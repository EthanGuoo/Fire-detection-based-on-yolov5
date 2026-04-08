import json
import math
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pyttsx3
import torch
from PySide6 import QtGui, QtCore
from PySide6.QtCore import Qt, QThread, QTimer, Signal, QSize, QRect
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QLinearGradient,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fire import Ui_MainWindow
from models.common import DetectMultiBackend
from utils.general import check_file, check_img_size, cv2 as yolov5_cv2, non_max_suppression, scale_boxes
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, smart_inference_mode

cv2 = yolov5_cv2


class MultiSourceCapture:
    """多源采集，支持多个摄像头或多个本地视频同时读取。"""

    def __init__(
        self,
        sources: List[dict],
        target_resolution: Tuple[int, int],
        target_fps: int,
        use_msmf_first: bool = True,
    ):
        self.sources = sources
        self.target_width, self.target_height = target_resolution
        self.target_fps = max(1, int(target_fps))
        self.use_msmf_first = use_msmf_first
        self.caps: List[cv2.VideoCapture] = []
        self.frame_interval = 1.0 / self.target_fps
        self.last_read_time = 0.0
        self.names = [src["name"] for src in sources]
        self._open_all()

    def _open_camera(self, index: int) -> Optional[cv2.VideoCapture]:
        backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW] if self.use_msmf_first else [cv2.CAP_DSHOW, cv2.CAP_MSMF]
        for backend in backends:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
                cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                ok, _ = cap.read()
                if ok:
                    return cap
            cap.release()
        return None

    def _open_video(self, path: str) -> Optional[cv2.VideoCapture]:
        real_path = str(check_file(path))
        cap = cv2.VideoCapture(real_path)
        if cap.isOpened():
            return cap
        cap.release()
        return None

    def _open_all(self) -> None:
        for src in self.sources:
            cap = None
            if src["type"] == "camera":
                cap = self._open_camera(int(src["value"]))
            elif src["type"] == "video":
                cap = self._open_video(str(src["value"]))

            if cap is None:
                self.release()
                raise RuntimeError(f"无法打开输入源：{src['name']}")
            self.caps.append(cap)

    def read(self) -> Tuple[List[bool], List[np.ndarray]]:
        now = time.time()
        elapsed = now - self.last_read_time
        if elapsed < self.frame_interval:
            time.sleep(self.frame_interval - elapsed)
        self.last_read_time = time.time()

        oks: List[bool] = []
        frames: List[np.ndarray] = []
        for cap in self.caps:
            ok, frame = cap.read()
            oks.append(bool(ok))
            frames.append(frame if ok else None)
        return oks, frames

    def release(self) -> None:
        for cap in self.caps:
            try:
                cap.release()
            except Exception:
                pass
        self.caps = []


class MyThread(QThread):
    send_img = Signal(np.ndarray)
    send_detectinfo_dic = Signal(dict)
    detect_speed = Signal(str)
    alarm_state = Signal(bool)
    status_text = Signal(str)

    def __init__(self):
        super().__init__()
        self.weights = 'fire.pt'
        self.conf = 0.25
        self.iou = 0.45
        self.is_save = Qt.Unchecked
        self.is_alarm = Qt.Unchecked
        self.end_loop = False
        self.source_name = '未选择'
        self.alarm_cooldown = 18
        self._alarm_counter = 0

        self.target_resolution = (1280, 720)
        self.target_fps = 20
        self.active_slots = 1
        self.sources_config: List[dict] = []

    def _safe_speak(self, text: str):
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass

    @staticmethod
    def _letterbox(image: np.ndarray, new_shape=(640, 640), color=(114, 114, 114)):
        shape = image.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return image

    def _make_mosaic(self, frames: List[np.ndarray], names: List[str]) -> np.ndarray:
        valid = [(f, n) for f, n in zip(frames, names) if f is not None]
        if not valid:
            return np.zeros((720, 1280, 3), dtype=np.uint8)

        panel_w = 640
        panel_h = 360
        panels = []
        for frame, name in valid[: max(1, self.active_slots)]:
            resized = cv2.resize(frame, (panel_w, panel_h))
            cv2.putText(
                resized,
                name,
                (14, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            panels.append(resized)

        cols = 2 if len(panels) > 1 else 1
        rows = math.ceil(len(panels) / cols)
        canvas = np.zeros((rows * panel_h, cols * panel_w, 3), dtype=np.uint8)

        for idx, panel in enumerate(panels):
            r = idx // cols
            c = idx % cols
            canvas[r * panel_h:(r + 1) * panel_h, c * panel_w:(c + 1) * panel_w] = panel
        return canvas

    def _detect_image(self, image_path: str, device, model, imgsz, names):
        """检测单张图片"""
        im0 = cv2.imread(image_path)
        if im0 is None:
            self.status_text.emit(f'无法读取图片: {image_path}')
            return

        # Letterbox
        letter = self._letterbox(im0, imgsz)
        letter = letter[:, :, ::-1].transpose(2, 0, 1)
        letter = np.ascontiguousarray(letter)

        # 推理
        t0 = time.time()
        im = torch.from_numpy(letter).to(model.device)
        im = im.half() if model.fp16 else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]

        pred = model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, self.conf, self.iou, None, False, max_det=1000)
        infer_time_ms = (time.time() - t0) * 1000

        # 绘制结果
        annotator = Annotator(im0, line_width=3, example=str(names))
        det = pred[0]
        detect_counts = {}

        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)
                label = f'{names[c]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(c, True))
                detect_counts[names[c]] = detect_counts.get(names[c], 0) + 1

        result_img = annotator.result()

        # 发送结果
        self.send_img.emit(result_img)
        self.send_detectinfo_dic.emit(detect_counts)
        self.detect_speed.emit(str(round(infer_time_ms, 1)))
        self.status_text.emit(f'图片检测完成 | 尺寸: {im0.shape[1]}x{im0.shape[0]} | 耗时: {infer_time_ms:.1f}ms')

        if len(det) > 0:
            self.alarm_state.emit(True)
        else:
            self.alarm_state.emit(False)

    @smart_inference_mode()
    def run(self):
        if not self.sources_config:
            self.status_text.emit('未配置检测源')
            return

        # 处理图片检测
        if self.sources_config[0]['type'] == 'image':
            try:
                device = select_device('')
                half = device.type != 'cpu'
                model = DetectMultiBackend(self.weights, device=device, dnn=False, data='data/custom.yaml', fp16=half)
                stride, names = model.stride, model.names
                imgsz = check_img_size((640, 640), s=stride)
                if half:
                    model.half()
                model.warmup(imgsz=(1, 3, *imgsz))

                image_path = self.sources_config[0]['value']
                self.status_text.emit(f'正在检测图片: {Path(image_path).name}')
                self._detect_image(image_path, device, model, imgsz, names)

            except Exception as e:
                self.alarm_state.emit(False)
                self.status_text.emit(f'错误: {e}')
                print(f'图片检测错误: {e}')
            return

        capture = None
        try:
            device = select_device('')
            half = device.type != 'cpu'
            model = DetectMultiBackend(self.weights, device=device, dnn=False, data='data/custom.yaml', fp16=half)
            stride, names = model.stride, model.names
            imgsz = check_img_size((640, 640), s=stride)
            if half:
                model.half()
            model.warmup(imgsz=(1, 3, *imgsz))

            active_sources = self.sources_config[: max(1, self.active_slots)]
            capture = MultiSourceCapture(active_sources, self.target_resolution, self.target_fps)

            speak_thread = None
            fixed_text = '警报，警报，发现火情，请迅速处理！'
            infer_time_ms = 0.0

            while not self.end_loop:
                oks, frames = capture.read()
                if not any(oks):
                    break

                annotated_frames: List[np.ndarray] = []
                merged_counts: Dict[str, int] = {}
                frame_has_alarm = False

                for ok, frame, src in zip(oks, frames, active_sources):
                    if not ok or frame is None:
                        blank = np.zeros((360, 640, 3), dtype=np.uint8)
                        cv2.putText(blank, f"{src['name']} 无信号", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                        annotated_frames.append(blank)
                        continue

                    im0 = frame.copy()
                    letter = self._letterbox(im0, imgsz)
                    letter = letter[:, :, ::-1].transpose(2, 0, 1)
                    letter = np.ascontiguousarray(letter)

                    t0 = time.time()
                    im = torch.from_numpy(letter).to(model.device)
                    im = im.half() if model.fp16 else im.float()
                    im /= 255
                    if len(im.shape) == 3:
                        im = im[None]

                    pred = model(im, augment=False, visualize=False)
                    pred = non_max_suppression(pred, self.conf, self.iou, None, False, max_det=1000)
                    infer_time_ms = (time.time() - t0) * 1000

                    annotator = Annotator(im0, line_width=2, example=str(names))
                    det = pred[0]
                    if len(det):
                        frame_has_alarm = True
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                        for *xyxy, conf, cls in reversed(det):
                            c = int(cls)
                            label = f'{names[c]} {conf:.2f}'
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            merged_counts[names[c]] = merged_counts.get(names[c], 0) + 1

                    out = annotator.result()
                    status = f"{src['name']} | {im0.shape[1]}x{im0.shape[0]} | {self.target_fps} FPS"
                    cv2.putText(out, status, (12, im0.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
                    annotated_frames.append(out)

                mosaic = self._make_mosaic(annotated_frames, [s['name'] for s in active_sources])
                self.send_img.emit(mosaic)
                self.send_detectinfo_dic.emit(merged_counts)
                self.detect_speed.emit(str(round(infer_time_ms, 1)))

                if frame_has_alarm:
                    self._alarm_counter += 1
                    self.alarm_state.emit(True)
                    if self.is_alarm == Qt.Checked and self._alarm_counter >= self.alarm_cooldown:
                        self._alarm_counter = 0
                        if not (speak_thread and speak_thread.is_alive()):
                            speak_thread = threading.Thread(target=self._safe_speak, args=(fixed_text,), daemon=True)
                            speak_thread.start()
                else:
                    self._alarm_counter = 0
                    self.alarm_state.emit(False)

                self.status_text.emit(
                    f"检测中 | 源数量: {len(active_sources)} | 分辨率: {self.target_resolution[0]}x{self.target_resolution[1]} | FPS: {self.target_fps}"
                )

            self.alarm_state.emit(False)
            self.status_text.emit('检测结束')
        except Exception as e:
            self.alarm_state.emit(False)
            self.status_text.emit(f'错误: {e}')
            print(f'出错了，出错的问题是: {e}')
            raise
        finally:
            if capture is not None:
                capture.release()


class MainWindow(QMainWindow, Ui_MainWindow):
    ACTION_META = (
        ('pushButton_img', '图片', '选择图片进行火焰检测'),
        ('pushButton_video', '视频', '选择视频文件进行火焰检测'),
        ('pushButton_camera', '摄像头', '使用摄像头实时检测火焰'),
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.hide_legacy_elements()

        self.setMinimumSize(1320, 780)
        self.setMaximumSize(2560, 1440)
        self.resize(1440, 860)

        self.my_thread = MyThread()
        self.my_thread.weights = ''
        self.my_thread.is_save = Qt.Unchecked
        self.my_thread.is_alarm = Qt.Unchecked

        self.flash_on = False
        self.flash_timer = QTimer(self)
        self.flash_timer.timeout.connect(self.toggle_alarm_flash)

        self.left_panel = None
        self._default_preview_active = True
        self._default_preview_message = '请从左侧选择图片、视频或摄像头，识别结果会居中显示在这里'
        self._last_frame = None

        self.build_scifi_style()
        self.build_premium_layout()
        self.remove_logo()
        self.adjust_control_sizes()
        self.methoBinding()

        self.my_thread.send_img.connect(self.show_image)
        self.my_thread.send_detectinfo_dic.connect(self.show_detect_info)
        self.my_thread.detect_speed.connect(self.show_detect_speed)
        self.my_thread.alarm_state.connect(self.handle_alarm_state)
        self.my_thread.status_text.connect(self.show_status)

        self.load_config()
        self.update_responsive_layout()
        QTimer.singleShot(0, lambda: self.set_default_preview(self._default_preview_message))


    def hide_legacy_elements(self):
        for name in (
            'label_weights',
            'label_source',
            'label_conf',
            'label_iou',
            'label_alarm_level',
            'label_is_save',
            'pushButton_speed',
            'gridLayoutWidget',
            'formLayoutWidget',
        ):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.hide()

    def _apply_shadow(self, widget, blur=30, y_offset=8, alpha=110):
        effect = QGraphicsDropShadowEffect(self)
        effect.setBlurRadius(blur)
        effect.setOffset(0, y_offset)
        effect.setColor(QColor(0, 0, 0, alpha))
        widget.setGraphicsEffect(effect)

    def _create_section_title(self, text: str, parent: QWidget) -> QLabel:
        label = QLabel(text, parent)
        label.setStyleSheet('color:#d7e3ff; font-size:13px; font-weight:700; letter-spacing:1px;')
        return label

    def _style_source_button(self, button: QPushButton, caption: str, tooltip: str) -> None:
        button.setToolTip(tooltip)
        button.setCursor(Qt.PointingHandCursor)
        button.setFixedSize(68, 68)
        button.setIconSize(QSize(28, 28))
        if button.icon().isNull():
            button.setText(caption[:1])
            button.setStyleSheet(
                """
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                        stop:0 rgba(86, 116, 255, 0.92), stop:1 rgba(82, 206, 255, 0.92));
                    border: 1px solid rgba(255, 255, 255, 0.18);
                    border-radius: 20px;
                    color: white;
                    font-size: 20px;
                    font-weight: 800;
                    padding: 0;
                }
                QPushButton:hover {
                    border: 1px solid rgba(255, 255, 255, 0.32);
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                        stop:0 rgba(106, 136, 255, 0.98), stop:1 rgba(110, 220, 255, 0.98));
                }
                QPushButton:pressed {
                    background: rgba(64, 93, 210, 0.96);
                }
                """
            )
        else:
            button.setText('')
            button.setStyleSheet(
                """
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                        stop:0 rgba(79, 101, 214, 0.95), stop:1 rgba(62, 171, 219, 0.94));
                    border: 1px solid rgba(255, 255, 255, 0.18);
                    border-radius: 20px;
                    padding: 0;
                }
                QPushButton:hover {
                    border: 1px solid rgba(255, 255, 255, 0.34);
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                        stop:0 rgba(102, 126, 255, 1.0), stop:1 rgba(84, 205, 255, 0.98));
                }
                QPushButton:pressed {
                    background: rgba(56, 86, 201, 0.96);
                }
                """
            )

    def _create_action_item(self, button: QPushButton, caption: str, tooltip: str) -> QFrame:
        card = QFrame(self.left_panel)
        card.setStyleSheet(
            """
            QFrame {
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 20px;
            }
            """
        )
        card.setMinimumWidth(96)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(8, 10, 8, 10)
        layout.setSpacing(8)

        button.setParent(card)
        self._style_source_button(button, caption, tooltip)
        layout.addWidget(button, 0, Qt.AlignCenter)

        label = QLabel(caption, card)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet('color:#cad8f6; font-size:12px; font-weight:600;')
        layout.addWidget(label)
        return card

    def build_scifi_style(self):
        self.setWindowTitle('智能火焰识别控制台')
        self.centralwidget.setStyleSheet(
            """
            QWidget#centralwidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #060a14, stop:0.48 #0c1220, stop:1 #101a2d);
                color: #edf3ff;
            }
            QLabel {
                color: #edf3ff;
                font-size: 13px;
            }
            QLineEdit, QTextBrowser, QDoubleSpinBox {
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.10);
                border-radius: 14px;
                color: #f6f8ff;
                padding: 10px 12px;
                font-size: 13px;
                selection-background-color: rgba(107, 141, 255, 0.65);
            }
            QLineEdit:focus, QDoubleSpinBox:focus {
                background: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(117, 154, 255, 0.92);
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(83, 111, 242, 0.97), stop:1 rgba(80, 172, 239, 0.95));
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 14px;
                color: white;
                font-size: 13px;
                font-weight: 700;
                padding: 0 16px;
            }
            QPushButton:hover {
                border: 1px solid rgba(255, 255, 255, 0.28);
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(103, 130, 255, 1.0), stop:1 rgba(92, 191, 255, 1.0));
            }
            QPushButton:pressed {
                background: rgba(64, 92, 216, 0.96);
            }
            QPushButton:disabled {
                background: rgba(255, 255, 255, 0.06);
                color: rgba(255, 255, 255, 0.45);
            }
            QCheckBox {
                color: #e7eeff;
                font-size: 13px;
                spacing: 10px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 6px;
                border: 1px solid rgba(255, 255, 255, 0.22);
                background: rgba(255, 255, 255, 0.04);
            }
            QCheckBox::indicator:checked {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(86, 117, 255, 0.98), stop:1 rgba(85, 201, 255, 0.98));
                border: 1px solid rgba(255, 255, 255, 0.36);
            }
            """
        )

        self.label_title.setText('智能火焰识别控制台')
        self.label_title.setFont(QFont('Microsoft YaHei', 20, QFont.Bold))
        self.label_title.setAlignment(Qt.AlignCenter)
        self.label_title.setStyleSheet(
            """
            QLabel {
                color: #f5f8ff;
                font-size: 24px;
                font-weight: 800;
                letter-spacing: 2px;
                background: rgba(255, 255, 255, 0.04);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 20px;
                padding: 10px 18px;
            }
            """
        )
        self._apply_shadow(self.label_title, blur=26, y_offset=6, alpha=90)
        self._apply_shadow(self.label_result, blur=30, y_offset=8, alpha=95)
        self._apply_shadow(self.textBrowser_result, blur=24, y_offset=8, alpha=80)

        if hasattr(self, 'label_speed'):
            self.label_speed.setAlignment(Qt.AlignCenter)
            self.label_speed.setText('0 ms')
            self.label_speed.setStyleSheet(
                """
                QLabel {
                    background: rgba(255, 255, 255, 0.06);
                    border: 1px solid rgba(255, 255, 255, 0.10);
                    border-radius: 14px;
                    color: #dce7ff;
                    font-size: 13px;
                    font-weight: 700;
                    padding: 4px 10px;
                }
                """
            )

        self._apply_normal_visual_state()
        self.statusbar.showMessage('系统就绪 | 图片 / 视频 / 摄像头模式已准备完成')

    def build_premium_layout(self):
        self.left_panel = QFrame(self.centralwidget)
        self.left_panel.setObjectName('leftPanel')
        self.left_panel.setStyleSheet(
            """
            QFrame#leftPanel {
                background: rgba(10, 14, 24, 0.82);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 26px;
            }
            """
        )
        self._apply_shadow(self.left_panel, blur=34, y_offset=10, alpha=110)

        left_layout = QVBoxLayout(self.left_panel)
        left_layout.setContentsMargins(20, 22, 20, 22)
        left_layout.setSpacing(18)

        panel_title = QLabel('控制面板', self.left_panel)
        panel_title.setStyleSheet('color:#ffffff; font-size:16px; font-weight:800; letter-spacing:1px;')
        left_layout.addWidget(panel_title)

        intro = QLabel('左侧控件已按功能重新分组，按钮横向整齐排列，适合直接开始识别。', self.left_panel)
        intro.setWordWrap(True)
        intro.setStyleSheet('color:#91a3c8; font-size:12px; line-height:1.6;')
        left_layout.addWidget(intro)

        left_layout.addWidget(self._create_section_title('模型权重', self.left_panel))
        weights_wrap = QWidget(self.left_panel)
        weights_layout = QHBoxLayout(weights_wrap)
        weights_layout.setContentsMargins(0, 0, 0, 0)
        weights_layout.setSpacing(10)
        self.lineEdit_weights.setParent(weights_wrap)
        self.lineEdit_weights.setPlaceholderText('请选择 .pt 权重文件')
        self.pushButton_weights.setParent(weights_wrap)
        self.pushButton_weights.setText('浏览')
        self.pushButton_weights.setCursor(Qt.PointingHandCursor)
        weights_layout.addWidget(self.lineEdit_weights, 1)
        weights_layout.addWidget(self.pushButton_weights)
        left_layout.addWidget(weights_wrap)

        left_layout.addWidget(self._create_section_title('检测源', self.left_panel))
        source_wrap = QWidget(self.left_panel)
        source_layout = QHBoxLayout(source_wrap)
        source_layout.setContentsMargins(0, 0, 0, 0)
        source_layout.setSpacing(10)
        for attr, caption, tooltip in self.ACTION_META:
            source_layout.addWidget(self._create_action_item(getattr(self, attr), caption, tooltip))
        left_layout.addWidget(source_wrap)

        left_layout.addWidget(self._create_section_title('识别参数', self.left_panel))
        param_card = QFrame(self.left_panel)
        param_card.setStyleSheet(
            """
            QFrame {
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 20px;
            }
            """
        )
        param_layout = QHBoxLayout(param_card)
        param_layout.setContentsMargins(14, 14, 14, 14)
        param_layout.setSpacing(12)

        conf_box = QWidget(param_card)
        conf_layout = QVBoxLayout(conf_box)
        conf_layout.setContentsMargins(0, 0, 0, 0)
        conf_layout.setSpacing(8)
        conf_title = QLabel('置信度', conf_box)
        conf_title.setStyleSheet('color:#c8d7f4; font-size:12px; font-weight:700;')
        self.doubleSpinBox_conf.setParent(conf_box)
        conf_layout.addWidget(conf_title)
        conf_layout.addWidget(self.doubleSpinBox_conf)

        iou_box = QWidget(param_card)
        iou_layout = QVBoxLayout(iou_box)
        iou_layout.setContentsMargins(0, 0, 0, 0)
        iou_layout.setSpacing(8)
        iou_title = QLabel('IOU 阈值', iou_box)
        iou_title.setStyleSheet('color:#c8d7f4; font-size:12px; font-weight:700;')
        self.doubleSpinBox_iou.setParent(iou_box)
        iou_layout.addWidget(iou_title)
        iou_layout.addWidget(self.doubleSpinBox_iou)

        param_layout.addWidget(conf_box)
        param_layout.addWidget(iou_box)
        left_layout.addWidget(param_card)

        left_layout.addWidget(self._create_section_title('运行选项', self.left_panel))
        option_card = QFrame(self.left_panel)
        option_card.setStyleSheet(
            """
            QFrame {
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 20px;
            }
            """
        )
        option_layout = QVBoxLayout(option_card)
        option_layout.setContentsMargins(16, 14, 16, 14)
        option_layout.setSpacing(12)
        self.checkBox_is_alarm.setParent(option_card)
        self.checkBox_is_alarm.setText('报警联动')
        self.checkBox_is_save.setParent(option_card)
        self.checkBox_is_save.setText('保存结果')
        option_layout.addWidget(self.checkBox_is_alarm)
        option_layout.addWidget(self.checkBox_is_save)
        left_layout.addWidget(option_card)

        info_chip = QLabel('支持图片、视频、摄像头三种模式快速切换', self.left_panel)
        info_chip.setAlignment(Qt.AlignCenter)
        info_chip.setStyleSheet(
            'background: rgba(93, 130, 255, 0.12); border: 1px solid rgba(108, 146, 255, 0.24); '
            'border-radius: 16px; color:#cddcff; padding: 10px 12px; font-size:12px;'
        )
        left_layout.addWidget(info_chip)

        left_layout.addStretch(1)

        self.pushButton_zz.setParent(self.left_panel)
        self.pushButton_zz.setText('停止 / 复位')
        self.pushButton_zz.setCursor(Qt.PointingHandCursor)
        left_layout.addWidget(self.pushButton_zz)

        self.left_panel.raise_()

    def remove_logo(self):
        if hasattr(self, 'pushButton_logo') and self.pushButton_logo is not None:
            self.pushButton_logo.hide()
            self.pushButton_logo.setEnabled(False)
            self.pushButton_logo.setText('')

    def adjust_control_sizes(self):
        self.lineEdit_weights.setMinimumHeight(46)
        self.pushButton_weights.setMinimumSize(92, 46)
        self.pushButton_weights.setStyleSheet(
            """
            QPushButton {
                background: rgba(255, 255, 255, 0.06);
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 14px;
                color: #eef4ff;
                font-weight: 700;
                padding: 0 16px;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 0.10);
                border: 1px solid rgba(123, 156, 255, 0.36);
            }
            QPushButton:pressed {
                background: rgba(255, 255, 255, 0.14);
            }
            """
        )

        spin_style = """
            QDoubleSpinBox {
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.10);
                border-radius: 14px;
                color: #f7f9ff;
                padding: 10px 12px;
                font-size: 13px;
                min-height: 24px;
            }
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                width: 22px;
                border-radius: 8px;
                background: rgba(255, 255, 255, 0.06);
                margin: 4px;
            }
            QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
                background: rgba(255, 255, 255, 0.12);
            }
        """
        self.doubleSpinBox_conf.setStyleSheet(spin_style)
        self.doubleSpinBox_iou.setStyleSheet(spin_style)
        self.doubleSpinBox_conf.setMinimumHeight(48)
        self.doubleSpinBox_iou.setMinimumHeight(48)
        self.doubleSpinBox_conf.setDecimals(2)
        self.doubleSpinBox_iou.setDecimals(2)
        self.doubleSpinBox_conf.setSingleStep(0.01)
        self.doubleSpinBox_iou.setSingleStep(0.01)

        self.pushButton_zz.setMinimumHeight(50)
        self.pushButton_zz.setStyleSheet(
            """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(255, 120, 131, 0.92), stop:1 rgba(255, 157, 83, 0.92));
                border: 1px solid rgba(255, 255, 255, 0.14);
                border-radius: 16px;
                color: #ffffff;
                font-size: 14px;
                font-weight: 800;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(255, 135, 145, 0.98), stop:1 rgba(255, 171, 98, 0.98));
            }
            QPushButton:pressed {
                background: rgba(222, 114, 96, 0.95);
            }
            """
        )

        for widget in [self.lineEdit_weights, self.checkBox_is_alarm, self.checkBox_is_save]:
            widget.setCursor(Qt.PointingHandCursor if hasattr(widget, 'setCursor') else Qt.ArrowCursor)

    def methoBinding(self):
        self.pushButton_img.clicked.connect(self.select_images)
        self.pushButton_video.clicked.connect(self.check_video)
        self.pushButton_camera.clicked.connect(self.open_camera)
        self.doubleSpinBox_conf.valueChanged.connect(self.change_conf)
        self.doubleSpinBox_iou.valueChanged.connect(self.change_iou)
        self.pushButton_weights.clicked.connect(self.select_weights)
        self.pushButton_zz.clicked.connect(self.clean)
        self.checkBox_is_alarm.clicked.connect(self.is_alarm)
        self.checkBox_is_save.clicked.connect(self.is_save)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_responsive_layout()

    def update_responsive_layout(self):
        margin = 22
        top_y = 18
        header_h = 52
        body_top = 92
        gap = 18
        panel_w = 360
        status_h = max(28, self.statusbar.sizeHint().height())
        available_h = self.height() - body_top - status_h - 22
        available_h = max(540, available_h)

        self.label_title.setGeometry(margin, top_y, self.width() - margin * 2, header_h)
        self.label_title.raise_()

        if hasattr(self, 'label_speed'):
            speed_w = 118
            self.label_speed.setGeometry(self.width() - margin - speed_w, top_y + 10, speed_w, 30)
            self.label_speed.raise_()

        self.left_panel.setGeometry(margin, body_top, panel_w, available_h)

        right_x = margin + panel_w + gap
        right_w = self.width() - right_x - margin
        right_w = max(600, right_w)
        preview_h = int(available_h * 0.68)
        preview_h = max(410, preview_h)
        info_y = body_top + preview_h + 16
        info_h = available_h - preview_h - 16
        info_h = max(158, info_h)

        self.label_result.setGeometry(right_x, body_top, right_w, preview_h)
        self.textBrowser_result.setGeometry(right_x, info_y, right_w, info_h)
        self.label_result.raise_()
        self.textBrowser_result.raise_()

        if self._default_preview_active:
            self.set_default_preview(self._default_preview_message)
        elif self._last_frame is not None:
            self._render_cv_image(self._last_frame)

    def _build_placeholder_pixmap(self, label_size: QSize, message: str) -> QPixmap:
        width = max(420, label_size.width() - 28)
        height = max(280, label_size.height() - 28)
        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        bg_gradient = QLinearGradient(0, 0, width, height)
        bg_gradient.setColorAt(0.0, QColor(11, 17, 28))
        bg_gradient.setColorAt(0.55, QColor(14, 23, 39))
        bg_gradient.setColorAt(1.0, QColor(18, 32, 54))
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(bg_gradient))
        painter.drawRoundedRect(0, 0, width, height, 28, 28)

        frame_rect = QRect(28, 28, width - 56, height - 56)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(130, 163, 255, 78), 2, Qt.DashLine))
        painter.drawRoundedRect(frame_rect, 24, 24)

        card_w = min(360, width - 80)
        card_h = min(220, height - 96)
        card_rect = QRect((width - card_w) // 2, (height - card_h) // 2 - 6, card_w, card_h)

        card_gradient = QLinearGradient(card_rect.left(), card_rect.top(), card_rect.right(), card_rect.bottom())
        card_gradient.setColorAt(0.0, QColor(255, 255, 255, 18))
        card_gradient.setColorAt(1.0, QColor(255, 255, 255, 8))
        painter.setBrush(QBrush(card_gradient))
        painter.setPen(QPen(QColor(255, 255, 255, 24), 1))
        painter.drawRoundedRect(card_rect, 24, 24)

        icon_rect = QRect(card_rect.center().x() - 34, card_rect.top() + 28, 68, 68)
        painter.setBrush(QColor(101, 137, 255, 36))
        painter.setPen(QPen(QColor(139, 174, 255, 220), 2))
        painter.drawRoundedRect(icon_rect, 18, 18)
        painter.drawEllipse(icon_rect.center(), 9, 9)
        painter.drawLine(icon_rect.center().x() - 18, icon_rect.center().y(), icon_rect.center().x() + 18, icon_rect.center().y())
        painter.drawLine(icon_rect.center().x(), icon_rect.center().y() - 18, icon_rect.center().x(), icon_rect.center().y() + 18)

        title_font = QFont('Microsoft YaHei', 18, QFont.Bold)
        painter.setFont(title_font)
        painter.setPen(QColor(243, 247, 255))
        painter.drawText(
            QRect(card_rect.left() + 24, card_rect.top() + 112, card_rect.width() - 48, 34),
            Qt.AlignCenter,
            '智能识别预览区'
        )

        body_font = QFont('Microsoft YaHei', 11)
        painter.setFont(body_font)
        painter.setPen(QColor(188, 201, 231))
        painter.drawText(
            QRect(card_rect.left() + 30, card_rect.top() + 148, card_rect.width() - 60, 42),
            Qt.AlignCenter | Qt.TextWordWrap,
            message
        )

        chip_rect = QRect((width - 300) // 2, min(height - 54, card_rect.bottom() + 20), 300, 34)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(90, 129, 255, 36))
        painter.drawRoundedRect(chip_rect, 16, 16)
        painter.setPen(QColor(205, 219, 255))
        painter.setFont(QFont('Microsoft YaHei', 10))
        painter.drawText(chip_rect, Qt.AlignCenter, '支持图片 / 视频 / 摄像头实时识别')

        painter.end()
        return pixmap

    def set_default_preview(self, message: str = None):
        if message:
            self._default_preview_message = message
        pixmap = self._build_placeholder_pixmap(self.label_result.size(), self._default_preview_message)
        self.label_result.setPixmap(pixmap)
        self.label_result.setAlignment(Qt.AlignCenter)
        self.label_result.setScaledContents(False)
        self._default_preview_active = True
        self._last_frame = None

    def _render_cv_image(self, image_path: np.ndarray):
        try:
            ih, iw, _ = image_path.shape
            w = max(1, self.label_result.geometry().width())
            h = max(1, self.label_result.geometry().height())
            if iw / w > ih / h:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(image_path, (nw, nh))
            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(image_path, (nw, nh))
            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1], QImage.Format.Format_RGB888)
            self.label_result.setPixmap(QPixmap.fromImage(img))
            self.label_result.setAlignment(Qt.AlignCenter)
        except Exception as e:
            print(repr(e))

    def keyPressEvent(self, event):
        super().keyPressEvent(event)

    def _choose_resolution_and_fps(self):
        resolutions = ['640x480', '1280x720', '1600x900', '1920x1080']
        res, ok = QInputDialog.getItem(self, '选择分辨率', '请选择采集分辨率：', resolutions, 1, False)
        if not ok:
            return None
        fps, ok = QInputDialog.getInt(self, '选择帧率', '请输入目标帧率：', self.my_thread.target_fps, 5, 60, 1)
        if not ok:
            return None
        w, h = res.split('x')
        return (int(w), int(h)), int(fps)

    def is_alarm(self):
        self.my_thread.is_alarm = Qt.Checked if self.checkBox_is_alarm.checkState() == Qt.Checked else Qt.Unchecked

    def is_save(self):
        self.my_thread.is_save = Qt.Checked if self.checkBox_is_save.checkState() == Qt.Checked else Qt.Unchecked

    def show_detect_speed(self, speed):
        if hasattr(self, 'label_speed'):
            self.label_speed.setText(speed + ' ms')

    def show_status(self, text):
        self.statusbar.showMessage(text)

    def clean(self):
        self.my_thread.end_loop = True
        if self.my_thread.isRunning():
            self.my_thread.wait()
        self.stop_alarm_flash()
        self.set_default_preview('请选择新的检测源开始识别')
        self.show_detect_info({})
        self.statusbar.showMessage('已停止检测')

    def select_images(self):
        if self.my_thread.isRunning():
            self.clean()
        image_path, _ = QFileDialog.getOpenFileName(self, '选择图片', '', '图像文件 (*.png *.jpg *.jpeg *.bmp *.webp)')
        if image_path:
            self.my_thread.sources_config = [
                {'type': 'image', 'value': image_path, 'name': Path(image_path).name}
            ]
            self.my_thread.active_slots = 1
            self.my_thread.source_name = f'图片: {Path(image_path).name}'
            self.start_detect()

    def check_video(self):
        if self.my_thread.isRunning():
            self.clean()

        video_paths, _ = QFileDialog.getOpenFileNames(
            self,
            '选择一个或多个视频',
            '',
            'Videos (*.mp4 *.avi *.mkv *.mov *.flv *.wmv);;All Files (*)'
        )

        if not video_paths:
            return

        if len(video_paths) > 1:
            picked = self._choose_resolution_and_fps()
            if picked is None:
                return
            self.my_thread.target_resolution, self.my_thread.target_fps = picked
        else:
            self.my_thread.target_resolution = (1280, 720)
            self.my_thread.target_fps = 20

        self.my_thread.sources_config = [
            {'type': 'video', 'value': p, 'name': Path(p).name}
            for p in video_paths
        ]
        self.my_thread.active_slots = len(self.my_thread.sources_config)
        self.my_thread.source_name = f'视频源 x {len(video_paths)}'
        self.start_detect()

    def get_camera_num(self):
        device_list = []
        for i in range(8):
            ok = False
            for backend in (cv2.CAP_MSMF, cv2.CAP_DSHOW):
                stream = cv2.VideoCapture(i, backend)
                if stream.isOpened():
                    ret, _ = stream.read()
                    ok = bool(ret)
                stream.release()
                if ok:
                    break
            if ok:
                device_list.append(i)
        return len(device_list), device_list

    def open_camera(self):
        if self.my_thread.isRunning():
            self.clean()

        picked = self._choose_resolution_and_fps()
        if picked is None:
            return
        self.my_thread.target_resolution, self.my_thread.target_fps = picked

        num, device_list = self.get_camera_num()
        if num == 0:
            QMessageBox.warning(self, '出错啦', '<p>未检测到有效的摄像头</p>')
            return

        manual, ok = QInputDialog.getText(
            self,
            '多摄像头输入',
            f'可用摄像头编号: {", ".join(str(i) for i in device_list)}\n请输入要识别的编号，逗号分隔：',
            text=','.join(str(i) for i in device_list[: min(2, len(device_list))])
        )
        if not ok or not manual.strip():
            return

        indexes = []
        for part in manual.split(','):
            part = part.strip()
            if part.isdigit():
                idx = int(part)
                if idx in device_list and idx not in indexes:
                    indexes.append(idx)

        if not indexes:
            QMessageBox.warning(self, '提示', '未输入有效摄像头编号')
            return

        self.my_thread.sources_config = [{'type': 'camera', 'value': idx, 'name': f'摄像头 {idx}'} for idx in indexes]
        self.my_thread.active_slots = len(self.my_thread.sources_config)
        self.my_thread.source_name = f'摄像头源 x {len(indexes)}'
        self.start_detect()

    def change_conf(self, x):
        self.my_thread.conf = round(float(x), 2)

    def change_iou(self, x):
        self.my_thread.iou = round(float(x), 2)

    def select_weights(self):
        weights_path, _ = QFileDialog.getOpenFileName(self, '选择权重', '', 'pt (*.pt)')
        if weights_path:
            self.lineEdit_weights.setText(weights_path)
            self.my_thread.weights = weights_path
            self.clean()

    def show_detect_info(self, names_dic):
        device_name = 'GPU' if torch.cuda.is_available() else 'CPU'
        source_name = self.my_thread.source_name or '未选择'
        alarm_state = '开启' if self.my_thread.is_alarm == Qt.Checked else '关闭'
        save_state = '开启' if self.my_thread.is_save == Qt.Checked else '关闭'

        detect_lines = []
        for key, value in names_dic.items():
            if int(value) >= 1:
                detect_lines.append(f'<div style="margin-bottom:6px;"><span style="color:#f5f8ff;">{key}</span>'
                                    f'<span style="float:right; color:#7fd3ff; font-weight:700;">{value}</span></div>')
        if not detect_lines:
            detect_lines.append('<div style="color:#93a7cb;">当前画面未发现目标</div>')

        html = f"""
        <div style="font-family:'Microsoft YaHei'; color:#eef3ff; font-size:13px; line-height:1.75;">
            <div style="font-size:15px; font-weight:800; color:#ffffff; margin-bottom:10px;">实时检测信息</div>
            <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); border-radius:14px; padding:12px 14px;">
                <div><span style="color:#93a7cb;">运行设备：</span>{device_name}</div>
                <div><span style="color:#93a7cb;">当前源：</span>{source_name}</div>
                <div><span style="color:#93a7cb;">启用源数量：</span>{self.my_thread.active_slots}</div>
                <div><span style="color:#93a7cb;">分辨率：</span>{self.my_thread.target_resolution[0]} x {self.my_thread.target_resolution[1]}</div>
                <div><span style="color:#93a7cb;">目标帧率：</span>{self.my_thread.target_fps}</div>
                <div><span style="color:#93a7cb;">置信度阈值：</span>{self.my_thread.conf}</div>
                <div><span style="color:#93a7cb;">IOU 阈值：</span>{self.my_thread.iou}</div>
                <div><span style="color:#93a7cb;">报警联动：</span>{alarm_state}</div>
                <div><span style="color:#93a7cb;">结果保存：</span>{save_state}</div>
            </div>
            <div style="margin-top:14px; font-size:15px; font-weight:800; color:#ffffff;">检测结果</div>
            <div style="margin-top:8px; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); border-radius:14px; padding:12px 14px;">
                {''.join(detect_lines)}
            </div>
        </div>
        """
        self.textBrowser_result.setHtml(html)

    def show_image(self, image_path):
        self._default_preview_active = False
        self._last_frame = image_path
        self._render_cv_image(image_path)

    def _apply_normal_visual_state(self):
        self.label_result.setStyleSheet(
            """
            QLabel {
                border: 1px solid rgba(255, 255, 255, 0.10);
                border-radius: 24px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(8, 14, 24, 0.96), stop:1 rgba(12, 22, 38, 0.96));
            }
            """
        )
        self.textBrowser_result.setStyleSheet(
            """
            QTextBrowser {
                background: rgba(10, 16, 27, 0.96);
                border: 1px solid rgba(255, 255, 255, 0.10);
                border-radius: 22px;
                color: #eef4ff;
                padding: 14px;
                font-size: 13px;
            }
            """
        )
        self.statusbar.setStyleSheet(
            """
            QStatusBar {
                background: rgba(8, 14, 24, 0.97);
                border-top: 1px solid rgba(255, 255, 255, 0.08);
                color: #dbe7ff;
                padding: 4px 10px;
            }
            """
        )

    def handle_alarm_state(self, has_alarm):
        if has_alarm:
            if not self.flash_timer.isActive():
                self.flash_on = False
                self.flash_timer.start(240)
        else:
            self.stop_alarm_flash()

    def toggle_alarm_flash(self):
        self.flash_on = not self.flash_on
        if self.flash_on:
            self.label_result.setStyleSheet('border:2px solid rgba(255, 92, 124, 0.96); border-radius:24px; background:rgba(43, 9, 18, 0.96);')
            self.textBrowser_result.setStyleSheet('background:rgba(36, 9, 19, 0.96); border:1px solid rgba(255, 105, 136, 0.68); border-radius:22px; color:#fff0f3; padding:14px; font-size:13px;')
            self.statusbar.setStyleSheet('background:rgba(76, 16, 30, 0.98); color:#fff3f5; border-top:1px solid rgba(255, 125, 145, 0.35); padding:4px 10px;')
            self.statusbar.showMessage('警报：检测到火情/烟雾目标')
        else:
            self.label_result.setStyleSheet('border:2px solid rgba(255, 162, 86, 0.92); border-radius:24px; background:rgba(45, 20, 10, 0.96);')
            self.textBrowser_result.setStyleSheet('background:rgba(35, 16, 10, 0.96); border:1px solid rgba(255, 169, 98, 0.54); border-radius:22px; color:#fff6ee; padding:14px; font-size:13px;')
            self.statusbar.setStyleSheet('background:rgba(76, 36, 16, 0.98); color:#fff5ef; border-top:1px solid rgba(255, 176, 110, 0.35); padding:4px 10px;')
            self.statusbar.showMessage('警报中：请尽快确认现场情况')

    def stop_alarm_flash(self):
        self.flash_timer.stop()
        self.flash_on = False
        self._apply_normal_visual_state()
        self.statusbar.showMessage('系统待命')

    def load_config(self):
        try:
            with open('config.json', 'r', encoding='utf-8') as jsonfile:
                loaded_config = json.load(jsonfile)
            self.my_thread.weights = loaded_config.get('weights', 'fire.pt')
            self.my_thread.conf = loaded_config.get('conf', 0.25)
            self.my_thread.iou = loaded_config.get('iou', 0.45)
            self.my_thread.is_alarm = loaded_config.get('is_alarm', Qt.Unchecked)
            self.my_thread.is_save = loaded_config.get('is_save', Qt.Unchecked)
            self.my_thread.target_fps = int(loaded_config.get('target_fps', 20))
            self.my_thread.target_resolution = tuple(loaded_config.get('target_resolution', [1280, 720]))
        except Exception:
            self.my_thread.weights = 'fire.pt'
            self.my_thread.conf = 0.25
            self.my_thread.iou = 0.45
            self.my_thread.is_alarm = Qt.Unchecked
            self.my_thread.is_save = Qt.Unchecked
            self.my_thread.target_resolution = (1280, 720)
            self.my_thread.target_fps = 20

        self.doubleSpinBox_conf.setProperty('value', self.my_thread.conf)
        self.doubleSpinBox_iou.setProperty('value', self.my_thread.iou)
        self.lineEdit_weights.setText(self.my_thread.weights)

        if self.my_thread.is_save == 'checked':
            self.checkBox_is_save.setCheckState(Qt.Checked)
            self.my_thread.is_save = Qt.Checked
        elif self.my_thread.is_save == Qt.Checked:
            self.checkBox_is_save.setCheckState(Qt.Checked)

        if self.my_thread.is_alarm == 'checked':
            self.checkBox_is_alarm.setCheckState(Qt.Checked)
            self.my_thread.is_alarm = Qt.Checked
        elif self.my_thread.is_alarm == Qt.Checked:
            self.checkBox_is_alarm.setCheckState(Qt.Checked)

        self.stop_alarm_flash()
        self.show_detect_info({})

    def closeEvent(self, event):
        confirm = QMessageBox.question(
            self,
            '关闭程序',
            '确定关闭？',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm == QMessageBox.StandardButton.Yes:
            config = {
                'conf': self.my_thread.conf,
                'iou': self.my_thread.iou,
                'weights': self.my_thread.weights,
                'is_save': 'checked' if self.my_thread.is_save == Qt.Checked else 'Unchecked',
                'is_alarm': 'checked' if self.my_thread.is_alarm == Qt.Checked else 'Unchecked',
                'target_fps': self.my_thread.target_fps,
                'target_resolution': list(self.my_thread.target_resolution),
            }
            with open('config.json', 'w', encoding='utf-8') as jsonfile:
                json.dump(config, jsonfile, indent=4, ensure_ascii=False)
            self.clean()
            event.accept()
        else:
            event.ignore()

    def start_detect(self):
        if not self.my_thread.weights:
            QMessageBox.warning(self, '提示', '请先选择权重文件')
            return
        if not self.my_thread.sources_config:
            QMessageBox.warning(self, '提示', '请先选择检测源')
            return
        self.my_thread.end_loop = False
        self.statusbar.showMessage(f'开始检测：{self.my_thread.source_name}')
        self.my_thread.start()

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
