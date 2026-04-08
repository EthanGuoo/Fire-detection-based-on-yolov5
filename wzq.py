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
from PySide6 import QtGui
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QImage, QKeyEvent, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGraphicsDropShadowEffect,
    QInputDialog,
    QMainWindow,
    QMessageBox,
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
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        self.my_thread = MyThread()
        self.my_thread.weights = ''
        self.my_thread.is_save = Qt.Unchecked
        self.my_thread.is_alarm = Qt.Unchecked

        self.flash_on = False
        self.flash_timer = QTimer(self)
        self.flash_timer.timeout.connect(self.toggle_alarm_flash)

        self.build_scifi_style()
        self.remove_logo()
        self.methoBinding()

        self.my_thread.send_img.connect(lambda x: self.show_image(x, self.label_result))
        self.my_thread.send_detectinfo_dic.connect(self.show_detect_info)
        self.my_thread.detect_speed.connect(self.show_detect_speed)
        self.my_thread.alarm_state.connect(self.handle_alarm_state)
        self.my_thread.status_text.connect(self.show_status)
        self.load_config()

    def build_scifi_style(self):
        self.setWindowTitle('YOLOv5 火焰检测系统')
        self.centralwidget.setStyleSheet("""
            QWidget#centralwidget {
                background:qlineargradient(x1:0,y1:0,x2:1,y2:1,
                    stop:0 #07111f, stop:0.45 #0b1f33, stop:1 #051521);
                color:#d8f6ff;
            }
            QLabel, QCheckBox { color:#d8f6ff; }
            QLineEdit, QTextBrowser, QDoubleSpinBox {
                background:rgba(7, 23, 39, 0.88);
                border:1px solid #2cc7ff;
                border-radius:10px;
                color:#c9f6ff;
                padding:6px;
            }
            QPushButton {
                background:qlineargradient(x1:0,y1:0,x2:1,y2:1,
                    stop:0 rgba(10, 82, 136, 220), stop:1 rgba(0, 214, 255, 180));
                border:1px solid #52e3ff;
                border-radius:12px;
                color:white;
                padding:6px 10px;
                font-weight:600;
            }
            QPushButton:hover {
                background:qlineargradient(x1:0,y1:0,x2:1,y2:1,
                    stop:0 rgba(21, 117, 188, 240), stop:1 rgba(84, 240, 255, 220));
            }
        """)
        self.label_title.setText('YOLOv5 火焰检测智能预警平台')
        self.label_title.setStyleSheet('color:#84f7ff; letter-spacing:2px;')
        self.label_result.setStyleSheet('border:2px solid #3ae6ff; border-radius:18px; background-color:rgba(3, 10, 18, 0.95);')
        self.textBrowser_result.setStyleSheet('background:rgba(3, 13, 24, 0.92); border:1px solid #41dfff; border-radius:14px; color:#d8f6ff;')
        self.lineEdit_weights.setStyleSheet('background:rgba(7, 23, 39, 0.88); border:1px solid #2cc7ff; border-radius:10px; color:#c9f6ff;')
        self.statusbar.showMessage('就绪 | 支持图片/视频/摄像头检测 | 按 + / - 调整识别源数量')

        for widget in [self.pushButton_img, self.pushButton_video, self.pushButton_camera, self.pushButton_weights, self.pushButton_zz]:
            effect = QGraphicsDropShadowEffect(self)
            effect.setBlurRadius(22)
            effect.setOffset(0, 0)
            effect.setColor(QtGui.QColor(42, 231, 255, 170))
            widget.setGraphicsEffect(effect)

        self.pushButton_img.setToolTip('选择单张图片进行火焰检测')
        self.pushButton_video.setToolTip('选择单个或多个视频文件进行火焰检测（支持并行处理）')
        self.pushButton_camera.setToolTip('选择单个或多个摄像头进行实时火焰检测')

    def remove_logo(self):
        self.pushButton_logo.hide()

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

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key_Plus, Qt.Key_Equal):
            self.my_thread.active_slots = min(max(1, len(self.my_thread.sources_config)), self.my_thread.active_slots + 1)
            self.statusbar.showMessage(f'已增加识别源数量到 {self.my_thread.active_slots}')
        elif event.key() == Qt.Key_Minus:
            self.my_thread.active_slots = max(1, self.my_thread.active_slots - 1)
            self.statusbar.showMessage(f'已减少识别源数量到 {self.my_thread.active_slots}')
        else:
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
        self.label_speed.setText(speed + ' ms')

    def show_status(self, text):
        self.statusbar.showMessage(text)

    def clean(self):
        self.my_thread.end_loop = True
        self.my_thread.wait()
        self.stop_alarm_flash()
        self.label_result.clear()
        self.label_result.setPixmap(QtGui.QPixmap('icon/test.png'))
        self.label_result.setAlignment(Qt.AlignCenter)
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

        # 如果选择了多个视频或需要自定义参数，才弹出分辨率选择
        if len(video_paths) > 1:
            picked = self._choose_resolution_and_fps()
            if picked is None:
                return
            self.my_thread.target_resolution, self.my_thread.target_fps = picked
        else:
            # 单视频使用默认参数
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
        self.textBrowser_result.clear()
        self.textBrowser_result.append('=== 实时检测信息 ===')
        self.textBrowser_result.append('运行设备: GPU' if torch.cuda.is_available() else '运行设备: CPU')
        self.textBrowser_result.append(f'当前源: {self.my_thread.source_name}')
        self.textBrowser_result.append(f'启用识别源数量: {self.my_thread.active_slots}')
        self.textBrowser_result.append(f'分辨率: {self.my_thread.target_resolution[0]}x{self.my_thread.target_resolution[1]}')
        self.textBrowser_result.append(f'目标帧率: {self.my_thread.target_fps}')
        self.textBrowser_result.append(f'置信度阈值: {self.my_thread.conf}')
        self.textBrowser_result.append(f'IOU 阈值: {self.my_thread.iou}')
        self.textBrowser_result.append('报警联动: 开启' if self.my_thread.is_alarm == Qt.Checked else '报警联动: 关闭')
        self.textBrowser_result.append('快捷键: + 增加识别源  |  - 减少识别源')
        self.textBrowser_result.append('------------------------')
        self.textBrowser_result.append('检测到的物体与数量：')
        if not names_dic:
            self.textBrowser_result.append('当前画面未发现目标')
            return
        shown = False
        for key, value in names_dic.items():
            if int(value) >= 1:
                shown = True
                self.textBrowser_result.append(f'{key}: {value}')
        if not shown:
            self.textBrowser_result.append('当前画面未发现目标')

    @staticmethod
    def show_image(image_path, label):
        try:
            ih, iw, _ = image_path.shape
            w = label.geometry().width()
            h = label.geometry().height()
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
            label.setPixmap(QPixmap.fromImage(img))
            label.setAlignment(Qt.AlignCenter)
        except Exception as e:
            print(repr(e))

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
            self.label_result.setStyleSheet('border:4px solid #ff365e; border-radius:18px; background-color:rgba(40, 5, 12, 0.93);')
            self.textBrowser_result.setStyleSheet('background:rgba(36, 8, 18, 0.92); border:2px solid #ff5b7f; border-radius:14px; color:#ffe8ef;')
            self.statusbar.setStyleSheet('background:#5a1022; color:#ffffff;')
            self.statusbar.showMessage('警报：检测到火情/烟雾目标')
        else:
            self.label_result.setStyleSheet('border:2px solid #3ae6ff; border-radius:18px; background-color:rgba(3, 10, 18, 0.95);')
            self.textBrowser_result.setStyleSheet('background:rgba(3, 13, 24, 0.92); border:1px solid #41dfff; border-radius:14px; color:#d8f6ff;')
            self.statusbar.setStyleSheet('background:#0b1f33; color:#d8f6ff;')
            self.statusbar.showMessage('警报中：请尽快确认现场情况')

    def stop_alarm_flash(self):
        self.flash_timer.stop()
        self.flash_on = False
        self.label_result.setStyleSheet('border:2px solid #3ae6ff; border-radius:18px; background-color:rgba(3, 10, 18, 0.95);')
        self.textBrowser_result.setStyleSheet('background:rgba(3, 13, 24, 0.92); border:1px solid #41dfff; border-radius:14px; color:#d8f6ff;')
        self.statusbar.setStyleSheet('background:#0b1f33; color:#d8f6ff;')
        self.statusbar.showMessage('监控中')

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
