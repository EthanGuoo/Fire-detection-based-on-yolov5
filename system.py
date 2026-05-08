import csv
import json
import math
import os
import sys
import threading
import time
from datetime import datetime
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
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QScrollArea,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from fire import Ui_MainWindow
from models.common import DetectMultiBackend
from utils.general import check_file, check_img_size, cv2 as yolov5_cv2, non_max_suppression, scale_boxes
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, smart_inference_mode

cv2 = yolov5_cv2

try:
    from PIL import Image as _PILImage, ImageDraw as _PILDraw, ImageFont as _PILFont
    _PIL_AVAILABLE = True
except Exception:
    _PIL_AVAILABLE = False


def _find_cjk_font() -> Optional[str]:
    """定位一个支持中文的 TTF / TTC 字体。"""
    candidates = [
        r'C:\Windows\Fonts\msyh.ttc',
        r'C:\Windows\Fonts\msyh.ttf',
        r'C:\Windows\Fonts\msyhbd.ttc',
        r'C:\Windows\Fonts\simhei.ttf',
        r'C:\Windows\Fonts\simsun.ttc',
        '/System/Library/Fonts/PingFang.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return None


_CJK_FONT_PATH = _find_cjk_font()
_CJK_FONT_CACHE: Dict[int, object] = {}


def _get_cjk_font(size: int):
    if not _PIL_AVAILABLE or _CJK_FONT_PATH is None:
        return None
    font = _CJK_FONT_CACHE.get(size)
    if font is None:
        try:
            font = _PILFont.truetype(_CJK_FONT_PATH, size)
            _CJK_FONT_CACHE[size] = font
        except Exception:
            return None
    return font


def draw_text_cn(
    image: np.ndarray,
    text: str,
    org: Tuple[int, int],
    font_size: int = 22,
    color_bgr: Tuple[int, int, int] = (0, 255, 255),
) -> np.ndarray:
    """在 OpenCV BGR 图像上绘制中文文字（仅渲染文字区域，性能友好）。"""
    font = _get_cjk_font(font_size)
    if font is None:
        cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_size / 30.0, color_bgr, 2, cv2.LINE_AA)
        return image

    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    bbox = font.getbbox(text)
    tw = bbox[2] - bbox[0] + 4
    th = bbox[3] - bbox[1] + 4
    if tw <= 0 or th <= 0:
        return image

    x0, y0 = int(org[0]), int(org[1])
    h, w = image.shape[:2]
    x1 = min(x0 + tw, w)
    y1 = min(y0 + th, h)
    if x0 >= w or y0 >= h or x1 <= x0 or y1 <= y0:
        return image

    roi = image[y0:y1, x0:x1]
    roi_rgb = roi[:, :, ::-1].copy()
    pil_roi = _PILImage.fromarray(roi_rgb)
    draw = _PILDraw.Draw(pil_roi)
    draw.text((-bbox[0], -bbox[1]), text, font=font, fill=color_rgb)
    image[y0:y1, x0:x1] = np.array(pil_roi)[:, :, ::-1]
    return image


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

    def _open_camera_url(self, url: str) -> Optional[cv2.VideoCapture]:
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
            cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            ok, _ = cap.read()
            if ok:
                return cap
        cap.release()
        return None

    def _open_all(self) -> None:
        for src in self.sources:
            cap = None
            if src["type"] == "camera":
                cap = self._open_camera(int(src["value"]))
            elif src["type"] == "camera_url":
                cap = self._open_camera_url(str(src["value"]))
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


class CameraRow(QFrame):
    """单台摄像头的卡片行，包含启用开关、名称、地址、重命名/删除按钮。"""

    toggled = Signal(int, bool)
    rename_requested = Signal(int)
    remove_requested = Signal(int)

    def __init__(self, index: int, cam: dict, parent=None):
        super().__init__(parent)
        self.index = index
        self.setObjectName('cameraRow')
        self.setStyleSheet(
            """
            QFrame#cameraRow {
                background: rgba(255, 255, 255, 0.04);
                border: 1px solid rgba(255, 255, 255, 0.10);
                border-radius: 14px;
            }
            QFrame#cameraRow:hover {
                background: rgba(123, 156, 255, 0.10);
                border: 1px solid rgba(123, 156, 255, 0.38);
            }
            """
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 10, 12, 10)
        layout.setSpacing(12)

        icon = QLabel(self)
        icon.setFixedSize(40, 40)
        icon_color = '#7fb2ff' if cam.get('type') == 'camera' else '#b784ff'
        icon_text = '本' if cam.get('type') == 'camera' else '网'
        icon.setAlignment(Qt.AlignCenter)
        icon.setText(icon_text)
        icon.setStyleSheet(
            f'background: rgba(123, 156, 255, 0.14); border: 1px solid {icon_color}55; '
            f'border-radius: 20px; color: {icon_color}; font-size: 14px; font-weight: 800;'
        )
        layout.addWidget(icon)

        text_box = QVBoxLayout()
        text_box.setContentsMargins(0, 0, 0, 0)
        text_box.setSpacing(2)
        self.name_label = QLabel(cam.get('name', ''), self)
        self.name_label.setStyleSheet('color:#f5f8ff; font-size:14px; font-weight:700;')
        self.addr_label = QLabel(self._format_addr(cam), self)
        self.addr_label.setStyleSheet('color:#9fb1d6; font-size:12px;')
        self.addr_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        text_box.addWidget(self.name_label)
        text_box.addWidget(self.addr_label)
        layout.addLayout(text_box, 1)

        self.enable_btn = QPushButton(self)
        self.enable_btn.setCheckable(True)
        self.enable_btn.setChecked(bool(cam.get('enabled', True)))
        self.enable_btn.setCursor(Qt.PointingHandCursor)
        self.enable_btn.setFixedSize(72, 32)
        self._refresh_enable_text()
        self.enable_btn.setStyleSheet(
            """
            QPushButton {
                background: rgba(255, 255, 255, 0.06);
                border: 1px solid rgba(255, 255, 255, 0.18);
                border-radius: 16px;
                color: #cfd9f0;
                font-size: 12px;
                font-weight: 700;
                padding: 0 10px;
            }
            QPushButton:checked {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(86, 117, 255, 0.98), stop:1 rgba(85, 201, 255, 0.98));
                border: 1px solid rgba(255, 255, 255, 0.30);
                color: #ffffff;
            }
            QPushButton:hover {
                border: 1px solid rgba(123, 156, 255, 0.48);
            }
            """
        )
        self.enable_btn.toggled.connect(self._on_toggle)
        layout.addWidget(self.enable_btn)

        self.rename_btn = self._make_tool_btn('改名')
        self.remove_btn = self._make_tool_btn('删除', danger=True)
        self.rename_btn.clicked.connect(lambda: self.rename_requested.emit(self.index))
        self.remove_btn.clicked.connect(lambda: self.remove_requested.emit(self.index))
        layout.addWidget(self.rename_btn)
        layout.addWidget(self.remove_btn)

    def _make_tool_btn(self, text: str, danger: bool = False) -> QPushButton:
        btn = QPushButton(text, self)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setFixedSize(54, 32)
        base = 'rgba(255, 122, 134, 0.22)' if danger else 'rgba(255, 255, 255, 0.05)'
        border = 'rgba(255, 122, 134, 0.55)' if danger else 'rgba(255, 255, 255, 0.18)'
        hover = 'rgba(255, 122, 134, 0.35)' if danger else 'rgba(123, 156, 255, 0.22)'
        color = '#ffd8dd' if danger else '#dbe4fa'
        btn.setStyleSheet(
            f"""
            QPushButton {{
                background: {base};
                border: 1px solid {border};
                border-radius: 16px;
                color: {color};
                font-size: 12px;
                font-weight: 700;
            }}
            QPushButton:hover {{
                background: {hover};
            }}
            """
        )
        return btn

    @staticmethod
    def _format_addr(cam: dict) -> str:
        if cam.get('type') == 'camera':
            return f"本机 USB  ·  设备编号 {cam.get('value')}"
        return f"网络流  ·  {cam.get('value')}"

    def _refresh_enable_text(self):
        self.enable_btn.setText('已启用' if self.enable_btn.isChecked() else '未启用')

    def _on_toggle(self, checked: bool):
        self._refresh_enable_text()
        self.toggled.emit(self.index, checked)

    def update_cam(self, cam: dict):
        self.name_label.setText(cam.get('name', ''))
        self.addr_label.setText(self._format_addr(cam))
        self.enable_btn.blockSignals(True)
        self.enable_btn.setChecked(bool(cam.get('enabled', True)))
        self._refresh_enable_text()
        self.enable_btn.blockSignals(False)


class CameraManagerDialog(QDialog):
    """摄像头管理对话框：扫描、添加、重命名、删除、选择启用。"""

    RESOLUTIONS = ['640x480', '1280x720', '1600x900', '1920x1080']

    def __init__(self, cameras: List[dict], resolution: Tuple[int, int], fps: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle('摄像头管理')
        self.setMinimumSize(560, 540)
        self.resize(620, 600)
        self.cameras: List[dict] = [dict(c) for c in cameras]
        self._rows: List[CameraRow] = []

        self.setStyleSheet(
            """
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0a1324, stop:1 #10203c);
                color: #eef3ff;
            }
            QLabel { color: #eef3ff; font-size: 13px; }
            QScrollArea { border: none; background: transparent; }
            QScrollBar:vertical {
                background: rgba(255, 255, 255, 0.03);
                width: 10px; border-radius: 5px; margin: 2px;
            }
            QScrollBar::handle:vertical {
                background: rgba(123, 156, 255, 0.35);
                border-radius: 5px; min-height: 24px;
            }
            QScrollBar::handle:vertical:hover { background: rgba(123, 156, 255, 0.55); }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            QComboBox, QSpinBox {
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.14);
                border-radius: 10px;
                color: #f2f5ff;
                padding: 6px 10px;
                min-height: 26px;
            }
            QComboBox:hover, QSpinBox:hover { border-color: rgba(123, 156, 255, 0.55); }
            QComboBox QAbstractItemView {
                background: #0f1a30;
                color: #eef3ff;
                selection-background-color: rgba(86, 117, 255, 0.55);
                border: 1px solid rgba(255, 255, 255, 0.10);
                outline: 0;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                width: 18px;
                background: rgba(255, 255, 255, 0.05);
                border: none;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background: rgba(123, 156, 255, 0.30);
            }
            """
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(22, 22, 22, 22)
        root.setSpacing(16)

        header = QVBoxLayout()
        header.setSpacing(4)
        title = QLabel('摄像头管理', self)
        title.setStyleSheet('color:#ffffff; font-size:20px; font-weight:800; letter-spacing:1px;')
        subtitle = QLabel('支持本机 USB 设备与网络摄像头（RTSP / HTTP），启用后点击一键接通立即开始检测。', self)
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet('color:#9fb1d6; font-size:12px;')
        header.addWidget(title)
        header.addWidget(subtitle)
        root.addLayout(header)

        action_row = QHBoxLayout()
        action_row.setSpacing(10)
        self.btn_scan = self._make_primary_btn('扫描本机设备', '#5d7cff', '#4aa5ff')
        self.btn_add_url = self._make_primary_btn('添加网络摄像头', '#8a5dff', '#5dc6ff')
        action_row.addWidget(self.btn_scan)
        action_row.addWidget(self.btn_add_url)
        action_row.addStretch(1)
        self.count_label = QLabel('', self)
        self.count_label.setStyleSheet('color:#9fb1d6; font-size:12px;')
        action_row.addWidget(self.count_label)
        root.addLayout(action_row)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_container = QWidget(self.scroll_area)
        self.rows_layout = QVBoxLayout(self.scroll_container)
        self.rows_layout.setContentsMargins(4, 4, 4, 4)
        self.rows_layout.setSpacing(10)
        self.rows_layout.addStretch(1)
        self.scroll_area.setWidget(self.scroll_container)
        self.scroll_area.setStyleSheet(
            'QScrollArea { background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 16px; }'
        )
        root.addWidget(self.scroll_area, 1)

        self.empty_label = QLabel('暂无摄像头，点击上方“扫描本机设备”或“添加网络摄像头”开始。', self.scroll_container)
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setStyleSheet('color:#8295b8; font-size:13px; padding: 40px 12px;')

        params_card = QFrame(self)
        params_card.setStyleSheet(
            'QFrame { background: rgba(255, 255, 255, 0.04); border: 1px solid rgba(255, 255, 255, 0.10); border-radius: 14px; }'
        )
        params_layout = QHBoxLayout(params_card)
        params_layout.setContentsMargins(16, 12, 16, 12)
        params_layout.setSpacing(20)

        res_box = QVBoxLayout()
        res_box.setSpacing(4)
        res_title = QLabel('采集分辨率', self)
        res_title.setStyleSheet('color:#9fb1d6; font-size:12px; font-weight:600;')
        self.combo_res = QComboBox(self)
        self.combo_res.addItems(self.RESOLUTIONS)
        current_res = f'{resolution[0]}x{resolution[1]}'
        if current_res in self.RESOLUTIONS:
            self.combo_res.setCurrentText(current_res)
        else:
            self.combo_res.addItem(current_res)
            self.combo_res.setCurrentText(current_res)
        res_box.addWidget(res_title)
        res_box.addWidget(self.combo_res)

        fps_box = QVBoxLayout()
        fps_box.setSpacing(4)
        fps_title = QLabel('目标帧率', self)
        fps_title.setStyleSheet('color:#9fb1d6; font-size:12px; font-weight:600;')
        self.spin_fps = QSpinBox(self)
        self.spin_fps.setRange(5, 60)
        self.spin_fps.setValue(int(fps))
        fps_box.addWidget(fps_title)
        fps_box.addWidget(self.spin_fps)

        params_layout.addLayout(res_box, 1)
        params_layout.addLayout(fps_box, 1)
        root.addWidget(params_card)

        footer = QHBoxLayout()
        footer.setSpacing(10)
        footer.addStretch(1)
        self.btn_cancel = QPushButton('取消', self)
        self.btn_cancel.setCursor(Qt.PointingHandCursor)
        self.btn_cancel.setFixedHeight(40)
        self.btn_cancel.setMinimumWidth(100)
        self.btn_cancel.setStyleSheet(
            """
            QPushButton {
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.18);
                border-radius: 14px;
                color: #dbe4fa;
                font-weight: 700;
            }
            QPushButton:hover { background: rgba(255, 255, 255, 0.10); }
            """
        )
        self.btn_ok = self._make_primary_btn('一键接通', '#5d7cff', '#4aa5ff', height=40, min_width=140)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_ok.clicked.connect(self.accept)
        footer.addWidget(self.btn_cancel)
        footer.addWidget(self.btn_ok)
        root.addLayout(footer)

        self.btn_scan.clicked.connect(self._on_scan)
        self.btn_add_url.clicked.connect(self._on_add_url)

        self._refresh_list()

    def _make_primary_btn(self, text: str, c1: str, c2: str, height: int = 36, min_width: int = 130) -> QPushButton:
        btn = QPushButton(text, self)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setFixedHeight(height)
        btn.setMinimumWidth(min_width)
        btn.setStyleSheet(
            f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {c1}, stop:1 {c2});
                border: 1px solid rgba(255, 255, 255, 0.22);
                border-radius: 14px;
                color: white;
                font-size: 13px;
                font-weight: 800;
                padding: 0 16px;
            }}
            QPushButton:hover {{ border: 1px solid rgba(255, 255, 255, 0.45); }}
            QPushButton:pressed {{ background: {c1}; }}
            """
        )
        return btn

    def _clear_rows(self):
        for row in self._rows:
            row.setParent(None)
            row.deleteLater()
        self._rows = []

    def _refresh_list(self):
        self._clear_rows()
        self.empty_label.setParent(None)

        for i, cam in enumerate(self.cameras):
            row = CameraRow(i, cam, self.scroll_container)
            row.toggled.connect(self._on_row_toggled)
            row.rename_requested.connect(self._on_rename)
            row.remove_requested.connect(self._on_remove)
            self.rows_layout.insertWidget(self.rows_layout.count() - 1, row)
            self._rows.append(row)

        if not self.cameras:
            self.rows_layout.insertWidget(0, self.empty_label)

        enabled = sum(1 for c in self.cameras if c.get('enabled', True))
        self.count_label.setText(f'共 {len(self.cameras)} 台  ·  已启用 {enabled} 台')

    def _on_row_toggled(self, idx: int, checked: bool):
        if 0 <= idx < len(self.cameras):
            self.cameras[idx]['enabled'] = checked
        enabled = sum(1 for c in self.cameras if c.get('enabled', True))
        self.count_label.setText(f'共 {len(self.cameras)} 台  ·  已启用 {enabled} 台')

    def _on_scan(self):
        found = []
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
                found.append(i)

        if not found:
            QMessageBox.information(self, '扫描结果', '未检测到本机摄像头')
            return

        existing_values = {str(c['value']) for c in self.cameras if c.get('type') == 'camera'}
        added = 0
        for idx in found:
            if str(idx) in existing_values:
                continue
            self.cameras.append({
                'type': 'camera',
                'value': idx,
                'name': f'本机摄像头 {idx}',
                'enabled': True,
            })
            added += 1
        self._refresh_list()
        QMessageBox.information(self, '扫描结果', f'检测到 {len(found)} 个设备，新增 {added} 个')

    def _on_add_url(self):
        url, ok = QInputDialog.getText(
            self,
            '添加网络摄像头',
            '请输入 RTSP / HTTP 地址：',
            text='rtsp://'
        )
        if not ok or not url.strip():
            return
        url = url.strip()
        default_name = f'网络摄像头 {len(self.cameras) + 1}'
        name, ok = QInputDialog.getText(self, '命名', '为该摄像头命名：', text=default_name)
        if not ok:
            return
        name = name.strip() or default_name
        self.cameras.append({'type': 'camera_url', 'value': url, 'name': name, 'enabled': True})
        self._refresh_list()

    def _on_rename(self, idx: int):
        if not (0 <= idx < len(self.cameras)):
            return
        cam = self.cameras[idx]
        name, ok = QInputDialog.getText(self, '重命名', '新名称：', text=cam['name'])
        if not ok:
            return
        cam['name'] = name.strip() or cam['name']
        self._refresh_list()

    def _on_remove(self, idx: int):
        if not (0 <= idx < len(self.cameras)):
            return
        confirm = QMessageBox.question(
            self,
            '删除摄像头',
            f"确定删除 “{self.cameras[idx]['name']}” 吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return
        del self.cameras[idx]
        self._refresh_list()

    def result_cameras(self) -> List[dict]:
        return self.cameras

    def selected_resolution(self) -> Tuple[int, int]:
        w, h = self.combo_res.currentText().split('x')
        return int(w), int(h)

    def selected_fps(self) -> int:
        return int(self.spin_fps.value())


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
        self.infer_every_n = 2
        self.infer_size = 640

        self._save_dir: Optional[Path] = None
        self._csv_file = None
        self._csv_writer = None
        self._video_writers: Dict[str, cv2.VideoWriter] = {}
        self._frame_count = 0
        self._alarm_frames_saved = 0
        self._total_detections: Dict[str, int] = {}
        self._session_start: Optional[str] = None

    def _init_save_session(self):
        if self.is_save != Qt.Checked:
            return
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._save_dir = Path('runs/detect') / f'exp_{ts}'
        self._save_dir.mkdir(parents=True, exist_ok=True)
        (self._save_dir / 'alarm_frames').mkdir(exist_ok=True)

        csv_path = self._save_dir / 'detections.csv'
        self._csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            '时间', '帧号', '源名称', '类别', '置信度', 'x1', 'y1', 'x2', 'y2'
        ])

        self._frame_count = 0
        self._alarm_frames_saved = 0
        self._total_detections = {}
        self._session_start = datetime.now().isoformat()

    def _save_detection_row(self, src_name: str, class_name: str, conf: float, xyxy):
        if self._csv_writer is None:
            return
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        self._csv_writer.writerow([
            now, self._frame_count, src_name, class_name, f'{conf:.4f}', x1, y1, x2, y2
        ])
        self._total_detections[class_name] = self._total_detections.get(class_name, 0) + 1

    def _save_alarm_frame(self, frame: np.ndarray, src_name: str):
        if self._save_dir is None:
            return
        if self._alarm_frames_saved >= 500:
            return
        self._alarm_frames_saved += 1
        ts = datetime.now().strftime('%H%M%S_%f')[:-3]
        filename = f'{src_name}_{ts}.jpg'
        filepath = self._save_dir / 'alarm_frames' / filename
        frame_copy = frame.copy()
        threading.Thread(target=cv2.imwrite, args=(str(filepath), frame_copy), daemon=True).start()

    def _get_video_writer(self, src_name: str, width: int, height: int) -> Optional[cv2.VideoWriter]:
        if self._save_dir is None:
            return None
        if src_name in self._video_writers:
            return self._video_writers[src_name]
        safe_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in src_name)
        filepath = self._save_dir / f'{safe_name}.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(filepath), fourcc, self.target_fps, (width, height))
        if writer.isOpened():
            self._video_writers[src_name] = writer
            return writer
        return None

    def _close_save_session(self):
        for writer in self._video_writers.values():
            try:
                writer.release()
            except Exception:
                pass
        self._video_writers = {}

        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

        if self._save_dir is not None:
            summary = {
                'session_start': self._session_start,
                'session_end': datetime.now().isoformat(),
                'total_frames': self._frame_count,
                'alarm_frames_saved': self._alarm_frames_saved,
                'detection_counts': self._total_detections,
                'sources': [s.get('name', '') for s in self.sources_config],
                'resolution': list(self.target_resolution),
                'fps': self.target_fps,
                'conf_threshold': self.conf,
                'iou_threshold': self.iou,
                'weights': self.weights,
            }
            summary_path = self._save_dir / 'summary.json'
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=4, ensure_ascii=False)

            self.status_text.emit(f'结果已保存至: {self._save_dir}')
        self._save_dir = None

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
            draw_text_cn(resized, name, (14, 10), font_size=26, color_bgr=(0, 255, 255))
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

        src_name = Path(image_path).stem
        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)
                label = f'{names[c]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(c, True))
                detect_counts[names[c]] = detect_counts.get(names[c], 0) + 1
                self._save_detection_row(src_name, names[c], float(conf), xyxy)

        result_img = annotator.result()

        if self.is_save == Qt.Checked and self._save_dir is not None:
            save_path = self._save_dir / f'{src_name}_result.jpg'
            cv2.imwrite(str(save_path), result_img)

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

        self._init_save_session()

        # 处理图片检测
        if self.sources_config[0]['type'] == 'image':
            try:
                device = select_device('')
                half = device.type != 'cpu'
                model = DetectMultiBackend(self.weights, device=device, dnn=False, data='data/custom.yaml', fp16=half)
                stride, names = model.stride, model.names
                imgsz = check_img_size((self.infer_size, self.infer_size), s=stride)
                if half:
                    model.half()
                model.warmup(imgsz=(1, 3, *imgsz))

                image_path = self.sources_config[0]['value']
                self.status_text.emit(f'正在检测图片: {Path(image_path).name}')
                self._frame_count = 1
                self._detect_image(image_path, device, model, imgsz, names)

            except Exception as e:
                self.alarm_state.emit(False)
                self.status_text.emit(f'错误: {e}')
                print(f'图片检测错误: {e}')
            finally:
                self._close_save_session()
            return

        capture = None
        try:
            device = select_device('')
            half = device.type != 'cpu'
            model = DetectMultiBackend(self.weights, device=device, dnn=False, data='data/custom.yaml', fp16=half)
            stride, names = model.stride, model.names
            imgsz = check_img_size((self.infer_size, self.infer_size), s=stride)
            if half:
                model.half()
            model.warmup(imgsz=(1, 3, *imgsz))

            active_sources = self.sources_config[: max(1, self.active_slots)]
            capture = MultiSourceCapture(active_sources, self.target_resolution, self.target_fps)

            speak_thread = None
            fixed_text = '警报，警报，发现火情，请迅速处理！'
            infer_time_ms = 0.0
            loop_index = 0
            last_annotated: List[np.ndarray] = []
            last_counts: Dict[str, int] = {}
            last_alarm = False

            while not self.end_loop:
                oks, frames = capture.read()
                if not any(oks):
                    break

                loop_index += 1
                do_infer = (loop_index % self.infer_every_n == 0) or loop_index == 1

                if not do_infer and last_annotated:
                    annotated_frames = last_annotated
                    merged_counts = last_counts
                    frame_has_alarm = last_alarm
                else:
                    annotated_frames = []
                    merged_counts = {}
                    frame_has_alarm = False

                    for ok, frame, src in zip(oks, frames, active_sources):
                        if not ok or frame is None:
                            blank = np.zeros((360, 640, 3), dtype=np.uint8)
                            draw_text_cn(blank, f"{src['name']} 无信号", (20, 20), font_size=28, color_bgr=(0, 0, 255))
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
                                self._save_detection_row(src['name'], names[c], float(conf), xyxy)

                        out = annotator.result()
                        status = f"{src['name']} | {im0.shape[1]}x{im0.shape[0]} | {self.target_fps} FPS"
                        draw_text_cn(out, status, (12, im0.shape[0] - 32), font_size=20, color_bgr=(0, 255, 255))
                        annotated_frames.append(out)

                        if self.is_save == Qt.Checked:
                            writer = self._get_video_writer(src['name'], out.shape[1], out.shape[0])
                            if writer is not None:
                                writer.write(out)

                    last_annotated = annotated_frames
                    last_counts = merged_counts
                    last_alarm = frame_has_alarm

                mosaic = self._make_mosaic(annotated_frames, [s['name'] for s in active_sources])
                self.send_img.emit(mosaic)
                self.send_detectinfo_dic.emit(merged_counts)
                self.detect_speed.emit(str(round(infer_time_ms, 1)))

                self._frame_count += 1

                if frame_has_alarm and self.is_save == Qt.Checked:
                    self._save_alarm_frame(mosaic, 'mosaic')

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
            self._close_save_session()


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
        self.saved_cameras: List[dict] = []

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
        button.setFixedSize(56, 56)
        button.setIconSize(QSize(24, 24))
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
        card = QFrame()
        card.setStyleSheet(
            """
            QFrame {
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 18px;
            }
            """
        )
        card.setMinimumWidth(88)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(6, 8, 6, 8)
        layout.setSpacing(6)

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

        outer_layout = QVBoxLayout(self.left_panel)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        self.left_scroll = QScrollArea(self.left_panel)
        self.left_scroll.setWidgetResizable(True)
        self.left_scroll.setFrameShape(QFrame.NoFrame)
        self.left_scroll.setStyleSheet(
            """
            QScrollArea { background: transparent; border: none; }
            QScrollBar:vertical {
                background: rgba(255, 255, 255, 0.02);
                width: 8px; border-radius: 4px; margin: 4px 2px;
            }
            QScrollBar::handle:vertical {
                background: rgba(123, 156, 255, 0.30);
                border-radius: 4px; min-height: 20px;
            }
            QScrollBar::handle:vertical:hover { background: rgba(123, 156, 255, 0.50); }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            """
        )
        scroll_content = QWidget()
        scroll_content.setStyleSheet('background: transparent;')
        left_layout = QVBoxLayout(scroll_content)
        left_layout.setContentsMargins(18, 18, 18, 18)
        left_layout.setSpacing(12)
        self.left_scroll.setWidget(scroll_content)
        outer_layout.addWidget(self.left_scroll)

        panel_title = QLabel('控制面板', scroll_content)
        panel_title.setStyleSheet('color:#ffffff; font-size:16px; font-weight:800; letter-spacing:1px;')
        left_layout.addWidget(panel_title)

        left_layout.addWidget(self._create_section_title('模型权重', scroll_content))
        weights_wrap = QWidget(scroll_content)
        weights_layout = QHBoxLayout(weights_wrap)
        weights_layout.setContentsMargins(0, 0, 0, 0)
        weights_layout.setSpacing(8)
        self.lineEdit_weights.setParent(weights_wrap)
        self.lineEdit_weights.setPlaceholderText('请选择 .pt 权重文件')
        self.pushButton_weights.setParent(weights_wrap)
        self.pushButton_weights.setText('浏览')
        self.pushButton_weights.setCursor(Qt.PointingHandCursor)
        weights_layout.addWidget(self.lineEdit_weights, 1)
        weights_layout.addWidget(self.pushButton_weights)
        left_layout.addWidget(weights_wrap)

        left_layout.addWidget(self._create_section_title('检测源', scroll_content))
        source_wrap = QWidget(scroll_content)
        source_layout = QHBoxLayout(source_wrap)
        source_layout.setContentsMargins(0, 0, 0, 0)
        source_layout.setSpacing(8)
        for attr, caption, tooltip in self.ACTION_META:
            source_layout.addWidget(self._create_action_item(getattr(self, attr), caption, tooltip))
        left_layout.addWidget(source_wrap)

        self.pushButton_manage_cams = QPushButton('管理摄像头', scroll_content)
        self.pushButton_manage_cams.setCursor(Qt.PointingHandCursor)
        self.pushButton_manage_cams.setToolTip('扫描本机 / 添加网络摄像头 / 重命名 / 删除 / 选择启用')
        self.pushButton_manage_cams.setFixedHeight(36)
        self.pushButton_manage_cams.setStyleSheet(
            """
            QPushButton {
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.14);
                border-radius: 12px;
                color: #dde8ff;
                font-weight: 600;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 0.10);
                border: 1px solid rgba(123, 156, 255, 0.40);
            }
            """
        )
        left_layout.addWidget(self.pushButton_manage_cams)

        left_layout.addWidget(self._create_section_title('识别参数', scroll_content))
        param_card = QFrame(scroll_content)
        param_card.setStyleSheet(
            'QFrame { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 16px; }'
        )
        param_layout = QHBoxLayout(param_card)
        param_layout.setContentsMargins(12, 10, 12, 10)
        param_layout.setSpacing(10)

        conf_box = QWidget(param_card)
        conf_layout = QVBoxLayout(conf_box)
        conf_layout.setContentsMargins(0, 0, 0, 0)
        conf_layout.setSpacing(4)
        conf_title = QLabel('置信度', conf_box)
        conf_title.setStyleSheet('color:#c8d7f4; font-size:12px; font-weight:700;')
        self.doubleSpinBox_conf.setParent(conf_box)
        self.doubleSpinBox_conf.setMinimumWidth(120)
        conf_layout.addWidget(conf_title)
        conf_layout.addWidget(self.doubleSpinBox_conf)

        iou_box = QWidget(param_card)
        iou_layout = QVBoxLayout(iou_box)
        iou_layout.setContentsMargins(0, 0, 0, 0)
        iou_layout.setSpacing(4)
        iou_title = QLabel('IOU 阈值', iou_box)
        iou_title.setStyleSheet('color:#c8d7f4; font-size:12px; font-weight:700;')
        self.doubleSpinBox_iou.setParent(iou_box)
        self.doubleSpinBox_iou.setMinimumWidth(120)
        iou_layout.addWidget(iou_title)
        iou_layout.addWidget(self.doubleSpinBox_iou)

        param_layout.addWidget(conf_box, 1)
        param_layout.addWidget(iou_box, 1)
        left_layout.addWidget(param_card)

        left_layout.addWidget(self._create_section_title('性能调优', scroll_content))
        perf_card = QFrame(scroll_content)
        perf_card.setStyleSheet(
            'QFrame { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 16px; }'
        )
        perf_layout = QHBoxLayout(perf_card)
        perf_layout.setContentsMargins(12, 10, 12, 10)
        perf_layout.setSpacing(10)

        infer_size_box = QWidget(perf_card)
        infer_size_layout = QVBoxLayout(infer_size_box)
        infer_size_layout.setContentsMargins(0, 0, 0, 0)
        infer_size_layout.setSpacing(4)
        infer_size_title = QLabel('推理尺寸', infer_size_box)
        infer_size_title.setStyleSheet('color:#c8d7f4; font-size:12px; font-weight:700;')
        self.combo_infer_size = QComboBox(infer_size_box)
        self.combo_infer_size.addItems(['320', '416', '512', '640'])
        self.combo_infer_size.setCurrentText(str(self.my_thread.infer_size))
        self.combo_infer_size.setMinimumWidth(100)
        self.combo_infer_size.setMinimumHeight(36)
        self.combo_infer_size.setStyleSheet(
            'QComboBox { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.10); '
            'border-radius: 10px; color: #f7f9ff; padding: 6px 10px; font-size: 13px; }'
            'QComboBox QAbstractItemView { background: #0f1a30; color: #eef3ff; '
            'selection-background-color: rgba(86,117,255,0.55); border: 1px solid rgba(255,255,255,0.10); }'
        )
        infer_size_layout.addWidget(infer_size_title)
        infer_size_layout.addWidget(self.combo_infer_size)

        skip_box = QWidget(perf_card)
        skip_layout = QVBoxLayout(skip_box)
        skip_layout.setContentsMargins(0, 0, 0, 0)
        skip_layout.setSpacing(4)
        skip_title = QLabel('跳帧推理', skip_box)
        skip_title.setStyleSheet('color:#c8d7f4; font-size:12px; font-weight:700;')
        self.combo_skip_frame = QComboBox(skip_box)
        self.combo_skip_frame.addItems(['每帧推理', '每2帧', '每3帧', '每4帧'])
        self.combo_skip_frame.setCurrentIndex(max(0, self.my_thread.infer_every_n - 1))
        self.combo_skip_frame.setMinimumWidth(100)
        self.combo_skip_frame.setMinimumHeight(36)
        self.combo_skip_frame.setStyleSheet(
            'QComboBox { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.10); '
            'border-radius: 10px; color: #f7f9ff; padding: 6px 10px; font-size: 13px; }'
            'QComboBox QAbstractItemView { background: #0f1a30; color: #eef3ff; '
            'selection-background-color: rgba(86,117,255,0.55); border: 1px solid rgba(255,255,255,0.10); }'
        )
        skip_layout.addWidget(skip_title)
        skip_layout.addWidget(self.combo_skip_frame)

        perf_layout.addWidget(infer_size_box, 1)
        perf_layout.addWidget(skip_box, 1)
        left_layout.addWidget(perf_card)

        left_layout.addWidget(self._create_section_title('运行选项', scroll_content))
        option_row = QHBoxLayout()
        option_row.setSpacing(16)
        self.checkBox_is_alarm.setParent(scroll_content)
        self.checkBox_is_alarm.setText('报警联动')
        self.checkBox_is_save.setParent(scroll_content)
        self.checkBox_is_save.setText('保存结果')
        option_row.addWidget(self.checkBox_is_alarm)
        option_row.addWidget(self.checkBox_is_save)
        option_row.addStretch(1)
        left_layout.addLayout(option_row)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self.pushButton_open_output = QPushButton('打开输出目录', scroll_content)
        self.pushButton_open_output.setCursor(Qt.PointingHandCursor)
        self.pushButton_open_output.setToolTip('打开保存检测结果的文件夹 (runs/detect)')
        self.pushButton_open_output.setFixedHeight(36)
        self.pushButton_open_output.setStyleSheet(
            """
            QPushButton {
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.14);
                border-radius: 12px;
                color: #dde8ff;
                font-weight: 600;
                padding: 0 14px;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 0.10);
                border: 1px solid rgba(123, 156, 255, 0.40);
            }
            """
        )
        btn_row.addWidget(self.pushButton_open_output)
        btn_row.addStretch(1)
        left_layout.addLayout(btn_row)

        left_layout.addStretch(1)

        self.pushButton_zz.setParent(scroll_content)
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
        self.lineEdit_weights.setMinimumHeight(40)
        self.lineEdit_weights.setStyleSheet(
            'QLineEdit { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.10); '
            'border-radius: 10px; color: #f6f8ff; padding: 8px 12px; font-size: 13px; }'
            'QLineEdit:focus { border: 1px solid rgba(117,154,255,0.92); }'
        )
        self.pushButton_weights.setMinimumSize(72, 40)
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
                border-radius: 10px;
                color: #f7f9ff;
                padding: 6px 10px;
                font-size: 14px;
                min-width: 90px;
                min-height: 32px;
            }
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                width: 24px;
                border-radius: 6px;
                background: rgba(255, 255, 255, 0.06);
                margin: 3px;
            }
            QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
                background: rgba(255, 255, 255, 0.12);
            }
        """
        self.doubleSpinBox_conf.setStyleSheet(spin_style)
        self.doubleSpinBox_iou.setStyleSheet(spin_style)
        self.doubleSpinBox_conf.setMinimumHeight(40)
        self.doubleSpinBox_iou.setMinimumHeight(40)
        self.doubleSpinBox_conf.setMinimumWidth(120)
        self.doubleSpinBox_iou.setMinimumWidth(120)
        self.doubleSpinBox_conf.setDecimals(2)
        self.doubleSpinBox_iou.setDecimals(2)
        self.doubleSpinBox_conf.setSingleStep(0.01)
        self.doubleSpinBox_iou.setSingleStep(0.01)

        self.pushButton_zz.setMinimumHeight(44)
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
        self.pushButton_manage_cams.clicked.connect(self.manage_cameras)
        self.doubleSpinBox_conf.valueChanged.connect(self.change_conf)
        self.doubleSpinBox_iou.valueChanged.connect(self.change_iou)
        self.pushButton_weights.clicked.connect(self.select_weights)
        self.pushButton_zz.clicked.connect(self.clean)
        self.pushButton_open_output.clicked.connect(self.open_output_dir)
        self.checkBox_is_alarm.clicked.connect(self.is_alarm)
        self.checkBox_is_save.clicked.connect(self.is_save)
        self.combo_infer_size.currentTextChanged.connect(self.change_infer_size)
        self.combo_skip_frame.currentIndexChanged.connect(self.change_skip_frame)

    def change_infer_size(self, text: str):
        try:
            self.my_thread.infer_size = int(text)
        except ValueError:
            pass

    def change_skip_frame(self, index: int):
        self.my_thread.infer_every_n = max(1, index + 1)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_responsive_layout()

    def update_responsive_layout(self):
        margin = 22
        top_y = 18
        header_h = 52
        body_top = 92
        gap = 18
        panel_w = min(420, max(360, int(self.width() * 0.28)))
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

    def open_output_dir(self):
        output_dir = Path('runs/detect')
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        abs_path = str(output_dir.resolve())
        if sys.platform == 'win32':
            os.startfile(abs_path)
        elif sys.platform == 'darwin':
            os.system(f'open "{abs_path}"')
        else:
            os.system(f'xdg-open "{abs_path}"')

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

    def _scan_local_cameras(self) -> List[int]:
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
        return device_list

    def _auto_populate_cameras_if_empty(self) -> bool:
        if self.saved_cameras:
            return False
        found = self._scan_local_cameras()
        for i, idx in enumerate(found):
            self.saved_cameras.append({
                'type': 'camera',
                'value': idx,
                'name': f'本机摄像头 {idx}',
                'enabled': i == 0,
            })
        return bool(found)

    def manage_cameras(self):
        if self.my_thread.isRunning():
            self.clean()

        dialog = CameraManagerDialog(
            self.saved_cameras,
            self.my_thread.target_resolution,
            self.my_thread.target_fps,
            parent=self,
        )
        if dialog.exec() != QDialog.Accepted:
            return

        self.saved_cameras = dialog.result_cameras()
        self.my_thread.target_resolution = dialog.selected_resolution()
        self.my_thread.target_fps = dialog.selected_fps()
        self._start_enabled_cameras(silent=False)

    def open_camera(self):
        """一键接通：已保存的启用项直接启动；若从未保存过则自动扫描本机并启动。"""
        if self.my_thread.isRunning():
            self.clean()

        if not self.saved_cameras:
            scanned = self._auto_populate_cameras_if_empty()
            if not scanned:
                QMessageBox.information(
                    self,
                    '没有可用摄像头',
                    '未检测到本机摄像头。\n点击"管理摄像头"添加网络摄像头（RTSP / HTTP）后再试。'
                )
                self.manage_cameras()
                return
            self.statusbar.showMessage(f'已自动发现 {len(self.saved_cameras)} 个本机摄像头，正在接通第 1 个')

        enabled = [c for c in self.saved_cameras if c.get('enabled', True)]
        if not enabled:
            QMessageBox.information(
                self,
                '未启用任何摄像头',
                '请在"管理摄像头"中勾选至少一个需要启用的摄像头。'
            )
            self.manage_cameras()
            return

        self._start_enabled_cameras(silent=False)

    def _start_enabled_cameras(self, silent: bool = False):
        enabled = [c for c in self.saved_cameras if c.get('enabled', True)]
        if not enabled:
            if not silent:
                QMessageBox.information(self, '提示', '请至少启用一个摄像头')
            return

        self.my_thread.sources_config = [
            {'type': c['type'], 'value': c['value'], 'name': c['name']}
            for c in enabled
        ]
        self.my_thread.active_slots = len(enabled)
        self.my_thread.source_name = f'摄像头源 x {len(enabled)}'
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
            self.saved_cameras = [dict(c) for c in loaded_config.get('cameras', [])]
            self.my_thread.infer_size = int(loaded_config.get('infer_size', 640))
            self.my_thread.infer_every_n = int(loaded_config.get('infer_every_n', 2))
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
        self.combo_infer_size.setCurrentText(str(self.my_thread.infer_size))
        self.combo_skip_frame.setCurrentIndex(max(0, self.my_thread.infer_every_n - 1))

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
                'cameras': self.saved_cameras,
                'infer_size': self.my_thread.infer_size,
                'infer_every_n': self.my_thread.infer_every_n,
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
