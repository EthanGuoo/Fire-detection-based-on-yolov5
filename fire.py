# -*- coding: utf-8 -*-
from PySide6.QtCore import QCoreApplication, QMetaObject, QRect, QSize, Qt
from PySide6.QtGui import QCursor, QFont, QIcon
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QStatusBar,
    QTextBrowser,
    QWidget,
)


class Ui_MainWindow(object):
    def setupUi(self, MainWindow: QMainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName("MainWindow")

        MainWindow.resize(1440, 860)
        MainWindow.setMinimumSize(QSize(1320, 780))
        MainWindow.setMaximumSize(QSize(2560, 1440))

        window_icon = QIcon()
        window_icon.addFile("icon/1000LOGO.jpg", QSize(), QIcon.Normal, QIcon.Off)
        MainWindow.setWindowIcon(window_icon)

        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.setStyleSheet("""
            QWidget#centralwidget {
                background: #0b1120;
                color: #eef4ff;
            }
        """)

        self.label_title = QLabel(self.centralwidget)
        self.label_title.setObjectName("label_title")
        self.label_title.setGeometry(QRect(22, 18, 1396, 52))
        title_font = QFont("Microsoft YaHei", 20, QFont.Bold)
        self.label_title.setFont(title_font)
        self.label_title.setAlignment(Qt.AlignCenter)

        self.label_speed = QLabel(self.centralwidget)
        self.label_speed.setObjectName("label_speed")
        self.label_speed.setGeometry(QRect(1292, 26, 108, 32))
        speed_font = QFont("Microsoft YaHei", 11, QFont.Bold)
        self.label_speed.setFont(speed_font)
        self.label_speed.setAlignment(Qt.AlignCenter)

        self.lineEdit_weights = QLineEdit(self.centralwidget)
        self.lineEdit_weights.setObjectName("lineEdit_weights")
        self.lineEdit_weights.setGeometry(QRect(34, 116, 214, 46))

        self.pushButton_weights = QPushButton(self.centralwidget)
        self.pushButton_weights.setObjectName("pushButton_weights")
        self.pushButton_weights.setGeometry(QRect(258, 116, 86, 46))
        self.pushButton_weights.setCursor(QCursor(Qt.PointingHandCursor))

        self.pushButton_img = QPushButton(self.centralwidget)
        self.pushButton_img.setObjectName("pushButton_img")
        self.pushButton_img.setGeometry(QRect(34, 196, 68, 68))
        self.pushButton_img.setCursor(QCursor(Qt.PointingHandCursor))
        img_icon = QIcon()
        img_icon.addFile("icon/照片_pic.png", QSize(), QIcon.Normal, QIcon.Off)
        self.pushButton_img.setIcon(img_icon)
        self.pushButton_img.setIconSize(QSize(28, 28))

        self.pushButton_video = QPushButton(self.centralwidget)
        self.pushButton_video.setObjectName("pushButton_video")
        self.pushButton_video.setGeometry(QRect(138, 196, 68, 68))
        self.pushButton_video.setCursor(QCursor(Qt.PointingHandCursor))
        video_icon = QIcon()
        video_icon.addFile("icon/视频_video.png", QSize(), QIcon.Normal, QIcon.Off)
        self.pushButton_video.setIcon(video_icon)
        self.pushButton_video.setIconSize(QSize(28, 28))

        self.pushButton_camera = QPushButton(self.centralwidget)
        self.pushButton_camera.setObjectName("pushButton_camera")
        self.pushButton_camera.setGeometry(QRect(242, 196, 68, 68))
        self.pushButton_camera.setCursor(QCursor(Qt.PointingHandCursor))
        camera_icon = QIcon()
        camera_icon.addFile("icon/摄像头_camera-one.png", QSize(), QIcon.Normal, QIcon.Off)
        self.pushButton_camera.setIcon(camera_icon)
        self.pushButton_camera.setIconSize(QSize(28, 28))

        self.doubleSpinBox_conf = QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_conf.setObjectName("doubleSpinBox_conf")
        self.doubleSpinBox_conf.setGeometry(QRect(34, 308, 150, 48))
        self.doubleSpinBox_conf.setDecimals(2)
        self.doubleSpinBox_conf.setMaximum(1.0)
        self.doubleSpinBox_conf.setSingleStep(0.01)
        self.doubleSpinBox_conf.setValue(0.25)

        self.doubleSpinBox_iou = QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_iou.setObjectName("doubleSpinBox_iou")
        self.doubleSpinBox_iou.setGeometry(QRect(194, 308, 150, 48))
        self.doubleSpinBox_iou.setDecimals(2)
        self.doubleSpinBox_iou.setMaximum(1.0)
        self.doubleSpinBox_iou.setSingleStep(0.01)
        self.doubleSpinBox_iou.setValue(0.45)

        self.checkBox_is_alarm = QCheckBox(self.centralwidget)
        self.checkBox_is_alarm.setObjectName("checkBox_is_alarm")
        self.checkBox_is_alarm.setGeometry(QRect(38, 392, 140, 26))

        self.checkBox_is_save = QCheckBox(self.centralwidget)
        self.checkBox_is_save.setObjectName("checkBox_is_save")
        self.checkBox_is_save.setGeometry(QRect(188, 392, 140, 26))

        self.pushButton_zz = QPushButton(self.centralwidget)
        self.pushButton_zz.setObjectName("pushButton_zz")
        self.pushButton_zz.setGeometry(QRect(34, 450, 310, 50))
        self.pushButton_zz.setCursor(QCursor(Qt.PointingHandCursor))
        stop_icon = QIcon()
        stop_icon.addFile("icon/zz.png", QSize(), QIcon.Normal, QIcon.Off)
        self.pushButton_zz.setIcon(stop_icon)
        self.pushButton_zz.setIconSize(QSize(22, 22))

        self.label_result = QLabel(self.centralwidget)
        self.label_result.setObjectName("label_result")
        self.label_result.setGeometry(QRect(400, 92, 1016, 500))
        self.label_result.setAlignment(Qt.AlignCenter)
        self.label_result.setScaledContents(False)

        self.textBrowser_result = QTextBrowser(self.centralwidget)
        self.textBrowser_result.setObjectName("textBrowser_result")
        self.textBrowser_result.setGeometry(QRect(400, 610, 1016, 216))

        self.pushButton_logo = QPushButton(self.centralwidget)
        self.pushButton_logo.setObjectName("pushButton_logo")
        self.pushButton_logo.setGeometry(QRect(0, 0, 1, 1))
        self.pushButton_logo.setVisible(False)
        self.pushButton_logo.setEnabled(False)

        # 兼容原工程中的旧控件名，保留但默认隐藏，避免联调时出现属性缺失
        self.label_weights = QLabel(self.centralwidget)
        self.label_weights.setObjectName("label_weights")
        self.label_weights.setGeometry(QRect(0, 0, 1, 1))
        self.label_weights.hide()

        self.label_source = QLabel(self.centralwidget)
        self.label_source.setObjectName("label_source")
        self.label_source.setGeometry(QRect(0, 0, 1, 1))
        self.label_source.hide()

        self.label_conf = QLabel(self.centralwidget)
        self.label_conf.setObjectName("label_conf")
        self.label_conf.setGeometry(QRect(0, 0, 1, 1))
        self.label_conf.hide()

        self.label_iou = QLabel(self.centralwidget)
        self.label_iou.setObjectName("label_iou")
        self.label_iou.setGeometry(QRect(0, 0, 1, 1))
        self.label_iou.hide()

        self.label_alarm_level = QLabel(self.centralwidget)
        self.label_alarm_level.setObjectName("label_alarm_level")
        self.label_alarm_level.setGeometry(QRect(0, 0, 1, 1))
        self.label_alarm_level.hide()

        self.label_is_save = QLabel(self.centralwidget)
        self.label_is_save.setObjectName("label_is_save")
        self.label_is_save.setGeometry(QRect(0, 0, 1, 1))
        self.label_is_save.hide()

        self.pushButton_speed = QPushButton(self.centralwidget)
        self.pushButton_speed.setObjectName("pushButton_speed")
        self.pushButton_speed.setGeometry(QRect(0, 0, 1, 1))
        self.pushButton_speed.hide()

        self.gridLayoutWidget = QWidget(self.centralwidget)
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayoutWidget.setGeometry(QRect(0, 0, 1, 1))
        self.gridLayoutWidget.hide()

        self.formLayoutWidget = QWidget(self.centralwidget)
        self.formLayoutWidget.setObjectName("formLayoutWidget")
        self.formLayoutWidget.setGeometry(QRect(0, 0, 1, 1))
        self.formLayoutWidget.hide()

        MainWindow.setCentralWidget(self.centralwidget)

        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow: QMainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", "智能火焰识别控制台", None))
        self.label_title.setText(QCoreApplication.translate("MainWindow", "智能火焰识别控制台", None))
        self.label_speed.setText(QCoreApplication.translate("MainWindow", "0 ms", None))
        self.lineEdit_weights.setPlaceholderText(QCoreApplication.translate("MainWindow", "请选择模型权重文件", None))
        self.pushButton_weights.setText(QCoreApplication.translate("MainWindow", "浏览", None))

        self.pushButton_img.setToolTip(QCoreApplication.translate("MainWindow", "选择图片进行检测", None))
        self.pushButton_img.setText("")
        self.pushButton_video.setToolTip(QCoreApplication.translate("MainWindow", "选择视频进行检测", None))
        self.pushButton_video.setText("")
        self.pushButton_camera.setToolTip(QCoreApplication.translate("MainWindow", "打开摄像头进行检测", None))
        self.pushButton_camera.setText("")

        self.checkBox_is_alarm.setText(QCoreApplication.translate("MainWindow", "报警联动", None))
        self.checkBox_is_save.setText(QCoreApplication.translate("MainWindow", "保存结果", None))
        self.pushButton_zz.setText(QCoreApplication.translate("MainWindow", "停止 / 复位", None))
        self.textBrowser_result.setPlaceholderText(QCoreApplication.translate("MainWindow", "检测结果信息会显示在这里", None))
        self.label_result.setText("")