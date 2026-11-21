"""Dark simulation dashboard and mouse-based screen calibration."""
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from time import sleep
import csv
import cv2
import mss
import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import (QColor, QFont, QImage, QLinearGradient, QPainter, QPainterPath,
                           QPen, QPixmap, QRadialGradient)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout,
    QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QSizeGrip, QSizePolicy,
    QSpinBox, QDoubleSpinBox, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)
from crash_watcher import StrategyEngine
from crash_watcher.core.enums import BetOutcome, CycleOutcome, StrategyState
from crash_watcher.vision.capture import ScreenCapture
from crash_watcher.vision.recognition import RapidOCRReader, tesseract_reader
from .clicker import AutoClicker
from .config import (DesktopSettings, Region, SETTINGS_PATH, PROJECT_ROOT, MAX_CLICK_DELAY_MS,
                     load_settings, save_settings)
from .final_result import BangRecognizer
from .smart_review import DEFAULT_BASE_URL, DEFAULT_MODEL, chat_completions_url, probe_models
from .worker import MonitorWorker

_FA_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")

STYLE = """
QWidget { background:transparent; color:#e8f4ec; font-family:'Segoe UI','Tahoma'; font-size:14px; }
QMainWindow, QDialog { background:qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #0d1a16, stop:0.45 #122620, stop:1 #0a1411); }
QLabel { background:transparent; }
QScrollArea { background:transparent; border:0; }
QScrollArea > QWidget > QWidget { background:transparent; }
QGroupBox { background:rgba(255,255,255,10); border:1px solid rgba(120,220,175,60); border-radius:18px; margin-top:16px; padding:18px 14px 14px; font-weight:600; }
QGroupBox::title { subcontrol-origin:margin; left:16px; padding:0 8px; color:#8ff0c0; }
QLineEdit,QSpinBox,QDoubleSpinBox,QComboBox,QPlainTextEdit { background:rgba(8,22,18,190); border:1px solid rgba(110,200,160,70); border-radius:10px; padding:8px 10px; min-height:26px; selection-background-color:#2f8f68; selection-color:#f2fff7; }
QLineEdit:hover,QSpinBox:hover,QDoubleSpinBox:hover,QComboBox:hover { border:1px solid rgba(140,230,190,120); }
QLineEdit:focus,QSpinBox:focus,QDoubleSpinBox:focus,QComboBox:focus,QPlainTextEdit:focus { border:1px solid #59d19b; background:rgba(10,30,24,235); }
QComboBox::drop-down { border:0; width:22px; }
QComboBox QAbstractItemView { background:#10241d; border:1px solid #2f6d54; selection-background-color:#2f8f68; padding:4px; }
QSpinBox::up-button,QDoubleSpinBox::up-button { subcontrol-origin:border; subcontrol-position:top right; width:18px; border-left:1px solid rgba(110,200,160,60); border-top-right-radius:9px; background:rgba(90,200,150,35); }
QSpinBox::down-button,QDoubleSpinBox::down-button { subcontrol-origin:border; subcontrol-position:bottom right; width:18px; border-left:1px solid rgba(110,200,160,60); border-bottom-right-radius:9px; background:rgba(90,200,150,20); }
QPushButton { background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 rgba(46,92,74,235), stop:1 rgba(20,44,35,235)); border:1px solid rgba(110,210,165,95); border-radius:12px; padding:10px 16px; font-weight:700; color:#e9fbf1; }
QPushButton:hover { border:1px solid #6fe3ac; background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 rgba(64,122,96,245), stop:1 rgba(28,60,47,245)); }
QPushButton:pressed { background:rgba(18,40,32,255); }
QPushButton:focus { border:2px solid #f5c86b; }
QPushButton:disabled { color:#6b8478; background:rgba(28,44,37,150); border:1px solid rgba(90,130,110,60); }
QPushButton#start { background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #7af0b8, stop:0.55 #38c98d, stop:1 #1d8f60); color:#06231a; border:1px solid #a9ffd4; font-size:15px; }
QPushButton#start:hover { background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #a2ffd0, stop:0.55 #4fdfa3, stop:1 #26a372); }
QPushButton#stop { background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #ff90a6, stop:1 #b3364f); color:#2a0710; border:1px solid #ffc0cd; }
QPushButton#stop:hover { background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #ffb0c1, stop:1 #cc4562); }
QPushButton#pause { background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #ffd98a, stop:1 #b8862b); color:#2b1c03; border:1px solid #ffe9b8; }
QPushButton#resume { background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #8fdcff, stop:1 #2a7fa8); color:#04222e; border:1px solid #c2f0ff; }
QPushButton#reset { background:rgba(60,82,74,170); border:1px solid rgba(160,200,180,90); color:#dfeee6; }
QPushButton#reset:hover { background:rgba(90,120,108,200); }
QPushButton#preset { background:rgba(245,200,107,28); border:1px solid rgba(245,200,107,110); color:#ffe9b5; border-radius:10px; padding:7px 12px; font-weight:700; }
QPushButton#preset:hover { background:rgba(245,200,107,60); border:1px solid #ffd98a; }
QTabWidget::pane { border:1px solid rgba(110,210,165,55); border-radius:18px; background:rgba(8,20,16,120); top:-1px; }
QTabBar::tab { background:rgba(22,46,37,200); color:#a9c7b8; border:1px solid rgba(90,170,135,50); border-radius:12px; padding:10px 18px; margin:2px 3px; font-weight:600; }
QTabBar::tab:hover { color:#eafff2; border:1px solid rgba(140,230,190,110); }
QTabBar::tab:selected { background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #3cc994, stop:1 #1f7d59); color:#04231a; border:1px solid #8bffd0; }
QTableWidget { background:rgba(8,20,16,200); alternate-background-color:rgba(20,42,34,200); border:1px solid rgba(110,210,165,60); border-radius:14px; gridline-color:rgba(110,210,165,40); }
QTableWidget::item { padding:6px; }
QTableWidget::item:selected { background:rgba(60,201,148,90); color:#f2fff8; }
QHeaderView::section { background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #235242, stop:1 #17342a); color:#a8f0cd; padding:8px; border:0; border-right:1px solid rgba(110,210,165,45); font-weight:700; }
QFrame#metric { background:qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 rgba(38,80,64,235), stop:1 rgba(16,36,29,235)); border:1px solid rgba(110,210,165,70); border-radius:16px; }
QFrame#metric:hover { border:1px solid rgba(150,245,200,150); }
QLabel#modeChip { background:rgba(10,26,20,200); border:1px solid rgba(110,210,165,80); border-radius:11px; color:#bcd8ca; font-weight:700; padding:5px 10px; font-size:12px; }
QLabel#modeChip[turnbal="true"] { background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #f0dfa6, stop:1 #b8912f); color:#231a04; border:1px solid #ffe9b8; }
QCheckBox { spacing:10px; padding:4px 0; }
QCheckBox::indicator { width:20px; height:20px; border-radius:6px; border:1px solid rgba(120,220,175,110); background:rgba(8,22,18,220); }
QCheckBox::indicator:hover { border:1px solid #8bffd0; }
QCheckBox::indicator:checked { background:qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #7af0b8, stop:1 #23a06f); border:1px solid #b8ffdb; }
QScrollBar:vertical { background:transparent; width:10px; margin:2px; }
QScrollBar::handle:vertical { background:rgba(110,210,165,110); border-radius:5px; min-height:30px; }
QScrollBar::handle:vertical:hover { background:rgba(140,240,190,170); }
QScrollBar::add-line,QScrollBar::sub-line { height:0; width:0; }
QScrollBar:horizontal { background:transparent; height:10px; margin:2px; }
QScrollBar::handle:horizontal { background:rgba(110,210,165,110); border-radius:5px; min-width:30px; }
QToolTip { background:#12241e; color:#e9fbf1; border:1px solid #3cc994; border-radius:8px; padding:6px 8px; }
"""

class BellToggle(QWidget):
    """زنگِ بالای پنجره: کلیک روی آن حالت ربات را کلاسیک ↔ ترنبال می‌کند.

    در حالت کلاسیک نقره‌ای و ساکن است؛ در ترنبال طلایی می‌شود، هاله می‌گیرد
    و آرام تاب می‌خورد تا حالت فعال از یک نگاه معلوم باشد.
    """
    toggled = Signal()

    def __init__(self, on: bool = False, parent=None) -> None:
        super().__init__(parent)
        self._on = bool(on)
        self._angle = 0.0
        self._step = 1.1
        self.setFixedSize(92, 92)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("کلیک: تغییر حالت ربات بین کلاسیک و ترنبال")
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._swing)
        self._timer.start(45)

    def is_on(self) -> bool:
        return self._on

    def set_on(self, on: bool) -> None:
        self._on = bool(on)
        self.update()

    def _swing(self) -> None:
        if not self._on:
            if abs(self._angle) > 0.2:
                self._angle *= 0.55
                self.update()
            elif self._angle:
                self._angle = 0.0
                self.update()
            return
        self._angle += self._step
        if abs(self._angle) >= 6.0:
            self._step = -self._step
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.toggled.emit()
        super().mousePressEvent(event)

    def paintEvent(self, _):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        width, height = self.width(), self.height()
        if self._on:
            glow = QRadialGradient(QPointF(width / 2, height * 0.58), width * 0.52)
            glow.setColorAt(0.0, QColor(255, 218, 140, 125))
            glow.setColorAt(0.55, QColor(255, 194, 94, 45))
            glow.setColorAt(1.0, QColor(255, 194, 94, 0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(glow)
            painter.drawEllipse(int(width * 0.05), int(height * 0.14),
                                int(width * 0.90), int(height * 0.76))
        painter.save()
        painter.translate(width / 2, 10)
        painter.rotate(self._angle)
        painter.translate(-width / 2, -10)
        base = QColor("#e9d9a8" if self._on else "#b7b5b4")
        gradient = QLinearGradient(0, height * 0.18, 0, height * 0.82)
        gradient.setColorAt(0.0, base.lighter(120))
        gradient.setColorAt(0.55, base)
        gradient.setColorAt(1.0, base.darker(175))
        left, right = width * 0.27, width * 0.73
        top, hip, skirt = height * 0.20, height * 0.52, height * 0.66
        path = QPainterPath()
        path.moveTo(left, hip)
        path.cubicTo(left, top + height * 0.04, width * 0.37, top, width / 2, top)
        path.cubicTo(width * 0.63, top, right, top + height * 0.04, right, hip)
        path.quadTo(right, skirt, width * 0.79, skirt + height * 0.05)
        path.lineTo(width * 0.21, skirt + height * 0.05)
        path.quadTo(left, skirt, left, hip)
        path.closeSubpath()
        painter.setPen(QPen(QColor(255, 255, 255, 80), 1))
        painter.setBrush(gradient)
        painter.drawPath(path)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(base.darker(140))
        painter.drawRoundedRect(QRectF(width / 2 - 5, top - 9, 10, 10), 4, 4)
        painter.setBrush(QColor("#7a6a44") if self._on else QColor("#5f5b56"))
        painter.drawEllipse(QPointF(width / 2, skirt + height * 0.115), 5.0, 5.0)
        painter.restore()


class RegionOverlay(QDialog):
    """Drag a rectangle on a screenshot of the chosen display.

    The main window is hidden before this overlay is created, so the captured
    image shows the live screen without the calibration window on top of it.
    """
    def __init__(self, monitor: int, parent=None) -> None:
        super().__init__(parent)
        self.monitor = monitor
        self.selected: Region | None = None
        self.setWindowTitle("انتخاب ناحیه")
        with mss.MSS() as backend:
            m = dict(backend.monitors[monitor])
            raw = backend.grab({"left": m["left"], "top": m["top"],
                                "width": m["width"], "height": m["height"]})
        self.setGeometry(QRect(m["left"], m["top"], m["width"], m["height"]))
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                            | Qt.WindowType.Tool)
        self.setModal(True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._start: QPoint | None = None
        self._end: QPoint | None = None
        frame = np.asarray(raw)[:, :, :3].copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy()

    def paintEvent(self, _):
        painter = QPainter(self)
        painter.drawImage(self.rect(), self._image)
        painter.fillRect(self.rect(), QColor(6, 9, 14, 90))
        if self._start and self._end:
            rect = QRect(self._start, self._end).normalized()
            painter.drawImage(rect, self._image, rect)
            painter.setPen(QPen(QColor("#e4c000"), 3))
            painter.drawRect(rect)
        painter.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        painter.setPen(QColor("#f4f6fa"))
        painter.drawText(self.rect().adjusted(0, 24, 0, 0),
                         Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                         "مستطیل را با ماوس بکشید — Esc برای انصراف")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.reject()
            return
        super().keyPressEvent(event)

    def mousePressEvent(self, event):
        self._start = event.position().toPoint()
        self._end = self._start
        self.update()

    def mouseMoveEvent(self, event):
        self._end = event.position().toPoint()
        self.update()

    def mouseReleaseEvent(self, event):
        self._end = event.position().toPoint()
        rect = QRect(self._start, self._end).normalized()
        if rect.width() >= 20 and rect.height() >= 20:
            geo = self.geometry()
            self.selected = Region(self.monitor, geo.x()+rect.x(), geo.y()+rect.y(), rect.width(), rect.height())
            self.accept()

class CalibrationWorker(QThread):
    done = Signal(object, object)
    failed = Signal(str)
    def __init__(self, settings: DesktopSettings):
        super().__init__()
        self.settings = settings
    def run(self):
        try:
            if self.settings.tesseract_path:
                import pytesseract
                pytesseract.pytesseract.tesseract_cmd = self.settings.tesseract_path
                pytesseract.get_tesseract_version()
                reader = tesseract_reader
            else:
                reader = RapidOCRReader()
            with ScreenCapture(self.settings.result_region.capture(self.settings.interval_ms)) as capture:
                frame, _ = capture.capture()
            reading = BangRecognizer(reader, self.settings.ocr_confidence).recognize(frame.image)
            self.done.emit(frame.image, reading)
        except Exception as exc:
            self.failed.emit(str(exc))

class SmartProbeWorker(QThread):
    """نسخهٔ /v1/models سرور محلی FreeLLMAPI را بررسی می‌کند."""
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, url: str, key: str, timeout: float = 15.0):
        super().__init__()
        self.url = url
        self.key = key
        self.timeout = timeout

    def run(self):
        try:
            self.done.emit(probe_models(self.url, self.key, self.timeout))
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = load_settings()
        self.strategy = StrategyEngine(self.settings.strategy())
        self.worker: MonitorWorker | None = None
        self.calibration: CalibrationWorker | None = None
        self.history: list[list[str]] = []
        self._preview_image: QImage | None = None
        self.success = self.expired = self.wins = self.losses = 0
        self.confirmed = 0
        self._metric_boxes: list[QFrame] = []
        self._metric_columns = 0
        self.setWindowTitle("Crash Watcher · SIMULATED")
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        self.setMinimumSize(460, 420)
        self.resize(900, 680)
        screen = (self.screen() or QApplication.primaryScreen()).geometry()
        self.move(screen.left(), screen.bottom() - self.height() + 1)
        self.setStyleSheet(STYLE)
        self.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(24, 18, 24, 20)
        header = QHBoxLayout()
        header.setSpacing(14)
        titles = QVBoxLayout()
        titles.setSpacing(2)
        title = QLabel("CRASH WATCHER")
        title.setStyleSheet("font-size:25px;font-weight:800;letter-spacing:2px;"
                            "color:#f2fff8;background:transparent")
        subtitle = QLabel("داشبورد هوشمند — پایش و کلیک")
        subtitle.setStyleSheet("font-size:12px;color:#7fa694;background:transparent")
        subtitle.setWordWrap(True)
        titles.addWidget(title)
        titles.addWidget(subtitle)
        header.addLayout(titles)
        header.addStretch()
        mode_column = QVBoxLayout()
        mode_column.setSpacing(4)
        self.bell = BellToggle(self.settings.bot_mode == "turnbal")
        self.bell.toggled.connect(self.toggle_bot_mode)
        self.mode_chip = QLabel("حالت: کلاسیک")
        self.mode_chip.setObjectName("modeChip")
        self.mode_chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mode_chip.setToolTip("کلیک روی زنگ: تغییر حالت کلاسیک ↔ ترنبال")
        mode_column.addWidget(self.bell, 0, Qt.AlignmentFlag.AlignHCenter)
        mode_column.addWidget(self.mode_chip)
        header.addLayout(mode_column)
        outer.addLayout(header)
        toggles = QHBoxLayout()
        toggles.setSpacing(12)
        self.mode_button = QPushButton("حالت شبیه‌سازی")
        self.autoclick_button = QPushButton("کلیک خودکار: خاموش")
        for button in (self.mode_button, self.autoclick_button):
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setMinimumHeight(48)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            toggles.addWidget(button)
        self.mode_button.clicked.connect(self.toggle_mode)
        self.autoclick_button.clicked.connect(self.toggle_autoclick)
        outer.addLayout(toggles)
        self.tabs = QTabWidget()
        self.tabs.setElideMode(Qt.TextElideMode.ElideRight)
        self.tabs.setUsesScrollButtons(True)
        self.tabs.tabBar().setExpanding(False)
        outer.addWidget(self.tabs)
        self._build_dashboard()
        self._build_calibration()
        self._build_history()
        self._build_smart_review()
        self._build_turnbal()
        grip_row = QHBoxLayout()
        grip = QSizeGrip(root)
        grip.setFixedSize(18, 18)
        grip.setToolTip("برای تغییر اندازهٔ پنجره بکشید")
        grip_row.addWidget(grip)
        grip_row.addStretch()
        outer.addLayout(grip_row)
        self._refresh_toggles()
        self._refresh()

    def _metric(self, layout, row, column, label):
        box = QFrame()
        box.setObjectName("metric")
        box.setMinimumWidth(132)
        box.setToolTip(label)
        inside = QVBoxLayout(box)
        inside.setContentsMargins(14, 11, 14, 11)
        inside.setSpacing(4)
        caption = QLabel(label)
        caption.setWordWrap(True)
        caption.setStyleSheet("color:#9fc4b1;border:0;background:transparent;font-size:12px")
        value = QLabel("—")
        value.setStyleSheet("font-size:22px;font-weight:700;color:#eafff4;"
                            "border:0;background:transparent")
        value.setAlignment(Qt.AlignmentFlag.AlignRight)
        value.setWordWrap(True)
        if label == "ضریب شناسایی‌شده":
            value.setStyleSheet("font-size:32px;font-weight:800;color:#f5c86b;"
                                "border:0;background:transparent")
        elif label == "وضعیت پایش":
            value.setStyleSheet("font-size:20px;font-weight:800;color:#6fe3ac;"
                                "border:0;background:transparent")
        inside.addWidget(caption)
        inside.addWidget(value)
        layout.addWidget(box, row, column)
        self._metric_boxes.append(box)
        return value

    @staticmethod
    def _scrollable(page: QWidget) -> QScrollArea:
        """Keep a page readable when the window is shrunk instead of squashing the text."""
        page.setMinimumWidth(430)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(page)
        return scroll

    def _build_dashboard(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(16)
        metrics = QGridLayout()
        labels = ["وضعیت پایش", "ضریب شناسایی‌شده", "دسته", "اطمینان OCR", "دور تأییدشده",
                  "وضعیت استراتژی", "شماره چرخه", "شماره تلاش", "تلاش‌های باقی‌مانده",
                  "مبلغ ثابت شبیه‌سازی", "چرخه‌های کامل", "چرخه‌های موفق", "چرخه‌های منقضی",
                  "برد شبیه‌سازی", "باخت شبیه‌سازی"]
        self._metric_grid = metrics
        self.values = {name: self._metric(metrics, i//4, i%4, name)
                       for i, name in enumerate(labels)}
        self._metric_columns = 4
        layout.addLayout(metrics)
        controls = QHBoxLayout()
        self.start_button = QPushButton("شروع پایش")
        self.start_button.setObjectName("start")
        self.stop_button = QPushButton("توقف")
        self.stop_button.setObjectName("stop")
        self.pause_button = QPushButton("مکث")
        self.pause_button.setObjectName("pause")
        self.resume_button = QPushButton("ادامه")
        self.resume_button.setObjectName("resume")
        self.reset_button = QPushButton("بازنشانی نشست")
        self.reset_button.setObjectName("reset")
        for button in (self.start_button,self.stop_button,self.pause_button,self.resume_button,self.reset_button):
            controls.addWidget(button)
        self.start_button.clicked.connect(self.start_monitoring)
        self.stop_button.clicked.connect(self.stop_monitoring)
        self.pause_button.clicked.connect(lambda: self.worker.pause() if self.worker else None)
        self.resume_button.clicked.connect(lambda: self.worker.resume() if self.worker else None)
        self.reset_button.clicked.connect(self.reset_session)
        layout.addLayout(controls)
        self.preview = QLabel("پیش‌نمایش ناحیهٔ نتیجه")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(150)
        self.preview.setStyleSheet("background:rgba(8,20,16,200);border:1px solid rgba(110,210,165,70);"
                                   "border-radius:16px;color:#7fa694")
        layout.addWidget(self.preview)
        self.note = QLabel("فقط پیام پایان دور ثبت می‌شود؛ هیچ کلیکی روی بازی انجام نمی‌شود.")
        self.note.setStyleSheet("color:#a9c7b8;background:transparent")
        layout.addWidget(self.note)
        self.tabs.addTab(self._scrollable(page), "داشبورد")

    def _build_calibration(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        form = QFormLayout()
        self.monitor = QComboBox()
        with mss.MSS() as backend:
            for i, screen in enumerate(backend.monitors[1:], 1):
                self.monitor.addItem(f"{i} — {screen['width']}×{screen['height']}", i)
        form.addRow("نمایشگر", self.monitor)
        self.region_fields = {}
        for name, title in (("x","X"),("y","Y"),("width","عرض"),("height","ارتفاع")):
            field = QSpinBox()
            field.setRange(-10000 if name in ("x","y") else 20, 20000)
            # PySide6 leaves the line edit empty when setRange keeps the value at
            # its default; force one refresh so X/Y never look blank.
            field.setValue(field.value())
            self.region_fields[name] = field
            form.addRow(title, field)
        self.amount = QLineEdit(self.settings.fixed_bet_amount)
        form.addRow("مبلغ ثابت شبیه‌سازی", self.amount)
        self.attempts = QSpinBox()
        self.attempts.setRange(1, 100)
        self.attempts.setValue(self.settings.max_attempts)
        form.addRow("حداکثر تلاش", self.attempts)
        self.threshold = QLineEdit(self.settings.yellow_threshold)
        form.addRow("آستانهٔ زرد", self.threshold)
        self.interval = QSpinBox()
        self.interval.setRange(20, 2000)
        self.interval.setValue(self.settings.interval_ms)
        form.addRow("فاصلهٔ ثبت تصویر (میلی‌ثانیه)", self.interval)
        self.confidence = QDoubleSpinBox()
        self.confidence.setRange(.01, 1)
        self.confidence.setSingleStep(.05)
        self.confidence.setValue(self.settings.ocr_confidence)
        form.addRow("حداقل اطمینان OCR", self.confidence)
        self.tesseract = QLineEdit(self.settings.tesseract_path)
        self.tesseract.setPlaceholderText("خالی = مدل محلی RapidOCR")
        form.addRow("مسیر اختیاری Tesseract", self.tesseract)
        self.autoclick = QCheckBox()
        self.autoclick.setChecked(self.settings.autoclick_enabled)
        self._refresh_autoclick_label()
        self.autoclick.toggled.connect(self._refresh_autoclick_label)
        form.addRow("کلیک خودکار", self.autoclick)
        self.autoclick_real = QCheckBox("کلیک واقعی (خاموش = فقط آزمایش)")
        self.autoclick_real.setChecked(not self.settings.autoclick_dry_run)
        form.addRow("حالت کلیک", self.autoclick_real)
        self.click_threshold = QLineEdit(self.settings.autoclick_threshold)
        form.addRow("آستانهٔ کلیک (ضریب ≥)", self.click_threshold)
        self.click_delay = QSpinBox()
        self.click_delay.setRange(0, MAX_CLICK_DELAY_MS)
        self.click_delay.setToolTip("هیچ کلیکی دیرتر از یک ثانیه پس از بسته‌شدن ضریب انجام نمی‌شود")
        self.click_delay.setSingleStep(100)
        self.click_delay.setValue(self.settings.autoclick_delay_ms)
        form.addRow("تأخیر کلیک (میلی‌ثانیه)", self.click_delay)
        self.cashout = QCheckBox("کش‌اوت هوشمند: بعد از هر شرط، مدل تعیین می‌کند کجا خارج شوی")
        self.cashout.setChecked(self.settings.cashout_enabled)
        self.cashout.setToolTip(
            "مدل بین 1.10 و 1.80 هدف خروج را انتخاب می‌کند؛ اگر دست قبلی زیر 1.10 بسته شده "
            "باشد تا 1.89 صبر می‌شود. دیرترین خروج همیشه 1.90 است و کلیک روی همان دکمهٔ شرط زده می‌شود.")
        form.addRow("کش‌اوت هوشمند", self.cashout)
        self.smart_click = QCheckBox("کلیک هوشمند: فقط وقتی پیش‌بینی رنگ به اطمینان برسد")
        self.smart_click.setChecked(self.settings.smart_click_enabled)
        form.addRow("کلیک هوشمند", self.smart_click)
        self.prediction_confidence = QSpinBox()
        self.prediction_confidence.setRange(50, 100)
        self.prediction_confidence.setSuffix(" %")
        self.prediction_confidence.setValue(round(self.settings.prediction_min_confidence * 100))
        form.addRow("حداقل اطمینان پیش‌بینی", self.prediction_confidence)
        self.review_minutes = QSpinBox()
        self.review_minutes.setRange(1, 120)
        self.review_minutes.setSuffix(" دقیقه")
        self.review_minutes.setValue(max(1, self.settings.review_window_s // 60))
        form.addRow("بازبینی خودکار هر", self.review_minutes)
        self.click_wait = QSpinBox()
        self.click_wait.setRange(0, 20)
        self.click_wait.setSuffix(" دست")
        self.click_wait.setValue(self.settings.autoclick_skip_rounds)
        form.addRow("انتظار بعد از زرد", self.click_wait)
        self.click_burst = QSpinBox()
        self.click_burst.setRange(1, 20)
        self.click_burst.setSuffix(" کلیک")
        self.click_burst.setValue(self.settings.autoclick_max_clicks)
        form.addRow("کلیک‌های بعد از انتظار", self.click_burst)
        self.pair_flip = QCheckBox("جفت سبز یا جفت قرمز ← کلیک روی رنگ مخالف")
        self.pair_flip.setChecked(self.settings.pair_flip_enabled)
        form.addRow("قانون جفت رنگ", self.pair_flip)
        self.row_signal = QCheckBox("سیگنال سطری: همان سطر ۴ ستون قبل زرد بوده ← کلیک زرد همین دست")
        self.row_signal.setChecked(self.settings.row_signal_enabled)
        form.addRow("قانون سطری", self.row_signal)
        self.row_gap = QSpinBox()
        self.row_gap.setRange(1, 10)
        self.row_gap.setSuffix(" ستون")
        self.row_gap.setValue(self.settings.row_signal_gap)
        form.addRow("فاصلهٔ ستون‌ها", self.row_gap)
        self.row_limit = QSpinBox()
        self.row_limit.setRange(1, 6)
        self.row_limit.setSuffix(" سطر")
        self.row_limit.setValue(self.settings.row_signal_rows)
        form.addRow("حداکثر سطر فعال", self.row_limit)
        self.streak_signal = QCheckBox("روند هم‌رنگ: یک رنگ بیشتر از ۶ بار پشت‌سرهم ← همان رنگ")
        self.streak_signal.setChecked(self.settings.streak_signal_enabled)
        form.addRow("قانون روند", self.streak_signal)
        self.streak_length = QSpinBox()
        self.streak_length.setRange(1, 20)
        self.streak_length.setSuffix(" بار")
        self.streak_length.setValue(self.settings.streak_signal_length)
        form.addRow("حداقل تکرار پشت‌سرهم", self.streak_length)
        self.streak_clicks = QSpinBox()
        self.streak_clicks.setRange(1, 30)
        self.streak_clicks.setSuffix(" دست")
        self.streak_clicks.setValue(self.settings.streak_signal_clicks)
        form.addRow("کلیک روند هم‌رنگ", self.streak_clicks)
        self.value_bets_edit = QLineEdit(", ".join(f"{key}:{color}" for key, color in
                                                    self.settings.value_bets.items()))
        self.value_bets_edit.setPlaceholderText("1.00:GREEN، 2.08:YELLOW")
        self.value_bets_edit.setToolTip("هر مورد به شکل ضریب:رنگ (RED/GREEN/YELLOW)، جداشده با کاما")
        form.addRow("ضریب‌های ویژه", self.value_bets_edit)
        layout.addLayout(form)
        regions = QHBoxLayout()
        for title, target in (("ناحیه ۱ — تاریخچه", "history"),
                              ("ناحیه ۲ — ضریب فعلی", "result"),
                              ("ناحیه ۳ — شبکهٔ رنگ", "chart"),
                              ("ناحیه ۴ — دکمه‌های شرط", "bet")):
            button = QPushButton(title)
            button.clicked.connect(lambda _=False, t=target: self.select_region(t))
            regions.addWidget(button)
        layout.addLayout(regions)
        controls = QHBoxLayout()
        for title, handler in (("ذخیره", self.save_ui_settings), ("آزمون OCR", self.calibrate),
                               ("کلیک تست", self.test_click),
                               ("حذف همهٔ تنظیمات", self.reset_all_settings)):
            button = QPushButton(title)
            button.clicked.connect(handler)
            controls.addWidget(button)
        layout.addLayout(controls)
        self.region_status = QLabel("ناحیهٔ نتیجه: انتخاب نشده")
        layout.addWidget(self.region_status)
        self.calibration_status = QLabel("آزمون OCR هنوز انجام نشده است.")
        layout.addWidget(self.calibration_status)
        self.click_status = QLabel("وضعیت کلیک: خاموش")
        layout.addWidget(self.click_status)
        self.report_status = QLabel("گزارش پایان جلسه: پس از پایان زمان جلسه یا توقف پایش در "
                                    "logs/session-report.txt نوشته می‌شود")
        self.report_status.setWordWrap(True)
        layout.addWidget(self.report_status)
        layout.addStretch()
        self.tabs.addTab(self._scrollable(page), "کالیبراسیون")
        self.tabs.setTabToolTip(1, "ناحیه‌ها، کلیک و همهٔ تنظیمات ربات")
        if self.settings.result_region:
            self._fill_region(self.settings.result_region)
        self._update_region_status()

    def _build_smart_review(self):
        """ناحیهٔ جدید «بررسی هوشمند ضریب‌ها»: اتصال به FreeLLMAPI و ارزیابی ناحیه ۱ تا ۴."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(10)
        self.smart_enabled = QCheckBox()
        self.smart_enabled.setChecked(self.settings.smart_review_enabled)
        self._refresh_smart_label()
        self.smart_enabled.toggled.connect(self._toggle_smart_review)
        layout.addWidget(self.smart_enabled)
        self.smart_play = QCheckBox()
        self.smart_play.setChecked(self.settings.smart_play_enabled)
        self.smart_play.setToolTip("پس از پایان دورهٔ یادگیری، مدل خودش تصمیم می‌گیرد و شرط می‌بندد")
        self._refresh_smart_play_label()
        self.smart_play.toggled.connect(self._toggle_smart_play)
        layout.addWidget(self.smart_play)
        self.smart_play_status = QLabel("دورهٔ یادگیری با شروع پایش شمرده می‌شود.")
        self.smart_play_status.setWordWrap(True)
        self.smart_play_status.setStyleSheet("color:#c7d0dc;")
        layout.addWidget(self.smart_play_status)
        form = QFormLayout()
        self.smart_url = QLineEdit(self.settings.smart_review_url or DEFAULT_BASE_URL)
        self.smart_url.setPlaceholderText(DEFAULT_BASE_URL)
        self.smart_url.setToolTip("آدرس هر پرووایدر سازگار با OpenAI — FreeLLMAPI، 9router، "
                                  "OmniRouter یا هر API دیگری؛ مثلاً http://127.0.0.1:31415/v1")
        form.addRow("Custom Provider", self.smart_url)
        self.smart_key = QLineEdit(self.settings.smart_review_key)
        self.smart_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.smart_key.setPlaceholderText("کلید یکپارچهٔ داشبورد FreeLLMAPI")
        form.addRow("کلید", self.smart_key)
        self.smart_model = QLineEdit(self.settings.smart_review_model or DEFAULT_MODEL)
        self.smart_model.setPlaceholderText(DEFAULT_MODEL)
        form.addRow("مدل", self.smart_model)
        self.smart_timeout = QSpinBox()
        self.smart_timeout.setRange(5, 300)
        self.smart_timeout.setSuffix(" ثانیه")
        self.smart_timeout.setValue(self.settings.smart_review_timeout_s)
        form.addRow("مهلت پاسخ مدل", self.smart_timeout)
        self.smart_every = QSpinBox()
        self.smart_every.setRange(1, 100)
        self.smart_every.setSuffix(" دست")
        self.smart_every.setValue(self.settings.smart_review_every_rounds)
        form.addRow("ارزیابی هر چند دست", self.smart_every)
        self.smart_rules = QLineEdit(self.settings.smart_review_rules)
        self.smart_rules.setPlaceholderText("قاعدهٔ اضافی اختیاری برای مدل")
        form.addRow("قاعدهٔ اضافی", self.smart_rules)
        self.smart_learning = QSpinBox()
        self.smart_learning.setRange(0, 600)
        self.smart_learning.setSuffix(" دقیقه")
        self.smart_learning.setValue(self.settings.smart_play_learning_minutes)
        self.smart_learning.setToolTip("در این مدت فقط یادگیری و ارزیابی انجام می‌شود؛ هیچ شرطی بسته نمی‌شود")
        form.addRow("دورهٔ یادگیری پیش از بازی", self.smart_learning)
        self.smart_min_conf = QSpinBox()
        self.smart_min_conf.setRange(20, 100)
        self.smart_min_conf.setSuffix(" %")
        self.smart_min_conf.setValue(round(self.settings.smart_play_min_confidence * 100))
        self.smart_min_conf.setToolTip("فقط وقتی احتمال اعلامی مدل از این مقدار بیشتر باشد شرط بسته می‌شود")
        form.addRow("حداقل احتمال برای شرط", self.smart_min_conf)
        self.smart_two = QCheckBox("اجازهٔ کلیک هم‌زمان روی دو رنگ (پوشش ضرر)")
        self.smart_two.setChecked(self.settings.smart_play_two_colors)
        form.addRow("", self.smart_two)
        layout.addLayout(form)
        buttons = QHBoxLayout()
        connect_button = QPushButton("تست اتصال")
        connect_button.clicked.connect(self.test_smart_connection)
        save_button = QPushButton("ذخیره")
        save_button.clicked.connect(self.save_ui_settings)
        buttons.addWidget(connect_button)
        buttons.addWidget(save_button)
        buttons.addStretch()
        layout.addLayout(buttons)
        self.smart_probe_status = QLabel("اتصال هنوز آزمایش نشده است.")
        self.smart_probe_status.setWordWrap(True)
        layout.addWidget(self.smart_probe_status)
        self.smart_status = QLabel("پیشنهاد مدل: —")
        self.smart_status.setWordWrap(True)
        self.smart_status.setStyleSheet("font-size:18px;font-weight:700;color:#c7d0dc;")
        layout.addWidget(self.smart_status)
        self.smart_raw = QPlainTextEdit()
        self.smart_raw.setReadOnly(True)
        self.smart_raw.setMinimumHeight(140)
        self.smart_raw.setPlaceholderText("آخرین پرسمان و پاسخ مدل اینجا نمایش داده می‌شود")
        layout.addWidget(self.smart_raw)
        note = QLabel("پیشنهاد مدل فقط جهت اطلاع است؛ کلیک‌ها همچنان با قواعد خود ربات انجام می‌شوند. "
                      "رویدادها در logs/smart-review.jsonl ذخیره می‌شوند.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#aeb8c5")
        layout.addWidget(note)
        layout.addStretch()
        self.tabs.addTab(self._scrollable(page), "بررسی هوشمند")
        self.tabs.setTabToolTip(3, "بررسی هوشمند ضریب‌ها — اتصال به FreeLLMAPI")

    def _build_turnbal(self):
        """تب ترنبال: تصمیم هوش مصنوعی و کالیبراسیون شرط سه‌رنگ."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(12)
        self.turnbal_status = QLabel("این تب فقط در حالت ترنبال اثر دارد.")
        self.turnbal_status.setWordWrap(True)
        self.turnbal_status.setStyleSheet("color:#c7d0dc;")
        layout.addWidget(self.turnbal_status)

        ai_group = QGroupBox("بررسی هوشمند ترنبال")
        ai_layout = QVBoxLayout(ai_group)
        ai_layout.setSpacing(8)
        hint = QLabel("مدل زبانی ضریب‌ها، رنگ‌ها و شبکهٔ نمودار را تحلیل می‌کند و برای دست بعد "
                      "قرمز، سبز یا زرد پیشنهاد می‌دهد؛ پیشنهاد پیش از کلیک با کالیبراسیون "
                      "سه‌رنگ پایین تطبیق داده می‌شود.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#9fc4b1;")
        ai_layout.addWidget(hint)
        self.turnbal_ai = QCheckBox()
        self.turnbal_ai.setChecked(self.settings.smart_review_enabled)
        self.turnbal_ai.toggled.connect(self._toggle_smart_review)
        ai_layout.addWidget(self.turnbal_ai)
        self.turnbal_play = QCheckBox()
        self.turnbal_play.setChecked(self.settings.smart_play_enabled)
        self.turnbal_play.setToolTip("در حالت ترنبال، پس از دورهٔ یادگیری فقط تصمیم مدل شرط می‌بندد")
        self.turnbal_play.toggled.connect(self._toggle_smart_play)
        ai_layout.addWidget(self.turnbal_play)
        self.turnbal_ai_status = QLabel("دورهٔ یادگیری از لحظهٔ شروع پایش شمرده می‌شود.")
        self.turnbal_ai_status.setWordWrap(True)
        self.turnbal_ai_status.setStyleSheet("color:#8ff0c0;")
        ai_layout.addWidget(self.turnbal_ai_status)
        provider = QLabel("آدرس، کلید و مدل از تب «بررسی هوشمند» (برچسب Custom Provider) خوانده "
                          "می‌شود؛ هر API سازگار با OpenAI کار می‌کند: FreeLLMAPI، 9router، "
                          "OmniRouter یا هر پرووایدر دیگر.")
        provider.setWordWrap(True)
        provider.setStyleSheet("color:#7fa694;font-size:12px")
        ai_layout.addWidget(provider)
        layout.addWidget(ai_group)

        calib_group = QGroupBox("کالیبراسیون شرط سه‌رنگ")
        grid = QGridLayout(calib_group)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        for column, text in enumerate(("رنگ", "فعال", "بازهٔ ضریب", "حداقل احتمال مدل",
                                        "حداکثر تلاش")):
            label = QLabel(text)
            label.setStyleSheet("color:#8ff0c0;font-weight:700;")
            grid.addWidget(label, 0, column)
        rows = (("red", "قرمز", "#e05a5a", "کمتر از آستانهٔ قرمز"),
                ("green", "سبز", "#2ecc71", "از آستانهٔ قرمز تا زرد"),
                ("yellow", "زرد (مون)", "#f0ca52", "آستانهٔ زرد و بالاتر"))
        for row, (key, name, color, caption) in enumerate(rows, 1):
            chip = QLabel(name)
            chip.setStyleSheet(f"color:{color};font-weight:800;")
            grid.addWidget(chip, row, 0)
            check = QCheckBox()
            check.setChecked(getattr(self.settings, f"turnbal_{key}_enabled"))
            setattr(self, f"tb_{key}_on", check)
            grid.addWidget(check, row, 1)
            span = QLabel(caption)
            span.setStyleSheet("color:#9fc4b1;font-size:12px")
            grid.addWidget(span, row, 2)
            conf = QSpinBox()
            conf.setRange(20, 100)
            conf.setSuffix(" %")
            conf.setValue(round(getattr(self.settings, f"turnbal_{key}_min_conf") * 100))
            setattr(self, f"tb_{key}_conf", conf)
            grid.addWidget(conf, row, 3)
            tries = QSpinBox()
            tries.setRange(1, 20)
            tries.setValue(getattr(self.settings, f"turnbal_{key}_attempts"))
            setattr(self, f"tb_{key}_tries", tries)
            grid.addWidget(tries, row, 4)
        layout.addWidget(calib_group)

        risk_group = QGroupBox("کش‌اوت و ریسک")
        risk_form = QFormLayout(risk_group)
        self.tb_cashout_min = QDoubleSpinBox()
        self.tb_cashout_min.setRange(1.40, 1.90)
        self.tb_cashout_min.setSingleStep(0.05)
        self.tb_cashout_min.setDecimals(2)
        self.tb_cashout_min.setValue(self.settings.turnbal_cashout_min)
        risk_form.addRow("حداقل کش‌اوت", self.tb_cashout_min)
        self.tb_cashout_max = QDoubleSpinBox()
        self.tb_cashout_max.setRange(1.40, 1.90)
        self.tb_cashout_max.setSingleStep(0.05)
        self.tb_cashout_max.setDecimals(2)
        self.tb_cashout_max.setValue(self.settings.turnbal_cashout_max)
        risk_form.addRow("حداکثر کش‌اوت", self.tb_cashout_max)
        self.tb_emergency = QDoubleSpinBox()
        self.tb_emergency.setRange(1.40, 1.90)
        self.tb_emergency.setSingleStep(0.05)
        self.tb_emergency.setDecimals(2)
        self.tb_emergency.setValue(self.settings.turnbal_emergency_exit)
        risk_form.addRow("خروج اضطراری", self.tb_emergency)
        self.tb_cycle_win = QSpinBox()
        self.tb_cycle_win.setRange(0, 100)
        self.tb_cycle_win.setSuffix(" %")
        self.tb_cycle_win.setValue(round(self.settings.turnbal_min_cycle_win * 100))
        risk_form.addRow("حداقل برد چرخه برای بازی", self.tb_cycle_win)
        self.tb_streak = QSpinBox()
        self.tb_streak.setRange(1, 50)
        self.tb_streak.setSuffix(" دست")
        self.tb_streak.setValue(self.settings.turnbal_max_streak)
        risk_form.addRow("سقف دست پشت‌سرهم", self.tb_streak)
        self.tb_gap = QSpinBox()
        self.tb_gap.setRange(0, 50)
        self.tb_gap.setSuffix(" دست")
        self.tb_gap.setValue(self.settings.turnbal_min_gap)
        risk_form.addRow("حداقل فاصلهٔ دو شرط", self.tb_gap)
        layout.addWidget(risk_group)

        buttons = QHBoxLayout()
        save_button = QPushButton("ذخیره")
        save_button.clicked.connect(self.save_ui_settings)
        defaults_button = QPushButton("بازگشت به پیش‌فرض‌ها")
        defaults_button.setObjectName("reset")
        defaults_button.clicked.connect(self.reset_turnbal_defaults)
        buttons.addWidget(save_button)
        buttons.addWidget(defaults_button)
        buttons.addStretch()
        layout.addLayout(buttons)
        note = QLabel("در حالت کلاسیک همین تنظیمات ذخیره می‌مانند ولی اثر ندارند؛ کلیک‌ها همان "
                      "قواعد فعلی ربات را دنبال می‌کنند. تصمیم‌های ترنبال در "
                      "logs/turnbal-review.jsonl ثبت می‌شوند.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#aeb8c5")
        layout.addWidget(note)
        layout.addStretch()
        self.tabs.addTab(self._scrollable(page), "ترنبال")
        self.tabs.setTabToolTip(4, "ترنبال: تصمیم هوش مصنوعی + کالیبراسیون شرط سه‌رنگ")

    def reset_turnbal_defaults(self):
        """کالیبراسیون سه‌رنگ را به مقدارهای پیش‌فرض برگردان و ذخیره کن."""
        defaults = DesktopSettings()
        for key in ("red", "green", "yellow"):
            getattr(self, f"tb_{key}_on").setChecked(getattr(defaults, f"turnbal_{key}_enabled"))
            getattr(self, f"tb_{key}_conf").setValue(round(getattr(defaults, f"turnbal_{key}_min_conf") * 100))
            getattr(self, f"tb_{key}_tries").setValue(getattr(defaults, f"turnbal_{key}_attempts"))
        self.tb_cashout_min.setValue(defaults.turnbal_cashout_min)
        self.tb_cashout_max.setValue(defaults.turnbal_cashout_max)
        self.tb_emergency.setValue(defaults.turnbal_emergency_exit)
        self.tb_cycle_win.setValue(round(defaults.turnbal_min_cycle_win * 100))
        self.tb_streak.setValue(defaults.turnbal_max_streak)
        self.tb_gap.setValue(defaults.turnbal_min_gap)
        self.save_ui_settings()

    def _build_history(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["شناسهٔ دور", "ضریب", "دسته", "اطمینان", "زمان", "استراتژی", "تلاش", "نتیجه"])
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)
        actions = QHBoxLayout()
        clear = QPushButton("پاک‌کردن تاریخچه")
        export = QPushButton("خروجی CSV")
        clear.clicked.connect(self.clear_history)
        export.clicked.connect(self.export_history)
        actions.addWidget(clear)
        actions.addWidget(export)
        actions.addStretch()
        layout.addLayout(actions)
        self.tabs.addTab(page, "تاریخچه")
        self.tabs.setTabToolTip(2, "تاریخچهٔ رویدادها")

    def _fill_region(self, region: Region):
        self.monitor.setCurrentIndex(region.monitor - 1)
        for name in self.region_fields:
            self.region_fields[name].setValue(getattr(region, name))

    def _current_region(self) -> Region:
        return Region(self.monitor.currentData(), **{name: field.value() for name, field in self.region_fields.items()})

    @staticmethod
    def _parse_value_bets(text: str) -> dict[str, str]:
        bets: dict[str, str] = {}
        for chunk in text.replace("،", ",").split(","):
            if not chunk.strip():
                continue
            key, separator, color = chunk.partition(":")
            if not separator:
                raise ValueError("هر مورد ضریب ویژه باید به شکل 1.00:GREEN باشد")
            bets[key.strip()] = color.strip().upper()
        return bets

    def _settings_from_form(self) -> DesktopSettings:
        return replace(self.settings, result_region=self._current_region(), fixed_bet_amount=self.amount.text(),
                       max_attempts=self.attempts.value(), yellow_threshold=self.threshold.text(),
                       interval_ms=self.interval.value(), ocr_confidence=self.confidence.value(),
                       tesseract_path=self.tesseract.text().strip(),
                       autoclick_enabled=self.autoclick.isChecked(),
                       autoclick_dry_run=not self.autoclick_real.isChecked(),
                       autoclick_threshold=self.click_threshold.text().strip(),
                       autoclick_delay_ms=self.click_delay.value(),
                       autoclick_skip_rounds=self.click_wait.value(),
                       autoclick_max_clicks=self.click_burst.value(),
                       cashout_enabled=self.cashout.isChecked(),
                       pair_flip_enabled=self.pair_flip.isChecked(),
                       row_signal_enabled=self.row_signal.isChecked(),
                       row_signal_gap=self.row_gap.value(),
                       row_signal_rows=self.row_limit.value(),
                       streak_signal_enabled=self.streak_signal.isChecked(),
                       streak_signal_length=self.streak_length.value(),
                       streak_signal_clicks=self.streak_clicks.value(),
                       value_bets=self._parse_value_bets(self.value_bets_edit.text()),
                       smart_click_enabled=self.smart_click.isChecked(),
                       prediction_min_confidence=self.prediction_confidence.value() / 100,
                       smart_review_enabled=self.smart_enabled.isChecked(),
                       smart_review_url=self.smart_url.text().strip() or DEFAULT_BASE_URL,
                       smart_review_key=self.smart_key.text().strip(),
                       smart_review_model=self.smart_model.text().strip() or DEFAULT_MODEL,
                       smart_review_timeout_s=self.smart_timeout.value(),
                       smart_review_every_rounds=self.smart_every.value(),
                       smart_review_rules=self.smart_rules.text().strip(),
                       smart_play_enabled=self.smart_play.isChecked(),
                       smart_play_learning_minutes=self.smart_learning.value(),
                       smart_play_min_confidence=self.smart_min_conf.value() / 100,
                       smart_play_two_colors=self.smart_two.isChecked(),
                       bot_mode=self.settings.bot_mode,
                       turnbal_red_enabled=self.tb_red_on.isChecked(),
                       turnbal_green_enabled=self.tb_green_on.isChecked(),
                       turnbal_yellow_enabled=self.tb_yellow_on.isChecked(),
                       turnbal_red_min_conf=self.tb_red_conf.value() / 100,
                       turnbal_green_min_conf=self.tb_green_conf.value() / 100,
                       turnbal_yellow_min_conf=self.tb_yellow_conf.value() / 100,
                       turnbal_red_attempts=self.tb_red_tries.value(),
                       turnbal_green_attempts=self.tb_green_tries.value(),
                       turnbal_yellow_attempts=self.tb_yellow_tries.value(),
                       turnbal_cashout_min=round(self.tb_cashout_min.value(), 2),
                       turnbal_cashout_max=round(self.tb_cashout_max.value(), 2),
                       turnbal_emergency_exit=round(self.tb_emergency.value(), 2),
                       turnbal_min_cycle_win=self.tb_cycle_win.value() / 100,
                       turnbal_max_streak=self.tb_streak.value(),
                       turnbal_min_gap=self.tb_gap.value(),
                       review_window_s=self.review_minutes.value() * 60)

    def _refresh_autoclick_label(self) -> None:
        """Make the on/off state of autoclick obvious: text and colour both change."""
        on = self.autoclick.isChecked()
        self.autoclick.setText("کلیک خودکار: روشن" if on else "کلیک خودکار: خاموش")
        color = "#6fe3ac" if on else "#95a5a6"
        self.autoclick.setStyleSheet(f"QCheckBox {{ color: {color}; font-weight: bold; }}")

    def _refresh_smart_label(self) -> None:
        """وضعیت روشن/خاموش «بررسی هوشمند» را با متن و رنگ نشان بده."""
        on = self.smart_enabled.isChecked()
        self.smart_enabled.setText("بررسی هوشمند ضریب‌ها: روشن" if on
                                  else "بررسی هوشمند ضریب‌ها: خاموش")
        color = "#6fe3ac" if on else "#95a5a6"
        self.smart_enabled.setStyleSheet(f"QCheckBox {{ color: {color}; font-weight: bold; }}")
        self._refresh_turnbal_labels()

    def _refresh_smart_play_label(self) -> None:
        """وضعیت روشن/خاموش «ورود هوشمند به بازی» را واضح نشان بده."""
        on = self.smart_play.isChecked()
        self.smart_play.setText("ورود هوشمند به بازی: روشن" if on
                               else "ورود هوشمند به بازی: خاموش")
        color = "#6fe3ac" if on else "#95a5a6"
        self.smart_play.setStyleSheet(f"QCheckBox {{ color: {color}; font-weight: bold; }}")
        self._refresh_turnbal_labels()

    def _toggle_smart_play(self, on: bool) -> None:
        self._refresh_smart_play_label()
        self.apply_settings(smart_play_enabled=bool(on))

    def _refresh_smart_play_status(self, report: dict) -> None:
        """مرحلهٔ یادگیری/بازی و دقت اندازه‌گیری‌شدهٔ مدل را بنویس."""
        names = {"RED": "قرمز", "GREEN": "سبز", "YELLOW": "زرد"}
        parts = []
        phase = report.get("phase")
        if phase == "learning":
            parts.append(f"دورهٔ یادگیری: {report.get('learning_left', 0):.0f} دقیقه مانده — "
                         "فقط ارزیابی، بدون شرط")
        elif phase == "trading":
            parts.append("دورهٔ یادگیری تمام شد؛ ورود هوشمند فعال است")
        tries = report.get("tries") or 0
        accuracy = report.get("accuracy")
        if tries and accuracy is not None:
            parts.append(f"دقت مدل: {accuracy:.0%} در {tries} پیش‌بینی")
        bets = report.get("bets") or ()
        if report.get("ok") and bets:
            parts.append("انتخاب برای دست بعد: " + " + ".join(names.get(bet, bet) for bet in bets))
        self.smart_play_status.setText(" | ".join(parts) or "—")

    def _toggle_smart_review(self, on: bool) -> None:
        self._refresh_smart_label()
        if self.worker and self.worker.isRunning():
            self.apply_settings(smart_review_enabled=bool(on))
        else:
            candidate = self._settings_from_form()
            try:
                save_settings(candidate)
            except Exception as exc:
                self.smart_enabled.setChecked(not on)
                self.smart_probe_status.setText("تنظیمات ذخیره نشد: " + str(exc))
                return
            self.settings = candidate
        if on:
            self.test_smart_connection()

    def test_smart_connection(self):
        """اتصال، پورت و کلید را قبل از شروع پایش آزمایش کن."""
        try:
            chat_completions_url(self.smart_url.text())
        except Exception as exc:
            QMessageBox.warning(self, "آدرس نامعتبر", str(exc))
            return
        self.smart_probe_status.setText("در حال اتصال به پرووایدر…")
        self.smart_probe = SmartProbeWorker(self.smart_url.text(), self.smart_key.text(),
                                            min(15, self.smart_timeout.value()))
        self.smart_probe.done.connect(self._smart_probe_done)
        self.smart_probe.failed.connect(
            lambda message: self.smart_probe_status.setText("خطای اتصال: " + message))
        self.smart_probe.start()

    def _smart_probe_done(self, models):
        names = "، ".join(str(name) for name in models[:5])
        if models:
            self.smart_probe_status.setText(
                f"متصل شد — {len(models)} مدل در دسترس: {names}"
                " (نام یکی از همین‌ها را در فیلد «مدل» بگذارید یا auto را نگه دارید)")
        else:
            self.smart_probe_status.setText("متصل شد (فهرست مدل‌ها خالی بود؛ مدل را دستی بگذارید).")

    def _on_smart(self, report):
        self._refresh_smart_play_status(report)
        if not report.get("ok"):
            self.smart_status.setText("خطا: " + str(report.get("error", "")))
            self.smart_status.setStyleSheet("font-size:16px;font-weight:700;color:#e07a86;")
        else:
            verdict = report.get("verdict")
            if verdict is None:
                self.smart_status.setText("پاسخ مدل قابل خواندن نبود — متن خام را ببینید")
                self.smart_status.setStyleSheet("font-size:16px;font-weight:700;color:#e6ca4f;")
            else:
                color = {"RED": "#e05a5a", "GREEN": "#2ecc71",
                         "YELLOW": "#f0ca52"}.get(verdict.bet, "#c7d0dc")
                reason = f" — {verdict.reason}" if verdict.reason else ""
                self.smart_status.setText(f"پیشنهاد: {verdict.label}{reason}")
                self.smart_status.setStyleSheet(
                    f"font-size:18px;font-weight:800;color:{color};")
        seconds = report.get("seconds")
        timing = f"\n\nزمان پاسخ: {seconds:.1f} ثانیه | دور: {report.get('round', '—')}" if seconds else ""
        self.smart_raw.setPlainText(
            f"پرسمان:\n{report.get('prompt', '')}{timing}\n\nپاسخ مدل:\n{report.get('raw', report.get('error', ''))}")

    @staticmethod
    def _third_point(region: Region, sixth: int) -> tuple[int, int]:
        return (region.x + region.width * sixth // 6, region.y + region.height // 2)

    def _region_state(self, region: Region | None) -> str:
        if region is None:
            return "انتخاب نشده"
        return f"{region.x},{region.y} {region.width}×{region.height}"

    def _update_region_status(self):
        parts = [f"ناحیه {str(number).translate(_FA_DIGITS)}: "
                 f"{self._region_state(getattr(self.settings, name))}"
                 for number, name in ((1, "history_region"), (2, "result_region"),
                                      (3, "chart_region"), (4, "bet_region"))]
        self.region_status.setText("  |  ".join(parts))

    def select_region(self, target: str = "result"):
        """Hide this window, let the user drag on the live screen, then come back."""
        error = None
        try:
            self.hide()
            QApplication.processEvents()
            sleep(0.3)  # let this window actually leave the screen before the capture
            overlay = RegionOverlay(self.monitor.currentData())
            if overlay.exec() and overlay.selected:
                if target == "history":
                    self.settings = replace(self.settings, history_region=overlay.selected)
                elif target == "bet":
                    chosen = overlay.selected
                    self.settings = replace(
                        self.settings, bet_region=chosen,
                        bet_point_red=self._third_point(chosen, 1),
                        bet_point_green=self._third_point(chosen, 3),
                        bet_point_yellow=self._third_point(chosen, 5))
                elif target == "click":
                    self.settings = replace(self.settings, click_region=overlay.selected)
                elif target == "chart":
                    self.settings = replace(self.settings, chart_region=overlay.selected)
                elif target.startswith("bet_"):
                    chosen = overlay.selected
                    self.settings = replace(self.settings, **{
                        f"bet_point_{target[4:]}": (chosen.x + chosen.width // 2,
                                                    chosen.y + chosen.height // 2)})
                else:
                    self._fill_region(overlay.selected)
                    self.settings = replace(self.settings, result_region=overlay.selected)
                self.save_ui_settings()  # فایل تنظیمات را ذخیره کن تا مقدارها به‌روز باشند
                self._update_region_status()
        except Exception as exc:
            error = exc
        finally:
            self.show()
            self.raise_()
            self.activateWindow()
            self.tabs.setCurrentIndex(1)
        if error is not None:
            QMessageBox.warning(self, "انتخاب ناحیه", str(error))

    def save_ui_settings(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "پایش", "ابتدا پایش را متوقف کنید.")
            return False
        try:
            previous = self.settings
            candidate = self._settings_from_form()
            save_settings(candidate)
            self.settings = candidate
            if self.settings.strategy() != previous.strategy():
                self.strategy = StrategyEngine(self.settings.strategy())
                self.success = self.expired = self.wins = self.losses = 0
            self._update_region_status()
            self._refresh_toggles()
            self._refresh()
            return True
        except Exception as exc:
            QMessageBox.warning(self, "تنظیمات نامعتبر", str(exc))
            return False

    def calibrate(self):
        if not self.save_ui_settings():
            return
        self.calibration_status.setText("در حال آزمون OCR…")
        self.calibration = CalibrationWorker(self.settings)
        self.calibration.done.connect(self._calibration_done)
        self.calibration.failed.connect(lambda message: self.calibration_status.setText("خطا: " + message))
        self.calibration.start()

    def _calibration_done(self, image, reading):
        self._show_image(image)
        self.calibration_status.setText(f"OCR: {reading.raw_text!r} | ضریب: {reading.value} | "
                                        f"دسته: {reading.category} | اطمینان: {reading.confidence:.2f} | "
                                        f"نامطمئن: {'بله' if reading.uncertain else 'خیر'}")

    def _show_image(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self._preview_image = QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                                     rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self._render_preview()

    def _render_preview(self):
        image = getattr(self, "_preview_image", None)
        preview = getattr(self, "preview", None)
        if image is None or preview is None:
            return
        preview.setPixmap(QPixmap.fromImage(image).scaled(preview.size(),
                                                          Qt.AspectRatioMode.KeepAspectRatio))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render_preview()
        self._relayout_metrics()

    def _relayout_metrics(self):
        """کارت‌های داشبورد را با عرض پنجره چندستونه کن تا متن در هیچ اندازه‌ای نچپد."""
        grid = getattr(self, "_metric_grid", None)
        if grid is None or not self._metric_boxes:
            return
        width = self.tabs.contentsRect().width() or self.width()
        columns = 4 if width >= 780 else 3 if width >= 600 else 2 if width >= 380 else 1
        if columns == self._metric_columns:
            return
        self._metric_columns = columns
        for index, box in enumerate(self._metric_boxes):
            grid.addWidget(box, index // columns, index % columns)

    def start_monitoring(self):
        if self.worker and self.worker.isRunning():
            return
        if self.settings.result_region is None:
            # بررسی اینکه مقدارها واقعاً تنظیم شده‌اند (نه پیش‌فرض)
            x_val = self.region_fields["x"].value()
            y_val = self.region_fields["y"].value()
            width_val = self.region_fields["width"].value()
            height_val = self.region_fields["height"].value()
            # اگر مقادیر معتبر نباشند (پیش‌فرض یا نا موثر)، جلب توجه کن
            if x_val == 0 and y_val == 0 and (width_val == 20 or width_val == 0) and (height_val == 20 or height_val == 0):
                QMessageBox.information(self, "ناحیهٔ نتیجه", "ابتدا ناحیهٔ پیام پایان دور را انتخاب کنید.")
                self.tabs.setCurrentIndex(1)
                return
        if not self.save_ui_settings():
            return
        self.worker = MonitorWorker(self.settings, self.strategy)
        self.worker.observation.connect(self._on_observation)
        self.worker.preview.connect(self._show_image)
        self.worker.problem.connect(self._on_problem)
        self.worker.status.connect(self._on_status)
        self.worker.click_report.connect(self._on_click)
        self.worker.review_report.connect(self._on_review)
        self.worker.smart_report.connect(self._on_smart)
        self.worker.start()
        self._refresh()

    def stop_monitoring(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(5000)
            if self.worker.isRunning():
                self._on_problem("پایش متوقف نشد؛ منتظر پایان OCR بمانید.")
            elif self.strategy.state == StrategyState.ACTIVE_CYCLE:
                self.strategy.reset()
        self._refresh()

    def reset_session(self):
        self.stop_monitoring()
        if self.worker and self.worker.isRunning():
            return
        self.strategy = StrategyEngine(self.settings.strategy())
        self.success = self.expired = self.wins = self.losses = 0
        self.confirmed = 0
        self._refresh()

    def reset_all_settings(self):
        if QMessageBox.question(self, "حذف تنظیمات", "همهٔ تنظیمات محلی حذف شوند؟") != QMessageBox.StandardButton.Yes:
            return
        self.stop_monitoring()
        if SETTINGS_PATH.exists():
            SETTINGS_PATH.unlink()
        self.settings = DesktopSettings()
        self.strategy = StrategyEngine(self.settings.strategy())
        self.confirmed = 0
        self._fill_region(Region(1,0,0,20,20))
        self._sync_ui_to_settings()
        self._update_region_status()
        self._refresh()

    def _sync_ui_to_settings(self):
        """همه فیلدهای UI را با تنظیمات فعلی همگام کن."""
        self.amount.setText(self.settings.fixed_bet_amount)
        self.attempts.setValue(self.settings.max_attempts)
        self.threshold.setText(self.settings.yellow_threshold)
        self.interval.setValue(self.settings.interval_ms)
        self.confidence.setValue(self.settings.ocr_confidence)
        self.tesseract.setText(self.settings.tesseract_path)
        self.autoclick.setChecked(self.settings.autoclick_enabled)
        self._refresh_autoclick_label()
        self.autoclick_real.setChecked(not self.settings.autoclick_dry_run)
        self.click_threshold.setText(self.settings.autoclick_threshold)
        self.click_delay.setValue(self.settings.autoclick_delay_ms)
        self.click_wait.setValue(self.settings.autoclick_skip_rounds)
        self.click_burst.setValue(self.settings.autoclick_max_clicks)
        self.pair_flip.setChecked(self.settings.pair_flip_enabled)
        self.row_signal.setChecked(self.settings.row_signal_enabled)
        self.row_gap.setValue(self.settings.row_signal_gap)
        self.row_limit.setValue(self.settings.row_signal_rows)
        self.streak_signal.setChecked(self.settings.streak_signal_enabled)
        self.streak_length.setValue(self.settings.streak_signal_length)
        self.streak_clicks.setValue(self.settings.streak_signal_clicks)
        self.value_bets_edit.setText(", ".join(f"{key}:{color}" for key, color in
                                               self.settings.value_bets.items()))
        self.smart_click.setChecked(self.settings.smart_click_enabled)
        self.smart_enabled.blockSignals(True)
        self.smart_enabled.setChecked(self.settings.smart_review_enabled)
        self.smart_enabled.blockSignals(False)
        self._refresh_smart_label()
        self.smart_url.setText(self.settings.smart_review_url)
        self.smart_key.setText(self.settings.smart_review_key)
        self.smart_model.setText(self.settings.smart_review_model)
        self.smart_timeout.setValue(self.settings.smart_review_timeout_s)
        self.smart_every.setValue(self.settings.smart_review_every_rounds)
        self.smart_rules.setText(self.settings.smart_review_rules)
        self.smart_play.blockSignals(True)
        self.smart_play.setChecked(self.settings.smart_play_enabled)
        self.smart_play.blockSignals(False)
        self._refresh_smart_play_label()
        self.smart_learning.setValue(self.settings.smart_play_learning_minutes)
        self.smart_min_conf.setValue(round(self.settings.smart_play_min_confidence * 100))
        self.smart_two.setChecked(self.settings.smart_play_two_colors)
        self.tb_red_on.setChecked(self.settings.turnbal_red_enabled)
        self.tb_green_on.setChecked(self.settings.turnbal_green_enabled)
        self.tb_yellow_on.setChecked(self.settings.turnbal_yellow_enabled)
        self.tb_red_conf.setValue(round(self.settings.turnbal_red_min_conf * 100))
        self.tb_green_conf.setValue(round(self.settings.turnbal_green_min_conf * 100))
        self.tb_yellow_conf.setValue(round(self.settings.turnbal_yellow_min_conf * 100))
        self.tb_red_tries.setValue(self.settings.turnbal_red_attempts)
        self.tb_green_tries.setValue(self.settings.turnbal_green_attempts)
        self.tb_yellow_tries.setValue(self.settings.turnbal_yellow_attempts)
        self.tb_cashout_min.setValue(self.settings.turnbal_cashout_min)
        self.tb_cashout_max.setValue(self.settings.turnbal_cashout_max)
        self.tb_emergency.setValue(self.settings.turnbal_emergency_exit)
        self.tb_cycle_win.setValue(round(self.settings.turnbal_min_cycle_win * 100))
        self.tb_streak.setValue(self.settings.turnbal_max_streak)
        self.tb_gap.setValue(self.settings.turnbal_min_gap)
        self._refresh_bot_mode()
        self.prediction_confidence.setValue(round(self.settings.prediction_min_confidence * 100))
        self.review_minutes.setValue(max(1, self.settings.review_window_s // 60))
        self._refresh_toggles()

    def _on_status(self, status):
        self.values["وضعیت پایش"].setText({"MONITORING":"در حال پایش", "PAUSED":"مکث", "STOPPED":"متوقف"}.get(status,status))
        if status == "STOPPED":
            self.report_status.setText(f"گزارش پایان جلسه آماده است: {PROJECT_ROOT / 'logs' / 'session-report.txt'}")
        self._refresh()

    def _on_problem(self, message):
        self.note.setText("خطا/هشدار: " + message)
        self._add_history(["", "", "", "", datetime.now().isoformat(timespec="seconds"), "", "", "UNCERTAIN: " + message])

    def test_click(self):
        """One manual click on the resolved target; dry-run only reports the point."""
        clicker = AutoClicker(self.settings)
        try:
            report = clicker.test()
        finally:
            clicker.close()
        label = {"clicked": "کلیک شد", "dry-run": "آزمایشی", "click failed": "کلیک ناموفق"}.get(
            report.reason, "انجام نشد")
        if report.point is None:
            self.click_status.setText(
                f"کلیک تست: انجام نشد — {report.reason}. "
                "هدف کلیک تنظیم نشده: ناحیهٔ کلیک/نقطهٔ کلیک یا ناحیهٔ ۴ را انتخاب کن.")
        else:
            self.click_status.setText(f"کلیک تست: {label} — {report.reason} {report.point}")

    def _on_click(self, report):
        label = {"clicked": "کلیک شد", "click failed": "کلیک ناموفق"}.get(report.reason, "انجام نشد")
        budget = (f" | باقی‌مانده {report.remaining}/{self.settings.autoclick_max_clicks}"
                  if report.remaining else "")
        forecast = f" | پیش‌بینی: {report.forecast}" if report.forecast else ""
        self.click_status.setText(f"وضعیت کلیک: {label} — {report.reason}{budget}{forecast} "
                                  f"{report.point or ''}")
        if report.clicked:
            self._add_history(["", "", "", "", datetime.now().isoformat(timespec="seconds"), "", "",
                               f"CLICK {report.reason}: {report.point}"])

    def _on_review(self, review):
        """Log the ten-minute review window and show its numbers."""
        hits = review.get("gated_total", 0)
        accuracy = (review.get("gated_correct", 0) / hits) if hits else 0.0
        self.click_status.setText(
            f"بازبینی {review['minutes']:.0f} دقیقه‌ای: دور {review['rounds']} | کلیک {review['clicks']} | "
            f"رد‌شده {review['skipped']} | پیش‌بینی کلیک‌شده {hits} (درست {accuracy:.0%}) | "
            f"دقت مدل {review.get('base_accuracy', 0):.0%}/{review.get('cond_accuracy', 0):.0%}")
        log_path = PROJECT_ROOT / "logs" / "reviews.csv"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not log_path.exists()
        with log_path.open("a", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            if new_file:
                writer.writerow(["زمان", "دقیقه", "دور", "کلیک", "ردشده", "پیش‌بینی", "درست",
                                 "دقت کلیک", "دقت پایه", "دقت شرطی", "مدل"])
            writer.writerow([datetime.now().isoformat(timespec="seconds"), f"{review['minutes']:.1f}",
                             review["rounds"], review["clicks"], review["skipped"], hits,
                             review.get("gated_correct", 0), f"{accuracy:.3f}",
                             f"{review.get('base_accuracy', 0):.3f}",
                             f"{review.get('cond_accuracy', 0):.3f}", review.get("model", "")])

    def _on_observation(self, reading, completed, result):
        if completed is not None:
            self.confirmed += 1
        self.values["ضریب شناسایی‌شده"].setText(str(reading.value) if reading.value else "—")
        self.values["دسته"].setText({"RED":"قرمز", "GREEN":"سبز", "YELLOW":"زرد"}.get(reading.category,"نامطمئن"))
        self.values["اطمینان OCR"].setText(f"{reading.confidence:.0%}")
        if result:
            if result.bet_outcome == BetOutcome.WON:
                self.wins += 1
            elif result.bet_outcome == BetOutcome.LOST:
                self.losses += 1
            if result.cycle_outcome == CycleOutcome.WON:
                self.success += 1
            elif result.cycle_outcome == CycleOutcome.EXPIRED:
                self.expired += 1
            outcome = ("TRIGGER" if result.cycle_triggered else "SIMULATED WIN" if result.bet_outcome == BetOutcome.WON
                       else "EXPIRED" if result.cycle_outcome == CycleOutcome.EXPIRED else "SIMULATED LOSS" if result.bet_registered else "NONE")
            self._add_history([completed.round_id, str(completed.multiplier), result.category.value,
                               f"{completed.confidence:.0%}", completed.timestamp.isoformat(timespec="seconds"),
                               result.current_state.value, str(result.attempt_number), outcome])
        self._refresh()

    @staticmethod
    def _toggle_style(active: bool) -> str:
        """Glossy dark-green look; bright when the feature is on."""
        if active:
            return ("QPushButton{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,"
                    "stop:0 #7af0b8,stop:0.55 #38c98d,stop:1 #1d8f60);"
                    "border:1px solid #a9ffd4;border-radius:15px;color:#06231a;"
                    "font-size:16px;font-weight:800;padding:12px 16px;}"
                    "QPushButton:hover{border:1px solid #d4ffe6;}")
        return ("QPushButton{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,"
                "stop:0 rgba(46,92,74,235),stop:1 rgba(20,44,35,235));"
                "border:1px solid rgba(110,210,165,95);border-radius:15px;color:#cfe9dc;"
                "font-size:16px;font-weight:800;padding:12px 16px;}"
                "QPushButton:hover{border:1px solid #8bffd0;color:#f2fff8;}")

    def _refresh_toggles(self):
        real = not self.settings.autoclick_dry_run
        self.mode_button.setText("حالت واقعی ⚡" if real else "حالت شبیه‌سازی")
        self.mode_button.setToolTip("کلیک روی بازی در هر دو حالت انجام می‌شود؛ این دکمه برچسب گزارش را عوض می‌کند")
        self.mode_button.setStyleSheet(self._toggle_style(real))
        on = self.settings.autoclick_enabled
        self.autoclick_button.setText("کلیک خودکار: روشن" if on else "کلیک خودکار: خاموش")
        self.autoclick_button.setToolTip("کلیک خودکار را روشن یا خاموش می‌کند")
        self.autoclick_button.setStyleSheet(self._toggle_style(on))
        self.setWindowTitle("Crash Watcher · REAL" if real else "Crash Watcher · SIMULATED")
        self._refresh_bot_mode()

    def apply_settings(self, **changes):
        """Save a header-toggle change and push it into a running monitor."""
        candidate = replace(self.settings, **changes)
        try:
            save_settings(candidate)
        except Exception as exc:
            QMessageBox.warning(self, "تنظیمات نامعتبر", str(exc))
            return
        self.settings = candidate
        if self.worker is not None:
            self.worker.configure(candidate)
        self.autoclick.setChecked(candidate.autoclick_enabled)
        self.autoclick_real.setChecked(not candidate.autoclick_dry_run)
        self.smart_enabled.blockSignals(True)
        self.smart_enabled.setChecked(candidate.smart_review_enabled)
        self.smart_enabled.blockSignals(False)
        self.smart_play.blockSignals(True)
        self.smart_play.setChecked(candidate.smart_play_enabled)
        self.smart_play.blockSignals(False)
        self._refresh_smart_label()
        self._refresh_smart_play_label()
        self._refresh_toggles()
        self._refresh_bot_mode()
        self._refresh()

    def toggle_bot_mode(self):
        """کلیک روی زنگ: کلاسیک ↔ ترنبال؛ انتخاب در settings.json می‌ماند."""
        new_mode = "classic" if self.settings.bot_mode == "turnbal" else "turnbal"
        self.apply_settings(bot_mode=new_mode)
        self._refresh_bot_mode()

    def _refresh_bot_mode(self):
        """زنگ، برچسب حالت و متن راهنمای تب ترنبال را با حالت فعلی همگام کن."""
        turnbal = self.settings.bot_mode == "turnbal"
        if hasattr(self, "bell"):
            self.bell.set_on(turnbal)
            self.mode_chip.setText("حالت: ترنبال" if turnbal else "حالت: کلاسیک")
            self.mode_chip.setProperty("turnbal", turnbal)
            self.mode_chip.style().unpolish(self.mode_chip)
            self.mode_chip.style().polish(self.mode_chip)
        if hasattr(self, "turnbal_status"):
            self.turnbal_status.setText(
                "حالت ترنبال فعال است: مدل رنگ دست بعد را تحلیل می‌کند و ربات فقط "
                "رنگ‌های مجاز کالیبراسیون را با همان احتمالی که مدل داده شرط می‌بندد."
                if turnbal else
                "حالت کلاسیک فعال است: ربات با قواعد خودش کار می‌کند؛ این تب برای تنظیم "
                "کالیبراسیون ترنبال است و در حالت کلاسیک اثری ندارد.")
        self._refresh_turnbal_labels()

    def _refresh_turnbal_labels(self):
        """برچسب‌های روشن/خاموش تب ترنبال را با دو کلید بررسی هوشمند همگام کن."""
        if not hasattr(self, "turnbal_ai"):
            return
        for box, source, on_text, off_text in (
                (self.turnbal_ai, self.smart_enabled, "تحلیل با هوش مصنوعی: روشن",
                 "تحلیل با هوش مصنوعی: خاموش"),
                (self.turnbal_play, self.smart_play, "ورود خودکار به بازی: روشن",
                 "ورود خودکار به بازی: خاموش")):
            on = source.isChecked()
            box.blockSignals(True)
            box.setChecked(on)
            box.blockSignals(False)
            box.setText(on_text if on else off_text)
            box.setStyleSheet("QCheckBox { color: %s; font-weight: bold; }"
                              % ("#6fe3ac" if on else "#95a5a6"))

    def toggle_mode(self):
        self.apply_settings(autoclick_dry_run=not self.settings.autoclick_dry_run)

    def toggle_autoclick(self):
        self.apply_settings(autoclick_enabled=not self.settings.autoclick_enabled)

    def _refresh(self):
        active = bool(self.worker and self.worker.isRunning())
        self.start_button.setEnabled(not active)
        self.stop_button.setEnabled(active)
        self.pause_button.setEnabled(active)
        self.resume_button.setEnabled(active)
        if not active:
            self.values["وضعیت پایش"].setText("متوقف")
        self.values["وضعیت استراتژی"].setText("در انتظار ضریب زرد" if self.strategy.state == StrategyState.WAITING_FOR_YELLOW else "چرخهٔ فعال")
        self.values["شماره چرخه"].setText(str(self.strategy._cycle_id))
        self.values["شماره تلاش"].setText(str(self.strategy.attempt_number))
        self.values["تلاش‌های باقی‌مانده"].setText(str(self.strategy.remaining_attempts))
        self.values["مبلغ ثابت شبیه‌سازی"].setText(str(self.strategy.config.fixed_bet_amount))
        self.values["چرخه‌های کامل"].setText(str(self.success + self.expired))
        self.values["چرخه‌های موفق"].setText(str(self.success))
        self.values["چرخه‌های منقضی"].setText(str(self.expired))
        self.values["برد شبیه‌سازی"].setText(str(self.wins))
        self.values["باخت شبیه‌سازی"].setText(str(self.losses))
        self.values["دور تأییدشده"].setText(str(self.confirmed))
        if self.settings.autoclick_enabled and self.settings.smart_click_enabled:
            self.note.setText(f"کلیک هوشمند فعال: فقط وقتی احتمال رنگ بعدی ≥ "
                              f"{self.settings.prediction_min_confidence:.0%} باشد کلیک می‌شود؛ "
                              f"دورهای کم‌اطمینان رد می‌شوند. قانون «{self.settings.autoclick_skip_rounds} "
                              f"دست انتظار و {self.settings.autoclick_max_clicks} کلیک بعد از هر ضریب ≥ "
                              f"{self.settings.autoclick_threshold}» پابرجاست.")
        elif self.settings.autoclick_enabled:
            self.note.setText(f"کلیک خودکار فعال: از هر ضریب ≥ {self.settings.autoclick_threshold}، "
                              f"{self.settings.autoclick_skip_rounds} دست صبر و بعد "
                              f"{self.settings.autoclick_max_clicks} کلیک پشت‌سرهم با تأخیر "
                              f"{self.settings.autoclick_delay_ms} میلی‌ثانیه انجام می‌شود")
        else:
            self.note.setText("کلیک خودکار خاموش است؛ هیچ کلیکی روی بازی انجام نمی‌شود.")
        if self.settings.bot_mode == "turnbal":
            self.note.setText("حالت ترنبال: مدل رنگ دست بعد را تحلیل می‌کند و ربات فقط "
                              "داخل کالیبراسیون سه‌رنگ شرط می‌بندد؛ تصمیم‌ها در "
                              "logs/turnbal-review.jsonl ثبت می‌شوند.")

    def _add_history(self, row):
        self.history.append(row)
        log_path = PROJECT_ROOT / "logs" / "events.csv"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            if log_path.stat().st_size == 0:
                writer.writerow([self.table.horizontalHeaderItem(i).text() for i in range(self.table.columnCount())])
            writer.writerow(row)
        index = self.table.rowCount()
        self.table.insertRow(index)
        for column, value in enumerate(row):
            self.table.setItem(index, column, QTableWidgetItem(value))
        self.table.scrollToBottom()

    def clear_history(self):
        self.history.clear()
        self.table.setRowCount(0)

    def export_history(self):
        path, _ = QFileDialog.getSaveFileName(self, "خروجی تاریخچه", str(PROJECT_ROOT / "history.csv"), "CSV (*.csv)")
        if path:
            with open(path, "w", encoding="utf-8-sig", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([self.table.horizontalHeaderItem(i).text() for i in range(self.table.columnCount())])
                writer.writerows(self.history)

    def closeEvent(self, event):
        self.stop_monitoring()
        if self.worker and self.worker.isRunning():
            event.ignore()
            return
        if self.calibration and self.calibration.isRunning():
            self.calibration.wait(5000)
            if self.calibration.isRunning():
                event.ignore()
                return
        event.accept()
