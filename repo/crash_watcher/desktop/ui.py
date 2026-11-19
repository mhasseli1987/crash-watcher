"""Dark simulation dashboard and mouse-based screen calibration."""
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from time import sleep
import csv
import cv2
import mss
import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, QThread, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout,
    QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSizeGrip, QSizePolicy,
    QSpinBox, QDoubleSpinBox, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)
from crash_watcher import StrategyEngine
from crash_watcher.core.enums import BetOutcome, CycleOutcome, StrategyState
from crash_watcher.vision.capture import ScreenCapture
from crash_watcher.vision.recognition import RapidOCRReader, tesseract_reader
from .clicker import AutoClicker
from .config import (DesktopSettings, Region, SETTINGS_PATH, PROJECT_ROOT, MAX_CLICK_DELAY_MS,
                     load_settings, save_settings)
from .final_result import BangRecognizer
from .smart_review import (DEFAULT_BASE_URL, DEFAULT_MODEL, OPENAI_BASE_URL, OPENAI_MODEL,
                           chat_completions_url, probe_models)
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
QTabBar::tab { background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #3cc994, stop:1 #1f7d59); color:#04231a; border:1px solid #8bffd0; border-radius:12px; padding:10px 18px; margin:2px 3px; font-weight:700; }
QTabBar::tab:hover { background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #55ddaa, stop:1 #2a9a6e); color:#02231a; border:1px solid #b8ffdb; }
QTabBar::tab:selected { background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #7af0b8, stop:1 #2aa877); color:#032018; border:1px solid #d6ffe9; }
QTableWidget { background:rgba(8,20,16,200); alternate-background-color:rgba(20,42,34,200); border:1px solid rgba(110,210,165,60); border-radius:14px; gridline-color:rgba(110,210,165,40); }
QTableWidget::item { padding:6px; }
QTableWidget::item:selected { background:rgba(60,201,148,90); color:#f2fff8; }
QHeaderView::section { background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #235242, stop:1 #17342a); color:#a8f0cd; padding:8px; border:0; border-right:1px solid rgba(110,210,165,45); font-weight:700; }
QFrame#metric { background:qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 rgba(38,80,64,235), stop:1 rgba(16,36,29,235)); border:1px solid rgba(110,210,165,70); border-radius:16px; }
QFrame#metric:hover { border:1px solid rgba(150,245,200,150); }
QProgressBar#session { background:rgba(10,26,21,220); border:1px solid rgba(110,210,165,60); border-radius:8px; min-height:16px; max-height:16px; text-align:center; color:#d8f5e7; font-size:11px; font-weight:700; }
QProgressBar#session::chunk { border-radius:7px; background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #38c98d, stop:0.6 #6fe3ac, stop:1 #f5c86b); }
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

class PointOverlay(QDialog):
    """Pick one point on a screenshot of the chosen display.

    Used for «محل کلیک کلاسیک»: the game's big Bet / Cash Out button. The main
    window is hidden first, so the screenshot shows the live screen.
    """
    def __init__(self, monitor: int, parent=None) -> None:
        super().__init__(parent)
        self.monitor = monitor
        self.selected: tuple[int, int] | None = None
        self.setWindowTitle("محل کلیک کلاسیک")
        with mss.MSS() as backend:
            m = dict(backend.monitors[monitor])
            raw = backend.grab({"left": m["left"], "top": m["top"],
                                "width": m["width"], "height": m["height"]})
        self.setGeometry(QRect(m["left"], m["top"], m["width"], m["height"]))
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                            | Qt.WindowType.Tool)
        self.setModal(True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._hover: QPoint | None = None
        frame = np.asarray(raw)[:, :, :3].copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0],
                             QImage.Format.Format_RGB888).copy()

    def paintEvent(self, _):
        painter = QPainter(self)
        painter.drawImage(self.rect(), self._image)
        painter.fillRect(self.rect(), QColor(6, 9, 14, 90))
        painter.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        painter.setPen(QColor("#f4f6fa"))
        painter.drawText(self.rect().adjusted(0, 24, 0, 0),
                         Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                         "روی دکمهٔ کلاسیک (Bet / Cash Out) کلیک کنید — Esc برای انصراف")
        if self._hover is not None:
            painter.setPen(QPen(QColor("#7af0b8"), 2))
            x, y = self._hover.x(), self._hover.y()
            painter.drawEllipse(x - 14, y - 14, 28, 28)
            painter.drawLine(x - 24, y, x + 24, y)
            painter.drawLine(x, y - 24, x, y + 24)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.reject()
            return
        super().keyPressEvent(event)

    def mouseMoveEvent(self, event):
        self._hover = event.position().toPoint()
        self.update()

    def mouseReleaseEvent(self, event):
        geo = self.geometry()
        point = event.position().toPoint()
        self.selected = (geo.x() + point.x(), geo.y() + point.y())
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
    """نسخهٔ /v1/models سرور را بررسی می‌کند (FreeLLMAPI محلی یا OpenAI)."""
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
        subtitle = QLabel("داشبورد هوشمند — پایش، کلیک و تایمر جلسه")
        subtitle.setStyleSheet("font-size:12px;color:#7fa694;background:transparent")
        subtitle.setWordWrap(True)
        titles.addWidget(title)
        titles.addWidget(subtitle)
        header.addLayout(titles)
        header.addStretch()
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
        session = QVBoxLayout()
        session.setSpacing(6)
        session_line = QHBoxLayout()
        session_line.setSpacing(10)
        self.session_left = QLabel("بدون تایمر")
        self.session_left.setStyleSheet("color:#f5c86b;font-weight:700;font-size:13px;"
                                        "background:transparent")
        self.session_left.setMinimumWidth(130)
        self.session_progress = QProgressBar()
        self.session_progress.setObjectName("session")
        self.session_progress.setRange(0, 100)
        self.session_progress.setValue(0)
        self.session_progress.setTextVisible(True)
        self.session_progress.setFormat("بدون محدودیت")
        self.session_progress.setToolTip("اگر «مدت جلسه» را عددی بگذارید، پیشرفت جلسه اینجا نشان داده می‌شود")
        session_line.addWidget(self.session_left)
        session_line.addWidget(self.session_progress, 1)
        session.addLayout(session_line)
        presets = QHBoxLayout()
        presets.setSpacing(6)
        for text, minutes in (("۳۰ دقیقه", 30), ("۱ ساعت", 60), ("۳ ساعت", 180),
                              ("بدون تایمر", 0)):
            preset = QPushButton(text)
            preset.setObjectName("preset")
            preset.setCursor(Qt.CursorShape.PointingHandCursor)
            preset.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
            preset.setToolTip(f"مدت جلسه را {text} کن" if minutes else "تایمر جلسه را بردار")
            preset.clicked.connect(lambda _=False, m=minutes: self._set_session_limit(m))
            presets.addWidget(preset)
        presets.addStretch()
        session.addLayout(presets)
        layout.addLayout(session)
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
        self.session_minutes = QSpinBox()
        self.session_minutes.setRange(0, 1440)
        self.session_minutes.setSuffix(" دقیقه")
        self.session_minutes.setSingleStep(30)
        self.session_minutes.setSpecialValueText("بدون محدودیت")
        self.session_minutes.setValue(self.settings.session_minutes)
        form.addRow("مدت جلسه", self.session_minutes)
        self.shutdown_after = QCheckBox("خاموش‌کردن ویندوز در پایان جلسه (۶۰ ثانیه فرصت، لغو: shutdown /a)")
        self.shutdown_after.setChecked(self.settings.shutdown_after_session)
        form.addRow("پایان جلسه", self.shutdown_after)
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
        self.signal_rule = QCheckBox("سیگنال شبکه: دو زرد پشت‌سرهم ← کلیک زرد بعدی")
        self.signal_rule.setChecked(self.settings.signal_rule_enabled)
        form.addRow("قانون سیگنال", self.signal_rule)
        self.signal_repeat = QSpinBox()
        self.signal_repeat.setRange(1, 10)
        self.signal_repeat.setSuffix(" بار")
        self.signal_repeat.setValue(self.settings.signal_repeat)
        form.addRow("تکرار سیگنال", self.signal_repeat)
        self.signal_burst = QSpinBox()
        self.signal_burst.setRange(1, 30)
        self.signal_burst.setSuffix(" دست")
        self.signal_burst.setValue(self.settings.signal_burst_rounds)
        form.addRow("کلیک پشت‌سرهم سیگنال", self.signal_burst)
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
        self.three_x = QCheckBox("موتور ۳x: فقط با احتمال ≥۴۰٪ از نمونهٔ مشابه وارد شو")
        self.three_x.setChecked(self.settings.three_x_enabled)
        form.addRow("موتور ورود", self.three_x)
        self.three_x_minutes = QSpinBox()
        self.three_x_minutes.setRange(0, 600)
        self.three_x_minutes.setSuffix(" دقیقه")
        self.three_x_minutes.setSingleStep(15)
        self.three_x_minutes.setValue(self.settings.three_x_observation_minutes)
        form.addRow("مشاهدهٔ اولیه", self.three_x_minutes)
        self.three_x_rounds = QSpinBox()
        self.three_x_rounds.setRange(0, 5000)
        self.three_x_rounds.setSuffix(" دور")
        self.three_x_rounds.setValue(self.settings.three_x_min_rounds)
        form.addRow("حداقل دور قبل از ورود", self.three_x_rounds)
        self.three_x_probability = QSpinBox()
        self.three_x_probability.setRange(20, 95)
        self.three_x_probability.setSuffix(" %")
        self.three_x_probability.setValue(round(self.settings.three_x_probability * 100))
        form.addRow("حداقل احتمال ورود", self.three_x_probability)
        self.three_x_samples = QSpinBox()
        self.three_x_samples.setRange(1, 2000)
        self.three_x_samples.setSuffix(" نمونه")
        self.three_x_samples.setValue(self.settings.three_x_min_samples)
        form.addRow("حداقل نمونهٔ مشابه", self.three_x_samples)
        self.three_x_ocr = QSpinBox()
        self.three_x_ocr.setRange(50, 100)
        self.three_x_ocr.setSuffix(" %")
        self.three_x_ocr.setValue(round(self.settings.three_x_min_ocr * 100))
        form.addRow("حداقل اطمینان OCR", self.three_x_ocr)
        self.three_x_exit = QLineEdit(self.settings.three_x_exit)
        form.addRow("ضریب خروج", self.three_x_exit)
        self.three_x_stake = QDoubleSpinBox()
        self.three_x_stake.setRange(.01, 10)
        self.three_x_stake.setDecimals(2)
        self.three_x_stake.setSuffix(" %")
        self.three_x_stake.setSingleStep(.05)
        self.three_x_stake.setValue(self.settings.three_x_stake_pct)
        form.addRow("حداکثر مبلغ شرط از موجودی", self.three_x_stake)
        self.three_x_bankroll = QLineEdit(self.settings.three_x_bankroll)
        form.addRow("موجودی مجازی جلسه", self.three_x_bankroll)
        self.three_x_profit = QDoubleSpinBox()
        self.three_x_profit.setRange(.1, 100)
        self.three_x_profit.setSuffix(" %")
        self.three_x_profit.setSingleStep(.5)
        self.three_x_profit.setValue(self.settings.three_x_profit_limit_pct)
        form.addRow("حد سود جلسه", self.three_x_profit)
        self.three_x_loss = QDoubleSpinBox()
        self.three_x_loss.setRange(.1, 100)
        self.three_x_loss.setSuffix(" %")
        self.three_x_loss.setSingleStep(.5)
        self.three_x_loss.setValue(self.settings.three_x_loss_limit_pct)
        form.addRow("حد ضرر جلسه", self.three_x_loss)
        self.three_x_streak = QSpinBox()
        self.three_x_streak.setRange(1, 10)
        self.three_x_streak.setSuffix(" باخت")
        self.three_x_streak.setValue(self.settings.three_x_stop_losses)
        form.addRow("توقف بعد از باخت متوالی", self.three_x_streak)
        self.three_x_pause = QSpinBox()
        self.three_x_pause.setRange(0, 200)
        self.three_x_pause.setSuffix(" دور")
        self.three_x_pause.setValue(self.settings.three_x_stop_rounds)
        form.addRow("مدت توقف بعد از باخت‌ها", self.three_x_pause)
        self.three_x_gap = QSpinBox()
        self.three_x_gap.setRange(1, 100)
        self.three_x_gap.setSuffix(" دور")
        self.three_x_gap.setValue(self.settings.three_x_min_gap_rounds)
        form.addRow("فاصلهٔ بین دو ورود", self.three_x_gap)
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
        classic_row = QHBoxLayout()
        self.classic_button = QPushButton("محل کلیک کلاسیک (دکمهٔ Bet / Cash Out)")
        self.classic_button.clicked.connect(self.select_classic_point)
        self.classic_test = QPushButton("کلیک تست کلاسیک")
        self.classic_test.clicked.connect(self.test_classic_click)
        classic_row.addWidget(self.classic_button)
        classic_row.addWidget(self.classic_test)
        layout.addLayout(classic_row)
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
        """زبانهٔ «AI Analysis»: تحلیل ناحیه ۱ تا ۴ با FreeLLMAPI محلی یا API رسمی OpenAI."""
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
        self.smart_cashout = QCheckBox()
        self.smart_cashout.setChecked(self.settings.smart_cashout_enabled)
        self.smart_cashout.setToolTip(
            "پیش از پایان ۶ ثانیه با یک کلیک روی «محل کلیک کلاسیک» وارد بازی می‌شود و با "
            "کلیک دوم، ضریب را زیر ۱٫۹۹ می‌بندد و خارج می‌شود. بعد از ۸ باخت متوالی، "
            "دوباره ۳۰ دقیقه فقط ضریب‌ها و ترکیب رنگ‌ها را تحلیل می‌کند.")
        self._refresh_smart_cashout_label()
        self.smart_cashout.toggled.connect(self._toggle_smart_cashout)
        layout.addWidget(self.smart_cashout)
        form = QFormLayout()
        self.smart_provider = QComboBox()
        for label, value in (("FreeLLMAPI (محلی)", "freellmapi"),
                             ("OpenAI GPT (رسمی)", "openai")):
            self.smart_provider.addItem(label, value)
        self.smart_provider.setCurrentIndex(
            max(0, self.smart_provider.findData(self.settings.smart_review_provider)))
        self.smart_url = QLineEdit(self.settings.smart_review_url or DEFAULT_BASE_URL)
        self.smart_url.setPlaceholderText(DEFAULT_BASE_URL)
        self.smart_url.setToolTip("سرور سازگار با OpenAI: آدرس محلی FreeLLMAPI یا "
                                  "https://api.openai.com/v1")
        self.smart_key = QLineEdit(self.settings.smart_review_key)
        self.smart_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.smart_model = QLineEdit(self.settings.smart_review_model or DEFAULT_MODEL)
        self.smart_model.setPlaceholderText(DEFAULT_MODEL)
        self._apply_provider_defaults()
        form.addRow("مغز متفکر", self.smart_provider)
        form.addRow("آدرس سرور", self.smart_url)
        form.addRow("کلید", self.smart_key)
        form.addRow("مدل", self.smart_model)
        self.smart_provider.currentIndexChanged.connect(self._on_provider_changed)
        self.smart_every = QSpinBox()
        self.smart_every.setRange(1, 100)
        self.smart_every.setSuffix(" دست")
        self.smart_every.setValue(self.settings.smart_review_every_rounds)
        form.addRow("ارزیابی هر چند دست", self.smart_every)
        self.smart_rules = QLineEdit(self.settings.smart_review_rules)
        self.smart_rules.setPlaceholderText("اختیاری — داده‌ها بر آن مقدم است")
        form.addRow("نکتهٔ اختیاری برای مدل", self.smart_rules)
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
        note = QLabel("در این زبانه هیچ‌کدام از قواعد ربات اجرا نمی‌شود؛ مدل خودش ترتیب ضرایب، "
                      "توالی رنگ‌ها و نمودار رنگی را تحلیل می‌کند و همان تصمیم کلیک می‌شود. "
                      "با «کلیک کلاسیک»، ربات قبل از پایان ۶ ثانیه وارد بازی می‌شود و روی "
                      "ضریبی که خودش زیر ۱٫۹۹ انتخاب کرده شرط را می‌بندد؛ ۸ باخت متوالی هم "
                      "دوباره ۳۰ دقیقه تحلیل بدون شرط را شروع می‌کند. برای فعال‌شدن باید "
                      "«AI Analysis» و «ورود هوشمند به بازی» هم روشن باشند. "
                      "رویدادها در logs/smart-review.jsonl ذخیره می‌شوند.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#aeb8c5")
        layout.addWidget(note)
        layout.addStretch()
        self.tabs.addTab(self._scrollable(page), "AI Analysis")
        self.tabs.setTabToolTip(3, "AI Analysis — تحلیل ناحیه ۱ تا ۴ با FreeLLMAPI محلی یا "
                                   "API رسمی OpenAI")

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
                       signal_rule_enabled=self.signal_rule.isChecked(),
                       signal_repeat=self.signal_repeat.value(),
                       signal_burst_rounds=self.signal_burst.value(),
                       pair_flip_enabled=self.pair_flip.isChecked(),
                       row_signal_enabled=self.row_signal.isChecked(),
                       row_signal_gap=self.row_gap.value(),
                       row_signal_rows=self.row_limit.value(),
                       streak_signal_enabled=self.streak_signal.isChecked(),
                       streak_signal_length=self.streak_length.value(),
                       streak_signal_clicks=self.streak_clicks.value(),
                       value_bets=self._parse_value_bets(self.value_bets_edit.text()),
                       three_x_enabled=self.three_x.isChecked(),
                       three_x_observation_minutes=self.three_x_minutes.value(),
                       three_x_min_rounds=self.three_x_rounds.value(),
                       three_x_probability=self.three_x_probability.value() / 100,
                       three_x_min_samples=self.three_x_samples.value(),
                       three_x_min_ocr=self.three_x_ocr.value() / 100,
                       three_x_exit=self.three_x_exit.text().strip() or "3",
                       three_x_stake_pct=self.three_x_stake.value(),
                       three_x_bankroll=self.three_x_bankroll.text().strip() or "1000",
                       three_x_profit_limit_pct=self.three_x_profit.value(),
                       three_x_loss_limit_pct=self.three_x_loss.value(),
                       three_x_stop_losses=self.three_x_streak.value(),
                       three_x_stop_rounds=self.three_x_pause.value(),
                       three_x_min_gap_rounds=self.three_x_gap.value(),
                       smart_click_enabled=self.smart_click.isChecked(),
                       prediction_min_confidence=self.prediction_confidence.value() / 100,
                       smart_review_enabled=self.smart_enabled.isChecked(),
                       smart_review_provider=self._provider(),
                       smart_review_url=self.smart_url.text().strip() or DEFAULT_BASE_URL,
                       smart_review_key=self.smart_key.text().strip(),
                       smart_review_model=self.smart_model.text().strip() or DEFAULT_MODEL,
                       smart_review_timeout_s=self.settings.smart_review_timeout_s,
                       smart_review_every_rounds=self.smart_every.value(),
                       smart_review_rules=self.smart_rules.text().strip(),
                       smart_play_enabled=self.smart_play.isChecked(),
                       smart_play_learning_minutes=self.smart_learning.value(),
                       smart_play_min_confidence=self.smart_min_conf.value() / 100,
                       smart_play_two_colors=self.smart_two.isChecked(),
                       smart_cashout_enabled=self.smart_cashout.isChecked(),
                       review_window_s=self.review_minutes.value() * 60,
                       session_minutes=self.session_minutes.value(),
                       shutdown_after_session=self.shutdown_after.isChecked())

    def _refresh_autoclick_label(self) -> None:
        """Make the on/off state of autoclick obvious: text and colour both change."""
        on = self.autoclick.isChecked()
        self.autoclick.setText("کلیک خودکار: روشن" if on else "کلیک خودکار: خاموش")
        color = "#6fe3ac" if on else "#95a5a6"
        self.autoclick.setStyleSheet(f"QCheckBox {{ color: {color}; font-weight: bold; }}")

    def _refresh_smart_label(self) -> None:
        """وضعیت روشن/خاموش «AI Analysis» را با متن و رنگ نشان بده."""
        on = self.smart_enabled.isChecked()
        self.smart_enabled.setText("AI Analysis: روشن" if on else "AI Analysis: خاموش")
        color = "#6fe3ac" if on else "#95a5a6"
        self.smart_enabled.setStyleSheet(f"QCheckBox {{ color: {color}; font-weight: bold; }}")

    def _provider(self) -> str:
        return self.smart_provider.currentData() or "freellmapi"

    def _apply_provider_defaults(self) -> None:
        """آدرس و مدل پیش‌فرض همان سرویس را بگذار؛ مقدار دستیِ کاربر را عوض نکن."""
        openai = self._provider() == "openai"
        url, other_url = ((OPENAI_BASE_URL, DEFAULT_BASE_URL) if openai
                          else (DEFAULT_BASE_URL, OPENAI_BASE_URL))
        model, other_model = ((OPENAI_MODEL, DEFAULT_MODEL) if openai
                              else (DEFAULT_MODEL, OPENAI_MODEL))
        for widget, value, other in ((self.smart_url, url, other_url),
                                     (self.smart_model, model, other_model)):
            current = widget.text().strip()
            if not current or current == other:
                widget.setText(value)
        self.smart_key.setPlaceholderText("کلید رسمی OpenAI (sk-...)" if openai
                                          else "کلید یکپارچهٔ داشبورد FreeLLMAPI")

    def _on_provider_changed(self) -> None:
        self._apply_provider_defaults()

    def _refresh_smart_cashout_label(self) -> None:
        """وضعیت گزینهٔ «کلیک کلاسیک + Cash Out» را با متن و رنگ نشان بده."""
        on = self.smart_cashout.isChecked()
        self.smart_cashout.setText(
            "کلیک کلاسیک + Cash Out زیر ۱٫۹۹: روشن" if on
            else "کلیک کلاسیک + Cash Out زیر ۱٫۹۹: خاموش")
        color = "#6fe3ac" if on else "#95a5a6"
        self.smart_cashout.setStyleSheet(f"QCheckBox {{ color: {color}; font-weight: bold; }}")

    def _toggle_smart_cashout(self, on: bool) -> None:
        self._refresh_smart_cashout_label()
        self.apply_settings(smart_cashout_enabled=bool(on))
        if on and self.settings.classic_click_point is None:
            self.smart_play_status.setText(
                "برای کلیک کلاسیک، در زبانهٔ کالیبراسیون «محل کلیک کلاسیک» را روی دکمهٔ "
                "Bet / Cash Out انتخاب کنید.")

    def _refresh_smart_play_label(self) -> None:
        """وضعیت روشن/خاموش «ورود هوشمند به بازی» را واضح نشان بده."""
        on = self.smart_play.isChecked()
        self.smart_play.setText("ورود هوشمند به بازی: روشن" if on
                               else "ورود هوشمند به بازی: خاموش")
        color = "#6fe3ac" if on else "#95a5a6"
        self.smart_play.setStyleSheet(f"QCheckBox {{ color: {color}; font-weight: bold; }}")

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
        self.smart_probe_status.setText("در حال اتصال…")
        self.smart_probe = SmartProbeWorker(self.smart_url.text(), self.smart_key.text())
        self.smart_probe.done.connect(self._smart_probe_done)
        self.smart_probe.failed.connect(
            lambda message: self.smart_probe_status.setText("خطای اتصال: " + message))
        self.smart_probe.start()

    def _smart_probe_done(self, models):
        names = "، ".join(str(name) for name in models[:5])
        if models:
            self.smart_probe_status.setText(
                f"متصل شد — {len(models)} مدل در دسترس: {names}"
                " (نام یکی از همین‌ها را در فیلد «مدل» بگذارید)")
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
        point = self.settings.classic_click_point
        parts.append("محل کلیک کلاسیک: "
                     + (f"{point[0]},{point[1]}" if point else "انتخاب نشده"))
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

    def select_classic_point(self):
        """انتخاب تک‌نقطه‌ی «محل کلیک کلاسیک» روی صفحهٔ زنده."""
        error = None
        try:
            self.hide()
            QApplication.processEvents()
            sleep(0.3)  # let this window actually leave the screen before the capture
            overlay = PointOverlay(self.monitor.currentData())
            if overlay.exec() and overlay.selected:
                self.settings = replace(self.settings,
                                        classic_click_point=overlay.selected)
                self.save_ui_settings()
                self._update_region_status()
        except Exception as exc:
            error = exc
        finally:
            self.show()
            self.raise_()
            self.activateWindow()
            self.tabs.setCurrentIndex(1)
        if error is not None:
            QMessageBox.warning(self, "محل کلیک کلاسیک", str(error))

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
        self.worker.session_finished.connect(self._on_session_finished)
        self.worker.session_tick.connect(self._on_session_tick)
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
        self.signal_rule.setChecked(self.settings.signal_rule_enabled)
        self.signal_repeat.setValue(self.settings.signal_repeat)
        self.signal_burst.setValue(self.settings.signal_burst_rounds)
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
        self.three_x.setChecked(self.settings.three_x_enabled)
        self.three_x_minutes.setValue(self.settings.three_x_observation_minutes)
        self.three_x_rounds.setValue(self.settings.three_x_min_rounds)
        self.three_x_probability.setValue(round(self.settings.three_x_probability * 100))
        self.three_x_samples.setValue(self.settings.three_x_min_samples)
        self.three_x_ocr.setValue(round(self.settings.three_x_min_ocr * 100))
        self.three_x_exit.setText(self.settings.three_x_exit)
        self.three_x_stake.setValue(self.settings.three_x_stake_pct)
        self.three_x_bankroll.setText(self.settings.three_x_bankroll)
        self.three_x_profit.setValue(self.settings.three_x_profit_limit_pct)
        self.three_x_loss.setValue(self.settings.three_x_loss_limit_pct)
        self.three_x_streak.setValue(self.settings.three_x_stop_losses)
        self.three_x_pause.setValue(self.settings.three_x_stop_rounds)
        self.three_x_gap.setValue(self.settings.three_x_min_gap_rounds)
        self.smart_enabled.blockSignals(True)
        self.smart_enabled.setChecked(self.settings.smart_review_enabled)
        self.smart_enabled.blockSignals(False)
        self._refresh_smart_label()
        self.smart_provider.blockSignals(True)
        self.smart_provider.setCurrentIndex(
            max(0, self.smart_provider.findData(self.settings.smart_review_provider)))
        self.smart_provider.blockSignals(False)
        self.smart_url.setText(self.settings.smart_review_url)
        self.smart_key.setText(self.settings.smart_review_key)
        self.smart_model.setText(self.settings.smart_review_model)
        self.smart_every.setValue(self.settings.smart_review_every_rounds)
        self.smart_rules.setText(self.settings.smart_review_rules)
        self.smart_play.blockSignals(True)
        self.smart_play.setChecked(self.settings.smart_play_enabled)
        self.smart_play.blockSignals(False)
        self._refresh_smart_play_label()
        self.smart_learning.setValue(self.settings.smart_play_learning_minutes)
        self.smart_min_conf.setValue(round(self.settings.smart_play_min_confidence * 100))
        self.smart_two.setChecked(self.settings.smart_play_two_colors)
        self.smart_cashout.blockSignals(True)
        self.smart_cashout.setChecked(self.settings.smart_cashout_enabled)
        self.smart_cashout.blockSignals(False)
        self._refresh_smart_cashout_label()
        self.prediction_confidence.setValue(round(self.settings.prediction_min_confidence * 100))
        self.review_minutes.setValue(max(1, self.settings.review_window_s // 60))
        self.session_minutes.setValue(self.settings.session_minutes)
        self.shutdown_after.setChecked(self.settings.shutdown_after_session)
        self._refresh_toggles()

    def _on_status(self, status):
        self.values["وضعیت پایش"].setText({"MONITORING":"در حال پایش", "PAUSED":"مکث", "STOPPED":"متوقف"}.get(status,status))
        if status == "STOPPED":
            self.report_status.setText(f"گزارش پایان جلسه آماده است: {PROJECT_ROOT / 'logs' / 'session-report.txt'}")
        self._refresh()

    def _on_session_tick(self, remaining: int, total: int) -> None:
        """شمارش معکوس جلسه را روی نوار داشبورد بنویس."""
        if total <= 0:
            self.session_left.setText("بدون تایمر")
            self.session_left.setStyleSheet("color:#f5c86b;font-weight:700;font-size:13px;"
                                            "background:transparent")
            self.session_progress.setValue(0)
            self.session_progress.setFormat("بدون محدودیت")
            return
        remaining = max(0, int(remaining))
        hours, rest = divmod(remaining, 3600)
        minutes, seconds = divmod(rest, 60)
        clock = (f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours
                 else f"{minutes:02d}:{seconds:02d}")
        elapsed = max(0, total - remaining)
        percent = int(round(elapsed * 100 / total)) if total else 0
        urgent = remaining <= 300
        self.session_left.setText("پایان جلسه در " + clock if remaining else "جلسه تمام شد")
        self.session_left.setStyleSheet(
            "color:#ff9db0;font-weight:800;font-size:13px;background:transparent" if urgent
            else "color:#f5c86b;font-weight:700;font-size:13px;background:transparent")
        self.session_progress.setValue(percent)
        self.session_progress.setFormat(f"{elapsed // 60} از {total // 60} دقیقه")

    def _set_session_limit(self, minutes: int) -> None:
        """تایمر جلسه را از داشبورد (بدون رفتن به کالیبراسیون) عوض کن."""
        self.session_minutes.setValue(minutes)
        self.apply_settings(session_minutes=int(minutes))
        if not minutes:
            self._on_session_tick(0, 0)
        elif self.worker is None or not self.worker.isRunning():
            self.session_left.setText(f"آمادهٔ شروع — {minutes} دقیقه")
            self.session_progress.setValue(0)
            self.session_progress.setFormat(f"0 از {minutes} دقیقه")

    def _on_session_finished(self, report_path, shutdown):
        self.session_left.setText("جلسه تمام شد")
        self.session_left.setStyleSheet("color:#ff9db0;font-weight:800;font-size:13px;"
                                        "background:transparent")
        self.session_progress.setValue(100)
        self.session_progress.setFormat("پایان جلسه")
        self.report_status.setText(
            f"جلسه تمام شد — گزارش: {report_path}"
            + ("  |  ویندوز تا ۶۰ ثانیه دیگر خاموش می‌شود (لغو: shutdown /a)" if shutdown else ""))
        QMessageBox.information(
            self, "پایان جلسه",
            "گزارش پایان جلسه نوشته شد:\n" + report_path
            + ("\n\nویندوز تا ۶۰ ثانیه دیگر خاموش می‌شود.\nبرای لغو: shutdown /a" if shutdown else ""))

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
        if report.reason == "target not found":
            self.click_status.setText(
                "کلیک تست: هدف پیدا نشد — ابتدا «ناحیه ۴ — دکمه‌های شرط» را کالیبره کنید")
            return
        label = {"clicked": "کلیک شد", "dry-run": "آزمایشی", "click failed": "کلیک ناموفق"}.get(
            report.reason, "انجام نشد")
        self.click_status.setText(f"کلیک تست: {label} — {report.reason} {report.point or ''}")

    def test_classic_click(self):
        """کلیک تست روی «محل کلیک کلاسیک» (دکمهٔ Bet / Cash Out)."""
        clicker = AutoClicker(self.settings)
        try:
            report = clicker.test_classic()
        finally:
            clicker.close()
        if report.reason == "classic click point not calibrated":
            self.click_status.setText(
                "کلیک تست کلاسیک: ابتدا «محل کلیک کلاسیک» را انتخاب کنید")
            return
        label = {"clicked": "کلیک شد", "click failed": "کلیک ناموفق"}.get(
            report.reason, "انجام نشد")
        self.click_status.setText(
            f"کلیک تست کلاسیک: {label} — {report.point or ''}")

    def _on_click(self, report):
        label = {"clicked": "کلیک شد", "click failed": "کلیک ناموفق"}.get(report.reason, "انجام نشد")
        budget = (f" | باقی‌مانده {report.remaining}/{self.settings.autoclick_max_clicks}"
                  if report.remaining else "")
        forecast = f" | پیش‌بینی: {report.forecast}" if report.forecast else ""
        detail = f" | {report.detail}" if report.detail else ""
        self.click_status.setText(f"وضعیت کلیک: {label} — {report.reason}{budget}{forecast}"
                                  f"{detail} {report.point or ''}")
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
        self.smart_cashout.blockSignals(True)
        self.smart_cashout.setChecked(candidate.smart_cashout_enabled)
        self.smart_cashout.blockSignals(False)
        self._refresh_smart_label()
        self._refresh_smart_play_label()
        self._refresh_smart_cashout_label()
        self._refresh_toggles()
        self._refresh()

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
