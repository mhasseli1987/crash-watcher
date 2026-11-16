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
    QMessageBox, QPushButton, QScrollArea, QSizeGrip, QSizePolicy, QSpinBox, QDoubleSpinBox, QTableWidget,
    QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)
from crash_watcher import StrategyEngine
from crash_watcher.core.enums import BetOutcome, CycleOutcome, StrategyState
from crash_watcher.vision.capture import ScreenCapture
from crash_watcher.vision.recognition import RapidOCRReader, tesseract_reader
from .clicker import RULE_LABELS, AutoClicker
from .config import (DesktopSettings, Region, SETTINGS_PATH, PROJECT_ROOT, MAX_CLICK_DELAY_MS,
                     load_settings, save_settings)
from .final_result import BangRecognizer
from .worker import MonitorWorker

_FA_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")

STYLE = """
QWidget { background:#10151f; color:#e9edf4; font-family:'Segoe UI'; font-size:14px; }
QMainWindow { background:#10151f; }
QGroupBox { border:1px solid #2a303b; border-radius:16px; margin-top:14px; padding:18px 12px 12px; font-weight:600; }
QGroupBox::title { subcontrol-origin:margin; left:14px; padding:0 6px; color:#b8c2d0; }
QLineEdit,QSpinBox,QDoubleSpinBox,QComboBox { background:#1b2533; border:1px solid #343c49; border-radius:7px; padding:7px; min-height:25px; }
QPushButton { background:#252d39; border:1px solid #3c4656; border-radius:8px; padding:9px 14px; font-weight:600; }
QPushButton:hover { background:#344052; } QPushButton:pressed { background:#40516a; } QPushButton:focus { border:2px solid #e6ca4f; } QLineEdit:focus,QSpinBox:focus,QDoubleSpinBox:focus { border:1px solid #e6ca4f; } QPushButton:disabled { color:#697586; background:#1c222b; }
QPushButton#start { background:#f0ca52; color:#111319; border:0; }
QPushButton#stop { background:#7d3544; border:0; }
QTableWidget { background:#151a22; alternate-background-color:#1b222c; border:1px solid #2a303b; gridline-color:#2a303b; }
QHeaderView::section { background:#252d38; color:#cbd3df; padding:7px; border:0; }
QTabWidget::pane { border:1px solid #29313d; } QTabBar::tab { background:#1b2533; padding:9px 18px; }
QTabBar::tab:selected { background:#343f50; }
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

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = load_settings()
        self.strategy = StrategyEngine(self.settings.strategy())
        self.worker: MonitorWorker | None = None
        self.calibration: CalibrationWorker | None = None
        self.history: list[list[str]] = []
        self._pending_clicks: list[tuple[str, str, tuple[int, int] | None, datetime]] = []
        self._preview_image: QImage | None = None
        self.success = self.expired = self.wins = self.losses = 0
        self.confirmed = 0
        self.setWindowTitle("Crash Watcher · SIMULATED")
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        self.setMinimumSize(480, 400)
        self.resize(650, 460)
        screen = (self.screen() or QApplication.primaryScreen()).geometry()
        self.move(screen.left(), screen.bottom() - self.height() + 1)
        self.setStyleSheet(STYLE)
        self.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(24, 18, 24, 20)
        header = QHBoxLayout()
        title = QLabel("CRASH WATCHER")
        title.setStyleSheet("font-size:25px;font-weight:800;letter-spacing:1px;color:#f4f6fa")
        header.addWidget(title)
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
        outer.addWidget(self.tabs)
        self._build_dashboard()
        self._build_calibration()
        self._build_history()
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
        box.setMinimumWidth(118)
        box.setStyleSheet("QFrame{background:#1b2533;border:1px solid #2f3744;border-radius:16px;}")
        inside = QVBoxLayout(box)
        caption = QLabel(label)
        caption.setWordWrap(True)
        caption.setStyleSheet("color:#9eabba;border:0;background:transparent")
        value = QLabel("—")
        value.setStyleSheet("font-size:22px;font-weight:700;border:0;background:transparent")
        value.setAlignment(Qt.AlignmentFlag.AlignRight)
        if label == "ضریب شناسایی‌شده":
            value.setStyleSheet("font-size:32px;font-weight:800;color:#f0ca52;border:0;background:transparent")
        inside.addWidget(caption)
        inside.addWidget(value)
        layout.addWidget(box, row, column)
        return value

    @staticmethod
    def _scrollable(page: QWidget) -> QScrollArea:
        """Keep a page readable when the window is shrunk instead of squashing the text."""
        page.setMinimumWidth(520)
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
        self.values = {name: self._metric(metrics, i//4, i%4, name) for i, name in enumerate(labels)}
        layout.addLayout(metrics)
        controls = QHBoxLayout()
        self.start_button = QPushButton("شروع پایش")
        self.start_button.setObjectName("start")
        self.stop_button = QPushButton("توقف")
        self.stop_button.setObjectName("stop")
        self.pause_button = QPushButton("مکث")
        self.resume_button = QPushButton("ادامه")
        self.reset_button = QPushButton("بازنشانی نشست")
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
        self.preview.setStyleSheet("background:#171b22;border:1px solid #343b46;border-radius:16px")
        layout.addWidget(self.preview)
        self.note = QLabel("فقط پیام پایان دور ثبت می‌شود؛ هیچ کلیکی روی بازی انجام نمی‌شود.")
        self.note.setStyleSheet("color:#aeb8c5")
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
        self.tabs.addTab(self._scrollable(page), "ناحیه و کالیبراسیون")
        if self.settings.result_region:
            self._fill_region(self.settings.result_region)
        self._update_region_status()

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
        self.tabs.addTab(page, "تاریخچهٔ رویدادها")

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
                       smart_click_enabled=self.smart_click.isChecked(),
                       prediction_min_confidence=self.prediction_confidence.value() / 100,
                       review_window_s=self.review_minutes.value() * 60,
                       session_minutes=self.session_minutes.value(),
                       shutdown_after_session=self.shutdown_after.isChecked())

    def _refresh_autoclick_label(self) -> None:
        """Make the on/off state of autoclick obvious: text and colour both change."""
        on = self.autoclick.isChecked()
        self.autoclick.setText("کلیک خودکار: روشن" if on else "کلیک خودکار: خاموش")
        color = "#2ecc71" if on else "#95a5a6"
        self.autoclick.setStyleSheet(f"QCheckBox {{ color: {color}; font-weight: bold; }}")

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
        self.worker.session_finished.connect(self._on_session_finished)
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

    def _on_session_finished(self, report_path, shutdown):
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
        label = {"clicked": "کلیک شد", "dry-run": "آزمایشی", "click failed": "کلیک ناموفق"}.get(
            report.reason, "انجام نشد")
        self.click_status.setText(f"کلیک تست: {label} — {report.reason} {report.point or ''}")

    def _on_click(self, report):
        label = {"clicked": "کلیک شد", "click failed": "کلیک ناموفق"}.get(report.reason, "انجام نشد")
        budget = (f" | باقی‌مانده {report.remaining}/{self.settings.autoclick_max_clicks}"
                  if report.remaining else "")
        forecast = f" | پیش‌بینی: {report.forecast}" if report.forecast else ""
        self.click_status.setText(f"وضعیت کلیک: {label} — {report.reason}{budget}{forecast} "
                                  f"{report.point or ''}")
        if report.clicked:
            # Hold the row until the round it was placed for closes, so the history
            # shows the rule, the colour it backed and whether that colour arrived.
            self._pending_clicks.append((report.rule, report.target, report.point,
                                         datetime.now()))

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
            self._settle_clicks(completed)
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

    def _settle_clicks(self, completed) -> None:
        """Write one history row per click placed for the round that just closed."""
        if not self._pending_clicks:
            return
        pending, self._pending_clicks = self._pending_clicks, []
        drawn = getattr(getattr(completed, "category", None), "value", "")
        for rule, target, _point, when in pending:
            targets = [part for part in (target or "").split("+") if part]
            won = bool(targets) and drawn in targets
            self._add_history([completed.round_id, str(completed.multiplier), drawn,
                               f"{completed.confidence:.0%}",
                               when.isoformat(timespec="seconds"),
                               RULE_LABELS.get(rule, rule or "—"), "",
                               f"{'WIN' if won else 'LOSS'} {'+'.join(targets)}"])

    @staticmethod
    def _toggle_style(active: bool) -> str:
        """Glossy dark-green look; bright when the feature is on."""
        if active:
            return ("QPushButton{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,"
                    "stop:0 #4aa871,stop:0.45 #2e7a4e,stop:1 #1a4a30);"
                    "border:1px solid #63c98c;border-radius:15px;color:#f3fff7;"
                    "font-size:16px;font-weight:800;padding:12px 16px;}"
                    "QPushButton:hover{border:1px solid #8be3ad;}")
        return ("QPushButton{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,"
                "stop:0 #2a3a31,stop:1 #151f19);"
                "border:1px solid #3a5847;border-radius:15px;color:#aebfb4;"
                "font-size:16px;font-weight:800;padding:12px 16px;}"
                "QPushButton:hover{border:1px solid #63c98c;color:#e9f6ee;}")

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
