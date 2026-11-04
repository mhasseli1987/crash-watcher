"""Dark simulation dashboard and mouse-based screen calibration."""
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import csv
import cv2
import mss
import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, QThread, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QApplication, QComboBox, QDialog, QFileDialog, QFormLayout,
    QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QPushButton, QSpinBox, QDoubleSpinBox, QTableWidget, QTableWidgetItem,
    QTabWidget, QVBoxLayout, QWidget)
from crash_watcher import StrategyEngine
from crash_watcher.core.enums import BetOutcome, CycleOutcome, StrategyState
from crash_watcher.vision.capture import ScreenCapture
from crash_watcher.vision.recognition import RapidOCRReader, tesseract_reader
from .config import DesktopSettings, Region, SETTINGS_PATH, PROJECT_ROOT, load_settings, save_settings
from .final_result import BangRecognizer
from .worker import MonitorWorker

STYLE = """
QWidget { background:#101319; color:#e9edf4; font-family:'Segoe UI'; font-size:13px; }
QMainWindow { background:#101319; }
QGroupBox { border:1px solid #2a303b; border-radius:12px; margin-top:14px; padding:18px 12px 12px; font-weight:600; }
QGroupBox::title { subcontrol-origin:margin; left:14px; padding:0 6px; color:#b8c2d0; }
QLineEdit,QSpinBox,QDoubleSpinBox,QComboBox { background:#1b2029; border:1px solid #343c49; border-radius:7px; padding:7px; min-height:25px; }
QPushButton { background:#252d39; border:1px solid #3c4656; border-radius:8px; padding:9px 14px; font-weight:600; }
QPushButton:hover { background:#344052; } QPushButton:disabled { color:#697586; background:#1c222b; }
QPushButton#start { background:#d7b300; color:#111319; border:0; }
QPushButton#stop { background:#82363e; border:0; }
QTableWidget { background:#151a22; alternate-background-color:#1b222c; border:1px solid #2a303b; gridline-color:#2a303b; }
QHeaderView::section { background:#252d38; color:#cbd3df; padding:7px; border:0; }
QTabWidget::pane { border:1px solid #29313d; } QTabBar::tab { background:#1b2029; padding:9px 18px; }
QTabBar::tab:selected { background:#343f50; }
"""

class RegionOverlay(QDialog):
    """Drag a rectangle on a screenshot of the chosen display."""
    def __init__(self, monitor: int, parent=None) -> None:
        super().__init__(parent)
        self.monitor = monitor
        self.selected: Region | None = None
        with mss.MSS() as backend:
            m = backend.monitors[monitor]
        geometry = QRect(m["left"], m["top"], m["width"], m["height"])
        self.setGeometry(geometry)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setModal(True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._start: QPoint | None = None
        self._end: QPoint | None = None
        with ScreenCapture(Region(monitor, geometry.x(), geometry.y(),
                                  geometry.width(), geometry.height()).capture(100)) as capture:
            frame, _ = capture.capture()
        rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        self._image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy()

    def paintEvent(self, _):
        painter = QPainter(self)
        painter.drawImage(self.rect(), self._image)
        painter.fillRect(self.rect(), QColor(6, 9, 14, 110))
        if self._start and self._end:
            rect = QRect(self._start, self._end).normalized()
            painter.drawImage(rect, self._image, rect)
            painter.setPen(QPen(QColor("#e4c000"), 3))
            painter.drawRect(rect)

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
        self.success = self.expired = self.wins = self.losses = 0
        self.setWindowTitle("Crash Watcher · SIMULATED")
        self.resize(1180, 790)
        self.setStyleSheet(STYLE)
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(24, 18, 24, 20)
        header = QHBoxLayout()
        title = QLabel("CRASH WATCHER")
        title.setStyleSheet("font-size:25px;font-weight:800;letter-spacing:1px;color:#f4f6fa")
        badge = QLabel("●  SIMULATED ONLY")
        badge.setStyleSheet("background:#363019;color:#e6ca4f;border-radius:9px;padding:9px 14px;font-weight:700")
        header.addWidget(title)
        header.addStretch()
        header.addWidget(badge)
        outer.addLayout(header)
        self.tabs = QTabWidget()
        outer.addWidget(self.tabs)
        self._build_dashboard()
        self._build_calibration()
        self._build_history()
        self._refresh()

    def _metric(self, layout, row, column, label):
        box = QFrame()
        box.setStyleSheet("QFrame{background:#1b2029;border:1px solid #2f3744;border-radius:12px;}")
        inside = QVBoxLayout(box)
        caption = QLabel(label)
        caption.setStyleSheet("color:#9eabba;border:0;background:transparent")
        value = QLabel("—")
        value.setStyleSheet("font-size:22px;font-weight:700;border:0;background:transparent")
        inside.addWidget(caption)
        inside.addWidget(value)
        layout.addWidget(box, row, column)
        return value

    def _build_dashboard(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(16)
        metrics = QGridLayout()
        labels = ["وضعیت پایش", "ضریب شناسایی‌شده", "دسته", "اطمینان OCR",
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
        self.preview.setStyleSheet("background:#171b22;border:1px solid #343b46;border-radius:12px")
        layout.addWidget(self.preview)
        self.note = QLabel("فقط پیام پایان دور ثبت می‌شود؛ هیچ کلیکی روی بازی انجام نمی‌شود.")
        self.note.setStyleSheet("color:#aeb8c5")
        layout.addWidget(self.note)
        self.tabs.addTab(page, "داشبورد")

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
        layout.addLayout(form)
        controls = QHBoxLayout()
        for title, handler in (("انتخاب ناحیهٔ نتیجه با ماوس", lambda: self.select_region(False)),
                               ("انتخاب خانهٔ آخر تاریخچه", lambda: self.select_region(True)),
                               ("ذخیره", self.save_ui_settings), ("آزمون OCR", self.calibrate),
                               ("حذف همهٔ تنظیمات", self.reset_all_settings)):
            button = QPushButton(title)
            button.clicked.connect(handler)
            controls.addWidget(button)
        layout.addLayout(controls)
        self.region_status = QLabel("ناحیهٔ نتیجه: انتخاب نشده")
        layout.addWidget(self.region_status)
        self.calibration_status = QLabel("آزمون OCR هنوز انجام نشده است.")
        layout.addWidget(self.calibration_status)
        layout.addStretch()
        self.tabs.addTab(page, "ناحیه و کالیبراسیون")
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

    def _settings_from_form(self) -> DesktopSettings:
        return replace(self.settings, result_region=self._current_region(), fixed_bet_amount=self.amount.text(),
                       max_attempts=self.attempts.value(), yellow_threshold=self.threshold.text(),
                       interval_ms=self.interval.value(), ocr_confidence=self.confidence.value(),
                       tesseract_path=self.tesseract.text().strip())

    def _update_region_status(self):
        self.region_status.setText(f"نتیجه: {self.settings.result_region} | خانهٔ آخر تاریخچه: {self.settings.history_region}")

    def select_region(self, history: bool):
        try:
            overlay = RegionOverlay(self.monitor.currentData(), self)
            if overlay.exec() and overlay.selected:
                if history:
                    self.settings = replace(self.settings, history_region=overlay.selected)
                else:
                    self._fill_region(overlay.selected)
                    self.settings = replace(self.settings, result_region=overlay.selected)
                self._update_region_status()
        except Exception as exc:
            QMessageBox.warning(self, "انتخاب ناحیه", str(exc))

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
        qimage = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self.preview.setPixmap(QPixmap.fromImage(qimage).scaled(self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def start_monitoring(self):
        if self.worker and self.worker.isRunning():
            return
        if self.settings.result_region is None and all(self.region_fields[name].value() in (0, 20) for name in self.region_fields):
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
        self._refresh()

    def reset_all_settings(self):
        if QMessageBox.question(self, "حذف تنظیمات", "همهٔ تنظیمات محلی حذف شوند؟") != QMessageBox.StandardButton.Yes:
            return
        self.stop_monitoring()
        if SETTINGS_PATH.exists():
            SETTINGS_PATH.unlink()
        self.settings = DesktopSettings()
        self.strategy = StrategyEngine(self.settings.strategy())
        self._fill_region(Region(1,0,0,20,20))
        self.amount.setText(self.settings.fixed_bet_amount)
        self.attempts.setValue(self.settings.max_attempts)
        self.threshold.setText(self.settings.yellow_threshold)
        self._update_region_status()
        self._refresh()

    def _on_status(self, status):
        self.values["وضعیت پایش"].setText(status)
        self._refresh()

    def _on_problem(self, message):
        self.note.setText("خطا/هشدار: " + message)
        self._add_history(["", "", "", "", datetime.now().isoformat(timespec="seconds"), "", "", "UNCERTAIN: " + message])

    def _on_observation(self, reading, completed, result):
        self.values["ضریب شناسایی‌شده"].setText(str(reading.value) if reading.value else "—")
        self.values["دسته"].setText(reading.category or "UNCERTAIN")
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

    def _refresh(self):
        active = bool(self.worker and self.worker.isRunning())
        self.start_button.setEnabled(not active)
        self.stop_button.setEnabled(active)
        self.pause_button.setEnabled(active)
        self.resume_button.setEnabled(active)
        if not active:
            self.values["وضعیت پایش"].setText("متوقف")
        self.values["وضعیت استراتژی"].setText(self.strategy.state.value)
        self.values["شماره چرخه"].setText(str(self.strategy._cycle_id))
        self.values["شماره تلاش"].setText(str(self.strategy.attempt_number))
        self.values["تلاش‌های باقی‌مانده"].setText(str(self.strategy.remaining_attempts))
        self.values["مبلغ ثابت شبیه‌سازی"].setText(str(self.strategy.config.fixed_bet_amount))
        self.values["چرخه‌های کامل"].setText(str(self.success + self.expired))
        self.values["چرخه‌های موفق"].setText(str(self.success))
        self.values["چرخه‌های منقضی"].setText(str(self.expired))
        self.values["برد شبیه‌سازی"].setText(str(self.wins))
        self.values["باخت شبیه‌سازی"].setText(str(self.losses))

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
