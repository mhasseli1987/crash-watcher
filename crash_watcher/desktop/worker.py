"""Qt worker: screenshots and OCR stay off the GUI thread."""
from datetime import datetime, timezone
from threading import Event
from time import perf_counter
import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from crash_watcher import StrategyEngine
from crash_watcher.core.enums import StrategyState
from crash_watcher.vision.capture import ScreenCapture, ScreenUnavailable
from crash_watcher.vision.recognition import RapidOCRReader, tesseract_reader
from .config import DesktopSettings
from .final_result import BangDetector, BangRecognizer, parse_history_entry

class MonitorWorker(QThread):
    observation = Signal(object, object, object)
    preview = Signal(object)
    history_observation = Signal(object, object, str)
    problem = Signal(str)
    status = Signal(str)

    def __init__(self, settings: DesktopSettings, strategy: StrategyEngine) -> None:
        super().__init__()
        if settings.result_region is None:
            raise ValueError("Result region is required")
        self.settings = settings
        self.strategy = strategy
        self._stop = Event()
        self._pause = Event()
        self._resync_on_resume = Event()
        self.detector = BangDetector()

    def stop(self) -> None:
        self._stop.set()

    def pause(self) -> None:
        self._pause.set()

    def resume(self) -> None:
        self._pause.clear()
        self._resync_on_resume.set()

    def run(self) -> None:
        try:
            if self.settings.tesseract_path:
                import pytesseract
                pytesseract.pytesseract.tesseract_cmd = self.settings.tesseract_path
                pytesseract.get_tesseract_version()
                reader = tesseract_reader
            else:
                reader = RapidOCRReader()
            recognizer = BangRecognizer(reader, self.settings.ocr_confidence)
            history_engine = None
            if self.settings.history_region:
                from rapidocr import RapidOCR
                history_engine = RapidOCR()
                # RapidOCR lazily loads its detection/classification models on
                # the first call. Paying that several-second cost when a Bang
                # first appears can make the final screen disappear before the
                # detector collects stable frames. Warm it before monitoring.
                warmup = np.zeros((64, 240, 3), dtype=np.uint8)
                cv2.putText(warmup, "1234567 1.23x", (4, 43),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                history_engine(warmup)
            with ScreenCapture(self.settings.result_region.capture(self.settings.interval_ms)) as capture:
                history_capture = (ScreenCapture(self.settings.history_region.capture(self.settings.interval_ms))
                                   if self.settings.history_region else None)
                if history_capture:
                    history_capture.__enter__()
                try:
                    self.status.emit("MONITORING")
                    previous = None
                    previous_reading = None
                    history_previous = None
                    history_id = None
                    history_value = None
                    while not self._stop.is_set():
                        if self._resync_on_resume.is_set():
                            self._resync_on_resume.clear()
                            self.detector.reset()
                            previous = previous_reading = None
                            if self.strategy.state == StrategyState.ACTIVE_CYCLE:
                                self.strategy.reset()
                        if self._pause.is_set():
                            self.status.emit("PAUSED")
                            self._stop.wait(.1)
                            continue
                        start = perf_counter()
                        try:
                            frame, events = capture.capture()
                            if events:
                                self.detector.reset()
                                if self.strategy.state == StrategyState.ACTIVE_CYCLE:
                                    self.strategy.reset()
                                previous = previous_reading = None
                                self.problem.emit("Screen geometry changed; cycle reset")
                            gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
                            thumb = cv2.resize(gray, (64, 32), interpolation=cv2.INTER_AREA)
                            changed = previous is None or float(np.mean(cv2.absdiff(thumb, previous))) >= 1.0
                            previous = thumb
                            if changed or previous_reading is None:
                                reading = recognizer.recognize(frame.image)
                                previous_reading = reading
                            else:
                                reading = previous_reading
                            # The site updates history after the Bang screen is
                            # replaced by the countdown, so observe history in
                            # every phase (OCR still runs only when it changes).
                            if history_capture:
                                strip, _ = history_capture.capture()
                                hthumb = cv2.resize(cv2.cvtColor(strip.image, cv2.COLOR_BGR2GRAY), (64, 32))
                                if history_previous is None or float(np.mean(cv2.absdiff(hthumb, history_previous))) >= 1:
                                    parsed = history_engine(strip.image)
                                    history_text = " ".join(parsed.txts or ())
                                    entry = parse_history_entry(history_text)
                                    history_id, history_value = entry if entry else (None, None)
                                    self.history_observation.emit(history_id, history_value, history_text)
                                    history_previous = hthumb
                            completed = self.detector.observe(frame.image, reading, frame.captured_at,
                                                              history_id, history_value,
                                                              require_history=history_capture is not None)
                            strategy_result = (self.strategy.process_round(completed.round_id, completed.multiplier)
                                               if completed else None)
                            self.observation.emit(reading, completed, strategy_result)
                            self.preview.emit(frame.image.copy())
                        except Exception as exc:
                            self.detector.reset()
                            previous = previous_reading = None
                            if self.strategy.state == StrategyState.ACTIVE_CYCLE:
                                self.strategy.reset()
                            self.problem.emit(str(exc))
                        self._stop.wait(max(0, self.settings.interval_ms / 1000 - (perf_counter() - start)))
                finally:
                    if history_capture:
                        history_capture.__exit__(None, None, None)
        except Exception as exc:
            self.problem.emit(str(exc))
        finally:
            self.status.emit("STOPPED")
