"""Qt worker: screenshots and OCR stay off the GUI thread."""
from collections import deque
from datetime import datetime, timezone
import subprocess
import sys
from threading import Event, Thread
from time import perf_counter, time
import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from crash_watcher import StrategyEngine
from crash_watcher.core.enums import StrategyState
from crash_watcher.vision.capture import ScreenCapture, ScreenUnavailable
from crash_watcher.vision.recognition import (RapidOCRReader, parse_multiplier, preprocess,
                                              tesseract_reader)
from .config import DesktopSettings, PROJECT_ROOT
from .clicker import AutoClicker
from .color_grid import read_grid
from .final_result import BangDetector, BangRecognizer, parse_history_entry
from .smart_play import SmartPlanner
from .smart_review import SmartSnapshot, append_log, build_prompt, grid_columns, review
from .turnbal import TurnbalCalibration, build_prompt as build_turnbal_prompt
from .turnbal import review as review_turnbal


def live_multiplier(reader, image, minimum: float = 0.55):
    """Read the big running multiplier (region 2) while a bet is open."""
    try:
        text, confidence = reader(preprocess(image)[0])
    except Exception:
        return None
    if not np.isfinite(confidence) or confidence < minimum:
        return None
    return parse_multiplier(text)


def session_seconds_left(started: float, limit_minutes: int,
                         now: float | None = None) -> int | None:
    """Seconds until the session timer fires, or None when no timer is set."""
    if type(limit_minutes) is not int or limit_minutes <= 0:
        return None
    moment = time() if now is None else now
    return max(0, int(round(started + limit_minutes * 60 - moment)))

class MonitorWorker(QThread):
    observation = Signal(object, object, object)
    preview = Signal(object)
    history_observation = Signal(object, object, str)
    click_report = Signal(object)
    review_report = Signal(object)
    smart_report = Signal(object)  # «بررسی هوشمند ضریب‌ها» verdict or error
    session_finished = Signal(str, bool)  # report path, shutdown requested
    session_tick = Signal(int, int)  # seconds remaining, total seconds (0 = no timer)
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
        self._clicker: AutoClicker | None = None
        self._recent: deque[str] = deque(maxlen=30)
        self._smart_seen = 0
        self._smart_rounds = 0
        self._smart_busy = False
        self._planner: SmartPlanner | None = None
        self._chart_backend = None
        self._session_started = time()
        self._session_limit_reached = False

    def configure(self, settings: DesktopSettings) -> None:
        """Apply click settings from the window while the loop is running."""
        self.settings = settings
        if self._clicker is not None:
            self._clicker.update(settings)
        if self._planner is not None:
            # در حالت ترنبال آستانهٔ احتمال هر رنگ جدا در کالیبراسیون سنجیده
            # می‌شود؛ پس آستانهٔ عمومی نباید دوباره جلوی تصمیم را بگیرد.
            threshold = 0.0 if settings.bot_mode == "turnbal" else settings.smart_play_min_confidence
            self._planner.update(settings.smart_play_learning_minutes, threshold,
                                settings.smart_play_two_colors)

    def stop(self) -> None:
        self._stop.set()

    def pause(self) -> None:
        self._pause.set()

    def resume(self) -> None:
        self._pause.clear()
        self._resync_on_resume.set()

    def run(self) -> None:
        clicker = self._clicker = AutoClicker(self.settings)
        # دورهٔ یادگیری از همین لحظهٔ شروع پایش شمرده می‌شود.
        planner = self._planner = SmartPlanner(
            self.settings.smart_play_learning_minutes,
            0.0 if self.settings.bot_mode == "turnbal" else self.settings.smart_play_min_confidence,
            self.settings.smart_play_two_colors)
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
                    reading = None
                    history_previous = None
                    history_id = None
                    history_value = None
                    last_tick = None
                    while not self._stop.is_set():
                        limit = self.settings.session_minutes
                        remaining = session_seconds_left(self._session_started, limit)
                        if remaining != last_tick:
                            # Countdown on the dashboard; the timer keeps running
                            # through a pause because it measures wall-clock time.
                            last_tick = remaining
                            self.session_tick.emit(remaining or 0, limit * 60 if limit else 0)
                        if limit and remaining is not None and remaining <= 0:
                            # The session is over: stop cleanly so the clicker
                            # writes its report before the machine goes down.
                            self._session_limit_reached = True
                            break
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
                            if completed is not None:
                                # اول نتیجهٔ دست تازه ثبت و پیش‌بینی قبلی سنجیده
                                # می‌شود، بعد تصمیمِ آمادهٔ مدل برای دست بعد به
                                # کلیکر داده می‌شود تا در پنجرهٔ کوتاه فقط کلیک کند.
                                planner.observe(completed.round_id,
                                                clicker.color(completed.multiplier))
                                ready = (planner.take() if self._smart_play_active()
                                         else None)
                                clicker.set_smart_play(ready.bets if ready else (),
                                                       ready.cashout if ready else None)
                                self.click_report.emit(clicker.fire(completed.round_id,
                                                                    completed.multiplier))
                                review = clicker.take_review()
                                if review is not None:
                                    self.review_report.emit(review)
                                self._recent.append(str(completed.multiplier))
                                self._smart_seen += 1
                                self._smart_rounds += 1
                                if (self.settings.smart_review_enabled
                                        and self._smart_rounds >= self.settings.smart_review_every_rounds):
                                    self._smart_rounds = 0
                                    self._start_smart_review(str(completed.multiplier))
                            if clicker.wants_cashout():
                                # ضریب زندهٔ ناحیهٔ ۲ (عدد در حال بالا رفتن) را جدا
                                # بخوان؛ خوانش Bang فقط عدد پایان دست را می‌دهد.
                                # اگر همین فریم Bang بود، دست تمام شده و کلیک روی
                                # دکمهٔ شرط فقط یک شرط تازه باز می‌کند: هرگز نزن.
                                ended = reading.value is not None
                                live = None if ended else live_multiplier(reader, frame.image)
                                clicker.tick(live, ended=ended)
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
            clicker.close()
            self._clicker = None
            if self._chart_backend is not None:
                self._chart_backend.close()
                self._chart_backend = None
            self.status.emit("STOPPED")
            if self._session_limit_reached:
                shutdown = self.settings.shutdown_after_session
                self.session_finished.emit(str(clicker.report_path), shutdown)
                if shutdown:
                    self._schedule_shutdown()

    def _chart_grid(self) -> tuple[tuple[str, ...], ...]:
        """Region 3 once, as letters; empty when the chart cannot be read."""
        region = self.settings.chart_region
        if region is None:
            return ()
        try:
            import mss
            if self._chart_backend is None:
                self._chart_backend = mss.mss()
            raw = self._chart_backend.grab({"left": region.x, "top": region.y,
                                            "width": region.width, "height": region.height})
            return grid_columns(read_grid(np.asarray(raw)[:, :, :3]))
        except Exception:
            return ()

    def _smart_play_active(self) -> bool:
        """ورود هوشمند فقط وقتی معنا دارد که بررسی هوشمند هم روشن باشد."""
        if self.settings.bot_mode == "turnbal":
            return bool(self.settings.smart_review_enabled)
        return bool(self.settings.smart_review_enabled and self.settings.smart_play_enabled)

    def _start_smart_review(self, current: str) -> None:
        """Ask the local model in a side thread so the monitoring loop never waits."""
        if self._smart_busy:
            return
        self._smart_busy = True
        snapshot = SmartSnapshot(recent=tuple(self._recent), current=current,
                                 grid_columns=self._chart_grid(), rounds_seen=self._smart_seen)
        Thread(target=self._smart_review_thread, args=(snapshot, self.settings),
               daemon=True).start()

    def _smart_review_thread(self, snapshot: SmartSnapshot, settings: DesktopSettings) -> None:
        started = perf_counter()
        turnbal = settings.bot_mode == "turnbal"
        calibration = TurnbalCalibration.from_settings(settings)
        try:
            if turnbal:
                # تصمیم مدل پیش از دیدنِ آستانه‌های رنگ گرفته و بعد با همان
                # کالیبراسیون سه‌رنگ تطبیق داده می‌شود.
                printed = build_turnbal_prompt(snapshot, calibration,
                                               settings.autoclick_threshold,
                                               settings.smart_review_rules,
                                               allow_two=settings.smart_play_two_colors)
                verdict, raw, prompt = review_turnbal(
                    settings.smart_review_url, settings.smart_review_key,
                    settings.smart_review_model, snapshot, calibration,
                    settings.smart_review_timeout_s, settings.autoclick_threshold,
                    settings.smart_review_rules,
                    allow_two=settings.smart_play_two_colors)
                prompt = prompt or printed
            else:
                verdict, raw, prompt = review(settings.smart_review_url, settings.smart_review_key,
                                              settings.smart_review_model, snapshot,
                                              settings.smart_review_timeout_s,
                                              settings.autoclick_threshold, settings.smart_review_rules,
                                              min_confidence=settings.smart_play_min_confidence,
                                              allow_two=settings.smart_play_two_colors)
        except Exception as exc:
            self._smart_busy = False
            append_log(PROJECT_ROOT / "logs" / ("turnbal-review.jsonl" if turnbal
                                                 else "smart-review.jsonl"),
                       {"round": snapshot.current, "ok": False, "mode": settings.bot_mode,
                        "error": str(exc)})
            fallback = (build_turnbal_prompt(snapshot, calibration, settings.autoclick_threshold,
                                             settings.smart_review_rules,
                                             allow_two=settings.smart_play_two_colors)
                        if turnbal else
                        build_prompt(snapshot, settings.autoclick_threshold,
                                     settings.smart_review_rules,
                                     min_confidence=settings.smart_play_min_confidence,
                                     allow_two=settings.smart_play_two_colors))
            self.smart_report.emit({"ok": False, "round": snapshot.current,
                                    "mode": settings.bot_mode,
                                    "error": str(exc),
                                    **(self._planner.stats() if self._planner else {}),
                                    "prompt": fallback})
            return
        self._smart_busy = False
        elapsed = perf_counter() - started
        planner = self._planner
        if planner is not None:
            # تصمیم برای دست بعد آماده می‌شود، پیش از بسته‌شدن ضریب جاری.
            planner.store(verdict, snapshot.current)
        append_log(PROJECT_ROOT / "logs" / ("turnbal-review.jsonl" if turnbal
                                             else "smart-review.jsonl"),
                   {"round": snapshot.current, "ok": True, "mode": settings.bot_mode,
                    "bet": verdict.bet if verdict else None,
                    "second": verdict.second if verdict else None,
                    "confidence": verdict.confidence if verdict else None,
                    "reason": verdict.reason if verdict else "", "raw": raw,
                    "phase": planner.phase if planner else None,
                    "seconds": round(elapsed, 2), "model": settings.smart_review_model})
        self.smart_report.emit({"ok": True, "round": snapshot.current, "verdict": verdict,
                                "mode": settings.bot_mode,
                                "raw": raw, "prompt": prompt, "seconds": elapsed,
                                "bets": verdict.bets if verdict else (),
                                **(planner.stats() if planner else {})})

    def _schedule_shutdown(self) -> None:
        """Ask Windows to power off shortly after the report is on disk."""
        try:
            if sys.platform != "win32":
                self.problem.emit("خاموش‌کردن خودکار فقط روی ویندوز انجام می‌شود.")
                return
            subprocess.Popen(["shutdown", "/s", "/t", "60", "/c",
                              "Crash Watcher: session finished"], close_fds=True)
        except Exception as exc:
            self.problem.emit(f"خاموش‌کردن ناموفق بود: {exc}")
