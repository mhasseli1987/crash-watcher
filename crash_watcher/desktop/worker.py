"""Qt worker: screenshots and OCR stay off the GUI thread."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import re
from threading import Event, Thread
from time import perf_counter
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
from .model_compare import (ModelAttempt, ModelLeaderboard, compare as compare_models,
                            parse_models)
from .smart_play import SmartPlanner
from .smart_review import SmartSnapshot, append_log, build_prompt, grid_columns, review
from .turnbal import TurnbalCalibration, TurnbalGuard, build_prompt as build_turnbal_prompt
from .turnbal import review as review_turnbal


def live_multiplier(reader, image, minimum: float = 0.55):
    """Read the big running multiplier (region 2) while a bet is open."""
    for variant in preprocess(image):
        try:
            text, confidence = reader(variant)
        except Exception:
            continue
        if np.isfinite(confidence) and confidence >= minimum:
            value = parse_multiplier(text)
            if value is not None:
                return value
    return None


def detected_lines(engine, image):
    result = engine(image)
    return [(str(text), float(score)) for text, score in
            zip(result.txts or (), result.scores or ())]


def has_win_message(lines):
    return any(score >= 0.8 and re.search(r"\byou\s+won\b", text, re.I)
               for text, score in lines)


class MonitorWorker(QThread):
    observation = Signal(object, object, object)
    preview = Signal(object)
    history_observation = Signal(object, object, str)
    click_report = Signal(object)
    review_report = Signal(object)
    smart_report = Signal(object)  # «بررسی هوشمند ضریب‌ها» verdict or error
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
        self._guard: TurnbalGuard | None = None
        # Brier هر مدل بین جلسه‌ها ذخیره می‌شود تا انتخاب خودکار واقعی باشد.
        self._leaderboard = ModelLeaderboard(PROJECT_ROOT / "logs" / "model-scores.json")
        self._compare_pending: list[tuple[str, tuple[str, ...], float]] = []
        self._chart_backend = None

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
        win_executor = None
        self._guard = (TurnbalGuard(TurnbalCalibration.from_settings(self.settings))
                       if self.settings.bot_mode == "turnbal" else None)
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
            from rapidocr import RapidOCR
            screen_engine = RapidOCR()
            win_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cashout-ocr")
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
                    cashout_wait_for_live = False
                    win_message_cleared = False
                    win_future = None
                    win_round = None
                    win_after_attempt = False
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
                            if win_future is not None and win_future.done():
                                lines = win_future.result()
                                if win_round == clicker._last_id and clicker._pending_rule is not None:
                                    if not has_win_message(lines):
                                        win_message_cleared = True
                                    elif win_message_cleared and win_after_attempt and clicker._cashout_attempted:
                                        clicker.confirm_cashout(clicker._cashout_attempt_multiplier)
                                win_future = None
                            if (win_future is None and clicker._pending_rule is not None
                                    and self.settings.cashout_enabled and not clicker._cashout_confirmed):
                                win_round = clicker._last_id
                                win_after_attempt = clicker._cashout_attempted
                                win_future = win_executor.submit(detected_lines, screen_engine, frame.image.copy())
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
                                color = clicker.color(completed.multiplier)
                                self._score_models(color)
                                if self._guard is not None:
                                    self._guard.observe(color)
                                planner.observe(completed.round_id, color)
                                ready = (planner.take() if self._smart_play_active()
                                         else None)
                                cashout = ready.cashout if ready else None
                                if ready is not None and self._guard is not None:
                                    # خروج اضطراری کالیبراسیون روی هدف کش‌اوت اعمال
                                    # می‌شود، نه فقط در متن پرامپت.
                                    cashout = self._guard.cap_cashout(cashout)
                                clicker.set_smart_play(ready.bets if ready else (), cashout)
                                report = clicker.fire(completed.round_id,
                                                      completed.multiplier)
                                if report.clicked:
                                    win_message_cleared = False
                                if report.clicked and clicker.wants_cashout():
                                    cashout_wait_for_live = True
                                if (ready is not None and self._guard is not None
                                        and report.rule == "smart-play"):
                                    self._guard.begin(ready.bets)
                                self.click_report.emit(report)
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
                            if completed is None and clicker.wants_cashout():
                                # ضریب زندهٔ ناحیهٔ ۲ (عدد در حال بالا رفتن) را جدا
                                # بخوان؛ خوانش Bang فقط عدد پایان دست را می‌دهد.
                                # اگر همین فریم Bang بود، دست تمام شده و کلیک روی
                                # دکمهٔ شرط فقط یک شرط تازه باز می‌کند: هرگز نزن.
                                if reading.value is None:
                                    live = live_multiplier(reader, frame.image)
                                    if live is not None:
                                        cashout_wait_for_live = False
                                        clicker.tick(live)
                                elif not cashout_wait_for_live:
                                    clicker.tick(ended=True)
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
            if win_executor is not None:
                win_executor.shutdown(wait=False, cancel_futures=True)
            clicker.close()
            self._clicker = None
            if self._chart_backend is not None:
                self._chart_backend.close()
                self._chart_backend = None
            self.status.emit("STOPPED")

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

    def _score_models(self, color: str) -> None:
        """پیش‌بینی همهٔ مدل‌های مقایسه‌شده را با رنگ واقعی بسنج."""
        if not self._compare_pending:
            return
        pending, self._compare_pending = self._compare_pending, []
        for model, bets, confidence in pending:
            if not bets:
                continue
            self._leaderboard.record(model, confidence, color == bets[0])
        self._leaderboard.save()

    def _ask(self, model, snapshot, settings, turnbal, calibration, timeout):
        """یک درخواست مدل؛ همین تابع در مقایسهٔ موازی هم استفاده می‌شود."""
        key = settings.resolved_key()
        if turnbal:
            return review_turnbal(settings.smart_review_url, key, model, snapshot,
                                  calibration, timeout, settings.autoclick_threshold,
                                  settings.smart_review_rules,
                                  allow_two=settings.smart_play_two_colors)
        return review(settings.smart_review_url, key, model, snapshot, timeout,
                      settings.autoclick_threshold, settings.smart_review_rules,
                      min_confidence=settings.smart_play_min_confidence,
                      allow_two=settings.smart_play_two_colors)

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
        log_path = PROJECT_ROOT / "logs" / ("turnbal-review.jsonl" if turnbal
                                           else "smart-review.jsonl")
        models = parse_models(settings.smart_compare_models)
        if len(models) < 2:
            models = [settings.smart_review_model or "auto"]
        # بودجه فقط تعیین می‌کند پاسخِ دیر برای دست بعد استفاده شود یا نه؛
        # timeout همان مقدار تنظیمات می‌ماند تا تماس عادی قطع نشود.
        budget = settings.smart_latency_budget_ms / 1000
        timeout = settings.smart_review_timeout_s
        attempts: list[ModelAttempt] = []
        chosen: ModelAttempt | None = None
        note = ""
        try:
            if len(models) == 1:
                # یک مدل: تماس مستقیم تا پیام خطای دقیق همان مدل بماند.
                verdict, raw, prompt = self._ask(models[0], snapshot, settings, turnbal,
                                                 calibration, timeout)
                attempts = [ModelAttempt(models[0], verdict, raw, prompt,
                                         perf_counter() - started)]
                chosen = attempts[0]
            else:
                # همان اسنپ‌شات موازی به همهٔ مدل‌ها؛ خطای یکی بقیه را خراب نمی‌کند.
                attempts = compare_models(
                    models, lambda model: self._ask(model, snapshot, settings, turnbal,
                                                    calibration, timeout))
                chosen, note = self._leaderboard.choose(attempts)
                if chosen is None:
                    errors = "; ".join(f"{item.model}: {item.error}"
                                       for item in attempts if item.error)
                    raise RuntimeError(note + (f" ({errors})" if errors else ""))
            verdict, raw, prompt = chosen.verdict, chosen.raw, chosen.prompt
            if turnbal:
                # محافظ‌های کالیبراسیون در کد اجرا می‌شوند، نه فقط در پرامپت.
                guard = self._guard or TurnbalGuard(calibration)
                self._guard = guard
                gated, guard_reason = guard.check(verdict)
                if gated is None and guard_reason:
                    note = guard_reason
                    raw = (raw or "") + f"\n[guard] {guard_reason}"
                verdict = gated
        except Exception as exc:
            self._smart_busy = False
            append_log(log_path,
                       {"round": snapshot.current, "ok": False, "mode": settings.bot_mode,
                        "error": str(exc),
                        "models": [{"model": item.model, "error": item.error}
                                   for item in attempts]})
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
        elapsed = perf_counter() - started
        if elapsed > budget:
            # پاسخ دیرتر از بودجه رسیده؛ برای دست بعد کهنه است و استفاده نمی‌شود.
            self._smart_busy = False
            append_log(log_path,
                       {"round": snapshot.current, "ok": False, "late": True,
                        "mode": settings.bot_mode, "seconds": round(elapsed, 2),
                        "budget": budget, "model": chosen.model if chosen else None,
                        "error": "latency budget exceeded"})
            self.smart_report.emit(
                {"ok": False, "round": snapshot.current, "mode": settings.bot_mode,
                 "late": True,
                 "error": (f"پاسخ مدل {elapsed:.1f} ثانیه طول کشید و از بودجهٔ "
                           f"{budget:.1f} ثانیه گذشت؛ برای دست بعد استفاده نشد."),
                 **(self._planner.stats() if self._planner else {})})
            return
        self._smart_busy = False
        planner = self._planner
        if planner is not None:
            # تصمیم برای دست بعد آماده می‌شود، پیش از بسته‌شدن ضریب جاری.
            planner.store(verdict, snapshot.current)
        if len(attempts) > 1 and verdict is not None:
            # پیش‌بینی همهٔ مدل‌ها با دور بعد سنجیده می‌شود تا Brier هرکدام واقعی باشد.
            self._compare_pending = [(item.model, item.verdict.bets, item.verdict.confidence)
                                     for item in attempts
                                     if item.verdict and item.verdict.bets]
        record = {"round": snapshot.current, "ok": True, "mode": settings.bot_mode,
                  "bet": verdict.bet if verdict else None,
                  "second": verdict.second if verdict else None,
                  "confidence": verdict.confidence if verdict else None,
                  "reason": verdict.reason if verdict else "", "raw": raw,
                  "phase": planner.phase if planner else None,
                  "seconds": round(elapsed, 2),
                  "model": chosen.model if chosen else settings.smart_review_model,
                  "chosen_model": chosen.model if chosen and len(attempts) > 1 else None,
                  "selection": note,
                  "models": [{"model": item.model,
                              "bet": item.verdict.bet if item.verdict else None,
                              "confidence": item.verdict.confidence if item.verdict else None,
                              "seconds": round(item.elapsed, 2), "error": item.error}
                             for item in attempts]}
        append_log(log_path, record)
        self.smart_report.emit({"ok": True, "round": snapshot.current, "verdict": verdict,
                                "mode": settings.bot_mode,
                                "raw": raw, "prompt": prompt, "seconds": elapsed,
                                "bets": verdict.bets if verdict else (),
                                "model": record["model"], "selection": note,
                                "models": record["models"],
                                "leaderboard": self._leaderboard.summary(),
                                **(planner.stats() if planner else {})})

