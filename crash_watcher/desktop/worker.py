"""Qt worker: screenshots and OCR stay off the GUI thread."""
from collections import deque
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import re
from threading import Event, Lock, Thread
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
from .prediction_manager import PredictionManager
from .ollama_provider import predict as predict_ollama, OllamaError
from .smart_review import SmartVerdict
from .smart_review import SmartSnapshot, append_log, ask, build_prompt, grid_columns, parse_verdict, review
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


def classic_prompt(snapshot: SmartSnapshot) -> str:
    """Use the confirmed multiplier history for a single play/skip decision."""
    history = []
    for raw in snapshot.recent[-30:]:
        try:
            value = Decimal(str(raw))
            if not value.is_finite() or value <= 0:
                continue
        except (InvalidOperation, ValueError, TypeError):
            continue
        color = "RED" if value < 2 else "GREEN" if value < 10 else "YELLOW"
        history.append(f"{value}={color}")
    return ("Evaluate whether to enter the NEXT classic crash round. "
            "Recent confirmed multipliers with historical color labels, oldest first: "
            + (", ".join(history) or "none")
            + ". Below 2 is a low round, 2 up to 10 is a mid round and 10 or more is a "
              "high round; the order of these values is context only and does not "
              "guarantee future outcomes. The classic game has a single bet button: do "
              "not pick a label. Choose SKIP if evidence is insufficient. Return JSON "
              "with bet PLAY or SKIP, confidence between 0 and 1, a short reason, and "
              "optional cashout from 1.40 to 1.90.")



def resolve_win_ocr(future, clicker, round_id, after_attempt, timeout=0):
    """Consume a completed OCR result only for the bet it was captured for."""
    try:
        lines = future.result(timeout=timeout)
    except FutureTimeout:
        return False
    if (round_id == clicker._last_id and clicker._pending_rule is not None
            and after_attempt and clicker._cashout_attempted and has_win_message(lines)):
        clicker.confirm_cashout(clicker._cashout_attempt_multiplier)
    return True


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
        self.detector = BangDetector(history_timeout_frames=3)
        self._clicker: AutoClicker | None = None
        self._recent: deque[str] = deque(maxlen=30)
        self._smart_seen = 0
        self._smart_rounds = 0
        self._smart_busy = False
        self._planner: SmartPlanner | None = None
        self._guard: TurnbalGuard | None = None
        # Brier هر مدل بین جلسه‌ها ذخیره می‌شود تا انتخاب خودکار واقعی باشد.
        score_file = "model-scores.json" if settings.bot_mode == "turnbal" else "classic-model-scores.json"
        self._leaderboard = ModelLeaderboard(PROJECT_ROOT / "logs" / score_file)
        self._compare_pending: list[tuple[str, tuple[str, ...], float]] = []
        self._chart_backend = None
        self._prediction = PredictionManager()
        self._prediction_target: str | None = None
        self._prediction_version = 0
        self._prediction_sequence = 0
        self._frozen_pending = None
        self._provider_locks = {source: Lock() for source in ("ROUTER", "OLLAMA")}

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
            warmup = np.zeros((64, 240, 3), dtype=np.uint8)
            cv2.putText(warmup, "You Won", (4, 43),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            screen_engine(warmup)
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
                    win_future = None
                    win_round = None
                    win_after_attempt = False
                    last_completed_at = perf_counter()
                    last_health_at = last_completed_at - 15
                    while not self._stop.is_set():
                        if self._resync_on_resume.is_set():
                            self._resync_on_resume.clear()
                            self.detector.reset()
                            self._prediction_target = None
                            self._frozen_pending = None
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
                                self._prediction_target = None
                                self._frozen_pending = None
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
                                finished, win_future = win_future, None
                                try:
                                    resolve_win_ocr(finished, clicker, win_round, win_after_attempt)
                                except Exception as exc:
                                    self.problem.emit("Cashout confirmation OCR failed: " + str(exc))
                            if (win_future is None and clicker._pending_rule is not None
                                    and self.settings.cashout_enabled and clicker._cashout_attempted
                                    and not clicker._cashout_confirmed):
                                win_round = clicker._last_id
                                win_after_attempt = clicker._cashout_attempted
                                win_future = win_executor.submit(detected_lines, screen_engine,
                                                                 frame.image.copy())
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
                            if (self._smart_play_active()
                                    and completed is None and self._prediction_target is None
                                    and self.detector.state.name == "ROUND_IN_PROGRESS"):
                                self._start_hybrid(clicker)
                            strategy_result = (self.strategy.process_round(completed.round_id, completed.multiplier)
                                               if completed and self.settings.bot_mode == "turnbal" else None)
                            self.observation.emit(reading, completed, strategy_result)
                            now = perf_counter()
                            if now - last_health_at >= 15:
                                last_health_at = now
                                append_log(PROJECT_ROOT / "logs" / "monitoring-health.jsonl",
                                           {"state": self.detector.state.name,
                                            "ocr_text": reading.raw_text[:160],
                                            "ocr_confidence": float(reading.confidence),
                                            "seconds_without_result": round(now - last_completed_at, 1),
                                            "rounds_seen": self._smart_seen,
                                            "provider": self.settings.ai_provider})
                                if now - last_completed_at >= 30:
                                    self.smart_report.emit({"ok": False, "status": "WAITING_FOR_ROUND",
                                        "error": "هنوز پایان دور شناسایی نشده است؛ ناحیهٔ ضریب فعلی را بررسی کنید. "
                                                 "آخرین متن OCR: " + reading.raw_text[:80]})
                            if completed is not None:
                                last_completed_at = perf_counter()
                                # اول نتیجهٔ دست تازه ثبت و پیش‌بینی قبلی سنجیده
                                # می‌شود، بعد تصمیمِ آمادهٔ مدل برای دست بعد به
                                # کلیکر داده می‌شود تا در پنجرهٔ کوتاه فقط کلیک کند.
                                turnbal = self.settings.bot_mode == "turnbal"
                                color = clicker.color(completed.multiplier) if turnbal else None
                                scoring_target = (self._frozen_pending or {}).get("cashout") or 1.4
                                outcome = (color if turnbal else
                                           "SUCCESS" if float(completed.multiplier) >= scoring_target else "FAIL")
                                if self._frozen_pending is not None:
                                    pending = self._frozen_pending
                                    self._frozen_pending = None
                                    append_log(PROJECT_ROOT / "logs" / "hybrid-predictions.jsonl",
                                               {"target_round_id": pending["target_round_id"],
                                                "actual_round_id": completed.round_id,
                                                "source": pending["source"],
                                                "prediction": pending["prediction"],
                                                "confidence": pending["confidence"],
                                                "actual_result": outcome,
                                                "prediction_correct": (pending["prediction"] == color if turnbal else
                                                                       outcome == "SUCCESS" if pending["prediction"] == "PLAY" else None),
                                                "context_version": pending["context_version"],
                                                "status": "SCORED",
                                                "deadline_met": pending["source"] != "NO_PREDICTION",
                                                "total_latency_ms": pending["total_latency_ms"],
                                                "providers": {source: {"prediction": answer.verdict.bet,
                                                                       "confidence": answer.verdict.confidence,
                                                                       "correct": (answer.verdict.bet == color if turnbal else
                                                                                   float(completed.multiplier) >= (answer.verdict.cashout or 1.4)
                                                                                   if answer.verdict.bet == "PLAY" else None)}
                                                              for source, answer in pending["answers"].items()},
                                                **pending["latencies"]})
                                if turnbal:
                                    self._score_models(color)
                                if self._guard is not None:
                                    self._guard.observe(color)
                                planner.observe(completed.round_id, outcome)
                                if self._smart_play_active() and self._prediction_target is not None:
                                    self._local_prediction(clicker, color)
                                    chosen = self._prediction.freeze(self._prediction_target)
                                    frozen_answers = self._prediction.answers()
                                    frozen_latencies = self._prediction.latencies_ms()
                                    frozen_total_ms = self._prediction.elapsed_ms()
                                    self._prediction_target = None
                                    self._frozen_pending = {
                                        "target_round_id": chosen.target_round_id if chosen else f"session-next-{self._prediction_sequence}",
                                        "context_version": chosen.context_version if chosen else self._prediction_version,
                                        "source": chosen.source if chosen else "NO_PREDICTION",
                                        "prediction": chosen.verdict.bet if chosen else None,
                                        "cashout": chosen.verdict.cashout if chosen else None,
                                        "confidence": chosen.verdict.confidence if chosen else None,
                                        "answers": frozen_answers,
                                        "total_latency_ms": frozen_total_ms,
                                        "latencies": frozen_latencies}
                                    verdict = chosen.verdict if chosen else None
                                    if verdict is not None and self._guard is not None:
                                        verdict, _ = self._guard.check(verdict)
                                    planner.store(verdict, completed.round_id)
                                    self.smart_report.emit({"ok": bool(chosen), "round": completed.round_id,
                                                            "target_round_id": chosen.target_round_id if chosen else None,
                                                            "source": chosen.source if chosen else "NO_PREDICTION",
                                                            "status": "FROZEN", "frozen": True,
                                                            "verdict": verdict,
                                                            **frozen_latencies,
                                                            **planner.stats()})
                                ready = (planner.take() if self._smart_play_active()
                                         else None)
                                cashout = ready.cashout if ready else None
                                if ready is not None and self._guard is not None:
                                    # خروج اضطراری کالیبراسیون روی هدف کش‌اوت اعمال
                                    # می‌شود، نه فقط در متن پرامپت.
                                    cashout = self._guard.cap_cashout(cashout)
                                clicker.set_smart_play(ready.bets if ready else (), cashout)
                                # A frame captured before the round ended may still be
                                # undergoing OCR. Allow it to finish before settling.
                                if win_future is not None and clicker._cashout_attempted:
                                    try:
                                        if resolve_win_ocr(win_future, clicker, win_round,
                                                           win_after_attempt, timeout=2.0):
                                            win_future = None
                                    except Exception as exc:
                                        win_future = None
                                        self.problem.emit("Cashout confirmation OCR failed: " + str(exc))
                                report = clicker.fire(completed.round_id,
                                                      completed.multiplier)
                                if report.clicked and clicker.wants_cashout():
                                    cashout_wait_for_live = True
                                if (ready is not None and self._guard is not None
                                        and report.rule == "smart-play"):
                                    self._guard.begin(ready.bets)
                                self.click_report.emit(report)
                                review = clicker.take_review() if turnbal else None
                                if review is not None:
                                    self.review_report.emit(review)
                                self._recent.append(str(completed.multiplier))
                                self._smart_seen += 1
                                self._smart_rounds += 1
                                if (not self._smart_play_active() and self.settings.smart_review_enabled
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
                            self._prediction_target = None
                            self._frozen_pending = None
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

    def _start_hybrid(self, clicker: AutoClicker) -> None:
        """Prepare a provisional next-round decision from confirmed history."""
        self._prediction_sequence += 1
        target = f"session-next-{self._prediction_sequence}"
        version = self._prediction.begin(target, 120.0)
        self._prediction_target, self._prediction_version = target, version
        self._local_prediction(clicker)
        settings = self.settings
        snapshot = SmartSnapshot(recent=tuple(self._recent), current="",
                                 rounds_seen=self._smart_seen)
        Thread(target=self._hybrid_ai, args=(target, version, snapshot, settings),
               daemon=True).start()

    def _local_prediction(self, clicker: AutoClicker, confirmed_color: str | None = None) -> None:
        target = self._prediction_target
        if target is None:
            return
        if self.settings.bot_mode == "classic":
            self._prediction.clear_local(target, self._prediction_version)
            return
        forecaster = deepcopy(clicker.forecaster) if confirmed_color else clicker.forecaster
        if confirmed_color:
            forecaster.observe(confirmed_color)
        forecast = forecaster.forecast()
        if (forecast.samples < forecaster.min_samples
                or forecast.probability < self.settings.prediction_min_confidence):
            self._prediction.clear_local(target, self._prediction_version)
            return
        verdict = SmartVerdict(forecast.color, forecast.probability,
                               f"local {forecast.model}")
        self._prediction.submit(target, self._prediction_version, "LOCAL_ENGINE", verdict)

    def _hybrid_ai(self, target: str, version: int, snapshot: SmartSnapshot,
                   settings: DesktopSettings) -> None:
        """Run local inference without waiting for the remote network timeout."""
        if settings.ai_provider == "local":
            self._hybrid_ollama(target, version, snapshot, settings)
            return
        if settings.ollama_enabled:
            Thread(target=self._hybrid_ollama, args=(target, version, snapshot, settings),
                   daemon=True).start()
        turnbal = settings.bot_mode == "turnbal"
        calibration = TurnbalCalibration.from_settings(settings) if turnbal else None
        lock = self._provider_locks["ROUTER"]
        if not lock.acquire(blocking=False):
            return
        try:
            if not self._prediction.reserve(target, version, "ROUTER"):
                return
            try:
                verdict, _, _ = self._ask(settings.smart_review_model, snapshot, settings,
                                          turnbal, calibration,
                                          min(settings.smart_review_timeout_s,
                                              settings.smart_latency_budget_ms / 1000))
                allowed = ("RED", "GREEN", "YELLOW", "SKIP") if turnbal else ("PLAY", "SKIP")
                if verdict is not None and verdict.bet in allowed and self._prediction.submit(target, version, "ROUTER", verdict):
                    self._provider_ready(target, "ROUTER", verdict)
                    return
            except Exception as exc:
                self._provider_failed(target, "ROUTER", exc)
        finally:
            lock.release()

    def _hybrid_ollama(self, target, version, snapshot, settings):
        lock = self._provider_locks["OLLAMA"]
        if not lock.acquire(blocking=False):
            return
        try:
            if not self._prediction.reserve(target, version, "OLLAMA"):
                return
            turnbal = settings.bot_mode == "turnbal"
            calibration = TurnbalCalibration.from_settings(settings) if turnbal else None
            try:
                prompt = (build_turnbal_prompt(snapshot, calibration,
                                               settings.autoclick_threshold,
                                               settings.smart_review_rules,
                                               allow_two=settings.smart_play_two_colors)
                          if turnbal else classic_prompt(snapshot))
                verdict = predict_ollama(settings, prompt)
                allowed = ("RED", "GREEN", "YELLOW", "SKIP") if turnbal else ("PLAY", "SKIP")
                if verdict.bet in allowed and self._prediction.submit(target, version, "OLLAMA", verdict):
                    self._provider_ready(target, "OLLAMA", verdict)
            except Exception as exc:
                self._provider_failed(target, "OLLAMA", exc)
        finally:
            lock.release()

    def _provider_ready(self, target, source, verdict):
        report = {"ok": True, "target_round_id": target, "source": source,
                  "status": "PREPARED", "verdict": verdict, "bets": verdict.bets,
                  "selection": "پاسخ مدل آماده است؛ اجرا پس از تأیید پایان دور بررسی می‌شود.",
                  **(self._planner.stats() if self._planner else {})}
        self.smart_report.emit(report)
        append_log(PROJECT_ROOT / "logs" / "provider-results.jsonl",
                   {"target_round_id": target, "source": source, "status": "PREPARED",
                    "bet": verdict.bet, "confidence": verdict.confidence, "reason": verdict.reason})


    def _provider_failed(self, target, source, error):
        message = source + ": " + str(error)
        self.problem.emit(message)
        self.smart_report.emit({"ok": False, "error": message, "source": source,
                                "status": "ERROR", "target_round_id": target})
        append_log(PROJECT_ROOT / "logs" / "provider-results.jsonl",
                   {"target_round_id": target, "source": source, "status": "ERROR", "error": str(error)})


    def _ask(self, model, snapshot, settings, turnbal, calibration, timeout):
        """یک درخواست مدل؛ همین تابع در مقایسهٔ موازی هم استفاده می‌شود."""
        if settings.ai_provider == "local":
            prompt = (build_turnbal_prompt(snapshot, calibration, settings.autoclick_threshold,
                                           settings.smart_review_rules,
                                           allow_two=settings.smart_play_two_colors)
                      if turnbal else classic_prompt(snapshot))
            verdict = predict_ollama(settings, prompt)
            return verdict, verdict.reason, prompt
        key = settings.resolved_key()
        if turnbal:
            return review_turnbal(settings.smart_review_url, key, model, snapshot,
                                  calibration, timeout, settings.autoclick_threshold,
                                  settings.smart_review_rules,
                                  allow_two=settings.smart_play_two_colors)
        prompt = classic_prompt(snapshot)
        raw = ask(settings.smart_review_url, key, model, prompt, timeout)
        verdict = parse_verdict(raw)
        return (verdict if verdict and verdict.bet in ("PLAY", "SKIP") else None), raw, prompt

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
                                 grid_columns=self._chart_grid() if self.settings.bot_mode == "turnbal" else (),
                                 rounds_seen=self._smart_seen)
        Thread(target=self._smart_review_thread, args=(snapshot, self.settings),
               daemon=True).start()

    def _smart_review_thread(self, snapshot: SmartSnapshot, settings: DesktopSettings) -> None:
        started = perf_counter()
        turnbal = settings.bot_mode == "turnbal"
        calibration = TurnbalCalibration.from_settings(settings) if turnbal else None
        log_path = PROJECT_ROOT / "logs" / ("turnbal-review.jsonl" if turnbal
                                           else "smart-review.jsonl")
        models = parse_models(settings.smart_compare_models)
        if settings.ai_provider == "local":
            models = [settings.ollama_model]
        if len(models) < 2 and settings.ai_provider != "local":
            models = [settings.smart_review_model or "auto"]
        # بودجه فقط تعیین می‌کند پاسخِ دیر برای دست بعد استفاده شود یا نه؛
        # timeout همان مقدار تنظیمات می‌ماند تا تماس عادی قطع نشود.
        budget = settings.smart_latency_budget_ms / 1000
        if settings.ai_provider == "local":
            budget = settings.ollama_timeout_ms / 1000
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
                        classic_prompt(snapshot))
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
        if turnbal and len(attempts) > 1 and verdict is not None:
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

