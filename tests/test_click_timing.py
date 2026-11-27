"""PHASE 7: click timing, dispatch results and decision logs.

These tests never touch a real screen, mouse or model.
"""
from decimal import Decimal
import json
import time

import pytest

from crash_watcher import StrategyEngine
from crash_watcher.core.config import StrategyConfig
from crash_watcher.desktop.clicker import AutoClicker, CLICK_PLAN_TTL_S
from crash_watcher.desktop.config import DesktopSettings, Region
from crash_watcher.desktop.worker import MonitorWorker


class Clock:
    """A monotonic clock the test can move without sleeping."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class Sender:
    """Records every dispatch and answers with a scripted plan of results."""

    def __init__(self, results=None) -> None:
        self.calls = []
        self.results = list(results or [])

    def __call__(self, x, y, restore=True):
        self.calls.append((x, y, restore))
        if self.results:
            return self.results.pop(0)
        return True


@pytest.fixture
def no_detection(monkeypatch):
    monkeypatch.setattr(AutoClicker, "_detect", lambda self: None)


def _settings(**overrides) -> DesktopSettings:
    values = dict(click_region=Region(1, 100, 200, 60, 40), autoclick_enabled=True,
                  autoclick_dry_run=False, autoclick_cooldown_ms=0, autoclick_delay_ms=0,
                  autoclick_skip_rounds=0, bot_mode="turnbal",
                  smart_play_learning_minutes=0)
    values.update(overrides)
    return DesktopSettings(**values)


def test_failed_sender_is_never_counted_as_a_click(no_detection):
    sender = Sender([False])
    clicker = AutoClicker(_settings(), sender=sender)
    clicker.set_smart_play(("RED",))
    report = clicker.fire("r1", Decimal("1.50"))
    assert not report.clicked and report.status == "INPUT FAILED"
    assert report.dispatch_result == "sent 0/1"
    assert clicker.clicks == 0 and clicker._pending_rule is None
    assert clicker._cashout_points == []


def test_cursor_moves_but_input_dispatch_fails(no_detection):
    class Failing:
        def __init__(self):
            self.calls = []

        def __call__(self, x, y, restore=True):
            self.calls.append((x, y))
            raise OSError("mouse_event failed")

    sender = Failing()
    clicker = AutoClicker(_settings(), sender=sender)
    clicker.set_smart_play(("RED",))
    report = clicker.fire("r1", Decimal("1.50"))
    assert len(sender.calls) == 1  # the cursor was moved to the point
    assert not report.clicked and report.status == "INPUT FAILED"
    assert clicker.clicks == 0


def test_duplicate_round_id_is_clicked_once(no_detection):
    sender = Sender()
    clicker = AutoClicker(_settings(), sender=sender)
    clicker.set_smart_play(("RED",))
    assert clicker.fire("r1", Decimal("1.50")).clicked
    again = clicker.fire("r1", Decimal("1.50"))
    assert not again.clicked and again.reason == "same round"
    assert len(sender.calls) == 1 and clicker.clicks == 1


def test_partial_multi_color_execution_tracks_the_sent_bet(no_detection):
    sender = Sender([False, True])
    clicker = AutoClicker(_settings(cashout_enabled=True, bet_point_red=(10, 20),
                                    bet_point_green=(30, 40)), sender=sender)
    clicker.set_smart_play(("RED", "GREEN"), 1.5)
    report = clicker.fire("r1", Decimal("1.50"))
    assert len(sender.calls) == 2
    assert report.clicked and report.status == "INPUT PARTIAL"
    assert report.dispatch_result == "sent 1/2" and report.target == "GREEN"
    assert clicker.clicks == 1 and clicker._pending_rule[0] == ("GREEN",)
    clicker.tick(Decimal("1.30"))
    clicker.tick(Decimal("1.50"))
    assert sender.calls[-1][:2] == (30, 40)


def test_scheduled_click_is_cancelled_during_detected_red_streak(no_detection):
    clock, sender = Clock(), Sender()
    settings = _settings(autoclick_delay_ms=900, result_region=Region(1, 0, 0, 100, 100))
    worker = MonitorWorker(settings, StrategyEngine(settings.strategy()))
    clicker = worker._clicker = AutoClicker(settings, sender=sender, clock=clock)
    clicker.set_smart_play(("RED",))
    assert clicker.fire("r1", Decimal("1.50")).status == "CLICK SCHEDULED"
    for _ in range(3):
        worker._observe_red_streak(clicker, Decimal("1.50"))
    assert not worker.red_hold_active()
    worker.submit_advice("GREEN")
    worker._observe_red_streak(clicker, Decimal("1.50"))
    assert worker.red_hold_active() and worker._take_advice() is None
    clock.advance(1)
    assert clicker.drain_click().status == "EXPIRED"
    clock.advance(60)
    assert not sender.calls and not worker.red_hold_active()


def test_click_deadline_expires_without_dispatch(no_detection):
    clock = Clock()
    sender = Sender()
    clicker = AutoClicker(_settings(autoclick_delay_ms=900), sender=sender, clock=clock)
    clicker.set_smart_play(("RED",))
    scheduled = clicker.fire("r1", Decimal("1.50"))
    assert scheduled.status == "CLICK SCHEDULED" and not sender.calls
    clock.advance(CLICK_PLAN_TTL_S + 1)
    expired = clicker.drain_click()
    assert expired is not None and expired.status == "EXPIRED"
    assert "click window closed" in expired.missed_deadline_reason
    assert not sender.calls and clicker.clicks == 0


def test_configured_click_delay_is_respected_without_sleeping(no_detection):
    clock = Clock()
    sender = Sender()
    clicker = AutoClicker(_settings(autoclick_delay_ms=250), sender=sender, clock=clock)
    clicker.set_smart_play(("RED",))
    started = time.perf_counter()
    assert clicker.fire("r1", Decimal("1.50")).status == "CLICK SCHEDULED"
    # Scheduling returns immediately: no sleep on the capture thread.
    assert time.perf_counter() - started < 0.2
    assert not sender.calls
    clock.advance(0.10)
    assert clicker.drain_click() is None and not sender.calls
    clock.advance(0.16)
    sent = clicker.drain_click()
    assert sent is not None and sent.clicked and sent.status == "INPUT SENT"
    assert len(sender.calls) == 1 and clicker.clicks == 1
    assert sent.latency_ms is not None and sent.dispatch_at and sent.intent_at


def test_higher_priority_rule_overrides_model_and_is_logged(no_detection, tmp_path):
    sender = Sender()
    log = tmp_path / "rounds.jsonl"
    settings = _settings(autoclick_threshold="100", value_bets={"1.50": "RED"})
    clicker = AutoClicker(settings, sender=sender, log_path=log)
    clicker.set_smart_play(("GREEN",), 1.5, prediction="GREEN", source="OLLAMA")
    report = clicker.fire("r1", Decimal("1.50"))
    assert report.clicked and report.rule == "value-bet" and report.target == "RED"
    assert "value-bet overrides model GREEN" in report.override_reason
    entry = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    assert entry["selected_strategy"] == "value-bet"
    assert entry["model_prediction"] == "GREEN"
    assert entry["prediction_source"] == "OLLAMA"
    assert entry["override_reason"]
    assert entry["final_status"] == "INPUT SENT"
    assert entry["click_intent_at"] and entry["input_dispatch_at"]
    assert entry["input_dispatch_result"] == "sent 1/1"


def test_stale_model_prediction_is_rejected(no_detection):
    clock = Clock()
    sender = Sender()
    clicker = AutoClicker(_settings(), sender=sender, clock=clock)
    clicker.set_smart_play(("GREEN",), 1.5, prediction="GREEN", source="OLLAMA")
    clock.advance(31)
    report = clicker.fire("r1", Decimal("1.50"))
    assert not report.clicked and not sender.calls
    assert clicker._smart_play == ()


def test_failed_click_does_not_corrupt_the_click_budget(no_detection):
    sender = Sender([True, False])
    settings = _settings(bankroll_enabled=True, bankroll_start="100",
                         fixed_bet_amount="10")
    clicker = AutoClicker(settings, sender=sender)
    clicker.set_smart_play(("YELLOW",))
    assert clicker.fire("r1", Decimal("12.00")).clicked
    level = clicker.double_level
    clicker.set_smart_play(("YELLOW",))
    failed = clicker.fire("r2", Decimal("12.00"))
    assert not failed.clicked and failed.status == "INPUT FAILED"
    # فقط شرط قبلی تسویه شد؛ دستِ ناموفق هیچ شرط، دوبل یا خروجی باز نکرد.
    assert clicker.clicks == 1
    assert clicker.double_level == level
    assert clicker._pending_rule is None
    assert clicker._cashout_points == []
    # The round is not retried into the wrong window.
    assert clicker.fire("r2", Decimal("12.00")).reason == "same round"


def test_pause_and_resume_drop_stale_click_without_duplicate(no_detection):
    clock = Clock()
    sender = Sender()
    clicker = AutoClicker(_settings(autoclick_delay_ms=800), sender=sender, clock=clock)
    clicker.set_smart_play(("RED",))
    assert clicker.fire("r1", Decimal("1.50")).status == "CLICK SCHEDULED"
    clicker.resume_cleanup()
    assert clicker.drain_click() is None and not sender.calls

    clicker.set_smart_play(("RED",))
    assert clicker.fire("r2", Decimal("1.60")).status == "CLICK SCHEDULED"
    clock.advance(0.9)
    sent = clicker.drain_click()
    assert sent is not None and sent.clicked and len(sender.calls) == 1
    assert clicker.fire("r2", Decimal("1.60")).reason == "same round"
    assert len(sender.calls) == 1


def test_model_inference_does_not_block_the_monitor_thread(monkeypatch):
    worker = MonitorWorker(
        _settings(result_region=Region(1, 0, 0, 100, 100), ai_provider="local",
                  ollama_model="test", cashout_enabled=False),
        StrategyEngine(StrategyConfig()))
    monkeypatch.setattr(worker, "_hybrid_ollama",
                        lambda *args, **kwargs: time.sleep(0.6))
    clicker = AutoClicker(_settings(), sender=Sender())
    worker._clicker = clicker
    worker._recent.append("1.50")
    started = time.perf_counter()
    worker._start_hybrid(clicker)
    assert time.perf_counter() - started < 0.2
    assert worker._prediction_target is not None


def test_repeated_reading_of_the_same_round_keeps_the_open_cashout(no_detection):
    """دو بار خواندن یک دست نباید پنجرهٔ خروج آن دست را پاک کند."""
    sender = Sender()
    clicker = AutoClicker(_settings(cashout_enabled=True), sender=sender)
    clicker.set_smart_play(("RED",), 1.5)
    assert clicker.fire("r1", Decimal("1.50")).clicked
    assert clicker.wants_cashout()
    deadline = clicker._cashout_deadline
    again = clicker.fire("r1", Decimal("1.50"))
    assert again.reason == "same round"
    assert clicker.wants_cashout() and clicker._cashout_deadline == deadline
    assert len(sender.calls) == 1
    # هدف خروج همان بازهٔ کاربر است و با رسیدن ضریب زنده کلیک می‌شود.
    clicker.tick(Decimal("1.70"))  # نمونهٔ قبلی برای تشخیص عدد زنده
    clicker.tick(Decimal("1.90"))
    assert clicker._cashout_confirmed is False and len(sender.calls) == 2


def test_stake_adjustment_failure_writes_one_log_line(no_detection, tmp_path):
    log = tmp_path / "rounds.jsonl"
    clicker = AutoClicker(_settings(double_bet_enabled=True, double_bet_point=None),
                          sender=Sender(), log_path=log)
    clicker.set_smart_play(("RED",))
    clicker._double_clicks = 1  # یک ×2 لازم است اما نقطهٔ آن تنظیم نشده
    report = clicker.fire("r1", Decimal("1.50"))
    assert not report.clicked and report.reason == "stake adjustment failed"
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["final_status"] == "INPUT FAILED"
    assert clicker.clicks == 0


def test_one_sequence_per_monitoring_run_with_no_round_model_requests(monkeypatch, tmp_path):
    from datetime import datetime, timedelta, timezone
    from types import SimpleNamespace

    import numpy as np
    import rapidocr
    from crash_watcher.desktop import worker as module
    from crash_watcher.desktop.final_result import ConfirmedFinal, FinalReading

    settings = _settings(result_region=Region(1, 0, 0, 100, 100), interval_ms=20,
                         smart_review_enabled=True, smart_play_enabled=True,
                         smart_click_enabled=True, turnbal_yellow_enabled=True,
                         color_sequence_count=4, bet_region=Region(1, 100, 200, 60, 40))
    worker = MonitorWorker(settings, StrategyEngine(settings.strategy()))
    sender = Sender()
    clicker = AutoClicker(settings, sender=sender)
    calls, reports = [], []
    worker.click_report.connect(reports.append)
    timestamp = lambda: datetime.now(timezone.utc)
    old = timestamp() - timedelta(seconds=8)

    def snapshot(_):
        calls.append("capture")
        return ("RED", "GREEN")

    def predict(*_):
        calls.append("model")
        return (("RED", .8), ("YELLOW", .8), ("GREEN", .2), ("GREEN", .8))

    class InlineThread:
        def __init__(self, target, args, **_):
            self.target, self.args = target, args

        def start(self):
            self.target(*self.args)

    class Capture:
        def __init__(self, _):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def capture(self):
            return SimpleNamespace(image=np.zeros((40, 100, 3), dtype=np.uint8),
                                   captured_at=timestamp()), ()

    outcomes = iter([(None, None), ("before-ready", old), ("r1", None),
                     ("r1", None), ("r2", None), ("r3", None), ("r4", None), ("r5", None)])

    def observe(*_args, **_kwargs):
        round_id, time = next(outcomes, ("stop", None))
        if round_id == "stop":
            worker.stop()
            return None
        return ConfirmedFinal(round_id, Decimal("12.00"), 1.0, time or timestamp()) if round_id else None

    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "ScreenCapture", Capture)
    monkeypatch.setattr(module, "Thread", InlineThread)
    monkeypatch.setattr(module, "AutoClicker", lambda _: clicker)
    monkeypatch.setattr(module, "capture_color_history", snapshot)
    monkeypatch.setattr(module, "request_color_sequence", predict)
    monkeypatch.setattr(module, "RapidOCRReader", lambda: None)
    monkeypatch.setattr(rapidocr, "RapidOCR", lambda: lambda _: None)
    reading = FinalReading(None, 0, "", "", False)
    monkeypatch.setattr(module, "BangRecognizer", lambda *_: SimpleNamespace(recognize=lambda _: reading))
    monkeypatch.setattr(worker.detector, "observe", observe)
    worker.detector.state = SimpleNamespace(name="ROUND_IN_PROGRESS")
    monkeypatch.setattr(worker, "_start_hybrid", lambda *_: pytest.fail("no per-round request"))
    monkeypatch.setattr(worker, "_start_smart_review", lambda *_: pytest.fail("no per-round review"))
    monkeypatch.setattr(worker, "_start_sequence_evaluation", lambda _: worker._prepare_color_sequence(settings))
    monkeypatch.setattr(clicker, "_chart_reading", lambda: None)
    worker.run()

    assert calls == ["capture", "model"]
    assert [(report.target_round_id, report.target) for report in reports if report.clicked] == [
        ("r1", "RED"), ("r2", "YELLOW"), ("r3", "YELLOW"), ("r4", "GREEN"), ("r5", "YELLOW")]
    assert sender.calls == [(110, 220, True), (150, 220, True), (150, 220, True),
                            (130, 220, True), (150, 220, True)]


def test_color_sequence_waits_for_autoclick_before_first_slot(monkeypatch, tmp_path):
    from dataclasses import replace
    from datetime import datetime, timezone
    from crash_watcher.desktop import worker as module
    from crash_watcher.desktop.final_result import ConfirmedFinal

    settings = _settings(autoclick_enabled=False, result_region=Region(1, 0, 0, 100, 100),
                         bet_region=Region(1, 100, 200, 60, 40))
    worker = MonitorWorker(settings, StrategyEngine(settings.strategy()))
    sender = Sender()
    clicker = AutoClicker(settings, sender=sender, log_path=tmp_path / "rounds.jsonl")
    worker._clicker = clicker
    monkeypatch.setattr(module, "capture_color_history", lambda _: ("GREEN",))
    monkeypatch.setattr(module, "request_color_sequence", lambda *_: (("RED", .9), ("GREEN", .9)))
    worker._prepare_color_sequence(settings)

    def fire(round_id):
        round_result = ConfirmedFinal(round_id, Decimal("3.00"), 1.0, datetime.now(timezone.utc))
        return worker._fire_color_sequence(clicker, round_result)

    assert fire("off").reason == "disabled"
    assert sender.calls == []
    worker.configure(replace(settings, autoclick_enabled=True))
    assert fire("on").target == "RED"
    assert fire("next").target == "GREEN"
    assert sender.calls == [(110, 220, True), (130, 220, True)]


def test_evaluation_discards_late_model_and_clicks_by_existing_rules(monkeypatch):
    from datetime import datetime, timezone
    from crash_watcher.desktop import worker as module
    from crash_watcher.desktop.final_result import ConfirmedFinal

    settings = _settings(result_region=Region(1, 0, 0, 30, 30),
                         bet_region=Region(1, 100, 200, 60, 40))
    worker = MonitorWorker(settings, StrategyEngine(settings.strategy()))
    clock = Clock()
    sender = Sender()
    clicker = AutoClicker(settings, sender=sender, clock=clock)
    worker._evaluation_deadline = clock.now + 60
    worker._recent.append("3.20")
    worker._first_evaluation_round.set()
    monkeypatch.setattr(module, "perf_counter", clock)
    monkeypatch.setattr(module, "capture_color_history", lambda _: ("GREEN",))
    monkeypatch.setattr(clicker, "_chart_reading", lambda: None)

    def completed(name):
        return ConfirmedFinal(name, Decimal("12.00"), 1.0, datetime.now(timezone.utc))

    assert not worker._fire_color_sequence(clicker, completed("during")).clicked

    def slow_response(_, chart, *, recent, deadline):
        assert chart == ("GREEN",) and recent == ("3.20",) and deadline == 1060
        clock.advance(60)
        return (("GREEN", .99),) * 3

    monkeypatch.setattr(module, "request_color_sequence", slow_response)
    worker._prepare_color_sequence(settings)
    assert not worker._sequence
    report = worker._fire_color_sequence(clicker, completed("after"))
    assert report.clicked and report.rule == "yellow-run" and report.target == "YELLOW"
    assert sender.calls == [(150, 220, True)]
