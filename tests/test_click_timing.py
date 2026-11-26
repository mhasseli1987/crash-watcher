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


def test_partial_multi_color_execution_is_reported_not_counted(no_detection):
    sender = Sender([True, False])
    clicker = AutoClicker(_settings(), sender=sender)
    clicker.set_smart_play(("RED", "GREEN"), 1.5)
    report = clicker.fire("r1", Decimal("1.50"))
    assert len(sender.calls) == 2
    assert not report.clicked and report.status == "INPUT FAILED"
    assert report.dispatch_result == "sent 1/2"
    assert clicker.clicks == 0 and clicker._pending_rule is None


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
