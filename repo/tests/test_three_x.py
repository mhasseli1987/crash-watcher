"""تست‌های موتور ۳x: مشاهدهٔ اولیه، احتمال، فاصلهٔ ورود، توقف‌ها و آمار ساعت."""
from decimal import Decimal

import pytest

from crash_watcher.desktop.clicker import AutoClicker
from crash_watcher.desktop.config import DesktopSettings, Region
from crash_watcher.desktop.three_x import ThreeXSession, bucket_of


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _settings(**changes) -> DesktopSettings:
    base = dict(three_x_enabled=True, three_x_observation_minutes=0, three_x_min_rounds=0,
                three_x_probability=0.40, three_x_min_samples=2, three_x_min_ocr=0.95,
                three_x_exit="3", three_x_stake_pct=10, three_x_bankroll="10",
                three_x_profit_limit_pct=100, three_x_loss_limit_pct=100,
                three_x_stop_losses=3, three_x_stop_rounds=30, three_x_min_gap_rounds=5)
    base.update(changes)
    return DesktopSettings(**base)


def test_observation_window_and_round_floor_block_entries():
    clock = _Clock()
    session = ThreeXSession(_settings(three_x_observation_minutes=60, three_x_min_rounds=100),
                            clock=clock)
    for index in range(120):
        session.observe(Decimal("5.00"))
    assert session.phase == "learning" and not session.decide().bet
    clock.now = 60 * 60
    session.observe(Decimal("5.00"))
    assert session.phase == "trading"


def test_probability_needs_similar_samples_and_forty_percent():
    session = ThreeXSession(_settings(three_x_probability=0.40, three_x_min_samples=20))
    for value in ("1.20", "1.20", "3.50") * 12:
        session.observe(Decimal(value))
    session.observe(Decimal("1.20"))
    probability, samples, base = session.estimate()
    assert samples >= 20 and probability >= 0.40 and base > 0
    assert session.decide().bet
    strict = ThreeXSession(_settings(three_x_probability=0.95))
    for value in ("1.20", "1.20", "3.50") * 12:
        strict.observe(Decimal(value))
    assert not strict.decide().bet
    scarce = ThreeXSession(_settings(three_x_min_samples=20))
    for value in ("1.20", "3.50"):
        scarce.observe(Decimal(value))
    assert not scarce.decide().bet


def test_entry_gap_and_loss_cooldown_block_the_next_bet():
    session = ThreeXSession(_settings(three_x_probability=0.0, three_x_min_samples=1,
                                      three_x_min_gap_rounds=5))
    session.observe(Decimal("1.20"))
    assert not session.decide().bet  # هنوز انتقالی برای نمونهٔ مشابه ثبت نشده
    session.observe(Decimal("1.20"))
    first = session.decide()
    assert first.bet and first.stake == Decimal("1.00")
    session.observe(Decimal("1.20"))  # the bet loses
    assert not session.decide().bet  # ۵ دور فاصله لازم است
    for value in ("3.50", "1.20", "3.50", "1.20"):
        assert not session.decide().bet
        session.observe(Decimal(value))
    session.observe(Decimal("1.20"))
    assert session.decide().bet


def test_three_losses_pause_thirty_rounds():
    session = ThreeXSession(_settings(three_x_probability=0.0, three_x_min_samples=1,
                                      three_x_min_gap_rounds=1, three_x_loss_limit_pct=100))
    for _ in range(10):
        session.observe(Decimal("1.20"))
        session.decide()
    assert session.losses >= 3 and session.stats()["cooldown_left"] >= 25
    assert not session.decide().bet


def test_profit_and_loss_limits_halt_the_session():
    losing = ThreeXSession(_settings(three_x_probability=0.0, three_x_min_samples=1,
                                     three_x_min_gap_rounds=1, three_x_stake_pct=10,
                                     three_x_bankroll="10", three_x_loss_limit_pct=20,
                                     three_x_stop_losses=10))
    for value in ("1.20", "1.20", "1.20") * 4:
        losing.observe(Decimal(value))
        losing.decide()
    assert losing.halted and "حد ضرر" in losing.halted_reason
    assert not losing.decide().bet
    winning = ThreeXSession(_settings(three_x_probability=0.0, three_x_min_samples=1,
                                      three_x_min_gap_rounds=1, three_x_stake_pct=10,
                                      three_x_bankroll="10", three_x_profit_limit_pct=5))
    winning.observe(Decimal("1.20"))
    winning.observe(Decimal("1.20"))
    winning.decide()
    winning.observe(Decimal("3.50"))
    assert winning.halted and "حد سود" in winning.halted_reason


def test_hourly_statistics_cover_buckets_recent_windows_and_streaks():
    session = ThreeXSession(_settings())
    for value in ("1.20", "1.20", "2.50", "3.50", "1.10", "6.00"):
        session.observe(Decimal(value))
    stats = session.stats()
    assert stats["rounds"] == 6
    assert stats["above_share"] == pytest.approx(2 / 6)
    assert stats["longest_below"] == 3
    assert stats["recent20"] == pytest.approx(2 / 6)
    assert stats["after_bucket"]["1–1.5"]["rounds"] == 3
    assert stats["after_bucket"]["1–1.5"]["above"] == 1
    lines = "\n".join(session.report_lines())
    assert "سهم ضرایب" in lines and "بعد از 1–1.5" in lines
    assert bucket_of(Decimal("9.99")) == "5+"


class _Sender:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, x, y, restore=True):
        self.calls.append((x, y, restore))
        return True


@pytest.fixture
def no_detection(monkeypatch):
    monkeypatch.setattr(AutoClicker, "_detect", lambda self: None)


def test_clicker_bets_green_only_after_enough_similar_samples(no_detection):
    sender = _Sender()
    settings = DesktopSettings(autoclick_enabled=True, autoclick_cooldown_ms=0,
                               autoclick_delay_ms=0, three_x_enabled=True,
                               three_x_observation_minutes=0, three_x_min_rounds=0,
                               three_x_min_samples=2, three_x_probability=0.40,
                               three_x_min_gap_rounds=1, three_x_stake_pct=0.25,
                               three_x_bankroll="1000", bet_point_green=(689, 834))
    clicker = AutoClicker(settings, sender=sender)
    for index, value in enumerate(("1.20", "3.50", "1.20", "3.50")):
        assert not clicker.fire(f"r{index}", Decimal(value)).clicked
    report = clicker.fire("r4", Decimal("1.20"))
    assert report.clicked and report.target == "GREEN" and report.point == (689, 834)
    assert report.rule == "three-x" and "احتمال" in report.detail
    assert sender.calls == [(689, 834, True)]


def test_clicker_skips_entries_with_low_ocr_confidence(no_detection):
    sender = _Sender()
    settings = DesktopSettings(autoclick_enabled=True, autoclick_cooldown_ms=0,
                               autoclick_delay_ms=0, three_x_enabled=True,
                               three_x_observation_minutes=0, three_x_min_rounds=0,
                               three_x_min_samples=2, three_x_probability=0.40,
                               three_x_min_gap_rounds=1, bet_point_green=(689, 834))
    clicker = AutoClicker(settings, sender=sender)
    for index, value in enumerate(("1.20", "3.50", "1.20", "3.50")):
        clicker.fire(f"r{index}", Decimal(value), confidence=0.99)
    report = clicker.fire("r4", Decimal("1.20"), confidence=0.80)
    assert not report.clicked and "OCR" in report.reason and sender.calls == []
