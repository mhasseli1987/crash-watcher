"""تست شیفت به ترنبال: کلیک کلید در یک دور، شرط از دور بعد."""
from decimal import Decimal

import pytest

from crash_watcher.desktop import clicker as clicker_module
from crash_watcher.desktop.clicker import AutoClicker
from crash_watcher.desktop.config import DesktopSettings, Region


class _Sender:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, x, y, restore=True):
        self.calls.append((x, y))
        return True


@pytest.fixture(autouse=True)
def instant_page_switch(monkeypatch):
    """مکث پس از تعویض صفحه در تست لازم نیست."""
    monkeypatch.setattr(clicker_module, "PAGE_SWITCH_PAUSE_S", 0)


def _settings() -> DesktopSettings:
    return DesktopSettings(
        autoclick_enabled=True, autoclick_cooldown_ms=0, autoclick_delay_ms=0,
        value_bet_enabled=False, pair_flip_enabled=False, streak_signal_enabled=False,
        signal_rule_enabled=False, row_signal_enabled=False, smart_click_enabled=False,
        three_x_enabled=False, stake_martingale_enabled=False,
        trenball_enabled=True, trenball_streak_length=3, trenball_stake_double_steps=3,
        trenball_toggle_point=(10, 10),
        trenball_bet_region=Region(1, 0, 0, 600, 40),
        bet_region=Region(1, 0, 100, 600, 40),
        classic_click_point=(5, 5))


def _ai_settings() -> DesktopSettings:
    return DesktopSettings(
        autoclick_enabled=True, autoclick_cooldown_ms=0, autoclick_delay_ms=0,
        value_bet_enabled=False, pair_flip_enabled=False, streak_signal_enabled=False,
        signal_rule_enabled=False, row_signal_enabled=False, smart_click_enabled=False,
        three_x_enabled=False, stake_martingale_enabled=False,
        trenball_enabled=False, trenball_stake_double_steps=3,
        trenball_toggle_point=(10, 10),
        trenball_bet_region=Region(1, 0, 0, 600, 40),
        bet_region=Region(1, 0, 100, 600, 40),
        classic_click_point=(5, 5),
        smart_play_two_colors=False, ai_trenball_enabled=True,
        ai_trenball_min_confidence=0.60)


def test_ai_confidence_switches_to_trenball_and_bets_the_model_color_same_round():
    sender = _Sender()
    clicker = AutoClicker(_ai_settings(), sender=sender)
    clicker.set_smart_play(("GREEN",), active=True, confidence=0.75)
    report = clicker.fire("r1", Decimal("1.20"))

    assert clicker._page == "trenball"
    assert report.clicked and report.rule == "trenball"
    assert report.target == "GREEN"
    # کلید تعویض صفحه و شرط ترنبال، هر دو در همان دور.
    assert sender.calls == [(10, 10), (300, 20)]


def test_ai_low_confidence_returns_to_classic():
    sender = _Sender()
    clicker = AutoClicker(_ai_settings(), sender=sender)
    clicker.set_smart_play(("RED",), active=True, confidence=0.75)
    clicker.fire("r1", Decimal("1.20"))
    sender.calls.clear()

    clicker.set_smart_play(("RED",), active=True, confidence=0.55)
    clicker.fire("r2", Decimal("1.20"))

    # زیر آستانه: کلید تعویض به کلاسیک زده می‌شود و کار عادی کلاسیک ادامه دارد
    # (اینجا ورود روی رنگ مدل، چون Cash Out کلاسیک خاموش است).
    assert clicker._page == "classic"
    assert sender.calls == [(10, 10), (100, 120)]


def test_switch_clicks_toggle_then_bets_trenball_only_next_round():
    sender = _Sender()
    clicker = AutoClicker(_settings(), sender=sender)
    reports = []
    for index in range(4):
        sender.calls.clear()
        reports.append((clicker.fire(f"r{index}", Decimal("1.20")), list(sender.calls)))

    switch_report, switch_calls = reports[2]
    # دور سومِ روند: فقط کلید Classic/Trenball زده می‌شود، هیچ شرطی.
    assert clicker._page == "trenball"
    assert not switch_report.clicked
    assert switch_calls == [(10, 10)]

    bet_report, bet_calls = reports[3]
    # دور بعد: شرط روی دکمهٔ قرمز ترنبال (یک‌سوم چپِ ناحیه ۵).
    assert bet_report.clicked and bet_report.rule == "trenball"
    assert bet_report.point == (100, 20)
    assert bet_calls == [(100, 20)]
