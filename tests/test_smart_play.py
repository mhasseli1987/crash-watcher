"""تست‌های ورود هوشمند: دورهٔ یادگیری، تصمیم آماده، پوشش دو رنگ و کلیک دوتایی."""
from decimal import Decimal

import pytest

from crash_watcher.desktop.clicker import AutoClicker
from crash_watcher.desktop.config import DesktopSettings
from crash_watcher.desktop.smart_play import LEARNING, TRADING, SmartPlanner
from crash_watcher.desktop.smart_review import SmartVerdict, parse_verdict


class _Clock:
    """ساعت قلابی تا دورهٔ یادگیری بدون انتظار واقعی جلو برود."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_learning_phase_blocks_bets_then_trading_allows_them():
    clock = _Clock()
    planner = SmartPlanner(learning_minutes=30, min_confidence=0.40, clock=clock)
    planner.store(SmartVerdict("GREEN", 0.70, "روند"), "12")
    assert planner.phase == LEARNING
    assert planner.decision() is None
    clock.now = 29 * 60
    assert planner.learning and planner.decision() is None
    clock.now = 30 * 60
    assert planner.phase == TRADING
    decision = planner.take()
    assert decision is not None and decision.bets == ("GREEN",)
    assert decision.confidence == 0.70
    assert planner.decision() is None  # تصمیم هر بار یک‌بار مصرف می‌شود


def test_low_confidence_and_skip_do_not_bet():
    planner = SmartPlanner(learning_minutes=0, min_confidence=0.40, clock=_Clock())
    planner.store(SmartVerdict("RED", 0.39, "شک"), "1")
    assert planner.decision() is None
    planner.store(SmartVerdict("SKIP", 0.9, "صبر"), "2")
    assert planner.decision() is None
    planner.store(None, "3")
    assert planner.decision() is None


def test_two_colours_are_kept_or_trimmed_by_setting():
    planner = SmartPlanner(learning_minutes=0, allow_two_colors=True)
    planner.store(SmartVerdict("YELLOW", 0.45, "شک", second="RED"), "3")
    assert planner.decision().bets == ("YELLOW", "RED")
    single = SmartPlanner(learning_minutes=0, allow_two_colors=False)
    single.store(SmartVerdict("YELLOW", 0.45, "شک", second="RED"), "4")
    assert single.decision().bets == ("YELLOW",)


def test_hedge_hit_is_break_even_not_a_hit_and_weak_accuracy_stops_betting():
    planner = SmartPlanner(learning_minutes=0, min_confidence=0.40)
    planner.store(SmartVerdict("YELLOW", 0.5, "شک", second="RED"), "1")
    planner.observe("2", "RED")  # رنگ دوم زد: سربه‌سر است، نه پیش‌بینی درست
    assert planner.hits == 0 and planner.break_even == 1 and planner.tries == 1
    planner.store(SmartVerdict("GREEN", 0.5, "x"), "2")
    planner.observe("3", "YELLOW")
    assert planner.hits == 0 and planner.tries == 2
    for index in range(20):
        planner.store(SmartVerdict("GREEN", 0.9, "x"), str(index))
        planner.observe(str(index + 100), "YELLOW")
    planner.store(SmartVerdict("GREEN", 0.9, "x"), "200")
    assert planner.measured_accuracy is not None and planner.measured_accuracy < 0.4
    assert planner.decision() is None  # دقت اندازه‌گیری‌شده زیر آستانه است


def test_stale_decision_is_dropped():
    """پاسخ دیرهنگام مدل نباید روی دستی که داده‌اش عوض شده اعمال شود."""
    planner = SmartPlanner(learning_minutes=0, clock=_Clock())
    planner.observe("1", "RED")
    planner.store(SmartVerdict("GREEN", 0.7, "x"), "1")
    planner.observe("2", "RED")
    assert planner.decision() is not None
    planner.observe("3", "RED")
    planner.observe("4", "RED")
    assert planner.decision() is None


def test_parse_verdict_reads_the_second_colour():
    verdict = parse_verdict('{"bet":"YELLOW","second":"red","confidence":45,"reason":"شک"}')
    assert verdict is not None and verdict.bets == ("YELLOW", "RED")
    assert verdict.label.startswith("زرد (مون) + قرمز")
    trimmed = parse_verdict('{"bet":"RED","second":"RED"}')
    assert trimmed is not None and trimmed.second is None
    none_second = parse_verdict('{"bet":"RED","second":"none"}')
    assert none_second is not None and none_second.second is None


def _settings(**changes) -> DesktopSettings:
    base = dict(autoclick_enabled=True, autoclick_dry_run=False,
                autoclick_delay_ms=0, autoclick_cooldown_ms=0,
                smart_review_enabled=True, smart_play_enabled=True,
                smart_play_learning_minutes=0, smart_play_two_colors=True,
                bet_point_red=(10, 20), bet_point_green=(15, 25), bet_point_yellow=(30, 40))
    base.update(changes)
    return DesktopSettings(**base)


def _clicker(tmp_path, sent, **changes) -> AutoClicker:
    return AutoClicker(_settings(**changes),
                       sender=lambda x, y, restore=True: sent.append((x, y)) or True,
                       log_path=tmp_path / "rounds.jsonl",
                       summary_path=tmp_path / "summary.txt",
                       report_path=tmp_path / "report.txt")


def test_smart_play_clicks_both_colours_in_one_round(tmp_path):
    sent = []
    clicker = _clicker(tmp_path, sent)
    clicker.set_smart_play(("YELLOW", "RED"))
    report = clicker.fire("7", Decimal("2.10"))
    assert report.clicked and report.rule == "smart-play"
    assert report.target == "YELLOW+RED"
    assert sent == [(30, 40), (10, 20)]


def test_existing_rules_keep_priority_over_the_model(tmp_path):
    sent = []
    clicker = _clicker(tmp_path, sent)
    clicker.set_smart_play(("YELLOW",))
    first = clicker.fire("1", Decimal("12.00"))  # زرد: صف ۷+۳ مسلح می‌شود
    assert not first.clicked and first.remaining == 3
    assert sent == []


def test_nothing_is_clicked_while_learning(tmp_path):
    sent = []
    clicker = _clicker(tmp_path, sent, smart_play_learning_minutes=30)
    planner = SmartPlanner(30, 0.4, True)
    planner.store(SmartVerdict("RED", 0.8, "x"), "1")
    decision = planner.decision()
    clicker.set_smart_play(decision.bets if decision else ())
    report = clicker.fire("2", Decimal("1.50"))
    assert not report.clicked and report.rule != "smart-play"
    assert sent == []


def test_smart_play_settings_are_validated():
    DesktopSettings(smart_play_enabled=True, smart_play_learning_minutes=0).validate()
    for bad in ({"smart_play_learning_minutes": -1}, {"smart_play_learning_minutes": 900},
                {"smart_play_min_confidence": 0.1}, {"smart_play_min_confidence": True},
                {"smart_play_enabled": "yes"}):
        with pytest.raises(ValueError):
            DesktopSettings(**bad).validate()


def test_smart_cashout_clicks_once_when_the_live_multiplier_reaches_the_target(tmp_path):
    sent = []
    clicker = _clicker(tmp_path, sent, cashout_enabled=True)
    clicker.set_smart_play(("RED",), 1.55)
    report = clicker.fire("7", Decimal("2.10"))
    assert report.clicked and sent == [(10, 20)]
    assert clicker.wants_cashout()
    clicker.tick(Decimal("1.30"))
    clicker.tick(Decimal("1.62"))
    assert sent == [(10, 20), (10, 20)]
    assert not clicker.wants_cashout()
    clicker.tick(Decimal("1.80"))
    assert sent == [(10, 20), (10, 20)]


def test_classic_cashout_uses_the_user_min_cashout_setting(tmp_path):
    """کلاسیک مدل ندارد: هدف خروج «حداقل کش‌اوت» تنظیمات است، نه عدد ثابت 1.60."""
    sent = []
    clicker = _clicker(tmp_path, sent, cashout_enabled=True,
                       value_bets={"2.08": "GREEN"},
                       turnbal_cashout_min=1.50, turnbal_cashout_max=1.90,
                       turnbal_emergency_exit=1.40)
    clicker.set_smart_play((), None)  # مسیر کلاسیک: بدون تصمیم مدل
    report = clicker.fire("7", Decimal("2.08"))  # ضریب ویژهٔ کاربر: شرط ثبت می‌شود
    assert report.clicked and sent == [(15, 25)]
    assert clicker._cashout_target == Decimal("1.50")
    clicker.tick(Decimal("1.30"))
    clicker.tick(Decimal("1.52"))
    assert sent == [(15, 25), (15, 25)]
    assert not clicker.wants_cashout()


def test_smart_cashout_target_stays_between_140_and_190(tmp_path):
    sent = []
    clicker = _clicker(tmp_path, sent, cashout_enabled=True)
    clicker.set_smart_play(("RED",), 0.9)
    clicker.fire("1", Decimal("1.05"))
    assert clicker._cashout_target == Decimal("1.40")
    clicker.set_smart_play(("RED",), 3.5)
    clicker.fire("2", Decimal("1.05"))
    assert clicker._cashout_target == Decimal("1.90")


def test_cashout_never_clicks_on_the_final_bang_screen(tmp_path):
    """خوانش دیرِ صفحهٔ پایان نباید روی دکمهٔ شرط کلیک کند (شرط تازه باز می‌شد)."""
    sent = []
    clicker = _clicker(tmp_path, sent, cashout_enabled=True)
    clicker.set_smart_play(("RED",), 1.55)
    clicker.fire("7", Decimal("2.10"))
    assert clicker.wants_cashout()
    clicker.tick(Decimal("1.42"))
    clicker.tick(Decimal("5.71"), ended=True)  # Bang @ 5.71
    assert sent == [(10, 20)]  # فقط کلیک ثبت شرط
    assert not clicker.wants_cashout()
    clicker.tick(Decimal("5.71"))
    clicker.tick(Decimal("5.90"))
    assert sent == [(10, 20)]


def test_cashout_ignores_a_late_bang_number_even_without_the_ended_flag(tmp_path):
    """جهش بزرگ بین دو نمونه عدد زنده نیست؛ پنجرهٔ خروج بسته می‌شود."""
    sent = []
    clicker = _clicker(tmp_path, sent, cashout_enabled=True)
    clicker.set_smart_play(("RED",), 1.50)
    clicker.fire("3", Decimal("1.05"))
    clicker.tick(Decimal("1.45"))
    clicker.tick(Decimal("5.71"))
    assert sent == [(10, 20)]
    assert not clicker.wants_cashout()
    clicker.tick(Decimal("1.02"))  # دست تازه شروع شده
    clicker.tick(Decimal("1.55"))
    assert sent == [(10, 20)]


def test_cashout_needs_a_rising_live_number_not_a_repeated_one(tmp_path):
    sent = []
    clicker = _clicker(tmp_path, sent, cashout_enabled=True)
    clicker.set_smart_play(("RED",), 1.50)
    clicker.fire("4", Decimal("1.05"))
    clicker.tick(Decimal("1.62"))  # اولین خوانش بالای هدف؛ هنوز بالا رفتن ثابت نشده
    assert sent == [(10, 20)]
    clicker.tick(Decimal("1.62"))  # عدد ثابت: ضریب زندهٔ در حال بالا رفتن نیست
    assert sent == [(10, 20)]
    clicker.tick(Decimal("1.70"))
    assert sent == [(10, 20), (10, 20)]
    assert not clicker.wants_cashout()
