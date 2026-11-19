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
    carry = SmartPlanner(learning_minutes=0)
    carry.store(SmartVerdict("GREEN", 0.7, "x", exit_at=1.75), "9")
    assert carry.decision().exit_at == 1.75  # عدد خروج مدل تا لحظهٔ کلیک حفظ می‌شود


def test_hedge_hit_counts_as_correct_and_weak_accuracy_stops_betting():
    planner = SmartPlanner(learning_minutes=0, min_confidence=0.40)
    planner.store(SmartVerdict("YELLOW", 0.5, "شک", second="RED"), "1")
    planner.observe("2", "RED")  # رنگ دوم زد: سربه‌سر، نه باخت
    assert planner.hits == 1 and planner.break_even == 1
    planner.store(SmartVerdict("GREEN", 0.5, "x"), "2")
    planner.observe("3", "YELLOW")
    assert planner.hits == 1 and planner.tries == 2
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
    base = dict(autoclick_enabled=True, autoclick_delay_ms=0, autoclick_cooldown_ms=0,
                three_x_enabled=False,
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


def test_ai_mode_ignores_the_old_rules(tmp_path):
    """در AI Analysis فقط مدل تصمیم می‌گیرد؛ زرد نباید صف ۷+۳ را مسلح کند."""
    sent = []
    clicker = _clicker(tmp_path, sent)
    clicker.set_smart_play(("YELLOW",), active=True)
    report = clicker.fire("1", Decimal("12.00"))
    assert report.clicked and report.rule == "smart-play"
    assert report.remaining == 0
    assert sent == [(30, 40)]


def test_ai_mode_skips_instead_of_falling_back_to_a_rule(tmp_path):
    sent = []
    clicker = _clicker(tmp_path, sent)
    clicker.set_smart_play((), active=True)
    report = clicker.fire("1", Decimal("12.00"))
    assert not report.clicked and "smart-play" in report.reason
    assert sent == []


def test_ai_mode_takes_priority_over_the_three_x_engine(tmp_path):
    sent = []
    clicker = _clicker(tmp_path, sent, three_x_enabled=True)
    clicker.set_smart_play(("GREEN",), active=True)
    report = clicker.fire("5", Decimal("3.40"))
    assert report.clicked and report.rule == "smart-play"
    assert sent == [(15, 25)]


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


def test_classic_cashout_enters_once_and_exits_below_199(tmp_path):
    """کلیک کلاسیک: یک کلیک ورود، یک کلیک خروج روی ضریب مدل (زیر ۱٫۹۹)."""
    sent = []
    clicker = _clicker(tmp_path, sent, smart_cashout_enabled=True,
                       classic_click_point=(50, 60))
    clicker.set_smart_play(("GREEN",), active=True, cashout=1.85)
    report = clicker.fire("10", Decimal("1.20"))
    assert report.clicked and report.rule == "smart-play"
    assert "ورود کلاسیک" in report.detail
    assert sent == [(50, 60)]  # فقط یک کلیک ورود، بدون دکمهٔ رنگ
    assert clicker.wants_live() and not clicker.live_due()
    assert clicker.watch_live(Decimal("1.02")) is None  # عدد قبلی که هنوز روی صفحه است
    assert clicker.watch_live(Decimal("1.03")) is None  # شروع دست تأیید شد
    exit_report = clicker.watch_live(Decimal("1.86"))
    assert exit_report is not None and exit_report.clicked
    assert sent == [(50, 60), (50, 60)]
    assert clicker.watch_live(Decimal("1.90")) is None  # فقط یک بار خروج


def test_classic_cashout_never_clicks_at_or_above_199(tmp_path):
    sent = []
    clicker = _clicker(tmp_path, sent, smart_cashout_enabled=True,
                       classic_click_point=(50, 60))
    clicker.set_smart_play(("RED",), active=True, cashout=1.85)
    clicker.fire("1", Decimal("1.10"))
    clicker.watch_live(Decimal("1.02"))
    clicker.watch_live(Decimal("1.02"))
    assert clicker.watch_live(Decimal("1.99")) is None  # قاعده: هرگز ۱٫۹۹ یا بیشتر
    assert clicker.watch_live(Decimal("2.20")) is None
    assert sent == [(50, 60)]


def test_eight_classic_losses_ask_for_a_relearn(tmp_path):
    """هشت باخت متوالی باید درخواست «دوباره نیم ساعت تحلیل» بدهد."""
    sent = []
    clicker = _clicker(tmp_path, sent, smart_cashout_enabled=True,
                       classic_click_point=(50, 60))
    for index in range(8):
        clicker.set_smart_play(("GREEN",), active=True, cashout=1.50)
        clicker.fire(str(index), Decimal("1.10"))
        assert clicker.take_relearn_request() is False
    clicker.set_smart_play((), active=True)
    report = clicker.fire("8", Decimal("1.10"))
    assert not report.clicked
    assert clicker.take_relearn_request() is True
    assert clicker.take_relearn_request() is False


def test_restart_learning_starts_thirty_minutes_again():
    clock = _Clock()
    planner = SmartPlanner(learning_minutes=0, clock=clock)
    planner.restart_learning()
    assert planner.phase == LEARNING
    clock.now = 29 * 60
    assert planner.phase == LEARNING
    clock.now = 30 * 60
    assert planner.phase == TRADING


def test_classic_cashout_settings_are_validated():
    DesktopSettings(smart_cashout_enabled=True, classic_click_point=(12, 34)).validate()
    for bad in ({"classic_click_point": (1,)}, {"classic_click_point": ("a", 2)},
                {"smart_cashout_enabled": "yes"}):
        with pytest.raises(ValueError):
            DesktopSettings(**bad).validate()


def test_smart_play_settings_are_validated():
    DesktopSettings(smart_play_enabled=True, smart_play_learning_minutes=0).validate()
    for bad in ({"smart_play_learning_minutes": -1}, {"smart_play_learning_minutes": 900},
                {"smart_play_min_confidence": 0.1}, {"smart_play_min_confidence": True},
                {"smart_play_enabled": "yes"}):
        with pytest.raises(ValueError):
            DesktopSettings(**bad).validate()
