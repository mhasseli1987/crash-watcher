from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
import json
import os
import numpy as np
import pytest
from crash_watcher import StrategyConfig, StrategyEngine
from crash_watcher.desktop.clicker import AutoClicker
from crash_watcher.desktop.config import DesktopSettings, Region, load_settings, save_settings
from crash_watcher.desktop.final_result import (BangDetector, BangRecognizer, FinalReading,
                                                 parse_bang, parse_history_entry)
from crash_watcher.desktop.forecast import ColorForecaster, Forecast
from crash_watcher.desktop.color_grid import read_grid
from crash_watcher.desktop.signals import (row_yellow_due, two_yellows_in_a_row,
                                            yellow_green_green_yellow)
from crash_watcher.vision.recognition import RapidOCRReader

@pytest.mark.parametrize("text,value", [("Bang @1.46x", "1.46"), ("Bang@7.50x", "7.50"),
                                        ("BANG @13.05×", "13.05"), ("Bang @1.07", "1.07")])
def test_bang_parser(text, value):
    assert parse_bang(text) == Decimal(value)

@pytest.mark.parametrize("text", ["Starts in 4.4 s", "1.27x", "Payout 10x", "1.8x", "Bang @1O.OOx", "Bang @1..46x"])
def test_bang_rejects_live_countdown_and_labels(text):
    assert parse_bang(text) is None

def test_history_entry_is_unambiguous():
    assert parse_history_entry("9342414 5.52x") == ("9342414", Decimal("5.52"))
    assert parse_history_entry("9356780 1 89x") == ("9356780", Decimal("1.89"))
    assert parse_history_entry("9356780 189x") == ("9356780", Decimal("1.89"))
    assert parse_history_entry("9342414 5.52x 9342413 2.02x") is None

def test_real_ocr_bang_and_countdown():
    import cv2
    recognizer = BangRecognizer(RapidOCRReader())
    for text, value in [("Bang @1.46x", Decimal("1.46")), ("Bang @7.50x", Decimal("7.50")),
                        ("Bang @13.05x", Decimal("13.05")), ("Starts in 4.4 s", None)]:
        image = np.zeros((120, 500, 3), np.uint8)
        cv2.putText(image, text, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        assert recognizer.recognize(image).value == value

def test_supplied_screenshots_do_not_create_false_final_results():
    import cv2
    root = Path(__file__).resolve().parents[1]
    recognizer = BangRecognizer(RapidOCRReader())
    live = cv2.imread(str(root / "sample.png"))[170:235, 270:460]
    countdown = cv2.imread(str(root / "screenshot2.jpg"))[300:365, 75:370]
    assert recognizer.recognize(live).value is None
    assert recognizer.recognize(countdown).value is None

def _reading(value):
    if value is None:
        return FinalReading(None, .0, "", None, True)
    value = Decimal(str(value))
    return FinalReading(value, .99, f"Bang @{value}x", "YELLOW" if value >= 10 else "RED", False)

def _image(shade):
    return np.full((40, 100, 3), shade, np.uint8)

def test_final_result_lifecycle_and_duplicate_guard():
    detector = BangDetector(stable_frames=2, absent_frames=2)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    # A final result already visible at startup must not be sent.
    for i in range(4):
        assert detector.observe(_image(30), _reading("10.00"), now+timedelta(seconds=i)) is None
    detector.observe(_image(0), _reading(None), now+timedelta(seconds=4))
    detector.observe(_image(0), _reading(None), now+timedelta(seconds=5))
    detector.observe(_image(30), _reading("10.00"), now+timedelta(seconds=6))
    assert detector.observe(_image(30), _reading("10.00"), now+timedelta(seconds=7)) is None
    result = detector.observe(_image(30), _reading("10.00"), now+timedelta(seconds=8))
    assert result and result.multiplier == Decimal("10.00")
    detector.observe(_image(30), _reading(None), now+timedelta(seconds=9))
    detector.observe(_image(30), _reading(None), now+timedelta(seconds=10))
    for i in range(11, 15):
        assert detector.observe(_image(30), _reading("10.00"), now+timedelta(seconds=i)) is None

def test_history_confirmation_and_strategy_handoff():
    detector = BangDetector(stable_frames=2, absent_frames=2)
    strategy = StrategyEngine(StrategyConfig())
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    detector.observe(_image(0), _reading(None), now)
    detector.observe(_image(0), _reading(None), now)
    detector.observe(_image(30), _reading("10.00"), now)
    detector.observe(_image(30), _reading("10.00"), now)
    assert detector.observe(_image(30), _reading("10.00"), now,
                            "9342414", Decimal("9.99"), True) is None
    confirmed = detector.observe(_image(30), _reading("10.00"), now,
                                 "9342414", Decimal("10.00"), True)
    assert confirmed and confirmed.round_id == "game-9342414"
    assert strategy.process_round(confirmed.round_id, confirmed.multiplier).cycle_triggered
    assert strategy.process_round(confirmed.round_id, confirmed.multiplier).duplicate

def test_history_can_confirm_after_bang_changes_to_countdown():
    detector = BangDetector(stable_frames=2, absent_frames=2)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    detector.observe(_image(0), _reading(None), now)
    detector.observe(_image(0), _reading(None), now)
    detector.observe(_image(30), _reading("2.89"), now, "9342414", Decimal("1.44"), True)
    detector.observe(_image(30), _reading("2.89"), now, "9342414", Decimal("1.44"), True)
    assert detector.observe(_image(30), _reading("2.89"), now,
                            "9342414", Decimal("1.44"), True) is None
    confirmed = detector.observe(_image(0), _reading(None), now,
                                 "9342415", Decimal("2.89"), True)
    assert confirmed and confirmed.round_id == "game-9342415"
    assert confirmed.multiplier == Decimal("2.89")

def test_settings_roundtrip_and_validation(tmp_path):
    path = tmp_path / "settings.json"
    settings = DesktopSettings(Region(1, 1, 2, 200, 100), Region(1, 5, 8, 80, 30),
                               fixed_bet_amount="12.5", max_attempts=10)
    save_settings(settings, path)
    assert load_settings(path) == settings
    assert load_settings(tmp_path / "absent.json") == DesktopSettings()
    with pytest.raises(ValueError):
        save_settings(DesktopSettings(fixed_bet_amount="-1"), path)

def test_gui_starts_idle_and_closes(monkeypatch):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PySide6.QtWidgets import QApplication
    from crash_watcher.desktop.ui import MainWindow
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    assert window.values["وضعیت پایش"].text() == "متوقف"
    assert window.worker is None
    assert window.start_button.isEnabled()
    window.close()

def test_worker_can_stop_and_restart():
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PySide6.QtWidgets import QApplication
    from crash_watcher.desktop.worker import MonitorWorker
    app = QApplication.instance() or QApplication([])
    settings = DesktopSettings(result_region=Region(999, 0, 0, 50, 50))
    for _ in range(2):
        worker = MonitorWorker(settings, StrategyEngine(StrategyConfig()))
        worker.start()
        worker.stop()
        assert worker.wait(5000)

class _Sender:
    def __init__(self):
        self.calls = []

    def __call__(self, x, y, restore=True):
        self.calls.append((x, y, restore))
        return True

@pytest.fixture
def no_detection(monkeypatch):
    """Never touch a real screen from the test suite."""
    monkeypatch.setattr(AutoClicker, "_detect", lambda self: None)

def _clicker(**overrides):
    values = dict(click_region=Region(1, 100, 200, 60, 40), autoclick_enabled=True,
                  autoclick_dry_run=False, autoclick_cooldown_ms=0, autoclick_delay_ms=0,
                  autoclick_skip_rounds=0, three_x_enabled=False)
    values.update(overrides)
    return DesktopSettings(**values)

def test_clicker_waits_for_a_high_round_then_clicks_the_region_centre(no_detection):
    sender = _Sender()
    clicker = AutoClicker(_clicker(), sender=sender)
    below = clicker.fire("low-1", Decimal("9.99"))
    assert not below.clicked and "waiting for a round" in below.reason and sender.calls == []
    hit = clicker.fire("high-1", Decimal("10.00"))
    assert hit.clicked and hit.reason == "clicked"
    assert hit.point == (130, 220) and sender.calls == [(130, 220, True)]

def test_yellow_waits_seven_rounds_then_clicks_eight_nine_ten(no_detection):
    sender = _Sender()
    clicker = AutoClicker(_clicker(autoclick_skip_rounds=7, autoclick_max_clicks=3,
                                   streak_signal_enabled=False), sender=sender)
    armed = clicker.fire("y", Decimal("10.40"))
    assert not armed.clicked and "waiting 7" in armed.reason and armed.remaining == 3
    for index in range(7):
        skipped = clicker.fire(f"w{index}", Decimal("1.50"))
        assert not skipped.clicked
    burst = [clicker.fire(f"c{index}", Decimal("1.10")) for index in range(3)]
    assert all(report.clicked for report in burst)
    assert [report.remaining for report in burst] == [2, 1, 0]
    after = clicker.fire("after", Decimal("8.00"))
    assert not after.clicked and after.remaining == 0
    assert len(sender.calls) == 3

def test_yellow_inside_the_wait_does_not_restart_the_run(no_detection):
    sender = _Sender()
    clicker = AutoClicker(_clicker(autoclick_skip_rounds=7, autoclick_max_clicks=3), sender=sender)
    clicker.fire("y", Decimal("19.00"))
    for index in range(3):
        assert not clicker.fire(f"w{index}", Decimal("1.50")).clicked
    during = clicker.fire("w-yellow", Decimal("12.00"))
    assert not during.clicked and during.remaining == 3 and clicker.waiting == 3
    for index in range(3):
        assert not clicker.fire(f"w2{index}", Decimal("1.50")).clicked
    assert clicker.fire("c0", Decimal("1.50")).clicked
    assert len(sender.calls) == 1

def test_high_round_inside_the_run_does_not_reset_the_count(no_detection):
    sender = _Sender()
    clicker = AutoClicker(_clicker(autoclick_max_clicks=3), sender=sender)
    assert clicker.fire("a", Decimal("10.00")).remaining == 2
    inside = clicker.fire("b", Decimal("25.00"))
    assert inside.clicked and inside.remaining == 1
    assert clicker.fire("c", Decimal("1.50")).remaining == 0
    assert not clicker.fire("d", Decimal("1.50")).clicked
    assert len(sender.calls) == 3
    # Only once the run is over does a new high round start a fresh one.
    assert clicker.fire("e", Decimal("10.00")).remaining == 2

def test_no_click_waits_longer_than_one_second(no_detection, monkeypatch):
    slept = []
    monkeypatch.setattr("crash_watcher.desktop.clicker.sleep", lambda seconds: slept.append(seconds))
    with pytest.raises(ValueError):
        _clicker(autoclick_delay_ms=1001).validate()
    clicker = AutoClicker(_clicker(autoclick_delay_ms=1000), sender=_Sender())
    assert clicker.fire("r", Decimal("12.00")).clicked
    assert clicker.test().clicked
    assert slept == [1.0, 1.0]


def test_clicker_waits_before_clicking(no_detection, monkeypatch):
    slept = []
    monkeypatch.setattr("crash_watcher.desktop.clicker.sleep", lambda seconds: slept.append(seconds))
    sender = _Sender()
    clicker = AutoClicker(_clicker(autoclick_delay_ms=1000), sender=sender)
    assert clicker.fire("r1", Decimal("12.00")).clicked
    assert slept == [1.0] and len(sender.calls) == 1
    slept.clear()
    AutoClicker(_clicker(autoclick_delay_ms=0), sender=sender).fire("r2", Decimal("12.00"))
    assert slept == []

def test_clicker_clicks_once_per_round_and_respects_cooldown(no_detection):
    sender = _Sender()
    clicker = AutoClicker(_clicker(), sender=sender)
    assert clicker.fire("r1", Decimal("12.00")).clicked
    again = clicker.fire("r1", Decimal("12.00"))
    assert not again.clicked and again.reason == "same round"
    assert clicker.fire("r2", Decimal("11.00")).clicked
    assert len(sender.calls) == 2
    slow = AutoClicker(_clicker(autoclick_cooldown_ms=60000), sender=sender)
    assert slow.fire("a", Decimal("10.00")).clicked
    blocked = slow.fire("b", Decimal("10.00"))
    assert not blocked.clicked and blocked.reason == "cooldown" and len(sender.calls) == 3

def test_clicker_clicks_in_dry_run_and_never_when_disabled(no_detection):
    sender = _Sender()
    dry = AutoClicker(_clicker(autoclick_dry_run=True), sender=sender)
    report = dry.fire("r1", Decimal("10.00"))
    assert report.clicked and report.reason == "clicked"
    assert sender.calls == [(130, 220, True)]
    off = AutoClicker(DesktopSettings(autoclick_enabled=False), sender=sender)
    assert off.fire("r1", Decimal("99.00")).reason == "disabled"
    assert len(sender.calls) == 1



def test_history_timeout_confirms_a_stable_bang():
    detector = BangDetector(stable_frames=2, absent_frames=2, history_timeout_frames=2)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    detector.observe(_image(0), _reading(None), now)
    detector.observe(_image(0), _reading(None), now)
    detector.observe(_image(30), _reading("12.00"), now, "111", Decimal("1.00"), True)
    detector.observe(_image(30), _reading("12.00"), now, "111", Decimal("1.00"), True)
    assert detector.observe(_image(30), _reading("12.00"), now,
                            "111", Decimal("1.00"), True) is None
    confirmed = detector.observe(_image(30), _reading("12.00"), now,
                                 "111", Decimal("1.00"), True)
    assert confirmed and confirmed.multiplier == Decimal("12.00")
    assert confirmed.round_id.startswith("visual-")

def test_confirmed_round_reaches_click_through_detector_and_strategy(no_detection):
    """Same sequence MonitorWorker.run() performs per frame: detect, strategy, click."""
    detector = BangDetector(stable_frames=2, absent_frames=2)
    strategy = StrategyEngine(StrategyConfig())
    sender = _Sender()
    clicker = AutoClicker(_clicker(), sender=sender)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    reports = []

    def feed(shade, reading, second):
        completed = detector.observe(_image(shade), reading, now + timedelta(seconds=second))
        result = strategy.process_round(completed.round_id, completed.multiplier) if completed else None
        if completed:
            reports.append(clicker.fire(completed.round_id, completed.multiplier))
        return result

    for second in (0, 1):
        assert feed(0, _reading(None), second) is None
    trigger = [feed(30, _reading("12.00"), second) for second in (2, 3, 4)][-1]
    assert trigger is not None and trigger.cycle_triggered
    assert len(reports) == 1 and reports[0].clicked and reports[0].point == (130, 220)
    assert sender.calls == [(130, 220, True)]

    for second in (5, 6):
        assert feed(0, _reading(None), second) is None
    bet = [feed(40, _reading("15.00"), second) for second in (7, 8, 9)][-1]
    assert bet is not None and bet.bet_registered
    assert len(reports) == 2 and sender.calls == [(130, 220, True), (130, 220, True)]

_DOT_BGR = {"GREEN": (113, 204, 46), "RED": (34, 126, 230), "YELLOW": (15, 196, 241)}

def _grid_image(columns):
    """Draw the site's layout: column-major, newest at the top-left cell."""
    import cv2
    image = np.full((200, 260, 3), 32, np.uint8)
    pitch = 30
    for column, colors in enumerate(columns):
        for row, color in enumerate(colors):
            cv2.circle(image, (20 + column * pitch, 20 + row * pitch), 8, _DOT_BGR[color], -1)
    return image

def test_grid_reader_returns_the_sites_column_major_order():
    reading = read_grid(_grid_image([["GREEN", "GREEN", "GREEN"], ["RED"], ["GREEN", "YELLOW"]]))
    assert reading is not None
    assert reading.sequence == ("GREEN", "GREEN", "GREEN", "RED", "GREEN", "YELLOW")
    assert reading.columns == 3 and reading.rows == 3
    assert reading.color_at(0, 0) == "GREEN" and reading.color_at(2, 1) == "YELLOW"
    assert reading.color_at(1, 1) is None

def test_two_yellows_in_a_row_signal():
    reading = read_grid(_grid_image([["GREEN"], ["YELLOW", "YELLOW"], ["RED"]]))
    assert reading is not None and two_yellows_in_a_row(reading)
    separated = read_grid(_grid_image([["YELLOW"], ["GREEN"], ["YELLOW"]]))
    assert separated is not None and not two_yellows_in_a_row(separated)
    assert two_yellows_in_a_row(None) is False

def test_signal_clicks_after_each_yellow_up_to_the_repeat_limit(no_detection, monkeypatch):
    sender = _Sender()
    clicker = AutoClicker(_clicker(autoclick_threshold="100", signal_repeat=2,
                                   signal_rule_enabled=True), sender=sender)
    monkeypatch.setattr(AutoClicker, "_chart_reading", lambda self: object())
    monkeypatch.setattr("crash_watcher.desktop.clicker.two_yellows_in_a_row", lambda reading: True)
    monkeypatch.setattr("crash_watcher.desktop.clicker.yellow_green_green_yellow", lambda reading: False)
    monkeypatch.setattr("crash_watcher.desktop.clicker.row_yellow_due",
                        lambda reading, gap=4, rows=4, recent_columns=4: False)
    assert clicker.fire("a", Decimal("12.00")).clicked
    assert clicker.fire("b", Decimal("11.00")).clicked
    third = clicker.fire("c", Decimal("10.50"))
    assert not third.clicked
    assert len(sender.calls) == 2

def test_signal_needs_a_rising_edge_to_arm_again(no_detection, monkeypatch):
    sender = _Sender()
    clicker = AutoClicker(_clicker(autoclick_threshold="100", signal_repeat=1,
                                   signal_rule_enabled=True), sender=sender)
    pattern = {"value": False}
    monkeypatch.setattr(AutoClicker, "_chart_reading", lambda self: object())
    monkeypatch.setattr("crash_watcher.desktop.clicker.yellow_green_green_yellow", lambda reading: False)
    monkeypatch.setattr("crash_watcher.desktop.clicker.row_yellow_due",
                        lambda reading, gap=4, rows=4, recent_columns=4: False)
    monkeypatch.setattr("crash_watcher.desktop.clicker.two_yellows_in_a_row",
                        lambda reading: pattern["value"])
    assert not clicker.fire("a", Decimal("12.00")).clicked
    pattern["value"] = True
    assert clicker.fire("b", Decimal("12.00")).clicked
    pattern["value"] = False
    assert not clicker.fire("c", Decimal("12.00")).clicked
    pattern["value"] = True
    assert clicker.fire("d", Decimal("12.00")).clicked


def test_yellow_green_green_yellow_shape():
    reading = read_grid(_grid_image([["YELLOW", "GREEN", "GREEN", "YELLOW"], ["RED"]]))
    assert reading is not None and yellow_green_green_yellow(reading)
    wrong = read_grid(_grid_image([["YELLOW", "GREEN", "RED", "YELLOW"]]))
    assert wrong is not None and not yellow_green_green_yellow(wrong)
    assert yellow_green_green_yellow(None) is False

def test_signal_burst_clicks_the_next_nine_rounds(no_detection, monkeypatch):
    sender = _Sender()
    clicker = AutoClicker(_clicker(autoclick_threshold="100", streak_signal_enabled=False,
                                   signal_rule_enabled=True), sender=sender)
    monkeypatch.setattr(AutoClicker, "_chart_reading", lambda self: object())
    monkeypatch.setattr("crash_watcher.desktop.clicker.two_yellows_in_a_row", lambda reading: False)
    monkeypatch.setattr("crash_watcher.desktop.clicker.yellow_green_green_yellow", lambda reading: True)
    monkeypatch.setattr("crash_watcher.desktop.clicker.row_yellow_due",
                        lambda reading, gap=4, rows=4, recent_columns=4: False)
    assert not clicker.fire("a", Decimal("1.20")).clicked
    for index in range(9):
        assert clicker.fire(f"c{index}", Decimal("1.10")).clicked
    assert not clicker.fire("after", Decimal("1.30")).clicked
    assert len(sender.calls) == 9

def test_pair_flip_clicks_the_opposite_color(no_detection):
    sender = _Sender()
    clicker = AutoClicker(_clicker(autoclick_threshold="100", bet_point_red=(11, 12),
                                   bet_point_green=(21, 22), pair_flip_enabled=True),
                          sender=sender)
    assert not clicker.fire("a", Decimal("5.00")).clicked
    assert not clicker.fire("b", Decimal("6.00")).clicked
    flip = clicker.fire("c", Decimal("1.20"))
    assert flip.clicked and flip.target == "RED" and flip.point == (11, 12)
    assert not clicker.fire("d", Decimal("1.10")).clicked
    assert len(sender.calls) == 1

def test_pair_flip_after_two_reds_targets_green(no_detection):
    sender = _Sender()
    clicker = AutoClicker(_clicker(autoclick_threshold="100", bet_point_green=(31, 32),
                                   pair_flip_enabled=True), sender=sender)
    assert not clicker.fire("a", Decimal("1.10")).clicked
    assert not clicker.fire("b", Decimal("1.20")).clicked
    flip = clicker.fire("c", Decimal("2.50"))
    assert flip.clicked and flip.target == "GREEN" and flip.point == (31, 32)

def test_smart_click_targets_the_predicted_color(no_detection):
    sender = _Sender()
    clicker = AutoClicker(_clicker(autoclick_threshold="100", smart_click_enabled=True,
                                   bet_point_green=(41, 42)), sender=sender)
    clicker.forecaster = _FixedForecast(probability=0.70)
    report = clicker.fire("r", Decimal("1.10"))
    assert report.clicked and report.target == "GREEN" and report.point == (41, 42)

def test_row_signal_is_due_when_the_next_row_was_yellow_four_columns_back():
    # The newest column already holds rows 0-2, so the next dot fills row 3 and
    # that same row was yellow four columns back.
    reading = read_grid(_grid_image([["GREEN", "GREEN", "GREEN"], ["RED"], ["RED"], ["RED"],
                                     ["RED", "RED", "RED", "YELLOW"]]))
    assert reading is not None and row_yellow_due(reading)
    # A yellow in another row of that column says nothing about the row being filled.
    other_row = read_grid(_grid_image([["GREEN", "GREEN", "GREEN"], ["RED"], ["RED"], ["RED"],
                                       ["RED", "RED", "YELLOW", "RED"]]))
    assert other_row is not None and not row_yellow_due(other_row)
    # Below the fourth row the user considers the odds too low.
    deep = read_grid(_grid_image([["GREEN", "GREEN", "GREEN", "GREEN"], ["RED"], ["RED"],
                                  ["RED"],
                                  ["RED", "RED", "RED", "RED", "YELLOW"]]))
    assert deep is not None and not row_yellow_due(deep)
    # Four columns of history is the least the rule can look back at.
    short = read_grid(_grid_image([["GREEN"], ["RED"], ["RED"], ["YELLOW"]]))
    assert short is not None and not row_yellow_due(short)
    assert row_yellow_due(None) is False

def test_row_signal_clicks_yellow_on_the_round_that_fills_the_row(no_detection, monkeypatch):
    sender = _Sender()
    clicker = AutoClicker(_clicker(autoclick_threshold="100", pair_flip_enabled=False,
                                   row_signal_enabled=True,
                                   bet_point_yellow=(51, 52)), sender=sender)
    monkeypatch.setattr(AutoClicker, "_chart_reading", lambda self: object())
    monkeypatch.setattr("crash_watcher.desktop.clicker.two_yellows_in_a_row", lambda reading: False)
    monkeypatch.setattr("crash_watcher.desktop.clicker.yellow_green_green_yellow", lambda reading: False)
    due = {"value": True}
    monkeypatch.setattr("crash_watcher.desktop.clicker.row_yellow_due",
                        lambda reading, gap=4, rows=4, recent_columns=4: due["value"])
    # The repeat is due on this very round, so the bet happens now, not one round later.
    clicked = clicker.fire("a", Decimal("1.10"))
    assert clicked.clicked and clicked.target == "YELLOW" and clicked.point == (51, 52)
    assert clicked.rule == "row-signal"
    due["value"] = False
    assert not clicker.fire("b", Decimal("1.20")).clicked

def test_seven_greens_in_a_row_bet_green_for_ten_rounds(no_detection):
    sender = _Sender()
    clicker = AutoClicker(_clicker(autoclick_threshold="100", pair_flip_enabled=False,
                                   streak_signal_enabled=True,
                                   bet_point_green=(91, 92), bet_point_red=(81, 82)),
                          sender=sender)
    # Six in a row are not enough for the user's streak rule.
    for index in range(6):
        assert not clicker.fire(f"g{index}", Decimal("3.00")).clicked
    # The seventh green arms the run, and that same round already bets green.
    report = clicker.fire("g6", Decimal("3.10"))
    assert report.clicked and report.rule == "streak-run" and report.target == "GREEN"
    for index in range(9):
        assert clicker.fire(f"s{index}", Decimal("1.10")).clicked
    # A yellow breaks the streak and the ten rounds are used up.
    assert not clicker.fire("after", Decimal("12.00")).clicked
    assert len(sender.calls) == 10 and all(call[0] == 91 for call in sender.calls)

def test_seven_reds_in_a_row_bet_red_for_ten_rounds(no_detection):
    sender = _Sender()
    clicker = AutoClicker(_clicker(autoclick_threshold="100", pair_flip_enabled=False,
                                   streak_signal_enabled=True,
                                   bet_point_green=(91, 92), bet_point_red=(81, 82)),
                          sender=sender)
    for index in range(7):
        clicker.fire(f"r{index}", Decimal("1.20"))
    assert clicker._streak_target == "RED"
    for index in range(9):
        assert clicker.fire(f"s{index}", Decimal("5.00")).clicked
    assert len(sender.calls) == 10 and all(call[0] == 81 for call in sender.calls)

@pytest.mark.parametrize("value,color,point", [("1.00", "GREEN", (61, 62)),
                                              ("1.11", "GREEN", (61, 62)),
                                              ("1.99", "GREEN", (61, 62)),
                                              ("2.08", "YELLOW", (71, 72))])
def test_special_multiplier_bets_the_named_color(value, color, point, no_detection):
    sender = _Sender()
    clicker = AutoClicker(_clicker(autoclick_threshold="100", pair_flip_enabled=False,
                                   value_bet_enabled=True,
                                   bet_point_green=(61, 62), bet_point_yellow=(71, 72)),
                          sender=sender)
    report = clicker.fire(f"r{value}", Decimal(value))
    assert report.clicked and report.target == color and report.point == point

def test_unlisted_multiplier_does_not_click(no_detection):
    sender = _Sender()
    clicker = AutoClicker(_clicker(autoclick_threshold="100", pair_flip_enabled=False),
                          sender=sender)
    assert not clicker.fire("r", Decimal("3.33")).clicked
    assert sender.calls == []

def test_bet_region_splits_its_row_into_three_buttons(no_detection):
    clicker = AutoClicker(_clicker(bet_region=Region(1, 100, 200, 500, 40), bet_point_red=None,
                                   bet_point_green=None, bet_point_yellow=None), sender=_Sender())
    assert clicker._bet_point("RED") == (183, 220)
    assert clicker._bet_point("GREEN") == (350, 220)
    assert clicker._bet_point("YELLOW") == (516, 220)

def test_session_length_and_shutdown_flag_are_validated():
    _clicker(session_minutes=0).validate()
    _clicker(session_minutes=360).validate()
    for bad in (-1, 1441):
        with pytest.raises(ValueError):
            _clicker(session_minutes=bad).validate()
    with pytest.raises(ValueError):
        _clicker(shutdown_after_session="yes").validate()

def test_session_settings_round_trip(tmp_path):
    path = tmp_path / "settings.json"
    save_settings(DesktopSettings(session_minutes=180, shutdown_after_session=False), path)
    loaded = load_settings(path)
    assert loaded.session_minutes == 180 and loaded.shutdown_after_session is False

def test_an_old_timer_and_shutdown_default_is_switched_off_on_load(tmp_path):
    path = tmp_path / "settings.json"
    save_settings(DesktopSettings(session_minutes=360, shutdown_after_session=True), path)
    loaded = load_settings(path)
    assert loaded.session_minutes == 0 and loaded.shutdown_after_session is False

def test_settings_file_with_removed_keys_still_loads(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"fixed_bet_amount": "12",
                                "pair_cycle_enabled": True,
                                "double_bet_enabled": False}), encoding="utf-8")
    loaded = load_settings(path)
    assert loaded.fixed_bet_amount == "12"
    # A file from before the 3x engine must not revive the old shape rules.
    assert loaded.signal_rule_enabled is False and loaded.streak_signal_enabled is False
    assert loaded.three_x_enabled is True

def test_live_multiplier_reads_the_running_number_only():
    """خروج کلاسیک باید فقط عدد زنده را بفهمد، نه پیام پایان دور یا عدد نامطمئن."""
    from crash_watcher.desktop.worker import live_multiplier
    image = np.zeros((40, 160, 3), dtype=np.uint8)
    assert live_multiplier(lambda _: ("1.73x", 0.95), image) == Decimal("1.73")
    assert live_multiplier(lambda _: ("Bang @1.73x", 0.95), image) is None
    assert live_multiplier(lambda _: ("1.73x", 0.30), image) is None


def test_session_seconds_left_counts_down_and_rounds_safely():
    from crash_watcher.desktop.worker import session_seconds_left
    assert session_seconds_left(1000.0, 0, now=1000.0) is None
    assert session_seconds_left(1000.0, 30, now=1000.0) == 30 * 60
    assert session_seconds_left(1000.0, 30, now=1000.0 + 600) == 20 * 60
    assert session_seconds_left(1000.0, 30, now=1000.0 + 5000) == 0
    assert session_seconds_left(1000.0, -5, now=1000.0) is None

def test_value_bets_reject_bad_entries():
    with pytest.raises(ValueError):
        _clicker(value_bets={"1.00": "BLUE"}).validate()
    with pytest.raises(ValueError):
        _clicker(value_bets={"abc": "GREEN"}).validate()

def test_clicks_are_attributed_to_a_rule_and_the_report_counts_hits(tmp_path, no_detection):
    sender = _Sender()
    clicker = AutoClicker(_clicker(autoclick_threshold="100", pair_flip_enabled=False,
                                   value_bet_enabled=True,
                                   bet_point_green=(61, 62)), sender=sender,
                          log_path=tmp_path / "rounds.jsonl",
                          summary_path=tmp_path / "summary.txt",
                          report_path=tmp_path / "report.txt")
    click = clicker.fire("a", Decimal("1.00"))
    assert click.clicked and click.rule == "value-bet" and click.target == "GREEN"
    clicker.fire("b", Decimal("3.00"))
    clicker.close()
    report = (tmp_path / "report.txt").read_text(encoding="utf-8")
    assert "ضریب ویژه: 1 کلیک، 1 درست (100%)" in report
    first = json.loads((tmp_path / "rounds.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["rule"] == "value-bet" and first["target"] == "GREEN"

def test_grid_reader_ignores_an_empty_region():
    assert read_grid(np.full((80, 120, 3), 32, np.uint8)) is None
    assert read_grid(None) is None

class _FixedForecast:
    """Deterministic stand-in for ColorForecaster in click-gate tests."""
    min_samples = 40

    def __init__(self, color="GREEN", probability=0.70):
        self.color = color
        self.probability = probability
        self.seen = []

    def observe(self, color):
        self.seen.append(color)

    def forecast(self):
        return Forecast(self.color, self.probability, 100, "base rate")

    def review(self):
        return {"samples": len(self.seen), "base_accuracy": 0.5, "cond_accuracy": 0.4,
                "base_scored": 1, "cond_scored": 1, "model": "base rate"}

def test_forecaster_tracks_the_following_color_when_it_is_real():
    forecaster = ColorForecaster(min_samples=10)
    for _ in range(40):
        forecaster.observe("RED")
        forecaster.observe("GREEN")
    assert forecaster.conditional_is_trusted()
    prediction = forecaster.forecast()
    assert prediction.samples == 80 and 0.5 <= sum(forecaster.probabilities().values()) <= 1.0001
    assert prediction.color == "RED" and prediction.probability >= 0.6

def test_forecaster_rejects_unknown_colors():
    with pytest.raises(ValueError):
        ColorForecaster().observe("BLUE")

def test_smart_click_skips_low_confidence_and_clicks_high(no_detection):
    sender = _Sender()
    clicker = AutoClicker(_clicker(smart_click_enabled=True), sender=sender)
    clicker.forecaster = _FixedForecast(probability=0.55)
    skipped = clicker.fire("r1", Decimal("1.20"))
    assert not skipped.clicked and skipped.reason.startswith("skip")
    assert sender.calls == [] and skipped.forecast.startswith("GREEN")
    clicker.forecaster = _FixedForecast(probability=0.70)
    clicked = clicker.fire("r2", Decimal("1.30"))
    assert clicked.clicked and len(sender.calls) == 1
    assert clicked.remaining == 0

def test_high_round_still_clicks_inside_smart_mode(no_detection):
    sender = _Sender()
    clicker = AutoClicker(_clicker(smart_click_enabled=True, autoclick_max_clicks=2), sender=sender)
    clicker.forecaster = _FixedForecast(probability=0.10)
    assert clicker.fire("a", Decimal("12.00")).clicked
    assert clicker.fire("b", Decimal("1.10")).clicked
    assert not clicker.fire("c", Decimal("1.10")).clicked
    assert len(sender.calls) == 2

def test_review_window_defaults_to_thirty_minutes():
    assert DesktopSettings().review_window_s == 1800

def test_every_round_is_logged_and_the_review_writes_a_summary(tmp_path, no_detection, monkeypatch):
    clock = {"now": 1000.0}
    monkeypatch.setattr("crash_watcher.desktop.clicker.monotonic", lambda: clock["now"])
    sender = _Sender()
    clicker = AutoClicker(_clicker(smart_click_enabled=True, review_window_s=60), sender=sender,
                          log_path=tmp_path / "rounds.jsonl", summary_path=tmp_path / "summary.txt")
    clicker.forecaster = _FixedForecast(probability=0.70)
    clicker.fire("r1", Decimal("12.00"))
    clicker.fire("r2", Decimal("1.50"))
    lines = (tmp_path / "rounds.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first, second = (json.loads(line) for line in lines)
    assert first["color"] == "YELLOW" and first["clicked"] is True and first["round"] == "r1"
    assert second["color"] == "RED"
    clock["now"] += 61
    review = clicker.take_review()
    assert review and review["rounds"] == 2 and review["minutes"] >= 1
    summary = (tmp_path / "summary.txt").read_text(encoding="utf-8")
    assert "بازبینی" in summary and "تصمیم خودکار" in summary

def test_review_window_reports_after_ten_minutes(no_detection, monkeypatch):
    clock = {"now": 1000.0}
    monkeypatch.setattr("crash_watcher.desktop.clicker.monotonic", lambda: clock["now"])
    sender = _Sender()
    clicker = AutoClicker(_clicker(smart_click_enabled=True, autoclick_max_clicks=1,
                                   review_window_s=600), sender=sender)
    clicker.forecaster = _FixedForecast(probability=0.70)
    assert clicker.take_review() is None
    clicker.fire("a", Decimal("12.00"))
    clock["now"] += 599
    assert clicker.take_review() is None
    clock["now"] += 2
    review = clicker.take_review()
    assert review and review["rounds"] == 1 and review["clicks"] == 1
    assert review["minutes"] >= 10 and review["gated_total"] == 0
    assert clicker.take_review() is None


def test_clicker_without_target_sends_nothing(no_detection):
    sender = _Sender()
    report = AutoClicker(DesktopSettings(autoclick_enabled=True, autoclick_dry_run=False,
                                         autoclick_cooldown_ms=0, autoclick_skip_rounds=0,
                                         three_x_enabled=False),
                         sender=sender).fire("r1", Decimal("10.00"))
    assert not report.clicked and report.reason == "target not found" and sender.calls == []


def test_test_click_uses_the_bet_region_when_no_click_area_is_saved(no_detection):
    sender = _Sender()
    clicker = AutoClicker(DesktopSettings(autoclick_enabled=True, autoclick_delay_ms=0,
                                          bet_region=Region(1, 369, 811, 640, 47)),
                          sender=sender)
    report = clicker.test()
    assert report.clicked and report.reason == "clicked"
    assert report.point == (689, 834) and sender.calls == [(689, 834, True)]


def test_test_click_uses_the_yellow_bet_button_as_the_last_fallback(no_detection):
    sender = _Sender()
    clicker = AutoClicker(DesktopSettings(autoclick_enabled=True, autoclick_delay_ms=0,
                                          bet_point_yellow=(902, 834)), sender=sender)
    assert clicker.test().point == (902, 834)
    assert sender.calls == [(902, 834, True)]
