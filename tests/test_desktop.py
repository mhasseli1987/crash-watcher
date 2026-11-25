from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
import json
import os
import numpy as np
import pytest
from crash_watcher import StrategyConfig, StrategyEngine
from crash_watcher.desktop.clicker import AutoClicker
from crash_watcher.desktop.bankroll import Bankroll
from crash_watcher.desktop.config import DesktopSettings, Region, load_settings, save_settings
from crash_watcher.desktop.final_result import (BangDetector, BangRecognizer, FinalReading,
                                                 parse_bang, parse_history_entry)
from crash_watcher.desktop.model_compare import (ModelAttempt, ModelLeaderboard,
                                                  compare as compare_models, parse_models)
from crash_watcher.desktop.preflight import run_preflight
from crash_watcher.desktop.secrets_store import is_protected, protect, unprotect
from crash_watcher.desktop.smart_play import SmartPlanner
from crash_watcher.desktop.smart_review import SmartSnapshot, SmartVerdict
from crash_watcher.desktop.turnbal import (ColorRule, TurnbalCalibration, TurnbalGuard,
                                            calibrate, build_prompt as turnbal_prompt)
from types import SimpleNamespace
from crash_watcher.desktop.forecast import ColorForecaster, Forecast
from crash_watcher.desktop.color_grid import read_grid
from crash_watcher.desktop.signals import row_yellow_due
from crash_watcher.vision.recognition import RapidOCRReader

@pytest.mark.parametrize("text,value", [("Bang @1.46x", "1.46"), ("Bang@7.50x", "7.50"),
                                        ("BANG @13.05×", "13.05"), ("Bang @1.07", "1.07"),
                                        ("Bang @2.87=", "2.87"), ("Bang @1.75:", "1.75"),
                                        ("Bang @4.20*", "4.20")])
def test_bang_parser(text, value):
    assert parse_bang(text) == Decimal(value)

@pytest.mark.parametrize("text", ["Starts in 4.4 s", "1.27x", "Payout 10x", "1.8x", "Bang @1O.OOx",
                                  "Bang @1..46x", "Bang @2.87==", "Bang @2.8=",
                                  "Bang @2.870", "Bang @@2.87x"])
def test_bang_rejects_live_countdown_and_labels(text):
    assert parse_bang(text) is None

def test_history_entry_is_unambiguous():
    assert parse_history_entry("9342414 5.52x") == ("9342414", Decimal("5.52"))
    assert parse_history_entry("9356780 1 89x") == ("9356780", Decimal("1.89"))
    assert parse_history_entry("9356780 189x") == ("9356780", Decimal("1.89"))
    # نوار تاریخچه چند نشان را در یک خط می‌دهد؛ تازه‌ترین (بزرگ‌ترین شناسه) برگردانده می‌شود.
    assert parse_history_entry("9342413 2.02x 9342414 5.52x") == ("9342414", Decimal("5.52"))
    assert parse_history_entry("9342410 5.31x 9342411 1.23x 9342414 5.52x") == ("9342414",
                                                                                Decimal("5.52"))
    assert parse_history_entry("9342414 5.52x 9342413") == ("9342414", Decimal("5.52"))
    # نشان آخر که ضریبش ناقص خوانده شده، جفت ناقص نمی‌سازد.
    assert parse_history_entry("9342413 2.02x 9342414") == ("9342413", Decimal("2.02"))

def test_real_ocr_bang_and_countdown():
    import cv2
    recognizer = BangRecognizer(RapidOCRReader())
    for text, value in [("Bang @1.46x", Decimal("1.46")), ("Bang @7.50x", Decimal("7.50")),
                        ("Bang @13.05x", Decimal("13.05")), ("Starts in 4.4 s", None)]:
        image = np.zeros((120, 500, 3), np.uint8)
        cv2.putText(image, text, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        assert recognizer.recognize(image).value == value


def test_misread_bang_uses_detected_line_without_guessing_digits():
    class ViewportReader:
        def __call__(self, image):
            return "Bearg 66。", 0.55

        def detect_lines(self, image):
            return [("Bang @6.60x", 0.99), ("House Edge 1%", 0.95)]

    recognizer = BangRecognizer(ViewportReader())
    assert recognizer.recognize(np.zeros((194, 410, 3), np.uint8)).value == Decimal("6.60")
    recognizer.reader.detect_lines = lambda image: [("Bearg 66。", 0.99)]
    assert recognizer.recognize(np.zeros((194, 410, 3), np.uint8)).value is None
    recognizer.reader.detect_lines = lambda image: [("Bang @6.60x", 0.7)]
    assert recognizer.recognize(np.zeros((194, 410, 3), np.uint8)).value is None


def test_real_ocr_reads_a_padded_chart_viewport():
    import cv2
    image = np.full((194, 410, 3), 24, np.uint8)
    cv2.line(image, (20, 170), (390, 170), (55, 55, 55), 1)
    cv2.putText(image, "Bang @6.60x", (70, 82), cv2.FONT_HERSHEY_SIMPLEX,
                0.85, (230, 230, 230), 2)
    assert BangRecognizer(RapidOCRReader()).recognize(image).value == Decimal("6.60")

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

def _bang_until_stable(detector, now, value="10.00"):
    """دو فریم خالی و دو فریم Bang تا شمارش پایداری کامل شود."""
    detector.observe(_image(0), _reading(None), now)
    detector.observe(_image(0), _reading(None), now)
    detector.observe(_image(30), _reading(value), now)
    return detector.observe(_image(30), _reading(value), now)

def test_history_confirmation_and_strategy_handoff():
    detector = BangDetector(stable_frames=2, absent_frames=2)
    strategy = StrategyEngine(StrategyConfig())
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert _bang_until_stable(detector, now) is None
    confirmed = detector.observe(_image(30), _reading("10.00"), now,
                                 "9342414", Decimal("10.00"))
    assert confirmed and confirmed.round_id == "game-9342414"
    assert strategy.process_round(confirmed.round_id, confirmed.multiplier).cycle_triggered
    assert strategy.process_round(confirmed.round_id, confirmed.multiplier).duplicate

def test_mismatched_history_badge_never_delays_the_round():
    detector = BangDetector(stable_frames=2, absent_frames=2)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert _bang_until_stable(detector, now) is None
    # نوار تاریخچه هنوز به‌روز نشده: دست بدون هیچ انتظاری با شناسهٔ بصری ثبت می‌شود.
    confirmed = detector.observe(_image(30), _reading("10.00"), now,
                                 "9342414", Decimal("9.99"))
    assert confirmed and confirmed.round_id.startswith("visual-")
    assert confirmed.multiplier == Decimal("10.00")

def test_settings_roundtrip_and_validation(tmp_path):
    path = tmp_path / "settings.json"
    settings = DesktopSettings(Region(1, 1, 2, 200, 100), Region(1, 5, 8, 80, 30),
                               fixed_bet_amount="12.5", max_attempts=10)
    save_settings(settings, path)
    assert load_settings(path) == settings
    assert load_settings(tmp_path / "absent.json") == DesktopSettings()
def test_saved_double_bet_points_reload_as_tuples(tmp_path):
    """نقاط دوبل/نصف در JSON به لیست تبدیل می‌شوند؛ خواندن دوباره نباید خطا بدهد."""
    path = tmp_path / "settings.json"
    settings = DesktopSettings(double_bet_point=(521, 787), halve_bet_point=(481, 791))
    save_settings(settings, path)
    loaded = load_settings(path)
    assert loaded.double_bet_point == (521, 787)
    assert loaded.halve_bet_point == (481, 791)

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
                  autoclick_skip_rounds=0, bot_mode="turnbal",
                  smart_play_learning_minutes=0)
    values.update(overrides)
    return DesktopSettings(**values)


def test_classic_ignores_turnbal_rules_and_clicks_one_button(no_detection):
    sender = _Sender()
    settings = _clicker(bot_mode="classic", bet_region=Region(1, 100, 200, 300, 40),
                        autoclick_skip_rounds=7, pair_flip_enabled=True,
                        row_signal_enabled=True, value_bet_enabled=True,
                        turnbal_cashout_min=1.8, cashout_enabled=True)
    clicker = AutoClicker(settings, sender=sender)
    clicker.set_smart_play(("PLAY",))
    report = clicker.fire("high", Decimal("12.00"))
    assert report.clicked and report.rule == "smart-play"
    assert sender.calls == [(250, 220, True)]
    assert clicker.armed == 0 and clicker.waiting == 0
    assert clicker._cashout_target == Decimal("1.40")


def test_classic_does_not_click_without_a_decision(no_detection):
    sender = _Sender()
    clicker = AutoClicker(_clicker(bot_mode="classic", smart_click_enabled=False), sender=sender)
    report = clicker.fire("high", Decimal("12.00"))
    assert not report.clicked and not sender.calls and clicker.waiting == 0


def test_classic_rejects_color_decisions_and_never_uses_color_forecast(no_detection, monkeypatch):
    sender = _Sender()
    clicker = AutoClicker(_clicker(bot_mode="classic", smart_click_enabled=True), sender=sender)
    monkeypatch.setattr(clicker, "_color", lambda value: pytest.fail("color used in classic"))
    monkeypatch.setattr(clicker.forecaster, "observe", lambda value: pytest.fail("color forecast used"))
    clicker.set_smart_play(("RED", "GREEN", "YELLOW"))
    assert not clicker.fire("r1", Decimal("12")).clicked
    clicker.set_smart_play(("PLAY",))
    assert clicker.fire("r2", Decimal("12")).clicked


def test_classic_loss_doubles_once_and_confirmed_win_resets(no_detection):
    sender = _Sender()
    settings = _clicker(bot_mode="classic", bet_region=Region(1, 100, 200, 300, 40),
                        cashout_enabled=True, double_bet_enabled=True,
                        double_bet_point=(10, 20), halve_bet_point=(30, 40))
    clicker = AutoClicker(settings, sender=sender)
    clicker.set_smart_play(("PLAY",))
    assert clicker.fire("r1", Decimal("2.20")).clicked
    clicker.set_smart_play(("PLAY",))
    assert clicker.fire("r2", Decimal("1.20")).clicked
    assert clicker.double_level == 1 and sender.calls[1:3] == [(10, 20, True), (250, 220, True)]
    clicker.tick(Decimal("1.30"))
    clicker.tick(Decimal("1.45"))
    clicker.confirm_cashout(Decimal("1.45"))
    clicker.set_smart_play(("PLAY",))
    assert clicker.fire("r3", Decimal("1.55")).clicked
    assert clicker.double_level == 0 and sender.calls[-2:] == [(30, 40, True), (250, 220, True)]

def test_red_streak_hold_blocks_new_bets_and_keeps_doubling(no_detection):
    """توقف قرمزهای پیاپی: شرط تازه بسته نمی‌شود و دوبلکردن از دست نمی‌رود."""
    sender = _Sender()
    settings = _clicker(bot_mode="classic", bet_region=Region(1, 100, 200, 300, 40),
                        cashout_enabled=True, double_bet_enabled=True,
                        double_bet_point=(10, 20), halve_bet_point=(30, 40))
    clicker = AutoClicker(settings, sender=sender)
    clicker.set_smart_play(("PLAY",))
    assert clicker.fire("r1", Decimal("2.20")).clicked
    clicker.set_smart_play(("PLAY",))
    assert clicker.fire("r2", Decimal("1.20")).clicked  # باخت دست اول → دوبل
    assert clicker.double_level == 1
    sent = list(sender.calls)
    clicker.hold_clicks(60, "red streak")
    clicker.set_smart_play(("PLAY",))
    report = clicker.fire("r3", Decimal("1.10"))
    assert not report.clicked and "hold red streak" in report.reason
    # تسویهٔ دو باخت در همین توقف هم شمرده شد؛ شرط تازه‌ای بسته نشد.
    assert sender.calls == sent and clicker.double_level == 2
    clicker.hold_clicks(0)
    clicker.set_smart_play(("PLAY",))
    assert clicker.fire("r4", Decimal("1.10")).stake == 40


def test_manual_click_test_clicks_even_in_dry_run(no_detection):
    sender = _Sender()
    clicker = AutoClicker(_clicker(autoclick_dry_run=True), sender=sender)
    report = clicker.test()
    assert report.clicked and sender.calls == [(130, 220, True)]


def test_button_picker_accepts_a_single_click(monkeypatch):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QDialog
    from crash_watcher.desktop import ui

    class Screen:
        monitors = [None, {"left": 0, "top": 0, "width": 100, "height": 100}]
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def grab(self, region): return np.zeros((100, 100, 4), dtype=np.uint8)

    monkeypatch.setattr(ui.mss, "MSS", Screen)
    app = QApplication.instance() or QApplication([])
    overlay = ui.RegionOverlay(1, point_mode=True)
    overlay.show()
    QTest.mouseClick(overlay, Qt.MouseButton.LeftButton, pos=QPoint(25, 35))
    assert overlay.result() == QDialog.DialogCode.Accepted
    assert (overlay.selected.x, overlay.selected.y) == (25, 35)


def test_loss_doubles_the_stake_and_a_win_halves_it_back(no_detection):
    sender = _Sender()
    settings = _clicker(autoclick_threshold="100",
                        value_bets={"1.50": "RED", "3.00": "GREEN"},
                        double_bet_enabled=True, double_bet_max=4,
                        double_bet_point=(10, 20), halve_bet_point=(30, 40),
                        fixed_bet_amount="10", bankroll_enabled=True,
                        bankroll_start="100")
    clicker = AutoClicker(settings, sender=sender)
    assert clicker.fire("r1", Decimal("1.50")).clicked
    assert sender.calls == [(130, 220, True)] and clicker.balance == 100
    lost = clicker.fire("r2", Decimal("3.00"))  # سبز: شرط قرمز باخت
    assert lost.stake == 20 and clicker.double_level == 1
    assert sender.calls[1:] == [(10, 20, True), (130, 220, True)]
    assert clicker.balance == 90
    won = clicker.fire("r3", Decimal("3.00"))  # سبز: شرط دوبل‌شده برد
    assert won.stake == 10 and clicker.double_level == 0
    assert sender.calls[3:] == [(30, 40, True), (130, 220, True)]
    assert clicker.balance == 110


def test_bet_doubling_resets_after_four_steps(no_detection):
    sender = _Sender()
    settings = _clicker(autoclick_threshold="100", value_bets={"3.00": "RED"},
                        double_bet_enabled=True, double_bet_max=4,
                        double_bet_point=(10, 20), halve_bet_point=(30, 40),
                        fixed_bet_amount="10")
    clicker = AutoClicker(settings, sender=sender)
    stakes = [clicker.fire(f"r{index}", Decimal("3.00")).stake
              for index in range(1, 8)]
    assert stakes == [10, 20, 40, 80, 160, 10, 20]
    assert sender.calls[9:14] == [(30, 40, True)] * 4 + [(130, 220, True)]


def test_ten_doubling_steps_reset_to_base(no_detection):
    settings = _clicker(autoclick_threshold="100", value_bets={"3.00": "RED"},
                        double_bet_enabled=True, double_bet_max=10,
                        double_bet_point=(10, 20), halve_bet_point=(30, 40),
                        fixed_bet_amount="10")
    settings.validate()
    clicker = AutoClicker(settings, sender=_Sender())
    stakes = [clicker.fire(f"r{index}", Decimal("3.00")).stake
              for index in range(12)]
    assert stakes == [10 * 2**level for level in range(11)] + [10]


def test_failed_stake_adjustment_does_not_repeat_successful_clicks(no_detection):
    calls = []
    def sender(x, y, restore=True):
        calls.append((x, y))
        return len(calls) != 2
    settings = _clicker(double_bet_enabled=True, double_bet_max=4,
                        double_bet_point=(10, 20), halve_bet_point=(30, 40))
    clicker = AutoClicker(settings, sender=sender)
    clicker._halve_clicks = 3
    assert not clicker._apply_double_clicks()
    assert clicker._halve_clicks == 2
    assert clicker._apply_double_clicks()
    assert len(calls) == 4 and clicker._halve_clicks == 0


def test_cashout_confirmation_overrides_final_color(no_detection, tmp_path):
    sender = _Sender()
    settings = _clicker(autoclick_threshold="100", value_bets={"1.50": "RED", "1.20": "GREEN"},
                        cashout_enabled=True, double_bet_enabled=True, double_bet_max=4,
                        double_bet_point=(10, 20), halve_bet_point=(30, 40))
    clicker = AutoClicker(settings, sender=sender, log_path=tmp_path / "rounds.jsonl")
    clicker.fire("r1", Decimal("1.50"))
    clicker.fire("r2", Decimal("1.20"))
    assert clicker.double_level == 1
    clicker.tick(Decimal("1.40"))
    clicker.tick(Decimal("1.50"))
    clicker.confirm_cashout(Decimal("1.50"))
    result = clicker.fire("r3", Decimal("1.50"))
    assert result.stake == 10 and clicker.double_level == 0
    assert (30, 40, True) in sender.calls


def test_unconfirmed_cashout_blocks_next_bet(no_detection, tmp_path):
    sender = _Sender()
    settings = _clicker(autoclick_threshold="100", value_bets={"1.50": "RED", "3.00": "GREEN"},
                        cashout_enabled=True, double_bet_enabled=True,
                        double_bet_point=(10, 20), halve_bet_point=(30, 40))
    clicker = AutoClicker(settings, sender=sender, log_path=tmp_path / "rounds.jsonl")
    clicker.fire("r1", Decimal("1.50"))
    clicker.tick(Decimal("1.40"))
    clicker.tick(Decimal("1.50"))
    count = len(sender.calls)
    # دست زیر ضریب کلیک خروج بسته شد: خروج قطعی نیست، پس شرط تازه بسته نمی‌شود.
    result = clicker.fire("r2", Decimal("1.20"))
    assert not result.clicked and "unconfirmed" in result.reason
    assert len(sender.calls) == count
    # برابر با ضریب کلیک («1.50») هنوز نامعلوم است: می‌تواند کرش همان لحظه باشد.
    equal = clicker.fire("r2a", Decimal("1.50"))
    assert not equal.clicked and "unconfirmed" in equal.reason
    # دست بعدی روشن بالای همان ضریب بسته شد: خروج انجام شده، مبلغ به پایه برمی‌گردد
    # و شرط تازه بسته می‌شود (انتظار برای پیام سایت هر دست یک شرط را می‌سوزاند).
    settled = clicker.fire("r2b", Decimal("3.00"))
    assert settled.clicked and clicker.double_level == 0


def test_inflight_win_ocr_is_settled_before_next_bet(no_detection, tmp_path):
    from concurrent.futures import Future
    from crash_watcher.desktop.worker import resolve_win_ocr

    sender = _Sender()
    settings = _clicker(autoclick_threshold="100", value_bets={"1.50": "RED", "3.00": "GREEN"},
                        cashout_enabled=True, double_bet_enabled=True,
                        double_bet_point=(10, 20), halve_bet_point=(30, 40))
    clicker = AutoClicker(settings, sender=sender, log_path=tmp_path / "rounds.jsonl")
    clicker.fire("r1", Decimal("1.50"))
    clicker.tick(Decimal("1.40"))
    clicker.tick(Decimal("1.50"))
    pending = Future()
    assert not resolve_win_ocr(pending, clicker, "r1", True)
    assert not clicker._cashout_confirmed
    pending.set_result([("You Won 4.24 CUB", 0.99)])
    assert resolve_win_ocr(pending, clicker, "different-round", True)
    assert not clicker._cashout_confirmed
    assert resolve_win_ocr(pending, clicker, "r1", False)
    assert not clicker._cashout_confirmed
    assert resolve_win_ocr(pending, clicker, "r1", True)
    assert clicker._cashout_confirmed
    assert clicker.fire("r2", Decimal("3.00")).clicked
    assert clicker.double_level == 0


def test_win_message_from_supplied_image():
    import cv2
    from rapidocr import RapidOCR
    from crash_watcher.desktop.worker import detected_lines, has_win_message
    img = cv2.imread(str(Path(__file__).resolve().parents[1] / "assets/successful-cashout-example.png"))
    assert has_win_message(detected_lines(RapidOCR(), img))
    assert not has_win_message([("Bang @1.16", 0.99), ("You Won", 0.3)])


def test_win_message_from_current_multiplier_reference():
    import cv2
    from rapidocr import RapidOCR
    from crash_watcher.desktop.worker import detected_lines, has_win_message
    img = cv2.imread(str(Path(__file__).resolve().parents[1] / "assets/cashout-you-won-reference.png"))
    assert has_win_message(detected_lines(RapidOCR(), img))

def test_win_message_split_into_separate_lines_still_confirms():
    """OCR واقعی «You Won» را دو خط جدا می‌خواند؛ باید باز هم برد شناخته شود."""
    import cv2
    from rapidocr import RapidOCR
    from crash_watcher.desktop.worker import detected_lines, has_win_message
    img = cv2.imread(str(Path(__file__).resolve().parents[1] / "assets/you-won-doge-reference.png"))
    lines = detected_lines(RapidOCR(), img)
    assert [text for text, _ in lines][:2] == ["You", "Won"]
    assert has_win_message(lines)


@pytest.mark.parametrize("losses", [1, 2, 3, 4])
def test_win_at_each_doubling_level_returns_to_base(no_detection, losses):
    sender = _Sender()
    settings = _clicker(autoclick_threshold="100",
                        value_bets={"1.50": "RED", "3.00": "GREEN"},
                        double_bet_enabled=True, double_bet_max=4,
                        double_bet_point=(10, 20), halve_bet_point=(30, 40),
                        fixed_bet_amount="10")
    clicker = AutoClicker(settings, sender=sender)
    clicker.fire("start", Decimal("1.50"))
    for index in range(losses):
        # Alternating results always lose against the preceding target.
        value = Decimal("3.00") if index % 2 == 0 else Decimal("1.50")
        clicker.fire(f"loss{index}", value)
    win_value = Decimal("3.00") if losses % 2 else Decimal("1.50")
    report = clicker.fire("win", win_value)
    assert report.stake == 10 and clicker.double_level == 0
    assert sender.calls[-(losses + 1):] == [(30, 40, True)] * losses + [(130, 220, True)]


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
    assert slept == [1.0]


def test_clicker_places_automatic_bet_without_delay(no_detection, monkeypatch):
    slept = []
    monkeypatch.setattr("crash_watcher.desktop.clicker.sleep", lambda seconds: slept.append(seconds))
    sender = _Sender()
    clicker = AutoClicker(_clicker(autoclick_delay_ms=1000), sender=sender)
    assert clicker.fire("r1", Decimal("12.00")).clicked
    assert slept == [] and len(sender.calls) == 1
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

def test_dry_run_reports_the_click_but_never_sends_it(no_detection):
    sender = _Sender()
    dry = AutoClicker(_clicker(autoclick_dry_run=True), sender=sender)
    report = dry.fire("r1", Decimal("10.00"))
    assert not report.clicked and report.dry_run
    assert report.reason == "dry run: would click"
    assert report.point == (130, 220) and sender.calls == []
    off = AutoClicker(DesktopSettings(autoclick_enabled=False), sender=sender)
    assert off.fire("r1", Decimal("99.00")).reason == "disabled"
    assert sender.calls == []




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






def test_pair_flip_clicks_the_opposite_color(no_detection):
    sender = _Sender()
    clicker = AutoClicker(_clicker(autoclick_threshold="100", bet_point_red=(11, 12),
                                   bet_point_green=(21, 22)), sender=sender)
    assert not clicker.fire("a", Decimal("5.00")).clicked
    assert not clicker.fire("b", Decimal("6.00")).clicked
    flip = clicker.fire("c", Decimal("1.20"))
    assert flip.clicked and flip.target == "RED" and flip.point == (11, 12)
    assert not clicker.fire("d", Decimal("1.10")).clicked
    assert len(sender.calls) == 1

def test_pair_flip_after_two_reds_targets_green(no_detection):
    sender = _Sender()
    clicker = AutoClicker(_clicker(autoclick_threshold="100", bet_point_green=(31, 32)),
                          sender=sender)
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
                                   bet_point_yellow=(51, 52)), sender=sender)
    monkeypatch.setattr(AutoClicker, "_chart_reading", lambda self: object())
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




def test_settings_file_with_removed_keys_still_loads(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"fixed_bet_amount": "12",
                                "pair_cycle_enabled": True,
                                "double_bet_enabled": False}), encoding="utf-8")
    loaded = load_settings(path)
    assert loaded.fixed_bet_amount == "12"


def test_value_bets_reject_bad_entries():
    with pytest.raises(ValueError):
        _clicker(value_bets={"1.00": "BLUE"}).validate()
    with pytest.raises(ValueError):
        _clicker(value_bets={"abc": "GREEN"}).validate()

def test_clicks_are_attributed_to_a_rule_and_the_report_counts_hits(tmp_path, no_detection):
    sender = _Sender()
    clicker = AutoClicker(_clicker(autoclick_threshold="100", pair_flip_enabled=False,
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
                                         smart_play_learning_minutes=0, bot_mode="turnbal"),
                         sender=sender).fire("r1", Decimal("10.00"))
    assert not report.clicked and report.reason == "target not found" and sender.calls == []


# --- shadow mode, bankroll, turnbal guards, model comparison, secrets ---------

def test_shadow_mode_never_sends_but_records_the_decision(no_detection):
    sender = _Sender()
    clicker = AutoClicker(_clicker(shadow_mode=True), sender=sender)
    report = clicker.fire("r1", Decimal("10.00"))
    assert not report.clicked and report.shadow
    assert report.reason == "shadow: would click" and report.target == "YELLOW"
    assert sender.calls == []


def test_dry_run_cashout_does_not_send_a_click(no_detection):
    sender = _Sender()
    clicker = AutoClicker(_clicker(autoclick_dry_run=True, cashout_enabled=True),
                          sender=sender)
    clicker.fire("r1", Decimal("10.00"))
    clicker.set_smart_play((), 1.5)
    clicker.fire("r2", Decimal("1.20"))
    clicker.tick(Decimal("1.45"))
    clicker.tick(Decimal("1.60"))
    assert sender.calls == []


def test_classic_cashout_two_percent_above_the_click_counts_as_a_win(no_detection):
    """بسته‌شدن دست ۲٪ بالای ضریب کلیک، بدون پیام سایت هم برد است."""
    sender = _Sender()
    clicker = AutoClicker(_clicker(bot_mode="classic", bet_region=Region(1, 100, 200, 300, 40),
                                   cashout_enabled=True), sender=sender)
    clicker.set_smart_play(("PLAY",))
    assert clicker.fire("r1", Decimal("2.20")).clicked
    clicker.tick(Decimal("1.30"))
    clicker.tick(Decimal("1.45"))
    assert clicker._cashout_attempt_multiplier == Decimal("1.45")
    assert clicker._cashout_settled(Decimal("1.46")) is False
    assert clicker._cashout_settled(Decimal("1.48")) is True
    clicker.set_smart_play(("PLAY",))
    assert clicker.fire("r2", Decimal("1.54")).clicked
    assert clicker._rule_stats["smart-play"] == [1, 2]


def test_bankroll_stops_the_session_after_the_loss_limit(no_detection):
    sender = _Sender()
    clicker = AutoClicker(_clicker(bankroll_enabled=True, bankroll_start="10",
                                   bankroll_stop_loss_pct=20, fixed_bet_amount="10"),
                          sender=sender)
    assert clicker.fire("r1", Decimal("10.00")).clicked
    assert len(sender.calls) == 1
    # The click was on yellow; a red round loses the whole stake.
    clicker.fire("r2", Decimal("1.50"))
    assert clicker.balance == 0 and clicker.bankroll.stopped
    blocked = clicker.fire("r3", Decimal("10.00"))
    assert not blocked.clicked and "capital stop-loss" in blocked.reason
    assert len(sender.calls) == 1


def test_first_minute_only_evaluates_and_places_no_bet(no_detection):
    """دقیقهٔ اول: مدل ارزیابی می‌کند و هیچ شرطی بسته نمی‌شود."""
    sender = _Sender()
    clicker = AutoClicker(_clicker(bot_mode="classic", bet_region=Region(1, 100, 200, 300, 40),
                                   smart_play_learning_minutes=1), sender=sender)
    clicker.set_smart_play(("PLAY",))
    assert clicker.hold_remaining() > 55
    report = clicker.fire("r1", Decimal("12.00"))
    assert not report.clicked and "evaluation" in report.reason
    assert sender.calls == []
    clicker._hold_until = None
    clicker.set_smart_play(("PLAY",))
    assert clicker.fire("r2", Decimal("12.00")).clicked


def test_session_profit_limit_stops_new_bets(no_detection):
    sender = _Sender()
    settings = _clicker(value_bets={"1.50": "RED", "3.00": "GREEN"},
                        autoclick_threshold="100", smart_play_learning_minutes=0,
                        bankroll_start="100", session_profit_limit="5",
                        fixed_bet_amount="10", bet_point_red=(10, 20),
                        bet_point_green=(30, 40))
    clicker = AutoClicker(settings, sender=sender)
    assert clicker.fire("r1", Decimal("1.50")).clicked
    stopped = clicker.fire("r2", Decimal("1.50"))
    assert clicker.balance == 110 and clicker.bankroll.profit_hit
    assert not stopped.clicked and "capital stop-loss" in stopped.reason


def test_session_loss_limit_stops_new_bets(no_detection):
    sender = _Sender()
    settings = _clicker(value_bets={"1.50": "RED"}, autoclick_threshold="100",
                        smart_play_learning_minutes=0, bankroll_start="100",
                        session_loss_limit="5", fixed_bet_amount="10",
                        bet_point_red=(10, 20), bet_point_green=(30, 40))
    clicker = AutoClicker(settings, sender=sender)
    assert clicker.fire("r1", Decimal("1.50")).clicked
    stopped = clicker.fire("r2", Decimal("3.00"))
    assert clicker.balance == 90 and clicker.bankroll.loss_hit
    assert not stopped.clicked and "capital stop-loss" in stopped.reason


def test_advice_cashout_target_is_kept_inside_the_safe_band(no_detection):
    def target(predicted):
        clicker = AutoClicker(_clicker(bot_mode="classic", smart_play_learning_minutes=0,
                                       bet_region=Region(1, 100, 200, 300, 40),
                                       cashout_enabled=True), sender=_Sender())
        clicker.set_smart_play(("PLAY",), predicted)
        assert clicker.fire("r1", Decimal("12.00")).clicked
        return clicker._cashout_target

    assert target(3.5) == Decimal("1.90")
    assert target(1.2) == Decimal("1.40")
    assert target(1.65) == Decimal("1.65")


def test_a_round_closed_below_1_10_bets_the_next_hand_until_1_90(no_detection):
    """بستن زیر ۱٫۱۰: کلیک برای دست بعد و صبر تا ۱٫۹۰ برای خروج."""
    sender = _Sender()
    clicker = AutoClicker(_clicker(bot_mode="classic", smart_play_learning_minutes=0,
                                   bet_region=Region(1, 100, 200, 300, 40),
                                   cashout_enabled=True), sender=sender)
    report = clicker.fire("r1", Decimal("1.08"))
    assert report.clicked and report.rule == "low-close"
    assert clicker._cashout_target == Decimal("1.90")
    assert sender.calls == [(250, 220, True)]

    normal = AutoClicker(_clicker(bot_mode="classic", smart_play_learning_minutes=0,
                                  bet_region=Region(1, 100, 200, 300, 40),
                                  cashout_enabled=True), sender=_Sender())
    assert not normal.fire("r1", Decimal("1.10")).clicked


def test_bankroll_settles_a_win_and_a_loss_with_the_colour_payout():
    account = Bankroll(10, 0.2)
    account.settle(2, 10, True)
    assert account.balance == 28 and account.realized == 18 and account.wins == 1
    account.settle(2, 2, False)
    assert account.balance == 26 and account.losses == 1 and not account.stopped


def test_bankroll_stops_at_the_profit_and_loss_amounts():
    profit = Bankroll(100, 0.2, profit_limit=5, loss_limit=8)
    profit.settle(10, 2.0, True)
    assert profit.realized == 10 and profit.profit_hit and profit.stopped
    assert "حد سود" in profit.label()
    loss = Bankroll(100, 0.2, profit_limit=5, loss_limit=8)
    loss.settle(10, 2.0, False)
    assert loss.realized == -10 and loss.loss_hit and loss.stopped
    assert "حد ضرر" in loss.label()


def test_bankroll_treats_a_hedge_and_a_simulated_stake_as_break_even():
    account = Bankroll(10, 0.2)
    assert account.settle(2, 2, None) == 0
    assert account.balance == 10 and account.break_even == 1
    assert account.wins == account.losses == 0
    assert account.settle(0, 2, False) == 0
    assert account.balance == 10 and account.losses == 0


def test_turnbal_calibration_rejects_disabled_and_low_confidence():
    calibration = TurnbalCalibration(yellow=ColorRule(False, 0.8, 2),
                                     green=ColorRule(True, 0.6, 3))
    rejected, reason = calibrate(SmartVerdict("YELLOW", 0.95, ""), calibration)
    assert rejected is None and "غیرفعال" in reason
    weak, reason = calibrate(SmartVerdict("GREEN", 0.5, ""), calibration)
    assert weak is None and "کمتر" in reason
    allowed, _ = calibrate(SmartVerdict("GREEN", 0.7, "", cashout=1.8), calibration)
    assert allowed is not None and allowed.cashout == 1.8


def test_turnbal_guard_enforces_max_streak():
    calibration = TurnbalCalibration(max_streak=2, min_gap=0, red=ColorRule(True, 0.5, 99))
    guard = TurnbalGuard(calibration)
    verdict = SmartVerdict("RED", 0.9, "")
    assert guard.check(verdict)[0] is verdict
    guard.begin(("RED",))
    guard.observe("RED")
    assert guard.check(verdict)[0] is verdict
    guard.begin(("RED",))
    guard.observe("GREEN")
    blocked, reason = guard.check(verdict)
    assert blocked is None and "پشت‌سرهم" in reason


def test_turnbal_guard_enforces_the_gap_between_bets():
    calibration = TurnbalCalibration(max_streak=99, min_gap=2, red=ColorRule(True, 0.5, 99))
    guard = TurnbalGuard(calibration)
    verdict = SmartVerdict("RED", 0.9, "")
    guard.begin(("RED",))
    guard.observe("GREEN")
    assert guard.check(verdict)[0] is None
    guard.observe("GREEN")
    assert guard.check(verdict)[0] is None       # only one round has passed
    guard.observe("GREEN")
    assert guard.check(verdict)[0] is verdict


def test_turnbal_guard_enforces_attempts_per_cycle():
    calibration = TurnbalCalibration(max_streak=99, min_gap=0, red=ColorRule(True, 0.5, 1))
    guard = TurnbalGuard(calibration)
    verdict = SmartVerdict("RED", 0.9, "")
    guard.begin(("RED",))
    guard.observe("GREEN")
    blocked, reason = guard.check(verdict)
    assert blocked is None and "تلاش" in reason


def test_turnbal_guard_blocks_below_the_cycle_win_rate():
    calibration = TurnbalCalibration(max_streak=99, min_gap=0, min_cycle_win=0.6,
                                     red=ColorRule(True, 0.5, 99))
    guard = TurnbalGuard(calibration, min_samples=2)
    verdict = SmartVerdict("RED", 0.9, "")
    for _ in range(2):
        guard.begin(("RED",))
        guard.observe("GREEN")
    blocked, reason = guard.check(verdict)
    assert blocked is None and "نرخ برد" in reason


def test_turnbal_guard_caps_the_cashout_at_the_emergency_exit():
    calibration = TurnbalCalibration(emergency_exit=1.55)
    assert TurnbalGuard(calibration).cap_cashout(1.90) == 1.55
    assert TurnbalGuard(calibration).cap_cashout(1.45) == 1.45


def test_turnbal_prompt_states_the_enforced_limits():
    calibration = TurnbalCalibration(max_streak=4, min_gap=2, emergency_exit=1.55,
                                     min_cycle_win=0.55)
    text = turnbal_prompt(SmartSnapshot(recent=("1.20", "9.00")), calibration)
    assert "4 دست" in text and "2 دست" in text
    assert "1.55" in text and "55%" in text


def test_smart_planner_scores_the_second_colour_as_break_even_only():
    planner = SmartPlanner(0, 0.0, True, clock=lambda: 0)
    planner.store(SmartVerdict("RED", 0.7, "", second="GREEN"), "r1")
    planner.observe("r2", "GREEN")
    assert planner.tries == 1 and planner.hits == 0 and planner.break_even == 1
    assert planner.measured_accuracy == 0.0
    assert planner.brier == pytest.approx(0.49)
    planner.store(SmartVerdict("RED", 0.9, ""), "r2")
    planner.observe("r3", "RED")
    assert planner.hits == 1 and planner.tries == 2
    assert planner.brier == pytest.approx((0.49 + 0.01) / 2)


def test_model_leaderboard_picks_the_lowest_brier_and_handles_cold_start(tmp_path):
    board = ModelLeaderboard(tmp_path / "scores.json", min_samples=3)
    for _ in range(3):
        board.record("good", 0.9, True)
        board.record("bad", 0.9, False)
    board.save()
    chosen, note = board.choose([ModelAttempt("good", SmartVerdict("RED", 0.5, "")),
                                 ModelAttempt("bad", SmartVerdict("GREEN", 0.99, ""))])
    assert chosen.model == "good" and "Brier" in note
    assert ModelLeaderboard(tmp_path / "scores.json", min_samples=3).best(["good"])
    cold = ModelLeaderboard(None, min_samples=3)
    chosen, note = cold.choose([ModelAttempt("a", SmartVerdict("RED", 0.4, "")),
                                ModelAttempt("b", SmartVerdict("GREEN", 0.9, ""))])
    assert chosen.model == "b" and "سرد" in note


def test_compare_runs_models_in_parallel_and_isolates_errors():
    attempts = compare_models(["a", "b", "c"],
                              lambda model: (SmartVerdict("RED", 0.6, model), model, "p"))
    assert [attempt.model for attempt in attempts] == ["a", "b", "c"]
    assert all(attempt.ok and attempt.verdict.reason == attempt.model for attempt in attempts)

    def boom(model):
        raise RuntimeError("unreachable")

    failed = compare_models(["x"], boom)
    assert failed[0].error == "unreachable" and not failed[0].ok
    assert parse_models("a, b، a ,, c") == ["a", "b", "c"]


def test_settings_key_is_encrypted_at_rest_and_env_wins(tmp_path, monkeypatch):
    sealed = protect("sk-secret-123456")
    assert unprotect(sealed) == "sk-secret-123456"
    stored = tmp_path / "settings.json"
    save_settings(DesktopSettings(smart_review_key="sk-secret-123456"), stored)
    if is_protected(sealed):
        assert "sk-secret-123456" not in stored.read_text(encoding="utf-8")
    assert load_settings(stored).smart_review_key == "sk-secret-123456"
    monkeypatch.setenv("CRASH_WATCHER_API_KEY", "env-key")
    assert DesktopSettings(smart_review_key="file-key").resolved_key() == "env-key"
    monkeypatch.delenv("CRASH_WATCHER_API_KEY")
    assert DesktopSettings(smart_review_key="file-key").resolved_key() == "file-key"


def test_preflight_flags_missing_regions_and_accepts_a_configured_setup():
    checks = run_preflight(DesktopSettings(), capture=lambda region: None, check_ocr=False)
    problems = {check.name for check in checks if not check.ok}
    assert "ناحیهٔ نتیجه (۲)" in problems and "هدف کلیک / دکمهٔ شرط" in problems
    image = np.zeros((20, 40, 3), np.uint8)
    good = DesktopSettings(result_region=Region(1, 0, 0, 40, 20),
                           chart_region=Region(1, 0, 0, 40, 20), click_point=(5, 5))
    checks = run_preflight(good, capture=lambda region: image, check_ocr=False,
                           grid=lambda picture: SimpleNamespace(columns=3, rows=2))
    assert all(check.ok for check in checks), [check.label for check in checks]
