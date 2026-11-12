from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
import os
import numpy as np
import pytest
from crash_watcher import StrategyConfig, StrategyEngine
from crash_watcher.desktop.config import DesktopSettings, Region, load_settings, save_settings
from crash_watcher.desktop.final_result import (BangDetector, BangRecognizer, FinalReading,
                                                 parse_bang, parse_history_entry)
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
