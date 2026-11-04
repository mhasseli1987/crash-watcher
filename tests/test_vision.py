from datetime import datetime, timedelta, timezone
from decimal import Decimal
import numpy as np
import pytest
from crash_watcher import StrategyConfig, StrategyEngine
from crash_watcher.core.enums import MultiplierCategory
from crash_watcher.vision.capture import CaptureConfig, ScreenCapture, ScreenUnavailable
from crash_watcher.vision.detection import DetectorConfig, RoundDetector
from crash_watcher.vision.models import (DetectionEventType, DetectionState, Frame, Recognition)
from crash_watcher.vision.pipeline import VisualPipeline
from crash_watcher.vision.recognition import MultiplierRecognizer, RapidOCRReader, category_for, parse_multiplier

@pytest.mark.parametrize("text,value", [("1.08x", "1.08"), ("2.45x", "2.45"),
    ("9.99x", "9.99"), ("10.00x", "10.00"), ("12.45x", "12.45"),
    ("25.67x", "25.67"), (" 1.08× ", "1.08")])
def test_parse(text, value):
    assert parse_multiplier(text) == Decimal(value)

@pytest.mark.parametrize("text", ["1O.OO", "1..00", "10.OO", "1,0O", "NaN", "Infinity",
    "", "0.00x", "-1.00x", "10x", "10.0x", "10.000x", "1 0.00x", "1,000.00x"])
def test_reject_malformed(text):
    assert parse_multiplier(text) is None

@pytest.mark.parametrize("text,category", [("1.99x", "RED"), ("2.00x", "GREEN"),
    ("9.99x", "GREEN"), ("10.00x", "YELLOW")])
def test_categories(text, category):
    assert category_for(parse_multiplier(text)).value == category

def test_recognition_synthetic_image():
    import cv2
    image = np.zeros((90, 300, 3), np.uint8)
    cv2.putText(image, "12.45x", (5, 63), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 220, 220), 3)
    seen = []
    def reader(variant):
        seen.append(variant)
        return "12.45x", .95
    result = MultiplierRecognizer(reader).recognize(image)
    assert result.multiplier == Decimal("12.45")
    assert result.category == MultiplierCategory.YELLOW
    assert result.color_category == MultiplierCategory.YELLOW
    assert len(seen) == 3 and all(v.ndim == 2 for v in seen)

@pytest.mark.parametrize("text", ["1.08x", "2.45x", "9.99x", "10.00x", "12.45x", "25.67x"])
def test_real_local_ocr_on_synthetic_text(text):
    import cv2
    image = np.zeros((120, 360, 3), np.uint8)
    cv2.putText(image, text, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    result = MultiplierRecognizer(RapidOCRReader()).recognize(image)
    assert result.multiplier == parse_multiplier(text)
    assert not result.uncertain

def test_uncertain_ocr():
    image = np.zeros((30, 90, 3), np.uint8)
    assert MultiplierRecognizer(lambda _: ("1O.OO", .99)).recognize(image).uncertain
    assert MultiplierRecognizer(lambda _: ("10.00x", .2)).recognize(image).uncertain
    readings = iter([("10.00x", .95), ("12.00x", .95), ("10.00x", .95)])
    assert MultiplierRecognizer(lambda _: next(readings)).recognize(image).uncertain

def _frame(pattern, index):
    image = np.full((40, 80, 3), pattern, np.uint8)
    return Frame(image, datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(milliseconds=index * 200))

def _rec(value):
    if value is None:
        return Recognition(None, None, 0, "", True)
    value = Decimal(str(value))
    return Recognition(value, category_for(value), .99, f"{value}x", False)

def test_visual_lifecycle_duplicate_and_identical_new_round():
    d = RoundDetector(DetectorConfig(stable_frames=2, motion_frames=2, min_round_ms=200))
    # Startup result stays visible; no completion.
    for i in range(3):
        assert d.inspect(_frame(20, i), _rec("10.00")).completed is None
    d.inspect(_frame(90, 3), None)
    started = d.inspect(_frame(180, 4), None)
    assert any(e.kind == DetectionEventType.ROUND_STARTED for e in started.events)
    d.inspect(_frame(180, 5), _rec(None))  # visible blank/uncertain phase
    d.inspect(_frame(30, 6), None)
    d.inspect(_frame(30, 7), _rec("10.00"))
    confirmed = d.inspect(_frame(30, 8), _rec("10.00"))
    assert confirmed.completed.multiplier == Decimal("10.00")
    assert confirmed.state == DetectionState.RESULT_CONFIRMED
    for i in range(9, 15):
        assert d.inspect(_frame(30, i), _rec("10.00")).completed is None
    d.inspect(_frame(90, 15), None)
    d.inspect(_frame(180, 16), None)
    d.inspect(_frame(180, 17), _rec(None))
    d.inspect(_frame(30, 18), None)
    d.inspect(_frame(30, 19), _rec("10.00"))
    again = d.inspect(_frame(30, 20), _rec("10.00"))
    assert again.completed and again.completed.round_id != confirmed.completed.round_id

def test_animation_and_missing_frames():
    d = RoundDetector(DetectorConfig(stable_frames=2, motion_frames=1, min_round_ms=0))
    d.inspect(_frame(0, 0), _rec(None))
    d.inspect(_frame(100, 1), None)
    for i in range(2, 8):
        assert d.inspect(_frame(100 + i * 10, i), _rec(i)).completed is None
    assert d.inspect(None, None).state == DetectionState.UNCERTAIN

def test_no_blank_no_confirmation():
    d = RoundDetector(DetectorConfig(stable_frames=2, motion_frames=1, min_round_ms=0))
    d.inspect(_frame(0, 0), _rec("1.00"))
    d.inspect(_frame(100, 1), None)
    for i in range(2, 7):
        assert d.inspect(_frame(100, i), _rec("1.00")).completed is None

class FakeBackend:
    def __init__(self):
        self.monitors = [{}, {"left": 0, "top": 0, "width": 100, "height": 100}]
        self.fail = False
    def grab(self, area):
        if self.fail:
            raise OSError("offline")
        return np.zeros((area["height"], area["width"], 4), np.uint8)

def test_capture_failures_and_resolution_change():
    b = FakeBackend()
    c = ScreenCapture(CaptureConfig(1, 0, 0, 20, 20), b)
    assert len(c.monitors()) == 1
    frame, events = c.capture()
    assert frame.image.shape == (20, 20, 3) and events == ()
    b.monitors[1]["width"] = 120
    assert c.capture()[1][0].kind == DetectionEventType.SCREEN_REGION_CHANGED
    b.monitors[1]["width"] = 10
    with pytest.raises(ScreenUnavailable):
        c.capture()
    b.monitors[1]["width"] = 100
    b.fail = True
    with pytest.raises(ScreenUnavailable):
        c.capture()

def test_detector_frame_dimensions_change():
    d = RoundDetector()
    d.inspect(_frame(0, 0), _rec(None))
    frame = Frame(np.zeros((30, 30, 3), np.uint8), _frame(0, 1).captured_at)
    assert d.inspect(frame, None).events[0].kind == DetectionEventType.SCREEN_REGION_CHANGED

def test_pipeline_integrates_once():
    e = StrategyEngine(StrategyConfig())
    events = []
    p = VisualPipeline(ScreenCapture(CaptureConfig(1, 0, 0, 20, 20), FakeBackend()),
                       MultiplierRecognizer(lambda _: ("10.00x", .99)),
                       RoundDetector(DetectorConfig(stable_frames=2, motion_frames=1, min_round_ms=0)),
                       e, on_event=events.append)
    # Explicit blank reading during a static in-progress frame.
    for i, color in enumerate([0, 100, 100, 30, 30, 30]):
        if i == 2:
            p.recognizer = MultiplierRecognizer(lambda _: ("", 0))
        if i == 4:
            p.recognizer = MultiplierRecognizer(lambda _: ("10.00x", .99))
        output = p.handle_frame(_frame(color, i))
    assert output.strategy and output.strategy.cycle_triggered
    assert e.state.value == "ACTIVE_CYCLE"
    assert p.handle_frame(_frame(30, 7)).strategy is None
    assert p.metrics.report()["processing_ms"] >= 0

def test_capture_gap_cancels_active_strategy_cycle():
    e = StrategyEngine(StrategyConfig())
    e.process_round("trigger", 10)
    events = []
    p = VisualPipeline(ScreenCapture(CaptureConfig(1, 0, 0, 20, 20), FakeBackend()),
                       MultiplierRecognizer(lambda _: ("", 0)), RoundDetector(), e,
                       on_event=events.append)
    p._resync("capture failure")
    assert e.state.value == "WAITING_FOR_YELLOW"
    assert events[-1].kind == DetectionEventType.MULTIPLIER_UNCERTAIN
