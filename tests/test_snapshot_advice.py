from pathlib import Path
import cv2
import numpy as np
import pytest
from crash_watcher.desktop import snapshot_advice as advice
from crash_watcher.desktop.config import DesktopSettings
from crash_watcher.desktop.color_grid import read_grid

ASSETS = Path(__file__).resolve().parents[1] / "assets"


def test_guide_examples_and_overflow():
    image = cv2.imread(str(ASSETS / "trend-chart-guide.png"))
    first = read_grid(image[99:204, :112])
    second = read_grid(image[235:338, :112])
    decode = lambda g: advice.chart_sequence(tuple((d.column, d.row, d.color) for d in g.dots))
    assert decode(first) == ("GREEN", "GREEN", "GREEN", "RED", "GREEN", "YELLOW")
    assert decode(second) == ("GREEN", "YELLOW", "GREEN", "GREEN", "GREEN", "GREEN",
                              "GREEN", "RED", "RED", "GREEN")


def test_supplied_images():
    from crash_watcher.vision.recognition import RapidOCRReader
    snapshot = advice.read_snapshot(cv2.imread(str(ASSETS / "advice-history.png")),
                                    cv2.imread(str(ASSETS / "advice-chart.png")), RapidOCRReader())
    assert snapshot.recent == (("1.04", "RED"), ("3.19", "GREEN"), ("3.70", "GREEN"),
                               ("2.66", "GREEN"), ("1.92", "RED"))
    assert len(snapshot.sequence) == len(snapshot.cells) == 40
    assert snapshot.sequence.count("YELLOW") == 3
    assert snapshot.sequence[-2:] == ("GREEN", "YELLOW")


@pytest.mark.parametrize("raw", ['{}', '{"color":"BLUE"}',
    '{"color":"GREEN","reason":5}'])
def test_invalid_answers_rejected(raw):
    with pytest.raises(ValueError):
        advice.parse_advice(raw)


@pytest.mark.parametrize("raw, color", [
    ('{"color":"GREEN","multiplier":1.9}', "GREEN"),
    ('{"color":"RED","multiplier":NaN}', "RED"),
    ('{"color":"RED","multiplier":true}', "RED"),
    ('{"color":"YELLOW","multiplier":5}', "YELLOW")])
def test_bad_multiplier_never_rejects_the_colour(raw, color):
    """ضریب مدل در کلیک/کش‌اوت استفاده نمی‌شود؛ ناسازگاری آن رنگ را باطل نمی‌کند."""
    parsed, value, _, _ = advice.parse_advice(raw)
    assert parsed == color and value is None


def test_unknown_and_estimate():
    assert advice.parse_advice('{"color":"UNKNOWN","multiplier":null}')[1] is None
    assert advice.parse_advice('{"color":"YELLOW","multiplier":12.5}')[1] == 12.5


@pytest.mark.parametrize("changed", [True, False])
def test_one_request_after_read_and_stale_result(monkeypatch, changed):
    worker = advice.SnapshotAdviceWorker(DesktopSettings())
    before = [np.zeros((2, 2, 3), dtype=np.uint8)] * 2
    after = [np.ones((2, 2, 3), dtype=np.uint8)] * 2 if changed else before
    captures = iter([before, after])
    monkeypatch.setattr(worker, "capture_regions", lambda: next(captures))
    monkeypatch.setattr(advice, "RapidOCRReader", lambda: None)
    events, results, errors, advice_events = [], [], [], []
    def read(*args):
        events.append("read")
        return advice.HistorySnapshot((("1.04", "RED"),), (), ())
    def ask(*args):
        events.append("ask")
        return '{"color":"UNKNOWN","multiplier":null}'
    monkeypatch.setattr(advice, "read_snapshot", read)
    monkeypatch.setattr(advice, "request_advice", ask)
    worker.done.connect(results.append)
    worker.failed.connect(errors.append)
    worker.advice.connect(lambda color, value, confidence: advice_events.append((color, value)))
    worker.run()
    assert events == ["read", "ask"]
    assert advice_events == [("UNKNOWN", None)]
    assert not errors and len(results) == 1
    assert ("قدیمی است" in results[0]) == changed


def test_model_confidence_and_persian_colour_are_accepted():
    """مدل باید احتمال بدهد؛ درصد و نام فارسی رنگ هم پذیرفته می‌شود."""
    color, value, _, confidence = advice.parse_advice(
        '{"color":"سبز","confidence":"45%","multiplier":2.4,"reason":"روند"}')
    assert (color, value, confidence) == ("GREEN", 2.4, 0.45)
    assert advice.parse_advice('{"color":"RED"}')[3] == 0.0
    assert advice.parse_advice('{"color":"red","confidence":1.2}')[3] == 1.0


def test_local_model_is_asked_before_the_cloud_route(monkeypatch):
    """حتی در حالت hybrid تحلیل از مدل محلی گرفته می‌شود."""
    from dataclasses import replace
    settings = replace(DesktopSettings(), ai_provider="hybrid", ollama_enabled=True,
                       smart_review_enabled=True)
    calls = []
    monkeypatch.setattr(advice, "_ask_local",
                        lambda *args: calls.append("local") or '{"color":"RED"}')
    monkeypatch.setattr(advice, "ask", lambda *args, **kwargs: calls.append("cloud") or "{}")
    assert advice.request_advice(settings, "prompt") == '{"color":"RED"}'
    assert calls == ["local"]


def test_low_confidence_advice_never_arms_a_click():
    """زیر ۴۰٪ شرط بسته نمی‌شود؛ از ۴۰٪ به بالا رنگ برای دست جدید ثبت می‌شود."""
    from types import SimpleNamespace
    from crash_watcher.desktop.ui import MainWindow
    submitted, texts = [], []
    window = SimpleNamespace(
        settings=DesktopSettings(),
        snapshot_output=SimpleNamespace(appendPlainText=texts.append),
        worker=SimpleNamespace(isRunning=lambda: True,
                               submit_advice=lambda color, value=None: submitted.append(color)))
    MainWindow._snapshot_advice_ready(window, "RED", 1.5, 0.39)
    assert submitted == [] and "کمتر" in texts[-1]
    MainWindow._snapshot_advice_ready(window, "RED", 1.5, 0.40)
    assert submitted == ["RED"]


def test_red_streak_is_counted_from_the_newest_rounds():
    reds = advice.HistorySnapshot((("1.20", "RED"),), (),
                                 ("GREEN", "RED", "RED", "RED", "RED"))
    assert advice.red_run(reds) == 4
    changed = advice.HistorySnapshot((("1.20", "RED"), ("2.50", "GREEN")), (),
                                     ("RED", "GREEN"))
    assert advice.red_run(changed) == 0
    extended = advice.HistorySnapshot((("1.10", "RED"),), (), ("RED", "RED"))
    assert advice.red_run(extended) == 2


def test_red_streak_snapshot_skips_the_model(monkeypatch):
    """چهار قرمز پیاپی: مدل پرسیده نمی‌شود و توقف یک‌دقیقه‌ای اعلام می‌شود."""
    # توقف قرمزهای پیاپی قاعدهٔ بازی رنگ ترنبال است (در کلاسیک قرمز نیمی از
    # دست‌هاست و ربات را یک‌دست‌درمیان می‌خواباند).
    worker = advice.SnapshotAdviceWorker(DesktopSettings(bot_mode="turnbal"))
    frames = [np.zeros((2, 2, 3), dtype=np.uint8)] * 2
    monkeypatch.setattr(worker, "capture_regions", lambda: frames)
    monkeypatch.setattr(advice, "RapidOCRReader", lambda: None)
    monkeypatch.setattr(advice, "warm_local", lambda *args: True)
    monkeypatch.setattr(advice, "read_snapshot", lambda *args: advice.HistorySnapshot(
        (("1.10", "RED"),), (), ("RED", "RED", "RED", "RED")))
    monkeypatch.setattr(advice, "request_advice",
                        lambda *args: pytest.fail("model must not be asked during a red run"))
    paused, advice_events = [], []
    worker.paused.connect(paused.append)
    worker.advice.connect(lambda *args: advice_events.append(args))
    worker.run()
    assert advice_events == [] and "60" in paused[0] and "قرمز" in paused[0]


def test_advice_reaches_the_click_decision(tmp_path):
    """پیشنهاد مدل باید همان کلیک شرط دست بعد را بسازد، با تنظیمات کلیک."""
    from dataclasses import replace
    from decimal import Decimal
    from crash_watcher import StrategyEngine
    from crash_watcher.desktop.clicker import AutoClicker
    from crash_watcher.desktop.config import Region
    from crash_watcher.desktop.worker import MonitorWorker
    base = replace(DesktopSettings(), result_region=Region(1, 0, 0, 20, 10),
                   bot_mode="turnbal", autoclick_enabled=True, cashout_enabled=False,
                   autoclick_dry_run=False, shadow_mode=False, bet_point_green=(70, 70),
                   autoclick_delay_ms=0, smart_play_learning_minutes=0,
                   autoclick_threshold="10.00")
    worker = MonitorWorker(base, StrategyEngine(base.strategy()))
    worker.submit_advice("GREEN")
    assert worker._take_advice() == (("GREEN",), None)
    assert worker._take_advice() is None  # یک‌بار مصرف
    classic = replace(base, bot_mode="classic")
    classic_worker = MonitorWorker(classic, StrategyEngine(classic.strategy()))
    classic_worker.submit_advice("GREEN")
    assert classic_worker._take_advice() == (("PLAY",), None)
    worker.submit_advice("UNKNOWN")
    assert worker._take_advice() is None
    sent = []
    clicker = AutoClicker(base, sender=lambda x, y, restore=True: sent.append((x, y)) or True,
                          log_path=tmp_path / "rounds.jsonl",
                          summary_path=tmp_path / "summary.txt",
                          report_path=tmp_path / "report.txt")
    clicker.set_smart_play(("GREEN",), None)
    report = clicker.fire("r1", Decimal("1.50"))
    assert report.clicked and report.rule == "smart-play"
    assert sent == [(70, 70)]


def test_chart_column_that_cannot_be_reached_is_left_out():
    """ستون ناقص/درحال‌کشیده‌شدن، کل خواندن جدول را باطل نمی‌کند."""
    cells = ((0, 5, "GREEN"), (1, 0, "RED"), (1, 1, "RED"), (2, 0, "GREEN"))
    assert advice.chart_sequence(cells) == ("RED", "RED", "GREEN")


def _fake_grid():
    from types import SimpleNamespace
    return SimpleNamespace(dots=[SimpleNamespace(column=0, row=0, color="RED")])


def test_any_number_of_recent_multipliers_is_accepted(monkeypatch):
    """نوار ضرایب ثابت پنج‌تایی نیست: هر تعداد کارت دیده شود خوانده می‌شود."""
    history = np.zeros((39, 300, 3), dtype=np.uint8)
    dots = [(5, 16, "GREEN"), (70, 16, "RED"), (135, 16, "GREEN"), (200, 16, "RED")]
    monkeypatch.setattr(advice, "_centres",
                        lambda image, color, area: [(x, y) for x, y, c in dots if c == color])
    monkeypatch.setattr(advice, "read_grid", lambda image: _fake_grid())
    values = iter(["3.46x", "1.14x", "3.08x", "1.72x"])
    snapshot = advice.read_snapshot(history, np.zeros((10, 10, 3), dtype=np.uint8),
                                    lambda crop: (next(values), 0.99))
    assert snapshot.recent == (("3.46", "GREEN"), ("1.14", "RED"),
                               ("3.08", "GREEN"), ("1.72", "RED"))
    assert snapshot.sequence == ("RED",)


def test_unreadable_edge_card_is_skipped_not_fatal(monkeypatch):
    history = np.zeros((39, 220, 3), dtype=np.uint8)
    dots = [(5, 16, "GREEN"), (70, 16, "GREEN"), (135, 16, "RED")]
    monkeypatch.setattr(advice, "_centres",
                        lambda image, color, area: [(x, y) for x, y, c in dots if c == color])
    monkeypatch.setattr(advice, "read_grid", lambda image: _fake_grid())
    values = iter([("", 0.0), ("2.06x", 0.95), ("1.10x", 0.97)])
    snapshot = advice.read_snapshot(history, np.zeros((10, 10, 3), dtype=np.uint8),
                                    lambda crop: next(values))
    assert snapshot.recent == (("2.06", "GREEN"), ("1.10", "RED"))


def test_read_with_retry_recovers_from_a_moving_strip(monkeypatch):
    worker = advice.SnapshotAdviceWorker(DesktopSettings())
    frames = [np.zeros((2, 2, 3), dtype=np.uint8)] * 2
    monkeypatch.setattr(worker, "capture_regions", lambda: frames)
    monkeypatch.setattr(advice, "RapidOCRReader", lambda: None)
    calls = []
    def read(*args):
        calls.append(1)
        if len(calls) == 1:
            raise ValueError("moving")
        return advice.HistorySnapshot((("1.04", "RED"),), (), ())
    monkeypatch.setattr(advice, "read_snapshot", read)
    monkeypatch.setattr(advice, "sleep", lambda delay: None)
    _, snapshot = worker.read_with_retry()
    assert len(calls) == 2 and snapshot.recent == (("1.04", "RED"),)


def test_failed_read_never_calls_model(monkeypatch):
    worker = advice.SnapshotAdviceWorker(DesktopSettings())
    monkeypatch.setattr(worker, "capture_regions", lambda: [None, None])
    monkeypatch.setattr(advice, "RapidOCRReader", lambda: None)
    def fail(*args):
        raise ValueError("unreadable")
    monkeypatch.setattr(advice, "read_snapshot", fail)
    monkeypatch.setattr(advice, "request_advice", lambda *args: pytest.fail("unexpected network call"))
    errors = []
    worker.failed.connect(errors.append)
    worker.run()
    assert errors == ["unreadable"]


def test_watcher_debounces_each_new_history(monkeypatch):
    worker = advice.SnapshotWatchWorker(DesktopSettings())
    # A transient animation is ignored; both subsequent stable rounds count.
    values = iter([0, 1, 0, 2, 2, 2, 3, 3])
    class Stop:
        ticks = 0
        def is_set(self):
            return self.ticks == 8
        def wait(self, delay):
            self.ticks += 1
    worker.stopped = Stop()
    monkeypatch.setattr(worker, "capture_regions", lambda: [np.full((2, 2, 3), next(values), dtype=np.uint8)])
    changes = []
    worker.changed.connect(lambda: changes.append(True))
    worker.run()
    assert len(changes) == 2


def test_changed_history_while_busy_is_queued(monkeypatch):
    from types import SimpleNamespace
    from crash_watcher.desktop.ui import MainWindow
    texts = []
    window = SimpleNamespace(_snapshot_pending=False,
        snapshot_output=SimpleNamespace(setPlainText=texts.append),
        snapshot_worker=SimpleNamespace(isRunning=lambda: True),
        _snapshot_finished=lambda: pytest.fail("must not start parallel requests"))
    MainWindow._snapshot_changed(window)
    MainWindow._snapshot_changed(window)
    assert window._snapshot_pending
    MainWindow._snapshot_done(window, "old recommendation")
    assert "old recommendation" not in texts
