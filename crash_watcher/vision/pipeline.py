"""Nonblocking capture worker and Phase 1 adapter."""
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from threading import Event, Thread
from time import perf_counter
import cv2
import numpy as np
from crash_watcher.core.models import RoundResult
from crash_watcher.core.strategy_engine import StrategyEngine
from crash_watcher.core.enums import StrategyState
from .capture import ScreenCapture, ScreenUnavailable
from .detection import RoundDetector
from .models import DetectionEvent, DetectionEventType, DetectionUpdate, Frame, Recognition, now_utc
from .models import DetectionState
from .recognition import MultiplierRecognizer

class Metrics:
    def __init__(self) -> None:
        self.capture_ms: deque[float] = deque(maxlen=1000)
        self.processing_ms: deque[float] = deque(maxlen=1000)
        self.ocr_ms: deque[float] = deque(maxlen=1000)
        self.detection_ms: deque[float] = deque(maxlen=1000)

    def report(self) -> dict[str, float]:
        return {name: round(sum(values) / len(values), 2) if values else 0.0
                for name, values in (("capture_ms", self.capture_ms), ("processing_ms", self.processing_ms),
                                     ("ocr_ms", self.ocr_ms), ("detection_ms", self.detection_ms))}

@dataclass(frozen=True)
class PipelineOutput:
    detection: DetectionUpdate
    strategy: RoundResult | None

class VisualPipeline:
    def __init__(self, capture: ScreenCapture, recognizer: MultiplierRecognizer,
                 detector: RoundDetector, strategy: StrategyEngine,
                 on_output: Callable[[PipelineOutput], None] | None = None,
                 on_event: Callable[[DetectionEvent], None] | None = None) -> None:
        self.capture, self.recognizer, self.detector, self.strategy = capture, recognizer, detector, strategy
        self.on_output, self.on_event = on_output, on_event
        self.metrics = Metrics()
        self._previous: np.ndarray | None = None
        self._previous_recognition: Recognition | None = None
        self._submitted: set[str] = set()
        self._stop = Event()
        self._thread: Thread | None = None

    def _emit(self, event: DetectionEvent) -> None:
        if self.on_event:
            self.on_event(event)

    def _resync(self, reason: str) -> None:
        """Discard a cycle when capture gaps could have hidden completed rounds."""
        self.detector.reset()
        self._previous = None
        self._previous_recognition = None
        if self.strategy.state == StrategyState.ACTIVE_CYCLE:
            self.strategy.reset()
            self._emit(DetectionEvent(DetectionEventType.MULTIPLIER_UNCERTAIN, now_utc(),
                                      f"strategy cycle reset after {reason}"))

    def handle_frame(self, frame: Frame) -> PipelineOutput:
        """One frame; call serially. OCR is skipped while image changes rapidly."""
        begin = perf_counter()
        gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
        thumb = cv2.resize(gray, (64, 32), interpolation=cv2.INTER_AREA)
        moving = self._previous is not None and float(np.mean(cv2.absdiff(thumb, self._previous))) >= self.detector.config.pixel_difference
        self._previous = thumb
        recognition: Recognition | None = None
        if not moving and self._previous_recognition is not None and self.detector.state in (DetectionState.WAITING, DetectionState.RESULT_CONFIRMED):
            recognition = self._previous_recognition
        elif not moving:
            recognition = self.recognizer.recognize(frame.image)
            self.metrics.ocr_ms.append(recognition.ocr_ms)
            self._previous_recognition = recognition
        else:
            self._previous_recognition = None
        detection = self.detector.inspect(frame, recognition)
        self.metrics.capture_ms.append(frame.capture_ms)
        self.metrics.processing_ms.append((perf_counter() - begin) * 1000)
        result = None
        if detection.completed is not None:
            completed = detection.completed
            if completed.round_id in self._submitted:
                self._emit(DetectionEvent(DetectionEventType.DUPLICATE_RESULT_IGNORED, frame.captured_at, round_id=completed.round_id))
            else:
                result = self.strategy.process_round(completed.round_id, completed.multiplier)
                self._submitted.add(completed.round_id)
                self.metrics.detection_ms.append(completed.detection_latency_ms)
        for event in detection.events:
            self._emit(event)
        output = PipelineOutput(detection, result)
        if self.on_output:
            self.on_output(output)
        return output

    def _run(self) -> None:
        try:
            with self.capture:
                self._emit(DetectionEvent(DetectionEventType.SCREEN_CAPTURE_STARTED, now_utc()))
                while not self._stop.is_set():
                    started = perf_counter()
                    try:
                        frame, events = self.capture.capture()
                        for event in events:
                            self._emit(event)
                        if events:
                            self._resync("screen geometry change")
                        self.handle_frame(frame)
                    except ScreenUnavailable as exc:
                        self._resync("capture failure")
                        self._emit(DetectionEvent(DetectionEventType.SCREEN_CAPTURE_FAILED, now_utc(), str(exc)))
                    except Exception as exc:
                        self._resync("detection error")
                        self._emit(DetectionEvent(DetectionEventType.DETECTION_ERROR, now_utc(), str(exc)))
                    self._stop.wait(max(0, self.capture.config.interval_ms / 1000 - (perf_counter() - started)))
        except Exception as exc:
            self._emit(DetectionEvent(DetectionEventType.SCREEN_CAPTURE_FAILED, now_utc(), str(exc)))

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("pipeline already running")
        self._stop.clear()
        self._thread = Thread(target=self._run, name="crash-watcher-capture", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                raise RuntimeError("capture worker did not stop")
