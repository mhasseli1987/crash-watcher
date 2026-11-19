"""Conservative visual round lifecycle; no value is emitted without re-arming."""
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from time import perf_counter
from uuid import uuid4
import cv2
import numpy as np
from .models import (CompletedRound, DetectionEvent, DetectionEventType, DetectionState,
                     DetectionUpdate, Frame, Recognition)

@dataclass(frozen=True)
class DetectorConfig:
    stable_frames: int = 3
    motion_frames: int = 2
    min_round_ms: int = 700
    pixel_difference: float = 3.0

    def __post_init__(self) -> None:
        if (type(self.stable_frames) is not int or self.stable_frames < 2 or
            type(self.motion_frames) is not int or self.motion_frames < 1 or
            type(self.min_round_ms) is not int or self.min_round_ms < 0 or
            not np.isfinite(self.pixel_difference) or self.pixel_difference <= 0):
            raise ValueError("invalid detector configuration")

class RoundDetector:
    """Requires observed result disappearance/motion then a stable OCR result.

    A startup result is deliberately ignored. Identical successive multipliers
    are allowed only after a fresh visible in-progress transition.
    """
    def __init__(self, config: DetectorConfig = DetectorConfig()) -> None:
        self.config = config
        self.state = DetectionState.WAITING
        self._previous: np.ndarray | None = None
        self._shape: tuple[int, ...] | None = None
        self._motion_count = 0
        self._stable_count = 0
        self._candidate: Decimal | None = None
        self._started_at: datetime | None = None
        self._last_result: Decimal | None = None
        self._emitted = False
        self._blank_seen = False

    def reset(self) -> None:
        self.__init__(self.config)

    def inspect(self, frame: Frame | None, recognition: Recognition | None) -> DetectionUpdate:
        start = perf_counter()
        events: list[DetectionEvent] = []
        if frame is None:
            self._stable_count = 0
            self._candidate = None
            self.state = DetectionState.UNCERTAIN
            return DetectionUpdate(self.state, (), processing_ms=(perf_counter()-start)*1000)
        if self._shape is not None and frame.image.shape != self._shape:
            self.reset()
            events.append(DetectionEvent(DetectionEventType.SCREEN_REGION_CHANGED, frame.captured_at, "frame dimensions changed"))
        self._shape = frame.image.shape
        gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
        thumb = cv2.resize(gray, (64, 32), interpolation=cv2.INTER_AREA)
        difference = (float(np.mean(cv2.absdiff(thumb, self._previous))) if self._previous is not None else 0.0)
        self._previous = thumb
        moving = difference >= self.config.pixel_difference
        value = recognition.multiplier if recognition is not None and not recognition.uncertain else None
        completed = None
        if self.state in (DetectionState.WAITING, DetectionState.RESULT_CONFIRMED, DetectionState.UNCERTAIN):
            if moving or (value is None and self._last_result is not None):
                self._motion_count += 1
            else:
                self._motion_count = 0
            if self._motion_count >= self.config.motion_frames:
                self.state = DetectionState.ROUND_IN_PROGRESS
                self._started_at = frame.captured_at
                self._candidate = None
                self._stable_count = 0
                self._emitted = False
                self._blank_seen = False
                events.append(DetectionEvent(DetectionEventType.ROUND_STARTED, frame.captured_at))
        elif self.state in (DetectionState.ROUND_IN_PROGRESS, DetectionState.RESULT_CANDIDATE):
            if moving or value is None:
                if not moving and recognition is not None and value is None:
                    self._blank_seen = True
                self._candidate = None
                self._stable_count = 0
                self.state = DetectionState.ROUND_IN_PROGRESS
                if value is None and recognition is not None:
                    events.append(DetectionEvent(DetectionEventType.MULTIPLIER_UNCERTAIN, frame.captured_at, recognition.text))
            else:
                elapsed = (frame.captured_at - self._started_at).total_seconds() * 1000 if self._started_at else 0
                if elapsed >= self.config.min_round_ms and self._blank_seen:
                    self._stable_count = self._stable_count + 1 if value == self._candidate else 1
                    self._candidate = value
                    self.state = DetectionState.RESULT_CANDIDATE
                    if self._stable_count == 1:
                        events.append(DetectionEvent(DetectionEventType.MULTIPLIER_CANDIDATE, frame.captured_at, multiplier=value))
                    if self._stable_count >= self.config.stable_frames and not self._emitted:
                        round_id = f"visual-{uuid4().hex}"
                        completed = CompletedRound(round_id, value, recognition.category, frame.captured_at, elapsed)
                        self._last_result = value
                        self._emitted = True
                        self.state = DetectionState.RESULT_CONFIRMED
                        events.extend((DetectionEvent(DetectionEventType.MULTIPLIER_CONFIRMED, frame.captured_at, round_id=round_id, multiplier=value),
                                       DetectionEvent(DetectionEventType.ROUND_COMPLETED, frame.captured_at, round_id=round_id, multiplier=value)))
        return DetectionUpdate(self.state, tuple(events), completed, recognition, (perf_counter()-start)*1000)
