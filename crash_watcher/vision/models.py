"""Typed observations and detection events."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
import numpy as np
from crash_watcher.core.enums import MultiplierCategory

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

class DetectionState(str, Enum):
    WAITING = "WAITING"
    ROUND_IN_PROGRESS = "ROUND_IN_PROGRESS"
    RESULT_CANDIDATE = "RESULT_CANDIDATE"
    RESULT_CONFIRMED = "RESULT_CONFIRMED"
    UNCERTAIN = "UNCERTAIN"

class DetectionEventType(str, Enum):
    SCREEN_CAPTURE_STARTED = "SCREEN_CAPTURE_STARTED"
    SCREEN_CAPTURE_FAILED = "SCREEN_CAPTURE_FAILED"
    SCREEN_REGION_CHANGED = "SCREEN_REGION_CHANGED"
    MULTIPLIER_CANDIDATE = "MULTIPLIER_CANDIDATE"
    MULTIPLIER_CONFIRMED = "MULTIPLIER_CONFIRMED"
    MULTIPLIER_UNCERTAIN = "MULTIPLIER_UNCERTAIN"
    ROUND_STARTED = "ROUND_STARTED"
    ROUND_COMPLETED = "ROUND_COMPLETED"
    DUPLICATE_RESULT_IGNORED = "DUPLICATE_RESULT_IGNORED"
    DETECTION_ERROR = "DETECTION_ERROR"

@dataclass(frozen=True)
class DetectionEvent:
    kind: DetectionEventType
    timestamp: datetime
    detail: str = ""
    round_id: str | None = None
    multiplier: Decimal | None = None

@dataclass(frozen=True)
class Recognition:
    multiplier: Decimal | None
    category: MultiplierCategory | None
    confidence: float
    text: str
    uncertain: bool
    color_category: MultiplierCategory | None = None
    latency_ms: float = 0.0
    ocr_ms: float = 0.0

@dataclass(frozen=True)
class Frame:
    image: np.ndarray = field(repr=False, compare=False)
    captured_at: datetime = field(default_factory=now_utc)
    capture_ms: float = 0.0

@dataclass(frozen=True)
class CompletedRound:
    round_id: str
    multiplier: Decimal
    category: MultiplierCategory
    confirmed_at: datetime
    detection_latency_ms: float

@dataclass(frozen=True)
class DetectionUpdate:
    state: DetectionState
    events: tuple[DetectionEvent, ...]
    completed: CompletedRound | None = None
    recognition: Recognition | None = None
    processing_ms: float = 0.0
