"""Strict final-result OCR and conservative visual confirmation."""
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
import re
from uuid import uuid4
import cv2
import numpy as np
from crash_watcher.vision.models import DetectionState
from crash_watcher.vision.recognition import category_for, preprocess
from crash_watcher.vision.recognition import parse_multiplier

# RapidOCR occasionally drops the small trailing multiplier glyph while still
# reading the distinctive, final-only ``Bang @`` prefix with high confidence.
# Keep the prefix and exactly two decimal places mandatory so live multipliers,
# payout labels and countdown text still cannot be accepted as final results.
_BANG = re.compile(r"^Bang\s*@\s*(\d{1,4}\.\d{2})\s*[xX×]?$", re.IGNORECASE | re.ASCII)
_HISTORY_ID = re.compile(r"\b(\d{6,12})\b")
_HISTORY_VALUE = re.compile(r"(?<![\d.])\d{1,4}\.\d{2}[xX×](?!\w)")
_HISTORY_VALUE_DROPPED_DOT = re.compile(r"(?<!\d)(\d{1,4})\s+(\d{2})[xX×](?!\w)")
_HISTORY_VALUE_COMPACT = re.compile(r"(?<!\d)(\d{1,4})(\d{2})[xX×](?!\w)")

def parse_bang(text: str) -> Decimal | None:
    match = _BANG.fullmatch(text.strip()) if isinstance(text, str) else None
    if not match:
        return None
    value = Decimal(match[1])
    return value if value > 0 else None

def parse_history_id(text: str) -> str | None:
    ids = _HISTORY_ID.findall(text)
    return ids[0] if len(ids) == 1 else None

def parse_history_entry(text: str) -> tuple[str, Decimal] | None:
    round_id = parse_history_id(text)
    values = _HISTORY_VALUE.findall(text)
    # At the site's small history-badge size RapidOCR can see the decimal dot
    # as whitespace (for example ``1 89x``). Only repair this exact shape: the
    # round id and trailing x remain mandatory and exactly two fractional
    # digits are required.
    if not values:
        dropped_dot = _HISTORY_VALUE_DROPPED_DOT.findall(text)
        values = [f"{whole}.{fraction}x" for whole, fraction in dropped_dot]
    if not values:
        compact = _HISTORY_VALUE_COMPACT.findall(text)
        values = [f"{whole}.{fraction}x" for whole, fraction in compact]
    if not round_id or len(values) != 1:
        return None
    value = parse_multiplier(values[0])
    return (round_id, value) if value is not None else None

@dataclass(frozen=True)
class FinalReading:
    value: Decimal | None
    confidence: float
    raw_text: str
    category: str | None
    uncertain: bool

class BangRecognizer:
    def __init__(self, reader, min_confidence: float = 0.80) -> None:
        self.reader = reader
        self.min_confidence = min_confidence

    def recognize(self, image: np.ndarray) -> FinalReading:
        readings = [self.reader(variant) for variant in preprocess(image)]
        accepted = [(parse_bang(text), text, confidence) for text, confidence in readings
                    if np.isfinite(confidence) and confidence >= self.min_confidence]
        valid = [(value, text, confidence) for value, text, confidence in accepted if value is not None]
        if not valid or len({value for value, _, _ in valid}) != 1:
            text, confidence = max(readings, key=lambda item: item[1], default=("", 0.0))
            return FinalReading(None, confidence, text, None, True)
        value, text, confidence = max(valid, key=lambda item: item[2])
        return FinalReading(value, confidence, text, category_for(value).value, False)

@dataclass(frozen=True)
class ConfirmedFinal:
    round_id: str
    multiplier: Decimal
    confidence: float
    timestamp: datetime

class BangDetector:
    """Requires a non-Bang phase then a stable Bang; ignores startup Bang."""
    def __init__(self, stable_frames: int = 3, absent_frames: int = 2, pixel_difference: float = 3.0) -> None:
        self.stable_frames = stable_frames
        self.absent_frames = absent_frames
        self.pixel_difference = pixel_difference
        self.reset()

    def reset(self) -> None:
        self.state = DetectionState.WAITING
        self._absent = 0
        self._stable = 0
        self._candidate: Decimal | None = None
        self._last: Decimal | None = None
        self._last_image: np.ndarray | None = None
        self._previous: np.ndarray | None = None
        self._last_history_id: str | None = None
        self._pending_history: tuple[Decimal, float, datetime] | None = None
        self._seen_ids: set[str] = getattr(self, "_seen_ids", set())

    def observe(self, image: np.ndarray, reading: FinalReading, timestamp: datetime,
                history_id: str | None = None, history_value: Decimal | None = None,
                require_history: bool = False) -> ConfirmedFinal | None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thumb = cv2.resize(gray, (64, 32), interpolation=cv2.INTER_AREA)
        moving = self._previous is not None and float(np.mean(cv2.absdiff(thumb, self._previous))) >= self.pixel_difference
        self._previous = thumb
        if reading.value is None:
            # Trenball adds the just-finished round to its history only after
            # the Bang screen changes to the countdown. Confirm a stable Bang
            # retained from the prior phase as soon as that history row lands.
            if (require_history and self._pending_history and history_id and
                    history_value == self._pending_history[0] and history_id not in self._seen_ids):
                value, confidence, final_timestamp = self._pending_history
                self._pending_history = None
                self._last = value
                self._last_image = thumb.copy()
                self._last_history_id = history_id
                self._seen_ids.add(history_id)
                self.state = DetectionState.RESULT_CONFIRMED
                return ConfirmedFinal(f"game-{history_id}", value, confidence, final_timestamp)
            self._absent += 1
            self._stable = 0
            self._candidate = None
            if self._absent >= self.absent_frames:
                self.state = DetectionState.ROUND_IN_PROGRESS
            return None
        if self.state not in (DetectionState.ROUND_IN_PROGRESS, DetectionState.RESULT_CANDIDATE, DetectionState.UNCERTAIN):
            return None
        if moving:
            self._stable = 0
            self._candidate = None
            return None
        self._stable = self._stable + 1 if reading.value == self._candidate else 1
        self._candidate = reading.value
        self.state = DetectionState.RESULT_CANDIDATE
        if self._stable < self.stable_frames:
            return None
        if require_history and (not history_id or history_value != reading.value):
            self._pending_history = (reading.value, reading.confidence, timestamp)
            self.state = DetectionState.UNCERTAIN
            return None
        if history_id and history_id in self._seen_ids:
            self.state = DetectionState.RESULT_CONFIRMED
            return None
        # Transient OCR failure cannot turn an unchanged old result into a new round.
        if self._last == reading.value and self._last_image is not None:
            same_image = float(np.mean(cv2.absdiff(thumb, self._last_image))) < self.pixel_difference
            if same_image and (not history_id or history_id == self._last_history_id):
                self.state = DetectionState.RESULT_CONFIRMED
                return None
        self._last = reading.value
        self._pending_history = None
        self._last_image = thumb.copy()
        self._last_history_id = history_id
        self._absent = 0
        self.state = DetectionState.RESULT_CONFIRMED
        round_id = f"game-{history_id}" if history_id else f"visual-{uuid4().hex}"
        self._seen_ids.add(round_id)
        if history_id:
            self._seen_ids.add(history_id)
        return ConfirmedFinal(round_id, reading.value, reading.confidence, timestamp)
