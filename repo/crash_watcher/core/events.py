"""Immutable GUI-independent events."""
from dataclasses import dataclass
from datetime import datetime
from .enums import EventType

@dataclass(frozen=True)
class StrategyEvent:
    event_type: EventType
    timestamp: datetime
    round_id: str | None
    cycle_id: int
    attempt_number: int
