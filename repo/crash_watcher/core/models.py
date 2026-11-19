"""Immutable result returned for every valid round submission."""
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from .enums import BetOutcome, CycleOutcome, MultiplierCategory, StrategyState
from .events import StrategyEvent

@dataclass(frozen=True)
class RoundResult:
    round_id: str
    multiplier: Decimal
    category: MultiplierCategory
    previous_state: StrategyState
    current_state: StrategyState
    cycle_triggered: bool
    bet_registered: bool
    bet_amount: Decimal | None
    attempt_number: int
    remaining_attempts: int
    bet_outcome: BetOutcome
    cycle_outcome: CycleOutcome
    timestamp: datetime
    cycle_id: int
    duplicate: bool
    events: tuple[StrategyEvent, ...]
