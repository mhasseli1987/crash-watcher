"""Sequential confirmed-round state machine; no external side effects."""
from collections.abc import Callable
from datetime import datetime, timezone
from .config import StrategyConfig, positive_number
from .enums import BetOutcome, CycleOutcome, EventType, MultiplierCategory, StrategyState
from .events import StrategyEvent
from .models import RoundResult

class StrategyEngine:
    """Process rounds serially. Duplicate history survives manual reset."""
    def __init__(self, config: StrategyConfig, *, clock: Callable[[], datetime] | None = None) -> None:
        if not isinstance(config, StrategyConfig):
            raise ValueError("config must be StrategyConfig")
        self._config = config
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._state = StrategyState.WAITING_FOR_YELLOW
        self._attempt = 0
        self._remaining = 0
        self._cycle_id = 0
        self._seen: set[str] = set()

    @property
    def config(self) -> StrategyConfig:
        return self._config

    @property
    def state(self) -> StrategyState:
        return self._state

    @property
    def attempt_number(self) -> int:
        return self._attempt

    @property
    def remaining_attempts(self) -> int:
        return self._remaining

    def _clear_cycle(self) -> None:
        self._state = StrategyState.WAITING_FOR_YELLOW
        self._attempt = self._remaining = 0

    def reset(self) -> StrategyEvent:
        """Cancel active cycle without allowing any old ID to be replayed."""
        event = StrategyEvent(EventType.CYCLE_RESET, self._clock(), None, self._cycle_id, self._attempt)
        self._clear_cycle()
        return event

    def process_round(self, round_id: str, multiplier: object) -> RoundResult:
        """Reject invalid input atomically; return immutable results and ordered events.

        Terminal results retain the executed attempt number; engine counters reset
        immediately. Remaining attempts always reflect the post-round engine state.
        Whitespace surrounding IDs is normalized. Multipliers must be positive.
        """
        if not isinstance(round_id, str) or not round_id.strip():
            raise ValueError("round_id must be a nonempty string")
        round_id = round_id.strip()
        value = positive_number(multiplier, "multiplier")
        category = (MultiplierCategory.YELLOW if value >= self.config.yellow_threshold else
                    MultiplierCategory.RED if value < self.config.red_threshold else MultiplierCategory.GREEN)
        timestamp = self._clock()
        previous = self.state
        events: list[StrategyEvent] = []
        attempt = self._attempt
        def emit(kind: EventType) -> None:
            events.append(StrategyEvent(kind, timestamp, round_id, self._cycle_id, attempt))
        duplicate = round_id in self._seen
        triggered = registered = False
        bet = BetOutcome.NONE
        cycle = CycleOutcome.ACTIVE if self.state == StrategyState.ACTIVE_CYCLE else CycleOutcome.NONE
        if duplicate:
            emit(EventType.DUPLICATE_ROUND_IGNORED)
        elif self.state == StrategyState.WAITING_FOR_YELLOW:
            if category == MultiplierCategory.YELLOW:
                self._cycle_id += 1
                self._state = StrategyState.ACTIVE_CYCLE
                self._remaining = self.config.max_attempts
                triggered = True
                cycle = CycleOutcome.ACTIVE
                emit(EventType.YELLOW_TRIGGER_DETECTED)
                emit(EventType.CYCLE_STARTED)
        else:
            registered = True
            attempt = self._attempt + 1
            self._attempt = attempt
            self._remaining -= 1
            emit(EventType.SIMULATED_BET_REGISTERED)
            if category == MultiplierCategory.YELLOW:
                bet, cycle = BetOutcome.WON, CycleOutcome.WON
                emit(EventType.SIMULATED_BET_WON)
                emit(EventType.CYCLE_WON)
            else:
                bet = BetOutcome.LOST
                emit(EventType.SIMULATED_BET_LOST)
                if self._remaining == 0:
                    cycle = CycleOutcome.EXPIRED
                    emit(EventType.CYCLE_EXPIRED)
            if cycle in (CycleOutcome.WON, CycleOutcome.EXPIRED):
                self._clear_cycle()
                emit(EventType.CYCLE_RESET)
        self._seen.add(round_id)
        return RoundResult(round_id, value, category, previous, self.state, triggered,
                           registered, self.config.fixed_bet_amount if registered else None,
                           attempt, self._remaining, bet, cycle, timestamp, self._cycle_id,
                           duplicate, tuple(events))
