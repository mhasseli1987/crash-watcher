"""Immutable, validated configuration."""
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

def positive_number(value: object, name: str) -> Decimal:
    """Accept finite positive int, float, or Decimal values only."""
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ValueError(f"{name} must be a positive number")
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"Invalid {name}") from exc
    if not result.is_finite() or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result

@dataclass(frozen=True)
class StrategyConfig:
    fixed_bet_amount: Decimal = Decimal("10")
    max_attempts: int = 10
    yellow_threshold: Decimal = Decimal("10")
    red_threshold: Decimal = Decimal("2")

    def __post_init__(self) -> None:
        for name in ("fixed_bet_amount", "yellow_threshold", "red_threshold"):
            object.__setattr__(self, name, positive_number(getattr(self, name), name))
        if type(self.max_attempts) is not int or self.max_attempts <= 0:
            raise ValueError("max_attempts must be a positive integer")
        if self.red_threshold >= self.yellow_threshold:
            raise ValueError("red_threshold must be below yellow_threshold")
