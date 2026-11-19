"""Forecast the next round's color from the colors already seen.

The game draws independent rounds, so this module claims nothing mystical: it
estimates the chance of the next color from the recent color mix (base rate) and
from what usually follows the previous color (conditional rate, shrunk towards
the base rate so a handful of coincidences cannot dominate). The clicker only
acts when that estimate reaches the configured confidence, otherwise it sits out.

Every forecast is scored against the round that actually followed, so the log
shows whether the model is worth anything or is just tracking the base rate.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

COLORS = ("RED", "GREEN", "YELLOW")


@dataclass(frozen=True)
class Forecast:
    """The most likely next color and how much evidence stands behind it."""

    color: str
    probability: float
    samples: int
    model: str

    @property
    def label(self) -> str:
        return f"{self.color} {self.probability:.0%} ({self.model}, n={self.samples})"


class ColorForecaster:
    """Rolling base-rate + previous-color model with its own accuracy record."""

    def __init__(self, window: int = 300, min_samples: int = 40, shrink: float = 20.0) -> None:
        self.window = window
        self.min_samples = min_samples
        self.shrink = shrink
        self._colors: deque[str] = deque(maxlen=window)
        self._after: dict[str, Counter] = {color: Counter() for color in COLORS}
        self._last: str | None = None
        # Rolling hit record of both models, one 1/0 per scored round.
        self._scores = {"base": deque(maxlen=40), "cond": deque(maxlen=40)}

    def reset(self) -> None:
        self._colors.clear()
        self._after = {color: Counter() for color in COLORS}
        self._last = None
        self._scores = {"base": deque(maxlen=40), "cond": deque(maxlen=40)}

    @property
    def samples(self) -> int:
        return len(self._colors)

    def _base(self) -> dict[str, float]:
        counts = Counter(self._colors)
        total = len(self._colors)
        return {color: (counts[color] + 1.0) / (total + len(COLORS)) for color in COLORS}

    def _conditional(self) -> dict[str, float]:
        base = self._base()
        if self._last is None:
            return base
        following = self._after[self._last]
        total = sum(following.values())
        if total < 8:
            return base
        weight = total / (total + self.shrink)
        shrunk = {color: (following[color] + self.shrink * base[color]) / (total + self.shrink)
                  for color in COLORS}
        return {color: weight * shrunk[color] + (1 - weight) * base[color] for color in COLORS}

    @staticmethod
    def _mean(scores: deque) -> float:
        return sum(scores) / len(scores) if scores else 0.0

    def conditional_is_trusted(self) -> bool:
        """Only prefer the conditional model when it has out-predicted the base rate."""
        cond, base = self._scores["cond"], self._scores["base"]
        return len(cond) >= 20 and len(base) >= 20 and self._mean(cond) > self._mean(base)

    def probabilities(self) -> dict[str, float]:
        return self._conditional() if self.conditional_is_trusted() else self._base()

    def forecast(self) -> Forecast:
        probabilities = self.probabilities()
        color = max(probabilities, key=probabilities.get)
        model = "previous color" if self.conditional_is_trusted() else "base rate"
        return Forecast(color, probabilities[color], len(self._colors), model)

    def observe(self, color: str) -> None:
        """Record a finished round and score the forecast made before it."""
        if color not in COLORS:
            raise ValueError(f"unknown color {color!r}")
        if len(self._colors) >= self.min_samples:
            base = self._base()
            cond = self._conditional()
            self._scores["base"].append(int(max(base, key=base.get) == color))
            self._scores["cond"].append(int(max(cond, key=cond.get) == color))
        if self._last is not None:
            self._after[self._last][color] += 1
        self._last = color
        self._colors.append(color)

    def review(self) -> dict:
        base, cond = self._mean(self._scores["base"]), self._mean(self._scores["cond"])
        return {
            "samples": len(self._colors),
            "base_accuracy": base,
            "cond_accuracy": cond,
            "base_scored": len(self._scores["base"]),
            "cond_scored": len(self._scores["cond"]),
            "model": "previous color" if self.conditional_is_trusted() else "base rate",
        }
