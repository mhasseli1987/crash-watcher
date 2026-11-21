"""ورود هوشمند به بازی: دورهٔ یادگیری، تصمیمِ آماده و سنجش دقت مدل.

ترتیب زمانی بازی این است: ضریب دست جاری نمایش داده می‌شود، بعد حدود شش ثانیه
فرصت هست تا برای دست بعد شرط بسته شود. یک تماس با مدل چند ثانیه طول می‌کشد،
پس اگر تصمیم را همان‌جا بگیریم پنجره از دست می‌رود. بنابراین تصمیم همیشه
*پیش از* بسته‌شدن ضریب جاری و در رشتهٔ جدا آماده می‌شود؛ سر بسته‌شدن ضریب
فقط یک کلیک لازم است.

دورهٔ یادگیری: تا پایان این مدت هیچ شرطی بسته نمی‌شود؛ ضریب‌ها و رنگ‌ها ثبت
می‌شوند و پیش‌بینی مدل با نتیجهٔ واقعی سنجیده می‌شود. بعد از آن، فقط وقتی
احتمال اعلامی مدل و دقت اندازه‌گیری‌شده‌اش از آستانه بالاتر باشد وارد بازی
می‌شود؛ در غیر این صورت دست را رد می‌کند.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import monotonic

from .smart_review import SmartVerdict

LEARNING = "learning"
TRADING = "trading"


@dataclass(frozen=True)
class SmartDecision:
    """Bet set for the next hand, prepared while the current one is running."""

    bets: tuple[str, ...]
    confidence: float
    reason: str = ""
    prepared_after: str | None = None  # round id the snapshot ended with
    cashout: float | None = None  # ضریب خروج پیشنهادی مدل برای همین دست

    @property
    def target(self) -> str:
        return "+".join(self.bets)


class SmartPlanner:
    """یادگیری، آماده‌سازی تصمیم دست بعد و اندازه‌گیری دقت مدل."""

    def __init__(self, learning_minutes: int = 30, min_confidence: float = 0.40,
                 allow_two_colors: bool = True, clock=monotonic,
                 max_stale_rounds: int = 3) -> None:
        self._clock = clock
        self._started = clock()
        self.learning_seconds = max(0, int(learning_minutes)) * 60
        self.min_confidence = float(min_confidence)
        self.allow_two_colors = bool(allow_two_colors)
        # (round id, color) of the rounds seen this session, newest last.
        self.rounds: deque[tuple[str, str]] = deque(maxlen=300)
        self._ready: SmartDecision | None = None
        self._prepared_at = -1  # عدد دورها در لحظهٔ آماده‌شدن تصمیم
        self.max_stale_rounds = max(1, int(max_stale_rounds))
        self._latest: SmartDecision | None = None
        self._latest_scored = True
        self.hits = 0
        self.tries = 0
        self.break_even = 0

    # -- phase ---------------------------------------------------------
    @property
    def phase(self) -> str:
        return (LEARNING if self._clock() - self._started < self.learning_seconds
                else TRADING)

    @property
    def learning(self) -> bool:
        return self.phase == LEARNING

    @property
    def remaining_minutes(self) -> float:
        elapsed = self._clock() - self._started
        return max(0.0, self.learning_seconds - elapsed) / 60

    def update(self, learning_minutes: int, min_confidence: float,
               allow_two_colors: bool) -> None:
        """تنظیمات تازه را بی‌آنکه ساعت دورهٔ یادگیری صفر شود اعمال کن."""
        self.learning_seconds = max(0, int(learning_minutes)) * 60
        self.min_confidence = float(min_confidence)
        self.allow_two_colors = bool(allow_two_colors)

    # -- prepare / consume ---------------------------------------------
    def store(self, verdict: SmartVerdict | None, prepared_after: str | None) -> None:
        """نتیجهٔ مدل را نگه دار تا سر بسته‌شدن ضریبِ بعدی استفاده شود."""
        if verdict is None or verdict.bet == "SKIP":
            self._ready = None
            return
        bets = verdict.bets
        if not self.allow_two_colors:
            bets = bets[:1]
        if not bets:
            self._ready = None
            return
        decision = SmartDecision(bets, verdict.confidence, verdict.reason, prepared_after,
                                 verdict.cashout)
        self._ready = decision
        self._prepared_at = len(self.rounds)
        self._latest = decision
        self._latest_scored = False

    @property
    def measured_accuracy(self) -> float | None:
        return self.hits / self.tries if self.tries else None

    def decision(self) -> SmartDecision | None:
        """تصمیم آمادهٔ دست بعد؛ None یعنی این دست بازی نکن."""
        if self.phase == LEARNING or self._ready is None:
            return None
        if not self._ready.bets or self._ready.confidence < self.min_confidence:
            return None
        if len(self.rounds) - self._prepared_at >= self.max_stale_rounds:
            # پاسخ مدل خیلی دیر رسیده؛ داده‌اش برای دست فعلی کهنه است.
            return None
        accuracy = self.measured_accuracy
        if accuracy is not None and self.tries >= 20 and accuracy < self.min_confidence:
            # مدل خودش هم به اندازهٔ کافی درست پیش‌بینی نکرده؛ ریسک نکن.
            return None
        return self._ready

    def take(self) -> SmartDecision | None:
        """تصمیم آماده را یک‌بار بردار؛ بعد از آن تازه‌سازی لازم است."""
        decision = self.decision()
        self._ready = None
        return decision

    # -- observe -------------------------------------------------------
    def observe(self, round_id: str, color: str) -> None:
        """ثبت دور تازه و سنجش آخرین پیش‌بینی با همین نتیجه."""
        if not self._latest_scored and self._latest is not None:
            self._latest_scored = True
            if self._latest.bets:
                self.tries += 1
                if color == self._latest.bets[0]:
                    self.hits += 1
                elif color in self._latest.bets[1:]:
                    self.hits += 1  # پوشش دو رنگ: برد یا سربه‌سر، نه باخت
                    self.break_even += 1
        self.rounds.append((round_id, color))

    def stats(self) -> dict:
        return {"phase": self.phase, "learning_left": self.remaining_minutes,
                "hits": self.hits, "tries": self.tries, "break_even": self.break_even,
                "accuracy": self.measured_accuracy}
