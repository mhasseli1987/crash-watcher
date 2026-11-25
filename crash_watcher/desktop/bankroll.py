"""سرمایهٔ جلسه: حساب سادهٔ برد/باخت روی همان تصمیم‌های کلیکر.

هر پیش‌بینی که سنجیده می‌شود یک شرط با مبلغ ثابت است. اگر رنگ اصلی بیاید،
سود برابر payout−1 ضرب در مبلغ است؛ اگر رنگ دوم (پوشش) بیاید شرط سربه‌سر
بسته می‌شود؛ در غیر این صورت همان مبلغ از دست می‌رود. علاوه بر کفِ درصدی،
حد سود و حد ضرر به مبلغ هم سنجیده می‌شوند: با رسیدن سود یا زیان جلسه به
آن عدد، ربات دیگر شرط تازه‌ای نمی‌بندد. هیچ پولی جابه‌جا نمی‌شود؛ این فقط
یک سنجهٔ ریسک است.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Bankroll:
    start: float
    stop_loss_fraction: float = 0.20
    profit_limit: float = 0.0
    loss_limit: float = 0.0
    balance: float = field(init=False)
    realized: float = field(default=0.0, init=False)
    wins: int = field(default=0, init=False)
    losses: int = field(default=0, init=False)
    break_even: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.start = max(0.0, float(self.start))
        self.stop_loss_fraction = min(1.0, max(0.0, float(self.stop_loss_fraction)))
        self.profit_limit = max(0.0, float(self.profit_limit))
        self.loss_limit = max(0.0, float(self.loss_limit))
        self.balance = self.start

    @property
    def floor(self) -> float:
        return self.start * (1.0 - self.stop_loss_fraction)

    @property
    def stop_loss_hit(self) -> bool:
        """آیا کفِ استاپ‌لاس رد شده است؟"""
        return self.start > 0 and self.balance <= self.floor

    @property
    def profit_hit(self) -> bool:
        """آیا سود جلسه به حد سود رسیده است؟"""
        return self.profit_limit > 0 and self.realized >= self.profit_limit

    @property
    def loss_hit(self) -> bool:
        """آیا زیان جلسه به حد ضرر رسیده است؟"""
        return self.loss_limit > 0 and self.realized <= -self.loss_limit

    @property
    def stopped(self) -> bool:
        """آیا یکی از سه حد جلسه رسیده است؟"""
        return self.stop_loss_hit or self.profit_hit or self.loss_hit

    def settle(self, stake: float, payout: float, won: bool | None) -> float:
        """یک شرط را تسویه کن و سود/زیان این دست را برگردان.

        ``won`` برابر True برد رنگ اصلی، False باخت و None پوشش رنگ دوم است.
        مبلغ صفر (حالت آزمایشی/سایه) هیچ شمارنده‌ای را تغییر نمی‌دهد.
        """
        stake = max(0.0, float(stake))
        if stake == 0:
            return 0.0
        if won is None:
            self.break_even += 1
            return 0.0
        if won:
            delta = stake * max(0.0, float(payout) - 1.0)
            self.wins += 1
        else:
            delta = -stake
            self.losses += 1
        self.balance += delta
        self.realized += delta
        return delta

    def state(self) -> str:
        """دلیل توقف جلسه یا «فعال»."""
        if self.profit_hit:
            return f"توقف حد سود ({self.profit_limit:.2f})"
        if self.loss_hit:
            return f"توقف حد ضرر ({self.loss_limit:.2f})"
        if self.stop_loss_hit:
            return "توقف استاپ‌لاس"
        return "فعال"

    def label(self) -> str:
        if self.start <= 0:
            return "بدون سرمایه"
        return (f"{self.balance:.2f} از {self.start:.2f} "
                f"(سود/زیان {self.realized:+.2f}) — {self.state()}")
