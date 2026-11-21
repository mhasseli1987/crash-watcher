"""استاپ‌لاس سرمایه: حساب سادهٔ برد/باخت روی همان تصمیم‌های کلیکر.

هر پیش‌بینی که سنجیده می‌شود یک شرط با مبلغ ثابت است. اگر رنگ اصلی بیاید،
سود برابر payout−1 ضرب در مبلغ است؛ اگر رنگ دوم (پوشش) بیاید شرط سربه‌سر
بسته می‌شود؛ در غیر این صورت همان مبلغ از دست می‌رود. وقتی موجودی به کفِ
تعیین‌شده برسد، ربات تا شروع دوبارهٔ جلسه دیگر شرط نمی‌بندد. هیچ پولی جابه‌جا
نمی‌شود؛ این فقط یک سنجهٔ ریسک است.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Bankroll:
    start: float
    stop_loss_fraction: float = 0.20
    balance: float = field(init=False)
    realized: float = field(default=0.0, init=False)
    wins: int = field(default=0, init=False)
    losses: int = field(default=0, init=False)
    break_even: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.start = max(0.0, float(self.start))
        self.stop_loss_fraction = min(1.0, max(0.0, float(self.stop_loss_fraction)))
        self.balance = self.start

    @property
    def floor(self) -> float:
        return self.start * (1.0 - self.stop_loss_fraction)

    @property
    def stopped(self) -> bool:
        """آیا کفِ استاپ‌لاس رد شده است؟"""
        return self.start > 0 and self.balance <= self.floor

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

    def label(self) -> str:
        if self.start <= 0:
            return "بدون سرمایه"
        state = "توقف استاپ‌لاس" if self.stopped else "فعال"
        return (f"{self.balance:.2f} از {self.start:.2f} "
                f"(کف {self.floor:.2f}) — {state}")
