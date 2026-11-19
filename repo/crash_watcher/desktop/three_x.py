"""موتور ورود ۳x: فقط وقتی وارد می‌شود که احتمال رسیدن دور بعد به آستانه کافی باشد.

قواعد این موتور جای حدس‌های «شکل رنگ‌ها» را می‌گیرد:

* ابتدا یک ساعت مشاهده و حداقل ۱۰۰ دور کامل؛ تا آن زمان هیچ ورودی بسته نمی‌شود.
* احتمال با نمونهٔ مشابه سنجیده می‌شود: دورهایی که ضریب قبلی‌شان در همان بازهٔ
  ضریب دور جاری بوده است. کمتر از ۲۰ نمونهٔ مشابه یعنی ورود ممنوع.
* نقطهٔ سربه‌سر ۱۰۰/ضریب خروج است (۳x ← ۳۳٫۳٪)، پس آستانهٔ ورود ۴۰٪ است.
* هر سیگنال یک ورود، حداقل ۵ دور فاصله بین دو ورود.
* مبلغ شرط ثابت و حداکثر ۰٫۲۵٪ موجودی؛ فیبوناچی و جبران ضرر ندارد.
* سه باخت پشت‌سرهم ← ۳۰ دور توقف؛ سود یا ضرر ۲٪ جلسه ← توقف کامل.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from decimal import Decimal
from time import monotonic

# بازه‌های ضریب که «رفتار بعد از هر بازه» بر اساس آن‌ها سنجیده می‌شود.
BUCKETS = ((Decimal("1"), Decimal("1.5")), (Decimal("1.5"), Decimal("2")),
           (Decimal("2"), Decimal("3")), (Decimal("3"), Decimal("5")),
           (Decimal("5"), None))
BUCKET_LABELS = ("1–1.5", "1.5–2", "2–3", "3–5", "5+")

LEARNING = "learning"
TRADING = "trading"
HALTED = "halted"


def bucket_of(value: Decimal) -> str:
    """نام بازهٔ ضریب؛ بالاترین بازه باز است."""
    for (low, high), label in zip(BUCKETS, BUCKET_LABELS):
        if value < low:
            continue
        if high is None or value < high:
            return label
    return BUCKET_LABELS[0]


@dataclass(frozen=True)
class ThreeXDecision:
    """یک تصمیم ورود/صبر برای دست بعد."""

    bet: bool
    reason: str
    probability: float = 0.0
    samples: int = 0
    stake: Decimal = Decimal("0")
    detail: str = ""


class ThreeXSession:
    """آمار ساعت، تخمین احتمال و مدیریت موجودی یک جلسه."""

    def __init__(self, settings, clock=monotonic) -> None:
        self._clock = clock
        self._started = clock()
        self._rounds: deque[Decimal] = deque(maxlen=6000)
        self._round_times: deque[float] = deque(maxlen=6000)
        self.update(settings)
        self._pending_stake: Decimal | None = None
        self._last_entry_index = -10**9
        self._cooldown_until = 0
        self._consecutive_losses = 0
        self.started_balance = Decimal(str(settings.three_x_bankroll))
        self.balance = self.started_balance
        self.entries = 0
        self.wins = 0
        self.losses = 0
        self.stake_won = Decimal("0")
        self.stake_lost = Decimal("0")
        self.halted_reason = ""
        self._last_detail = ""

    # -- settings ------------------------------------------------------
    def update(self, settings) -> None:
        self.settings = settings
        self.observation_seconds = max(0, int(settings.three_x_observation_minutes)) * 60
        self.min_rounds = max(0, int(settings.three_x_min_rounds))
        self.min_probability = float(settings.three_x_probability)
        self.min_samples = max(1, int(settings.three_x_min_samples))
        self.min_ocr = float(settings.three_x_min_ocr)
        self.exit_factor = Decimal(str(settings.three_x_exit))
        self.stake_pct = float(settings.three_x_stake_pct)
        self.profit_limit = float(settings.three_x_profit_limit_pct)
        self.loss_limit = float(settings.three_x_loss_limit_pct)
        self.stop_losses = max(1, int(settings.three_x_stop_losses))
        self.stop_rounds = max(0, int(settings.three_x_stop_rounds))
        self.min_gap_rounds = max(1, int(settings.three_x_min_gap_rounds))

    # -- phase ---------------------------------------------------------
    @property
    def phase(self) -> str:
        if self.halted_reason:
            return HALTED
        elapsed = self._clock() - self._started
        if elapsed < self.observation_seconds or len(self._rounds) < self.min_rounds:
            return LEARNING
        return TRADING

    @property
    def learning(self) -> bool:
        return self.phase == LEARNING

    @property
    def halted(self) -> bool:
        return bool(self.halted_reason)

    @property
    def remaining_minutes(self) -> float:
        elapsed = self._clock() - self._started
        return max(0.0, self.observation_seconds - elapsed) / 60

    @property
    def break_even(self) -> float:
        return float(1 / self.exit_factor) if self.exit_factor else 0.0

    # -- statistics ----------------------------------------------------
    def _window(self) -> list[Decimal]:
        """دورهای یک ساعت اخیر (یا از شروع، اگر کمتر گذشته باشد)."""
        now = self._clock()
        span = max(3600.0, float(self.observation_seconds))
        recent = [value for value, moment in zip(self._rounds, self._round_times)
                  if now - moment <= span]
        return recent or list(self._rounds)

    def estimate(self, confidence: float | None = None) -> tuple[float, int, float]:
        """(احتمال شرطی رسیدن به آستانه، نمونهٔ مشابه، نرخ پایهٔ ساعت)."""
        window = self._window()
        total = len(window)
        if total < 2:
            return 0.0, 0, 0.0
        above = sum(1 for value in window if value >= self.exit_factor)
        base = (above + 1) / (total + 2)
        current = bucket_of(window[-1])
        similar = hits = 0
        for previous, following in zip(window, window[1:]):
            if bucket_of(previous) == current:
                similar += 1
                if following >= self.exit_factor:
                    hits += 1
        # نمونهٔ کم با نرخ پایه نرم می‌شود تا چند تصادف، احتمال نسازد.
        probability = ((hits + 2 * base) / (similar + 2)) if similar else base
        return probability, similar, base

    def stake(self) -> Decimal:
        """مبلغ ثابت: ۰٫۲۵٪ موجودی شروع، هرگز بیشتر از ۰٫۲۵٪ موجودی فعلی."""
        cap = self.balance * Decimal(str(self.stake_pct)) / Decimal("100")
        flat = self.started_balance * Decimal(str(self.stake_pct)) / Decimal("100")
        value = min(flat, cap)
        return value.quantize(Decimal("0.01")) if value > 0 else Decimal("0")

    def _longest_below(self, window: list[Decimal]) -> int:
        longest = run = 0
        for value in window:
            if value < self.exit_factor:
                run += 1
                longest = max(longest, run)
            else:
                run = 0
        return longest

    @staticmethod
    def _hit_rate(values: list[Decimal], exit_factor: Decimal) -> float:
        return (sum(1 for value in values if value >= exit_factor) / len(values)
                if values else 0.0)

    def stats(self) -> dict:
        """خلاصهٔ ساعت برای گزارش جلسه و تحلیل بعدی."""
        window = self._window()
        exit_factor = self.exit_factor
        recent20 = self._hit_rate(window[-20:], exit_factor)
        overall = self._hit_rate(window, exit_factor)
        after: dict[str, dict] = {}
        for previous, following in zip(window, window[1:]):
            label = bucket_of(previous)
            entry = after.setdefault(label, {"rounds": 0, "above": 0})
            entry["rounds"] += 1
            if following >= exit_factor:
                entry["above"] += 1
        return {
            "phase": self.phase,
            "rounds": len(window),
            "session_rounds": len(self._rounds),
            "above_share": overall,
            "longest_below": self._longest_below(window),
            "recent20": recent20,
            "recent50": self._hit_rate(window[-50:], exit_factor),
            "recent100": self._hit_rate(window[-100:], exit_factor),
            "delta_vs_hour": recent20 - overall,
            "after_bucket": after,
            "balance": self.balance,
            "started_balance": self.started_balance,
            "pnl_pct": ((self.balance - self.started_balance) / self.started_balance * 100
                        if self.started_balance else 0.0),
            "entries": self.entries,
            "wins": self.wins,
            "losses": self.losses,
            "consecutive_losses": self._consecutive_losses,
            "cooldown_left": max(0, self._cooldown_until - len(self._rounds)),
            "halted": self.halted_reason,
            "learning_minutes": self.remaining_minutes,
        }

    def report_lines(self) -> list[str]:
        stats = self.stats()
        lines = [
            f"موتور ۳x — مرحله: {stats['phase']} | دور ساعت: {stats['rounds']} | "
            f"موجودی: {stats['balance']:.2f} ({stats['pnl_pct']:+.2f}٪)",
            f"سهم ضرایب ≥{self.exit_factor}: {stats['above_share']:.0%} | "
            f"بلندترین زنجیرهٔ زیر {self.exit_factor}: {stats['longest_below']}",
            f"عملکرد ۲۰/۵۰/۱۰۰ دور اخیر: {stats['recent20']:.0%}/{stats['recent50']:.0%}/"
            f"{stats['recent100']:.0%} | تفاوت ۲۰ دور با ساعت: {stats['delta_vs_hour']:+.0%}",
        ]
        for label in BUCKET_LABELS:
            entry = stats["after_bucket"].get(label)
            if entry and entry["rounds"]:
                lines.append(f"  بعد از {label}: {entry['above']}/{entry['rounds']} "
                             f"({entry['above'] / entry['rounds']:.0%} بالای {self.exit_factor})")
        lines.append(f"ورود: {stats['entries']} | برد: {stats['wins']} | باخت: {stats['losses']}"
                     + (f" | توقف: {stats['halted']}" if stats["halted"] else ""))
        return lines

    # -- betting -------------------------------------------------------
    def _check_limits(self) -> None:
        if not self.started_balance:
            return
        pnl = (self.balance - self.started_balance) / self.started_balance * 100
        if pnl >= self.profit_limit:
            self.halted_reason = f"حد سود جلسه ({pnl:+.2f}٪) — توقف تا جلسهٔ بعد"
        elif pnl <= -self.loss_limit:
            self.halted_reason = f"حد ضرر جلسه ({pnl:+.2f}٪) — توقف تا جلسهٔ بعد"

    def observe(self, value: Decimal, confidence: float | None = None) -> str:
        """ثبت دور تازه و تسویهٔ شرطی که روی همین دور بسته شده بود."""
        self._rounds.append(value)
        self._round_times.append(self._clock())
        outcome = ""
        if self._pending_stake is not None:
            stake = self._pending_stake
            self._pending_stake = None
            if value >= self.exit_factor:
                profit = stake * (self.exit_factor - 1)
                self.balance += profit
                self.stake_won += profit
                self.wins += 1
                self._consecutive_losses = 0
                outcome = f"برد ({value}×، +{profit:.2f})"
            else:
                self.balance -= stake
                self.stake_lost += stake
                self.losses += 1
                self._consecutive_losses += 1
                outcome = f"باخت ({value}×، -{stake:.2f})"
                if self._consecutive_losses >= self.stop_losses:
                    self._cooldown_until = len(self._rounds) + self.stop_rounds
                    self._consecutive_losses = 0
                    outcome += f" — {self.stop_rounds} دور توقف"
            self._check_limits()
        return outcome

    def decide(self, confidence: float | None = None) -> ThreeXDecision:
        """تصمیم برای دست بعد: ورود فقط با احتمال و نمونهٔ کافی."""
        if self.halted:
            return ThreeXDecision(False, self.halted_reason)
        if self.phase == LEARNING:
            missing = max(0, self.min_rounds - len(self._rounds))
            return ThreeXDecision(
                False,
                f"مشاهده: {self.remaining_minutes:.0f} دقیقه و {missing} دور مانده")
        if confidence is not None and confidence < self.min_ocr:
            return ThreeXDecision(False, f"اطمینان OCR {confidence:.0%} < {self.min_ocr:.0%}")
        gap = len(self._rounds) - self._last_entry_index
        if gap < self.min_gap_rounds:
            return ThreeXDecision(False, f"فاصلهٔ ورود: {self.min_gap_rounds - gap} دور")
        if len(self._rounds) < self._cooldown_until:
            return ThreeXDecision(False, f"توقف بعد از باخت‌ها: "
                                         f"{self._cooldown_until - len(self._rounds)} دور")
        probability, samples, base = self.estimate(confidence)
        detail = (f"احتمال {probability:.0%} از {samples} نمونهٔ مشابه "
                  f"(پایه {base:.0%})")
        if samples < self.min_samples:
            return ThreeXDecision(False, f"نمونهٔ مشابه کم: {samples}/{self.min_samples}",
                                  probability, samples, detail=detail)
        if probability < self.min_probability:
            return ThreeXDecision(False, f"{probability:.0%} < {self.min_probability:.0%}",
                                  probability, samples, detail=detail)
        stake = self.stake()
        if stake <= 0:
            return ThreeXDecision(False, "موجودی کافی نیست", probability, samples)
        self._pending_stake = stake
        self._last_entry_index = len(self._rounds)
        self.entries += 1
        return ThreeXDecision(True, "ورود ۳x", probability, samples, stake,
                              detail + f" | شرط {stake:.2f}")
