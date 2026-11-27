"""حالت ترنبال: تصمیمِ مدل، محدودشده با کالیبراسیون شرط سه‌رنگ.

در حالت کلاسیک قواعد خود ربات تصمیم می‌گیرند. در حالت ترنبال، همان مدلِ
«بررسی هوشمند» ضریب‌ها و رنگ‌ها و شبکهٔ نمودار را تحلیل می‌کند، ولی اینجا
پاسخ مدل پیش از رسیدن به کلیکر از فیلتر کالیبراسیون هر رنگ می‌گذرد:

* رنگ باید در کالیبراسیون فعال باشد؛
* احتمال اعلامی مدل از حداقلِ همان رنگ کمتر نباشد؛
* ضریب خروج فقط داخل بازهٔ اعلام‌شده (پیش‌فرض ۱٫۴۰ تا ۱٫۹۰) بسته شود.

هیچ‌کدام از این قواعد جای مدل را نمی‌گیرد؛ فقط جلوی شرطِ خارج از کالیبراسیون
را می‌گیرد. تنظیمات در settings.json ذخیره می‌شوند.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .smart_review import (EXIT_MAX, EXIT_MIN, SmartSnapshot, SmartVerdict,
                           ask, parse_verdict)

COLORS = ("RED", "GREEN", "YELLOW")
NAMES = {"RED": "قرمز", "GREEN": "سبز", "YELLOW": "زرد (مون)"}


@dataclass(frozen=True)
class ColorRule:
    enabled: bool
    min_conf: float
    max_attempts: int


@dataclass(frozen=True)
class TurnbalCalibration:
    """کالیبراسیون شرط سه‌رنگ؛ از تنظیمات ذخیره‌شده ساخته می‌شود."""

    red: ColorRule = ColorRule(True, 0.55, 4)
    green: ColorRule = ColorRule(True, 0.60, 6)
    yellow: ColorRule = ColorRule(False, 0.75, 2)
    cashout_min: float = EXIT_MIN
    cashout_max: float = EXIT_MAX
    min_cycle_win: float = 0.50
    max_streak: int = 3
    min_gap: int = 2
    emergency_exit: float = EXIT_MIN

    def rule(self, color: str) -> ColorRule:
        return {"RED": self.red, "GREEN": self.green,
                "YELLOW": self.yellow}.get(color, ColorRule(False, 1.0, 1))

    @property
    def enabled_colors(self) -> tuple[str, ...]:
        return tuple(color for color in COLORS if self.rule(color).enabled)

    def cashout(self, value: float | None) -> float:
        """ضریب خروج را داخل بازهٔ کالیبراسیون بسته کن."""
        chosen = self.cashout_max if value is None else float(value)
        return round(min(max(chosen, self.cashout_min), self.cashout_max), 2)

    @classmethod
    def from_settings(cls, settings) -> "TurnbalCalibration":
        return cls(
            red=ColorRule(settings.turnbal_red_enabled, settings.turnbal_red_min_conf,
                          settings.turnbal_red_attempts),
            green=ColorRule(settings.turnbal_green_enabled, settings.turnbal_green_min_conf,
                            settings.turnbal_green_attempts),
            yellow=ColorRule(settings.turnbal_yellow_enabled, settings.turnbal_yellow_min_conf,
                             settings.turnbal_yellow_attempts),
            cashout_min=float(settings.turnbal_cashout_min),
            cashout_max=float(settings.turnbal_cashout_max),
            min_cycle_win=float(settings.turnbal_min_cycle_win),
            max_streak=int(settings.turnbal_max_streak),
            min_gap=int(settings.turnbal_min_gap),
            emergency_exit=float(settings.turnbal_emergency_exit),
        )


SYSTEM_PROMPT = (
    "تو موتور پیش‌بینی رنگ یک بازی سه‌شرط (Trenball) هستی. ورودی، شبکهٔ رنگ "
    "نمودار است. فقط رنگ دست بعد را پیش‌بینی کن؛ ضریب‌ها و خروج هیچ نقشی در "
    "تصمیم تو ندارند. پاسخ را همیشه در قالب JSON بده."
)


def _rules_text(calibration: TurnbalCalibration, threshold: str) -> list[str]:
    del threshold  # اثری در تصمیم رنگ ندارد
    lines = ["کالیبراسیون شرط سه‌رنگ (فقط رنگ‌های زیر مجازند):"]
    for color in COLORS:
        rule = calibration.rule(color)
        if rule.enabled:
            lines.append(f"- {NAMES[color]}: حداقل احتمال {rule.min_conf:.0%} | "
                         f"حداکثر {rule.max_attempts} تلاش در چرخه")
        else:
            lines.append(f"- {NAMES[color]}: غیرفعال — پیشنهادش پذیرفته نمی‌شود")
    lines += [
        f"- اگر نرخ برد اندازه‌گیری‌شده از {calibration.min_cycle_win:.0%} کمتر شد، "
        "محافظه‌کارتر شو",
        f"- بیش از {calibration.max_streak} دست پشت‌سرهم شرط نبند؛ بین دو شرط دست‌کم "
        f"{calibration.min_gap} دست فاصله بگذار",
        "- لازم نیست هر دست بازی کنی؛ دست‌های بی‌اطمینان را SKIP کن",
    ]
    return lines


def build_prompt(snapshot: SmartSnapshot, calibration: TurnbalCalibration,
                 threshold: str = "10.00", extra_rules: str = "",
                 *, allow_two: bool = True) -> str:
    """پرسمان کامل ترنبال؛ رنگ‌ها و حداقل‌ها را صریح می‌گوید."""
    from .smart_review import _grid_text  # همان متن شبکهٔ رنگ بخش بررسی هوشمند

    bets = "، ".join(f"{NAMES[name]} ({payout})" for name, payout in snapshot.bets)
    rules = _rules_text(calibration, threshold)
    if allow_two:
        rules.append("- اگر بین دو رنگ شک داری می‌توانی هر دو را بزنی و رنگ دوم را در "
                     "second بگذار؛ شرط روی دو رنگ با پرداخت دوبرابر سربه‌سر می‌شود")
    if extra_rules.strip():
        rules.append("- " + extra_rules.strip())
    return (
        "وضعیت فعلی بازی:\n"
        "۱) شبکهٔ رنگ نمودار (هر ستون از بالا جدید به پایین قدیمی):\n"
        f"{_grid_text(snapshot)}\n"
        f"۲) دکمه‌های شرط این دست: {bets}\n"
        f"تعداد دورهای دیده‌شده در این جلسه: {snapshot.rounds_seen}\n\n"
        "قواعد و کالیبراسیون:\n" + "\n".join(rules) + "\n\n"
        "برای دست بعد کدام رنگ بیشترین شانس را دارد؟ فقط رنگ و احتمال را بده؛ "
        "ضریب و خروج لازم نیست. فقط JSON برگردان:\n"
        + ('{"bet":"RED|GREEN|YELLOW|SKIP","second":"NONE|RED|GREEN|YELLOW",'
           '"confidence":0.0,"reason":"یک جمله فارسی"}' if allow_two else
           '{"bet":"RED|GREEN|YELLOW|SKIP","confidence":0.0,'
           '"reason":"یک جمله فارسی"}')
    )


@dataclass
class TurnbalGuard:
    """اجرای واقعی محافظ‌های کالیبراسیون در کد، نه فقط در پرامپت.

    * `max_streak`: بیش از این تعداد شرط پشت‌سرهم بسته نمی‌شود؛
    * `min_gap`: بعد از هر شرط دست‌کم این تعداد دست صبر می‌شود؛
    * `*_attempts`: هر رنگ در هر چرخه فقط تا این تعداد تلاش می‌شود و با
      بردِ همان رنگ شمارنده‌های چرخه صفر می‌شوند؛
    * `min_cycle_win`: وقتی نرخ برد اندازه‌گیری‌شده از این عدد کمتر شد،
      تا پایان جلسه هیچ شرط تازه‌ای بسته نمی‌شود.
    """

    calibration: TurnbalCalibration
    min_samples: int = 20
    _streak: int = field(default=0, init=False)
    # اولین شرطِ جلسه فاصله نمی‌خواهد؛ فقط بین دو شرط شمرده می‌شود.
    _gap: int = field(default=10 ** 6, init=False)
    _attempts: dict[str, int] = field(default_factory=dict, init=False)
    _wins: int = field(default=0, init=False)
    _tries: int = field(default=0, init=False)
    _pending: tuple[str, ...] | None = field(default=None, init=False)

    # -- state ---------------------------------------------------------
    def observe(self, color: str) -> None:
        """دست تازه بسته شد: شرط قبلی را بسنج و شمارنده‌ها را جلو ببر."""
        if self._pending is not None:
            bets, self._pending = self._pending, None
            won = bool(bets) and color == bets[0]
            self._tries += 1
            self._wins += 1 if won else 0
            if bets:
                self._attempts[bets[0]] = self._attempts.get(bets[0], 0) + 1
            if won:
                self._attempts.clear()  # چرخهٔ تازه بعد از برد
            self._streak += 1
            self._gap = 0
        else:
            self._streak = 0
            self._gap += 1

    def begin(self, bets) -> None:
        """شرطی که همین حالا واقعاً بسته شد را ثبت کن (بعد از کلیک)."""
        self._pending = tuple(color for color in (bets or ()) if color)

    # -- checks --------------------------------------------------------
    @property
    def win_rate(self) -> float | None:
        return self._wins / self._tries if self._tries else None

    def check(self, verdict: SmartVerdict | None) -> tuple[SmartVerdict | None, str]:
        """آیا این تصمیم با محافظ‌های کالیبراسیون سازگار است؟"""
        if verdict is None or verdict.bet == "SKIP" or not verdict.bets:
            return verdict, ""
        rate = self.win_rate
        if (rate is not None and self._tries >= self.min_samples
                and rate < self.calibration.min_cycle_win):
            return None, (f"نرخ برد {rate:.0%} کمتر از کف "
                          f"{self.calibration.min_cycle_win:.0%} است")
        if self._streak >= self.calibration.max_streak:
            return None, (f"سقف {self.calibration.max_streak} شرط پشت‌سرهم پر است؛ "
                          "دست‌کم یک دست صبر")
        if self._gap < self.calibration.min_gap:
            return None, (f"باید {self.calibration.min_gap} دست بین دو شرط فاصله باشد "
                          f"({self._gap} دست گذشته)")
        primary = verdict.bets[0]
        rule = self.calibration.rule(primary)
        used = self._attempts.get(primary, 0)
        if used >= rule.max_attempts:
            return None, (f"سهمیهٔ {rule.max_attempts} تلاش "
                          f"{NAMES.get(primary, primary)} در این چرخه پر شده است")
        return verdict, ""

    def cap_cashout(self, value: float | None) -> float:
        """هدف خروج هرگز از خروج اضطراری دیرتر نباشد."""
        target = self.calibration.cashout(value)
        ceiling = float(self.calibration.emergency_exit)
        return round(max(self.calibration.cashout_min, min(target, ceiling)), 2)

    def stats(self) -> dict:
        return {"win_rate": self.win_rate, "tries": self._tries, "streak": self._streak,
                "gap": self._gap, "attempts": dict(self._attempts)}


def calibrate(verdict: SmartVerdict | None,
              calibration: TurnbalCalibration) -> tuple[SmartVerdict | None, str]:
    """پاسخ مدل را با کالیبراسیون سه‌رنگ تطبیق بده.

    خروجی: (تصمیم مجاز یا None، دلیلِ رد). رنگِ اصلیِ غیرفعال یا کم‌اطمینان کل
    تصمیم را باطل می‌کند؛ رنگ دوم فقط اگر خودش مجاز باشد نگه داشته می‌شود.
    """
    if verdict is None:
        return None, "پاسخ مدل خوانده نشد"
    if verdict.bet == "SKIP" or not verdict.bets:
        return None, "مدل این دست را رد کرد"
    primary = verdict.bets[0]
    rule = calibration.rule(primary)
    if not rule.enabled:
        return None, f"{NAMES.get(primary, primary)} در کالیبراسیون غیرفعال است"
    if verdict.confidence < rule.min_conf:
        return None, (f"احتمال {verdict.confidence:.0%} کمتر از حداقل "
                      f"{rule.min_conf:.0%} برای {NAMES.get(primary, primary)} است")
    second = None
    if len(verdict.bets) > 1:
        candidate = verdict.bets[1]
        other = calibration.rule(candidate)
        if other.enabled and verdict.confidence >= other.min_conf:
            second = candidate
    return (SmartVerdict(primary, verdict.confidence, verdict.reason, second,
                         calibration.cashout(verdict.cashout)),
            "")


def review(base_url: str, api_key: str, model: str, snapshot: SmartSnapshot,
           calibration: TurnbalCalibration, timeout: float = 45.0,
           threshold: str = "10.00", extra_rules: str = "",
           allow_two: bool = True) -> tuple[SmartVerdict | None, str, str]:
    """پرسمان ترنبال + تطبیق با کالیبراسیون: (تصمیم، پاسخ خام، پرسمان)."""
    prompt = build_prompt(snapshot, calibration, threshold, extra_rules,
                          allow_two=allow_two)
    raw = ask(base_url, api_key, model, prompt, timeout, SYSTEM_PROMPT)
    verdict, _reason = calibrate(parse_verdict(raw), calibration)
    return verdict, raw, prompt
