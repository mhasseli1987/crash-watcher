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

from dataclasses import dataclass

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
    "تو موتور تصمیم یک ربات Crash (Trenball) هستی. ورودی، ضریب آخرین دست‌ها و "
    "شبکهٔ رنگ نمودار است. فقط بر اساس همین داده و کالیبراسیون کاربر تحلیل کن، "
    "هیچ ادعای قطعی نکن و پاسخ را همیشه در قالب JSON بده."
)


def _rules_text(calibration: TurnbalCalibration, threshold: str) -> list[str]:
    lines = ["کالیبراسیون شرط سه‌رنگ (فقط رنگ‌های زیر مجازند):"]
    ranges = {"RED": f"کمتر از ۲٫۰۰", "GREEN": f"۲٫۰۰ تا {threshold}",
              "YELLOW": f"{threshold} و بالاتر"}
    for color in COLORS:
        rule = calibration.rule(color)
        if rule.enabled:
            lines.append(f"- {NAMES[color]}: بازهٔ ضریب {ranges[color]} | حداقل احتمال "
                         f"{rule.min_conf:.0%} | حداکثر {rule.max_attempts} تلاش در چرخه")
        else:
            lines.append(f"- {NAMES[color]}: غیرفعال — پیشنهادش پذیرفته نمی‌شود")
    lines += [
        f"- ضریب خروج (کش‌اوت) فقط بین {calibration.cashout_min:.2f} و "
        f"{calibration.cashout_max:.2f}",
        f"- اگر نرخ برد اندازه‌گیری‌شده از {calibration.min_cycle_win:.0%} کمتر شد، "
        "محافظه‌کارتر شو",
        f"- بیش از {calibration.max_streak} دست پشت‌سرهم شرط نبند؛ بین دو شرط دست‌کم "
        f"{calibration.min_gap} دست فاصله بگذار",
        f"- خروج اضطراری روی {calibration.emergency_exit:.2f} است؛ دیرتر از آن نمان",
        "- لازم نیست هر دست بازی کنی؛ دست‌های بی‌اطمینان را SKIP کن",
    ]
    return lines


def build_prompt(snapshot: SmartSnapshot, calibration: TurnbalCalibration,
                 threshold: str = "10.00", extra_rules: str = "",
                 *, allow_two: bool = True) -> str:
    """پرسمان کامل ترنبال؛ رنگ‌ها و حداقل‌ها را صریح می‌گوید."""
    from .smart_review import _grid_text  # همان متن شبکهٔ رنگ بخش بررسی هوشمند

    recent = "، ".join(snapshot.recent[-20:]) or "ندارد"
    bets = "، ".join(f"{NAMES[name]} ({payout})" for name, payout in snapshot.bets)
    rules = _rules_text(calibration, threshold)
    if allow_two:
        rules.append("- اگر بین دو رنگ شک داری می‌توانی هر دو را بزنی و رنگ دوم را در "
                     "second بگذار؛ شرط روی دو رنگ با پرداخت دوبرابر سربه‌سر می‌شود")
    if extra_rules.strip():
        rules.append("- " + extra_rules.strip())
    return (
        "وضعیت فعلی بازی:\n"
        f"۱) ضریب آخرین دست‌ها (قدیمی به جدید): {recent}\n"
        f"۲) ضریب دست فعلی/آخر: {snapshot.current or 'خوانده نشد'}\n"
        "۳) شبکهٔ رنگ نمودار (هر ستون از بالا جدید به پایین قدیمی):\n"
        f"{_grid_text(snapshot)}\n"
        f"۴) دکمه‌های شرط این دست: {bets}\n"
        f"تعداد دورهای دیده‌شده در این جلسه: {snapshot.rounds_seen}\n\n"
        "قواعد و کالیبراسیون:\n" + "\n".join(rules) + "\n\n"
        "برای دست بعد کدام رنگ بیشترین شانس را دارد؟ فقط JSON برگردان:\n"
        + ('{"bet":"RED|GREEN|YELLOW|SKIP","second":"NONE|RED|GREEN|YELLOW",'
           '"cashout":1.45,"confidence":0.0,"reason":"یک جمله فارسی"}' if allow_two else
           '{"bet":"RED|GREEN|YELLOW|SKIP","cashout":1.45,"confidence":0.0,'
           '"reason":"یک جمله فارسی"}')
    )


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
