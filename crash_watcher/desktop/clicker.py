"""کلیک موس روی دایرهٔ زرد؛ فقط کتابخانهٔ استاندارد."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from time import monotonic, sleep
import ctypes
import ctypes.wintypes
import json
import cv2
import numpy as np
from .bankroll import Bankroll
from .config import DesktopSettings, PROJECT_ROOT
from .color_grid import read_grid
from .forecast import ColorForecaster
from .signals import next_dot_row, row_yellow_due

_DOWN = 0x0002
# خروج هوشمند: هدف همیشه بین ۱٫۴۰ و ۱٫۹۰ است؛ هیچ شرطی از ۱٫۹۰ جلوتر باز نمی‌ماند.
CASHOUT_MIN_TARGET = Decimal("1.40")
CASHOUT_MAX_TARGET = Decimal("1.90")
# ضریب زندهٔ واقعی بین دو نمونهٔ پشت‌سرهم حداکثر دو برابر می‌شود؛ جهش بزرگ‌تر
# عدد ثابتِ پایان دست (Bang) است که دیر خوانده شده.
CASHOUT_MAX_STEP = Decimal("2.00")
# دستی که زیر این ضریب بسته می‌شود (کرک لحظه‌ای مثل ۱٫۰۵ یا ۱٫۰۸) یعنی دست
# بعد بی‌درنگ شرط بسته شود و همان دست تا ۱٫۹۰ برای کلیک خروج صبر کند.
LOW_CLOSE = Decimal("1.10")
# برای پذیرش خروج بدون پیام سایت، ضریب پایان دست باید دست‌کم این‌قدر بالای ضریب
# لحظهٔ کلیک باشد؛ عدد برابر با هدف می‌تواند کرش همان لحظه باشد.
CASHOUT_CONFIRM_MARGIN = Decimal("1.02")
_UP = 0x0004
try:
    _user32 = ctypes.windll.user32
except AttributeError:
    _user32 = None


def click(x: int, y: int, restore: bool = True) -> bool:
    if _user32 is None:
        return False
    origin = None
    if restore:
        point = ctypes.wintypes.POINT()
        if _user32.GetCursorPos(ctypes.byref(point)):
            origin = (point.x, point.y)
    if not _user32.SetCursorPos(int(x), int(y)):
        return False
    sleep(0.01)
    _user32.mouse_event(_DOWN, 0, 0, 0, 0)
    sleep(0.03)
    _user32.mouse_event(_UP, 0, 0, 0, 0)
    if origin is not None:
        sleep(0.02)
        _user32.SetCursorPos(*origin)
    return True


def find_yellow(image: np.ndarray):
    """بزرگ‌ترین لکهٔ زرد فشرده: (x, y, fill) یا None."""
    if image is None or image.size == 0 or image.ndim != 3:
        return None
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (20, 110, 110), (38, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total = float(image.shape[0] * image.shape[1])
    best = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < max(40.0, total * 0.0008) or area > total * 0.5:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if not 0.45 <= w / max(h, 1) <= 2.2:
            continue
        fill = area / float(w * h)
        if fill < 0.45:
            continue
        if best is None or area * fill > best[0]:
            best = (area * fill, x + w / 2, y + h / 2)
    return (int(best[1]), int(best[2]), 1.0) if best else None


@dataclass(frozen=True)
class ClickReport:
    clicked: bool
    reason: str
    point: tuple[int, int] | None = None
    count: int = 0
    remaining: int = 0
    forecast: str = ""
    target: str = ""
    rule: str = ""
    detail: str = ""
    dry_run: bool = False
    shadow: bool = False
    balance: float | None = None
    stake: float = 0.0
    double_level: int = 0


RULE_LABELS = {"yellow-run": "صف ۷+۳ زرد",
               "row-signal": "سیگنال سطری",
               "streak-run": "روند هم‌رنگ ۷+", "value-bet": "ضریب ویژه",
               "pair-flip": "جفت رنگ", "smart": "کلیک هوشمند",
               "smart-play": "ورود هوشمند (مدل)",
               "low-close": "بستن زیر ۱٫۱۰"}


class AutoClicker:
    """زرد که بسته شد: ۷ دست صبر، بعد سه کلیک پشت‌سرهم؛ بدون ری‌ست با زردهای بعدی."""

    def __init__(self, settings: DesktopSettings, sender=click, log_path: Path | None = None,
                 summary_path: Path | None = None, report_path: Path | None = None) -> None:
        self.settings = settings
        self.sender = sender
        self.log_path = log_path or PROJECT_ROOT / "logs" / "rounds.jsonl"
        self.summary_path = summary_path or PROJECT_ROOT / "logs" / "summary.txt"
        self.report_path = report_path or PROJECT_ROOT / "logs" / "session-report.txt"
        self._rule_stats: dict[str, list[int]] = {}
        self._pending_rule: tuple[str, str] | None = None
        self._rounds_seen = 0
        self._started = datetime.now()
        self._backend = None
        self._chart_backend = None
        self._last_at = 0.0
        self._last_id: str | None = None
        self._clicks = 0
        self._armed = 0
        self._waiting = 0
        self._row_uses = 0
        self._row_cell: tuple[int, int] | None = None
        self._streak_color: str | None = None
        self._streak = 0
        self._streak_clicks = 0
        self._streak_target: str | None = None
        self._flip_color: str | None = None
        self.forecaster = ColorForecaster()
        self._pending_color: str | None = None
        self._smart_play: tuple[str, ...] = ()
        self._smart_cashout: float | None = None
        # استاپ‌لاس سرمایه: فقط وقتی کاربر سرمایهٔ جلسه را روشن کند.
        self.bankroll: Bankroll | None = None
        self._refresh_bankroll()
        # «2 /2»: هر باخت یک دوبل و هر برد یک برگشت به مبلغ پایه.
        self._double_level = 0
        self._double_clicks = 0
        # توقف کلیکِ تازه (مثل قرمزهای پیاپی)؛ شرط باز و کش‌اوت را نمی‌بندد.
        self._hold_until: datetime | None = None
        self._hold_reason = ""
        # ارزیابی اولیه: دقیقهٔ اول پایش فقط ضریب‌ها و رنگ‌ها دیده می‌شوند و
        # هیچ شرط تازه‌ای بسته نمی‌شود؛ ساعت از همین لحظهٔ شروع پایش است.
        minutes = max(0, int(getattr(settings, "smart_play_learning_minutes", 0) or 0))
        if minutes:
            self._hold_until = datetime.now(timezone.utc) + timedelta(minutes=minutes)
            self._hold_reason = "evaluation"
        self._halve_clicks = 0
        self._cashout_target: Decimal | None = None
        self._cashout_points: list[tuple[int, int]] = []
        self._cashout_prev: Decimal | None = None
        self._cashout_confirmed = False
        self._cashout_attempted = False
        self._cashout_attempt_multiplier: Decimal | None = None
        self._cashout_planned_target: Decimal | None = None
        self._window = {"rounds": 0, "clicks": 0, "skipped": 0,
                        "gated_correct": 0, "gated_total": 0}
        self._window_started = monotonic()

    @property
    def clicks(self) -> int:
        """Clicks sent in this monitoring session."""
        return self._clicks

    @property
    def armed(self) -> int:
        """Clicks left in the run started by the last high round."""
        return self._armed

    @property
    def waiting(self) -> int:
        """Rounds still to skip before the click window of the current run."""
        return self._waiting

    def _session_limits(self) -> tuple[float, float]:
        """حد سود و حد ضرر جلسه (مبلغ)؛ ۰ یعنی خاموش."""
        def amount(text) -> float:
            try:
                value = float(Decimal(str(text or "0")))
            except Exception:
                return 0.0
            return max(0.0, value)
        return (amount(self.settings.session_profit_limit),
                amount(self.settings.session_loss_limit))

    def _refresh_bankroll(self) -> None:
        """سرمایهٔ جلسه را از تنظیمات بساز؛ حد سود یا حد ضرر هم حساب را روشن می‌کند."""
        profit, loss = self._session_limits()
        if not (self.settings.bankroll_enabled or profit or loss):
            self.bankroll = None
            return
        try:
            start = float(Decimal(self.settings.bankroll_start))
        except Exception:
            self.bankroll = None
            return
        fraction = self.settings.bankroll_stop_loss_pct / 100
        if (self.bankroll is None or self.bankroll.start != start
                or self.bankroll.stop_loss_fraction != fraction
                or self.bankroll.profit_limit != profit
                or self.bankroll.loss_limit != loss):
            self.bankroll = Bankroll(start, fraction, profit, loss)

    def update(self, settings: DesktopSettings) -> None:
        """Apply a settings change made while monitoring is already running."""
        if settings.bot_mode != self.settings.bot_mode:
            self._armed = self._waiting = self._row_uses = self._streak_clicks = 0
            self._flip_color = None
        self.settings = settings
        self._refresh_bankroll()

    @property
    def enabled(self) -> bool:
        return self.settings.autoclick_enabled

    @property
    def _simulated_label(self) -> str:
        """برچسب حالتِ بدون کلیک برای گزارش‌ها."""
        return "shadow" if self.settings.shadow_mode else "dry run"

    @property
    def balance(self) -> float | None:
        """موجودی سرمایهٔ جلسه؛ None یعنی استاپ‌لاس خاموش است."""
        return self.bankroll.balance if self.bankroll is not None else None

    def hold_clicks(self, seconds: float, reason: str = "") -> None:
        """تا این مدت هیچ شرط تازه‌ای بسته نشود؛ تسویه و کش‌اوت دست نمی‌خورند."""
        self._hold_until = datetime.now(timezone.utc) + timedelta(seconds=max(0.0, float(seconds)))
        self._hold_reason = reason

    def hold_remaining(self) -> float:
        """چند ثانیه از توقف کلیک مانده است."""
        if self._hold_until is None:
            return 0.0
        return max(0.0, (self._hold_until - datetime.now(timezone.utc)).total_seconds())

    def _stake(self) -> float:
        """مبلغ این شرط: مبلغ پایه ضرب در دوبل‌های جبران باخت."""
        try:
            base = float(Decimal(self.settings.fixed_bet_amount))
        except Exception:
            return 0.0
        return base * (2 ** self._double_level)

    @property
    def double_level(self) -> int:
        """تعداد دوبل‌های فعال مبلغ شرط در جبران باخت."""
        return self._double_level

    def _track_doubling(self, outcome: bool | None) -> None:
        """باخت = یک دوبل تازه؛ برد یا سربه‌سر = برگشت به مبلغ پایه."""
        if not self.settings.double_bet_enabled:
            return
        if outcome is False and self._double_level < self.settings.double_bet_max:
            self._double_level += 1
            self._double_clicks += 1
            return
        if self._double_level:
            self._halve_clicks += self._double_level
            self._double_level = 0

    def _apply_double_clicks(self) -> bool:
        """دکمه‌های ×2/÷2 بازی را پیش از ثبت شرط بعدی بزن تا مبلغ واقعی درست شود."""
        plan = ((self.settings.double_bet_point, "_double_clicks"),
                (self.settings.halve_bet_point, "_halve_clicks"))
        for point, attribute in plan:
            if getattr(self, attribute) and point is None:
                return False
            while getattr(self, attribute):
                if not self.sender(point[0], point[1],
                                   restore=self.settings.autoclick_restore_cursor):
                    return False
                setattr(self, attribute, getattr(self, attribute) - 1)
        return True

    def _payout(self, color: str | None) -> float:
        """پرداخت هر رنگ برای تسویهٔ سرمایه."""
        return {"RED": self.settings.payout_red, "GREEN": self.settings.payout_green,
                "YELLOW": self.settings.payout_yellow}.get(color or "GREEN", 2.0)

    def _detect(self):
        region = self.settings.click_region
        if region is None:
            return None
        try:
            import mss
            if self._backend is None:
                self._backend = mss.mss()
            raw = self._backend.grab({"left": region.x, "top": region.y,
                                      "width": region.width, "height": region.height})
            found = find_yellow(np.asarray(raw)[:, :, :3])
        except Exception:
            return None
        return (region.x + found[0], region.y + found[1]) if found else None

    def _resolve(self):
        """Target: detected yellow, fixed point, else the saved click region centre."""
        if self.settings.bot_mode == "turnbal" and self.settings.autoclick_mode == "detect":
            found = self._detect() or self.settings.click_point
            if found is not None:
                return found
        elif self.settings.click_point is not None:
            return self.settings.click_point
        region = self.settings.click_region
        if region is not None:
            return (region.x + region.width // 2, region.y + region.height // 2)
        return self.settings.click_point

    def _chart_reading(self):
        """Read the chart region once per round; None when no chart is usable."""
        region = self.settings.chart_region
        if region is None:
            return None
        try:
            import mss
            if self._chart_backend is None:
                self._chart_backend = mss.mss()
            raw = self._chart_backend.grab({"left": region.x, "top": region.y,
                                            "width": region.width, "height": region.height})
            return read_grid(np.asarray(raw)[:, :, :3])
        except Exception:
            return None

    def _value_bet(self, value: Decimal) -> str | None:
        """A colour the user attached to this exact multiplier, if any."""
        if not self.settings.value_bet_enabled:
            return None
        for text, color in self.settings.value_bets.items():
            try:
                if value == Decimal(text):
                    return color
            except Exception:
                continue
        return None

    def _bet_point(self, color: str | None):
        """The saved button for one colour, else the normal click target."""
        if self.settings.bot_mode == "classic":
            region = self.settings.bet_region
            if region is not None:
                return (region.x + region.width // 2, region.y + region.height // 2)
            return self._resolve()
        if color:
            point = getattr(self.settings, f"bet_point_{color.lower()}", None)
            if point is not None:
                return point
            region = self.settings.bet_region
            if region is not None:
                # Region 4 is the row of three buttons: red, green, moon (yellow),
                # each one a third of the row wide.
                sixth = {"RED": 1, "GREEN": 3, "YELLOW": 5}.get(color, 3)
                return (region.x + region.width * sixth // 6,
                        region.y + region.height // 2)
        return self._resolve()

    def _color(self, value: Decimal) -> str:
        """The game's dot color for a multiplier (red < 2, green < 10, else yellow)."""
        threshold = (Decimal(self.settings.yellow_threshold)
                     if self.settings.bot_mode == "turnbal" else Decimal("10"))
        if value >= threshold:
            return "YELLOW"
        return "RED" if value < Decimal(self.settings.red_threshold) else "GREEN"

    def _wait_before_click(self) -> None:
        """Permanent rule: never click more than one second after the round closed."""
        if self.settings.autoclick_delay_ms:
            sleep(self.settings.autoclick_delay_ms / 1000)

    def take_review(self) -> dict | None:
        """Summarise the last review window once its time is up; None otherwise."""
        now = monotonic()
        elapsed = now - self._window_started
        if elapsed < self.settings.review_window_s:
            return None
        stats, self._window = self._window, {"rounds": 0, "clicks": 0, "skipped": 0,
                                            "gated_correct": 0, "gated_total": 0}
        self._window_started = now
        review = self.forecaster.review()
        review.update(stats)
        review["minutes"] = elapsed / 60
        self._write_summary(review)
        return review

    def test(self) -> ClickReport:
        """One manual attempt at the resolved target; does not use the click budget."""
        point = self._resolve()
        if point is None:
            # ناحیهٔ کلیک تنظیم نشده: همان دکمهٔ شرط قرمز (ناحیهٔ ۴) را آزمایش کن.
            point = self._bet_point("RED")
        if point is None:
            return ClickReport(False, "target not found", count=self._clicks)
        # درخواست دستی کاربر «بدون کلیک» نیست: یک کلیک واقعی برای آزمون ارسال می‌شود.
        self._wait_before_click()
        ok = self.sender(point[0], point[1], restore=self.settings.autoclick_restore_cursor)
        return ClickReport(True, "clicked" if ok else "click failed", point, self._clicks,
                           self._armed, balance=self.balance)

    def color(self, value: Decimal) -> str:
        """رنگ ضریب — همین قاعدهٔ رنگ در بخش بررسی هوشمند هم استفاده می‌شود."""
        return self._color(value)

    def set_smart_play(self, bets, cashout=None) -> None:
        """تصمیم آمادهٔ مدل برای دست بعد: یک یا دو رنگ، به‌همراه ضریب خروج."""
        allowed = ("RED", "GREEN", "YELLOW") if self.settings.bot_mode == "turnbal" else ("PLAY",)
        self._smart_play = tuple(bet for bet in (bets or ()) if bet in allowed)
        self._smart_cashout = cashout

    def fire(self, round_id: str, value: Decimal) -> ClickReport:
        report = self._decide(round_id, value)
        if self.settings.autoclick_enabled:
            self._record(round_id, value, report)
        return report

    def _record(self, round_id: str, value: Decimal, report: ClickReport) -> None:
        """Append one JSON line per round so the session can be replayed offline."""
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            entry = {"time": datetime.now().isoformat(timespec="seconds"), "round": round_id,
                     "value": str(value), "color": self._color(value) if self.settings.bot_mode == "turnbal" else None,
                     "clicked": report.clicked, "reason": report.reason,
                     "forecast": report.forecast, "target": report.target,
                     "rule": report.rule, "detail": report.detail,
                     "clicks": report.count, "remaining": report.remaining,
                     "dry_run": report.dry_run, "shadow": report.shadow,
                     "balance": report.balance, "stake": report.stake,
                     "double_level": report.double_level}
            with self.log_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass  # logging must never break a click decision

    def _write_summary(self, review: dict) -> None:
        """Write a plain-text review the user can paste back for a deeper analysis."""
        try:
            self.summary_path.parent.mkdir(parents=True, exist_ok=True)
            hits = review.get("gated_total", 0)
            accuracy = (review.get("gated_correct", 0) / hits) if hits else 0.0
            model = review.get("model", "base rate")
            decision = ("مدل شرطی بهتر عمل کرد و فعال ماند" if model == "previous color"
                        else "نرخ پایه بهتر بود؛ مدل شرطی کنار گذاشته شد")
            lines = [
                f"بازبینی {review.get('minutes', 0):.0f} دقیقه‌ای — {datetime.now().isoformat(timespec='seconds')}",
                f"دور: {review.get('rounds', 0)} | کلیک: {review.get('clicks', 0)} | "
                f"ردشده: {review.get('skipped', 0)}",
                f"پیش‌بینی کلیک‌شده: {hits} | درست: {review.get('gated_correct', 0)} ({accuracy:.0%})",
                f"مدل فعال: {model} | دقت پایه: {review.get('base_accuracy', 0):.0%} | "
                f"دقت شرطی: {review.get('cond_accuracy', 0):.0%}",
                f"تصمیم خودکار: {decision}",
                *self._rule_lines(),
                f"لاگ هر دور: {self.log_path}",
            ]
            self.summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception:
            pass

    def _rule_lines(self) -> list[str]:
        if not self._rule_stats:
            return ["به تفکیک قانون: هنوز کلیکی ثبت نشده"]
        lines = ["به تفکیک قانون:"]
        for rule, (hits, total) in sorted(self._rule_stats.items(), key=lambda item: -item[1][1]):
            rate = hits / total if total else 0.0
            lines.append(f"  {RULE_LABELS.get(rule, rule)}: {total} کلیک، {hits} درست ({rate:.0%})")
        return lines

    def _write_session(self) -> None:
        """Write the end-of-session report the user can hand back for analysis."""
        try:
            self.report_path.parent.mkdir(parents=True, exist_ok=True)
            hits = sum(stats[0] for stats in self._rule_stats.values())
            clicks = sum(stats[1] for stats in self._rule_stats.values())
            rate = hits / clicks if clicks else 0.0
            model = self.forecaster.review()
            lines = [
                f"گزارش جلسه — شروع {self._started.isoformat(timespec='seconds')} | "
                f"پایان {datetime.now().isoformat(timespec='seconds')}",
                f"دورهای دیده‌شده: {self._rounds_seen} | کلیک: {self._clicks} | "
                f"دقت رنگ کلیک‌ها: {rate:.0%} ({hits}/{clicks})",
                f"مدل فعال: {model['model']} | دقت پایه: {model['base_accuracy']:.0%} | "
                f"دقت شرطی: {model['cond_accuracy']:.0%}",
                "",
                *self._rule_lines(),
                (f"سرمایهٔ جلسه: {self.bankroll.label()}" if self.bankroll else
                 "سرمایهٔ جلسه: خاموش"),
                "",
                f"دادهٔ کامل هر دور: {self.log_path}",
                f"بازبینی‌های ۳۰ دقیقه‌ای: {self.summary_path.parent / 'reviews.csv'}",
                "این فایل را برای تحلیل به دستیار بدهید.",
            ]
            self.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception:
            pass

    def _decide(self, round_id: str, value: Decimal) -> ClickReport:
        # دستِ در جریان بسته شد: پنجرهٔ کش‌اوتِ شرط قبلی تمام است.
        self._cashout_points = []
        self._cashout_prev = None
        if not self.settings.autoclick_enabled:
            return ClickReport(False, "disabled", count=self._clicks)
        if self.bankroll is not None and self.bankroll.stopped:
            # کف استاپ‌لاس رد شده: تا شروع دوبارهٔ جلسه هیچ شرطی بسته نمی‌شود.
            return ClickReport(False, f"capital stop-loss: {self.bankroll.label()}",
                               count=self._clicks, balance=self.bankroll.balance)
        if round_id == self._last_id:
            return ClickReport(False, "same round", count=self._clicks, remaining=self._armed)
        turnbal = self.settings.bot_mode == "turnbal"
        color = self._color(value) if turnbal else None
        if self._pending_rule is not None:
            target_colors, rule, stake, payouts = self._pending_rule
            settled = self._cashout_settled(value)
            # خروجی که سایت تأییدش نکرد فقط «نامعلوم» است و هیچ‌وقت شرط دست بعد را
            # قفل نمی‌کند: ربات بدون توقف به دست بعد می‌رود.
            unconfirmed = (self.settings.cashout_enabled and not self.settings.simulated
                           and not settled
                           and (self._cashout_attempted or self._cashout_planned_target is None
                                or value >= self._cashout_planned_target))
            self._pending_rule = None
            if unconfirmed:
                self._log_cashout(value, False, "cashout unconfirmed: continuing", ())
            # حالت کلاسیک مدل رنگ ندارد: هم پیام تأیید سایت و هم بسته‌شدن روشن
            # دست‌کم ۲٪ بالای ضریب کلیک، برد کش‌اوت است.
            hit = ((bool(target_colors) and color in target_colors) if turnbal
                   else bool(target_colors) and settled)
            if hit:
                self._rule_stats.setdefault(rule, [0, 0])[0] += 1
            outcome = None
            if target_colors:
                # برد رنگ اصلی، پوشش رنگ دوم (سربه‌سر) یا باخت. خروج فقط وقتی
                # برد است که «settled» باشد: پیام سایت یا بسته‌شدن روشن بالای
                # ضریب کلیک. خروج نامعلوم نه برد است نه باخت: دوبل و تسویهٔ سرمایه دست
                # نمی‌خورند تا دست بعد بدون توقف بسته شود.
                outcome = (None if unconfirmed
                           else True if settled else False
                           if self.settings.cashout_enabled and not self.settings.simulated
                           else True if turnbal and color == target_colors[0]
                           else None if not turnbal or color in target_colors[1:] else False)
                if turnbal or outcome is not None:
                    self._track_doubling(outcome)
            if self.bankroll is not None and target_colors and (turnbal or outcome is not None):
                payout = (float(self._cashout_confirmed_multiplier) if self._cashout_confirmed
                          else payouts.get(color) or min(payouts.values()))
                self.bankroll.settle(stake, payout, outcome)
                if self.bankroll.stopped:
                    # همین شرط کف را رد کرد؛ دیگر در همین دست هم کلیکی نمی‌شود.
                    return ClickReport(False, f"capital stop-loss: {self.bankroll.label()}",
                                       count=self._clicks, balance=self.bankroll.balance)
        if self._pending_color is not None:
            # Score the forecast that was acted on against the round just finished.
            self._window["gated_total"] += 1
            if color == self._pending_color:
                self._window["gated_correct"] += 1
            self._pending_color = None
        if turnbal:
            self.forecaster.observe(color)
        self._window["rounds"] += 1
        self._rounds_seen += 1
        flip = self._flip_color  # decided on the previous round, so it applies here only
        self._flip_color = None
        if turnbal and color == self._streak_color:
            self._streak += 1
        elif turnbal:
            self._streak_color, self._streak = color, 1
        if (turnbal and self.settings.pair_flip_enabled and self._streak == 2
                and color in ("RED", "GREEN")):
            # Two reds or two greens in a row: the user expects the opposite colour next.
            self._flip_color = "GREEN" if color == "RED" else "RED"
        if (turnbal and self.settings.streak_signal_enabled
                and self._streak > self.settings.streak_signal_length
                and color in ("RED", "GREEN") and self._streak_clicks <= 0):
            # Seven reds or greens in a row: the user reads a streak that long as
            # a trend that keeps going, so the same colour is bet for ten rounds.
            self._streak_clicks = self.settings.streak_signal_clicks
            self._streak_target = color
        # دستی که زیر ۱٫۱۰ بسته شد: شرط دست بعد همین حالا بسته می‌شود.
        low_close = value < LOW_CLOSE
        remaining_hold = self.hold_remaining()
        if remaining_hold > 0:
            # توقف کلیک: شرط قبلی همین بالاتر تسویه شد، اما شرط تازه بسته نمی‌شود.
            return ClickReport(False, f"hold {self._hold_reason}: {remaining_hold:.0f}s left",
                               count=self._clicks, remaining=self._armed)
        reading = self._chart_reading() if turnbal else None
        row_gap = self.settings.row_signal_gap
        row_due = (turnbal and self.settings.row_signal_enabled
                   and row_yellow_due(reading, row_gap, self.settings.row_signal_rows,
                                      self.settings.row_signal_recent_columns))
        row_cell = None
        if row_due:
            row = next_dot_row(reading)
            row_cell = (row, row_gap) if row is not None else None
        if (row_due and self._row_uses <= 0
                and (row_cell is None or row_cell != self._row_cell)):
            # The row this round fills was yellow `gap` columns back, so the repeat
            # is due right now: bet it on this round, never on the one after it.
            self._row_uses = self.settings.row_signal_clicks
        if row_cell is not None:
            self._row_cell = row_cell
        threshold = Decimal(self.settings.autoclick_threshold)
        if turnbal and value >= threshold and self._armed <= 0 and self._waiting <= 0:
            # A yellow arms a run: skip autoclick_skip_rounds rounds, then click the
            # next autoclick_max_clicks rounds whatever they close at. A second
            # yellow inside the run neither restarts the wait nor extends the burst.
            self._waiting = self.settings.autoclick_skip_rounds
            self._armed = self.settings.autoclick_max_clicks
            if self._waiting > 0:
                return ClickReport(False, f"run armed: waiting {self._waiting} rounds before clicking",
                                   count=self._clicks, remaining=self._armed)
        row_click = self._row_uses > 0
        if row_click:
            self._row_uses -= 1
        row_detail = (f"row {row_cell[0]} was yellow {row_gap} columns back"
                      if row_click and row_cell else "")
        value_color = (None if (not turnbal or row_click or self._armed > 0)
                       else self._value_bet(value))
        streak_click = (not row_click
                        and self._armed <= 0 and value_color is None
                        and self._streak_clicks > 0 and self._streak_target is not None)
        if streak_click:
            self._streak_clicks -= 1
        target_color = ("YELLOW" if row_click
                        else self._streak_target if streak_click else value_color)
        rule = ("row-signal" if row_click else "value-bet" if value_color
                else "streak-run" if streak_click else "")
        if not (row_click or streak_click):
            if self._waiting > 0 and value_color is None:
                self._waiting -= 1
                return ClickReport(False, f"skip: {self._waiting} rounds before the click window",
                                   count=self._clicks, remaining=self._armed)
            if self._armed <= 0:
                if value_color is not None:
                    target_color = value_color
                elif flip is not None:
                    target_color = flip
                    rule = "pair-flip"
                elif self._smart_play:
                    # تصمیم آمادهٔ مدل: یکی از قواعد بالا فعال نشده، پس نظر مدل
                    # را اجرا کن — یک رنگ یا دو رنگ هم‌زمان.
                    rule = "smart-play"
                elif low_close and not turnbal:
                    # قاعدهٔ ثابت کاربر: ضریب بسته‌شده زیر ۱٫۱۰ (۱٫۰۵، ۱٫۰۸ …)
                    # یعنی همین حالا برای دست بعد روی دکمهٔ شرط کلیک شود.
                    target_color = "PLAY"
                    rule = "low-close"
                elif not turnbal or not self.settings.smart_click_enabled:
                    return ClickReport(False, (f"waiting for a round >= {threshold}" if turnbal
                                               else "waiting for a smart decision"),
                                       count=self._clicks, remaining=0)
                else:
                    # Smart mode: only a colour the model can back with real confidence
                    # is worth a click; otherwise the round is skipped on purpose.
                    prediction = self.forecaster.forecast()
                    minimum = self.settings.prediction_min_confidence
                    if prediction.samples < self.forecaster.min_samples:
                        self._window["skipped"] += 1
                        return ClickReport(False, f"skip: only {prediction.samples} rounds seen",
                                           count=self._clicks, remaining=0,
                                           forecast=prediction.label)
                    if prediction.probability < minimum:
                        self._window["skipped"] += 1
                        return ClickReport(False, f"skip: {prediction.color} "
                                           f"{prediction.probability:.0%} < {minimum:.0%}",
                                           count=self._clicks, remaining=0,
                                           forecast=prediction.label)
                    target_color = prediction.color
                    rule = "smart"
        now = monotonic()
        if now - self._last_at < self.settings.autoclick_cooldown_ms / 1000:
            return ClickReport(False, "cooldown", count=self._clicks, remaining=self._armed)
        if self._armed > 0 and target_color is None:
            target_color = "YELLOW"  # the run after a yellow is a yellow bet
        if self._armed > 0 and not rule:
            rule = "yellow-run"
        if rule == "smart-play":
            # حداکثر دو رنگ، هر کدام با همان قاعدهٔ یک ثانیه‌ای کلیک می‌شود.
            targets: list[str | None] = list(dict.fromkeys(self._smart_play))
            if not turnbal:
                targets = targets[:1]
            self._smart_play = ()
        else:
            targets = [target_color]
        points = [(color, self._bet_point(color)) for color in targets]
        points = [(color, point) for color, point in points if point is not None]
        if not points:
            return ClickReport(False, "target not found", count=self._clicks, remaining=self._armed)
        simulated = self.settings.simulated
        gated = (turnbal and not row_click
                 and value_color is None and self._armed <= 0
                 and self.settings.smart_click_enabled)
        if gated:
            # Remember what was predicted so the next round can score it.
            self._pending_color = self.forecaster.forecast().color
        self._last_id, self._last_at = round_id, now
        self._clicks += 1
        self._window["clicks"] += 1
        if target_color or rule == "smart-play":
            self._cashout_confirmed = False
            self._cashout_attempted = False
            colors = tuple(color for color, _ in points if color)
            # در حالت آزمایشی/سایه سرمایه جابه‌جا نمی‌شود؛ مبلغ صفر ثبت می‌شود.
            stake = 0.0 if simulated else self._stake()
            payouts = {color: self._payout(color) for color in colors}
            self._pending_rule = (colors, rule or "unknown", stake, payouts)
            self._rule_stats.setdefault(rule or "unknown", [0, 0])[1] += 1
        if self._armed > 0:
            # Every clicked round of a run spends one, the trigger included, so a
            # run is exactly autoclick_max_clicks clicks and then stops.
            self._armed -= 1
        ok = True
        if not simulated and self.settings.double_bet_enabled:
            # مبلغ شرط در بازی پیش از کلیک شرط تنظیم می‌شود: ×2 پس از باخت و
            # ÷2 پس از برد تا برگشت به مبلغ پایه.
            ok = self._apply_double_clicks() and ok
        if not ok:
            self._pending_rule = None
            return ClickReport(False, "stake adjustment failed", count=self._clicks,
                               remaining=self._armed)
        for _, point in points:
            if simulated:
                continue
            # Permanent rule: each click waits at most one second after the round.
            ok = self.sender(point[0], point[1],
                             restore=self.settings.autoclick_restore_cursor) and ok
        if self.settings.cashout_enabled and (simulated or ok):
            # شرط ثبت شد: هدفِ خروج همیشه داخل بازهٔ کاربر (تب ترنبال، پیش‌فرض
            # ۱٫۴۰ تا ۱٫۹۰) نگه داشته می‌شود تا هیچ خروجی پایین‌تر از عددی که
            # کاربر گذاشته کلیک نشود.
            floor, ceiling = self._cashout_bounds()
            if rule == "low-close":
                # دست بعد از یک بستن زیر ۱٫۱۰ تا سقف بازه صبر می‌کند.
                target = ceiling
            else:
                try:
                    target = Decimal(str(self._smart_cashout))
                except Exception:
                    target = floor
            self._cashout_target = min(max(target, floor), ceiling)
            self._cashout_planned_target = self._cashout_target
            self._cashout_points = [point for _, point in points]
            self._cashout_prev = None
        self._smart_cashout = None
        colors = tuple(color for color, _ in points if color)
        reason = (f"{self._simulated_label}: would click" if simulated
                  else "clicked" if ok else "click failed")
        return ClickReport(not simulated and ok, reason, points[0][1], self._clicks,
                           self._armed, forecast=self.forecaster.forecast().label if turnbal else "",
                           target="+".join(colors) or (target_color or ""),
                           rule=rule, detail=row_detail,
                           dry_run=self.settings.autoclick_dry_run,
                           shadow=self.settings.shadow_mode, balance=self.balance,
                           stake=0.0 if simulated else self._stake(),
                           double_level=self._double_level)

    def _cashout_bounds(self) -> tuple[Decimal, Decimal]:
        """بازهٔ مجاز خروج از تنظیمات کاربر، همیشه داخل ۱٫۴۰ تا ۱٫۹۰."""
        def number(text, default: Decimal) -> Decimal:
            try:
                value = Decimal(str(text))
            except Exception:
                return default
            return value if value.is_finite() and value > 0 else default

        floor = min(max(number(self.settings.turnbal_cashout_min, CASHOUT_MIN_TARGET),
                        CASHOUT_MIN_TARGET), CASHOUT_MAX_TARGET)
        ceiling = min(max(number(self.settings.turnbal_cashout_max, CASHOUT_MAX_TARGET),
                          floor), CASHOUT_MAX_TARGET)
        return floor, ceiling

    def wants_cashout(self) -> bool:
        """شرطی باز است و هنوز روی هدف خروج کلیک نشده؟"""
        return bool(self.settings.cashout_enabled and self._cashout_points
                    and self._cashout_target is not None)

    def _cashout_settled(self, value: Decimal) -> bool:
        """آیا خروج انجام شده است؟ فقط با تأیید سایت یا بسته‌شدن روشن بالای ضریب
        کلیک؛ در حالت نامعلوم دست بعد بسته نمی‌شود تا شرطی بیهوده باز نماند."""
        if self._cashout_confirmed:
            return True
        attempt = self._cashout_attempt_multiplier
        return bool(self._cashout_attempted and attempt is not None
                    and value >= attempt * CASHOUT_CONFIRM_MARGIN)

    def confirm_cashout(self, multiplier: Decimal) -> None:
        """Only an observed site win message can confirm an attempted cashout."""
        if self._pending_rule is not None and self._cashout_attempted and not self._cashout_confirmed:
            self._cashout_confirmed = True
            self._cashout_confirmed_multiplier = multiplier
            self._log_cashout(multiplier, False, "cashout confirmed by site", (), confirmed=True)

    def tick(self, live=None, ended: bool = False) -> None:
        """کش‌اوت هوشمند: ضریب جاری که به هدفِ مدل رسید، روی همان دکمهٔ شرط کلیک کن.

        `live` همان خوانش OCR ناحیهٔ ۲ در این لحظه است و `ended` یعنی همین حالا
        صفحهٔ پایان دست (Bang) دیده شد. کلیک فقط وقتی انجام می‌شود که عدد واقعاً
        ضریب زندهٔ در حال بالارفتن باشد؛ عدد ثابتِ پایان دست هرگز کلیک نمی‌شود،
        چون آن کلیک به‌جای خروج، یک شرط تازه باز می‌کند.
        """
        if not (self.settings.cashout_enabled and self._cashout_points
                and self._cashout_target is not None):
            return
        if ended:
            # دست پیش از رسیدن به هدف بسته شد: پنجرهٔ خروجِ این شرط تمام است.
            self._close_cashout("cashout skipped: round ended before the target")
            return
        if live is None:
            return
        try:
            value = live if isinstance(live, Decimal) else Decimal(str(live))
        except Exception:
            return
        prev = self._cashout_prev
        if prev is not None and value < prev * Decimal("0.8"):
            # افت بزرگ یعنی صفحهٔ پایان/شمارش معکوس گذشته و پرواز تازه از ۱.۰۰ شروع شده.
            prev = None
        if prev is not None and value > prev * CASHOUT_MAX_STEP:
            # جهش بزرگ بین دو نمونهٔ پشت‌سرهم عدد زنده نیست؛ عدد ثابتِ Bang است
            # که دیر رسیده. کلیک روی دکمهٔ شرط اینجا شرط تازه باز می‌کرد.
            self._close_cashout(f"cashout skipped: {value} is not a live reading")
            return
        self._cashout_prev = value
        if prev is None or value <= prev or value < self._cashout_target:
            return
        target, self._cashout_target = self._cashout_target, None
        points, self._cashout_points = self._cashout_points, []
        self._cashout_prev = None
        ok = True
        simulated = self.settings.simulated
        for point in points:
            if simulated:
                continue
            ok = self.sender(point[0], point[1],
                             restore=self.settings.autoclick_restore_cursor) and ok
        self._cashout_attempted = bool(ok and not simulated)
        self._cashout_attempt_multiplier = value if self._cashout_attempted else None
        reason = (f"smart cashout target {target} on the bet button" if not simulated else
                  f"{self._simulated_label}: would cash out at {target}")
        self._log_cashout(value, ok and not simulated, reason, points)

    def _log_cashout(self, value, clicked: bool, reason: str, points=(), confirmed=False) -> None:
        """ثبت هر تصمیم خروج؛ لاگ هرگز نباید تصمیم کلیک را خراب کند."""
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            entry = {"time": datetime.now().isoformat(timespec="seconds"),
                     "round": self._last_id, "value": str(value), "clicked": bool(clicked),
                     "reason": reason, "site_confirmed": confirmed,
                     "points": [list(point) for point in points],
                     "clicks": self._clicks}
            with self.log_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _close_cashout(self, reason: str, value="") -> None:
        """پنجرهٔ خروج را بدون کلیک ببند تا شرط تازه‌ای باز نشود."""
        target, self._cashout_target = self._cashout_target, None
        points, self._cashout_points = self._cashout_points, []
        self._cashout_prev = None
        self._log_cashout(value or target, False, reason, points)

    def close(self) -> None:
        self._write_session()
        for name in ("_backend", "_chart_backend"):
            backend = getattr(self, name)
            if backend is not None:
                backend.close()
                setattr(self, name, None)
