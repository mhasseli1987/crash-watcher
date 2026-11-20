"""کلیک موس روی دایرهٔ زرد؛ فقط کتابخانهٔ استاندارد."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from time import monotonic, sleep
import ctypes
import ctypes.wintypes
import json
import cv2
import numpy as np
from .config import DesktopSettings, PROJECT_ROOT
from .color_grid import read_grid
from .forecast import ColorForecaster
from .signals import (next_dot_row, row_yellow_due, two_yellows_in_a_row,
                      yellow_green_green_yellow)
from .smart_play import RESTART_AFTER_LOSSES
from .smart_review import CASHOUT_LIMIT, clamp_exit, default_exit
from .three_x import ThreeXSession

_DOWN = 0x0002
# «کلیک کلاسیک»: ورود پیش از پایان ۶ ثانیه، خروج روی ضریبی زیر ۱٫۹۹.
CASHOUT_CEILING = Decimal(str(CASHOUT_LIMIT))
CLASSIC_ENTRY_WINDOW_S = 6.0
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


RULE_LABELS = {"yellow-run": "صف ۷+۳ زرد", "yellow-pair": "جفت زرد",
               "yellow-burst": "۹ کلیک زرد", "row-signal": "سیگنال سطری",
               "streak-run": "روند هم‌رنگ ۷+", "value-bet": "ضریب ویژه",
               "pair-flip": "جفت رنگ", "smart": "کلیک هوشمند",
               "smart-play": "ورود هوشمند (مدل)", "three-x": "ورود ۳x"}


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
        self._signal_uses = 0
        self._yellow_burst_uses = 0
        self._row_uses = 0
        self._row_cell: tuple[int, int] | None = None
        self._streak_color: str | None = None
        self._streak = 0
        self._streak_clicks = 0
        self._streak_target: str | None = None
        self._flip_color: str | None = None
        self._pattern_seen = False
        self._run_seen = False
        self.forecaster = ColorForecaster()
        self.three_x = ThreeXSession(settings)
        self._pending_color: str | None = None
        self._smart_play: tuple[str, ...] = ()
        self._smart_play_active = False
        self._smart_cashout: Decimal | None = None
        self._stake_steps = 0  # چند بار مبلغ پایه با x2 دو برابر شده است
        self._ai_bet: dict | None = None
        self._ai_losses = 0
        self._ai_relearn = False
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

    @property
    def halted(self) -> bool:
        """True once the 3x session hit its profit or loss limit."""
        return self.settings.three_x_enabled and self.three_x.halted

    def update(self, settings: DesktopSettings) -> None:
        """Apply a settings change made while monitoring is already running."""
        self.settings = settings
        self.three_x.update(settings)

    @property
    def enabled(self) -> bool:
        return self.settings.autoclick_enabled

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
        """Target: detected yellow, fixed point, the saved click region, else a bet button."""
        if self.settings.autoclick_mode == "detect":
            found = self._detect()
            if found is not None:
                return found
        if self.settings.click_point is not None:
            return self.settings.click_point
        region = self.settings.click_region
        if region is not None:
            return (region.x + region.width // 2, region.y + region.height // 2)
        # This build's calibration only stores the four regions, so the bet
        # buttons from region 4 are the click target when no click area was saved.
        if self.settings.bet_region is not None:
            region = self.settings.bet_region
            return (region.x + region.width // 2, region.y + region.height // 2)
        return self.settings.bet_point_yellow

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
        if value >= Decimal(self.settings.yellow_threshold):
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
            return ClickReport(False, "target not found", count=self._clicks)
        self._wait_before_click()
        ok = self.sender(point[0], point[1], restore=self.settings.autoclick_restore_cursor)
        return ClickReport(True, "clicked" if ok else "click failed", point, self._clicks, self._armed)

    def test_classic(self) -> ClickReport:
        """کلیک تست روی «محل کلیک کلاسیک» (دکمهٔ Bet / Cash Out)."""
        point = self.settings.classic_click_point
        if point is None:
            return ClickReport(False, "classic click point not calibrated", count=self._clicks)
        self._wait_before_click()
        ok = self.sender(point[0], point[1], restore=self.settings.autoclick_restore_cursor)
        return ClickReport(True, "clicked" if ok else "click failed", point,
                           self._clicks, 0, rule="classic")

    def color(self, value: Decimal) -> str:
        """رنگ ضریب — همین قاعدهٔ رنگ در بخش بررسی هوشمند هم استفاده می‌شود."""
        return self._color(value)

    def set_smart_play(self, bets, active: bool = False, cashout=None) -> None:
        """تصمیم آمادهٔ مدل برای دست بعد: یک یا دو رنگ و ضریب خروج کلاسیک.

        `active` یعنی زبانهٔ AI Analysis در حال کار است؛ در این حالت هیچ‌کدام از
        قواعد ربات اجرا نمی‌شوند و فقط تصمیم مدل کلیک می‌شود. `cashout` ضریبی
        است که مدل برای «کلیک کلاسیک» انتخاب کرده (همیشه زیر ۱٫۹۹).
        """
        self._smart_play = tuple(color for color in (bets or ()) if color)
        self._smart_play_active = bool(active)
        try:
            self._smart_cashout = Decimal(str(cashout)) if cashout is not None else None
        except Exception:
            self._smart_cashout = None

    def fire(self, round_id: str, value: Decimal, confidence: float | None = None) -> ClickReport:
        report = self._decide(round_id, value, confidence)
        if self.settings.autoclick_enabled:
            self._record(round_id, value, report)
        return report

    def _record(self, round_id: str, value: Decimal, report: ClickReport) -> None:
        """Append one JSON line per round so the session can be replayed offline."""
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            entry = {"time": datetime.now().isoformat(timespec="seconds"), "round": round_id,
                     "value": str(value), "color": self._color(value),
                     "clicked": report.clicked, "reason": report.reason,
                     "forecast": report.forecast, "target": report.target,
                     "rule": report.rule, "detail": report.detail,
                     "clicks": report.count, "remaining": report.remaining}
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
                *self._three_x_lines(),
                *self._rule_lines(),
                f"لاگ هر دور: {self.log_path}",
            ]
            self.summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception:
            pass

    def _three_x_lines(self) -> list[str]:
        if not self.settings.three_x_enabled:
            return []
        try:
            return self.three_x.report_lines()
        except Exception:
            return []

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
                *self._three_x_lines(),
                "",
                *self._rule_lines(),
                "",
                f"دادهٔ کامل هر دور: {self.log_path}",
                f"بازبینی‌های ۳۰ دقیقه‌ای: {self.summary_path.parent / 'reviews.csv'}",
                "این فایل را برای تحلیل به دستیار بدهید.",
            ]
            self.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception:
            pass

    def wants_live(self) -> bool:
        """یک شرط کلاسیک باز است و باید ضریب زنده زیر نظر باشد؟"""
        return self._ai_bet is not None and not self._ai_bet["cashed"]

    def live_due(self) -> bool:
        """خواندن ضریب زنده فقط پس از پایان پنجرهٔ ۶ ثانیه‌ای شرط معنا دارد."""
        bet = self._ai_bet
        if bet is None or bet["cashed"]:
            return False
        return monotonic() - bet["entered"] >= CLASSIC_ENTRY_WINDOW_S - 2.5

    def watch_live(self, value: Decimal) -> ClickReport | None:
        """ضریب زندهٔ دست جاری: سر هدفِ زیر ۱٫۹۹ کلیک خروج کلاسیک زده می‌شود."""
        bet = self._ai_bet
        if bet is None or bet["cashed"] or value is None:
            return None
        if not bet["started"]:
            # تا شروع دست همان عدد قبلی روی صفحه می‌ماند؛ دو خوانش کوچک
            # پشت‌سرهم یعنی شمارش تازه شروع شده است.
            if value <= Decimal("1.20"):
                bet["low"] += 1
                bet["started"] = bet["low"] >= 2
            else:
                bet["low"] = 0
            return None
        if value >= CASHOUT_CEILING or value < bet["target"]:
            return None
        point = self.settings.classic_click_point
        if point is None:
            return None
        ok = self.sender(point[0], point[1], restore=self.settings.autoclick_restore_cursor)
        bet["cashed"] = True
        self._clicks += 1
        self._window["clicks"] += 1
        self._last_at = monotonic()
        return ClickReport(True, "clicked" if ok else "click failed", point,
                           self._clicks, 0, target=str(bet["target"]),
                           rule="smart-play", detail=f"خروج کلاسیک روی {value}×")

    def take_relearn_request(self) -> bool:
        """آیا ۸ باخت متوالی رسیده و باید دوباره نیم ساعت فقط تحلیل کرد؟"""
        requested, self._ai_relearn = self._ai_relearn, False
        return requested

    def _stake(self, target: int) -> None:
        """مبلغ شرط را با دکمه‌های x2 و /2 بازی روی تعداد مرحلهٔ خواسته‌شده ببر."""
        if not self.settings.stake_martingale_enabled:
            self._stake_steps = 0
            return
        target = max(0, min(int(target), int(self.settings.stake_double_steps)))
        while self._stake_steps < target:
            point = self.settings.stake_double_point
            if point is None:
                return
            self.sender(point[0], point[1], restore=self.settings.autoclick_restore_cursor)
            self._stake_steps += 1
        while self._stake_steps > target:
            point = self.settings.stake_half_point
            if point is None:
                return
            self.sender(point[0], point[1], restore=self.settings.autoclick_restore_cursor)
            self._stake_steps -= 1

    def _settle_ai_bet(self) -> None:
        """دست کلاسیکی که تمام شده را تسویه کن و باخت‌های متوالی را بشمار."""
        bet = self._ai_bet
        if bet is None:
            return
        self._ai_bet = None
        if bet["cashed"]:
            self._ai_losses = 0
            self._stake(0)  # برد: مبلغ پایه به مقدار اول برمی‌گردد
            return
        # بدون خروج: ضریب پیش از رسیدن به هدف بسته شده است.
        self._ai_losses += 1
        if self._ai_losses >= RESTART_AFTER_LOSSES:
            self._ai_losses = 0
            self._ai_relearn = True
        limit = int(self.settings.stake_double_steps)
        # باخت: تا مرحلهٔ آخر مبلغ دو برابر می‌شود؛ بعد از آن دوباره از پایه.
        self._stake(self._stake_steps + 1 if self._stake_steps < limit else 0)

    def _enter_classic(self, round_id: str, bets: tuple[str, ...],
                       cashout: Decimal | None,
                       last: Decimal | None = None) -> ClickReport:
        """AI Analysis + کلیک کلاسیک: یک کلیک برای ورود، کلیک دوم برای خروج."""
        if self._ai_bet is not None:
            return ClickReport(False, "smart-play: شرط قبلی هنوز باز است",
                               count=self._clicks)
        if not bets:
            return ClickReport(False, "smart-play: مدل برای این دست تصمیمی نداد",
                               count=self._clicks)
        point = self.settings.classic_click_point
        if point is None:
            return ClickReport(False, "smart-play: محل کلیک کلاسیک کالیبره نشده است",
                               count=self._clicks)
        # عدد خروج همیشه بین ۱٫۳۰ و ۱٫۸۵ است؛ اگر دست قبلی زیر ۱٫۱۰ بسته شده
        # باشد سقف تا ۱٫۹۰ باز می‌شود و همیشه زیر ۱٫۹۹ می‌ماند.
        target = Decimal(str(clamp_exit(cashout, last))) if cashout is not None \
            else Decimal(str(default_exit(last)))
        if monotonic() - self._last_at < self.settings.autoclick_cooldown_ms / 1000:
            return ClickReport(False, "cooldown", count=self._clicks)
        self._last_id, self._last_at = round_id, monotonic()
        self._clicks += 1
        self._window["clicks"] += 1
        self._pending_rule = (bets, "smart-play")
        self._rule_stats.setdefault("smart-play", [0, 0])[1] += 1
        # Permanent rule: the entry click happens at most one second after the
        # round closed, well inside the game's six-second betting window.
        self._wait_before_click()
        ok = self.sender(point[0], point[1], restore=self.settings.autoclick_restore_cursor)
        if ok:
            self._ai_bet = {"target": target, "colors": bets, "entered": monotonic(),
                            "low": 0, "started": False, "cashed": False}
        stake = (f" | مبلغ مرحله {self._stake_steps} (×{2 ** self._stake_steps})"
                 if self.settings.stake_martingale_enabled and self._stake_steps else "")
        return ClickReport(True, "clicked" if ok else "click failed", point,
                           self._clicks, 0, target="+".join(bets),
                           rule="smart-play",
                           detail=f"ورود کلاسیک — خروج زیر {target:.2f}{stake}")

    def _decide_smart_play(self, round_id: str, value: Decimal) -> ClickReport:
        """AI Analysis: فقط تصمیم مدل اجرا می‌شود؛ هیچ قانونی از ربات اجرا نمی‌شود."""
        self._settle_ai_bet()
        color = self._color(value)
        if self._pending_rule is not None:
            target_colors, rule = self._pending_rule
            self._pending_rule = None
            if target_colors and color in target_colors:
                self._rule_stats.setdefault(rule, [0, 0])[0] += 1
        self.forecaster.observe(color)
        self._window["rounds"] += 1
        self._rounds_seen += 1
        bets = tuple(dict.fromkeys(self._smart_play))
        self._smart_play = ()
        cashout, self._smart_cashout = self._smart_cashout, None
        if self.settings.smart_cashout_enabled:
            return self._enter_classic(round_id, bets, cashout, value)
        if not bets:
            return ClickReport(False, "smart-play: مدل برای این دست تصمیمی نداد",
                               count=self._clicks)
        points = [(bet, self._bet_point(bet)) for bet in bets]
        points = [(bet, point) for bet, point in points if point is not None]
        if not points:
            return ClickReport(False, "target not found", count=self._clicks)
        if monotonic() - self._last_at < self.settings.autoclick_cooldown_ms / 1000:
            return ClickReport(False, "cooldown", count=self._clicks)
        self._last_id = round_id
        self._last_at = monotonic()
        self._clicks += 1
        self._window["clicks"] += 1
        colors = tuple(bet for bet, _ in points)
        self._pending_rule = (colors, "smart-play")
        self._rule_stats.setdefault("smart-play", [0, 0])[1] += 1
        # Permanent rule: one second at most after the round, so both colours are
        # clicked right after the same wait instead of waiting twice.
        self._wait_before_click()
        ok = True
        for _, point in points:
            ok = self.sender(point[0], point[1],
                             restore=self.settings.autoclick_restore_cursor) and ok
        return ClickReport(True, "clicked" if ok else "click failed", points[0][1],
                           self._clicks, 0, target="+".join(colors), rule="smart-play")

    def _decide(self, round_id: str, value: Decimal,
                confidence: float | None = None) -> ClickReport:
        if not self.settings.autoclick_enabled:
            return ClickReport(False, "disabled", count=self._clicks)
        if round_id == self._last_id:
            return ClickReport(False, "same round", count=self._clicks, remaining=self._armed)
        if self._smart_play_active:
            # AI Analysis روشن است: مغز ربات مدل است، نه قواعد قبلی.
            return self._decide_smart_play(round_id, value)
        if self.settings.three_x_enabled:
            return self._decide_three_x(round_id, value, confidence)
        color = self._color(value)
        if self._pending_rule is not None:
            target_colors, rule = self._pending_rule
            self._pending_rule = None
            if target_colors and color in target_colors:
                self._rule_stats.setdefault(rule, [0, 0])[0] += 1
        if self._pending_color is not None:
            # Score the forecast that was acted on against the round just finished.
            self._window["gated_total"] += 1
            if color == self._pending_color:
                self._window["gated_correct"] += 1
            self._pending_color = None
        self.forecaster.observe(color)
        self._window["rounds"] += 1
        self._rounds_seen += 1
        flip = self._flip_color  # decided on the previous round, so it applies here only
        self._flip_color = None
        if color == self._streak_color:
            self._streak += 1
        else:
            self._streak_color, self._streak = color, 1
        if (self.settings.pair_flip_enabled and self._streak == 2
                and color in ("RED", "GREEN")):
            # Two reds or two greens in a row: the user expects the opposite colour next.
            self._flip_color = "GREEN" if color == "RED" else "RED"
        if (self.settings.streak_signal_enabled
                and self._streak > self.settings.streak_signal_length
                and color in ("RED", "GREEN") and self._streak_clicks <= 0):
            # Seven reds or greens in a row: the user reads a streak that long as
            # a trend that keeps going, so the same colour is bet for ten rounds.
            self._streak_clicks = self.settings.streak_signal_clicks
            self._streak_target = color
        burst_pending = self._yellow_burst_uses  # armed now? then it starts next round
        reading = self._chart_reading()
        pattern = self.settings.signal_rule_enabled and two_yellows_in_a_row(reading)
        if pattern and not self._pattern_seen and self._signal_uses <= 0:
            # Rising edge only: the same shape still on screen must not keep re-arming.
            self._signal_uses = self.settings.signal_repeat
        self._pattern_seen = pattern
        run_shape = self.settings.signal_rule_enabled and yellow_green_green_yellow(reading)
        if run_shape and not self._run_seen and self._yellow_burst_uses <= 0:
            # Yellow-green-green-yellow in one column: yellows are "on the way".
            self._yellow_burst_uses = self.settings.signal_burst_rounds
        self._run_seen = run_shape
        row_gap = self.settings.row_signal_gap
        row_due = (self.settings.row_signal_enabled
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
        follow_click = burst_pending > 0
        if follow_click:
            # The yellow-run signal: bet yellow on each of the next N rounds.
            self._yellow_burst_uses -= 1
            if self._waiting > 0:
                self._waiting -= 1
        signal_click = (not follow_click and self.settings.signal_rule_enabled
                        and self._signal_uses > 0 and color == "YELLOW")
        if signal_click:
            # The two-yellows-in-a-row signal: click now, betting on the next round.
            self._signal_uses -= 1
        elif not follow_click and value >= threshold and self._armed <= 0 and self._waiting <= 0:
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
        value_color = (None if (signal_click or follow_click or row_click or self._armed > 0)
                       else self._value_bet(value))
        streak_click = (not (signal_click or follow_click or row_click)
                        and self._armed <= 0 and value_color is None
                        and self._streak_clicks > 0 and self._streak_target is not None)
        if streak_click:
            self._streak_clicks -= 1
        target_color = ("YELLOW" if (signal_click or follow_click or row_click)
                        else self._streak_target if streak_click else value_color)
        rule = ("yellow-burst" if follow_click else "yellow-pair" if signal_click
                else "row-signal" if row_click else "value-bet" if value_color
                else "streak-run" if streak_click else "")
        if not (signal_click or follow_click or row_click or streak_click):
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
                elif not self.settings.smart_click_enabled:
                    return ClickReport(False, f"waiting for a round >= {threshold}",
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
            self._smart_play = ()
        else:
            targets = [target_color]
        points = [(color, self._bet_point(color)) for color in targets]
        points = [(color, point) for color, point in points if point is not None]
        if not points:
            return ClickReport(False, "target not found", count=self._clicks, remaining=self._armed)
        gated = (not signal_click and not follow_click and not row_click
                 and value_color is None and self._armed <= 0
                 and self.settings.smart_click_enabled)
        if gated:
            # Remember what was predicted so the next round can score it.
            self._pending_color = self.forecaster.forecast().color
        self._last_id, self._last_at = round_id, now
        self._clicks += 1
        self._window["clicks"] += 1
        if target_color or rule == "smart-play":
            colors = tuple(color for color, _ in points if color)
            self._pending_rule = (colors, rule or "unknown")
            self._rule_stats.setdefault(rule or "unknown", [0, 0])[1] += 1
        if self._armed > 0:
            # Every clicked round of a run spends one, the trigger included, so a
            # run is exactly autoclick_max_clicks clicks and then stops.
            self._armed -= 1
        ok = True
        for _, point in points:
            # Permanent rule: each click waits at most one second after the round.
            self._wait_before_click()
            # Every mode clicks: autoclick_dry_run only labels the report.
            ok = self.sender(point[0], point[1],
                             restore=self.settings.autoclick_restore_cursor) and ok
        colors = tuple(color for color, _ in points if color)
        return ClickReport(True, "clicked" if ok else "click failed", points[0][1], self._clicks,
                           self._armed, forecast=self.forecaster.forecast().label,
                           target="+".join(colors) or (target_color or ""),
                           rule=rule, detail=row_detail)

    def _decide_three_x(self, round_id: str, value: Decimal,
                        confidence: float | None) -> ClickReport:
        """ورود ۳x: تسویهٔ شرط قبلی، بعد تصمیم برای دست بعد."""
        outcome = self.three_x.observe(value, confidence)
        decision = self.three_x.decide(confidence)
        stats = self.three_x.stats()
        detail = decision.detail or decision.reason
        if outcome:
            detail = f"{outcome} | {detail}"
        report = ClickReport(
            False, decision.reason, count=self._clicks, remaining=0,
            detail=detail, target=f"احتمال {decision.probability:.0%}",
            rule="three-x")
        if not decision.bet:
            return report
        color = self._color(self.three_x.exit_factor)
        point = self._bet_point(color)
        if point is None:
            self.three_x._pending_stake = None
            return ClickReport(False, "target not found", count=self._clicks,
                               detail=detail, rule="three-x")
        now = monotonic()
        if now - self._last_at < self.settings.autoclick_cooldown_ms / 1000:
            self.three_x._pending_stake = None
            return ClickReport(False, "cooldown", count=self._clicks, detail=detail,
                               rule="three-x")
        self._last_id, self._last_at = round_id, now
        self._clicks += 1
        self._window["clicks"] += 1
        self._pending_rule = ((color,), "three-x")
        self._rule_stats.setdefault("three-x", [0, 0])[1] += 1
        self._wait_before_click()
        ok = self.sender(point[0], point[1],
                         restore=self.settings.autoclick_restore_cursor)
        return ClickReport(True, "clicked" if ok else "click failed", point,
                           self._clicks, 0, target=color, rule="three-x",
                           detail=detail)

    def close(self) -> None:
        self._write_session()
        for name in ("_backend", "_chart_backend"):
            backend = getattr(self, name)
            if backend is not None:
                backend.close()
                setattr(self, name, None)