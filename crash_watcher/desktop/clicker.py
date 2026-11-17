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
                      yellow_green_green_yellow, yellow_green_then_red)

_DOWN = 0x0002
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
               "yellow-pair-cycle": "جفت زرد ← زرد تکی",
               "row-signal": "سیگنال سطری (تکرار زرد در سطر)",
               "yellow-green-red": "زرد-سبز ← قرمز",
               "yellow-red-double": "زرد+قرمز هم‌زمان"}


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
        self._pending_rules: list[tuple[str, str]] = []
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
        self._prev_color: str | None = None
        self._pair_ready = False
        self._pair_cycles = 0
        self._double_left = 0

        self._yg_locked = False
        self.forecaster = ColorForecaster()
        self._pending_color: str | None = None
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

    def update(self, settings: DesktopSettings) -> None:
        """Apply a settings change made while monitoring is already running."""
        self.settings = settings

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
        """Target: detected yellow, fixed point, else the saved click region centre."""
        if self.settings.autoclick_mode == "detect":
            found = self._detect() or self.settings.click_point
            if found is not None:
                return found
        elif self.settings.click_point is not None:
            return self.settings.click_point
        if self.settings.bet_point_yellow is not None:
            return self.settings.bet_point_yellow
        bet_region = self.settings.bet_region
        if bet_region is not None:
            return (bet_region.x + bet_region.width * 5 // 6,
                    bet_region.y + bet_region.height // 2)
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
                "",
                f"دادهٔ کامل هر دور: {self.log_path}",
                f"بازبینی‌های ۳۰ دقیقه‌ای: {self.summary_path.parent / 'reviews.csv'}",
                "این فایل را برای تحلیل به دستیار بدهید.",
            ]
            self.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception:
            pass

    def _decide(self, round_id: str, value: Decimal) -> ClickReport:
        if not self.settings.autoclick_enabled:
            return ClickReport(False, "disabled", count=self._clicks)
        if round_id == self._last_id:
            return ClickReport(False, "same round", count=self._clicks, remaining=self._armed)
        color = self._color(value)
        if self._pending_rules:
            for target_color, rule in self._pending_rules:
                if target_color and color == target_color:
                    self._rule_stats.setdefault(rule, [0, 0])[0] += 1
            self._pending_rules = []
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
        burst_pending = self._yellow_burst_uses  # snapshot: a shape found now starts next round
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
        row_due = False
        row_cell = None
        if self.settings.row_signal_enabled and reading is not None:
            # One rule, two spacings the user has seen: four or five columns back.
            for candidate in (row_gap, row_gap + 1):
                if row_yellow_due(reading, candidate, self.settings.row_signal_rows,
                                  self.settings.row_signal_recent_columns):
                    row_due = True
                    row = next_dot_row(reading)
                    row_cell = (candidate, row) if row is not None else None
                    break
        if (row_due and self._row_uses <= 0
                and (row_cell is None or row_cell != self._row_cell)):
            # The row this round fills was yellow `gap` columns back, so the repeat
            # is due right now: bet it on this round, never on the one after it.
            self._row_uses = self.settings.row_signal_clicks
        if row_cell is not None:
            self._row_cell = row_cell
        # --- every rule keeps its own state and may fire in the same round as any
        # other one; a round collects one bet per colour that wants it.
        prev_color, self._prev_color = self._prev_color, color
        bets: list[tuple[str, str]] = []
        pair_click = color == "YELLOW" and prev_color != "YELLOW" and self._pair_ready
        if color == "YELLOW" and self._pair_cycles < 2:
            if prev_color == "YELLOW":
                self._pair_ready = True  # a back-to-back yellow pair just closed
            elif pair_click:
                self._pair_ready = False
                self._pair_cycles += 1
        if pair_click and self.settings.pair_cycle_enabled:
            bets.append(("YELLOW", "yellow-pair-cycle"))
        if prev_color == "YELLOW" and color == "RED":
            self._double_left = 4  # yellow then red: bet both colours for four rounds
        if self._double_left > 0:
            self._double_left -= 1
            if self.settings.double_bet_enabled:
                bets.append(("YELLOW", "yellow-red-double"))
                bets.append(("RED", "yellow-red-double"))
        shaped = (reading is not None and hasattr(reading, "color_at")
                  and self.settings.row_signal_enabled)
        yg_red_click = False
        if shaped and not self._yg_locked:
            yg_red_click = yellow_green_then_red(reading, self.settings.row_signal_gap + 1)
        if color != "RED":
            self._yg_locked = False
        elif yg_red_click:
            self._yg_locked = True
        if yg_red_click:
            bets.append(("YELLOW", "yellow-green-red"))
        threshold = Decimal(self.settings.autoclick_threshold)
        armed_now = False
        if self._waiting > 0:
            # The 7+3 yellow run waits on its own counter; it never blocks a rule.
            self._waiting -= 1
        elif self._armed > 0:
            bets.append(("YELLOW", "yellow-run"))
            self._armed -= 1
        elif value >= threshold:
            # A yellow arms a run: skip autoclick_skip_rounds rounds, then click the
            # next autoclick_max_clicks rounds whatever they close at.
            self._waiting = self.settings.autoclick_skip_rounds
            self._armed = self.settings.autoclick_max_clicks
            if self._waiting > 0:
                armed_now = True
            elif self._armed > 0:
                bets.append(("YELLOW", "yellow-run"))
                self._armed -= 1
        if burst_pending > 0:
            # The yellow-run signal: bet yellow on each of the next N rounds.
            bets.append(("YELLOW", "yellow-burst"))
            self._yellow_burst_uses -= 1
        if self.settings.signal_rule_enabled and self._signal_uses > 0 and color == "YELLOW":
            # The two-yellows-in-a-row signal: click now, betting on the next round.
            bets.append(("YELLOW", "yellow-pair"))
            self._signal_uses -= 1
        if self._row_uses > 0:
            bets.append(("YELLOW", "row-signal"))
            self._row_uses -= 1
        value_color = self._value_bet(value)
        if value_color:
            bets.append((value_color, "value-bet"))
        if self._streak_clicks > 0 and self._streak_target is not None:
            bets.append((self._streak_target, "streak-run"))
            self._streak_clicks -= 1
        if flip is not None:
            bets.append((flip, "pair-flip"))
        smart_gated = False
        if not bets:
            if armed_now:
                return ClickReport(False, f"run armed: waiting {self._waiting} rounds before clicking",
                                   count=self._clicks, remaining=self._armed)
            if self._waiting > 0:
                return ClickReport(False, f"skip: {self._waiting} rounds before the click window",
                                   count=self._clicks, remaining=self._armed)
            if not self.settings.smart_click_enabled:
                return ClickReport(False, f"waiting for a round >= {threshold}",
                                   count=self._clicks, remaining=0)
            # Smart mode: only a colour the model can back with real confidence
            # is worth a click; otherwise the round is skipped on purpose.
            prediction = self.forecaster.forecast()
            minimum = self.settings.prediction_min_confidence
            if prediction.samples < self.forecaster.min_samples:
                self._window["skipped"] += 1
                return ClickReport(False, f"skip: only {prediction.samples} rounds seen",
                                   count=self._clicks, remaining=0, forecast=prediction.label)
            if prediction.probability < minimum:
                self._window["skipped"] += 1
                return ClickReport(False, f"skip: {prediction.color} "
                                   f"{prediction.probability:.0%} < {minimum:.0%}",
                                   count=self._clicks, remaining=0, forecast=prediction.label)
            bets.append((prediction.color, "smart"))
            smart_gated = True
        now = monotonic()
        if now - self._last_at < self.settings.autoclick_cooldown_ms / 1000:
            return ClickReport(False, "cooldown", count=self._clicks, remaining=self._armed)
        chosen: list[tuple[str, str, tuple[int, int]]] = []
        for target, rule in bets:
            if any(target == taken for taken, _rule, _point in chosen):
                continue  # one bet per colour; two rules on one colour share it
            point = self._bet_point(target)
            if point is not None:
                chosen.append((target, rule, point))
        if not chosen:
            return ClickReport(False, "target not found", count=self._clicks,
                               remaining=self._armed)
        if smart_gated:
            # Remember what was predicted so the next round can score it.
            self._pending_color = self.forecaster.forecast().color
        self._last_id, self._last_at = round_id, now
        self._wait_before_click()
        sent = 0
        for index, (target, rule, point) in enumerate(chosen):
            if index:
                sleep(1.0)  # the game drops a second bet placed too soon after the first
            if self.sender(point[0], point[1], restore=self.settings.autoclick_restore_cursor):
                sent += 1
            self._rule_stats.setdefault(rule, [0, 0])[1] += 1
            self._pending_rules.append((target, rule))
        self._clicks += sent
        self._window["clicks"] += sent
        names = "+".join(target for target, _rule, _point in chosen)
        return ClickReport(sent > 0, "clicked" if sent == len(chosen) else "click failed",
                           chosen[0][2], self._clicks, self._armed,
                           forecast=self.forecaster.forecast().label,
                           target=names, rule=chosen[0][1], detail=names)

    def close(self) -> None:
        self._write_session()
        for name in ("_backend", "_chart_backend"):
            backend = getattr(self, name)
            if backend is not None:
                backend.close()
                setattr(self, name, None)
