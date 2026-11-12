"""کلیک موس روی دایرهٔ زرد؛ فقط کتابخانهٔ استاندارد."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from time import monotonic, sleep
import ctypes
import ctypes.wintypes
import cv2
import numpy as np
from .config import DesktopSettings

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


class AutoClicker:
    """یک کلیک برای هر دور تأییدشده، با محدودیت آستانه و فاصلهٔ زمانی."""

    def __init__(self, settings: DesktopSettings, sender=click) -> None:
        self.settings = settings
        self.sender = sender
        self._backend = None
        self._last_at = 0.0
        self._last_id: str | None = None

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
        if self.settings.autoclick_mode == "detect":
            return self._detect() or self.settings.click_point
        return self.settings.click_point

    def fire(self, round_id: str, value: Decimal) -> ClickReport:
        if not self.settings.autoclick_enabled:
            return ClickReport(False, "disabled")
        if value < Decimal(self.settings.autoclick_threshold):
            return ClickReport(False, f"below threshold {value}")
        if round_id == self._last_id:
            return ClickReport(False, "same round")
        now = monotonic()
        if now - self._last_at < self.settings.autoclick_cooldown_ms / 1000:
            return ClickReport(False, "cooldown")
        point = self._resolve()
        if point is None:
            return ClickReport(False, "target not found")
        self._last_id, self._last_at = round_id, now
        if self.settings.autoclick_delay_ms:
            sleep(self.settings.autoclick_delay_ms / 1000)
        if self.settings.autoclick_dry_run:
            return ClickReport(True, "dry-run", point)
        ok = self.sender(point[0], point[1], restore=self.settings.autoclick_restore_cursor)
        return ClickReport(True, "clicked" if ok else "click failed", point)

    def close(self) -> None:
        if self._backend is not None:
            self._backend.close()
            self._backend = None