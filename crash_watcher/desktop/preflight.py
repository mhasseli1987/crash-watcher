"""پیش‌بررسی نواحی و OCR پیش از شروع پایش.

خطاهای تنظیمات (ناحیهٔ انتخاب‌نشده، نقطهٔ کلیک ناموجود، مدلِ بی‌پاسخ) بهتر
است پیش از شروعِ پول‌دادن دیده شوند. این ماژول بدون کلیک و بدون تغییر تنظیمات،
هر ناحیه را یک‌بار می‌خواند و نتیجه را به‌صورت فهرست بررسی برمی‌گرداند.
توابع کپچر/OCR/مدل تزریق‌پذیرند تا تست بدون صفحهٔ واقعی اجرا شود.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str = ""

    @property
    def label(self) -> str:
        mark = "✓" if self.ok else "✗"
        return f"{mark} {self.name}" + (f" — {self.detail}" if self.detail else "")


def check_settings(settings) -> list[Check]:
    """بررسی‌های فقط-تنظیماتی؛ بدون لمس صفحه."""
    target = settings.click_point or settings.bet_region or settings.click_region
    checks = [
        Check("ناحیهٔ نتیجه (۲)", settings.result_region is not None,
              "انتخاب نشده" if settings.result_region is None else "تنظیم شده"),
        Check("هدف کلیک / دکمهٔ شرط", target is not None,
              "هیچ نقطه یا ناحیه‌ای برای کلیک تنظیم نشده" if target is None else "تنظیم شده"),
    ]
    if settings.smart_review_enabled:
        url = str(settings.smart_review_url).strip()
        checks.append(Check("آدرس پرووایدر", bool(url), "آدرس خالی است" if not url else url))
    return checks


def _grab(region):
    """یک فریم از ناحیه با mss؛ None در هر خطا."""
    try:
        import mss
        with mss.mss() as backend:
            raw = backend.grab({"left": region.x, "top": region.y,
                                "width": region.width, "height": region.height})
            return np.asarray(raw)[:, :, :3].copy()
    except Exception:
        return None


def _default_reader():
    try:
        from crash_watcher.vision.recognition import RapidOCRReader
        return RapidOCRReader()
    except Exception:
        return None


def _read_region(name, region, capture):
    """تصویر ناحیه را بگیر؛ ناحیهٔ تنظیم‌نشده در check_settings گزارش شده است."""
    if region is None:
        return None, None
    image = capture(region)
    if image is None or not getattr(image, "size", 0):
        return Check(name, False, "تصویر گرفته نشد (مختصات یا نمایشگر؟)"), None
    return Check(name, True, f"{image.shape[1]}×{image.shape[0]} پیکسل"), image


def run_preflight(settings, capture=None, reader=None, grid=None, probe=None,
                  check_ocr: bool = True) -> list[Check]:
    """همهٔ بررسی‌ها: نواحی، خواندن ضریب با OCR، شبکهٔ رنگ و اتصال مدل."""
    capture = capture or _grab
    checks = check_settings(settings)
    result_check, image = _read_region("تصویر ناحیهٔ ۲", settings.result_region, capture)
    if result_check is not None:
        checks.append(result_check)
    if check_ocr and image is not None:
        checker = reader or _default_reader()
        if checker is None:
            checks.append(Check("OCR ضریب", False, "موتور OCR در دسترس نیست"))
        else:
            try:
                from crash_watcher.vision.recognition import preprocess
                text, confidence = checker(preprocess(image)[0])
                text = (text or "").strip()
                ok = bool(text) and float(confidence) >= settings.ocr_confidence
                checks.append(Check("OCR ضریب", ok,
                                    f"«{text}» با اطمینان {float(confidence):.2f}"
                                    if text else "متنی خوانده نشد"))
            except Exception as exc:  # noqa: BLE001
                checks.append(Check("OCR ضریب", False, str(exc)))
    chart_check, chart_image = _read_region("تصویر ناحیهٔ ۳", settings.chart_region, capture)
    if chart_check is not None:
        checks.append(chart_check)
    if chart_image is not None:
        try:
            from crash_watcher.desktop.color_grid import read_grid
            result = (grid or read_grid)(chart_image)
            columns = int(getattr(result, "columns", 0))
            checks.append(Check("شبکهٔ رنگ", columns > 0,
                                f"{columns} ستون" if columns else "هیچ ستونی خوانده نشد"))
        except Exception as exc:  # noqa: BLE001
            checks.append(Check("شبکهٔ رنگ", False, str(exc)))
    if settings.smart_review_enabled:
        checker = probe
        if checker is None:
            from crash_watcher.desktop.smart_review import probe_models
            checker = lambda: probe_models(settings.smart_review_url,
                                           settings.resolved_key(), 10.0)  # noqa: E731
        try:
            models = checker()
            checks.append(Check("اتصال مدل", True,
                                f"{len(models)} مدل" if models else "پاسخ داد، فهرست مدل خالی"))
        except Exception as exc:  # noqa: BLE001
            checks.append(Check("اتصال مدل", False, str(exc)))
    return checks


def failed(checks) -> list[Check]:
    return [check for check in checks if not check.ok]


def summary(checks) -> str:
    return " | ".join(check.label for check in checks)
