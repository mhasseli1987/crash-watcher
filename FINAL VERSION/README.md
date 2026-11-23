# FINAL VERSION

پوشهٔ نسخهٔ نهاییِ ویندوز. اینجا فقط خروجی قابل توزیع نگه داشته می‌شود، نه کد یا وابستگی‌های تکراری.

## چه چیزی در گیت می‌ماند

- `repo 1/CrashWatcher/CrashWatcher.exe` — آخرین exe ساخته‌شده (بدون دادهٔ جلسه).

## چه چیزی داخل گیت نیست

| مسیر | چرا |
| --- | --- |
| `repo 1/CrashWatcher/_internal/` | ۳۲۱ مگابایت وابستگی؛ با `python tools/release.py` از نو ساخته می‌شود. |
| `repo 1/CrashWatcher/settings.json` و `logs/` | کلید API و دادهٔ بازی‌های کاربر. |
| `repo 1/CrashWatcher.old-*` | بکاپ بیلدهای قبلی. |
| `repo/` | کپی قدیمی سورس؛ سورس اصلی در ریشهٔ ریپو است. |
| `CrashWatcher-Windows-x64.zip` | بستهٔ قابل توزیع (exe + `_internal` + راهنما)، فقط لوکال. |

## ساخت مجدد

- بیلد و نصب: `python tools/release.py`
- بستهٔ قابل توزیع (زیپ): از پوشهٔ نصب‌شده، exe و `_internal` را کنار هم zip کنید.
