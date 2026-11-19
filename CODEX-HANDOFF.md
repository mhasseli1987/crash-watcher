# Crash Watcher — complete handoff brief for Codex

Last updated: 2026-11-19 (فیکس کرش settings.json قدیمی + بیلد دوبارهٔ exe). This file is the single source of truth for the
current state of the project. Read it fully before touching code.

---

## 2026-11-19 (ج) — فیکس کرش «unexpected keyword argument» هنگام اجرای exe (آخرین تغییر)

### خلاصهٔ فارسی

- کاربر exe را روی ویندوز اجرا کرد و پنجرهٔ خطا داد:
  `DesktopSettings.__init__() got an unexpected keyword argument 'pair_cycle_enabled'`.
- **علت**: خود exe سالم بود؛ فایل `settings.json` کنارش از یک نسخهٔ قدیمی‌تر مانده بود و
  کلیدهای حذف‌شده (`pair_cycle_enabled`, `double_bet_enabled`) داشت. `load_settings`
  همهٔ کلیدهای فایل را مستقیم به `DesktopSettings(**data)` می‌داد و همین باعث `TypeError`
  می‌شد — یعنی هر settings.json قدیمی می‌توانست اپ را از کار بیندازد.
- **رفع**: `load_settings` حالا کلیدهای ناشناس را حذف می‌کند (`fields(DesktopSettings)`)
  و اگر فایل JSON دیکشنری نباشد، تنظیمات پیش‌فرض برمی‌گرداند. تست جدید:
  `test_settings_file_with_removed_keys_still_loads` → **198 passed**.
- exe با همین کد دوباره ساخته و در `F:\CRASH WATCHER\FINAL VERSION\repo 1\CrashWatcher`
  جای‌گذاری شد (settings.json و logs کاربر دست‌نخورده)، و روی دسکتاپ واقعی اجرا شد
  (PID زنده، بدون پنجرهٔ خطا).

### English technical state

- `config.py` `load_settings`: `data` is filtered against
  `{item.name for item in dataclasses.fields(DesktopSettings)}` before
  `DesktopSettings(**data)`; non-dict JSON returns `DesktopSettings()`. Later
  migrations (delay clamp, skip-rounds defaults, timer/shutdown reset) still run on the
  filtered dict, so old files keep getting the new defaults.
- Build unchanged: PyInstaller onedir, same command as the previous entry.

---

## 2026-11-19 (ب) — تایمر جلسه، تم رنگی رابط، بیلد exe (آخرین تغییر)

### خلاصهٔ فارسی

- **تایمر جلسه کامل شد** (قابلیت پایه بود، نمایش نداشت): نشان شمارش معکوس کنار عنوان پنجره
  (`HH:MM:SS`، زیر ۵ دقیقه قرمز می‌شود)، نوار پیشرفت جلسه روی داشبورد («X از Y دقیقه»)، و
  چهار دکمهٔ آمادهٔ «۳۰ دقیقه / ۱ ساعت / ۳ ساعت / بدون تایمر» که بدون رفتن به تب کالیبراسیون
  کار می‌کنند. تایمر بر پایهٔ زمان واقعی است، پس در مکث هم می‌شمارد. پایان جلسه: گزارش
  `logs/session-report.txt` و در صورت تیک، `shutdown /s /t 60`.
- **تم رابط کاملاً عوض شد**: پالت شیشه‌ای سبز/کهربایی با گرادیان تیره، کارت‌های متریک
  شیشه‌ای، تب‌های قرصی، دکمه‌های رنگی (شروع سبز، توقف سرخ، مکث کهربایی، ادامه آبی،
  بازنشانی خاکستری). الهام از کامپوننت React «adjustable glass dashboard» که کاربر فرستاد؛
  چون ربات PySide6 است، فقط زبان بصری‌اش در QSS پیاده شد — React/Tailwind/shadcn در این
  پروژه نصب نمی‌شود و نباید نصب شود.
- **ریسپانسیو**: گرید کارت‌ها با عرض پنجره ۴/۳/۲ ستون می‌شود، همهٔ صفحه‌ها داخل QScrollArea،
  برچسب تب‌ها کوتاه + elide + tooltip تا در عرض کم بریده و ناخوانا نشوند.
- **باگ خوانایی رفع شد**: فیلدهای X و Y در کالیبراسیون خالی دیده می‌شدند (PySide6 بعد از
  `setRange` که مقدار را عوض نمی‌کند، متن line edit را به‌روز نمی‌کند).
- **بیلد exe**: PyInstaller در venv نصب شد و خروجی onedir در
  `F:\CRASH WATCHER\FINAL VERSION\repo 1\CrashWatcher` کپی شد (settings.json و logs کاربر
  دست‌نخورده ماندند). exe با پلتفرم offscreen اجرا شد و سالم بالا آمد، بعد بسته شد.
- تست‌ها: `197 passed`.

### English technical state

- `worker.py`: new `session_tick(int remaining, int total)` signal emitted only when the
  remaining second changes (total `0` = no timer), plus module-level
  `session_seconds_left(started, limit_minutes, now=None) -> int | None` (unit-tested).
  The loop still sets `_session_limit_reached` and breaks at zero, so `session_finished`
  fires and the clicker writes its report before any shutdown.
- `ui.py`: `session_badge` (objectName `sessionBadge`, property `urgent=true` under five
  minutes), `session_left`, `session_progress` (`QProgressBar#session`), preset buttons
  (`#preset`), `_on_session_tick`, `_set_session_limit`, `_set_badge_urgent`,
  `_relayout_metrics` (4/3/2 columns from the tab width), tab elide + tooltips, shortened
  tab labels. Button object names: `start`, `stop`, `pause`, `resume`, `reset`.
  The QSS `STYLE` block is the single place for colours; `_toggle_style()` keeps the header
  toggles in the same palette.
- `tests/test_desktop.py`: `test_session_seconds_left_counts_down_and_rounds_safely`.
- Build: `build-windows.ps1` unchanged (`--onedir --windowed --collect-all rapidocr
  --collect-all onnxruntime`); output is ~352 MB, exe ≈ 9 MB.
- Not done / caveats: no click-through testing of the live game; the exe was only verified
  to boot (offscreen), not to OCR a real screen.

## 2026-11-19 — بررسی هوشمند ضریب‌ها + ورود هوشمند به بازی (آخرین تغییر)

این بخش تازه‌ترین وضعیت است و بر بخش‌های قدیمی‌تر اولویت دارد.

### خلاصهٔ فارسی

- زبانهٔ چهارم «بررسی هوشمند ضریب‌ها» به سرور محلی FreeLLMAPI وصل می‌شود:
  `http://127.0.0.1:31415/v1` (سرور بدون کلید `401` می‌دهد، پس کلید داشبورد لازم است)،
  مدل `auto`. چهار ناحیهٔ علامت‌زدهٔ کاربر به متن تبدیل و به مدل داده می‌شود و مدل فقط یک
  JSON برمی‌گرداند: `{"bet":..., "second":..., "confidence":..., "reason":...}`.
- «ورود هوشمند به بازی» (تیک جدا): اول دورهٔ یادگیری ۳۰ دقیقه‌ای (از لحظهٔ شروع پایش)،
  بدون شرط‌بندی؛ ضریب‌ها و رنگ‌ها ثبت و پیش‌بینی‌ها با نتیجهٔ واقعی سنجیده می‌شوند. بعد از
  آن فقط دست‌هایی بازی می‌شود که احتمال مدل از ۴۰٪ کمتر نباشد (و دقت اندازه‌گیری‌شده هم
  زیر ۴۰٪ نباشد). لازم نیست هر دست بازی شود.
- پنجرهٔ ۶ ثانیه‌ای بین دو ضریب: تصمیم برای دست بعد در رشتهٔ جدا و *پیش از* بسته‌شدن
  ضریب جاری آماده می‌شود؛ سر بسته‌شدن ضریب فقط کلیک می‌ماند (قاعدهٔ «حداکثر یک ثانیه پیش
  از هر کلیک» حفظ شده). پاسخ دیررس که بیش از سه دست کهنه شود دور ریخته می‌شود.
- دو رنگ هم‌زمان: مدل می‌تواند رنگ دوم را در `second` بدهد؛ هر دو دکمه زده می‌شود. چون
  پرداخت هر رنگ ≥ ۲ برابر است، بدترین حالت سربه‌سر است و ضرر نمی‌دهد.
- قواعد قبلی (۷+۳ زرد، جفت زرد، سطری، روند هم‌رنگ، ضریب ویژه، جفت رنگ) دست‌نخورده و
  اولویت‌دارند؛ مدل فقط وقتی وارد می‌شود که هیچ قاعده‌ای فعال نباشد.
- اندازه‌گیری واقعی روی سرور کاربر: `auto` گاهی `502` می‌دهد (خطای سرویس بالادست) و
  `glm-4.7-flash` یک بار ۴۷ ثانیه طول کشید؛ برای بازی واقعی مدل سریع لازم است.

### English technical state

- New modules: `repo/crash_watcher/desktop/smart_review.py` (prompt, HTTP via `urllib`,
  JSON parsing, models probe, `logs/smart-review.jsonl`) and
  `repo/crash_watcher/desktop/smart_play.py` (`SmartPlanner`: learning phase, prepared
  decision per hand, staleness guard, accuracy scoring).
- `worker.py`: `smart_report` signal, `_recent` deque, `_chart_grid()`, `_start_smart_review()`
  in a side thread, planner created in `run()` (learning clock starts with monitoring),
  `_smart_play_active()`. Flow per completed round: `planner.observe(round_id, color)` →
  `planner.take()` → `clicker.set_smart_play(bets)` → `clicker.fire(...)`.
- `clicker.py`: `color()`, `set_smart_play()`, `smart-play` rule (highest existing-rule
  priority is preserved), multi-target clicking with the one-second wait before *each* click,
  `_pending_rule` now holds a tuple of colours so a two-colour hand is scored correctly.
- `config.py`: `smart_review_enabled/url/key/model/timeout_s/every_rounds/rules` and
  `smart_play_enabled`, `smart_play_learning_minutes` (30), `smart_play_min_confidence` (0.40),
  `smart_play_two_colors` (true), all validated.
- `ui.py`: tab «بررسی هوشمند ضریب‌ها» with on/off checkbox, URL/key/model/timeout/every/rules,
  learning minutes, min confidence, two-colour checkbox, probe + save buttons, live status line,
  raw prompt/answer pane.
- Tests: `repo/tests/test_smart_review.py` (11) and `repo/tests/test_smart_play.py` (10);
  full suite `196 passed` with
  `cd repo && PYTHONIOENCODING=utf-8 QT_QPA_PLATFORM=offscreen ./.venv/Scripts/python.exe -m pytest -q`.
- Packaging: `CrashWatcher-smart-review.zip` in the workspace root (created with a Python
  `zipfile` script because Git Bash has no `zip`; `.venv`, caches, `logs` and `settings.json`
  are excluded).
- Not verified live: the click path against the real game window, and a full 30-minute learning
  session. The FreeLLMAPI connection itself was verified with the user's key (330 models,
  verdicts parsed).

## 0. خلاصهٔ فارسی (برای کاربر)

- برنامه یک ربات دسکتاپ ویندوزی (Python + PySide6) است که ناحیهٔ نتیجهٔ بازی را
  اسکرین‌شات می‌گیرد، با OCR متن `Bang @x.xx` را می‌خواند، با ردیف تاریخچه
  کراس‌چک می‌کند و بعد استراتژی شبیه‌سازی و «کلیک خودکار» را اجرا می‌کند.
- وضعیت فعلی کلیک خودکار (قانون جدید جایگزین «ده کلیک» شده): با دیدن ضریب
  ≥ آستانه (زرد)، ربات **۷ دست صبر** می‌کند و بعد دست‌های **۸، ۹ و ۱۰** را
  کلیک می‌کند — سه کلیک، بدون توجه به ضریب آن دست‌ها. خودِ دور زرد کلیک
  نمی‌شود. اگر داخل این بازه دوباره زرد بیاید، شمارش **ریست نمی‌شود** و
  شمارنده تمدید هم نمی‌شود؛ بعد از سه کلیک فرایند ریست می‌شود. این اعداد
  (`autoclick_skip_rounds=7`، `autoclick_max_clicks=3`) از UI قابل تغییرند و
  فایل تنظیمات قدیمی خودکار به همین مقادیر مهاجرت می‌کند.
- هر کلیک **۱ ثانیه** بعد از بسته‌شدن ضریب زده می‌شود (قبلاً بلافاصله بود و
  بعضی کلیک‌ها در بازی ثبت نمی‌شد). این مقدار در `settings.json` قدیمی که صفر
  ذخیره شده بود، هنگام لود خودکار به ۱۰۰۰ میلی‌ثانیه افزایش می‌یابد و در UI هم
  حداقل ۱۰۰۰ است (تا ۵۵۰۰ قابل افزایش).
- کلیک در هر دو حالت «شبیه‌سازی» و «واقعی» واقعی است؛ دکمهٔ حالت فقط برچسب
  گزارش را عوض می‌کند.
- تست‌ها: ۱۵۰ تست، ۱۴۸ پاس؛ ۲ تست فقط به‌خاطر نبود Qt/`libxkbcommon` روی لینوکس
  اجرا نمی‌شوند و باید روی ویندوز اجرا شوند.
- کارهای باقی‌مانده و باگ‌های شناخته‌شده در بخش ۹ آمده است.

---

## 1. Where the code lives

| Item | Value |
|---|---|
| Upstream repo | `https://github.com/mhasseli1987/crash-watcher` |
| Base commit already on GitHub | `724d77d` — *Restore desktop/vision modules and fix import paths* |
| Local branch | `main`, HEAD `dafdd86` (base + 8 commits) |
| Delivered patch | `crash-watcher-update.patch` (workspace root, `724d77d..HEAD`) |
| Delivered source zip | `CrashWatcher-START.zip` (workspace root, repo without `.venv`/`.git`/caches) |

The eight commits in the patch (oldest first):

| Commit | Subject | Why |
|---|---|---|
| `e9aa9f9` | Restore settings validation, wire auto-click, and add START.bat | `DesktopSettings.validate()` was missing → every save/start raised `AttributeError`; auto-click was never connected to the worker |
| `cd86474` | Hide the main window while picking a screen region | Region screenshots included the app window itself |
| `0097f10` | Add real/simulation and auto-click toggles, and a resizable window | Large glossy green header toggles; resizable window with size grip |
| `59f1d95` | Click on every high round in both modes, capped at ten clicks | Product request: click in dry-run too, budget of ten |
| `23a680d` | Click the ten rounds after a high round and keep tabs readable when narrow | Product request: the ten rounds after a trigger must all be clicked |
| `cccadef` | Delay every auto-click by a second and keep a run at ten clicks | Product request: 1 s delay (clicks were not registering) and no reset by a high round inside the run |
| `615c226` | Add a measured color-forecast click gate and a ten-minute review | Product request: click only at ≥60% predicted confidence, skip weak rounds, review every ten minutes |
| `0721afc` | Read the trend chart's colour grid in the site's column-major order | Position/shape rules need dots mapped to cells, not one blob |
| `dafdd86` | Replace the ten-click rule with wait seven, then click three | Product request: click only the 8th, 9th and 10th round after a yellow |

**How to apply (on the user's machine):**

```
git pull
git am crash-watcher-update.patch
git push
```

Or start from the zip: unzip `CrashWatcher-START.zip`, then run `START.bat`.
Do **not** commit `settings.json`, `logs/`, `.venv/`, `dist/`, `build/` — all gitignored.

## 2. First steps for Codex (Windows)

```
cd crash-watcher
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --no-cache-dir -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q          # expect 150 passed
.\.venv\Scripts\python.exe desktop_app.py
```

`START.bat` is the one-click path: it creates `.venv` if needed, runs
`build-windows.ps1` (pip install → pytest → PyInstaller `--onedir` → copy
`settings.json` if present), then launches `dist\CrashWatcher\CrashWatcher.exe`.
`RunCrashWatcher.bat` launches that exe directly.
On a non-Windows machine you can run the logic tests but **cannot** build the exe
and **cannot** verify real clicks; say so instead of claiming a build.

## 2.5 Color vocabulary the user uses (remember this)

From now on the user describes tasks with the game's dot colors. These map exactly to the
thresholds already implemented (defaults `red_threshold = 2.00`, `yellow_threshold = 10.00`):

| Color | Range | `MultiplierCategory` |
|---|---|---|
| قرمز (red) | below 1.99, i.e. `< 2.00` | `RED` |
| سبز (green) | 2.00 – 9.99 | `GREEN` |
| زرد (yellow) | 10.00 – 999 | `YELLOW` |

Example: "دو کلیک زرد بعدش سه کلیک سبز" means two clicks after rounds ≥ 10.00 and then three
after rounds in 2.00–9.99. Do not invent extra colors or re-map these boundaries.

### Trend-chart layout (from the site's own "Understanding Trend Chart" dialog)

The history grid is **column-major**: the newest round sits at the top-left cell, the next
older round directly **below** it in the same column, and when a column is full the history
continues at the top of the column to its right. So consecutive rounds are vertical
neighbours, and the top row holds every Nth round (N = rows in the grid), not consecutive
rounds. The site's example: a column of three greens followed by a column with one orange
and a column with green→yellow reads `5.64 → 2.63 → 4.15 → 1.82 → 2.31 → 18.78`.

The legend on that dialog: **green ≥ 2x, orange < 2x, yellow ≥ 10x** — identical to the
three categories above (the user calls orange “red”).

Consequence for prediction work: the click region must cover the whole grid, colors are read
by hue (no OCR), and any position/shape rule has to be expressed against this layout, not
against the visual top rows.

## 3. What the application is

A **Windows-only desktop app** (Python 3.11+, PySide6) that watches a crash-game
window, reads the finished-round multiplier from the `Bang @x.xx` screen with local
OCR, cross-checks it against the game's history row, runs a fixed-bet simulation,
and optionally clicks a saved screen region. Everything is local: screenshots + OCR.
No network calls, no login automation, no scraping.

Hard constraints — do not break without being asked:

- No network access at runtime; no secrets in the repo.
- OCR/regex stays strict: a wrong multiplier is worse than a missed round.
- Auto-click defaults stay **off** (`autoclick_enabled=False`). A default must never
  make the app click on its own.
- UI strings are Persian and RTL. Code, comments, tests, commits are English.
- `Decimal` for multipliers/money; frozen dataclasses for models.

## 4. Layout

```
desktop_app.py              entry point (also used by PyInstaller)
crash_watcher/core/         phase 1: pure strategy engine, no I/O
  config.py                 StrategyConfig + positive_number() guard
  enums.py                  StrategyState / MultiplierCategory / BetOutcome / ...
  events.py, models.py      frozen dataclasses
  strategy_engine.py        the state machine
crash_watcher/vision/       phase 2: generic capture/recognition/detection stack
crash_watcher/desktop/      the shipped app
  config.py                 DesktopSettings / Region / load+save settings JSON
  worker.py                 MonitorWorker(QThread): capture → OCR → detect → strategy → click
  final_result.py           BangRecognizer + BangDetector (the detector in use)
  clicker.py                AutoClicker + win32 click() + find_yellow()
  forecast.py               ColorForecaster: next-color probability + review stats
  ui.py                     MainWindow, RegionOverlay, CalibrationWorker, stylesheet
  app.py, __main__.py       launch; app.py sets DPI awareness before Qt import
  ocr_test.py, strategy.py  debug helper / unused re-export
examples/                   strategy_simulation, visual_diagnostic, vision_benchmark
tests/                      150 tests (59 test functions, parametrized)
START.bat / build-windows.ps1 / RunCrashWatcher.bat / run-dev.ps1 / run.bat
README.md / راهنمای-فارسی.md / PHASE-2-DIAGNOSTICS.md / PHASE-3-TEST-REPORT.md
sample.png, screenshot1.jpg, screenshot2.jpg   real game screenshots used by tests
```

`vision/detection.py`, `vision/pipeline.py`, `vision/recognition.py` are used by
`examples/` and tests, **not** by the desktop app. The app path is
`worker.py → final_result.py → core/strategy_engine.py → clicker.py`. Thumbnail-motion
logic is duplicated in `vision/detection.py`, `vision/pipeline.py`, `worker.py`,
`final_result.py` with different thresholds (3.0 / 3.0 / 1.0 / 3.0) — change the one
you actually need, not all four.

## 5. Runtime flow (`MonitorWorker.run()`, one iteration per `interval_ms` ≈ 100 ms)

1. `ScreenCapture.capture()` grabs `result_region`. A geometry change or failure resets
   the detector and cancels an active strategy cycle.
2. A 64×32 grayscale thumbnail diff decides whether OCR is needed; otherwise the last
   reading is reused (`worker.py:111-115`).
3. `BangRecognizer.recognize()` → `FinalReading(value, confidence, raw_text, category,
   uncertain)`. Accepted only if the text fullmatches
   `^Bang\s*@\s*(\d{1,4}\.\d{2})\s*[xX×]?$` and confidence ≥ `ocr_confidence`.
4. With `history_region` set, the last history row is OCR'd and parsed by
   `parse_history_entry()` (round id + exactly one multiplier). A round is confirmed only
   when the history row matches the Bang value — the anti-false-positive cross-check.
5. `BangDetector.observe()` confirms a round only after a non-Bang phase, `stable_frames=3`
   stable frames and no motion; a Bang already visible at startup is ignored. If the history
   row never matches, `history_timeout_frames=15` (~1.5 s) confirms the strict, stable Bang
   so the 6-second click window is not lost.
6. `StrategyEngine.process_round(round_id, multiplier)` → immutable `RoundResult`.
7. `AutoClicker.fire(round_id, multiplier)` → `click_report` signal (see §6).
8. GUI signals: `observation`, `preview`, `history_observation`, `click_report`,
   `problem`, `status`. `MonitorWorker.configure()` pushes live settings changes into the
   running `AutoClicker` via `AutoClicker.update()`.

## 6. Auto-click rules — current, exact

```python
def fire(round_id, value):
    if not enabled:                                   return "disabled"
    if round_id == self._last_id:                     return "same round"
    if value >= threshold and _armed <= 0 and _waiting <= 0:
        _waiting = autoclick_skip_rounds              # 7
        _armed = autoclick_max_clicks                 # 3
        if _waiting:                                  return "run armed: waiting 7 rounds"
    if _waiting > 0:
        _waiting -= 1                                 return "skip: n rounds before the window"
    if _armed <= 0:                                   return "waiting for a round >= <threshold>"
    if now - _last_at < cooldown_ms/1000:             return "cooldown"
    point = _resolve();  if point is None:            return "target not found"
    _last_id, _last_at = round_id, now
    _clicks += 1; _armed -= 1
    sleep(autoclick_delay_ms / 1000)                  # default 1000 ms
    send_click(point)                                 # both "modes" click for real
    return ClickReport(True, "clicked"/"click failed", point, _clicks, _armed)
```

Semantics the user explicitly asked for (do not "improve" these back):

- A yellow (≥ threshold) arms a run: **skip 7 rounds, then click the 8th, 9th and 10th**.
  The yellow round itself is not clicked. After the third click the process resets.
- A yellow **inside** that window does not reset the wait, extend it, or add clicks; it is
  just one of the skipped/clicked rounds.
- `autoclick_skip_rounds` and `autoclick_max_clicks` are editable in the calibration tab;
  settings files written before this rule are migrated on load to 7/3.
- Every clicked round waits `autoclick_delay_ms` (default and floor 1000 ms) after the round
  is confirmed, because immediate clicks were sometimes not registered by the game.
- Both label modes send a real click; `autoclick_dry_run` only changes nothing but the
  report label. Never reintroduce a `dry-run` bypass of the click.
- Target resolution (`_resolve`): in `autoclick_mode="detect"` (default) a yellow blob found
  inside `click_region` by `find_yellow()`, else `click_point`, else the **centre of
  `click_region`**; in `"point"` mode `click_point`, else the region centre.
- A click is `SetCursorPos` + `mouse_event` (win32), with optional cursor restore.
- `AutoClicker.test()` is the manual «کلیک تست» button: one attempt, no budget, no delay.

`ClickReport(clicked, reason, point, count, remaining, forecast)` — `remaining` is the clicks
left in the run, shown in the UI as «باقی‌مانده n/10»; `forecast` carries the model's label
(`GREEN 55% (base rate, n=87)`).

### Smart click (opt-in, `smart_click_enabled`)

When no run is active, `fire()` asks `ColorForecaster` (`desktop/forecast.py`) for the most
likely next dot color:

- probabilities come from the recent color mix (Dirichlet-smoothed base rate over a 300-round
  window) blended with the rate conditional on the previous color, shrunk towards the base
  rate so a few coincidences cannot dominate;
- after 40 scored rounds, the conditional model is used only if it has out-predicted the base
  rate (rolling 40-round hit record) — that is the automatic “improve after the review” step
  the user asked for;
- the forecast is scored against every round that follows, and clicks made through the gate
  are scored separately (`gated_correct` / `gated_total`);
- the click happens only when `probability ≥ prediction_min_confidence` and at least
  `min_samples` (40) rounds were seen; otherwise the round is skipped with a reason such as
  `skip: GREEN 46% < 60%`;
- a high round still arms the unconditional wait-7-then-click-3 run, so that rule survives
  smart mode;
- every `review_window_s` (600 s, i.e. ten minutes) `AutoClicker.take_review()` returns a
  summary that `MonitorWorker` emits as `review_report`; the GUI writes it to
  `logs/reviews.csv` and shows rounds / clicks / skips / forecast hit rate / model accuracies.

Honest expectation: the game draws independent rounds, so the forecast usually sits at the
base rates (red ≈ 50%, green ≈ 40%, yellow ≈ 10%) and the gate mostly *does not* fire. That
is the intended behaviour — it keeps the clicker idle when there is no evidence — and the
review CSV is what proves or disproves any edge.

## 7. Settings (`crash_watcher/desktop/config.py`)

Stored in `settings.json` next to the exe (frozen exe) or project root (source run).

| Field | Default | Notes |
|---|---|---|
| `result_region`, `history_region`, `click_region` | `None` | `Region(monitor, x, y, width, height)`, absolute desktop pixels |
| `click_point` | `None` | `tuple[int, int]` |
| `autoclick_enabled` | `False` | |
| `autoclick_mode` | `"detect"` | `detect` or `point` |
| `autoclick_threshold` | `"10.00"` | string, parsed with `Decimal` |
| `autoclick_cooldown_ms` | `1500` | between two clicks |
| `autoclick_delay_ms` | `1000` | floor `MIN_CLICK_DELAY_MS = 1000`, clamped on load; UI range 1000–5500 |
| `autoclick_max_clicks` | `3` | clicks in the burst after the wait |
| `autoclick_skip_rounds` | `7` | rounds skipped after the yellow before the burst |
| `autoclick_dry_run` | `True` | label only; clicks still happen |
| `autoclick_restore_cursor` | `True` | |
| `smart_click_enabled` | `False` | only click when the color forecast is confident |
| `prediction_min_confidence` | `0.60` | 0.5–1; below it the round is skipped |
| `review_window_s` | `600` | ten-minute review window; ≥ 60 |
| `fixed_bet_amount` / `max_attempts` | `"10"` / `8` | |
| `yellow_threshold` / `red_threshold` | `"10.00"` / `"2.00"` | |
| `interval_ms` | `100` | ≥ 20 |
| `ocr_confidence` | `0.55` | (0, 1] |
| `theme` | `"dark"` | unused |
| `tesseract_path` | `""` | empty = bundled RapidOCR |

- `load_settings()` rebuilds the three `Region`s, converts `click_point` back to a tuple,
  raises `autoclick_delay_ms` below 1000 up to 1000 (old files saved `0`), then calls
  `DesktopSettings.validate()`.
- `validate()` must raise `ValueError` (never `AttributeError`, never
  `InvalidOperation`): the GUI turns it into a Persian dialog. A missing `validate()` once
  made the whole app unusable — keep `test_settings_roundtrip_and_validation` green.
- `save_settings()` writes `settings.tmp` then replaces, so a crash cannot leave half a file.

## 8. UI (`crash_watcher/desktop/ui.py`)

- Dark stylesheet (`STYLE`), RTL, always-on-top, **resizable**: `setMinimumSize(480, 400)`,
  `QSizeGrip` at the bottom, preview rescales in `resizeEvent`.
- Window title shows the mode: `Crash Watcher · REAL` / `· SIMULATED`.
- Header: two large glossy dark-green buttons (`_toggle_style()`, bright when active):
  `حالت شبیه‌سازی` / `حالت واقعی ⚡` and `کلیک خودکار: روشن/خاموش`. Both save immediately
  and push into a running worker through `apply_settings()` → `MonitorWorker.configure()`.
- Tab 1 «داشبورد»: 15 metric cards (each `minimumWidth 118`, caption `wordWrap(True)`),
  start/stop/pause/resume/reset, live preview, status note (now also states the click
  count and the delay).
- Tab 2 «ناحیه و کالیبراسیون»: monitor, X/Y/W/H, amount, attempts, threshold, interval,
  OCR confidence, optional Tesseract path, auto-click checkbox, real-click checkbox,
  click threshold, **click delay spin box (1000–5500 ms)**, three region buttons
  (result / history / click), save, OCR test, **کلیک تست**, delete-all-settings, status labels.
- Tab 3 «تاریخچهٔ رویدادها»: 8-column table, clear, CSV export; every row is also appended
  to `logs/events.csv`.
- Narrow-window fix: every tab page is wrapped in `_scrollable()` — a `QScrollArea` with
  `page.setMinimumWidth(520)` — so shrinking the window scrolls instead of overlapping text.
- `RegionOverlay`: the main window **hides first**, waits 0.3 s, then screenshots the chosen
  monitor dimmed; drag a rectangle, Esc cancels, and the window returns on the calibration
  tab. Never capture with the app window visible — that was an earlier user-visible bug.

## 9. Tests

```
QT_QPA_PLATFORM=offscreen ./.venv/bin/python -m pytest -q     # Linux, logic only
.\.venv\Scripts\python.exe -m pytest -q                        # Windows, full
```

- 150 tests total. On this headless Linux box: **148 passed, 2 failed**, both with
  `ImportError: libxkbcommon.so.0` (`test_gui_starts_idle_and_closes`,
  `test_worker_can_stop_and_restart`). They are not broken code; run them on Windows.
- Click tests use `_Sender` (fake win32 click) plus the `no_detection` fixture, which stubs
  `AutoClicker._detect` so the suite never captures a real screen. `_clicker()` defaults to
  `autoclick_delay_ms=0` to keep the suite fast; delay behaviour has its own test with a
  monkeypatched `sleep`.
- Key click tests:
  - `test_clicker_waits_for_a_high_round_then_clicks_the_region_centre`
  - `test_yellow_waits_seven_rounds_then_clicks_eight_nine_ten` (the current rule)
  - `test_yellow_inside_the_wait_does_not_restart_the_run`
  - `test_high_round_inside_the_run_does_not_reset_the_count`
  - `test_clicker_waits_before_clicking` (1.0 s before the sender runs)
  - `test_clicker_clicks_once_per_round_and_respects_cooldown`
  - `test_clicker_clicks_in_dry_run_and_never_when_disabled`
  - `test_confirmed_round_reaches_click_through_detector_and_strategy`
- `test_history_timeout_confirms_a_stable_bang` guards the history timeout.
- OCR tests load the real bundled RapidOCR models on synthetic images; keep them.
- `test_supplied_screenshots_do_not_create_false_final_results` guards against accepting a
  live multiplier or countdown as a final result.

## 10. Open problems (fix before adding features)

Severity order; none fixed yet.

1. **No OCR retry after a transient failure** (`desktop/worker.py:111-128`). The failed
   reading is cached in `previous_reading`, and `history_previous` is advanced even when
   `parse_history_entry` failed, so one bad read on a static screen can drop the round.
   Fix: don't cache uncertain readings, retry every N frames, only advance `history_previous`
   after a successful parse.
2. **Calibration guard is bypassable** (`desktop/ui.py:354`). `_settings_from_form()` always
   writes `result_region` from the spin boxes (default `1,0,0,20,20`), so after one «ذخیره»
   the "no region selected" check never fires and monitoring can start on a 20×20 square.
3. **Unbounded logging from the GUI thread** (`desktop/ui.py:646`). Every `problem` signal
   both adds a table row and appends to `logs/events.csv`; with a bad region that is
   ~10 rows/second.
4. **Data written next to the exe** (`desktop/config.py:10`). `settings.json`, `logs/`,
   `self-test.json` land in the program directory; under `Program Files` or a read-only
   share saving fails. Move to `%LOCALAPPDATA%\CrashWatcher` with a migration path.
5. **No schema tolerance** (`desktop/config.py:103`). `DesktopSettings(**data)` raises
   `TypeError` on any unknown/renamed key, so the app refuses to start. Add a `version` field
   and ignore/migrate unknown keys.
6. **Torn reads of strategy state.** The GUI reads `state`, `attempt_number`,
   `remaining_attempts` separately while the worker mutates them, and `ui.py:629` reads the
   private `strategy._cycle_id`. Expose a public `cycle_id` and one consistent snapshot
   (or lock) before displaying them.
7. **Docs drift.** `PHASE-3-TEST-REPORT.md` claims 131 passing tests (now 150). Keep README,
   راهنمای-فارسی.md and this file in sync when behaviour changes.
8. **The forecast layer is an honest filter, not an edge.** `forecast.py` only measures; it
   cannot make an independent RNG predictable. Do not tune it to produce 60% confidence by
   shortening the window or dropping the shrinkage — that would only manufacture false
   certainty. If the user reports “it never clicks in smart mode”, explain the base rates
   rather than weakening the gate.
9. **Minor cleanup.** Unused imports in `ui.py` (`Path`, `QFont`, `QGroupBox`) and
   `worker.py`; `desktop/strategy.py` is an unused re-export; `ocr_test.py` is a debug
   script inside the package; `find_yellow()` computes a fill ratio then returns a hardcoded
   `1.0`.
10. **Live-game accuracy is still unverified.** Only synthetic images and three screenshots
   (`sample.png`, `screenshot1.jpg`, `screenshot2.jpg`) were tested. Real `Bang @…` captures
   from the user's game window are the missing evidence for OCR thresholds and latency.
11. **Real click registration is unverified.** The 1-second delay was added because the user
    reported some clicks not registering. Whether 1000 ms is enough (or too much inside the
    6-second window) must be confirmed by the user on Windows. If clicks still miss, options:
    larger default delay, double-click, or `mouse_event` with absolute coordinates.
12. **Click delay blocks the worker thread** (`clicker.py`), so capture/OCR pause for that
    second. It works with the current round cadence; if rounds ever get faster, move the
    delayed click to its own thread/timer.

## 11. Verification workflow used so far (reuse it)

```
# tests
QT_QPA_PLATFORM=offscreen ./.venv/bin/python -m pytest -q

# commit, then rebuild the deliverables from the workspace root
git -C repo format-patch 724d77d..HEAD --stdout > crash-watcher-update.patch
zip -qr CrashWatcher-START.zip repo -x "repo/.venv/*" "repo/.git/*" "*/__pycache__/*" "repo/.pytest_cache/*"

# prove the patch applies cleanly on the real base
git clone repo /tmp/verify && cd /tmp/verify && git reset --hard 724d77d
git -c user.name=v -c user.email=v@e.com am ../crash-watcher-update.patch
diff -r -x .git -x .venv -x __pycache__ -x .pytest_cache . <repo>   # must print nothing
```

If you add commits, regenerate both the patch and the zip, and re-run the verification
above. The patch must be generated from `724d77d`, not from the previous HEAD.

## 12. Conventions

- Immutable dataclasses for models; `Decimal` for multipliers and money; no floats in state.
- Public behaviour documented in docstrings and in README / راهنمای-فارسی.md.
- Tests: plain pytest functions, no network, no real screen, no Qt needed for logic tests;
  Persian UI labels asserted as strings.
- Commits: imperative subject, short body explaining why.
- Never claim an exe was built or a UI/click change was visually verified from a
  non-Windows/headless machine.
