"""One-shot, read-only advice from two captured history regions."""
from dataclasses import dataclass
from datetime import datetime
import json
import math
import re
import hashlib
from threading import Event
from time import perf_counter, sleep
from urllib import request

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from .color_grid import _centres, read_grid
from .config import PROJECT_ROOT
from .smart_review import ask, _json_object, append_log, _COLOR_ALIASES
from crash_watcher.vision.capture import ScreenCapture
from crash_watcher.vision.recognition import RapidOCRReader

NAMES = {"RED": "نارنجی / قرمز", "GREEN": "سبز", "YELLOW": "زرد", "UNKNOWN": "نامشخص"}
COLORS = ("RED", "GREEN", "YELLOW")
# مدل محلی پاسخ را در ۲ ثانیه می‌دهد؛ سقف کوتاه، مسیر ابری را از معادله خارج می‌کند.
LOCAL_ADVICE_TIMEOUT_S = 12.0
# قرمزهای پیاپی یعنی احتمال باخت بالا: یک دقیقه شرط تازه بسته نمی‌شود تا
# روند تغییر کند؛ این آستانه از جدول و نوار ضرایب خوانده می‌شود.
RED_STREAK_LIMIT = 4
RED_PAUSE_SECONDS = 60.0
ADVICE_SCHEMA = {
    "type": "object",
    "properties": {
        "color": {"type": "string", "enum": list(COLORS)},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "multiplier": {"type": ["number", "null"], "minimum": 1},
        "reason": {"type": "string", "maxLength": 120}},
    "required": ["color", "confidence", "multiplier", "reason"],
    "additionalProperties": False}


@dataclass(frozen=True)
class HistorySnapshot:
    recent: tuple[tuple[str, str], ...]
    cells: tuple[tuple[int, int, str], ...]
    sequence: tuple[str, ...] = ()

    def text(self):
        recent = "، ".join(f"{value}x ({NAMES[color]})" for value, color in self.recent)
        counts = {name: sum(c == name for _, _, c in self.cells) for name in ("RED", "GREEN", "YELLOW")}
        return "ضریب‌های آخر، چپ به راست: " + recent + "\nجدول: " + "، ".join(
            f"{NAMES[c]}: {n}" for c, n in counts.items()) + "\nرنگ‌ها، قدیمی به جدید: " + " ← ".join(
                NAMES[c] for c in self.sequence)


def red_run(snapshot):
    """چند دور پیاپی آخر قرمز (کمتر از ۲) بوده است.

    هر دو منبع چک می‌شوند و بیشترین دیده‌شده برگردانده می‌شود؛ هیچ‌کدام
    بیش از طول خودش شمرده نمی‌شود، پس برآورد بالا نمی‌رود.
    """
    def trailing(colors):
        run = 0
        for color in reversed(tuple(colors)):
            if color != "RED":
                break
            run += 1
        return run
    return max(trailing(snapshot.sequence), trailing([c for _, c in snapshot.recent]))


def _walk_chart(board, rows):
    visited, sequence = set(), []
    for start in sorted(col for col, row in board if row == 0):
        col, row = start, 0
        group = board[col, row] != "RED"
        horizontal = False
        while (col, row) in board and (col, row) not in visited:
            color = board[col, row]
            if (color != "RED") != group:
                break
            visited.add((col, row))
            sequence.append(color)
            if not horizontal and row + 1 < rows and (col, row + 1) not in visited:
                candidate = (col, row + 1)
                if candidate not in board:
                    break
                col, row = candidate
            else:
                horizontal = True
                col += 1
    return visited, sequence


def chart_sequence(cells, rows=6):
    """Site trend chart: >=2 (including yellow) shares a run; tails go right.

    Each new run starts at the next top-row column. Descend until the bottom
    or an older tail blocks the cell, then continue horizontally. A column the
    walk cannot reach (a run cut by the strip edge or still being drawn) is
    left out instead of rejecting the whole reading: the model needs the
    ordered outcomes, not every cell.
    """
    board = {(col, row): color for col, row, color in cells}
    while board:
        visited, sequence = _walk_chart(board, rows)
        orphans = set(board) - visited
        if not orphans:
            return tuple(sequence)
        columns = {col for col, _ in orphans}
        board = {cell: color for cell, color in board.items() if cell[0] not in columns}
    raise ValueError("ترتیب جدول مبهم است؛ جدول شش‌ردیفی را کامل انتخاب کنید.")


def read_snapshot(history, chart, reader, minimum=0.55):
    # Dot positions delimit the recent cards. A card is always entering or
    # leaving the strip, so the count is never a fixed five: read however many
    # coloured dots are visible and skip one whose number is not readable yet.
    # Only the coloured text is read; round IDs above it must never be taken
    # for multipliers.
    dots = sorted((x, y, color) for color in ("RED", "GREEN", "YELLOW")
                  for x, y in _centres(history, color, 40))
    recent = []
    for i, (x, y, color) in enumerate(dots):
        end = dots[i + 1][0] - 8 if i + 1 < len(dots) else history.shape[1]
        crop = history[max(0, y):min(history.shape[0], y + 22), x + 8:end]
        if not crop.size:
            continue
        scaled = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        text, confidence = reader(scaled)
        match = re.fullmatch(r"\s*(\d{1,4}\.\d{2})\s*[xX×]?\s*", text)
        if not match or not math.isfinite(confidence) or confidence < minimum:
            continue
        value = float(match[1])
        expected = "RED" if value < 2 else "GREEN" if value < 10 else "YELLOW"
        if value < 1 or expected != color:
            continue
        recent.append((match[1], color))
    if not recent:
        # ترتیب رنگ‌ها از جدول می‌آید؛ یک ضریب تازه هم بهتر از هیچ است.
        raise ValueError("ناحیهٔ ضرایب هیچ ضریب خوانایی ندارد؛ نوار ضرایب را دوباره انتخاب کنید.")
    grid = read_grid(chart)
    if grid is None:
        raise ValueError("جدول رنگ‌ها خوانده نشد؛ ناحیهٔ دوم را دوباره انتخاب کنید.")
    cells = tuple(sorted((d.column, d.row, d.color) for d in grid.dots))
    return HistorySnapshot(tuple(recent), cells, chart_sequence(cells))


def build_advice_prompt(snapshot):
    return (
        "Analyze this fixed snapshot for the round AFTER the displayed history. "
        "History is left-to-right, oldest-to-newest; it lists whichever recent multipliers "
        "were visible (the count can vary), so treat it as the last outcomes. Chart sequence is oldest-to-newest: "
        "start at top-left, descend within each run of <2 or >=2. YELLOW belongs to >=2. "
        "After six rows or an occupied cell, the same run continues right horizontally; "
        "a new run starts at the top of the next column. The chart has ALREADY been decoded "
        "into the chronological sequence supplied below; do not reconstruct coordinates. "
        "RED means an outcome multiplier <2x, "
        "GREEN is >=2x and <10x, YELLOW is >=10x. Past outcomes do not establish "
        "predictability or calibrated probabilities. Never claim guaranteed profit. "
        "Always name the single most likely colour of the three; do not answer UNKNOWN "
        "while any history is supplied. confidence is your own probability from 0 to 1 "
        "that this colour really is the next outcome: give it honestly and use 0.4 or more "
        "only when the supplied pattern supports that colour. "
        "Give a multiplier only if justified, otherwise null. Multiplier is a guessed final "
        "result, NOT a cashout target. "
        "10x refers to multiplier value, NEVER the number of cells or run length. "
        "These two images may represent different moments: never assume they match. "
        "Write reason in Persian only, seven words at most. Return JSON only: {\"color\":\"RED|GREEN|YELLOW\","
        "\"confidence\":0.0,\"multiplier\":null,\"reason\":\"short Persian explanation\"}.\n"
        + json.dumps({"recent": snapshot.recent,
                      "chart_oldest_to_newest": snapshot.sequence}, ensure_ascii=False)
    )


def warm_local(settings):
    """مدل محلی را پیش از تحلیل در حافظه بگذار.

    بارگذاری سرد مدل چند ثانیه است و اگر با همان مهلت کوتاه تحلیل انجام
    شود، درخواست به مسیر ابری می‌افتد؛ یک گرم‌کردن یک‌باره این را حل می‌کند.
    """
    if not (settings.ollama_enabled or settings.ai_provider == "local"):
        return False
    payload = {"model": settings.ollama_model, "stream": False, "think": False,
               "keep_alive": settings.ollama_keep_alive, "options": {"num_predict": 1},
               "messages": [{"role": "user", "content": "."}]}
    call = request.Request(settings.ollama_base_url.rstrip("/") + "/api/chat",
                           data=json.dumps(payload).encode("utf-8"),
                           headers={"Content-Type": "application/json"}, method="POST")
    try:
        with request.urlopen(call, timeout=max(settings.ollama_timeout_ms / 1000,
                                               LOCAL_ADVICE_TIMEOUT_S)) as response:
            json.load(response)
    except Exception:
        return False
    return True


def _ask_local(settings, prompt, schema):
    payload = {"model": settings.ollama_model, "stream": False, "think": False,
               "format": schema, "keep_alive": settings.ollama_keep_alive,
               # طول پاسخ همان هزینهٔ زمانی است: با ۶۴ توکن پاسخ کامل و سالم
               # در حدود ۳ ثانیه می‌رسد؛ با ۳۲ توکن JSON نیمه‌کاره می‌ماند.
               "options": {"temperature": 0.2, "num_predict": 64},
               "messages": [{"role": "user", "content": prompt}]}
    call = request.Request(settings.ollama_base_url.rstrip("/") + "/api/chat",
                           data=json.dumps(payload).encode("utf-8"),
                           headers={"Content-Type": "application/json"}, method="POST")
    timeout = min(settings.ollama_timeout_ms / 1000, LOCAL_ADVICE_TIMEOUT_S)
    with request.urlopen(call, timeout=timeout) as response:
        return json.load(response)["message"]["content"]


def request_advice(settings, prompt, schema=None):
    """مدل محلی اول جواب می‌دهد؛ مسیر ابری فقط پشتیبان است.

    در حالت hybrid درخواست‌های این تب پیش‌تر به مسیر ابری می‌رفتند و ۱۲ تا ۳۸
    ثانیه طول می‌کشیدند یا ۵۰۲ می‌دادند؛ همان مدل محلی همیشه آماده است و پاسخ
    را در حدود دو ثانیه برمی‌گرداند، پس تحلیل قبل از دست همیشه محلی است و
    ابری فقط وقتی محلی خطا بدهد و کاربر local را انتخاب نکرده باشد.
    """
    schema = schema or ADVICE_SCHEMA
    if settings.ollama_enabled or settings.ai_provider == "local":
        try:
            return _ask_local(settings, prompt, schema)
        except Exception:
            if settings.ai_provider == "local" or not settings.smart_review_enabled:
                raise
    return ask(settings.smart_review_url, settings.resolved_key(), settings.smart_review_model,
               prompt, settings.smart_review_timeout_s,
               system="Analyze supplied data only. Return the requested JSON. No tools or actions.")


def _confidence(data):
    """احتمال خود مدل که رنگ درست است؛ نبودش یعنی هیچ ادعایی ندارد."""
    raw = data.get("confidence", data.get("probability"))
    text = str(raw).strip()
    percent = "%" in text
    try:
        value = float(text.replace("%", "").strip())
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value):
        return 0.0
    if percent:
        value /= 100
    elif value > 1:
        # ۴۵ یعنی ۴۵٪؛ عددی مثل ۱٫۲ احتمال بی‌معناست و به سقف بسته می‌شود.
        value = value / 100 if value >= 10 else 1.0
    return min(1.0, max(0.0, value))


def parse_advice(raw):
    data = _json_object(raw)
    if not data:
        raise ValueError("پاسخ مدل JSON قابل خواندن نبود.")
    text = str(data.get("color", data.get("bet", data.get("choice", "")))).strip()
    color = _COLOR_ALIASES.get(text.lower(), _COLOR_ALIASES.get(text, text.upper()))
    if color == "SKIP":
        color = "UNKNOWN"
    if color not in NAMES:
        raise ValueError("پاسخ مدل رنگ معتبر ندارد.")
    confidence = _confidence(data)
    value = data.get("multiplier")
    # ضریب مدل فقط تخمین نتیجهٔ نهایی است و هیچ‌جای کلیک/کش‌اوت استفاده نمی‌شود؛
    # پس عدد نامعتبر یا ناسازگار با رنگ، خود رنگ را باطل نمی‌کند.
    if value is not None and (type(value) not in (int, float)
                              or not math.isfinite(value) or value < 1):
        value = None
    if value is not None:
        expected = "RED" if value < 2 else "GREEN" if value < 10 else "YELLOW"
        if expected != color:
            value = None
    reason = data.get("reason", "")
    if not isinstance(reason, str):
        raise ValueError("توضیح مدل معتبر نیست.")
    return color, value, reason, confidence


class SnapshotAdviceWorker(QThread):
    captured = Signal(str)
    advice = Signal(str, object, float)  # رنگ، ضریب تخمینی و احتمال خود مدل
    paused = Signal(str)  # قرمزهای پیاپی: توقف کلیک تا تغییر روند
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, settings):
        super().__init__()
        self.settings = settings

    def capture_regions(self):
        frames = []
        for region in (self.settings.history_region, self.settings.chart_region):
            with ScreenCapture(region.capture(self.settings.interval_ms)) as capture:
                frame, _ = capture.capture()
                frames.append(frame.image)
        return frames

    def read_with_retry(self, attempts=4, pause=0.6):
        """کارتی که در حال واردشدن/خروج از نوار است یک لحظه بعد خوانا می‌شود."""
        reader = RapidOCRReader()
        last = None
        for attempt in range(attempts):
            frames = self.capture_regions()
            try:
                return frames, read_snapshot(*frames, reader, self.settings.ocr_confidence)
            except ValueError as exc:
                last = exc
                if attempt + 1 < attempts:
                    sleep(pause)
        raise last

    def run(self):
        started = perf_counter()
        try:
            warm_local(self.settings)
            frames, snapshot = self.read_with_retry()
            stamp = datetime.now().strftime("%H:%M:%S")
            streak = red_run(snapshot)
            if streak >= RED_STREAK_LIMIT:
                # مدل لازم نیست: قاعده‌ها می‌گویند با این‌همه قرمز پیاپی، شرط
                # بستن در این یک دقیقه بد است. تا تغییر روند کلیک نمی‌شود.
                append_log(PROJECT_ROOT / "logs" / "snapshot-advice.jsonl",
                           {"time": stamp, "red_streak": streak,
                            "paused_seconds": RED_PAUSE_SECONDS, "error": ""})
                self.paused.emit(
                    snapshot.text() + f"\nزمان عکس: {stamp}\n"
                    f"{streak} دور پیاپی قرمز بوده؛ احتمال باخت بالا است. کلیک تازه برای "
                    f"{int(RED_PAUSE_SECONDS)} ثانیه متوقف شد تا روند تغییر کند.")
                return
            self.captured.emit(snapshot.text() + f"\nزمان عکس: {stamp} — منتظر پاسخ مدل…")
            raw = request_advice(self.settings, build_advice_prompt(snapshot))
            try:
                color, value, reason, confidence = parse_advice(raw)
            except ValueError as exc:
                raise ValueError(f"{exc} پاسخ خام: {raw[:160]!r}") from exc
            value = float(value) if value is not None else None
            self.advice.emit(color, value, confidence)
            append_log(PROJECT_ROOT / "logs" / "snapshot-advice.jsonl",
                       {"time": stamp, "color": color, "multiplier": value,
                        "confidence": confidence,
                        "seconds": round(perf_counter() - started, 2), "error": ""})
            after = self.capture_regions()
            stale = any(not np.array_equal(a, b) for a, b in zip(frames, after))
            result = (f"رنگ تخمینی: {NAMES[color]} | احتمال مدل: {confidence:.0%} | ضریب تخمینی: "
                      + (f"{value:.2f}x" if value is not None else "نامشخص") + "\n" + reason)
            self.done.emit(snapshot.text() + f"\nزمان عکس: {stamp}\n" + result + "\n"
                           + ("تاریخچه هنگام تحلیل تغییر کرد؛ این پیشنهاد قدیمی است. دوباره بررسی کنید." if stale else
                              "این پیشنهاد فقط مربوط به عکس ثبت‌شده است؛ پیش از دست جدید دوباره بررسی کنید."))
        except Exception as exc:
            append_log(PROJECT_ROOT / "logs" / "snapshot-advice.jsonl",
                       {"time": datetime.now().strftime("%H:%M:%S"), "color": None,
                        "multiplier": None, "error": str(exc)})
            self.failed.emit(str(exc))


class SnapshotWatchWorker(SnapshotAdviceWorker):
    """Observe stable pixel changes only: no OCR, model requests or clicks."""
    changed = Signal()

    def __init__(self, settings):
        super().__init__(settings)
        self.stopped = Event()

    def stop(self):
        self.stopped.set()

    def run(self):
        previous = candidate = None
        while not self.stopped.is_set():
            try:
                frames = self.capture_regions()
                digest = hashlib.sha256(b"".join(f.tobytes() for f in frames)).digest()
                if previous is None:
                    previous = digest
                elif digest != previous and digest == candidate:
                    previous = digest
                    self.changed.emit()
                candidate = digest
            except Exception as exc:
                self.failed.emit(str(exc))
                return
            self.stopped.wait(0.5)
