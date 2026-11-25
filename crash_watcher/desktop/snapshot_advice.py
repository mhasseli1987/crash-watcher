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
from .smart_review import ask, _json_object, append_log
from crash_watcher.vision.capture import ScreenCapture
from crash_watcher.vision.recognition import RapidOCRReader

NAMES = {"RED": "نارنجی / قرمز", "GREEN": "سبز", "YELLOW": "زرد", "UNKNOWN": "نامشخص"}


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
        "Give a tentative color and multiplier estimate only if justified, otherwise UNKNOWN "
        "and null. Multiplier is a guessed final result, NOT a cashout target. "
        "10x refers to multiplier value, NEVER the number of cells or run length. "
        "These two images may represent different moments: never assume they match. "
        "Write reason in Persian only, twelve words at most. Return JSON only: {\"color\":\"RED|GREEN|YELLOW|UNKNOWN\","
        "\"multiplier\":null,\"reason\":\"short Persian explanation\"}.\n"
        + json.dumps({"recent": snapshot.recent,
                      "chart_oldest_to_newest": snapshot.sequence}, ensure_ascii=False)
    )


def request_advice(settings, prompt, schema=None):
    if settings.ai_provider != "local":
        return ask(settings.smart_review_url, settings.resolved_key(), settings.smart_review_model,
                   prompt, settings.smart_review_timeout_s,
                   system="Analyze supplied data only. Return the requested JSON. No tools or actions.")
    schema = schema or {"type": "object", "properties": {
        "color": {"type": "string", "enum": list(NAMES)},
        "multiplier": {"type": ["number", "null"], "minimum": 1},
        "reason": {"type": "string", "maxLength": 120}},
        "required": ["color", "multiplier", "reason"], "additionalProperties": False}
    payload = {"model": settings.ollama_model, "stream": False, "think": False,
               "format": schema, "keep_alive": settings.ollama_keep_alive,
               "options": {"temperature": 0.2, "num_predict": 160},
               "messages": [{"role": "user", "content": prompt}]}
    call = request.Request(settings.ollama_base_url.rstrip("/") + "/api/chat",
                           data=json.dumps(payload).encode("utf-8"),
                           headers={"Content-Type": "application/json"}, method="POST")
    with request.urlopen(call, timeout=settings.ollama_timeout_ms / 1000) as response:
        return json.load(response)["message"]["content"]


def parse_advice(raw):
    data = _json_object(raw)
    if not data or data.get("color") not in NAMES:
        raise ValueError("پاسخ مدل رنگ معتبر ندارد.")
    color, value = data["color"], data.get("multiplier")
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
    return color, value, reason


class SnapshotAdviceWorker(QThread):
    captured = Signal(str)
    advice = Signal(str, object)  # رنگ و ضریب پیشنهادی برای دست بعد
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
            frames, snapshot = self.read_with_retry()
            stamp = datetime.now().strftime("%H:%M:%S")
            self.captured.emit(snapshot.text() + f"\nزمان عکس: {stamp} — منتظر پاسخ مدل…")
            raw = request_advice(self.settings, build_advice_prompt(snapshot))
            try:
                color, value, reason = parse_advice(raw)
            except ValueError as exc:
                raise ValueError(f"{exc} پاسخ خام: {raw[:160]!r}") from exc
            self.advice.emit(color, float(value) if value is not None else None)
            append_log(PROJECT_ROOT / "logs" / "snapshot-advice.jsonl",
                       {"time": stamp, "color": color,
                        "multiplier": float(value) if value is not None else None,
                        "seconds": round(perf_counter() - started, 2), "error": ""})
            after = self.capture_regions()
            stale = any(not np.array_equal(a, b) for a, b in zip(frames, after))
            result = (f"رنگ تخمینی: {NAMES[color]} | ضریب تخمینی: "
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
