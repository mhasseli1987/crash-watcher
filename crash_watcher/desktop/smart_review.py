"""بررسی هوشمند ضریب‌ها: ارسال ناحیه‌های ۱ تا ۴ به مدل زبانی محلی.

The bot already reads four screen regions: the history multipliers (1), the
current round's multiplier (2), the colour-dot chart (3) and the row of bet
buttons (4). This module packs them into one compact, human-readable prompt and
sends it to a local OpenAI-compatible server — the user runs FreeLLMAPI on
`http://127.0.0.1:31415/v1` — then parses the model's JSON answer into a bet.

Nothing here touches the mouse: the verdict is advice, and the clicker keeps
following its own rules.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
from urllib import error as urlerror
from urllib import request as urlrequest

DEFAULT_BASE_URL = "http://127.0.0.1:31415/v1"
DEFAULT_MODEL = "auto"

# Region 4 is the row of three buttons the user marked: red, green, moon.
DEFAULT_BETS: tuple[tuple[str, str], ...] = (("RED", "2x"), ("GREEN", "2x"), ("YELLOW", "10x"))

_LETTER = {"RED": "R", "GREEN": "G", "YELLOW": "Y"}
_LETTER_FA = {"R": "قرمز", "G": "سبز", "Y": "زرد"}
_COLOR_ALIASES = {"قرمز": "RED", "رد": "RED", "red": "RED", "r": "RED",
                  "سبز": "GREEN", "green": "GREEN", "g": "GREEN",
                  "زرد": "YELLOW", "مون": "YELLOW", "moon": "YELLOW",
                  "yellow": "YELLOW", "y": "YELLOW",
                  "هیچ": "SKIP", "بدون": "SKIP", "skip": "SKIP", "none": "SKIP",
                  "صبر": "SKIP", "wait": "SKIP"}

SYSTEM_PROMPT = (
    "تو دستیار تحلیل یک بازی Crash (Trenball) هستی. ورودی، وضعیت آخرین دست‌ها و "
    "شبکهٔ رنگ نمودار است. فقط بر اساس همین داده و قواعدی که کاربر داده تحلیل کن؛ "
    "هیچ ادعای قطعی نکن و پاسخ را همیشه در قالب JSON بده."
)


def chat_completions_url(base_url: str) -> str:
    """Accept `http://host:port`, `.../v1` or a full chat-completions URL."""
    url = (base_url or "").strip().rstrip("/")
    if not url:
        raise ValueError("آدرس FreeLLMAPI خالی است")
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError("آدرس باید با http:// یا https:// شروع شود")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return url + "/chat/completions"
    return url + "/v1/chat/completions"


def models_url(base_url: str) -> str:
    url = chat_completions_url(base_url)
    return url[: -len("/chat/completions")] + "/models"


def _is_local(url: str) -> bool:
    host = url.split("//", 1)[-1].split("/", 1)[0].split(":", 1)[0].lower()
    return host in ("127.0.0.1", "localhost", "::1", "[::1]")


@dataclass(frozen=True)
class SmartSnapshot:
    """What the user marked on the screen, already reduced to text."""
    recent: tuple[str, ...] = ()            # region 1: multipliers, oldest first
    current: str | None = None              # region 2: the last closed round
    grid_columns: tuple[tuple[str, ...], ...] = ()   # region 3: R/G/Y per column
    bets: tuple[tuple[str, str], ...] = DEFAULT_BETS  # region 4: button -> payout
    rounds_seen: int = 0

    @property
    def empty(self) -> bool:
        return not self.recent and self.current is None and not self.grid_columns


def grid_columns(reading, max_columns: int = 8) -> tuple[tuple[str, ...], ...]:
    """Turn a GridReading into letters, newest column first, top to bottom."""
    if reading is None or getattr(reading, "columns", 0) < 1:
        return ()
    columns = []
    for column in range(min(reading.columns, max_columns)):
        cells = []
        for row in range(reading.rows):
            color = reading.color_at(column, row)
            if color is None:
                break
            cells.append(_LETTER.get(color, "?"))
        columns.append(tuple(cells))
    return tuple(columns)


def _grid_text(snapshot: SmartSnapshot) -> str:
    if not snapshot.grid_columns:
        return "شبکهٔ رنگ خوانده نشد"
    lines = []
    for index, column in enumerate(snapshot.grid_columns, 1):
        letters = " ".join(_LETTER_FA.get(cell, "?") for cell in column)
        lines.append(f"  ستون {index} (جدیدترین بالا): {letters}")
    return "\n".join(lines)


# خروج هوشمند: هدف همیشه بین ۱٫۴۰ و ۱٫۹۰ است و هیچ شرطی از ۱٫۹۰ جلوتر باز نمی‌ماند.
EXIT_MIN = 1.40
EXIT_MAX = 1.90


def build_prompt(snapshot: SmartSnapshot, threshold: str = "10.00",
                 extra_rules: str = "", *, min_confidence: float = 0.40,
                 allow_two: bool = True) -> str:
    """The exact user message sent to the model."""
    recent = "، ".join(snapshot.recent[-20:]) or "ندارد"
    bets = "، ".join(f"{_LETTER_FA.get(_LETTER.get(color, ''), color)} "
                     f"({payout})" for color, payout in snapshot.bets)
    rules = [
        "- قرمز: ضریب کمتر از ۲ | سبز: بین ۲ و آستانهٔ زرد | زرد (مون): ≥ " + threshold,
        "- هفت دست صبر بعد از هر ضریب زرد، سپس سه کلیک",
        "- جفت سبز یا جفت قرمز پشت‌سرهم → رنگ مخالف",
        "- یک رنگ بیشتر از ۶ بار پشت‌سرهم → همان رنگ برای ده دست",
        "- سطر زرد با چهار ستون فاصله (چهار ستون آخر، چهار سطر اول) → زرد همان دست",
        "- ضرایب 1.00، 1.11 و 1.99 → سبز؛ 2.08 → زرد",
        f"- فقط وقتی شرط ببند که احتمالِ درست بودن پیش‌بینی‌ات {min_confidence:.0%} یا بیشتر باشد؛ در غیر این صورت bet را SKIP بگذار",
        "- لازم نیست هر دست بازی کنی؛ دست‌های بی‌اطمینان را رد کن",
        "- کش‌اوت: اگر شرط بستی، ضریبی بین 1.40 و 1.90 انتخاب کن که روی آن از دست خارج شوی",
        "- اگر آخرین ضریب بسته‌شده زیر 1.10 بود (مثلاً 1.08 یا 1.07) می‌توانی برای دست بعد تا 1.90 صبر کنی؛ "
        "در هر حالتی دیرترین خروج 1.90 است، نه بیشتر",
    ]
    if allow_two:
        rules.append("- اگر بین دو رنگ شک داری می‌توانی هر دو را بزنی: رنگ دوم را در second بگذار. "
                     "چون پرداخت هر رنگ دست‌کم دو برابر است، شرط روی دو رنگ در بدترین حالت "
                     "سربه‌سر می‌شود و ضرر نمی‌دهد؛ برای ضریب زرد که احتمال کمتری دارد همین را در نظر بگیر")
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
        "قواعد کاربر:\n" + "\n".join(rules) + "\n\n"
        "برای دست بعد کدام رنگ بیشترین شانس را دارد؟ فقط JSON برگردان، بدون توضیح اضافه:\n"
        + ('{"bet":"RED|GREEN|YELLOW|SKIP","second":"NONE|RED|GREEN|YELLOW","cashout":1.45,'
           '"confidence":0.0,"reason":"یک جمله فارسی"}' if allow_two else
           '{"bet":"RED|GREEN|YELLOW|SKIP","cashout":1.45,"confidence":0.0,"reason":"یک جمله فارسی"}')
    )


@dataclass(frozen=True)
class SmartVerdict:
    bet: str          # RED / GREEN / YELLOW / SKIP
    confidence: float  # 0..1
    reason: str
    second: str | None = None  # رنگ دوم اختیاری (پوشش ضرر)
    cashout: float | None = None  # ضریب خروج پیشنهادی مدل، بین 1.40 و 1.90

    @property
    def bets(self) -> tuple[str, ...]:
        """رنگ‌هایی که باید هم‌زمان زده شوند."""
        if self.bet == "SKIP":
            return ()
        return (self.bet,) if self.second is None else (self.bet, self.second)

    @property
    def label(self) -> str:
        names = {"RED": "قرمز", "GREEN": "سبز", "YELLOW": "زرد (مون)",
                 "SKIP": "بدون شرط"}
        color = " + ".join(names.get(bet, bet) for bet in self.bets) or "بدون شرط"
        return f"{color} — اطمینان {self.confidence:.0%}"


def _json_object(text: str) -> dict | None:
    if not isinstance(text, str):
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = [fenced[1]] if fenced else []
    # مدل‌های "auto" گاهی قبل از JSON توضیح می‌نویسند و ممکن است آکولاد دیگری
    # هم در متن باشد؛ همهٔ اشیای کوتاه را امتحان می‌کنیم تا اولین JSON معتبر پیدا شود.
    candidates += [match[0] for match in re.finditer(r"\{.*?\}", text, re.DOTALL)]
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict):
            return value
    return None


def parse_verdict(text: str) -> SmartVerdict | None:
    """Read the model's JSON answer; None when it cannot be understood."""
    payload = _json_object(text)
    if payload is None:
        return None
    bet = payload.get("bet", payload.get("color", payload.get("choice", "")))
    bet = str(bet).strip()
    color = _COLOR_ALIASES.get(bet.lower(), _COLOR_ALIASES.get(bet, bet.upper()))
    if color not in ("RED", "GREEN", "YELLOW", "SKIP"):
        return None
    raw_confidence = payload.get("confidence", payload.get("probability", 0))
    try:
        confidence = float(str(raw_confidence).replace("%", "").strip())
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence > 1:
        confidence /= 100
    confidence = min(1.0, max(0.0, confidence))
    reason = str(payload.get("reason", payload.get("why", ""))).strip()
    second = _second_color(payload, color)
    cashout = payload.get("cashout", payload.get("cash_out", payload.get("exit")))
    try:
        cashout = float(str(cashout).replace("x", "").strip())
        if not 1.0 < cashout < 10.0:
            cashout = None
    except (TypeError, ValueError):
        cashout = None
    if cashout is not None:
        # هر عددی که مدل بدهد داخل بازهٔ خروج ۱٫۴۰ تا ۱٫۹۰ بسته می‌شود.
        cashout = round(min(max(cashout, EXIT_MIN), EXIT_MAX), 2)
    return SmartVerdict(color, confidence, reason, second, cashout)


def _second_color(payload: dict, first: str) -> str | None:
    """رنگ دومی که مدل برای پوشش ضرر پیشنهاد کرده، اگر معتبر باشد."""
    raw = payload.get("second", payload.get("also", payload.get("hedge")))
    if raw is None:
        bets = payload.get("bets")
        if isinstance(bets, list) and len(bets) > 1:
            raw = bets[1]
    if raw is None:
        return None
    text = str(raw).strip()
    color = _COLOR_ALIASES.get(text.lower(), _COLOR_ALIASES.get(text, text.upper()))
    if color not in ("RED", "GREEN", "YELLOW") or color == first:
        return None
    return color


def _error_message(exc: Exception, base_url: str) -> str:
    if isinstance(exc, urlerror.HTTPError):
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace")[:300]
        except Exception:
            detail = ""
        if exc.code in (401, 403):
            return f"کلید FreeLLMAPI پذیرفته نشد (HTTP {exc.code})؛ کلید یکپارچهٔ داشبورد را بگذارید."
        return f"خطای HTTP {exc.code} از {base_url}: {detail}".strip()
    if isinstance(exc, urlerror.URLError):
        return (f"اتصال به {base_url} برقرار نشد ({exc.reason}). "
                "مطمئن شوید FreeLLMAPI روی همان پورت در حال اجراست.")
    return f"خطا در تماس با {base_url}: {exc}"


def _headers(api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"
    return headers


def ask(base_url: str, api_key: str, model: str, prompt: str,
        timeout: float = 45.0, system: str = SYSTEM_PROMPT) -> str:
    """POST one chat completion and return the assistant text."""
    url = chat_completions_url(base_url)
    payload = {
        "model": model.strip() or DEFAULT_MODEL,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": prompt}],
        "temperature": 0.2,
        "stream": False,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urlrequest.Request(url, data=data, headers=_headers(api_key), method="POST")
    try:
        with urlrequest.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI as a message
        raise RuntimeError(_error_message(exc, url)) from exc
    try:
        parsed = json.loads(body)
        return str(parsed["choices"][0]["message"]["content"])
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("پاسخ سرور شکل chat/completions نداشت") from exc


def review(base_url: str, api_key: str, model: str, snapshot: SmartSnapshot,
           timeout: float = 45.0, threshold: str = "10.00",
           extra_rules: str = "", *, min_confidence: float = 0.40,
           allow_two: bool = True) -> tuple[SmartVerdict | None, str, str]:
    """Ask for a verdict: (parsed, raw answer, prompt sent)."""
    prompt = build_prompt(snapshot, threshold, extra_rules,
                          min_confidence=min_confidence, allow_two=allow_two)
    raw = ask(base_url, api_key, model, prompt, timeout)
    return parse_verdict(raw), raw, prompt


def probe_models(base_url: str, api_key: str, timeout: float = 15.0) -> list[str]:
    """GET /v1/models: proves the port answers and the key is accepted."""
    url = models_url(base_url)
    request = urlrequest.Request(url, headers=_headers(api_key), method="GET")
    try:
        with urlrequest.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(_error_message(exc, url)) from exc
    try:
        data = json.loads(body)
        return [str(item.get("id", "")) for item in data.get("data", []) if item.get("id")]
    except (ValueError, AttributeError, TypeError):
        return []


def append_log(path: Path, entry: dict) -> None:
    """One JSON line per review so a session can be audited afterwards."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"time": datetime.now().isoformat(timespec="seconds"), **entry}
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass  # a log must never break monitoring
