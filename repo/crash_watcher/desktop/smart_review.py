"""AI Analysis: ارسال ناحیه‌های ۱ تا ۴ به مدل زبانی (FreeLLMAPI محلی یا OpenAI).

The bot already reads four screen regions: the history multipliers (1), the
current round's multiplier (2), the colour-dot chart (3) and the row of bet
buttons (4). This module packs them into one compact, human-readable prompt and
sends it to any OpenAI-compatible server — the local FreeLLMAPI on
`http://127.0.0.1:31415/v1` or the official OpenAI API on
`https://api.openai.com/v1` — then parses the model's JSON answer into a bet.

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
# API رسمی OpenAI؛ کلید از platform.openai.com گرفته می‌شود.
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_MODEL = "gpt-4o-mini"
# بارگذاری سرد مدل محلی می‌تواند بیش از یک دقیقه طول بکشد؛ مهلت ۴۵ ثانیه‌ای
# باعث می‌شد درخواست‌ها بی‌دلیل با «timed out» شکست بخورند.
DEFAULT_TIMEOUT = 120.0
# «کلیک کلاسیک»: شرط باید همیشه زیر این ضریب بسته شود.
CASHOUT_LIMIT = 1.99

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
    "تو مغز تحلیل یک بازی Crash (Trenball) هستی. هیچ قاعدهٔ آماده‌ای نداری؛ فقط دادهٔ "
    "خام همین لحظه را داری: ترتیب ضرایب گذشته، رنگ آن‌ها و شبکهٔ رنگ نمودار. خودت "
    "الگو را کشف کن، ادعای قطعی نکن و پاسخ را همیشه در قالب JSON بده."
)


def chat_completions_url(base_url: str) -> str:
    """Accept `http://host:port`, `.../v1` or a full chat-completions URL."""
    url = (base_url or "").strip().rstrip("/")
    if not url:
        raise ValueError("آدرس سرور خالی است")
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


def _color_of(text: str, threshold: str) -> str:
    """رنگ هر ضریب از روی خود عدد، تا مدل توالی رنگ‌ها را ببیند."""
    try:
        value = float(str(text).replace(",", ""))
        limit = float(str(threshold).replace(",", ""))
    except (TypeError, ValueError):
        return "?"
    if value >= limit:
        return "زرد"
    return "قرمز" if value < 2 else "سبز"


def build_prompt(snapshot: SmartSnapshot, threshold: str = "10.00",
                 extra_rules: str = "", *, min_confidence: float = 0.40,
                 allow_two: bool = True, classic: bool = False) -> str:
    """The exact user message sent to the model: raw data, no fixed rules."""
    history = " | ".join(f"{index}. {value} ({_color_of(value, threshold)})"
                         for index, value in enumerate(snapshot.recent[-30:], 1)) or "ندارد"
    bets = "، ".join(f"{_LETTER_FA.get(_LETTER.get(color, ''), color)} "
                     f"({payout})" for color, payout in snapshot.bets)
    lines = [
        "تو مغز تحلیل این ربات هستی. هیچ قاعدهٔ آماده و هیچ فرض ثابتی نداری؛ فقط همین "
        "دادهٔ خام را داری و باید خودت الگو را از دل آن پیدا کنی.",
        "",
        "داده‌های این لحظه:",
        f"۱) ضرایب دست‌های بسته‌شده به ترتیب زمان (قدیمی ← جدید): {history}",
        f"۲) ضریب دست فعلی/آخر: {snapshot.current or 'خوانده نشد'}",
        "۳) شبکهٔ رنگ نمودار (ستون ۱ جدیدترین است؛ هر ستون از بالا جدید به پایین قدیمی):",
        f"{_grid_text(snapshot)}",
        f"۴) دکمه‌های شرط این دست: {bets}",
        f"۵) تعداد دورهای دیده‌شده در این جلسه: {snapshot.rounds_seen}",
        f"۶) رنگ هر ضریب: کمتر از ۲ قرمز، از ۲ تا {threshold} سبز، {threshold} و بیشتر زرد (مون).",
        "",
        "خودت این داده‌ها را با دقت بررسی کن و به نتیجه برس:",
        "- ترتیب و روند عددها: صعودی یا نزولی بودن، جهش‌های بزرگ، برگشت به محدودهٔ قبلی و "
        "فاصلهٔ تکرار هر رنگ.",
        "- توالی رنگ‌ها در ضرایب گذشته و در ستون‌های نمودار: رنگ‌های پشت‌سرهم، جای خالی‌ها و "
        "شکل ستون‌ها.",
        "- شباهت همین لحظه با نمونه‌های مشابه در همین فهرست و اینکه آن نمونه‌ها بعدش چه رنگی دادند.",
        "- نتیجه باید یک پیش‌بینی عملی برای دست بعد باشد، نه توضیح کلی.",
        f"- احتمال پیش‌بینی‌ات را صادقانه بده؛ اگر از {min_confidence:.0%} کمتر است bet را "
        "SKIP بگذار. لازم نیست هر دست بازی کنی.",
    ]
    if allow_two:
        lines.append("- اگر بین دو رنگ شک داری می‌توانی هر دو را بزنی: رنگ دوم را در second بگذار.")
    if classic:
        lines.append(
            "- خروج کلاسیک: ربات پیش از پایان ۶ ثانیه با یک کلیک روی دکمهٔ big Bet / Cash Out "
            "وارد بازی می‌شود و بعد با کلیک دوباره خارج می‌شود. در فیلد exit عدد خروج را خودت "
            f"انتخاب کن؛ هرچه ریسک را کمتر می‌بینی پایین‌تر (مثلاً 1.30) و هرچه مطمئن‌تری "
            f"بالاتر (مثلاً 1.85)، ولی همیشه زیر {CASHOUT_LIMIT} (حداکثر 1.98).")
    if extra_rules.strip():
        lines.append("- نکتهٔ اختیاری کاربر: " + extra_rules.strip() + " (اجباری نیست؛ داده بر آن مقدم است)")
    lines += ["", "برای دست بعد کدام رنگ بیشترین شانس را دارد؟ فقط JSON برگردان، بدون توضیح اضافه:"]
    answer = '{"bet":"RED|GREEN|YELLOW|SKIP",'
    if allow_two:
        answer += '"second":"NONE|RED|GREEN|YELLOW",'
    if classic:
        answer += '"exit":1.85,'
    answer += '"confidence":0.0,"reason":"یک جمله فارسی"}'
    lines.append(answer)
    return "\n".join(lines)


@dataclass(frozen=True)
class SmartVerdict:
    bet: str          # RED / GREEN / YELLOW / SKIP
    confidence: float  # 0..1
    reason: str
    second: str | None = None  # رنگ دوم اختیاری (پوشش ضرر)
    exit_at: float | None = None  # ضریب خروج کلاسیک، همیشه زیر ۱٫۹۹

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
    return SmartVerdict(color, confidence, reason, second, _exit_target(payload))


def _exit_target(payload: dict) -> float | None:
    """ضریب خروجی که مدل برای «کلیک کلاسیک» انتخاب کرده؛ هرگز ۱٫۹۹ یا بیشتر."""
    raw = payload.get("exit", payload.get("cashout", payload.get("cash_out", payload.get("target"))))
    if raw is None:
        return None
    try:
        value = float(str(raw).replace("%", "").strip())
    except (TypeError, ValueError):
        return None
    if not 1.0 < value < 10.0:
        return None
    return min(value, CASHOUT_LIMIT - 0.01)


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


def _error_message(exc: Exception, base_url: str, timeout: float = DEFAULT_TIMEOUT) -> str:
    local = _is_local(base_url)
    if isinstance(exc, TimeoutError):
        hint = ("مطمئن شوید FreeLLMAPI روی همین آدرس باز است و صف درخواست ندارد؛ "
                "یک مدل سبک‌تر بگذارید." if local else
                "اتصال اینترنت یا صف API را بررسی کنید و دوباره تلاش کنید.")
        return (f"سرور در مهلت {timeout:.0f} ثانیه پاسخ نداد ({base_url}). "
                f"اگر مدل تازه در حال بارگذاری است یا سنگین است، {hint}")
    if isinstance(exc, urlerror.HTTPError):
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace")[:300]
        except Exception:
            detail = ""
        if exc.code in (401, 403):
            who = ("کلید یکپارچهٔ داشبورد FreeLLMAPI" if local
                   else "کلید رسمی OpenAI (sk-...)")
            return f"کلید پذیرفته نشد (HTTP {exc.code})؛ {who} را بررسی کنید."
        return f"خطای HTTP {exc.code} از {base_url}: {detail}".strip()
    if isinstance(exc, urlerror.URLError):
        if local:
            return (f"اتصال به {base_url} برقرار نشد ({exc.reason}). "
                    "مطمئن شوید FreeLLMAPI روی همان پورت در حال اجراست.")
        return (f"اتصال به {base_url} برقرار نشد ({exc.reason}). "
                "اینترنت، VPN یا صحت آدرس را بررسی کنید.")
    return f"خطا در تماس با {base_url}: {exc}"


def _model_name(base_url: str, model: str) -> str:
    """`auto` مخصوص سرور محلی است؛ API رسمی OpenAI نام مدل واقعی می‌خواهد."""
    name = (model or "").strip()
    if name and name.lower() != "auto":
        return name
    return DEFAULT_MODEL if _is_local(base_url) else OPENAI_MODEL


def _headers(api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"
    return headers


def ask(base_url: str, api_key: str, model: str, prompt: str,
        timeout: float = DEFAULT_TIMEOUT, system: str = SYSTEM_PROMPT) -> str:
    """POST one chat completion and return the assistant text."""
    url = chat_completions_url(base_url)
    payload = {
        "model": _model_name(url, model),
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
        raise RuntimeError(_error_message(exc, url, timeout)) from exc
    try:
        parsed = json.loads(body)
        return str(parsed["choices"][0]["message"]["content"])
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("پاسخ سرور شکل chat/completions نداشت") from exc


def review(base_url: str, api_key: str, model: str, snapshot: SmartSnapshot,
           timeout: float = DEFAULT_TIMEOUT, threshold: str = "10.00",
           extra_rules: str = "", *, min_confidence: float = 0.40,
           allow_two: bool = True, classic: bool = False) -> tuple[SmartVerdict | None, str, str]:
    """Ask for a verdict: (parsed, raw answer, prompt sent)."""
    prompt = build_prompt(snapshot, threshold, extra_rules, min_confidence=min_confidence,
                          allow_two=allow_two, classic=classic)
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
        raise RuntimeError(_error_message(exc, url, timeout)) from exc
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
