"""Project-local validated desktop settings."""
from dataclasses import asdict, dataclass, field, fields
from decimal import Decimal, InvalidOperation
import json
import os
import sys
from pathlib import Path
from crash_watcher.core.config import StrategyConfig, positive_number
from crash_watcher.vision.capture import CaptureConfig
from .secrets_store import protect, unprotect

PROJECT_ROOT = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[2]
SETTINGS_PATH = PROJECT_ROOT / "settings.json"
# Permanent rule: no click may be issued more than one second after the round
# closes, so the window before every click is capped at this value.
MAX_CLICK_DELAY_MS = 1000

@dataclass(frozen=True)
class Region:
    monitor: int
    x: int
    y: int
    width: int
    height: int

    def capture(self, interval_ms: int) -> CaptureConfig:
        return CaptureConfig(self.monitor, self.x, self.y, self.width, self.height, interval_ms)

@dataclass(frozen=True)
class DesktopSettings:
    result_region: Region | None = None
    history_region: Region | None = None
    click_region: Region | None = None
    chart_region: Region | None = None
    bet_region: Region | None = None  # the row of the three bet buttons (region 4)
    click_point: tuple[int, int] | None = None
    bet_point_red: tuple[int, int] | None = None
    bet_point_green: tuple[int, int] | None = None
    bet_point_yellow: tuple[int, int] | None = None
    # کش‌اوت هوشمند: بعد از هر شرط، مدل تعیین می‌کند روی چه ضریبی خارج شود.
    cashout_enabled: bool = False
    autoclick_enabled: bool = False
    autoclick_mode: str = "detect"
    autoclick_threshold: str = "10.00"
    autoclick_cooldown_ms: int = 1500
    autoclick_delay_ms: int = MAX_CLICK_DELAY_MS
    autoclick_max_clicks: int = 3
    autoclick_skip_rounds: int = 7
    autoclick_dry_run: bool = True
    # حالت سایه: همهٔ تصمیم‌ها و گزارش‌ها مثل حالت واقعی ساخته می‌شوند اما هیچ
    # کلیکی فرستاده نمی‌شود؛ برای سنجش عملکرد بدون ریسک.
    shadow_mode: bool = False
    autoclick_restore_cursor: bool = True
    smart_click_enabled: bool = False
    prediction_min_confidence: float = 0.60
    # «بررسی هوشمند ضریب‌ها»: پرسش از سرور محلی FreeLLMAPI (OpenAI-compatible).
    smart_review_enabled: bool = False
    smart_review_url: str = "http://127.0.0.1:31415/v1"
    smart_review_key: str = ""
    # کلید می‌تواند در متغیر محیطی بماند و اصلاً در settings.json ذخیره نشود.
    smart_review_key_env: str = "CRASH_WATCHER_API_KEY"
    smart_review_model: str = "auto"
    # فهرست مدل‌ها برای مقایسهٔ موازی؛ خالی = فقط همان smart_review_model.
    smart_compare_models: str = ""
    smart_review_timeout_s: int = 45
    # بودجهٔ تأخیر: پاسخی که دیرتر از این برسد برای دست بعد کهنه است.
    smart_latency_budget_ms: int = 4000
    smart_review_every_rounds: int = 1
    smart_review_rules: str = ""
    ollama_enabled: bool = False
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3:1.7b"
    ollama_timeout_ms: int = 4000
    ollama_keep_alive: str = "30s"
    # «ورود هوشمند به بازی»: اول یادگیری، بعد شرط‌بندی بر پایهٔ تصمیم آمادهٔ مدل.
    smart_play_enabled: bool = False
    smart_play_learning_minutes: int = 30
    smart_play_min_confidence: float = 0.40
    smart_play_two_colors: bool = True
    # حالت ربات: "classic" (قواعد فعلی) یا "turnbal" (تصمیم هوش مصنوعی + شرط سه‌رنگ)
    bot_mode: str = "classic"
    # «حالت ترنبال»: کالیبراسیون شرط سه‌رنگ؛ هر رنگ جدا سنجیده می‌شود
    turnbal_red_enabled: bool = True
    turnbal_green_enabled: bool = True
    turnbal_yellow_enabled: bool = False
    turnbal_red_min_conf: float = 0.55
    turnbal_green_min_conf: float = 0.60
    turnbal_yellow_min_conf: float = 0.75
    turnbal_red_attempts: int = 4
    turnbal_green_attempts: int = 6
    turnbal_yellow_attempts: int = 2
    turnbal_cashout_min: float = 1.40
    turnbal_cashout_max: float = 1.90
    turnbal_min_cycle_win: float = 0.50
    turnbal_max_streak: int = 3
    turnbal_min_gap: int = 2
    turnbal_emergency_exit: float = 1.40
    pair_flip_enabled: bool = True
    row_signal_enabled: bool = True
    row_signal_gap: int = 4
    row_signal_rows: int = 4
    row_signal_recent_columns: int = 4
    row_signal_clicks: int = 1
    streak_signal_enabled: bool = True
    streak_signal_length: int = 6
    streak_signal_clicks: int = 10
    value_bet_enabled: bool = True
    value_bets: dict[str, str] = field(default_factory=lambda: {
        "1.00": "GREEN", "1.11": "GREEN", "1.99": "GREEN", "2.08": "YELLOW"})
    review_window_s: int = 1800
    # استاپ‌لاس سرمایه: هیچ پولی جابه‌جا نمی‌شود؛ فقط سنجهٔ ریسک جلسه است.
    bankroll_enabled: bool = False
    bankroll_start: str = "100"
    bankroll_stop_loss_pct: float = 20.0
    payout_red: float = 2.0
    payout_green: float = 2.0
    payout_yellow: float = 10.0
    fixed_bet_amount: str = "10"
    # «2 /2»: دوبل کردن مبلغ شرط برای جبران باخت (تا ۴ بار) با دکمهٔ ×2 بازی
    # و برگشت به مبلغ پایه با دکمهٔ ÷2 پس از برد.
    double_bet_enabled: bool = False
    double_bet_max: int = 4
    double_bet_point: tuple[int, int] | None = None
    halve_bet_point: tuple[int, int] | None = None
    max_attempts: int = 8
    yellow_threshold: str = "10.00"
    red_threshold: str = "2.00"
    interval_ms: int = 100
    ocr_confidence: float = 0.55
    theme: str = "dark"
    tesseract_path: str = ""

    def resolved_key(self) -> str:
        """کلید مؤثر: متغیر محیطی بر مقدار ذخیره‌شده اولویت دارد."""
        name = (self.smart_review_key_env or "").strip()
        if name:
            value = os.environ.get(name, "").strip()
            if value:
                return value
        return self.smart_review_key

    @property
    def simulated(self) -> bool:
        """هیچ کلیکی نباید فرستاده شود."""
        return bool(self.autoclick_dry_run or self.shadow_mode)

    def strategy(self) -> StrategyConfig:
        return StrategyConfig(
            Decimal(self.fixed_bet_amount),
            self.max_attempts,
            Decimal(self.yellow_threshold),
            Decimal(self.red_threshold),
        )

    def validate(self) -> None:
        """Reject settings that cannot build a usable session; raise ValueError."""
        self.strategy()
        if type(self.interval_ms) is not int or self.interval_ms < 20:
            raise ValueError("interval_ms must be an integer of at least 20")
        for name in ("result_region", "history_region", "click_region", "chart_region",
                     "bet_region"):
            region = getattr(self, name)
            if region is None:
                continue
            if not isinstance(region, Region):
                raise ValueError(f"{name} must be a Region")
            region.capture(self.interval_ms)
        for name in ("click_point", "bet_point_red", "bet_point_green", "bet_point_yellow",
                     "double_bet_point", "halve_bet_point"):
            point = getattr(self, name)
            if point is None:
                continue
            if (not isinstance(point, tuple) or len(point) != 2
                    or any(type(value) is not int for value in point)):
                raise ValueError(f"{name} must be a pair of integers")
        if not 0 < self.ocr_confidence <= 1:
            raise ValueError("ocr_confidence must be between 0 and 1")
        if self.autoclick_mode not in ("detect", "point"):
            raise ValueError("autoclick_mode must be 'detect' or 'point'")
        try:
            threshold = Decimal(self.autoclick_threshold)
        except (InvalidOperation, TypeError, ValueError) as exc:
            raise ValueError("autoclick_threshold must be a positive number") from exc
        positive_number(threshold, "autoclick_threshold")
        if type(self.autoclick_cooldown_ms) is not int or self.autoclick_cooldown_ms < 0:
            raise ValueError("autoclick_cooldown_ms must be a non-negative integer")
        if (type(self.autoclick_delay_ms) is not int
                or not 0 <= self.autoclick_delay_ms <= MAX_CLICK_DELAY_MS):
            raise ValueError(f"autoclick_delay_ms must be an integer between 0 and {MAX_CLICK_DELAY_MS}")
        if type(self.autoclick_max_clicks) is not int or self.autoclick_max_clicks <= 0:
            raise ValueError("autoclick_max_clicks must be a positive integer")
        if type(self.autoclick_skip_rounds) is not int or self.autoclick_skip_rounds < 0:
            raise ValueError("autoclick_skip_rounds must be a non-negative integer")
        if not 0.5 <= self.prediction_min_confidence <= 1:
            raise ValueError("prediction_min_confidence must be between 0.5 and 1")
        if type(self.smart_review_enabled) is not bool:
            raise ValueError("smart_review_enabled must be a boolean")
        if type(self.ollama_enabled) is not bool:
            raise ValueError("ollama_enabled must be a boolean")
        from urllib.parse import urlsplit
        ollama_url = None
        try:
            ollama_url = urlsplit(self.ollama_base_url)
            local_host = ollama_url.hostname in ("127.0.0.1", "localhost", "::1")
        except (TypeError, ValueError, AttributeError):
            local_host = False
        if not local_host or ollama_url.scheme != "http" or ollama_url.username or ollama_url.password or ollama_url.path.strip("/") or ollama_url.query or ollama_url.fragment:
            raise ValueError("ollama_base_url must be a local HTTP origin")
        if not isinstance(self.ollama_model, str) or not self.ollama_model.strip():
            raise ValueError("ollama_model must be nonempty text")
        if type(self.ollama_timeout_ms) is not int or not 100 <= self.ollama_timeout_ms <= 120000:
            raise ValueError("ollama_timeout_ms must be between 100 and 120000")
        if not isinstance(self.ollama_keep_alive, str) or not self.ollama_keep_alive.strip():
            raise ValueError("ollama_keep_alive must be nonempty text")
        if (not isinstance(self.smart_review_url, str)
                or not self.smart_review_url.strip().lower().startswith(("http://", "https://"))):
            raise ValueError("smart_review_url must start with http:// or https://")
        if not isinstance(self.smart_review_key, str) or not isinstance(self.smart_review_rules, str):
            raise ValueError("smart review key and rules must be text")
        if not isinstance(self.smart_review_model, str):
            raise ValueError("smart_review_model must be text")
        for name in ("smart_review_key_env", "smart_compare_models"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} must be text")
        if (type(self.smart_review_timeout_s) is not int
                or not 5 <= self.smart_review_timeout_s <= 300):
            raise ValueError("smart_review_timeout_s must be an integer between 5 and 300")
        if (type(self.smart_latency_budget_ms) is not int
                or not 500 <= self.smart_latency_budget_ms <= 120000):
            raise ValueError("smart_latency_budget_ms must be between 500 and 120000")
        if type(self.shadow_mode) is not bool:
            raise ValueError("shadow_mode must be a boolean")
        if type(self.bankroll_enabled) is not bool:
            raise ValueError("bankroll_enabled must be a boolean")
        if not isinstance(self.bankroll_start, str):
            raise ValueError("bankroll_start must be text")
        try:
            positive_number(Decimal(self.bankroll_start), "bankroll_start")
        except (InvalidOperation, TypeError, ValueError) as exc:
            raise ValueError("bankroll_start must be a positive number") from exc
        if (not isinstance(self.bankroll_stop_loss_pct, (int, float))
                or isinstance(self.bankroll_stop_loss_pct, bool)
                or not 0 < self.bankroll_stop_loss_pct <= 100):
            raise ValueError("bankroll_stop_loss_pct must be between 0 and 100")
        if type(self.double_bet_enabled) is not bool:
            raise ValueError("double_bet_enabled must be a boolean")
        if type(self.double_bet_max) is not int or not 1 <= self.double_bet_max <= 10:
            raise ValueError("double_bet_max must be an integer between 1 and 10")
        for name in ("payout_red", "payout_green", "payout_yellow"):
            value = getattr(self, name)
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not 1.0 <= value <= 100):
                raise ValueError(f"{name} must be between 1 and 100")
        if (type(self.smart_review_every_rounds) is not int
                or not 1 <= self.smart_review_every_rounds <= 100):
            raise ValueError("smart_review_every_rounds must be an integer between 1 and 100")
        if type(self.smart_play_enabled) is not bool or type(self.smart_play_two_colors) is not bool:
            raise ValueError("smart_play_enabled and smart_play_two_colors must be booleans")
        if self.bot_mode not in ("classic", "turnbal"):
            raise ValueError("bot_mode must be 'classic' or 'turnbal'")
        for name in ("turnbal_red_enabled", "turnbal_green_enabled", "turnbal_yellow_enabled"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        for name in ("turnbal_red_min_conf", "turnbal_green_min_conf", "turnbal_yellow_min_conf"):
            value = getattr(self, name)
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not 0.2 <= value <= 1):
                raise ValueError(f"{name} must be a number between 0.2 and 1")
        for name in ("turnbal_red_attempts", "turnbal_green_attempts", "turnbal_yellow_attempts"):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= 20:
                raise ValueError(f"{name} must be an integer between 1 and 20")
        for name in ("turnbal_cashout_min", "turnbal_cashout_max", "turnbal_emergency_exit"):
            value = getattr(self, name)
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not 1.4 <= value <= 1.9):
                raise ValueError(f"{name} must be between 1.40 and 1.90")
        if self.turnbal_cashout_min > self.turnbal_cashout_max:
            raise ValueError("turnbal_cashout_min must not exceed turnbal_cashout_max")
        if (not isinstance(self.turnbal_min_cycle_win, (int, float))
                or isinstance(self.turnbal_min_cycle_win, bool)
                or not 0 <= self.turnbal_min_cycle_win <= 1):
            raise ValueError("turnbal_min_cycle_win must be between 0 and 1")
        if type(self.turnbal_max_streak) is not int or not 1 <= self.turnbal_max_streak <= 50:
            raise ValueError("turnbal_max_streak must be an integer between 1 and 50")
        if type(self.turnbal_min_gap) is not int or not 0 <= self.turnbal_min_gap <= 50:
            raise ValueError("turnbal_min_gap must be an integer between 0 and 50")
        if (type(self.smart_play_learning_minutes) is not int
                or not 0 <= self.smart_play_learning_minutes <= 600):
            raise ValueError("smart_play_learning_minutes must be an integer between 0 and 600")
        if (not isinstance(self.smart_play_min_confidence, (int, float))
                or isinstance(self.smart_play_min_confidence, bool)
                or not 0.2 <= self.smart_play_min_confidence <= 1):
            raise ValueError("smart_play_min_confidence must be a number between 0.2 and 1")
        if type(self.review_window_s) is not int or self.review_window_s < 60:
            raise ValueError("review_window_s must be an integer of at least 60")
        if type(self.row_signal_gap) is not int or self.row_signal_gap < 1:
            raise ValueError("row_signal_gap must be a positive integer")
        if type(self.row_signal_clicks) is not int or self.row_signal_clicks < 1:
            raise ValueError("row_signal_clicks must be a positive integer")
        for name in ("row_signal_rows", "row_signal_recent_columns",
                     "streak_signal_length", "streak_signal_clicks"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.value_bets, dict):
            raise ValueError("value_bets must map a multiplier to a colour")
        for text, color in self.value_bets.items():
            try:
                amount = Decimal(text)
            except (InvalidOperation, TypeError, ValueError) as exc:
                raise ValueError(f"value_bets key {text!r} must be a number") from exc
            positive_number(amount, f"value_bets[{text}]")
            if color not in ("RED", "GREEN", "YELLOW"):
                raise ValueError(f"value_bets[{text}] must be RED, GREEN or YELLOW")

def load_settings(path: Path = SETTINGS_PATH) -> DesktopSettings:
    if not path.exists():
        return DesktopSettings()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return DesktopSettings()
    # A settings file written by another build may still carry keys this version
    # no longer has (for example "pair_cycle_enabled"). Drop them instead of
    # letting the whole app fail to start.
    known = {item.name for item in fields(DesktopSettings)}
    data = {key: value for key, value in data.items() if key in known}
    for key in ("result_region", "history_region", "click_region", "chart_region", "bet_region"):
        if data.get(key):
            data[key] = Region(**data[key])
    for key in ("click_point", "bet_point_red", "bet_point_green", "bet_point_yellow",
                "double_bet_point", "halve_bet_point"):
        if data.get(key) is not None:
            data[key] = tuple(data[key])
    # Older settings files still carry the immediate-click default of 0.
    delay = data.get("autoclick_delay_ms")
    if type(delay) is not int or delay < 0:
        data["autoclick_delay_ms"] = MAX_CLICK_DELAY_MS
    elif delay > MAX_CLICK_DELAY_MS:
        data["autoclick_delay_ms"] = MAX_CLICK_DELAY_MS
    if "autoclick_skip_rounds" not in data:
        # Files written for the old "ten clicks right after the yellow" rule must not
        # keep clicking ten rounds; the run is now wait 7, click 3.
        data["autoclick_skip_rounds"] = 7
        data["autoclick_max_clicks"] = 3
    if isinstance(data.get("smart_review_key"), str):
        data["smart_review_key"] = unprotect(data["smart_review_key"])
    result = DesktopSettings(**data)
    result.validate()
    return result

def save_settings(settings: DesktopSettings, path: Path = SETTINGS_PATH) -> None:
    settings.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(settings)
    # کلید در settings.json همیشه رمزشده است؛ متن‌ساده هرگز روی دیسک نمی‌ماند.
    data["smart_review_key"] = protect(settings.smart_review_key)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)
