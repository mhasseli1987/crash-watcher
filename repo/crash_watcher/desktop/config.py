"""Project-local validated desktop settings."""
from dataclasses import asdict, dataclass, field, fields
from decimal import Decimal, InvalidOperation
import json
import sys
from pathlib import Path
from crash_watcher.core.config import StrategyConfig, positive_number
from crash_watcher.vision.capture import CaptureConfig

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
    # «محل کلیک کلاسیک»: همان دکمهٔ بزرگ Bet / Cash Out بازی.
    classic_click_point: tuple[int, int] | None = None
    # «کلید Classic/Trenball»: با کلیک روی همین نقطه صفحهٔ بازی عوض می‌شود.
    trenball_toggle_point: tuple[int, int] | None = None
    # ناحیهٔ سه دکمهٔ شرط ترنبال: Bet Red / Bet Green / Bet Moon.
    trenball_bet_region: Region | None = None
    # شیفت خودکار به ترنبال وقتی یک رنگ پشت سر هم تکرار می‌شود.
    trenball_enabled: bool = False
    trenball_streak_length: int = 3
    # در ترنبال فقط تا این تعداد مرحله مبلغ بعد از باخت دو برابر می‌شود.
    trenball_stake_double_steps: int = 3
    # مارتینگل: بعد از هر باخت، دکمهٔ x2 مبلغ بازی زده می‌شود و مبلغ پایه تا این
    # تعداد مرحله دو برابر می‌ماند؛ بعد از برد (یا پایان مرحلهٔ آخر) دکمهٔ /2 مبلغ
    # را به پایه برمی‌گرداند.
    stake_martingale_enabled: bool = True
    stake_double_steps: int = 3
    stake_double_point: tuple[int, int] | None = None
    stake_half_point: tuple[int, int] | None = None
    bet_point_red: tuple[int, int] | None = None
    bet_point_green: tuple[int, int] | None = None
    bet_point_yellow: tuple[int, int] | None = None
    autoclick_enabled: bool = False
    autoclick_mode: str = "detect"
    autoclick_threshold: str = "10.00"
    autoclick_cooldown_ms: int = 1500
    autoclick_delay_ms: int = MAX_CLICK_DELAY_MS
    autoclick_max_clicks: int = 3
    autoclick_skip_rounds: int = 7
    autoclick_dry_run: bool = True
    autoclick_restore_cursor: bool = True
    smart_click_enabled: bool = False
    prediction_min_confidence: float = 0.60
    # AI Analysis: پرسش از سرور محلی FreeLLMAPI یا API رسمی OpenAI.
    smart_review_provider: str = "freellmapi"  # freellmapi | openai
    smart_review_enabled: bool = False
    smart_review_url: str = "http://127.0.0.1:31415/v1"
    smart_review_key: str = ""
    smart_review_model: str = "auto"
    smart_review_timeout_s: int = 120
    smart_review_every_rounds: int = 1
    smart_review_rules: str = ""
    # «ورود هوشمند به بازی»: اول یادگیری، بعد شرط‌بندی بر پایهٔ تصمیم آمادهٔ مدل.
    smart_play_enabled: bool = False
    smart_play_learning_minutes: int = 30
    smart_play_min_confidence: float = 0.40
    smart_play_two_colors: bool = True
    # «شیفت هوشمند به ترنبال»: وقتی مدل رنگ دست بعد را با اطمینان ≥ این درصد
    # پیش‌بینی کند، صفحه سریع به ترنبال می‌رود و روی همان رنگ شرط بسته می‌شود؛
    # زیر این درصد دوباره به کلاسیک برمی‌گردد و تنظیمات عادی اجرا می‌شوند.
    ai_trenball_enabled: bool = True
    ai_trenball_min_confidence: float = 0.60
    # «کلیک کلاسیک»: ورود با یک کلیک پیش از پایان ۶ ثانیه و خروج با کلیک دوم زیر ۱٫۹۹.
    smart_cashout_enabled: bool = False
    # قوانین قدیمی «شکل رنگ» در تحلیل لاگ بازدهی نداشتند (۱۱٪ تا ۴۵٪) و پیش‌فرض
    # خاموش‌اند؛ فقط موتور ۳x تصمیم ورود می‌گیرد.
    signal_rule_enabled: bool = False
    signal_repeat: int = 2
    signal_burst_rounds: int = 9
    pair_flip_enabled: bool = False
    row_signal_enabled: bool = False
    row_signal_gap: int = 4
    row_signal_rows: int = 4
    row_signal_recent_columns: int = 4
    row_signal_clicks: int = 1
    streak_signal_enabled: bool = False
    streak_signal_length: int = 6
    streak_signal_clicks: int = 10
    value_bet_enabled: bool = False
    # موتور ۳x: تنها مسیر ورود. یک ساعت مشاهده، ۱۰۰ دور، احتمال ≥۴۰٪ از ۲۰ نمونهٔ
    # مشابه، شرط ثابت ≤۰٫۲۵٪ موجودی، توقف در ±۲٪ جلسه.
    three_x_enabled: bool = True
    three_x_observation_minutes: int = 60
    three_x_min_rounds: int = 100
    three_x_probability: float = 0.40
    three_x_min_samples: int = 20
    three_x_min_ocr: float = 0.95
    three_x_exit: str = "3"
    three_x_stake_pct: float = 0.25
    three_x_bankroll: str = "1000"
    three_x_profit_limit_pct: float = 2.0
    three_x_loss_limit_pct: float = 2.0
    three_x_stop_losses: int = 3
    three_x_stop_rounds: int = 30
    three_x_min_gap_rounds: int = 5
    value_bets: dict[str, str] = field(default_factory=lambda: {
        "1.00": "GREEN", "1.11": "GREEN", "1.99": "GREEN", "2.08": "YELLOW"})
    review_window_s: int = 1800
    session_minutes: int = 0
    shutdown_after_session: bool = False
    fixed_bet_amount: str = "10"
    max_attempts: int = 8
    yellow_threshold: str = "10.00"
    red_threshold: str = "2.00"
    interval_ms: int = 100
    ocr_confidence: float = 0.55
    theme: str = "dark"
    tesseract_path: str = ""

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
                     "bet_region", "trenball_bet_region"):
            region = getattr(self, name)
            if region is None:
                continue
            if not isinstance(region, Region):
                raise ValueError(f"{name} must be a Region")
            region.capture(self.interval_ms)
        for name in ("click_point", "classic_click_point", "trenball_toggle_point",
                     "stake_double_point", "stake_half_point", "bet_point_red",
                     "bet_point_green", "bet_point_yellow"):
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
        if self.smart_review_provider not in ("freellmapi", "openai"):
            raise ValueError("smart_review_provider must be 'freellmapi' or 'openai'")
        if type(self.smart_review_enabled) is not bool:
            raise ValueError("smart_review_enabled must be a boolean")
        if (not isinstance(self.smart_review_url, str)
                or not self.smart_review_url.strip().lower().startswith(("http://", "https://"))):
            raise ValueError("smart_review_url must start with http:// or https://")
        if not isinstance(self.smart_review_key, str) or not isinstance(self.smart_review_rules, str):
            raise ValueError("smart review key and rules must be text")
        if not isinstance(self.smart_review_model, str):
            raise ValueError("smart_review_model must be text")
        if (type(self.smart_review_timeout_s) is not int
                or not 60 <= self.smart_review_timeout_s <= 300):
            raise ValueError("smart_review_timeout_s must be an integer between 60 and 300")
        if (type(self.smart_review_every_rounds) is not int
                or not 1 <= self.smart_review_every_rounds <= 100):
            raise ValueError("smart_review_every_rounds must be an integer between 1 and 100")
        if (type(self.smart_play_enabled) is not bool or type(self.smart_play_two_colors) is not bool
                or type(self.smart_cashout_enabled) is not bool
                or type(self.stake_martingale_enabled) is not bool):
            raise ValueError("smart_play_enabled, smart_play_two_colors, "
                             "smart_cashout_enabled and stake_martingale_enabled must be booleans")
        if type(self.stake_double_steps) is not int or not 0 <= self.stake_double_steps <= 5:
            raise ValueError("stake_double_steps must be an integer between 0 and 5")
        if type(self.trenball_enabled) is not bool:
            raise ValueError("trenball_enabled must be a boolean")
        if (type(self.trenball_streak_length) is not int
                or not 2 <= self.trenball_streak_length <= 10):
            raise ValueError("trenball_streak_length must be an integer between 2 and 10")
        if (type(self.trenball_stake_double_steps) is not int
                or not 0 <= self.trenball_stake_double_steps <= 5):
            raise ValueError("trenball_stake_double_steps must be an integer between 0 and 5")
        if (type(self.smart_play_learning_minutes) is not int
                or not 0 <= self.smart_play_learning_minutes <= 600):
            raise ValueError("smart_play_learning_minutes must be an integer between 0 and 600")
        if (not isinstance(self.smart_play_min_confidence, (int, float))
                or isinstance(self.smart_play_min_confidence, bool)
                or not 0.2 <= self.smart_play_min_confidence <= 1):
            raise ValueError("smart_play_min_confidence must be a number between 0.2 and 1")
        if type(self.ai_trenball_enabled) is not bool:
            raise ValueError("ai_trenball_enabled must be a boolean")
        if (not isinstance(self.ai_trenball_min_confidence, (int, float))
                or isinstance(self.ai_trenball_min_confidence, bool)
                or not 0.2 <= self.ai_trenball_min_confidence <= 1):
            raise ValueError("ai_trenball_min_confidence must be a number between 0.2 and 1")
        if type(self.review_window_s) is not int or self.review_window_s < 60:
            raise ValueError("review_window_s must be an integer of at least 60")
        if (type(self.session_minutes) is not int or self.session_minutes < 0
                or self.session_minutes > 1440):
            raise ValueError("session_minutes must be an integer between 0 and 1440")
        if type(self.shutdown_after_session) is not bool:
            raise ValueError("shutdown_after_session must be a boolean")
        if type(self.signal_repeat) is not int or self.signal_repeat < 1:
            raise ValueError("signal_repeat must be a positive integer")
        if type(self.signal_burst_rounds) is not int or self.signal_burst_rounds < 1:
            raise ValueError("signal_burst_rounds must be a positive integer")
        if type(self.row_signal_gap) is not int or self.row_signal_gap < 1:
            raise ValueError("row_signal_gap must be a positive integer")
        if type(self.row_signal_clicks) is not int or self.row_signal_clicks < 1:
            raise ValueError("row_signal_clicks must be a positive integer")
        for name in ("row_signal_rows", "row_signal_recent_columns",
                     "streak_signal_length", "streak_signal_clicks"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("three_x_observation_minutes", "three_x_min_rounds", "three_x_min_samples",
                     "three_x_stop_losses", "three_x_stop_rounds", "three_x_min_gap_rounds"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.three_x_min_samples < 1 or self.three_x_stop_losses < 1 or self.three_x_min_gap_rounds < 1:
            raise ValueError("three_x_min_samples, three_x_stop_losses and "
                             "three_x_min_gap_rounds must be at least 1")
        for name in ("three_x_probability", "three_x_min_ocr"):
            value = getattr(self, name)
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not 0.2 <= value <= 1):
                raise ValueError(f"{name} must be a number between 0.2 and 1")
        for name in ("three_x_profit_limit_pct", "three_x_loss_limit_pct"):
            value = getattr(self, name)
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not 0.1 <= value <= 100):
                raise ValueError(f"{name} must be a number between 0.1 and 100")
        if (not isinstance(self.three_x_stake_pct, (int, float))
                or isinstance(self.three_x_stake_pct, bool)
                or not 0.01 <= self.three_x_stake_pct <= 10):
            raise ValueError("three_x_stake_pct must be between 0.01 and 10")
        try:
            exit_factor = Decimal(self.three_x_exit)
            bankroll = Decimal(self.three_x_bankroll)
        except (InvalidOperation, TypeError, ValueError) as exc:
            raise ValueError("three_x_exit and three_x_bankroll must be numbers") from exc
        positive_number(exit_factor, "three_x_exit")
        positive_number(bankroll, "three_x_bankroll")
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
    for key in ("result_region", "history_region", "click_region", "chart_region",
                "bet_region", "trenball_bet_region"):
        if data.get(key):
            data[key] = Region(**data[key])
    for key in ("click_point", "classic_click_point", "trenball_toggle_point",
                "stake_double_point", "stake_half_point", "bet_point_red", "bet_point_green",
                "bet_point_yellow"):
        if data.get(key) is not None:
            data[key] = tuple(data[key])
    # Older settings files still carry the immediate-click default of 0.
    delay = data.get("autoclick_delay_ms")
    if type(delay) is not int or delay < 0:
        data["autoclick_delay_ms"] = MAX_CLICK_DELAY_MS
    elif delay > MAX_CLICK_DELAY_MS:
        data["autoclick_delay_ms"] = MAX_CLICK_DELAY_MS
    if data.get("session_minutes") == 360 and data.get("shutdown_after_session") is True:
        # The timer and the shutdown were the old defaults; the user stops the
        # session by hand now, so an old file must not shut the machine down.
        data["session_minutes"] = 0
        data["shutdown_after_session"] = False
    if "three_x_enabled" not in data:
        # A file written before the 3x engine still enables the old shape rules;
        # those rules lost money, so a loaded file must not bring them back.
        for key in ("signal_rule_enabled", "pair_flip_enabled", "row_signal_enabled",
                    "streak_signal_enabled", "value_bet_enabled"):
            data[key] = False
    if "smart_review_timeout_s" in data:
        timeout = data["smart_review_timeout_s"]
        if type(timeout) is not int or timeout < 60:
            # نسخه‌های قبلی ۳۰/۴۵ ثانیه ذخیره می‌کردند و مدل محلی در این مهلت پاسخ
            # نمی‌داد؛ فقط همین مقدار اصلاح می‌شود، بقیهٔ فایل دست‌نخورده می‌ماند.
            data["smart_review_timeout_s"] = 120
    if "autoclick_skip_rounds" not in data:
        # Files written for the old "ten clicks right after the yellow" rule must not
        # keep clicking ten rounds; the run is now wait 7, click 3.
        data["autoclick_skip_rounds"] = 7
        data["autoclick_max_clicks"] = 3
    result = DesktopSettings(**data)
    result.validate()
    return result

def save_settings(settings: DesktopSettings, path: Path = SETTINGS_PATH) -> None:
    settings.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")
    temporary.replace(path)
