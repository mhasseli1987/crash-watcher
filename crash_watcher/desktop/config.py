"""Project-local validated desktop settings."""
from dataclasses import asdict, dataclass
from decimal import Decimal
import json
import sys
from pathlib import Path
from crash_watcher.core.config import StrategyConfig
from crash_watcher.vision.capture import CaptureConfig

PROJECT_ROOT = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[2]
SETTINGS_PATH = PROJECT_ROOT / "settings.json"

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
    click_point: tuple[int, int] | None = None
    autoclick_enabled: bool = False
    autoclick_mode: str = "detect"
    autoclick_threshold: str = "10.00"
    autoclick_cooldown_ms: int = 1500
    autoclick_delay_ms: int = 0
    autoclick_dry_run: bool = True
    autoclick_restore_cursor: bool = True
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
def load_settings(path: Path = SETTINGS_PATH) -> DesktopSettings:
    if not path.exists():
        return DesktopSettings()
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("result_region", "history_region"):
        if data.get(key):
            data[key] = Region(**data[key])
    result = DesktopSettings(**data)
    result.validate()
    return result

def save_settings(settings: DesktopSettings, path: Path = SETTINGS_PATH) -> None:
    settings.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")
    temporary.replace(path)
