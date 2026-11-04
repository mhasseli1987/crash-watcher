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
    fixed_bet_amount: str = "10"
    max_attempts: int = 10
    yellow_threshold: str = "10.00"
    interval_ms: int = 100
    ocr_confidence: float = 0.80
    tesseract_path: str = ""
    theme: str = "dark"

    def validate(self) -> None:
        StrategyConfig(fixed_bet_amount=Decimal(self.fixed_bet_amount),
                       max_attempts=self.max_attempts, yellow_threshold=Decimal(self.yellow_threshold))
        if not 0 < self.ocr_confidence <= 1:
            raise ValueError("OCR confidence must be between 0 and 1")
        if self.interval_ms < 20:
            raise ValueError("capture interval must be at least 20 ms")
        for region in (self.result_region, self.history_region):
            if region:
                region.capture(self.interval_ms)

    def strategy(self) -> StrategyConfig:
        self.validate()
        return StrategyConfig(fixed_bet_amount=Decimal(self.fixed_bet_amount),
                              max_attempts=self.max_attempts, yellow_threshold=Decimal(self.yellow_threshold))

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
