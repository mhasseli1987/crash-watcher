"""MSS capture with monitor/region validation and safe lifecycle."""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from .models import DetectionEvent, DetectionEventType, Frame, now_utc

@dataclass(frozen=True)
class CaptureConfig:
    monitor: int
    x: int
    y: int
    width: int
    height: int
    interval_ms: int = 100

    def __post_init__(self) -> None:
        if any(type(v) is not int for v in (self.monitor, self.x, self.y, self.width, self.height, self.interval_ms)):
            raise ValueError("capture fields must be integers")
        if self.monitor < 1 or self.width <= 0 or self.height <= 0 or self.interval_ms < 20:
            raise ValueError("invalid monitor, rectangle or interval")

class ScreenUnavailable(RuntimeError):
    pass

class ScreenCapture:
    def __init__(self, config: CaptureConfig, backend=None) -> None:
        self.config = config
        self._backend = backend
        self._owns_backend = backend is None
        self._last_geometry: tuple[int, int, int, int] | None = None

    def __enter__(self):
        if self._backend is None:
            import mss
            self._backend = mss.MSS()
        return self

    def __exit__(self, *_):
        if self._owns_backend and self._backend is not None:
            self._backend.close()
            self._backend = None

    def monitors(self) -> tuple[dict, ...]:
        if self._backend is None:
            raise ScreenUnavailable("capture backend is closed")
        return tuple(dict(m) for m in self._backend.monitors[1:])

    def capture(self) -> tuple[Frame, tuple[DetectionEvent, ...]]:
        if self._backend is None:
            raise ScreenUnavailable("capture backend is closed")
        monitors = self.monitors()
        if self.config.monitor > len(monitors):
            raise ScreenUnavailable("selected monitor is unavailable")
        m = monitors[self.config.monitor - 1]
        bounds = (m["left"], m["top"], m["width"], m["height"])
        c = self.config
        if (c.x < m["left"] or c.y < m["top"] or c.x + c.width > m["left"] + m["width"]
                or c.y + c.height > m["top"] + m["height"]):
            raise ScreenUnavailable("region is outside selected monitor; recalibrate")
        events = []
        if self._last_geometry is not None and bounds != self._last_geometry:
            events.append(DetectionEvent(DetectionEventType.SCREEN_REGION_CHANGED, now_utc(), str(bounds)))
        self._last_geometry = bounds
        start = perf_counter()
        try:
            raw = self._backend.grab({"left": c.x, "top": c.y, "width": c.width, "height": c.height})
            image = np.asarray(raw)[:, :, :3].copy()
        except Exception as exc:
            raise ScreenUnavailable(f"capture failed: {exc}") from exc
        return Frame(image, now_utc(), (perf_counter() - start) * 1000), tuple(events)
