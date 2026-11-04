"""Strict OCR multiplier parsing and local image preprocessing."""
from collections.abc import Callable
from decimal import Decimal
import re
from time import perf_counter
import cv2
import numpy as np
from crash_watcher.core.enums import MultiplierCategory
from .models import Recognition

_VALUE = re.compile(r"^(\d{1,4})\.(\d{2})[xX]?$", re.ASCII)

def parse_multiplier(text: str) -> Decimal | None:
    """Accept only an unambiguous two-decimal multiplier with optional x."""
    if not isinstance(text, str):
        return None
    normalized = text.strip().replace("×", "x")
    match = _VALUE.fullmatch(normalized)
    if not match:
        return None
    value = Decimal(f"{match[1]}.{match[2]}")
    return value if value > 0 else None

def category_for(value: Decimal, red: Decimal = Decimal("2"), yellow: Decimal = Decimal("10")) -> MultiplierCategory:
    if value < red:
        return MultiplierCategory.RED
    if value < yellow:
        return MultiplierCategory.GREEN
    return MultiplierCategory.YELLOW

def preprocess(image: np.ndarray) -> list[np.ndarray]:
    if image is None or image.size == 0 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected nonempty BGR image")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return [enhanced, binary, cv2.bitwise_not(binary)]

def sample_color(image: np.ndarray) -> MultiplierCategory | None:
    """Diagnostic-only saturated text color estimate; never determines value."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 100)
    if int(mask.sum()) < max(8, image.shape[0] * image.shape[1] // 100):
        return None
    hues = hsv[:, :, 0][mask]
    red = np.count_nonzero((hues < 12) | (hues > 170))
    green = np.count_nonzero((hues >= 35) & (hues < 90))
    yellow = np.count_nonzero((hues >= 12) & (hues < 35))
    votes = [red, green, yellow]
    if max(votes) < len(hues) * 0.45:
        return None
    return [MultiplierCategory.RED, MultiplierCategory.GREEN, MultiplierCategory.YELLOW][int(np.argmax(votes))]

def tesseract_reader(image: np.ndarray) -> tuple[str, float]:
    import pytesseract
    data = pytesseract.image_to_data(image, config="--psm 7 -c tessedit_char_whitelist=0123456789.xX", output_type=pytesseract.Output.DICT)
    parts = [(text.strip(), float(conf)) for text, conf in zip(data["text"], data["conf"])
             if text.strip() and float(conf) >= 0]
    if not parts:
        return "", 0.0
    return "".join(text for text, _ in parts), min(conf for _, conf in parts) / 100

class RapidOCRReader:
    """Bundled local model; rejects multiple text lines as ambiguous."""
    def __init__(self) -> None:
        from rapidocr import RapidOCR
        self._engine = RapidOCR()

    def __call__(self, image: np.ndarray) -> tuple[str, float]:
        result = self._engine(image, use_det=False, use_cls=False)
        if result.txts is None or result.scores is None or len(result.txts) != 1:
            return "", 0.0
        return str(result.txts[0]), float(result.scores[0])

class MultiplierRecognizer:
    def __init__(self, reader: Callable[[np.ndarray], tuple[str, float]] = tesseract_reader,
                 min_confidence: float = 0.80, red: Decimal = Decimal("2"), yellow: Decimal = Decimal("10")) -> None:
        if not 0 < min_confidence <= 1 or not 0 < red < yellow:
            raise ValueError("invalid recognition configuration")
        self.reader, self.min_confidence, self.red, self.yellow = reader, min_confidence, red, yellow

    def recognize(self, image: np.ndarray) -> Recognition:
        start = perf_counter()
        variants = preprocess(image)
        ocr_start = perf_counter()
        readings = [self.reader(variant) for variant in variants]
        ocr_ms = (perf_counter() - ocr_start) * 1000
        valid = [(parse_multiplier(text), text, confidence) for text, confidence in readings]
        accepted = [(value, text, confidence) for value, text, confidence in valid
                    if value is not None and np.isfinite(confidence) and confidence >= self.min_confidence]
        color = sample_color(image)
        if not accepted or len({value for value, _, _ in accepted}) != 1:
            best = max(readings, key=lambda x: x[1], default=("", 0.0))
            return Recognition(None, None, max(0.0, min(1.0, best[1])), best[0], True, color,
                               (perf_counter() - start) * 1000, ocr_ms)
        value, text, confidence = max(accepted, key=lambda x: x[2])
        return Recognition(value, category_for(value, self.red, self.yellow), confidence, text,
                           False, color, (perf_counter() - start) * 1000, ocr_ms)
