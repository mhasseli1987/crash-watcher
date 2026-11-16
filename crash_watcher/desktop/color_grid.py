"""Read the game's colour-dot trend chart from a screenshot region.

The site explains its own layout: dots are laid out column-major. The newest
round is the top-left dot, the next older one sits directly below it and, once a
column is full, history continues at the top of the column to the right. So the
ordered sequence is (column 0, row 0), (column 0, row 1), ..., (column 1, row 0),
and consecutive rounds are vertical neighbours.

Colours are read by hue, never OCR: green is >= 2x, orange (called red in the app)
is below 2x and yellow is >= 10x.
"""
from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np

# OpenCV hue is 0-179. Measured from the game's palette:
# green #2ecc71 -> ~72, yellow #f1c40f -> ~25, orange #e67e22 -> ~14.
_HSV_RANGES = {
    "RED": [((0, 90, 90), (19, 255, 255)), ((170, 90, 90), (180, 255, 255))],
    "YELLOW": [((20, 80, 110), (35, 255, 255))],
    "GREEN": [((40, 60, 60), (95, 255, 255))],
}


@dataclass(frozen=True)
class Dot:
    color: str
    column: int
    row: int
    x: int
    y: int


@dataclass(frozen=True)
class GridReading:
    dots: tuple[Dot, ...]
    columns: int
    rows: int
    pitch: float

    @property
    def sequence(self) -> tuple[str, ...]:
        """Colours in the site's order: newest first, down each column, left to right."""
        return tuple(dot.color for dot in sorted(self.dots, key=lambda dot: (dot.column, dot.row)))

    def color_at(self, column: int, row: int) -> str | None:
        for dot in self.dots:
            if dot.column == column and dot.row == row:
                return dot.color
        return None


def _centres(image: np.ndarray, color: str, min_area: float) -> list[tuple[int, int]]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in _HSV_RANGES[color]:
        mask |= cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        if not 0.6 <= width / max(height, 1) <= 1.7:
            continue
        if area / float(max(width * height, 1)) < 0.55:
            continue
        found.append((x + width // 2, y + height // 2))
    return found


def _pitch(values: list[int]) -> float:
    unique = sorted(set(values))
    steps = [b - a for a, b in zip(unique, unique[1:]) if b - a > 3]
    return float(np.median(steps)) if steps else 0.0


def read_grid(image: np.ndarray, min_dot_area: float = 40.0) -> GridReading | None:
    """Map every coloured dot to its grid cell; None when nothing usable was found."""
    if image is None or image.size == 0 or image.ndim != 3:
        return None
    centres = []
    for color in _HSV_RANGES:
        centres.extend((color, x, y) for x, y in _centres(image, color, min_dot_area))
    if not centres:
        return None
    pitch = _pitch([x for _, x, _ in centres]) or _pitch([y for _, _, y in centres])
    if pitch <= 0:
        return None
    origin_x = min(x for _, x, _ in centres)
    origin_y = min(y for _, _, y in centres)
    dots: list[Dot] = []
    seen: set[tuple[int, int]] = set()
    for color, x, y in centres:
        column = round((x - origin_x) / pitch)
        row = round((y - origin_y) / pitch)
        if column < 0 or row < 0 or (column, row) in seen:
            return None
        seen.add((column, row))
        dots.append(Dot(color, column, row, x, y))
    return GridReading(tuple(dots), max(d.column for d in dots) + 1,
                       max(d.row for d in dots) + 1, pitch)
