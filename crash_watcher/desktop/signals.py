"""Shape signals read from the game's colour-dot trend chart.

The chart is column-major (see color_grid): the newest dot is the top-left cell
and history fills down a column before moving one column to the right. A signal
is therefore a shape of adjacent cells, exactly as the user sees it.
"""
from __future__ import annotations

from .color_grid import GridReading


def next_dot_row(reading: GridReading | None) -> int | None:
    """Row the next round's dot lands in: the newest column fills top to bottom."""
    if reading is None or getattr(reading, "columns", 0) < 1:
        return None
    rows = getattr(reading, "rows", 0)
    row = 0
    while row < rows and reading.color_at(0, row) is not None:
        row += 1
    return row if row < rows else None


def row_yellow_due(reading: GridReading | None, gap: int = 4, rows: int = 4,
                   recent_columns: int = 4) -> bool:
    """True when the row the next dot fills was yellow exactly `gap` columns back.

    The chart is column-major, so the next round lands in row `next_dot_row` of
    the newest column. When that very row is yellow in the column `gap` columns
    older, the repetition the user watches for is due on that round: it has to
    be bet now, not after the repeat has already been drawn. Only the first
    `rows` rows count — below row four the user rates the odds too low.
    """
    if reading is None or gap < 1 or rows < 1 or recent_columns < 1:
        return False
    if reading.columns <= max(gap, recent_columns):
        return False
    row = next_dot_row(reading)
    if row is None or row >= rows:
        return False
    return reading.color_at(gap, row) == "YELLOW"


