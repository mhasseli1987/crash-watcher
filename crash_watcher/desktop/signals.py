"""Shape signals read from the game's colour-dot trend chart.

The chart is column-major (see color_grid): the newest dot is the top-left cell
and history fills down a column before moving one column to the right. A signal
is therefore a shape of adjacent cells, exactly as the user sees it.
"""
from __future__ import annotations

from .color_grid import GridReading


def yellow_green_green_yellow(reading: GridReading | None) -> bool:
    """True when one column reads yellow, green, green, yellow from the top.

    The user reads that sandwich as "more yellows are coming": the clicker then
    bets yellow on each of the next nine rounds.
    """
    if reading is None:
        return False
    for column in range(reading.columns):
        shape = tuple(reading.color_at(column, row) for row in range(4))
        if shape == ("YELLOW", "GREEN", "GREEN", "YELLOW"):
            return True
    return False


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


def row_yellow_repeat(reading: GridReading | None, gap: int = 4,
                      last_row: int = 3) -> int | None:
    """Rule 2: the row the next dot fills was yellow `gap` columns back.

    The user watches yellows repeating along one row of the chart — in any of
    the first four rows, never row five or lower — and wants the bet placed on
    the round that is about to land there, before that yellow is drawn.

    Returns the row the repeat is expected in, or None.
    """
    if reading is None or gap < 1:
        return None
    if reading.columns <= gap:
        return None
    row = next_dot_row(reading)
    if row is None or row > last_row:
        return None
    return row if reading.color_at(gap, row) == "YELLOW" else None


def yellow_green_then_red(reading: GridReading | None, gap: int = 5) -> bool:
    """Rule 3: yellow then green in a column, and `gap` columns later a red one.

    The user reads that as "yellow is due on the next round" and clicks yellow.
    """
    if reading is None or gap < 1:
        return False
    if reading.color_at(0, 0) != "RED":
        return False
    return (reading.color_at(gap, 0) == "YELLOW"
            and reading.color_at(gap, 1) == "GREEN")


def two_yellows_in_a_row(reading: GridReading | None) -> bool:
    """True when two consecutive rounds (vertical neighbours) are both yellow.

    The user's rule ignores what came before the pair: two yellows in a row are
    enough, and the next yellow is then expected to double again, so it is worth
    a yellow bet.
    """
    if reading is None:
        return False
    for column in range(reading.columns):
        for row in range(reading.rows - 1):
            if (reading.color_at(column, row) == "YELLOW"
                    and reading.color_at(column, row + 1) == "YELLOW"):
                return True
    return False
