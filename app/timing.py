"""The editable lyric timing table.

Rows are the unit of editing: one row is one word or syllable with its own
start and end time. Validation lives here, not in the UI, so the same rules
apply to the app and to the tests.

Errors are written for a beginner to act on: they name the row number, say what
is wrong, and say what to do.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .lyrics import Syllable, tokenize

# Spacing used when lyrics are first spread across the timeline. These are only
# starting values; every one of them is editable afterwards.
DEFAULT_UNIT_SECONDS = 0.45
DEFAULT_START_SECONDS = 0.0
DEFAULT_LINE_GAP_SECONDS = 0.0


@dataclass
class LyricRow:
    """One editable lyric entry."""

    line: int
    text: str
    start: float
    end: float
    word_end: bool = True

    @property
    def duration(self) -> float:
        return self.end - self.start


def rows_from_text(
    text: str,
    *,
    mode: str = "words",
    start: float = DEFAULT_START_SECONDS,
    unit: float = DEFAULT_UNIT_SECONDS,
    line_gap: float = DEFAULT_LINE_GAP_SECONDS,
) -> list[LyricRow]:
    """Split pasted lyrics into evenly spaced, editable rows.

    `line_gap` defaults to 0 so row N sits at start + N*unit. A gap would push
    every row after a line break off its expected slot.
    """
    cursor = start
    rows: list[LyricRow] = []

    for line_number, units in enumerate(tokenize(text, mode=mode), start=1):
        for text_piece, is_word_end in units:
            rows.append(
                LyricRow(
                    line=line_number,
                    text=text_piece,
                    start=round(cursor, 3),
                    end=round(cursor + unit, 3),
                    word_end=is_word_end,
                )
            )
            cursor += unit
        cursor += line_gap

    return rows


def validate_rows(rows: list[LyricRow]) -> list[str]:
    """Return a list of problems. An empty list means the rows are usable."""
    problems: list[str] = []

    if not rows:
        return [
            "No lyrics yet. Paste lyrics in the text box and press "
            "'Split lyrics into rows'."
        ]

    seen_text = False
    previous: LyricRow | None = None

    for position, row in enumerate(rows, start=1):
        label = f"Row {position}"

        if not isinstance(row.text, str) or not row.text.strip():
            problems.append(f"{label}: the text is empty. Type a word, or delete the row.")
        else:
            seen_text = True

        for field_name, value in (("start", row.start), ("end", row.end)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                problems.append(
                    f"{label}: '{field_name}' must be a number, but it is {value!r}."
                )
            elif not math.isfinite(float(value)):
                problems.append(f"{label}: '{field_name}' must be a real number, not {value!r}.")
            elif float(value) < 0:
                problems.append(
                    f"{label}: '{field_name}' is {value}. Times start at 0, so it cannot be negative."
                )

        if isinstance(row.start, (int, float)) and isinstance(row.end, (int, float)):
            if not math.isfinite(float(row.start)) or not math.isfinite(float(row.end)):
                pass  # already reported above
            elif float(row.end) < float(row.start):
                problems.append(
                    f"{label}: 'end' ({row.end}) is before 'start' ({row.start})."
                )

        if previous is not None and _usable(row.start) and _usable(previous.start):
            if float(row.start) < float(previous.start) - 1e-9:
                problems.append(
                    f"{label}: starts at {row.start}s, which is before the previous row "
                    f"('{previous.text}' at {previous.start}s). Rows must stay in time order."
                )

        previous = row

    if not seen_text:
        problems.append("Every row is empty. Add lyric text before exporting.")

    return problems


def _usable(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def rows_to_lyric_lines(rows: list[LyricRow]) -> list[list[Syllable]]:
    """Group rows by their line number, preserving order within a line."""
    grouped: dict[int, list[Syllable]] = {}
    for row in rows:
        grouped.setdefault(row.line, []).append(
            Syllable(
                text=row.text,
                start=float(row.start),
                end=float(row.end),
                is_word_end=bool(row.word_end),
                is_line_end=False,
            )
        )

    return [grouped[line_number] for line_number in sorted(grouped)]


def group_by_line(rows: list[LyricRow]) -> list[list[LyricRow]]:
    """Rows grouped into display lines, for the lyric sequence preview."""
    grouped: dict[int, list[LyricRow]] = {}
    for row in rows:
        grouped.setdefault(row.line, []).append(row)
    return [grouped[line_number] for line_number in sorted(grouped)]


def respace(rows: list[LyricRow], *, start: float = DEFAULT_START_SECONDS, unit: float = DEFAULT_UNIT_SECONDS) -> list[LyricRow]:
    """Re-apply even spacing, keeping text and line numbers. An 'undo' helper."""
    return [
        LyricRow(
            line=row.line,
            text=row.text,
            start=round(start + index * unit, 3),
            end=round(start + (index + 1) * unit, 3),
            word_end=row.word_end,
        )
        for index, row in enumerate(rows)
    ]


@dataclass
class LyricTable:
    """Convenience wrapper so the UI can hold one object in session state."""

    rows: list[LyricRow] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not validate_rows(self.rows)

    @property
    def problems(self) -> list[str]:
        return validate_rows(self.rows)

    @property
    def last_end(self) -> float:
        usable = [float(r.end) for r in self.rows if _usable(r.end)]
        return max(usable) if usable else 0.0