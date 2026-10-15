"""Lyric text and timing model.

Scope note for Milestone 1
-------------------------
Here one whitespace-separated token is one timed unit. Splitting words into
individual sung syllables is deliberately NOT implemented yet: naive syllable
splitting produces visibly wrong results on real lyrics (silent trailing
'e', consonant clusters, loanwords), and it is a Milestone 2 concern tied to
the timing editor. Getting this deliberately dumb now keeps it auditable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Syllable:
    """One timed lyric unit."""

    text: str
    start: float  # seconds from start of song
    end: float  # seconds from start of song
    is_word_end: bool = True
    is_line_end: bool = False

    @property
    def duration(self) -> float:
        return self.end - self.start


def parse_lyrics(
    text: str,
    *,
    start: float = 0.0,
    unit_seconds: float = 0.45,
    line_gap: float = 0.6,
) -> list[list[Syllable]]:
    """Turn plain lyric lines into uniformly timed units.

    Blank lines are dropped. Real per-word/per-syllable timings arrive with the
    Milestone 2 timing editor; this exists so Milestone 1 can produce a valid
    file without any timing UI.
    """
    lines: list[list[Syllable]] = []
    cursor = start

    for raw_line in text.splitlines():
        units = raw_line.split()
        if not units:
            continue

        line: list[Syllable] = []
        for index, unit in enumerate(units):
            is_last = index == len(units) - 1
            line.append(
                Syllable(
                    text=unit,
                    start=cursor,
                    end=cursor + unit_seconds,
                    is_word_end=True,
                    is_line_end=is_last,
                )
            )
            cursor += unit_seconds

        lines.append(line)
        cursor += line_gap

    return lines


def all_syllables(lines: list[list[Syllable]]) -> list[Syllable]:
    """Flatten lines into a single time-ordered list."""
    return [syl for line in lines for syl in line]