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


VOWELS = frozenset("aeiouy")


def split_syllables(word: str) -> list[str]:
    """Best-effort split of one word into sung syllables.

    This is a heuristic, not linguistics. It finds vowel groups and cuts in the
    consonant run that follows each one, which gets simple words right
    ("hello" -> "hel"/"lo") and is wrong for English spelling exceptions
    ("every" -> "ev"/"ry", "love" -> "l"/"ove"), loanwords and non-Latin text.

    That is why every generated syllable is an EDITABLE row in the timing table.
    Do not present the output as correct syllable division.
    """
    if not word:
        return []

    lowered = word.lower()
    if len(lowered) < 3:
        return [word]

    groups: list[list[int]] = []
    for index, character in enumerate(lowered):
        if character not in VOWELS:
            continue
        if groups and index == groups[-1][1]:
            groups[-1][1] = index
        else:
            groups.append([index, index])

    if len(groups) < 2:
        return [word]

    cuts: list[int] = []
    first_vowel = groups[0][0]
    for group in groups[:-1]:
        after = group[1] + 1
        while after < len(lowered) and lowered[after] not in VOWELS:
            after += 1
        # Give the final consonant of the run to the NEXT syllable, unless that
        # would strip the current syllable of its vowel.
        cut = after - 1 if (after - 1) > first_vowel else group[1] + 1
        cuts.append(max(1, min(cut, len(lowered) - 1)))

    parts: list[str] = []
    position = 0
    for cut in cuts:
        if position < cut < len(lowered):
            parts.append(lowered[position:cut])
            position = cut
    parts.append(lowered[position:])
    parts = [part for part in parts if part]

    return parts if len(parts) > 1 else [word]


def tokenize(text: str, mode: str = "words") -> list[list[tuple[str, bool]]]:
    """Split lyric lines into (text, is_word_end) units.

    `mode` is "words" (one unit per whitespace-separated token, the safer
    default) or "syllables" (heuristic split, editable afterwards).
    """
    if mode not in ("words", "syllables"):
        raise ValueError(f"unknown mode: {mode!r} (use 'words' or 'syllables')")

    lines: list[list[tuple[str, bool]]] = []
    for raw_line in text.splitlines():
        tokens = raw_line.split()
        if not tokens:
            continue

        units: list[tuple[str, bool]] = []
        for token in tokens:
            if mode == "words":
                units.append((token, True))
                continue

            pieces = split_syllables(token)
            for piece in pieces:
                units.append((piece, piece is pieces[-1]))

        lines.append(units)

    return lines