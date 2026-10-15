"""Milestone 2 tests: lyric splitting, the timing table, and its validation."""

from __future__ import annotations

import math

import pytest

from app.lyrics import split_syllables, tokenize
from app.timing import (
    LyricRow,
    group_by_line,
    respace,
    rows_from_text,
    rows_to_lyric_lines,
    validate_rows,
)

LYRICS = "hello world\nsecond line here"


# --- splitting ------------------------------------------------------------


def test_words_mode_keeps_every_token():
    units = tokenize(LYRICS, mode="words")

    assert [line for line in units] == [
        [("hello", True), ("world", True)],
        [("second", True), ("line", True), ("here", True)],
    ]


def test_syllables_marks_only_the_last_piece_of_each_word():
    units = tokenize("hello world", mode="syllables")
    flat = [piece for line in units for piece in line]

    assert [text for text, _ in flat] == ["hel", "lo", "world"]
    # Only the final piece of a word closes the word.
    assert [is_end for _, is_end in flat] == [False, True, True]


def test_split_syllables_handles_obvious_cases():
    assert split_syllables("hello") == ["hel", "lo"]
    assert split_syllables("cat") == ["cat"]  # too short to split
    assert split_syllables("sky") == ["sky"]  # one vowel group
    assert split_syllables("") == []


def test_split_syllables_never_loses_characters():
    for word in ("hello", "twinkle", "diamond", "beautiful", "karaoke", "rhythm"):
        assert "".join(split_syllables(word)) == word.lower()


def test_unknown_split_mode_is_rejected():
    with pytest.raises(ValueError):
        tokenize("hello", mode="phonemes")


# --- building rows --------------------------------------------------------


def test_rows_from_text_spaces_rows_evenly():
    rows = rows_from_text("hello world", start=1.0, unit=0.5, line_gap=0.0)

    assert [r.text for r in rows] == ["hello", "world"]
    assert rows[0].start == 1.0
    assert rows[1].start == 1.5
    assert rows[0].end == 1.5


def test_line_gap_default_is_zero_so_rows_stay_on_their_slots():
    rows = rows_from_text("one two\nthree four", unit=0.5)

    assert [r.start for r in rows] == [0.0, 0.5, 1.0, 1.5]


def test_blank_lines_do_not_create_rows():
    rows = rows_from_text("one\n\n\ntwo")

    assert [r.text for r in rows] == ["one", "two"]
    assert [r.line for r in rows] == [1, 2]


def test_rows_convert_back_into_lyric_lines():
    rows = rows_from_text(LYRICS)
    lines = rows_to_lyric_lines(rows)

    assert len(lines) == 2
    assert [s.text for s in lines[0]] == ["hello", "world"]
    assert [s.text for s in lines[1]] == ["second", "line", "here"]


def test_group_by_line_preserves_order():
    rows = rows_from_text(LYRICS)
    grouped = group_by_line(rows)

    assert [len(group) for group in grouped] == [2, 3]


def test_respace_reapplies_even_timings_but_keeps_text():
    rows = rows_from_text("one two three")
    moved = respace(rows, start=2.0, unit=1.0)

    assert [r.text for r in moved] == ["one", "two", "three"]
    assert [r.start for r in moved] == [2.0, 3.0, 4.0]


# --- validation -----------------------------------------------------------


def test_valid_rows_produce_no_problems():
    assert validate_rows(rows_from_text(LYRICS)) == []


def test_empty_lyrics_is_reported_clearly():
    problems = validate_rows([])

    assert len(problems) == 1
    assert "No lyrics yet" in problems[0]


def test_rows_where_every_text_is_blank_is_reported():
    rows = [LyricRow(line=1, text="   ", start=0.0, end=1.0) for _ in range(2)]
    problems = validate_rows(rows)

    assert any("text is empty" in p for p in problems)
    assert any("Every row is empty" in p for p in problems)


def test_non_numeric_time_is_reported_with_the_row_number():
    rows = [LyricRow(line=1, text="ok", start=0.0, end=1.0), LyricRow(line=1, text="bad", start="abc", end=2.0)]
    problems = validate_rows(rows)

    assert any("Row 2" in p and "must be a number" in p for p in problems)


def test_nan_time_is_reported_as_not_a_real_number():
    rows = [LyricRow(line=1, text="ok", start=0.0, end=1.0), LyricRow(line=1, text="gap", start=math.nan, end=2.0)]
    problems = validate_rows(rows)

    assert any("Row 2" in p and "real number" in p for p in problems)


def test_infinite_time_is_rejected():
    rows = [LyricRow(line=1, text="ok", start=0.0, end=float("inf"))]
    problems = validate_rows(rows)

    assert problems, "infinity must not be accepted as a time"


def test_negative_time_is_rejected_and_explained():
    rows = [LyricRow(line=1, text="ok", start=-0.5, end=1.0)]
    problems = validate_rows(rows)

    assert any("cannot be negative" in p for p in problems)


def test_end_before_start_is_rejected():
    rows = [LyricRow(line=1, text="backwards", start=5.0, end=4.0)]
    problems = validate_rows(rows)

    assert any("is before 'start'" in p for p in problems)


def test_out_of_order_rows_are_rejected():
    rows = [
        LyricRow(line=1, text="first", start=0.0, end=1.0),
        LyricRow(line=1, text="second", start=2.0, end=3.0),
        LyricRow(line=1, text="latecomer", start=0.5, end=1.5),
    ]
    problems = validate_rows(rows)

    assert any("must stay in time order" in p for p in problems)


def test_equal_start_times_are_allowed():
    """Two words sung together share a start; that is not out of order."""
    rows = [
        LyricRow(line=1, text="a", start=1.0, end=1.5),
        LyricRow(line=1, text="b", start=1.0, end=1.5),
    ]

    assert validate_rows(rows) == []


def test_validation_survives_a_completely_broken_row():
    rows = [LyricRow(line=1, text=None, start=math.nan, end=-1.0)]
    problems = validate_rows(rows)

    assert len(problems) >= 2