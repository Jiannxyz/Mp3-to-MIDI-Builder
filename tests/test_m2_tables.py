"""Milestone 2 tests: the DataFrame round trip used by st.data_editor.

A cell the user clears comes back from the widget as NaN, so these cover the
conversion in both directions and the NaN case specifically.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from app.demo import demo_lyric_rows, demo_note_rows
from app.notes import NoteRow, validate_note_rows
from app.tables import (
    blank_note_frame,
    frame_to_lyric_rows,
    frame_to_note_rows,
    lyric_frame_for,
    lyric_rows_to_frame,
    lyrics_preview_frame,
    note_frame_for,
    note_rows_to_frame,
    to_bool,
    to_float,
    to_int,
)
from app.timing import LyricRow, rows_from_text, validate_rows


# --- scalar coercion ------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [(1, 1.0), (1.5, 1.5), ("2.5", 2.5), (None, float("nan")), (float("nan"), float("nan")), ("abc", float("nan"))],
)
def test_to_float(value, expected):
    result = to_float(value)
    if math.isnan(expected):
        assert math.isnan(result)
    else:
        assert result == expected


def test_to_int_keeps_unusable_values_visible():
    """Returning the original lets validation explain the problem."""
    assert to_int("loud") == "loud"
    assert to_int(None) is None
    assert to_int(60) == 60
    assert to_int(60.4) == 60


@pytest.mark.parametrize("value,expected", [(True, True), (False, False), ("True", True), ("no", False), (None, False)])
def test_to_bool(value, expected):
    assert to_bool(value) is expected


# --- lyric table round trip ----------------------------------------------


def test_lyric_rows_survive_a_frame_round_trip():
    rows = rows_from_text("hello world\nsecond line", unit=0.5)
    restored = frame_to_lyric_rows(lyric_rows_to_frame(rows))

    assert [(r.line, r.text, r.start, r.end, r.word_end) for r in restored] == [
        (r.line, r.text, r.start, r.end, r.word_end) for r in rows
    ]


def test_lyric_frame_has_stable_columns_even_when_empty():
    frame = lyric_rows_to_frame([])
    assert list(frame.columns) == ["line", "text", "start (s)", "end (s)", "word end"]
    assert len(frame) == 0


def test_cleared_lyric_cell_becomes_an_empty_string_not_the_text_nan():
    frame = lyric_rows_to_frame(rows_from_text("hello"))
    frame.loc[0, "text"] = None

    rows = frame_to_lyric_rows(frame)
    assert rows[0].text == ""
    assert any("text is empty" in p for p in validate_rows(rows))


def test_cleared_time_cell_becomes_nan_and_is_reported():
    frame = lyric_rows_to_frame(rows_from_text("hello"))
    frame.loc[0, "start (s)"] = float("nan")

    rows = frame_to_lyric_rows(frame)
    assert math.isnan(rows[0].start)
    assert validate_rows(rows), "a NaN time must be reported, not silently dropped"


def test_a_user_edited_frame_round_trips_back():
    frame = lyric_rows_to_frame(rows_from_text("one two"))
    frame.loc[1, "start (s)"] = 9.5
    frame.loc[1, "text"] = "changed"

    rows = frame_to_lyric_rows(frame)
    assert rows[1].text == "changed"
    assert rows[1].start == 9.5


# --- note table round trip ------------------------------------------------


def test_note_rows_survive_a_frame_round_trip():
    rows = [NoteRow(channel=0, pitch=60, start=1.0, duration=0.5, velocity=90)]
    restored = frame_to_note_rows(note_rows_to_frame(rows))

    assert [(r.channel, r.pitch, r.start, r.duration, r.velocity) for r in restored] == [
        (r.channel, r.pitch, r.start, r.duration, r.velocity) for r in rows
    ]


def test_blank_note_frame_has_the_right_columns_and_dtypes():
    frame = blank_note_frame()

    assert list(frame.columns) == ["channel", "pitch", "start (s)", "duration (s)", "velocity", "source"]
    assert len(frame) == 0
    assert str(frame["start (s)"].dtype) == "float64"


def test_bad_note_cell_survives_the_round_trip_and_is_reported():
    # pandas refuses to put a string into an int64 column, so widen the column
    # first - that is what a non-numeric value from the widget would look like.
    frame = note_rows_to_frame([NoteRow()])
    frame["pitch"] = frame["pitch"].astype(object)
    frame.loc[0, "pitch"] = "loud"

    rows = frame_to_note_rows(frame)
    assert validate_note_rows(rows), "a non-numeric pitch must be reported"


def test_cleared_note_time_becomes_nan():
    frame = note_rows_to_frame([NoteRow()])
    frame.loc[0, "duration (s)"] = float("nan")

    rows = frame_to_note_rows(frame)
    assert math.isnan(rows[0].duration)
    assert validate_note_rows(rows)


# --- preview --------------------------------------------------------------


def test_lyric_preview_groups_words_into_lines():
    rows = rows_from_text("hello world\nsecond line", unit=0.5)
    preview = lyrics_preview_frame(rows)

    assert list(preview.columns) == ["Line", "Starts", "Text"]
    assert preview.iloc[0]["Text"] == "hello world"
    assert preview.iloc[1]["Text"] == "second line"
    assert preview.iloc[1]["Starts"] == "1.00s"


def test_lyric_preview_of_nothing_is_empty_but_well_formed():
    preview = lyrics_preview_frame([])

    assert len(preview) == 0
    assert list(preview.columns) == ["Line", "Starts", "Text"]


# --- regression: DataFrame truthiness -------------------------------------


def test_note_frame_for_empty_rows_does_not_raise():
    """`frame or fallback` raised 'truth value of a DataFrame is ambiguous'.

    The note table crashed in the app on first load because `or` calls
    `DataFrame.__bool__`. These helpers test the row list instead.
    """
    frame = note_frame_for([])

    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 0
    assert list(frame.columns) == ["channel", "pitch", "start (s)", "duration (s)", "velocity", "source"]


def test_note_frame_for_keeps_existing_rows():
    rows = [NoteRow(pitch=64, start=1.0)]
    frame = note_frame_for(rows)

    assert len(frame) == 1
    assert frame.iloc[0]["pitch"] == 64


def test_lyric_frame_for_handles_both_empty_and_populated():
    empty = lyric_frame_for([])
    assert isinstance(empty, pd.DataFrame) and len(empty) == 0

    populated = lyric_frame_for(rows_from_text("hello world"))
    assert len(populated) == 2


def test_a_dataframe_cannot_be_truth_tested_which_is_why_helpers_exist():
    """Documents the constraint the helpers exist to satisfy."""
    frame = note_frame_for([])

    with pytest.raises(ValueError, match="truth value"):
        bool(frame)


# --- demo rows feed the tables --------------------------------------------


def test_demo_rows_are_the_right_shape_for_the_tables():
    """Regression: the demo button stored Syllable/NoteTrack, not rows.

    `DemoSong.lyric_lines` is nested Syllable lists and `note_tracks` are
    NoteTrack objects. Neither is what the tables consume, so the app crashed
    on 'Load demo song'.
    """
    lyrics = demo_lyric_rows()
    notes = demo_note_rows()

    assert lyrics and all(isinstance(row, LyricRow) for row in lyrics)
    assert notes and all(isinstance(row, NoteRow) for row in notes)

    assert validate_rows(lyrics) == []
    assert validate_note_rows(notes) == []

    # And they must survive the widget round trip.
    assert len(frame_to_lyric_rows(lyric_frame_for(lyrics))) == len(lyrics)
    assert len(frame_to_note_rows(note_frame_for(notes))) == len(notes)


def test_demo_rows_use_line_numbers_so_lines_group_correctly():
    lyrics = demo_lyric_rows()

    assert len({row.line for row in lyrics}) > 1
    assert [row.text for row in lyrics] == [row.text for row in lyrics]