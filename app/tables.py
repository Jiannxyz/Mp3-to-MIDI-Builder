"""Conversions between the edit tables and pandas DataFrames.

Streamlit's `st.data_editor` works with DataFrames, so rows go out to the
widget and come back as a DataFrame. That round trip is where NaN appears: a
cell the user clears comes back as NaN rather than a number.

These helpers live outside `streamlit_app.py` on purpose - importing that module
runs the whole UI, which would make this logic untestable.
"""

from __future__ import annotations

import math

import pandas as pd

from .notes import NoteRow
from .timing import LyricRow

LYRIC_COLUMNS = ["line", "text", "start (s)", "end (s)", "word end"]
NOTE_COLUMNS = ["channel", "pitch", "start (s)", "duration (s)", "velocity"]


def to_float(value) -> float:
    """Coerce a table cell to float, keeping NaN so validation can report it."""
    if value is None:
        return float("nan")
    try:
        if pd.isna(value):
            return float("nan")
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def to_int(value):
    """Coerce to int, returning the value unchanged when it is not numeric.

    Returning the original keeps a bad cell visibly invalid so validation can
    explain it, instead of quietly turning "loud" into 0.
    """
    number = to_float(value)
    if math.isnan(number):
        return value
    return int(round(number))


def to_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return bool(value)


# --- lyric table ----------------------------------------------------------


def lyric_rows_to_frame(rows: list[LyricRow]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "line": pd.Series([r.line for r in rows], dtype="int64"),
            "text": pd.Series([r.text for r in rows], dtype="string"),
            "start (s)": pd.Series([r.start for r in rows], dtype="float64"),
            "end (s)": pd.Series([r.end for r in rows], dtype="float64"),
            "word end": pd.Series([r.word_end for r in rows], dtype="bool"),
        },
        columns=LYRIC_COLUMNS,
    )


def frame_to_lyric_rows(frame: pd.DataFrame) -> list[LyricRow]:
    rows: list[LyricRow] = []
    for _, record in frame.iterrows():
        line = to_int(record.get("line"))
        text = record.get("text")
        rows.append(
            LyricRow(
                line=line if isinstance(line, int) else 1,
                text="" if text is None or pd.isna(text) else str(text),
                start=to_float(record.get("start (s)")),
                end=to_float(record.get("end (s)")),
                word_end=to_bool(record.get("word end")),
            )
        )
    return rows


# --- note table -----------------------------------------------------------


def note_rows_to_frame(rows: list[NoteRow]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "channel": pd.Series([r.channel for r in rows], dtype="int64"),
            "pitch": pd.Series([r.pitch for r in rows], dtype="int64"),
            "start (s)": pd.Series([r.start for r in rows], dtype="float64"),
            "duration (s)": pd.Series([r.duration for r in rows], dtype="float64"),
            "velocity": pd.Series([r.velocity for r in rows], dtype="int64"),
        },
        columns=NOTE_COLUMNS,
    )


def blank_note_frame() -> pd.DataFrame:
    """An empty frame with explicit dtypes, so st.data_editor behaves."""
    return pd.DataFrame(
        {
            "channel": pd.Series(dtype="int64"),
            "pitch": pd.Series(dtype="int64"),
            "start (s)": pd.Series(dtype="float64"),
            "duration (s)": pd.Series(dtype="float64"),
            "velocity": pd.Series(dtype="int64"),
        },
        columns=NOTE_COLUMNS,
    )


def frame_to_note_rows(frame: pd.DataFrame) -> list[NoteRow]:
    rows: list[NoteRow] = []
    for _, record in frame.iterrows():
        rows.append(
            NoteRow(
                channel=to_int(record.get("channel")),
                pitch=to_int(record.get("pitch")),
                start=to_float(record.get("start (s)")),
                duration=to_float(record.get("duration (s)")),
                velocity=to_int(record.get("velocity")),
            )
        )
    return rows


def note_frame_for(rows: list[NoteRow]) -> pd.DataFrame:
    """The frame to hand to `st.data_editor` for the note table.

    Never write `some_frame or other_frame`. Python evaluates that by calling
    `DataFrame.__bool__`, which pandas refuses for any frame with more than one
    column, raising "The truth value of a DataFrame is ambiguous". Test the row
    list instead.
    """
    return note_rows_to_frame(rows) if rows else blank_note_frame()


def lyric_frame_for(rows: list[LyricRow]) -> pd.DataFrame:
    """The frame to hand to `st.data_editor` for the lyric table.

    Empty input already produces a correctly typed, correctly columned frame,
    so there is no `or fallback` here to get wrong.
    """
    return lyric_rows_to_frame(rows)


def lyrics_preview_frame(rows: list[LyricRow]) -> pd.DataFrame:
    """Line-by-line summary shown next to the audio player."""
    from .timing import group_by_line

    preview = [
        {
            "Line": group[0].line,
            "Starts": f"{group[0].start:.2f}s",
            "Text": " ".join(row.text for row in group),
        }
        for group in group_by_line(rows)
    ]
    return pd.DataFrame(preview, columns=["Line", "Starts", "Text"])