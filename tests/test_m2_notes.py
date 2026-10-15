"""Milestone 2 tests: the note table, and end-to-end export from edited tables."""

from __future__ import annotations

import pytest

from app.export_service import export_song_to_bytes
from app.karaoke_dialects import DEFAULT_DIALECT
from app.midi_build import seconds_to_ticks
from app.notes import NoteRow, empty_note_rows, example_note_rows, rows_to_note_tracks, validate_note_rows
from app.timing import rows_from_text, rows_to_lyric_lines, validate_rows
from app.validate_midi import parse_bytes

LYRICS = "hello world\nthis is a test"


# --- note table -----------------------------------------------------------


def test_empty_note_table_is_valid():
    """No notes is a normal result, not an error: the file is lyrics-only."""
    assert validate_note_rows(empty_note_rows()) == []


def test_example_notes_are_valid():
    assert validate_note_rows(example_note_rows(4)) == []


def test_example_notes_are_placeholders_at_expected_times():
    rows = example_note_rows(3, start=1.0, spacing=0.5)

    assert [r.start for r in rows] == [1.0, 1.5, 2.0]
    assert all(r.pitch == 60 for r in rows)


def test_pitch_outside_midi_range_is_rejected():
    rows = [NoteRow(pitch=200)]
    problems = validate_note_rows(rows)

    assert any("between 0 and 127" in p for p in problems)


def test_negative_pitch_is_rejected():
    assert validate_note_rows([NoteRow(pitch=-1)])


def test_fractional_pitch_is_rejected():
    problems = validate_note_rows([NoteRow(pitch=60.5)])

    assert any("whole number" in p for p in problems)


def test_velocity_must_be_within_range():
    assert validate_note_rows([NoteRow(velocity=0)])
    assert validate_note_rows([NoteRow(velocity=200)])


def test_channel_must_be_within_range():
    assert validate_note_rows([NoteRow(channel=16)])


def test_zero_duration_note_is_rejected():
    problems = validate_note_rows([NoteRow(duration=0)])

    assert any("longer than 0" in p for p in problems)


def test_negative_duration_is_rejected():
    assert validate_note_rows([NoteRow(duration=-1.0)])


def test_negative_start_is_rejected():
    problems = validate_note_rows([NoteRow(start=-2.0)])

    assert any("cannot be negative" in p for p in problems)


def test_non_numeric_note_values_are_reported_with_the_row():
    rows = [NoteRow(), NoteRow(pitch="loud")]
    problems = validate_note_rows(rows)

    assert any("Row 2" in p and "whole number" in p for p in problems)


def test_problem_messages_name_the_row_number():
    rows = [NoteRow(), NoteRow(), NoteRow(pitch=999)]
    problems = validate_note_rows(rows)

    assert any(p.startswith("Row 3:") for p in problems)


# --- note tracks ----------------------------------------------------------


def test_notes_are_grouped_into_one_track_per_channel():
    rows = [
        NoteRow(channel=0, pitch=60, start=0.0, duration=0.5),
        NoteRow(channel=1, pitch=48, start=0.0, duration=1.0),
        NoteRow(channel=0, pitch=62, start=0.5, duration=0.5),
    ]
    tracks = rows_to_note_tracks(rows)

    assert [t.name for t in tracks] == ["melody", "ch1"]
    assert len(tracks[0].notes) == 2
    assert tracks[0].program == 0, "channel 0 is the vocal melody"
    assert tracks[1].program is None


def test_notes_within_a_track_come_out_in_time_order():
    rows = [
        NoteRow(channel=0, pitch=60, start=2.0, duration=0.5),
        NoteRow(channel=0, pitch=62, start=0.5, duration=0.5),
    ]
    tracks = rows_to_note_tracks(rows)

    assert [n.start for n in tracks[0].notes] == [0.5, 2.0]


def test_empty_note_rows_produce_no_tracks():
    assert rows_to_note_tracks(empty_note_rows()) == []


# --- end-to-end export ----------------------------------------------------


def _export(lyric_rows, note_rows, **kwargs):
    return export_song_to_bytes(
        title=kwargs.pop("title", "Test Song"),
        artist=kwargs.pop("artist", "Test Singer"),
        lyric_lines=rows_to_lyric_lines(lyric_rows),
        note_tracks=rows_to_note_tracks(note_rows),
        dialect=DEFAULT_DIALECT,
        **kwargs,
    )


def test_export_from_edited_tables_succeeds():
    result = _export(rows_from_text(LYRICS), example_note_rows(4, start=0.0, spacing=0.5))

    assert result.ok, result.summary
    assert result.filename == "Test Song - Test Singer.mid"
    assert result.data[:4] == b"MThd"


def test_exported_file_contains_the_lyrics_and_the_notes():
    lyric_rows = rows_from_text(LYRICS)
    note_rows = example_note_rows(4, start=0.0, spacing=0.5)
    result = _export(lyric_rows, note_rows)

    parsed = parse_bytes(result.data)
    words = [e.text for e in parsed.meta_events(0x05) if e.text not in ("\r", "\n")]
    note_ons = [e for e in parsed.channel_events() if e.kind == "note_on"]

    assert words == [row.text for row in lyric_rows]
    assert len(note_ons) == len(note_rows)
    assert sorted(e.tick for e in note_ons) == sorted(
        seconds_to_ticks(row.start) for row in note_rows
    )


def test_lyrics_only_export_is_allowed_and_has_no_notes():
    """Missing notes must not block export - it yields a lyrics-only file."""
    result = _export(rows_from_text(LYRICS), [])

    assert result.ok, result.summary
    parsed = parse_bytes(result.data)
    assert parsed.meta_events(0x05), "lyrics should still be written"
    assert [e for e in parsed.channel_events() if e.kind == "note_on"] == []


def test_export_refuses_out_of_order_lyric_timings():
    from app.timing import LyricRow

    rows = [
        LyricRow(line=1, text="first", start=0.0, end=1.0),
        LyricRow(line=1, text="late", start=5.0, end=6.0),
        LyricRow(line=1, text="early", start=2.0, end=3.0),
    ]
    result = _export(rows, [])

    assert not result.ok
    assert any("words must stay in order" in p for p in result.problems)
    assert "words must stay in order" in result.problems[0], "cause must be reported first"


def test_export_refuses_invalid_times():
    from app.timing import LyricRow

    rows = [LyricRow(line=1, text="bad", start=-1.0, end=0.5)]
    result = _export(rows, [])

    assert not result.ok
    assert result.problems


def test_empty_lyrics_cannot_be_exported():
    result = _export([], [])

    assert not result.ok
    assert any("No lyrics yet" in p for p in result.problems)


def test_timing_rows_must_pass_validation_before_export():
    """A guard mirroring what the UI does before offering the button."""
    rows = rows_from_text(LYRICS)
    assert validate_rows(rows) == []

    broken = rows + [rows[0].__class__(line=1, text="dup", start=0.0, end=0.1)]
    assert validate_rows(broken) != []


def test_lyric_offset_still_works_on_exported_tables():
    lyric_rows = rows_from_text(LYRICS)
    result = _export(lyric_rows, [], lyric_offset=0.25)

    assert result.ok, result.summary
    parsed = parse_bytes(result.data)
    words = [e.tick for e in parsed.meta_events(0x05) if e.text not in ("\r", "\n")]

    assert words[0] == seconds_to_ticks(lyric_rows[0].start + 0.25)