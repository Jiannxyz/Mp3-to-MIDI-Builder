"""Milestone 1 tests: the generated karaoke files must be structurally correct.

These assert on raw bytes parsed by app.validate_midi, not on mido's own
round-trip, so a wrong encoding cannot pass.
"""

from __future__ import annotations

import pytest

from app.karaoke_dialects import DIALECTS
from app.lyrics import all_syllables, parse_lyrics
from app.make_m1_samples import build_for_dialect, validate_for_dialect
from app.midi_build import NoteEvent, NoteTrack, build_midi, save_midi, seconds_to_ticks
from app.validate_midi import MidiParseError, parse_bytes, parse_file

DIALECT_IDS = list(DIALECTS)


@pytest.mark.parametrize("dialect_id", DIALECT_IDS)
def test_sample_files_are_structurally_valid(dialect_id, tmp_path):
    dialect = DIALECTS[dialect_id]
    midi, lines, tracks = build_for_dialect(dialect)
    path = tmp_path / "sample.mid"
    save_midi(midi, path)

    assert validate_for_dialect(path, dialect, lines, tracks) == []


@pytest.mark.parametrize("dialect_id", DIALECT_IDS)
def test_baseline_and_variants_differ_by_one_dimension(dialect_id):
    """B/C/D each change exactly one thing from A, so results are attributable."""
    baseline = DIALECTS["A_fmt1_lyrics_ff05"]
    variant = DIALECTS[dialect_id]
    if dialect_id == "A_fmt1_lyrics_ff05":
        return

    # Separator handling (CR word breaks and LF line breaks together) counts as
    # a single dimension: D turns that handling off wholesale.
    dimensions = [
        variant.smf_format != baseline.smf_format,
        variant.lyric_meta != baseline.lyric_meta,
        (variant.word_separator, variant.line_separator)
        != (baseline.word_separator, baseline.line_separator),
        variant.banner != baseline.banner,
    ]
    assert sum(dimensions) == 1, f"{dialect_id} changes {sum(dimensions)} dimensions from the baseline"


@pytest.mark.parametrize("dialect_id", DIALECT_IDS)
def test_lyric_units_and_ticks_are_preserved(dialect_id, tmp_path):
    dialect = DIALECTS[dialect_id]
    midi, lines, _ = build_for_dialect(dialect)
    path = tmp_path / "sample.mid"
    save_midi(midi, path)

    parsed = parse_file(path)
    syllables = all_syllables(lines)
    lyric_ff = 0x05 if dialect.lyric_meta == "lyrics" else 0x01
    events = parsed.meta_events(lyric_ff)
    units = [e for e in events if e.text not in ("\r", "\n")]

    assert [e.text for e in units] == [s.text for s in syllables]
    assert [e.tick for e in units] == [seconds_to_ticks(s.start) for s in syllables]


def test_format_0_writes_a_single_merged_track(tmp_path):
    midi, _, _ = build_for_dialect(DIALECTS["B_fmt0_lyrics_ff05"])
    path = tmp_path / "fmt0.mid"
    save_midi(midi, path)

    parsed = parse_file(path)
    assert parsed.fmt == 0
    assert len(parsed.tracks) == 1


def test_format_1_keeps_note_tracks_separate(tmp_path):
    midi, _, _ = build_for_dialect(DIALECTS["A_fmt1_lyrics_ff05"])
    path = tmp_path / "fmt1.mid"
    save_midi(midi, path)

    parsed = parse_file(path)
    assert parsed.fmt == 1
    assert len(parsed.tracks) == 3  # metadata+lyrics, lead, bass


def test_notes_are_staggered_in_time_not_stacked_on_one_tick(tmp_path):
    """Regression: notes once all carried time=0 and played as a single sound.

    Counts passed while every note sat on tick 0, which SongHub rendered as one
    note followed by silence. Timing must therefore be asserted, not just count.
    """
    midi, _, tracks = build_for_dialect(DIALECTS["A_fmt1_lyrics_ff05"])
    path = tmp_path / "sample.mid"
    save_midi(midi, path)

    parsed = parse_file(path)
    on_ticks = [e.tick for e in parsed.channel_events() if e.kind == "note_on"]

    expected = [seconds_to_ticks(n.start) for t in tracks for n in t.notes]
    assert sorted(on_ticks) == sorted(expected)

    # Notes may legitimately coincide across channels (lead + bass together),
    # so uniqueness is asserted per channel, not globally.
    for channel in {e.channel for e in parsed.channel_events()}:
        per_channel = [e.tick for e in parsed.channel_events() if e.kind == "note_on" and e.channel == channel]
        assert len(set(per_channel)) == len(per_channel), (
            f"channel {channel} has notes sharing timestamps: {sorted(per_channel)}"
        )

    # The melody must actually advance through the song, not just start at 0.
    assert on_ticks[-1] > on_ticks[0]


def test_note_spacing_matches_the_rhythm(tmp_path):
    midi, _, _ = build_for_dialect(DIALECTS["A_fmt1_lyrics_ff05"])
    path = tmp_path / "sample.mid"
    save_midi(midi, path)

    parsed = parse_file(path)
    on_ticks = [e.tick for e in parsed.channel_events() if e.kind == "note_on"]
    gaps = {b - a for a, b in zip(on_ticks, on_ticks[1:]) if b > a}

    # Lead notes are 0.45s apart; bass repeats every 1.8s, so a few distinct gaps.
    assert 432 in gaps, f"expected 432-tick (0.45s) lead spacing, found {sorted(gaps)}"
    assert max(gaps) > 0


def test_lyric_offset_moves_lyrics_but_never_notes(tmp_path):
    """The lyric offset knob must shift text only, and clamp at zero."""
    lines = parse_lyrics("one two three", start=1.0, unit_seconds=0.5, line_gap=0.0)
    tracks = [NoteTrack("lead", notes=[NoteEvent(start=1.0, duration=0.5, pitch=60, channel=0)])]

    def ticks_at(offset):
        midi = build_midi(
            title="t",
            artist="",
            lines=lines,
            note_tracks=tracks,
            dialect=DIALECTS["A_fmt1_lyrics_ff05"],
            lyric_offset=offset,
        )
        path = tmp_path / f"off{offset}.mid"
        save_midi(midi, path)
        parsed = parse_file(path)
        lyric = [e.tick for e in parsed.meta_events(0x05) if e.text not in ("\r", "\n")]
        note = [e.tick for e in parsed.channel_events() if e.kind == "note_on"]
        return lyric, note

    base_lyrics, base_notes = ticks_at(0.0)
    shifted_lyrics, shifted_notes = ticks_at(0.25)

    assert shifted_notes == base_notes, "notes must never move when only lyrics shift"
    assert shifted_lyrics[0] == base_lyrics[0] + seconds_to_ticks(0.25)

    # A negative offset that would go before the start clamps to tick 0
    # instead of producing a negative time delta.
    clamped, _ = ticks_at(-5.0)
    assert min(clamped) >= 0


def test_notes_are_balanced_and_in_range(tmp_path):
    midi, _, tracks = build_for_dialect(DIALECTS["A_fmt1_lyrics_ff05"])
    path = tmp_path / "sample.mid"
    save_midi(midi, path)

    parsed = parse_file(path)
    ons = [e for e in parsed.channel_events() if e.kind == "note_on"]
    offs = [e for e in parsed.channel_events() if e.kind == "note_off"]

    assert len(ons) == len(offs) == sum(len(t.notes) for t in tracks)
    assert all(0 <= e.note <= 127 for e in ons)
    assert all(0 <= e.channel <= 15 for e in ons)


def test_tempo_and_division_are_written(tmp_path):
    midi, _, _ = build_for_dialect(DIALECTS["A_fmt1_lyrics_ff05"])
    path = tmp_path / "sample.mid"
    save_midi(midi, path)

    parsed = parse_file(path)
    assert parsed.division == 480
    assert not parsed.smpte_timing
    assert parsed.meta_events(0x51), "set_tempo must be present"


def test_ticks_scale_with_tempo():
    assert seconds_to_ticks(1.0) == 960  # 480 PPQ at 120 BPM
    assert seconds_to_ticks(0.0) == 0


def test_parser_rejects_non_midi_bytes():
    with pytest.raises(MidiParseError):
        parse_bytes(b"this is not a midi file")


def test_parser_rejects_truncated_header():
    with pytest.raises(MidiParseError):
        parse_bytes(b"MThd\x00")


def test_lyric_parsing_skips_blank_lines_and_advances_time():
    lines = parse_lyrics("one two\n\nthree", start=1.0, unit_seconds=0.5, line_gap=0.25)

    assert len(lines) == 2
    flat = all_syllables(lines)
    assert [s.text for s in flat] == ["one", "two", "three"]
    assert flat[0].start == 1.0
    # line gap pushes "three" past the two units plus the gap
    assert flat[2].start == pytest.approx(1.0 + 0.5 * 2 + 0.25)


def test_invalid_note_data_is_rejected():
    for bad_note in (
        NoteEvent(start=0, duration=0.5, pitch=999),  # pitch out of 0-127
        NoteEvent(start=0, duration=0, pitch=60),  # zero duration
        NoteEvent(start=0, duration=0.5, pitch=60, channel=99),  # channel out of 0-15
    ):
        with pytest.raises(ValueError):
            build_midi(
                title="t",
                artist="",
                lines=[],
                note_tracks=[NoteTrack("x", [bad_note])],
                dialect=DIALECTS["A_fmt1_lyrics_ff05"],
            )


def test_negative_time_is_rejected():
    with pytest.raises(ValueError):
        seconds_to_ticks(-1.0)