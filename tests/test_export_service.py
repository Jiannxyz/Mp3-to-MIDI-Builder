"""Tests for the export service and the demo song.

Requirement coverage: a generated MIDI is reopened and checked for note events,
lyric events and their expected timings.
"""

from __future__ import annotations

import io

import mido
import pytest

from app.demo import MELODY, SPACING, build_demo_song
from app.export_service import export_song_to_bytes, safe_filename, validate_export
from app.karaoke_dialects import DEFAULT_DIALECT
from app.lyrics import all_syllables
from app.midi_build import NoteEvent, NoteTrack, build_midi, save_midi, seconds_to_ticks
from app.validate_midi import MidiParseError, parse_bytes


@pytest.fixture
def demo():
    return build_demo_song()


def _export(song, **overrides):
    options = dict(
        title=song.title,
        artist=song.artist,
        lyric_lines=song.lyric_lines,
        note_tracks=song.note_tracks,
        bpm=song.tempo_bpm,
    )
    options.update(overrides)
    return export_song_to_bytes(**options)


# --- basic contract -------------------------------------------------------


def test_export_produces_valid_verified_bytes(demo):
    result = _export(demo)

    assert result.ok, result.summary
    assert result.data[:4] == b"MThd"
    assert result.filename.endswith(".mid")
    assert "Twinkle" in result.filename


def test_export_contains_note_events_lyric_events_and_tempo(demo):
    result = _export(demo)
    parsed = parse_bytes(result.data)

    note_ons = [e for e in parsed.channel_events() if e.kind == "note_on"]
    lyric_ff = 0x05 if DEFAULT_DIALECT.lyric_meta == "lyrics" else 0x01
    lyrics = [e for e in parsed.meta_events(lyric_ff) if e.text not in ("\r", "\n")]

    assert len(note_ons) == len(MELODY) + 4  # melody plus 4 bass notes
    assert len(lyrics) == len(all_syllables(demo.lyric_lines))
    assert parsed.meta_events(0x51), "tempo event (FF 51) must be present"
    assert parsed.division == 480


def test_reopening_gives_back_the_same_words_and_times(demo):
    result = _export(demo)
    parsed = parse_bytes(result.data)

    expected = all_syllables(demo.lyric_lines)
    lyric_ff = 0x05 if DEFAULT_DIALECT.lyric_meta == "lyrics" else 0x01
    got = [e for e in parsed.meta_events(lyric_ff) if e.text not in ("\r", "\n")]

    assert [e.text for e in got] == [s.text for s in expected]
    assert [e.tick for e in got] == [seconds_to_ticks(s.start) for s in expected]


def test_note_timings_survive_the_round_trip(demo):
    result = _export(demo)
    parsed = parse_bytes(result.data)

    expected = sorted(
        seconds_to_ticks(n.start) for track in demo.note_tracks for n in track.notes
    )
    got = sorted(e.tick for e in parsed.channel_events() if e.kind == "note_on")

    assert got == expected


def test_tempo_event_encodes_the_requested_bpm(demo):
    result = _export(demo, bpm=100)

    midi = mido.MidiFile(file=io.BytesIO(result.data))
    tempos = [msg.tempo for track in midi.tracks for msg in track if msg.type == "set_tempo"]

    assert len(tempos) == 1, "exactly one tempo event expected"
    assert mido.tempo2bpm(tempos[0]) == 100


# --- ordering and delta times (requirement 4) -----------------------------


def test_events_are_ordered_and_deltas_are_never_negative(tmp_path):
    """Absolute times must be non-decreasing, so every delta is >= 0.

    Delta times are what a player actually reads. A negative delta means the
    stream is corrupt, and mido would not notice.
    """
    demo = build_demo_song()
    midi = build_midi(
        title=demo.title,
        artist=demo.artist,
        lines=demo.lyric_lines,
        note_tracks=demo.note_tracks,
        dialect=DEFAULT_DIALECT,
        bpm=demo.tempo_bpm,
    )
    path = tmp_path / "d.mid"
    save_midi(midi, path)

    parsed = parse_bytes(path.read_bytes())
    for index, track in enumerate(parsed.tracks):
        ticks = [event.tick for event in track]
        assert ticks == sorted(ticks), f"track {index} events are not in time order"


def test_delta_times_reconstruct_the_intended_timeline(demo):
    """Sum of deltas must equal each event's absolute tick.

    This is the check that would have caught the 'all notes at tick 0' bug:
    counts were right, but the deltas summed to nothing.
    """
    data = _export(demo).data
    raw_tracks = _raw_delta_sums(data)

    for absolute_ticks, deltas in raw_tracks:
        running = 0
        for delta, expected in zip(deltas, absolute_ticks):
            assert delta >= 0, "delta times must never be negative"
            running += delta
            assert running == expected


def _raw_delta_sums(data: bytes):
    """Decode delta times directly, independently of our event parser."""
    index = 14  # 'MThd' + 6-byte header
    out = []

    def read_vlq(chunk: bytes, pos: int):
        value = 0
        while True:
            byte = chunk[pos]
            pos += 1
            value = (value << 7) | (byte & 0x7F)
            if not byte & 0x80:
                return value, pos

    while index < len(data):
        assert data[index : index + 4] == b"MTrk"
        length = int.from_bytes(data[index + 4 : index + 8], "big")
        chunk = data[index + 8 : index + 8 + length]
        index += 8 + length

        deltas: list[int] = []
        absolute: list[int] = []
        pos = 0
        tick = 0
        running_status: int | None = None

        while pos < len(chunk):
            delta, pos = read_vlq(chunk, pos)
            tick += delta
            deltas.append(delta)
            absolute.append(tick)

            byte = chunk[pos]
            if byte & 0x80:
                pos += 1
                status = byte
                if status < 0xF0:
                    running_status = status
            else:
                # Running status: the status byte was omitted by the writer.
                status = running_status

            if status == 0xFF:  # meta event
                pos += 1  # meta type
                payload_length, pos = read_vlq(chunk, pos)
                pos += payload_length
            elif status in (0xF0, 0xF7):  # sysex
                payload_length, pos = read_vlq(chunk, pos)
                pos += payload_length
            else:
                pos += 1 if (status & 0xF0) in (0xC0, 0xD0) else 2

        out.append((absolute, deltas))

    return out


def test_notes_are_not_collapsed_onto_one_tick(demo):
    """Regression guard: notes once all sat at tick 0 and sounded as one note."""
    data = _export(demo).data
    parsed = parse_bytes(data)

    on_ticks = [e.tick for e in parsed.channel_events() if e.kind == "note_on"]
    assert len(set(on_ticks)) > 1, "every note shares a timestamp"
    assert max(on_ticks) > 0


# --- lyrics as editable, individually timed events -----------------------


def test_each_lyric_word_is_its_own_event_at_its_own_time(demo):
    result = _export(demo)
    parsed = parse_bytes(result.data)

    lyric_ff = 0x05 if DEFAULT_DIALECT.lyric_meta == "lyrics" else 0x01
    words = [e for e in parsed.meta_events(lyric_ff) if e.text not in ("\r", "\n")]

    assert len(words) == 23
    # Each word is separately timed and increases monotonically.
    assert [w.tick for w in words] == sorted(w.tick for w in words)
    gaps = {b.tick - a.tick for a, b in zip(words, words[1:])}
    assert gaps == {seconds_to_ticks(SPACING)}


def test_lyric_offset_shifts_words_and_leaves_notes_alone(demo):
    base = _export(demo)
    shifted = _export(demo, lyric_offset=0.25)

    base_parsed = parse_bytes(base.data)
    shifted_parsed = parse_bytes(shifted.data)
    ff = 0x05 if DEFAULT_DIALECT.lyric_meta == "lyrics" else 0x01

    def words(parsed):
        return [e.tick for e in parsed.meta_events(ff) if e.text not in ("\r", "\n")]

    def notes(parsed):
        return [e.tick for e in parsed.channel_events() if e.kind == "note_on"]

    assert words(shifted_parsed) == [t + seconds_to_ticks(0.25) for t in words(base_parsed)]
    assert notes(shifted_parsed) == notes(base_parsed)


def test_word_times_can_be_edited_independently():
    """Milestone 2 will edit these times; prove the model supports it."""
    from dataclasses import replace

    demo = build_demo_song()
    # Shift the final line; order is preserved, so the export stays valid.
    last = demo.lyric_lines[-1]
    shifted = [replace(syl, start=syl.start + 0.5, end=syl.end + 0.5) for syl in last]
    edited_lines = [*demo.lyric_lines[:-1], shifted]

    result = export_song_to_bytes(
        title=demo.title,
        artist=demo.artist,
        lyric_lines=edited_lines,
        note_tracks=demo.note_tracks,
        bpm=demo.tempo_bpm,
    )
    assert result.ok, result.summary

    parsed = parse_bytes(result.data)
    got = [e.tick for e in parsed.meta_events(0x05) if e.text not in ("\r", "\n")]
    base = parse_bytes(_export(demo).data)
    base_ticks = [e.tick for e in base.meta_events(0x05) if e.text not in ("\r", "\n")]

    assert got[-len(last) :] == [t + seconds_to_ticks(0.5) for t in base_ticks[-len(last) :]]


def test_out_of_order_lyric_times_are_refused_with_a_clear_message():
    """MIDI deltas cannot be negative, so bad timings must not be scrambled silently."""
    from dataclasses import replace

    demo = build_demo_song()
    first_line = demo.lyric_lines[0]
    # Push one word past the start of the next line's first word.
    broken = [replace(first_line[-1], start=first_line[-1].start + 5.0, end=first_line[-1].end + 5.0)]
    edited_lines = [[*first_line[:-1], *broken], *demo.lyric_lines[1:]]

    result = export_song_to_bytes(
        title=demo.title,
        artist=demo.artist,
        lyric_lines=edited_lines,
        note_tracks=demo.note_tracks,
        bpm=demo.tempo_bpm,
    )

    assert not result.ok
    assert any("words must stay in order" in p for p in result.problems)
    # The actionable cause must lead the message, not be buried under symptoms.
    assert "words must stay in order" in result.problems[0]


# --- validation catches bad input ----------------------------------------


def test_validation_reports_tampered_bytes():
    demo = build_demo_song()
    midi = build_midi(
        title=demo.title,
        artist=demo.artist,
        lines=demo.lyric_lines,
        note_tracks=demo.note_tracks,
        dialect=DEFAULT_DIALECT,
    )

    buffer = io.BytesIO()
    midi.save(file=buffer)
    data = bytearray(buffer.getvalue())

    # Corrupt the SMF format field from 1 to 7.
    data[9] = 7

    problems = validate_export(
        bytes(data),
        dialect=DEFAULT_DIALECT,
        lyric_lines=demo.lyric_lines,
        note_tracks=demo.note_tracks,
    )
    assert problems
    assert any("format" in p for p in problems)


def test_non_midi_bytes_are_rejected():
    with pytest.raises(MidiParseError):
        parse_bytes(b"definitely not a midi file")


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Song / Name: *?", "Song Name"),
        ("  spaced   out  ", "spaced out"),
        ("", "song"),
        ("???", "song"),
    ],
)
def test_safe_filename(raw, expected):
    assert safe_filename(raw) == expected