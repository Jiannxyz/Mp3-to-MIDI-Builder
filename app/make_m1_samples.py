"""Milestone 1: generate the tiny dialect-calibration MIDI files.

Run:  .venv\\Scripts\\python.exe -m app.make_m1_samples

Produces one small .mid per dialect in out/, validates each one against the raw
bytes, and prints what to test in SongHub.

Lyrics here are deliberately neutral placeholder text. Use only songs and
lyrics you are authorized to use.
"""

from __future__ import annotations

from pathlib import Path

from .karaoke_dialects import DIALECTS, Dialect
from .lyrics import Syllable, all_syllables, parse_lyrics
from .midi_build import BPM, PPQ, NoteEvent, NoteTrack, build_midi, save_midi, seconds_to_ticks
from .validate_midi import check_midi, parse_file

TITLE = "Milestone 1 Dialect Test"
ARTIST = "Local Test"
LYRIC_TEXT = """hello karaoke test line
this checks lyric timing"""

START = 1.0  # leave a second of silence before the first lyric
UNIT = 0.45


def lead_track() -> NoteTrack:
    # A short original C-major phrase; nothing here is a copyrighted melody.
    pitches = [60, 62, 64, 65, 67, 65, 64, 62]
    return NoteTrack(
        name="lead",
        notes=[
            NoteEvent(start=START + i * UNIT, duration=UNIT * 0.9, pitch=p, velocity=88, channel=0)
            for i, p in enumerate(pitches)
        ],
    )


def bass_track() -> NoteTrack:
    bass = [36, 36, 41, 41]
    return NoteTrack(
        name="bass",
        notes=[
            NoteEvent(start=START + i * UNIT * 4, duration=UNIT * 3.5, pitch=p, velocity=72, channel=1)
            for i, p in enumerate(bass)
        ],
    )


def build_for_dialect(dialect: Dialect):
    lines = parse_lyrics(LYRIC_TEXT, start=START, unit_seconds=UNIT)
    tracks = [lead_track(), bass_track()]
    midi = build_midi(
        title=TITLE,
        artist=ARTIST,
        lines=lines,
        note_tracks=tracks,
        dialect=dialect,
        bpm=BPM,
        ppq=PPQ,
    )
    return midi, lines, tracks


def validate_for_dialect(path: Path, dialect: Dialect, lines, tracks) -> list[str]:
    parsed = parse_file(path)
    syllables = all_syllables(lines)
    expected_notes = sum(len(track.notes) for track in tracks)
    expected_note_ticks = [
        seconds_to_ticks(note.start) for track in tracks for note in track.notes
    ]

    return check_midi(
        parsed,
        expected_format=dialect.smf_format,
        expected_units=[syl.text for syl in syllables],
        expected_unit_ticks=[seconds_to_ticks(syl.start) for syl in syllables],
        expected_notes=expected_notes,
        expected_note_ticks=expected_note_ticks,
        lyric_meta=dialect.lyric_meta,
        expect_word_separator=dialect.word_separator is not None,
        expect_line_separator=dialect.line_separator is not None,
    )


def main() -> int:
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)

    print("=" * 72)
    print("Milestone 1 - karaoke MIDI dialect calibration samples")
    print("=" * 72)
    print(f"Lyrics : {LYRIC_TEXT!r}")
    print(f"Tempo  : {BPM} BPM, {PPQ} PPQ, first event at {START}s")
    print()

    failures = 0

    for key, dialect in DIALECTS.items():
        midi, lines, tracks = build_for_dialect(dialect)
        path = out_dir / f"m1_{key}.mid"
        save_midi(midi, path)
        problems = validate_for_dialect(path, dialect, lines, tracks)

        status = "PASS" if not problems else f"FAIL ({len(problems)})"
        print(f"[{status}] {path.name}  ({path.stat().st_size} bytes)")
        print(f"         {dialect.description}")
        for problem in problems:
            print(f"         - {problem}")
        print()

        if problems:
            failures += 1

    if failures:
        print(f"{failures} sample(s) failed structural validation. Fix before testing in SongHub.")
        return 1

    print("All samples passed raw-byte structural validation.")
    print()
    print("-" * 72)
    print("Now test in SongHub Karaoke PC, in this order:")
    print("-" * 72)
    print("1. Copy the 4 files from out/ into your SongHub songs folder.")
    print("2. Play each one. For EACH file record:")
    print("     [ ] Does a title/artist show?")
    print("     [ ] Do lyrics appear on screen at all?")
    print("     [ ] Do the lyrics highlight in time with the music?")
    print("     [ ] Do you hear notes playing?")
    print("3. Report which variant (A/B/C/D) works best, and what fails for the others.")
    print()
    print("Until SongHub is tested, none of these are proven compatible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())