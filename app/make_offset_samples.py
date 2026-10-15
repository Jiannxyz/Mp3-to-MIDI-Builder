"""Milestone 1c: calibrate the lyric offset.

Run:  .venv\\Scripts\\python.exe -m app.make_offset_samples

The melody now plays correctly, but SongHub highlights the text BEFORE the word
is sung. Since words and notes share identical timestamps at offset 0, any lead
the player shows is its own; this generates files at several offsets so the
right value can be measured by ear instead of guessed.

Offsets are applied to LYRIC EVENTS ONLY. Notes never move, so a larger offset
always means "highlight later".

The lyrics here are broken into four musical lines. `line_gap` is 0 on purpose:
word i must sit at i * SPACING so it lines up with melody note i. A line gap
would silently shift every word after the first line break.
"""

from __future__ import annotations

from pathlib import Path

from .karaoke_dialects import DIALECTS
from .lyrics import all_syllables, parse_lyrics
from .make_note_samples import MELODY, SPACING, VELOCITY
from .midi_build import BPM, PPQ, NoteEvent, NoteTrack, build_midi, save_midi, seconds_to_ticks
from .validate_midi import check_midi, parse_file

DIALECT = DIALECTS["A_fmt1_lyrics_ff05"]

TITLE = "Lyric Offset Test"
ARTIST = "Local Test"

# Four phrases, 23 words total, one word per melody note.
LYRIC_TEXT = """twinkle twinkle little star
how I wonder what you are
up above all the world
so high like a diamond in the sky"""

OFFSETS = [0.0, 0.20, 0.40, 0.60]


def melody_track() -> NoteTrack:
    return NoteTrack(
        name="melody",
        program=0,
        channel_prefix=True,
        notes=[
            NoteEvent(start=i * SPACING, duration=SPACING, pitch=p, velocity=VELOCITY, channel=0)
            for i, p in enumerate(MELODY)
        ],
    )


def build(offset: float):
    # line_gap=0 keeps word i aligned with melody note i across line breaks.
    lines = parse_lyrics(LYRIC_TEXT, start=0.0, unit_seconds=SPACING, line_gap=0.0)
    tracks = [melody_track()]
    midi = build_midi(
        title=TITLE,
        artist=ARTIST,
        lines=lines,
        note_tracks=tracks,
        dialect=DIALECT,
        bpm=BPM,
        ppq=PPQ,
        lyric_offset=offset,
    )
    return midi, lines, tracks


def validate(path: Path, offset: float, lines, tracks) -> list[str]:
    parsed = parse_file(path)
    syllables = all_syllables(lines)
    return check_midi(
        parsed,
        expected_format=DIALECT.smf_format,
        expected_units=[s.text for s in syllables],
        expected_unit_ticks=[seconds_to_ticks(s.start + offset) for s in syllables],
        expected_notes=sum(len(t.notes) for t in tracks),
        expected_note_ticks=[seconds_to_ticks(n.start) for t in tracks for n in t.notes],
        lyric_meta=DIALECT.lyric_meta,
        expect_word_separator=DIALECT.word_separator is not None,
        expect_line_separator=DIALECT.line_separator is not None,
    )


def main() -> int:
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)

    print("=" * 74)
    print("Milestone 1c - lyric offset calibration (SongHub highlights too early)")
    print("=" * 74)
    print(f"Melody : {len(MELODY)} notes over {len(MELODY) * SPACING:.1f}s")
    print(f"Lyrics : {len(LYRIC_TEXT.split())} words in 4 lines, one word per note")
    print()

    failures = 0
    for offset in OFFSETS:
        midi, lines, tracks = build(offset)
        name = f"o_offset_{int(offset * 1000):+04d}ms.mid".replace("+", "plus")
        path = out_dir / name
        save_midi(midi, path)
        problems = validate(path, offset, lines, tracks)

        print(f"[{'PASS' if not problems else 'FAIL'}] {path.name} (lyric offset {offset:+.2f}s)")
        for problem in problems:
            print(f"         - {problem}")
        if problems:
            failures += 1

    print()
    if failures:
        print(f"{failures} sample(s) failed structural validation.")
        return 1

    print("All samples passed raw-byte structural validation.")
    print()
    print("-" * 74)
    print("Test in SongHub. Notes are identical in every file; only the text moves.")
    print("-" * 74)
    print("  o_offset_0000ms  = control, words sit exactly on their notes")
    print("  o_offset_0200ms  = text 200ms later")
    print("  o_offset_0400ms  = text 400ms later")
    print("  o_offset_0600ms  = text 600ms later")
    print()
    print("Pick the one where each word highlights as you actually SING it.")
    print()
    print("ALSO watch one thing: is the lead CONSTANT, or does the text fall")
    print("progressively further behind as the song goes on?")
    print("  constant lead   -> an offset value fixes it")
    print("  growing lead    -> SongHub is ignoring our tempo map, a different fix")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())