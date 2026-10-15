"""Milestone 1b: diagnose why SongHub played only one note.

Run:  .venv\\Scripts\\python.exe -m app.make_note_samples

SongHub showed lyrics with correct in-time highlighting, but only ONE note was
audible. Rather than guess, this generates four files that each vary exactly one
variable against a common baseline:

  n1_program_fmt1    melody ch0, FF 08 program change + FF 20 prefix, format 1
  n2_noprogram_fmt1  melody ch0, NO program change, NO prefix,        format 1
  n3_program_fmt0    melody ch0, FF 08 + FF 20,                      format 0
  n4_twocahnnel      melody ch0 + bass ch1, both with FF 08,         format 1

Comparing n1 against n2/n3/n4 isolates: program change, track format, and
multi-channel layout.

Melody is "Ah vous dirai-je, Maman" (Twinkle Twinkle), traditional/public
domain, chosen because an unmistakable tune makes "did the melody play?" easy to
answer by ear.
"""

from __future__ import annotations

from pathlib import Path

from .karaoke_dialects import DIALECTS
from .lyrics import all_syllables, parse_lyrics
from .midi_build import BPM, PPQ, NoteEvent, NoteTrack, build_midi, save_midi, seconds_to_ticks
from .validate_midi import check_midi, parse_file

DIALECT = DIALECTS["A_fmt1_lyrics_ff05"]  # confirmed working in SongHub

TITLE = "Note Diagnostic"
ARTIST = "Local Test"
SPACING = 0.45
VELOCITY = 100

# 23 notes, matching the 23 lyric tokens one-for-one.
MELODY = [
    60, 60, 67, 67, 69, 69, 67,  # C C G G A A G
    65, 65, 64, 64, 62, 62, 60,  # F F E E D D C
    67, 67, 65, 65, 64, 64, 62,  # G G F F E E D
    67, 67,                      # G G
]

LYRIC_TEXT = (
    "twinkle twinkle little star how I wonder what you are "
    "up above all the world so high like a diamond in the sky"
)


def melody_track(*, program: int | None, channel_prefix: bool) -> NoteTrack:
    # Duration == spacing so notes are legato with no gaps between them.
    return NoteTrack(
        name="melody",
        program=program,
        channel_prefix=channel_prefix,
        notes=[
            NoteEvent(
                start=i * SPACING,
                duration=SPACING,
                pitch=pitch,
                velocity=VELOCITY,
                channel=0,
            )
            for i, pitch in enumerate(MELODY)
        ],
    )


def bass_track() -> NoteTrack:
    """One long bass note at the start of each 7-note phrase."""
    phrase = 7 * SPACING
    roots = [36, 41, 36, 36]  # C, F, C, C
    return NoteTrack(
        name="bass",
        program=32,  # acoustic bass
        channel_prefix=True,
        notes=[
            NoteEvent(start=i * phrase, duration=phrase * 0.95, pitch=p, velocity=90, channel=1)
            for i, p in enumerate(roots)
        ],
    )


VARIANTS = {
    "n1_program_fmt1": {
        "dialect": DIALECT,
        "tracks": lambda: [melody_track(program=0, channel_prefix=True)],
        "tests": "program change + channel prefix, single channel",
    },
    "n2_noprogram_fmt1": {
        "dialect": DIALECT,
        "tracks": lambda: [melody_track(program=None, channel_prefix=False)],
        "tests": "identical but NO program change and NO prefix",
    },
    "n3_program_fmt0": {
        "dialect": DIALECTS["B_fmt0_lyrics_ff05"],
        "tracks": lambda: [melody_track(program=0, channel_prefix=True)],
        "tests": "identical notes but merged into a single format 0 track",
    },
    "n4_twocahnnel_fmt1": {
        "dialect": DIALECT,
        "tracks": lambda: [melody_track(program=0, channel_prefix=True), bass_track()],
        "tests": "adds a second channel with bass (multi-channel layout)",
    },
}


def build_variant(spec: dict):
    lines = parse_lyrics(LYRIC_TEXT, start=0.0, unit_seconds=SPACING, line_gap=0.3)
    tracks = spec["tracks"]()
    midi = build_midi(
        title=TITLE,
        artist=ARTIST,
        lines=lines,
        note_tracks=tracks,
        dialect=spec["dialect"],
        bpm=BPM,
        ppq=PPQ,
    )
    return midi, lines, tracks


def validate_variant(path: Path, spec: dict, lines, tracks) -> list[str]:
    parsed = parse_file(path)
    syllables = all_syllables(lines)
    return check_midi(
        parsed,
        expected_format=spec["dialect"].smf_format,
        expected_units=[s.text for s in syllables],
        expected_unit_ticks=[seconds_to_ticks(s.start) for s in syllables],
        expected_notes=sum(len(t.notes) for t in tracks),
        expected_note_ticks=[seconds_to_ticks(n.start) for t in tracks for n in t.notes],
        lyric_meta=spec["dialect"].lyric_meta,
        expect_word_separator=spec["dialect"].word_separator is not None,
        expect_line_separator=spec["dialect"].line_separator is not None,
    )


def main() -> int:
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)

    print("=" * 74)
    print("Milestone 1b - note playback diagnostic (SongHub heard 1 note only)")
    print("=" * 74)
    print(f"Melody : {len(MELODY)} notes over {len(MELODY) * SPACING:.1f}s, legato, velocity {VELOCITY}")
    print("Tune   : 'Twinkle Twinkle' (traditional, public domain)")
    print()

    failures = 0
    for name, spec in VARIANTS.items():
        midi, lines, tracks = build_variant(spec)
        path = out_dir / f"{name}.mid"
        save_midi(midi, path)
        problems = validate_variant(path, spec, lines, tracks)

        print(f"[{'PASS' if not problems else 'FAIL'}] {path.name} ({path.stat().st_size} bytes)")
        print(f"         {spec['tests']}")
        for problem in problems:
            print(f"         - {problem}")
        print()

        if problems:
            failures += 1

    if failures:
        print(f"{failures} variant(s) failed structural validation.")
        return 1

    print("All variants passed raw-byte structural validation.")
    print()
    print("-" * 74)
    print("Test in SongHub, in this order. For each file note TWO things:")
    print("-" * 74)
    print("  1. Does the melody play through (all 23 notes), or stop early?")
    print("  2. Do the lyrics keep highlighting for the full ~10s, or stop early?")
    print()
    print("  n1 = baseline        ->  if this one fails, stop and report n1")
    print("  n2 vs n1             ->  does the program change matter?")
    print("  n3 vs n1             ->  does the track format matter?")
    print("  n4 vs n1             ->  does the second channel matter?")
    print()
    print("Question 2 matters as much as 1: if the lyrics stop highlighting early,")
    print("SongHub is likely ending playback at the end of the first track.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())