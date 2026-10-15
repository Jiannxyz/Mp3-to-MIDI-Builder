"""Milestone 1d: match the layout of a known-good karaoke file.

Run:  .venv\\Scripts\\python.exe -m app.make_structure_samples

Two known-good SongHub karaoke files were inspected. Both share one layout our
builder does NOT follow:

  track 0  conductor only - track name, tempo, time/key signature, marker.
           NO lyrics, NO notes.
  track 1  the vocal melody, channel 0, with lyric events INTERLEAVED into it.
           Each word is written at the SAME tick as its note_on, immediately
           after it, and the CR/LF separator is written at that same tick.
           Every track also carries FF 21 (midi_port).

In our files the lyrics sat in their own track with no notes. That is the most
likely cause of the highlight arriving early: SongHub may schedule a lyric
track and a melody track with a fixed skew, which cannot happen when both live
in one track.

Each variant below changes exactly ONE thing from `s1`, so one SongHub session
identifies the culprit.
"""

from __future__ import annotations

from pathlib import Path

from .demo import build_demo_song
from .karaoke_dialects import DEFAULT_DIALECT
from .midi_build import BPM, build_midi, save_midi, seconds_to_ticks
from .validate_midi import check_midi, parse_file

TITLE_SUFFIX = ""

VARIANTS = {
    "s1_reference_layout": {
        "lyrics_in_melody_track": True,
        "separator_timing": "same_tick",
        "emit_midi_port": True,
        "ppq": 480,
        "tests": "reference layout: lyrics in melody track, CR/LF at same tick",
    },
    "s2_ppq120": {
        "lyrics_in_melody_track": True,
        "separator_timing": "same_tick",
        "emit_midi_port": True,
        "ppq": 120,
        "tests": "same as s1 but PPQ 120, matching the reference files exactly",
    },
    "s3_separators_at_word_end": {
        "lyrics_in_melody_track": True,
        "separator_timing": "word_end",
        "emit_midi_port": True,
        "ppq": 480,
        "tests": "same as s1 but CR/LF at the word's END instead of its start",
    },
    "s4_separate_lyric_track": {
        "lyrics_in_melody_track": False,
        "separator_timing": "same_tick",
        "emit_midi_port": True,
        "ppq": 480,
        "tests": "same as s1 but lyrics back in their own track (the old layout)",
    },
}


def build_variant(spec: dict, ppq: int):
    song = build_demo_song()
    midi = build_midi(
        title=song.title,
        artist=song.artist,
        lines=song.lyric_lines,
        note_tracks=song.note_tracks,
        dialect=DEFAULT_DIALECT,
        bpm=song.tempo_bpm,
        ppq=ppq,
        lyrics_in_melody_track=spec["lyrics_in_melody_track"],
        separator_timing=spec["separator_timing"],
        emit_midi_port=spec["emit_midi_port"],
    )
    return midi, song


def validate_variant(path: Path, spec: dict, song, ppq: int) -> list[str]:
    parsed = parse_file(path)
    syllables = [syl for line in song.lyric_lines for syl in line]

    def to_ticks(seconds: float) -> int:
        return seconds_to_ticks(max(0.0, seconds), ppq=ppq, bpm=song.tempo_bpm)

    return check_midi(
        parsed,
        expected_format=DEFAULT_DIALECT.smf_format,
        expected_units=[s.text for s in syllables],
        expected_unit_ticks=[to_ticks(s.start) for s in syllables],
        expected_notes=sum(len(t.notes) for t in song.note_tracks),
        expected_note_ticks=[
            to_ticks(n.start) for t in song.note_tracks for n in t.notes
        ],
        lyric_meta=DEFAULT_DIALECT.lyric_meta,
        expect_word_separator=DEFAULT_DIALECT.word_separator is not None,
        expect_line_separator=DEFAULT_DIALECT.line_separator is not None,
    )


def main() -> int:
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)

    print("=" * 74)
    print("Milestone 1d - match the layout of known-good karaoke files")
    print("=" * 74)
    print("Reference files: lyrics interleaved into the melody track, track 0")
    print("carries no lyrics. Ours put lyrics in their own track.")
    print()

    failures = 0
    for name, spec in VARIANTS.items():
        ppq = spec["ppq"]
        midi, song = build_variant(spec, ppq)
        path = out_dir / f"{name}.mid"
        save_midi(midi, path)
        problems = validate_variant(path, spec, song, ppq)

        print(f"[{'PASS' if not problems else 'FAIL'}] {path.name} ({path.stat().st_size} bytes, PPQ {ppq})")
        print(f"         {spec['tests']}")
        for problem in problems:
            print(f"         - {problem}")
        if problems:
            failures += 1

    print()
    if failures:
        print(f"{failures} variant(s) failed structural validation.")
        return 1

    print("All variants passed raw-byte structural validation.")
    print()
    print("-" * 74)
    print("Test in SongHub. Test s1 FIRST - it is the candidate fix.")
    print("-" * 74)
    print("  s1  lyrics in melody track, CR/LF at same tick   <-- reference layout")
    print("  s2  s1 with PPQ 120                              -> does resolution matter?")
    print("  s3  s1 with CR/LF at word end                    -> does separator tick matter?")
    print("  s4  s1 with lyrics back in a separate track      -> confirms the track layout is it")
    print()
    print("For each: does the text still highlight EARLY, or now line up with the")
    print("singing? The winning variant should be judged by listening to the first")
    print("few words, not the whole song.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())