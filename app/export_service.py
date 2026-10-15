"""MIDI export service.

One entry point that turns a song into validated `.mid` bytes. The Streamlit
page and the tests both go through here, so there is a single definition of
"exported correctly".

Every export is validated by re-parsing the bytes it just produced with the
independent parser in `validate_midi`. Re-reading with mido would not be a real
check: writer and reader would agree on the same wrong bytes.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

from .karaoke_dialects import DEFAULT_DIALECT, Dialect
from .lyrics import Syllable, all_syllables
from .midi_build import NoteTrack, build_midi, seconds_to_ticks
from .validate_midi import check_midi, parse_bytes

_UNSAFE = re.compile(r"[^A-Za-z0-9 _-]")


def safe_filename(text: str, fallback: str = "song") -> str:
    """Song title -> filesystem-safe name. SongHub reads the name, so keep it plain."""
    cleaned = _UNSAFE.sub("", text or "").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned or fallback


@dataclass
class ExportResult:
    data: bytes
    filename: str
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    @property
    def summary(self) -> str:
        if self.ok:
            return f"OK - {len(self.data)} bytes, structure verified"
        return f"{len(self.problems)} problem(s): " + "; ".join(self.problems)


def _lyric_order_problems(syllables: list[Syllable]) -> list[str]:
    """Flag lyric timings that would have to be reordered to be written.

    MIDI delta times can never be negative, so events must be emitted in
    non-decreasing time. If an edit puts a word before the word before it, the
    only ways out are reordering (which scrambles the text) or refusing. This
    refuses, and says exactly which pair is wrong, because a silently scrambled
    lyric line is far harder to notice than an error message.
    """
    problems = []
    for previous, current in zip(syllables, syllables[1:]):
        if current.start < previous.start - 1e-9:
            problems.append(
                f"lyric '{current.text}' starts at {current.start:.2f}s, before "
                f"'{previous.text}' at {previous.start:.2f}s - words must stay in order"
            )
    return problems


def export_song_to_bytes(
    *,
    title: str,
    artist: str,
    lyric_lines: list[list[Syllable]],
    note_tracks: list[NoteTrack],
    dialect: Dialect = DEFAULT_DIALECT,
    bpm: int = 120,
    ppq: int = 480,
    lyric_offset: float = 0.0,
) -> ExportResult:
    """Build, validate and serialise a karaoke MIDI file.

    `lyric_offset` shifts lyric events only; notes are never moved. Positive
    values delay the highlight, negative values advance it.
    """
    midi = build_midi(
        title=title,
        artist=artist,
        lines=lyric_lines,
        note_tracks=note_tracks,
        dialect=dialect,
        bpm=bpm,
        ppq=ppq,
        lyric_offset=lyric_offset,
    )

    buffer = io.BytesIO()
    midi.save(file=buffer)
    data = buffer.getvalue()

    problems = validate_export(
        data,
        dialect=dialect,
        lyric_lines=lyric_lines,
        note_tracks=note_tracks,
        lyric_offset=lyric_offset,
        bpm=bpm,
        ppq=ppq,
    )

    # Order problems come first: they are the actionable cause, and the
    # structural mismatch further down is only a symptom of them.
    order_problems = _lyric_order_problems(all_syllables(lyric_lines))
    if order_problems:
        problems = order_problems + [p for p in problems if "lyric" not in p]

    name = f"{safe_filename(title)} - {safe_filename(artist, 'Unknown')}".strip()
    return ExportResult(data=data, filename=f"{name}.mid", problems=problems)


def validate_export(
    data: bytes,
    *,
    dialect: Dialect,
    lyric_lines: list[list[Syllable]],
    note_tracks: list[NoteTrack],
    lyric_offset: float = 0.0,
    bpm: int = 120,
    ppq: int = 480,
) -> list[str]:
    """Re-parse exported bytes and check structure. Empty list means it passed."""
    parsed = parse_bytes(data)
    syllables = all_syllables(lyric_lines)

    def to_ticks(seconds: float) -> int:
        return seconds_to_ticks(max(0.0, seconds), ppq=ppq, bpm=bpm)

    return check_midi(
        parsed,
        expected_format=dialect.smf_format,
        expected_units=[syl.text for syl in syllables],
        expected_unit_ticks=[to_ticks(syl.start + lyric_offset) for syl in syllables],
        expected_notes=sum(len(track.notes) for track in note_tracks),
        expected_note_ticks=[
            to_ticks(note.start) for track in note_tracks for note in track.notes
        ],
        lyric_meta=dialect.lyric_meta,
        expect_word_separator=dialect.word_separator is not None,
        expect_line_separator=dialect.line_separator is not None,
    )