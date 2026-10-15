"""The editable MIDI note table.

This is a MANUAL tool. It does not derive an arrangement from audio - audio to
MIDI transcription arrives in a later milestone. These rows are notes you type
in yourself, as a fallback for when transcription is unavailable or wrong.

Exporting with no notes at all is allowed and produces a lyrics-only karaoke
file; that is a normal result, not an error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .midi_build import NoteEvent, NoteTrack

MIN_PITCH, MAX_PITCH = 0, 127
MIN_VELOCITY, MAX_VELOCITY = 1, 127
MIN_CHANNEL, MAX_CHANNEL = 0, 15

DEFAULT_PITCH = 60  # middle C
DEFAULT_DURATION = 0.5


@dataclass
class NoteRow:
    """One editable note."""

    channel: int = 0
    pitch: int = DEFAULT_PITCH
    start: float = 0.0
    duration: float = DEFAULT_DURATION
    velocity: int = 80


def empty_note_rows() -> list[NoteRow]:
    """No notes. Exporting this still works and yields a lyrics-only file."""
    return []


def example_note_rows(count: int = 4, *, start: float = 0.0, spacing: float = 0.5) -> list[NoteRow]:
    """A few example rows so the table is not bafflingly empty.

    These are PLACEHOLDERS to edit or delete. They are not a transcription and
    they are not derived from the audio.
    """
    if count <= 0:
        return []

    return [
        NoteRow(channel=0, pitch=DEFAULT_PITCH, start=round(start + i * spacing, 3), duration=spacing)
        for i in range(count)
    ]


def validate_note_rows(rows: list[NoteRow]) -> list[str]:
    """Return a list of problems. Empty list means the notes are usable.

    An empty list of notes is NOT a problem.
    """
    problems: list[str] = []

    for position, note in enumerate(rows, start=1):
        label = f"Row {position}"

        pitch = _check_int(note.pitch, label, "pitch", MIN_PITCH, MAX_PITCH, problems)
        velocity = _check_int(note.velocity, label, "velocity", MIN_VELOCITY, MAX_VELOCITY, problems)
        channel = _check_int(note.channel, label, "channel", MIN_CHANNEL, MAX_CHANNEL, problems)

        start = _check_float(note.start, label, "start", problems, allow_zero=True)
        duration = _check_float(note.duration, label, "duration", problems, allow_zero=False)

        if start is not None and start < 0:
            problems.append(f"{label}: 'start' is {note.start}. Times start at 0, so it cannot be negative.")

        if duration is not None and duration <= 0:
            problems.append(
                f"{label}: 'duration' is {note.duration}. A note must last longer than 0 seconds."
            )

        _ = (pitch, velocity, channel)  # validated above; values used by caller

    return problems


def _check_int(value, label: str, name: str, low: int, high: int, problems: list[str]) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        problems.append(f"{label}: '{name}' must be a whole number, but it is {value!r}.")
        return None
    if isinstance(value, float) and not float(value).is_integer():
        problems.append(f"{label}: '{name}' must be a whole number, but it is {value}.")
        return None
    if not low <= int(value) <= high:
        problems.append(f"{label}: '{name}' is {value}. It must be between {low} and {high}.")
        return None
    return int(value)


def _check_float(value, label: str, name: str, problems: list[str], *, allow_zero: bool) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        problems.append(f"{label}: '{name}' must be a number, but it is {value!r}.")
        return None
    if not math.isfinite(float(value)):
        problems.append(f"{label}: '{name}' must be a real number, not {value!r}.")
        return None
    return float(value)


def rows_to_note_tracks(rows: list[NoteRow], *, tempo_bpm: int = 120, ppq: int = 480) -> list[NoteTrack]:
    """Group validated note rows into one NoteTrack per MIDI channel.

    Channel 0 is conventionally the vocal melody, which is what the SongHub
    reference files use, so it becomes the melody track the lyrics sit in.
    """
    by_channel: dict[int, list[NoteEvent]] = {}
    for note in rows:
        by_channel.setdefault(int(note.channel), []).append(
            NoteEvent(
                start=float(note.start),
                duration=float(note.duration),
                pitch=int(note.pitch),
                velocity=int(note.velocity),
                channel=int(note.channel),
            )
        )

    tracks: list[NoteTrack] = []
    for channel in sorted(by_channel):
        notes = sorted(by_channel[channel], key=lambda n: (n.start, n.pitch))
        tracks.append(
            NoteTrack(
                name="melody" if channel == 0 else f"ch{channel}",
                notes=notes,
                program=0 if channel == 0 else None,
                channel_prefix=True,
            )
        )
    return tracks


@dataclass
class NoteTable:
    """Convenience wrapper for session state."""

    rows: list[NoteRow] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not validate_note_rows(self.rows)

    @property
    def problems(self) -> list[str]:
        return validate_note_rows(self.rows)

    @property
    def is_empty(self) -> bool:
        return not self.rows