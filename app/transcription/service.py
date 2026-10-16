"""Facade the Streamlit UI talks to.

Everything model-specific lives behind `service.estimate_notes`, so replacing
Basic Pitch later means changing this file and `worker.py` only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import audio_convert, bridge
from .contract import TranscriptionError, TranscriptionResult

# Defaults are Basic Pitch's own defaults; exposed so the UI can tune them.
DEFAULT_ONSET_THRESHOLD = 0.5
DEFAULT_FRAME_THRESHOLD = 0.3
DEFAULT_MINIMUM_NOTE_MS = 127.7


@dataclass(frozen=True)
class Availability:
    """Whether note estimation can run right now, and why not if it cannot."""

    available: bool
    backend: str
    interpreter: str | None
    reason: str = ""

    @property
    def backend_label(self) -> str:
        return {
            "side-venv": "separate AI environment (.venv-ai)",
            "in-process-venv": "this environment",
            "none": "not installed",
        }.get(self.backend, self.backend)


def check_availability(venv_path: Path = bridge.DEFAULT_AI_VENV) -> Availability:
    """Detect an installed backend without raising."""
    try:
        backend, interpreter = bridge.resolve_backend(venv_path)
    except OSError as error:
        return Availability(False, "none", None, f"Could not inspect the environment: {error}")

    if backend == "none":
        return Availability(
            False,
            "none",
            None,
            "The optional AI extras (basic-pitch) are not installed.",
        )
    return Availability(True, backend, str(interpreter) if interpreter else None)


def estimate_notes(
    source: Path,
    workdir: Path,
    *,
    onset_threshold: float = DEFAULT_ONSET_THRESHOLD,
    frame_threshold: float = DEFAULT_FRAME_THRESHOLD,
    minimum_note_length_ms: float = DEFAULT_MINIMUM_NOTE_MS,
    minimum_frequency: float | None = None,
    maximum_frequency: float | None = None,
    timeout: int = 900,
    venv_path: Path = bridge.DEFAULT_AI_VENV,
) -> TranscriptionResult:
    """Estimate notes for one audio file.

    Converts the input to WAV if needed, runs the worker in whichever
    environment has basic_pitch, and returns the documented result type.

    Raises TranscriptionError with a `kind` and a user-facing `hint`.
    """
    wav_path = audio_convert.ensure_wav(Path(source), Path(workdir))

    payload = bridge.run_worker(
        wav_path,
        venv_path=venv_path,
        timeout=timeout,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=minimum_note_length_ms,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
    )

    return TranscriptionResult.from_dict(payload)


def amplitude_to_velocity(amplitude: float) -> int:
    """Basic Pitch amplitude (about 0..1) -> MIDI velocity (1..127)."""
    clamped = min(1.0, max(0.0, float(amplitude)))
    return max(1, min(127, round(clamped * 127)))


def estimates_to_note_rows(
    estimates,
    *,
    channel: int = 0,
) -> list["NoteRow"]:
    """Turn selected estimates into note-table rows for the existing exporter.

    Rows are marked `source="estimated"` so the UI can keep them visibly
    distinct from hand-typed notes. Lyric rows are untouched by this.
    """
    from app.notes import NoteRow  # imported here to keep the dependency one-way

    rows: list[NoteRow] = []
    for estimate in estimates:
        rows.append(
            NoteRow(
                channel=channel,
                pitch=int(estimate.pitch),
                start=float(estimate.start),
                duration=max(0.01, float(estimate.end) - float(estimate.start)),
                velocity=amplitude_to_velocity(estimate.amplitude),
                source="estimated",
            )
        )
    return rows


def assign_groups(estimates, *, by: str = "label") -> list[tuple]:
    """Flatten estimates into (estimate, group_name) pairs, in time order.

    Used to build the review table so a whole group can be excluded at once.
    """
    from .contract import group_estimates

    items: list[tuple] = []
    for name, group in group_estimates(list(estimates), by=by).items():
        for estimate in sorted(group, key=lambda note: (note.start, note.pitch)):
            items.append((estimate, name))
    return items


__all__ = [
    "Availability",
    "TranscriptionError",
    "TranscriptionResult",
    "amplitude_to_velocity",
    "assign_groups",
    "check_availability",
    "estimate_notes",
    "estimates_to_note_rows",
]