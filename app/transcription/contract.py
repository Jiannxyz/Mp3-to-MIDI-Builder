"""The documented note-estimate contract.

This is the boundary between the model and everything else. `worker.py` turns
Basic Pitch output into this shape; the rest of the app only ever sees it, so
the model can be swapped without touching the UI or the exporter.

Named `contract.py`, NOT `types.py`: this file sits next to `worker.py`, and
running a script puts its own directory first on `sys.path`. A module named
`types.py` here would shadow the standard library's `types` module and break
every stdlib import inside the worker before Basic Pitch is ever reached.
`test_m3_worker.py` guards against that class of mistake.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

# Error kinds, so the UI can say something useful instead of "it failed".
KIND_DEPENDENCY = "dependency"
KIND_CONVERSION = "conversion"
KIND_MODEL = "model"
KIND_RUNTIME = "runtime"
KIND_TIMEOUT = "timeout"

ERROR_HINTS = {
    KIND_DEPENDENCY: (
        "The AI extras are not installed. They are optional on purpose so the "
        "rest of the app stays fast. To enable note estimation, follow the "
        "'Optional AI extras' section of README.md."
    ),
    KIND_CONVERSION: (
        "The audio file could not be converted to WAV. Installing FFmpeg and "
        "putting it on PATH fixes this."
    ),
    KIND_MODEL: "The model failed to run. The first run downloads the model, so it needs a connection.",
    KIND_RUNTIME: "Note estimation failed unexpectedly.",
    KIND_TIMEOUT: (
        "Note estimation was stopped because it ran too long. Basic Pitch normally "
        "needs no model download, so if this happens on a first run something else "
        "is wrong - try a shorter piece of audio."
    ),
    "stalled-download": (
        "A model download stalled. Basic Pitch normally needs no download, so check "
        "your setup - or skip it and enter notes by hand in Step 5."
    ),
}


class TranscriptionError(Exception):
    """A note-estimation failure the UI can explain."""

    def __init__(self, message: str, kind: str = KIND_RUNTIME, detail: str = ""):
        super().__init__(message)
        self.kind = kind
        self.detail = detail

    @property
    def hint(self) -> str:
        return ERROR_HINTS.get(self.kind, "")

    def as_dict(self) -> dict:
        return {"kind": self.kind, "message": str(self), "detail": self.detail, "hint": self.hint}


@dataclass(frozen=True)
class NoteEstimate:
    """One estimated note.

    `start` and `end` are SECONDS from the start of the audio, matching
    `app.timing` and `app.notes` so estimates drop straight into the tables.
    `amplitude` is Basic Pitch's confidence/velocity proxy in roughly 0..1.
    `label` is the instrument name Basic Pitch guessed ("piano", "guitar", ...),
    empty when it did not report one.
    """

    start: float
    end: float
    pitch: int
    amplitude: float = 1.0
    label: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> "NoteEstimate":
        return cls(
            start=float(raw["start"]),
            end=float(raw["end"]),
            pitch=int(raw["pitch"]),
            amplitude=float(raw.get("amplitude", 1.0)),
            label=str(raw.get("label", "")),
        )

    @property
    def is_valid(self) -> bool:
        return (
            math.isfinite(self.start)
            and math.isfinite(self.end)
            and self.start >= 0
            and self.end > self.start
            and 0 <= self.pitch <= 127
        )


@dataclass
class TranscriptionResult:
    """Everything one estimation run produced."""

    estimates: list[NoteEstimate] = field(default_factory=list)
    model: str = ""
    sample_rate: int = 0
    audio_duration: float = 0.0
    backend: str = ""
    warnings: list[str] = field(default_factory=list)
    dropped: int = 0

    def __len__(self) -> int:
        return len(self.estimates)

    @property
    def last_end(self) -> float:
        return max((note.end for note in self.estimates), default=0.0)

    @property
    def summary(self) -> str:
        if not self.estimates:
            return "No notes were detected."
        parts = [f"{len(self.estimates)} notes"]
        groups = group_estimates(self.estimates)
        if len(groups) > 1:
            parts.append(f"{len(groups)} instrument groups")
        if self.dropped:
            parts.append(f"{self.dropped} invalid dropped")
        return ", ".join(parts) + "."

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "sample_rate": self.sample_rate,
            "audio_duration": self.audio_duration,
            "backend": self.backend,
            "warnings": list(self.warnings),
            "dropped": self.dropped,
            "notes": [note.to_dict() for note in self.estimates],
        }

    @classmethod
    def from_dict(cls, raw: dict, *, backend: str = "") -> "TranscriptionResult":
        estimates = [NoteEstimate.from_dict(item) for item in raw.get("notes", [])]
        kept: list[NoteEstimate] = []
        dropped = 0
        for estimate in estimates:
            if estimate.is_valid:
                kept.append(estimate)
            else:
                dropped += 1

        kept.sort(key=lambda note: (note.start, note.pitch))

        return cls(
            estimates=kept,
            model=str(raw.get("model", "")),
            sample_rate=int(raw.get("sample_rate", 0)),
            audio_duration=float(raw.get("audio_duration", 0.0)),
            backend=backend or str(raw.get("backend", "")),
            warnings=list(raw.get("warnings", [])),
            dropped=dropped,
        )


# Rough pitch bands, useful for picking out a melody when Basic Pitch reports
# no instrument labels (which is the usual case in basic_pitch 0.4.0).
PITCH_BANDS = (
    ("bass", 0, 51),
    ("mid", 52, 71),
    ("treble", 72, 127),
)


def pitch_band(pitch: int) -> str:
    for name, low, high in PITCH_BANDS:
        if low <= pitch <= high:
            return name
    return "unlabelled"


def group_estimates(estimates: list[NoteEstimate], by: str = "label") -> dict[str, list[NoteEstimate]]:
    """Group notes so whole groups can be selected for export.

    `by="label"` uses the instrument name Basic Pitch reported. Since 0.4.0
    reports none, this normally yields a single "unlabelled" group, so
    `by="pitch"` is also offered: it splits into bass/mid/treble, which is a
    practical way to keep a melody and drop accompaniment.
    """
    if by == "pitch":
        groups: dict[str, list[NoteEstimate]] = {}
        for estimate in estimates:
            groups.setdefault(pitch_band(estimate.pitch), []).append(estimate)
        return dict(sorted(groups.items(), key=lambda item: item[0]))

    if by != "label":
        raise ValueError(f"unknown grouping: {by!r} (use 'label' or 'pitch')")

    groups = {}
    for estimate in estimates:
        groups.setdefault(estimate.label or "unlabelled", []).append(estimate)
    return dict(sorted(groups.items(), key=lambda item: item[0]))


def estimates_from_rows(rows) -> list[NoteEstimate]:
    """Turn selected table rows into estimates, for re-export or inspection."""
    result: list[NoteEstimate] = []
    for row in rows:
        try:
            result.append(
                NoteEstimate(
                    start=float(row.start),
                    end=float(row.start) + float(row.duration),
                    pitch=int(row.pitch),
                    amplitude=float(getattr(row, "amplitude", 100)) / 100.0,
                    label=str(getattr(row, "label", "")),
                )
            )
        except (TypeError, ValueError, AttributeError):
            continue
    return result