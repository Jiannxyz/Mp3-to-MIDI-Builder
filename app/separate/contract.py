"""The documented stem-separation contract.

Mirrors `app.transcription.contract`: this is the boundary between the model
and everything else, so Demucs can be replaced without touching the UI.

Named `contract.py`, not `types.py` — see `app/transcription/contract.py` for
why a module name that shadows the standard library would be fatal here.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field

KIND_DEPENDENCY = "dependency"
KIND_CONVERSION = "conversion"
KIND_MODEL = "model"
KIND_RUNTIME = "runtime"
KIND_TIMEOUT = "timeout"

ERROR_HINTS = {
    KIND_DEPENDENCY: (
        "Demucs is not installed. It is an optional extra - see 'Optional AI "
        "extras' in README.md. Everything else in the app keeps working."
    ),
    KIND_CONVERSION: (
        "The audio could not be read. Installing FFmpeg and putting it on PATH "
        "fixes most cases; Demucs falls back to torchaudio without it."
    ),
    KIND_MODEL: (
        "The separation model could not be loaded. First use downloads the model "
        "weights, so it needs a connection once."
    ),
    KIND_RUNTIME: "Stem separation failed unexpectedly.",
    KIND_TIMEOUT: (
        "This was stopped because it ran too long. If it had just started, it may "
        "have been stuck downloading the model rather than separating - the first "
        "run needs internet to fetch the weights."
    ),
    "stalled-download": (
        "The model weights could not be downloaded. Your network reaches the site "
        "but not the model files. Run this once on a network that allows the "
        "download, or skip separation - the rest of the app works without it."
    ),
}

# Demucs model -> stems it produces. htdemucs is the default; the *_6s models
# add piano and guitar.
KNOWN_MODELS = {
    "htdemucs": ["drums", "bass", "other", "vocals", "guitar", "piano"],
    "htdemucs_ft": ["drums", "bass", "other", "vocals", "guitar", "piano"],
    "htdemucs_6s": ["drums", "bass", "other", "vocals", "guitar", "piano"],
    "htdemucs_ft_6s": ["drums", "bass", "other", "vocals", "guitar", "piano"],
    "mdx_extra": ["drums", "bass", "other", "vocals"],
    "mdx_extra_q": ["no_vocals", "vocals"],
    "mdx": ["drums", "bass", "other", "vocals"],
    "mdx_extra_9s": ["drums", "bass", "other", "vocals", "guitar", "piano", "strings", "keys"],
}

DEFAULT_MODEL = "htdemucs"

# Order matters: this is the advice the UI gives for picking a stem to run
# Basic Pitch on. Vocals first because a sung melody is monophonic, which is
# what a note tracker handles best; then bass, also mostly single-line.
PREFERRED_STEM_ORDER = [
    "vocals",
    "no_vocals",
    "bass",
    "guitar",
    "piano",
    "keys",
    "strings",
    "other",
    "drums",
]

# Drums are percussive and transient, not pitched notes in the musical sense,
# so they are a poor note-tracking source.
UNSUITABLE_STEMS = {"drums"}

SEPARATION_CAVEAT = (
    "Separated stems are NOT clean isolated instruments. Each one still contains "
    "bleed from the others, so a note estimate from a stem will be approximate."
)


class SeparationError(Exception):
    """A separation failure the UI can explain."""

    def __init__(self, message: str, kind: str = KIND_RUNTIME, detail: str = ""):
        super().__init__(message)
        self.kind = kind
        self.detail = detail

    @property
    def hint(self) -> str:
        return ERROR_HINTS.get(self.kind, "")


@dataclass(frozen=True)
class StemInfo:
    """One separated stem written to disk.

    `path` is always inside the caller's output directory; the worker's input
    file is never written to or removed.
    """

    name: str
    path: str
    duration: float = 0.0

    @property
    def is_suitable_for_notes(self) -> bool:
        """Whether running a note tracker on this stem is worth doing."""
        return self.name not in UNSUITABLE_STEMS

    @property
    def suitability_reason(self) -> str:
        if not self.is_suitable_for_notes:
            return "Drums are percussive; note tracking on them is rarely useful."
        return ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SeparationResult:
    stems: list[StemInfo] = field(default_factory=list)
    model: str = DEFAULT_MODEL
    device: str = "cpu"
    source: str = ""
    audio_duration: float = 0.0
    backend: str = ""
    elapsed_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.stems)

    @property
    def stem_names(self) -> list[str]:
        return [stem.name for stem in self.stems]

    def stem(self, name: str) -> StemInfo | None:
        return next((s for s in self.stems if s.name == name), None)

    @property
    def suitable_stems(self) -> list[StemInfo]:
        """Stems worth running note tracking on, best first."""
        candidates = [s for s in self.stems if s.is_suitable_for_notes]
        return sorted(candidates, key=lambda s: _preference_rank(s.name))

    @property
    def recommended_stem(self) -> StemInfo | None:
        """The single best stem to estimate a melody from, if any."""
        suitable = self.suitable_stems
        return suitable[0] if suitable else None

    @property
    def summary(self) -> str:
        if not self.stems:
            return "No stems were produced."
        names = ", ".join(self.stem_names)
        elapsed = f" in {self.elapsed_seconds:.0f}s" if self.elapsed_seconds else ""
        return f"{len(self.stems)} stems{elapsed}: {names}."

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "device": self.device,
            "source": self.source,
            "audio_duration": self.audio_duration,
            "backend": self.backend,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "warnings": list(self.warnings),
            "stems": [stem.to_dict() for stem in self.stems],
        }

    @classmethod
    def from_dict(cls, raw: dict, *, backend: str = "") -> "SeparationResult":
        stems = []
        for item in raw.get("stems", []):
            if not isinstance(item, dict) or not is_valid_stem(item):
                continue
            try:
                stems.append(
                    StemInfo(
                        name=str(item["name"]).strip(),
                        path=str(item["path"]),
                        duration=float(item.get("duration", 0.0) or 0.0),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue

        elapsed = float(raw.get("elapsed_seconds", 0.0) or 0.0)
        if not elapsed:
            elapsed = float(raw.get("elapsed", 0.0) or 0.0)

        return cls(
            stems=stems,
            model=str(raw.get("model", DEFAULT_MODEL)),
            device=str(raw.get("device", "cpu")),
            source=str(raw.get("source", "")),
            audio_duration=float(raw.get("audio_duration", 0.0)),
            backend=backend or str(raw.get("backend", "")),
            elapsed_seconds=elapsed,
            warnings=list(raw.get("warnings", [])),
        )


def _preference_rank(name: str) -> int:
    try:
        return PREFERRED_STEM_ORDER.index(name)
    except ValueError:
        return len(PREFERRED_STEM_ORDER)


def cpu_warning(device: str, audio_duration: float) -> str | None:
    """Honest expectation-setting for CPU runs, or None on GPU."""
    if device != "cpu":
        return None
    if audio_duration <= 0:
        return (
            "Separation is running on the CPU and is usually several times slower "
            "than the song's length. Be patient, or use a shorter clip."
        )
    return (
        f"Separation is running on the CPU for {audio_duration / 60:.1f} minutes of "
        "audio and usually takes several times longer than that. Be patient."
    )


class Stopwatch:
    """Tiny helper so the worker does not need datetime plumbing."""

    def __init__(self) -> None:
        self.started = time.monotonic()
        self.elapsed = 0.0

    def __enter__(self) -> "Stopwatch":
        self.started = time.monotonic()
        return self

    def __exit__(self, *args) -> None:
        self.elapsed = time.monotonic() - self.started

    @property
    def seconds(self) -> float:
        return self.elapsed if self.elapsed else time.monotonic() - self.started


def is_valid_stem(item: dict) -> bool:
    """Guard against a malformed stem entry reaching the UI."""
    name = str(item.get("name", "")).strip()
    duration = item.get("duration", 0.0)
    if not name or not str(item.get("path", "")):
        return False
    try:
        return math.isfinite(float(duration))
    except (TypeError, ValueError):
        return False