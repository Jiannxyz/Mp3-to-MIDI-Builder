"""The documented lyric-draft contract.

A "draft" is generated text, not a transcription of the lyrics. Speech
recognition is trained on speech, not singing, so expect wrong words, wrong
line breaks and roughly-plausible timings. Everything here is treated as a
starting point the user corrects or replaces.

Named `contract.py` because a `types.py` beside the worker would shadow the
standard library (see `app/transcription/contract.py`).
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
        "The speech-recognition extra is not installed. Paste your lyrics in "
        "Step 3 instead - the manual editor works exactly the same."
    ),
    KIND_CONVERSION: "The audio could not be read. Installing FFmpeg usually fixes this.",
    KIND_MODEL: "The transcription model could not be loaded. First use downloads it, so it needs a connection once.",
    KIND_RUNTIME: "Lyric drafting failed unexpectedly.",
    KIND_TIMEOUT: (
        "This was stopped because it ran too long. If it had just started, it may "
        "have been stuck downloading the model rather than transcribing - the "
        "first run needs internet to fetch the weights."
    ),
    "stalled-download": (
        "The model weights could not be downloaded. Your network reaches the site "
        "but not the model files. Run this once on a network that allows the "
        "download, or type your own lyrics - the editor works without this."
    ),
}

# Human-readable sizes, so the UI can set expectations before downloading.
MODEL_CHOICES = {
    "tiny": "tiny - fastest, poor quality, smallest download",
    "base": "base - a reasonable starting point",
    "small": "small - better, noticeably slower on CPU",
    "medium": "medium - slow on CPU, large download",
    "large-v3": "large-v3 - best quality, very slow and very large on CPU",
}

DEFAULT_MODEL = "base"

DRAFT_CAVEAT = (
    "This is a DRAFT produced by speech recognition, which is trained on speech "
    "rather than singing. The words, the spelling and the timings are all "
    "guesses. Replace the text with your own lyrics and correct the timings."
)


class DraftError(Exception):
    """A lyric-drafting failure the UI can explain."""

    def __init__(self, message: str, kind: str = KIND_RUNTIME, detail: str = ""):
        super().__init__(message)
        self.kind = kind
        self.detail = detail

    @property
    def hint(self) -> str:
        return ERROR_HINTS.get(self.kind, "")


@dataclass(frozen=True)
class DraftWord:
    """One recognised word with ESTIMATED timings in seconds."""

    text: str
    start: float
    end: float
    confidence: float = 0.0

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def is_valid(self) -> bool:
        return (
            bool(self.text.strip())
            and math.isfinite(self.start)
            and math.isfinite(self.end)
            and self.start >= 0
            and self.end > self.start
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DraftLine:
    words: list[DraftWord] = field(default_factory=list)

    @property
    def start(self) -> float:
        return min((w.start for w in self.words), default=0.0)

    @property
    def end(self) -> float:
        return max((w.end for w in self.words), default=0.0)

    @property
    def text(self) -> str:
        return " ".join(word.text for word in self.words).strip()

    def to_dict(self) -> dict:
        return {"start": self.start, "end": self.end, "text": self.text,
                "words": [word.to_dict() for word in self.words]}


@dataclass
class LyricDraft:
    lines: list[DraftLine] = field(default_factory=list)
    model: str = DEFAULT_MODEL
    language: str = ""
    device: str = "cpu"
    backend: str = ""
    elapsed_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)
    dropped: int = 0

    def __len__(self) -> int:
        return len(self.lines)

    @property
    def words(self) -> list[DraftWord]:
        return [word for line in self.lines for word in line.words]

    @property
    def text(self) -> str:
        """The draft as pasteable lyric text, one song line per line."""
        return "\n".join(line.text for line in self.lines if line.text)

    @property
    def word_count(self) -> int:
        return len(self.words)

    @property
    def summary(self) -> str:
        if not self.lines:
            return "No speech was recognised, so the draft is empty."
        dropped = f", {self.dropped} unusable dropped" if self.dropped else ""
        return f"{self.word_count} words in {len(self.lines)} lines{dropped}."

    @property
    def timings_are_estimates(self) -> bool:
        """Always True. There is no reliable alignment for singing here."""
        return True

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "language": self.language,
            "device": self.device,
            "backend": self.backend,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "warnings": list(self.warnings),
            "dropped": self.dropped,
            "lines": [line.to_dict() for line in self.lines],
        }

    @classmethod
    def from_dict(cls, raw: dict, *, backend: str = "") -> "LyricDraft":
        lines: list[DraftLine] = []
        dropped = 0

        for entry in raw.get("lines", []):
            words: list[DraftWord] = []
            for item in (entry or {}).get("words", []):
                try:
                    word = DraftWord(
                        text=str(item.get("text", "")).strip(),
                        start=float(item.get("start", 0.0)),
                        end=float(item.get("end", 0.0)),
                        confidence=float(item.get("confidence", 0.0) or 0.0),
                    )
                except (TypeError, ValueError):
                    dropped += 1
                    continue

                if word.is_valid:
                    words.append(word)
                else:
                    dropped += 1

            if words:
                lines.append(DraftLine(words=words))

        return cls(
            lines=lines,
            model=str(raw.get("model", DEFAULT_MODEL)),
            language=str(raw.get("language", "")),
            device=str(raw.get("device", "cpu")),
            backend=backend or str(raw.get("backend", "")),
            elapsed_seconds=float(raw.get("elapsed_seconds", 0.0) or 0.0),
            warnings=list(raw.get("warnings", [])),
            dropped=dropped,
        )


class Stopwatch:
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