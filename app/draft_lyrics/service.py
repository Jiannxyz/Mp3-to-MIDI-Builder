"""Facade the Streamlit UI talks to for lyric drafting.

Mirrors the other AI services. Drafting is optional: the manual lyric editor is
always available, and this never becomes the only way to enter lyrics.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from ..ai_process import AiWorkerError, interpreter_in, module_available, run_python_worker
from .contract import DEFAULT_MODEL, MODEL_CHOICES, DraftError, LyricDraft

DEFAULT_AI_VENV = Path(".venv-ai")
DEFAULT_TIMEOUT = 1800


@dataclass(frozen=True)
class DraftAvailability:
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


def whisper_available(python: Path | None = None) -> bool:
    return module_available(python, "faster_whisper", "WhisperModel")


def resolve_backend(venv_path: Path = DEFAULT_AI_VENV) -> tuple[str, Path | None]:
    side = interpreter_in(venv_path)
    if side is not None and whisper_available(side):
        return "side-venv", side
    if whisper_available():
        return "in-process-venv", Path(sys.executable)
    return "none", None


def check_availability(venv_path: Path = DEFAULT_AI_VENV) -> DraftAvailability:
    """Detect an installed ASR backend without raising."""
    try:
        backend, interpreter = resolve_backend(venv_path)
    except OSError as error:
        return DraftAvailability(False, "none", None, f"Could not inspect the environment: {error}")

    if backend == "none":
        return DraftAvailability(
            False,
            "none",
            None,
            "The speech-recognition extra is not installed, so drafts are unavailable.",
        )
    return DraftAvailability(True, backend, str(interpreter) if interpreter else None)


def draft_lyrics(
    source: Path,
    workdir: Path,
    *,
    model_name: str = DEFAULT_MODEL,
    language: str | None = None,
    device: str = "cpu",
    timeout: int = DEFAULT_TIMEOUT,
    venv_path: Path = DEFAULT_AI_VENV,
    worker: Path | None = None,
) -> LyricDraft:
    """Produce a DRAFT of lyrics with estimated timings.

    The result is a guess, not a transcription. It is returned as a draft the
    user replaces or corrects; nothing here edits the user's existing lyrics.
    """
    source = Path(source)
    if not source.exists():
        raise DraftError(f"Audio file not found: {source.name}", "conversion")

    backend, interpreter = resolve_backend(venv_path)
    if interpreter is None:
        raise DraftError("Lyric drafting is unavailable: the ASR extra is not installed.", "dependency")

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    worker_path = worker or Path(__file__).resolve().parent / "worker.py"
    result_file = workdir / "draft.json"

    try:
        payload = run_python_worker(
            worker_path,
            interpreter,
            source,
            output_path=result_file,
            extra_args={"model": model_name, "language": language or None, "device": device},
            timeout=timeout,
            cwd=worker_path.parents[2],
        )
    except AiWorkerError as error:
        raise DraftError(str(error), error.kind, error.detail) from error

    payload["backend"] = backend
    return LyricDraft.from_dict(payload, backend=backend)


def draft_to_lyric_rows(draft: LyricDraft, *, unit: float = 0.45, mark_estimated: bool = True):
    """Turn a draft into editable timing rows, ready for the existing editor.

    `mark_estimated` is kept for symmetry with note estimates; lyric rows have
    no source column, so the UI labels the whole table instead.
    """
    from app.timing import LyricRow

    rows: list[LyricRow] = []
    for line_number, line in enumerate(draft.lines, start=1):
        for index, word in enumerate(line.words):
            start = float(word.start)
            end = max(start + unit, float(word.end))
            rows.append(
                LyricRow(
                    line=line_number,
                    text=word.text,
                    start=round(start, 3),
                    end=round(end, 3),
                    word_end=True,
                )
            )
            _ = index
    return rows


__all__ = [
    "DEFAULT_MODEL",
    "MODEL_CHOICES",
    "DraftAvailability",
    "DraftError",
    "LyricDraft",
    "check_availability",
    "draft_lyrics",
    "draft_to_lyric_rows",
    "whisper_available",
]
