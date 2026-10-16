"""Run the note-estimation worker in a separate interpreter.

Two supported layouts, tried in order:

1. A side environment at `.venv-ai` with the AI dependencies (preferred - it
   keeps TensorFlow out of the app environment entirely).
2. The current interpreter, if it can already import basic_pitch (handy when
   someone just installs the extras into the main venv).

If neither works, the app keeps running and note estimation is simply reported
as unavailable. Nothing about note estimation is required to export a MIDI.

The subprocess mechanics live in `app.ai_process`, shared with the separation
and lyric-draft services. `resolve_backend` stays a module-level name here so
tests can patch it in this namespace.
"""

from __future__ import annotations

import sys
from pathlib import Path

from ..ai_process import (
    MARKER,
    AiWorkerError,
    extract_payload,
    interpreter_in,
    module_available,
    run_python_worker,
)
from .contract import KIND_DEPENDENCY, KIND_RUNTIME, KIND_TIMEOUT, TranscriptionError

WORKER = Path(__file__).resolve().parent / "worker.py"
DEFAULT_AI_VENV = Path(".venv-ai")

EXIT_CODES = {
    3: KIND_DEPENDENCY,
    4: "model",
    5: KIND_RUNTIME,
}


def ai_python(venv_path: Path = DEFAULT_AI_VENV) -> Path | None:
    """Interpreter inside the side environment, if one was created."""
    return interpreter_in(venv_path)


def basic_pitch_available(python: Path | None = None) -> bool:
    """True if that interpreter can run the worker (cached per interpreter).

    Probes the real entry point, not just the package: `import basic_pitch`
    succeeds while `basic_pitch.inference` fails if a dependency is broken.
    """
    return module_available(python, "basic_pitch.inference", "predict")


def resolve_backend(venv_path: Path = DEFAULT_AI_VENV) -> tuple[str, Path | None]:
    """Pick an interpreter for the worker. Returns (kind, path)."""
    side = ai_python(venv_path)
    if side is not None:
        return "side-venv", side
    if basic_pitch_available():
        return "in-process-venv", Path(sys.executable)
    return "none", None


def _extract(payload_text: str) -> dict | None:
    """Pull the marker line out of whatever the libraries printed."""
    return extract_payload(payload_text)


def run_worker(
    wav_path: Path,
    *,
    venv_path: Path = DEFAULT_AI_VENV,
    timeout: int = 900,
    **params,
) -> dict:
    """Execute the worker and return its JSON payload.

    Raises TranscriptionError with a `kind` the UI can explain.
    """
    backend, interpreter = resolve_backend(venv_path)
    if interpreter is None:
        raise TranscriptionError(
            "Note estimation is unavailable: the AI extras are not installed.",
            KIND_DEPENDENCY,
        )

    try:
        payload = run_python_worker(
            WORKER,
            interpreter,
            Path(wav_path),
            timeout=timeout,
            extra_args=params,
            cwd=WORKER.parents[2],
        )
    except AiWorkerError as error:
        raise TranscriptionError(str(error), error.kind, error.detail) from error

    payload["backend"] = backend
    return payload