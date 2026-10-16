"""Facade the Streamlit UI talks to for stem separation.

Mirrors `app.transcription.service`. Demucs is optional: when it is missing the
app still exports MIDI, and the UI says so instead of failing.
"""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from ..ai_process import AiWorkerError, interpreter_in, module_available, run_python_worker
from .contract import (
    DEFAULT_MODEL,
    KNOWN_MODELS,
    SeparationError,
    SeparationResult,
    cpu_warning,
)

DEFAULT_AI_VENV = Path(".venv-ai")
DEFAULT_TIMEOUT = 3600  # CPU separation of a full song is slow; be generous.


@dataclass(frozen=True)
class SeparationAvailability:
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


def demucs_available(python: Path | None = None) -> bool:
    return module_available(python, "demucs.api", "Separator")


def resolve_backend(venv_path: Path = DEFAULT_AI_VENV) -> tuple[str, Path | None]:
    side = interpreter_in(venv_path)
    if side is not None and demucs_available(side):
        return "side-venv", side
    if demucs_available():
        return "in-process-venv", Path(sys.executable)
    return "none", None


def check_availability(venv_path: Path = DEFAULT_AI_VENV) -> SeparationAvailability:
    """Detect an installed Demucs backend without raising."""
    try:
        backend, interpreter = resolve_backend(venv_path)
    except OSError as error:
        return SeparationAvailability(False, "none", None, f"Could not inspect the environment: {error}")

    if backend == "none":
        return SeparationAvailability(
            False,
            "none",
            None,
            "Demucs is not installed, so stems cannot be separated.",
        )
    return SeparationAvailability(True, backend, str(interpreter) if interpreter else None)


def separate_stems(
    source: Path,
    output_dir: Path,
    *,
    model: str = DEFAULT_MODEL,
    device: str = "auto",
    shifts: int = 0,
    timeout: int = DEFAULT_TIMEOUT,
    venv_path: Path = DEFAULT_AI_VENV,
    worker: Path | None = None,
) -> SeparationResult:
    """Split `source` into stems inside `output_dir`.

    The source file is read-only here: it is never written to, moved, renamed or
    deleted. Everything produced lands under `output_dir`.
    """
    source = Path(source)
    output_dir = Path(output_dir)

    if not source.exists():
        raise SeparationError(f"Audio file not found: {source.name}", "conversion")

    # Guard the promise: the output must not sit on top of the input.
    try:
        if output_dir.resolve() == source.parent.resolve():
            raise SeparationError(
                "The output folder must not be the same folder as the original audio.",
                "conversion",
            )
    except OSError:
        pass

    if output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    backend, interpreter = resolve_backend(venv_path)
    if interpreter is None:
        raise SeparationError("Stem separation is unavailable: demucs is not installed.", "dependency")

    worker_path = worker or Path(__file__).resolve().parent / "worker.py"
    result_file = output_dir / "separation.json"

    try:
        payload = run_python_worker(
            worker_path,
            interpreter,
            source,
            output_path=result_file,
            extra_args={
                "output_dir": output_dir,
                "model": model,
                "device": device,
                "shifts": shifts,
            },
            timeout=timeout,
            cwd=worker_path.parents[2],
        )
    except AiWorkerError as error:
        raise SeparationError(str(error), error.kind, error.detail) from error

    payload["backend"] = backend
    result = SeparationResult.from_dict(payload, backend=backend)

    if not result.stems:
        raise SeparationError("Separation finished but produced no stems.", "runtime")

    # The original must still be exactly where it was.
    if not source.exists():
        raise SeparationError(
            "Separation finished but the original audio is missing. "
            "Something removed your file - check the output folder settings.",
            "runtime",
        )

    if model not in KNOWN_MODELS:
        result.warnings.append(
            f"'{model}' is not a model this app recognises; the stems may differ from the usual set."
        )

    warning = cpu_warning(result.device, result.audio_duration)
    if warning and warning not in result.warnings:
        result.warnings.append(warning)

    return result


__all__ = [
    "DEFAULT_MODEL",
    "KNOWN_MODELS",
    "SeparationAvailability",
    "SeparationError",
    "SeparationResult",
    "check_availability",
    "demucs_available",
    "separate_stems",
]
