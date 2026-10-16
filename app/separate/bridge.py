"""Run the Demucs worker in a separate interpreter.

Thin wrapper over `app.ai_process`. `resolve_backend` stays a module-level name
so tests can patch it here.
"""

from __future__ import annotations

import sys
from pathlib import Path

from ..ai_process import (
    MARKER,
    AiWorkerError,
    interpreter_in,
    module_available,
    run_python_worker,
)
from .contract import SeparationError

WORKER = Path(__file__).resolve().parent / "worker.py"
DEFAULT_AI_VENV = Path(".venv-ai")


def ai_python(venv_path: Path = DEFAULT_AI_VENV) -> Path | None:
    return interpreter_in(venv_path)


def demucs_available(python: Path | None = None) -> bool:
    """Probe the real entry point, not just the package name."""
    return module_available(python, "demucs.api", "Separator")


def resolve_backend(venv_path: Path = DEFAULT_AI_VENV) -> tuple[str, Path | None]:
    side = ai_python(venv_path)
    if side is not None:
        return "side-venv", side
    if demucs_available():
        return "in-process-venv", Path(sys.executable)
    return "none", None


def run_worker(
    input_path: Path,
    output_dir: Path,
    *,
    venv_path: Path = DEFAULT_AI_VENV,
    timeout: int = 3600,
    **params,
) -> dict:
    """Execute the Demucs worker and return its JSON payload."""
    backend, interpreter = resolve_backend(venv_path)
    if interpreter is None:
        raise SeparationError("Stem separation is unavailable: demucs is not installed.", "dependency")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        payload = run_python_worker(
            WORKER,
            interpreter,
            Path(input_path),
            output_path=output_dir / "separation.json",
            extra_args={"output_dir": output_dir, **params},
            timeout=timeout,
            cwd=WORKER.parents[2],
        )
    except AiWorkerError as error:
        raise SeparationError(str(error), error.kind, error.detail) from error

    payload["backend"] = backend
    return payload
