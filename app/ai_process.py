"""Shared plumbing for running AI workers in a separate interpreter.

Used by every optional AI service (note estimation, stem separation, lyric
drafting) so the subprocess protocol exists in one place:

    <ai python> worker.py --input X --output Y   ->  JSON file  ->  dict

Two conventions that matter and are easy to get wrong:

* **Results go to a file, not stdout.** TensorFlow, torch and librosa all log to
  the console; a stdout protocol would be corrupted by a warning line. A
  `__KMB_JSON__` marker line on stdout is kept as a fallback.
* **A worker's own directory shadows the standard library**, because running a
  script puts that directory first on `sys.path`. Never name a worker module
  after a stdlib module.

This module deliberately does NOT resolve which interpreter to use. Each
service exposes its own `resolve_backend()` so tests can patch it in that
service's namespace.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

MARKER = "__KMB_JSON__"
DEFAULT_AI_VENV = Path(".venv-ai")

EXIT_DEPENDENCY = 3
EXIT_MODEL = 4
EXIT_RUNTIME = 5

EXIT_CODES = {
    EXIT_DEPENDENCY: "dependency",
    EXIT_MODEL: "model",
    EXIT_RUNTIME: "runtime",
}


class AiWorkerError(Exception):
    """A worker failed in a way the UI can explain.

    `kind` is one of: dependency, conversion, model, runtime, timeout.
    """

    def __init__(self, kind: str, message: str, detail: str = ""):
        super().__init__(message)
        self.kind = kind
        self.detail = detail


# A worker reports which stage it is in by writing this next to its output.
# Without it, a stalled model download and genuinely slow processing look
# identical from the outside: both are a silent worker until the timeout, and
# blaming "this is slow on a CPU" for a blocked download sends people off to
# tune the wrong thing.
PHASE_DOWNLOADING = "downloading-model"
PHASE_PROCESSING = "processing"

PHASE_TIMEOUT_MESSAGES = {
    PHASE_DOWNLOADING: (
        "Downloading the model timed out after {timeout} seconds. The first run "
        "has to fetch the model weights over the internet, and that download "
        "appears to be stalled. This is not slow processing - check your "
        "connection, or run it somewhere model downloads are allowed."
    ),
    PHASE_PROCESSING: (
        "Still working after {timeout} seconds. This is the slow stage, which is "
        "expected on a CPU for a full song. Try a shorter clip, lower the quality "
        "settings, or run it when the machine is idle."
    ),
}

PHASE_FALLBACK_MESSAGE = (
    "The AI process was stopped after {timeout} seconds. It did not report which "
    "stage it had reached, so it may have been downloading a model or still "
    "processing."
)

KIND_STALLED = "stalled-download"

# A blocked model download shows no progress at all, so the only useful signal
# is the cache not growing. Without this, finding out costs the full timeout -
# measured at 3600s in the wild, all of it spent waiting for bytes that were
# never coming.
DEFAULT_STALL_SECONDS = 90.0
DEFAULT_POLL_SECONDS = 5.0
DEFAULT_HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

STALLED_DOWNLOAD_MESSAGE = (
    "The model download made no progress for {stall} seconds and was stopped. "
    "This is a blocked or stalled download, not slow processing. Your network "
    "reaches the site but not the model files. Either run this once on a "
    "network that allows the download, copy the model cache across from a "
    "machine that has it, or skip this step - the rest of the app works."
)


def phase_file(output_path: Path) -> Path:
    return Path(str(output_path) + ".phase")


def import_error_message(package: str, error: BaseException, *, hint: str = "") -> str:
    """Classify an ImportError: is the package missing, or is it installed but broken?

    These need completely different fixes. Reporting "basic_pitch is not
    installed" when basic_pitch is fine and `resampy` cannot import `pkg_resources`
    tells the user to reinstall something they already have. `error.name` is set
    by ModuleNotFoundError and names the module that actually failed.
    """
    missing = getattr(error, "name", "") or ""
    suffix = f" {hint}" if hint else ""

    if not missing or missing == package or missing.startswith(package + "."):
        return f"{package} is not installed in this environment ({error}).{suffix}"

    return (
        f"{package} is installed, but importing it failed because '{missing}' could "
        f"not be imported ({error}). Run: pip install -r requirements-ai.txt"
    )


def read_phase(output_path: Path) -> str:
    """Which stage the worker last entered, or '' if it never said.

    Workers may append '|detail' for diagnostics; only the stage is returned.
    """
    try:
        raw = phase_file(output_path).read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    return raw.split("|", 1)[0].strip()


def read_phase_detail(output_path: Path) -> str:
    """The diagnostic a worker attached to its last phase, if any."""
    try:
        raw = phase_file(output_path).read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    if "|" not in raw:
        return ""
    return raw.split("|", 1)[1].strip()


def _directory_size(path: Path) -> int:
    total = 0
    try:
        for entry in path.rglob("*"):
            try:
                if entry.is_file():
                    total += entry.stat().st_size
            except OSError:
                continue
    except OSError:
        return total
    return total


def interpreter_in(venv_path: Path = DEFAULT_AI_VENV) -> Path | None:
    """The interpreter inside a side environment, if one exists."""
    candidate = Path(venv_path) / "Scripts" / "python.exe"
    if candidate.exists():
        return candidate
    posix = Path(venv_path) / "bin" / "python"
    return posix if posix.exists() else None


_MODULE_PROBE: dict[tuple[str, str], bool] = {}


def module_available(python: Path | None, module: str, symbol: str = "") -> bool:
    """True if that interpreter can import `module` (and `symbol`, if given).

    Pass the SAME import the worker will do, not just the package name. A
    shallow `import basic_pitch` succeeds even when the worker cannot run,
    because a transitive dependency is broken - `basic_pitch/__init__.py` only
    probes optional extras, while `basic_pitch.inference` pulls in resampy,
    which needs pkg_resources. Probing the package alone reported the service
    as available and then every real run failed.

    Cached per (interpreter, module, symbol): this spawns a process, and the
    Streamlit script re-runs on every interaction. Installing an extra requires
    restarting the app anyway, so a cached negative cannot go stale in practice.
    """
    interpreter = Path(python or Path(sys.executable))
    key = (str(interpreter), module, symbol)

    cached = _MODULE_PROBE.get(key)
    if cached is not None:
        return cached

    statement = f"from {module} import {symbol}" if symbol else f"import {module}"
    try:
        completed = subprocess.run(
            [str(interpreter), "-c", statement],
            capture_output=True,
            timeout=60,
        )
        found = completed.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        found = False

    _MODULE_PROBE[key] = found
    return found


def extract_payload(text: str) -> dict | None:
    """Pull the marker line out of whatever the libraries printed."""
    for line in (text or "").splitlines():
        if line.startswith(MARKER):
            try:
                return json.loads(line[len(MARKER) :])
            except json.JSONDecodeError:
                return None
    return None


def run_python_worker(
    worker: Path,
    interpreter: Path,
    input_path: Path,
    *,
    output_path: Path | None = None,
    extra_args: dict | None = None,
    timeout: int = 900,
    cwd: Path | None = None,
    cache_dir: Path | None = DEFAULT_HF_CACHE,
    stall_seconds: float = DEFAULT_STALL_SECONDS,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
) -> dict:
    """Run `worker --input X --output Y` and return the JSON payload.

    While the worker says it is downloading a model, the HuggingFace cache is
    watched for growth. No growth for `stall_seconds` means the download is
    blocked, and the process is stopped immediately with `KIND_STALLED` rather
    than making the user wait out the whole `timeout`.

    Raises AiWorkerError with a usable `kind`. The caller is responsible for
    choosing the interpreter, so tests can substitute one.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise AiWorkerError("conversion", f"Input not found: {input_path.name}")

    if output_path is None:
        output_path = input_path.with_suffix(".kmbnotes.json")
    output_path = Path(output_path)

    command = [str(interpreter), str(worker), "--input", str(input_path), "--output", str(output_path)]
    for name, value in (extra_args or {}).items():
        if value is None:
            continue
        command.extend(["--" + name.replace("_", "-"), str(value)])

    # stdout/stderr go to files, not PIPE: torch and librosa log heavily and a
    # full pipe buffer would deadlock the child while we poll it.
    #
    # The directory is cleaned with ignore_errors rather than by
    # TemporaryDirectory: when we kill the child, its inherited handles may not be
    # released immediately on Windows, and TemporaryDirectory raises
    # PermissionError during cleanup, which would mask the real error.
    log_dir = Path(tempfile.mkdtemp(prefix="kmb-log-"))
    out_file = log_dir / "out.txt"
    err_file = log_dir / "err.txt"
    try:
        with out_file.open("w", encoding="utf-8") as out_handle, err_file.open("w", encoding="utf-8") as err_handle:
            process = subprocess.Popen(
                command,
                stdout=out_handle,
                stderr=err_handle,
                cwd=str(cwd) if cwd else None,
            )

            deadline = time.monotonic() + timeout
            watched = cache_dir is not None
            last_size = _directory_size(Path(cache_dir)) if watched else 0
            last_growth = time.monotonic()

            while process.poll() is None:
                time.sleep(poll_seconds)

                if time.monotonic() > deadline:
                    process.kill()
                    process.wait(timeout=30)
                    phase = read_phase(output_path)
                    template = PHASE_TIMEOUT_MESSAGES.get(phase, PHASE_FALLBACK_MESSAGE)
                    raise AiWorkerError("timeout", template.format(timeout=timeout))

                if watched and read_phase(output_path) == PHASE_DOWNLOADING:
                    size = _directory_size(Path(cache_dir))
                    if size > last_size:
                        last_size = size
                        last_growth = time.monotonic()
                    elif time.monotonic() - last_growth > stall_seconds:
                        process.kill()
                        process.wait(timeout=30)
                        detail = read_phase_detail(output_path)
                        message = STALLED_DOWNLOAD_MESSAGE.format(stall=int(stall_seconds))
                        if detail:
                            message += f" Worker reported: {detail}."
                        raise AiWorkerError(KIND_STALLED, message)

            process.wait(timeout=30)

        stdout_text = out_file.read_text(encoding="utf-8", errors="replace")
        stderr_text = err_file.read_text(encoding="utf-8", errors="replace")
    finally:
        shutil.rmtree(log_dir, ignore_errors=True)

    payload: dict | None = None
    if output_path.exists():
        try:
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            payload = None
    if payload is None:
        payload = extract_payload(stdout_text)

    if payload is None:
        tail = (stderr_text.strip().splitlines() or ["no output"])[-1]
        raise AiWorkerError(
            EXIT_CODES.get(process.returncode, "runtime"),
            f"The AI process did not return a result. Last message: {tail}",
            detail=stderr_text[-2000:],
        )

    if not payload.get("ok"):
        raise AiWorkerError(
            payload.get("kind") or EXIT_CODES.get(process.returncode, "runtime"),
            payload.get("message") or "The AI process failed.",
            payload.get("detail", ""),
        )

    return payload