"""Upload handling, duration probing and safe temporary storage.

Everything here is local. Audio is written to a temporary folder, processed,
and removed. Nothing is uploaded anywhere.

Safety rule: cleanup refuses to delete anything that is not inside the
workspace directory it was given, so a bad path can never remove the user's
files or a parent directory.
"""

from __future__ import annotations

import shutil
import tempfile
import wave
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

ALLOWED_EXTENSIONS = frozenset({".mp3", ".wav"})

# A karaoke song is a few minutes of audio; 50 MB is generous for MP3 or WAV
# and small enough that a mistake cannot fill the disk.
MAX_UPLOAD_MB = 50
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

DEFAULT_WORKSPACE = Path("data") / "workspace"


class AudioError(ValueError):
    """Raised for an unusable upload. The message is shown to the user."""


def format_duration(seconds: float) -> str:
    """Seconds -> '3:45' or '1:02:03'. Used for display only."""
    if seconds is None or seconds < 0:
        return "unknown"

    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def check_extension(filename: str) -> str:
    """Validate the file extension, returning it in lower case."""
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise AudioError(f"'{filename}' is not supported. Allowed file types: {allowed}.")
    return suffix


def check_size(num_bytes: int) -> None:
    if num_bytes <= 0:
        raise AudioError("That file is empty.")
    if num_bytes > MAX_UPLOAD_BYTES:
        raise AudioError(
            f"That file is {num_bytes / 1024 / 1024:.1f} MB. "
            f"The limit is {MAX_UPLOAD_MB} MB."
        )


def safe_cleanup(path: Path, base: Path) -> bool:
    """Delete a file or directory, but ONLY if it is inside `base`.

    Returns True if something was removed. Never deletes `base` itself, and
    never follows a path that points outside the workspace.
    """
    try:
        resolved = Path(path).resolve()
        resolved_base = Path(base).resolve()
    except OSError:
        return False

    if resolved == resolved_base:
        return False
    if resolved_base not in resolved.parents:
        return False

    if resolved.is_dir():
        shutil.rmtree(resolved, ignore_errors=True)
        return True

    try:
        resolved.unlink()
        return True
    except OSError:
        return False


@contextmanager
def workspace(base: Path | str = DEFAULT_WORKSPACE) -> Iterator[Path]:
    """Create a temporary directory and always remove it afterwards."""
    base_path = Path(base)
    base_path.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="kmb-", dir=base_path))
    try:
        yield directory
    finally:
        safe_cleanup(directory, base_path)


def save_upload(data: bytes, filename: str, directory: Path) -> Path:
    """Validate and write an uploaded file into `directory`."""
    check_extension(filename)
    check_size(len(data))

    # Keep only the base name: an upload name could contain path separators.
    safe_name = Path(filename).name or "upload"
    destination = Path(directory) / safe_name
    destination.write_bytes(data)
    return destination


def probe_duration(path: Path) -> float:
    """Duration in seconds for a WAV or MP3 file.

    WAV is read with the standard library; MP3 needs mutagen, which reads the
    frame headers directly and needs no FFmpeg.
    """
    path = Path(path)
    if not path.exists():
        raise AudioError(f"File not found: {path.name}")

    suffix = path.suffix.lower()

    if suffix == ".wav":
        try:
            with wave.open(str(path), "rb") as handle:
                frames = handle.getnframes()
                rate = handle.getframerate()
                if rate:
                    return frames / float(rate)
        except (wave.Error, OSError):
            pass  # fall through to mutagen, which handles more variants

    try:
        import mutagen

        parsed = mutagen.File(str(path))
        if parsed is not None and parsed.info is not None:
            length = getattr(parsed.info, "length", None)
            if length and length > 0:
                return float(length)
    except ImportError:
        raise AudioError(
            "Reading MP3 duration needs the 'mutagen' package. "
            "Install it with: pip install mutagen"
        ) from None
    except Exception as error:  # noqa: BLE001 - corrupt files raise many types
        raise AudioError(f"Could not read the audio file: {error}") from error

    raise AudioError(
        f"Could not determine the duration of '{path.name}'. "
        "The file may be corrupt or not really audio."
    )