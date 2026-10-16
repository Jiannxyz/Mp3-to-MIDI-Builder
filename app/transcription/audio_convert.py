"""Make any audio file readable by Basic Pitch.

Basic Pitch wants a WAV path. WAV passes straight through; anything else is
converted with FFmpeg. FFmpeg is an OS-level dependency we do not bundle, so a
clear, actionable error is raised when it is missing rather than a raw
FileNotFoundError from subprocess.

No AI libraries are imported here, so this module is safe in the core app.
"""

from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

from .contract import KIND_CONVERSION, KIND_DEPENDENCY, TranscriptionError

# Basic Pitch's model works on 22050 Hz mono audio (basic_pitch.constants
# AUDIO_SAMPLE_RATE).
TARGET_SAMPLE_RATE = 22050
TARGET_CHANNELS = 1


def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def _wav_info(path: Path) -> tuple[int, int]:
    with wave.open(str(path), "rb") as handle:
        return handle.getframerate(), handle.getnchannels()


def ensure_wav(source: Path, workdir: Path) -> Path:
    """Return a WAV path for `source`, converting it if necessary.

    The result lives inside `workdir` so temporary conversions are removed with
    the rest of the temporary files.
    """
    source = Path(source)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    if not source.exists():
        raise TranscriptionError(f"Audio file not found: {source.name}", KIND_CONVERSION)

    if source.suffix.lower() == ".wav":
        try:
            rate, channels = _wav_info(source)
        except (wave.Error, OSError) as error:
            raise TranscriptionError(
                f"'{source.name}' has a .wav extension but could not be read as WAV: {error}",
                KIND_CONVERSION,
            ) from error

        destination = workdir / f"{source.stem}-22k.wav"
        if rate == TARGET_SAMPLE_RATE and channels == TARGET_CHANNELS:
            shutil.copy2(source, destination)
        else:
            _convert(source, destination, reason="resampling")
        return destination

    destination = workdir / f"{source.stem}-22k.wav"
    _convert(source, destination, reason="format conversion")
    return destination


def _convert(source: Path, destination: Path, *, reason: str) -> None:
    binary = ffmpeg_path()
    if binary is None:
        raise TranscriptionError(
            f"Converting '{source.name}' needs FFmpeg, which is not on PATH "
            f"({reason} required). Install FFmpeg and make sure 'ffmpeg' runs "
            f"from a command prompt.",
            KIND_DEPENDENCY,
        )

    command = [
        binary,
        "-nostdin",
        "-y",
        "-i",
        str(source),
        "-ac",
        str(TARGET_CHANNELS),
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-f",
        "wav",
        str(destination),
    ]

    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired as error:
        raise TranscriptionError(
            f"FFmpeg timed out while converting '{source.name}'.", KIND_CONVERSION
        ) from error

    if completed.returncode != 0:
        tail = (completed.stderr or "").strip().splitlines()[-1:] or ["no output"]
        raise TranscriptionError(
            f"FFmpeg could not convert '{source.name}': {tail[0]}",
            KIND_CONVERSION,
            detail=completed.stderr or "",
        )

    if not destination.exists() or destination.stat().st_size == 0:
        raise TranscriptionError(
            f"Conversion of '{source.name}' produced an empty file.", KIND_CONVERSION
        )


def write_tone_wav(path: Path, *, seconds: float = 1.0, rate: int = TARGET_SAMPLE_RATE, frequency: float = 440.0) -> Path:
    """Write a small sine WAV. Used by tests; never shipped as user data.

    Generated locally with the standard library so tests need no audio fixture
    files and no AI dependencies.
    """
    import math
    import struct

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    frames = int(seconds * rate)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        samples = bytearray()
        for index in range(frames):
            value = int(0.3 * 32767 * math.sin(2 * math.pi * frequency * index / rate))
            samples += struct.pack("<h", value)
        handle.writeframes(bytes(samples))

    return path