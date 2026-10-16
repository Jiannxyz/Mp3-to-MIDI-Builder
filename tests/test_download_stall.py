"""Tests for the download stall watchdog.

A blocked model download makes zero progress, so the only usable signal is the
HuggingFace cache not growing. Observed in the wild: 3600 seconds spent waiting
for bytes that were never coming. The watchdog turns that into ~90 seconds with
an accurate message.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from app.ai_process import (
    DEFAULT_STALL_SECONDS,
    KIND_STALLED,
    PHASE_DOWNLOADING,
    PHASE_PROCESSING,
    STALLED_DOWNLOAD_MESSAGE,
    AiWorkerError,
    run_python_worker,
)
from app.transcription.audio_convert import write_tone_wav

FAST = {"poll_seconds": 0.2, "stall_seconds": 1.0}


def _worker(tmp_path, body: str) -> Path:
    path = tmp_path / "w.py"
    path.write_text(body, encoding="utf-8")
    return path


def _downloading_worker(tmp_path) -> Path:
    """Writes the downloading phase then hangs, growing nothing."""
    return _worker(
        tmp_path,
        "import sys, time, pathlib\n"
        "out = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])\n"
        "pathlib.Path(str(out) + '.phase').write_text('downloading-model')\n"
        "time.sleep(60)\n",
    )


def _growing_worker(tmp_path, cache: Path) -> Path:
    """Pretends to download by growing the cache, then succeeds."""
    return _worker(
        tmp_path,
        "import sys, time, pathlib\n"
        "out = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])\n"
        f"cache = pathlib.Path({str(cache)!r})\n"
        "pathlib.Path(str(out) + '.phase').write_text('downloading-model')\n"
        "blob = cache / 'model.bin'\n"
        "for i in range(12):\n"
        "    blob.write_bytes(b'x' * (i + 1) * 4096)\n"
        "    time.sleep(0.25)\n"
        "pathlib.Path(str(out) + '.phase').write_text('processing')\n"
        "out.write_text('{\"ok\": true, \"notes\": []}')\n",
    )


def _slow_processing_worker(tmp_path) -> Path:
    """Reports processing and takes a while; the cache never grows, and that is fine."""
    return _worker(
        tmp_path,
        "import sys, time, pathlib, json\n"
        "out = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])\n"
        "pathlib.Path(str(out) + '.phase').write_text('processing')\n"
        "time.sleep(3)\n"
        "out.write_text('{\"ok\": true, \"notes\": []}')\n",
    )


# --- the watchdog ---------------------------------------------------------


def test_a_stalled_download_is_caught_quickly(tmp_path):
    """The whole point: do not make the user sit out the full timeout."""
    audio = write_tone_wav(tmp_path / "a.wav", seconds=0.2)
    cache = tmp_path / "cache"
    cache.mkdir()
    worker = _downloading_worker(tmp_path)

    started = time.monotonic()
    with pytest.raises(AiWorkerError) as error:
        run_python_worker(
            worker, Path(sys.executable), audio,
            output_path=tmp_path / "out.json",
            cache_dir=cache, timeout=600, **FAST,
        )
    elapsed = time.monotonic() - started

    assert error.value.kind == KIND_STALLED
    assert elapsed < 30, f"must not wait out the 600s timeout (took {elapsed:.0f}s)"
    assert "no progress" in str(error.value)


def test_the_stall_message_explains_what_to_do(tmp_path):
    audio = write_tone_wav(tmp_path / "a.wav", seconds=0.2)
    cache = tmp_path / "cache"
    cache.mkdir()

    with pytest.raises(AiWorkerError) as error:
        run_python_worker(
            _downloading_worker(tmp_path), Path(sys.executable), audio,
            output_path=tmp_path / "out.json", cache_dir=cache, timeout=600, **FAST,
        )

    message = str(error.value)
    assert "not slow processing" in message
    assert "network" in message.lower()
    assert "skip" in message.lower(), "must offer a way forward"


def test_a_download_that_makes_progress_is_not_killed(tmp_path):
    audio = write_tone_wav(tmp_path / "a.wav", seconds=0.2)
    cache = tmp_path / "cache"
    cache.mkdir()

    result = run_python_worker(
        _growing_worker(tmp_path, cache), Path(sys.executable), audio,
        output_path=tmp_path / "out.json", cache_dir=cache, timeout=120, **FAST,
    )

    assert result["ok"] is True


def test_slow_processing_is_not_mistaken_for_a_stalled_download(tmp_path):
    """The cache legitimately stops growing once processing starts."""
    audio = write_tone_wav(tmp_path / "a.wav", seconds=0.2)
    cache = tmp_path / "cache"
    cache.mkdir()

    result = run_python_worker(
        _slow_processing_worker(tmp_path), Path(sys.executable), audio,
        output_path=tmp_path / "out.json", cache_dir=cache, timeout=120, **FAST,
    )

    assert result["ok"] is True


def test_a_very_long_processing_run_still_times_out(tmp_path):
    audio = write_tone_wav(tmp_path / "a.wav", seconds=0.2)
    cache = tmp_path / "cache"
    cache.mkdir()
    worker = _worker(
        tmp_path,
        "import sys, time, pathlib\n"
        "out = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])\n"
        "pathlib.Path(str(out) + '.phase').write_text('processing')\n"
        "time.sleep(60)\n",
    )

    with pytest.raises(AiWorkerError) as error:
        run_python_worker(
            worker, Path(sys.executable), audio,
            output_path=tmp_path / "out.json", cache_dir=cache,
            timeout=2, poll_seconds=0.2, stall_seconds=600,
        )

    assert error.value.kind == "timeout"
    assert "Still working" in str(error.value)


def test_watching_can_be_switched_off(tmp_path):
    audio = write_tone_wav(tmp_path / "a.wav", seconds=0.2)
    worker = _downloading_worker(tmp_path)

    with pytest.raises(AiWorkerError) as error:
        run_python_worker(
            worker, Path(sys.executable), audio,
            output_path=tmp_path / "out.json", cache_dir=None,
            timeout=2, poll_seconds=0.2, stall_seconds=600,
        )

    assert error.value.kind == "timeout", "without a cache to watch, only the timeout applies"


def test_the_stall_window_is_generous_enough_for_a_slow_but_live_download():
    """90s of zero progress on an ~80MB file is dead, not slow."""
    assert DEFAULT_STALL_SECONDS >= 60
    assert DEFAULT_STALL_SECONDS <= 300


def test_a_failed_run_still_reports_its_own_error(tmp_path):
    audio = write_tone_wav(tmp_path / "a.wav", seconds=0.2)
    cache = tmp_path / "cache"
    cache.mkdir()
    worker = _worker(
        tmp_path,
        "import sys, pathlib\n"
        "out = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])\n"
        "out.write_text('{\"ok\": false, \"kind\": \"model\", \"message\": \"weights bad\"}')\n",
    )

    with pytest.raises(AiWorkerError) as error:
        run_python_worker(
            worker, Path(sys.executable), audio,
            output_path=tmp_path / "out.json", cache_dir=cache, **FAST,
        )

    assert error.value.kind == "model"
    assert "weights bad" in str(error.value)


def test_successful_runs_are_unaffected(tmp_path):
    audio = write_tone_wav(tmp_path / "a.wav", seconds=0.2)
    cache = tmp_path / "cache"
    cache.mkdir()
    worker = _worker(
        tmp_path,
        "import sys, pathlib\n"
        "out = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])\n"
        "out.write_text('{\"ok\": true, \"notes\": [1, 2, 3]}')\n",
    )

    result = run_python_worker(
        worker, Path(sys.executable), audio,
        output_path=tmp_path / "out.json", cache_dir=cache, **FAST,
    )

    assert result["notes"] == [1, 2, 3]


# --- the message reaches the services -------------------------------------


def test_separation_surfaces_the_stall_with_a_hint(tmp_path, monkeypatch):
    from app.separate import service
    from app.separate.contract import SeparationError

    audio = write_tone_wav(tmp_path / "a.wav", seconds=0.2)
    monkeypatch.setattr(
        "app.separate.service.resolve_backend", lambda *a, **k: ("side-venv", Path(sys.executable))
    )
    monkeypatch.setattr(
        "app.separate.service.run_python_worker",
        lambda *a, **k: (_ for _ in ()).throw(
            AiWorkerError(KIND_STALLED, STALLED_DOWNLOAD_MESSAGE.format(stall=90))
        ),
    )

    with pytest.raises(SeparationError) as error:
        service.separate_stems(audio, tmp_path / "stems")

    assert error.value.kind == KIND_STALLED
    assert "skip separation" in error.value.hint


def test_drafting_surfaces_the_stall_with_a_hint(tmp_path, monkeypatch):
    from app.draft_lyrics import service
    from app.draft_lyrics.contract import DraftError

    audio = write_tone_wav(tmp_path / "a.wav", seconds=0.2)
    monkeypatch.setattr(
        "app.draft_lyrics.service.resolve_backend", lambda *a, **k: ("side-venv", Path(sys.executable))
    )
    monkeypatch.setattr(
        "app.draft_lyrics.service.run_python_worker",
        lambda *a, **k: (_ for _ in ()).throw(
            AiWorkerError(KIND_STALLED, STALLED_DOWNLOAD_MESSAGE.format(stall=90))
        ),
    )

    with pytest.raises(DraftError) as error:
        service.draft_lyrics(audio, tmp_path / "work")

    assert error.value.kind == KIND_STALLED
    assert "type your own lyrics" in error.value.hint


def test_estimation_surfaces_the_stall_with_a_hint(tmp_path, monkeypatch):
    from app.transcription import TranscriptionError, estimate_notes

    audio = write_tone_wav(tmp_path / "a.wav", seconds=0.2)
    monkeypatch.setattr(
        "app.transcription.bridge.resolve_backend", lambda *a, **k: ("side-venv", Path(sys.executable))
    )
    # Patch one level below run_worker so the real bridge wraps it, which is the
    # path the app actually takes.
    monkeypatch.setattr(
        "app.transcription.bridge.run_python_worker",
        lambda *a, **k: (_ for _ in ()).throw(
            AiWorkerError(KIND_STALLED, STALLED_DOWNLOAD_MESSAGE.format(stall=90))
        ),
    )

    with pytest.raises(TranscriptionError) as error:
        estimate_notes(audio, tmp_path / "work")

    assert error.value.kind == KIND_STALLED
    assert "skip" in error.value.hint
