"""Tests for phase-aware timeouts.

A blocked model download and genuinely slow processing are indistinguishable
from the outside: both are a silent worker until the timeout fires. Reporting
"this is slow on a CPU" for a download that will never finish sends people off
to tune the wrong thing, so the workers record their stage and the bridge uses
it to name the real cause.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.ai_process import (
    PHASE_DOWNLOADING,
    PHASE_FALLBACK_MESSAGE,
    PHASE_PROCESSING,
    PHASE_TIMEOUT_MESSAGES,
    AiWorkerError,
    phase_file,
    read_phase,
    run_python_worker,
)
from app.draft_lyrics.contract import DraftError
from app.separate.contract import SeparationError
from app.transcription.audio_convert import write_tone_wav

ROOT = Path(__file__).resolve().parents[1]
SEP_WORKER = ROOT / "app" / "separate" / "worker.py"
DRAFT_WORKER = ROOT / "app" / "draft_lyrics" / "worker.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def wav(tmp_path):
    return write_tone_wav(tmp_path / "song.wav", seconds=1.0)


def _timeout_worker(message: str):
    """A fake worker that sleeps, so run_python_worker times out on it."""
    return (
        "import sys, time, pathlib\n"
        f"out = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])\n"
        f"phase = pathlib.Path(str(out) + '.phase')\n"
        + message
        + "\ntime.sleep(120)\n"
    )


# --- messages -------------------------------------------------------------


def test_download_stall_is_not_reported_as_slow_processing():
    message = PHASE_TIMEOUT_MESSAGES[PHASE_DOWNLOADING].format(timeout=3600)

    assert "Downloading the model" in message
    assert "stalled" in message
    assert "not slow processing" in message


def test_slow_processing_is_reported_as_slow_processing():
    message = PHASE_TIMEOUT_MESSAGES[PHASE_PROCESSING].format(timeout=3600)

    assert "Still working" in message
    assert "expected on a CPU" in message
    assert "stalled" not in message


def test_unknown_phase_falls_back_to_saying_so():
    message = PHASE_FALLBACK_MESSAGE.format(timeout=60)

    assert "did not report which stage" in message
    assert "downloading a model or still processing" in message


def test_phase_file_round_trips(tmp_path):
    output = tmp_path / "result.json"
    assert read_phase(output) == ""

    phase_file(output).write_text(PHASE_DOWNLOADING, encoding="utf-8")
    assert read_phase(output) == PHASE_DOWNLOADING


# --- the bridge uses the phase --------------------------------------------


def test_timeout_names_a_stalled_download(tmp_path, monkeypatch):
    script = _timeout_worker("phase.write_text('downloading-model')")
    worker = tmp_path / "w.py"
    worker.write_text(script)
    audio = write_tone_wav(tmp_path / "a.wav", seconds=0.2)
    out = tmp_path / "out.json"

    with pytest.raises(AiWorkerError) as error:
        run_python_worker(worker, Path(sys.executable), audio, output_path=out, timeout=5)

    assert error.value.kind == "timeout"
    assert "Downloading the model" in str(error.value)
    assert "stalled" in str(error.value)


def test_timeout_names_slow_processing(tmp_path):
    script = _timeout_worker("phase.write_text('processing')")
    worker = tmp_path / "w.py"
    worker.write_text(script)
    audio = write_tone_wav(tmp_path / "a.wav", seconds=0.2)
    out = tmp_path / "out.json"

    with pytest.raises(AiWorkerError) as error:
        run_python_worker(worker, Path(sys.executable), audio, output_path=out, timeout=5)

    assert "Still working" in str(error.value)
    assert "expected on a CPU" in str(error.value)


def test_timeout_with_no_phase_reports_the_uncertainty(tmp_path):
    script = _timeout_worker("pass")
    worker = tmp_path / "w.py"
    worker.write_text(script)
    audio = write_tone_wav(tmp_path / "a.wav", seconds=0.2)
    out = tmp_path / "out.json"

    with pytest.raises(AiWorkerError) as error:
        run_python_worker(worker, Path(sys.executable), audio, output_path=out, timeout=5)

    assert "did not report which stage" in str(error.value)


# --- the services surface it ----------------------------------------------


def test_separation_timeout_carries_the_download_message(tmp_path, wav, monkeypatch):
    from app.separate import service

    def fake_run(worker, interpreter, input_path, *, output_path=None, extra_args=None, **kwargs):
        phase_file(output_path).write_text(PHASE_DOWNLOADING, encoding="utf-8")
        raise AiWorkerError(
            "timeout",
            PHASE_TIMEOUT_MESSAGES[PHASE_DOWNLOADING].format(timeout=kwargs.get("timeout", 3600)),
        )

    monkeypatch.setattr("app.separate.service.resolve_backend", lambda *a, **k: ("side-venv", Path(sys.executable)))
    monkeypatch.setattr("app.separate.service.run_python_worker", fake_run)

    with pytest.raises(SeparationError) as error:
        service.separate_stems(wav, tmp_path / "stems")

    assert "stalled" in str(error.value)
    assert "internet" in error.value.hint.lower()


def test_draft_timeout_carries_the_download_message(tmp_path, wav, monkeypatch):
    from app.draft_lyrics import service

    def fake_run(worker, interpreter, input_path, *, output_path=None, **kwargs):
        raise AiWorkerError(
            "timeout",
            PHASE_TIMEOUT_MESSAGES[PHASE_DOWNLOADING].format(timeout=kwargs.get("timeout", 1800)),
        )

    monkeypatch.setattr("app.draft_lyrics.service.resolve_backend", lambda *a, **k: ("side-venv", Path(sys.executable)))
    monkeypatch.setattr("app.draft_lyrics.service.run_python_worker", fake_run)

    with pytest.raises(DraftError) as error:
        service.draft_lyrics(wav, tmp_path / "work")

    assert "stalled" in str(error.value)


# --- the workers really do record their stage -----------------------------


def test_separation_worker_records_the_downloading_phase(tmp_path, wav, monkeypatch):
    worker = _load(SEP_WORKER, "sep_phase_worker")
    output = tmp_path / "result.json"

    def blocked(model="htdemucs", device="cpu", shifts=0, segment=None):
        raise SystemExit("stalled while fetching weights")

    import types as _types

    module = _types.ModuleType("demucs.api")
    module.Separator = blocked
    module.save_audio = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "demucs.api", module)

    with pytest.raises(SystemExit):
        worker.separate(wav, tmp_path / "stems", output_path=output)

    assert read_phase(output) == PHASE_DOWNLOADING


def test_separation_worker_moves_to_processing_once_loaded(tmp_path, wav, monkeypatch):
    worker = _load(SEP_WORKER, "sep_phase_worker2")
    output = tmp_path / "result.json"

    def fake_demucs(stems):
        import types as _types

        class Separator:
            def __init__(self, model="htdemucs", device="cpu", shifts=0, segment=None):
                self.samplerate = 22050

            def separate_audio_file(self, path):
                output_phase = output.with_suffix(output.suffix + ".processing")
                return object(), {name: object() for name in stems}

        module = _types.ModuleType("demucs.api")
        module.Separator = Separator
        module.save_audio = lambda src, dest, samplerate=None, **k: write_tone_wav(
            Path(dest), seconds=0.3, rate=samplerate or 22050
        )
        return module

    monkeypatch.setitem(sys.modules, "demucs.api", fake_demucs(["vocals"]))
    worker.separate(wav, tmp_path / "stems", device="cpu", output_path=output)

    assert read_phase(output) == PHASE_PROCESSING


def test_draft_worker_records_the_downloading_phase(tmp_path, wav, monkeypatch):
    worker = _load(DRAFT_WORKER, "draft_phase_worker")
    output = tmp_path / "draft.json"

    import types as _types

    module = _types.ModuleType("faster_whisper")

    def blocked(*a, **k):
        raise SystemExit("stalled while fetching weights")

    module.WhisperModel = blocked
    monkeypatch.setitem(sys.modules, "faster_whisper", module)

    with pytest.raises(SystemExit):
        worker.draft(wav, output_path=output)

    assert read_phase(output) == PHASE_DOWNLOADING


def test_draft_worker_moves_to_processing_once_loaded(tmp_path, wav, monkeypatch):
    worker = _load(DRAFT_WORKER, "draft_phase_worker2")
    output = tmp_path / "draft.json"

    import types as _types

    module = _types.ModuleType("faster_whisper")

    class Model:
        def __init__(self, *a, **k):
            pass

        def transcribe(self, path, **kwargs):
            return iter([]), _types.SimpleNamespace(language="en")

    module.WhisperModel = Model
    monkeypatch.setitem(sys.modules, "faster_whisper", module)

    worker.draft(wav, output_path=output)

    assert read_phase(output) == PHASE_PROCESSING
