"""Milestone 3 tests for the worker itself, with the model mocked.

`basic_pitch` is injected into `sys.modules` as a fake, so the worker's parsing
and error handling are exercised without TensorFlow or a model download.
"""

from __future__ import annotations

import builtins
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from app.transcription.audio_convert import write_tone_wav

WORKER_PATH = Path(__file__).resolve().parents[1] / "app" / "transcription" / "worker.py"


@pytest.fixture
def worker():
    """Load worker.py as a module, independently of the app package."""
    spec = importlib.util.spec_from_file_location("kmb_worker", WORKER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def wav(tmp_path):
    return write_tone_wav(tmp_path / "tone.wav", seconds=1.0)


# --- the basic_pitch 0.4.0 return shape ----------------------------------


def test_worker_parses_the_three_value_return(worker, wav, monkeypatch):
    """predict() returns (model_output, midi_data, note_events) in 0.4.0."""
    calls = {}

    def fake_predict(audio_path, **kwargs):
        calls["path"] = str(audio_path)
        calls["kwargs"] = kwargs
        return (
            {"note_onset": [1]},  # model_output
            object(),  # midi_data
            [(0.0, 0.5, 60, 0.9, None), (0.5, 1.0, 62, 0.8, None)],
        )

    fake = types.ModuleType("basic_pitch.inference")
    fake.predict = fake_predict
    monkeypatch.setitem(sys.modules, "basic_pitch.inference", fake)
    monkeypatch.setitem(sys.modules, "basic_pitch", types.ModuleType("basic_pitch"))

    payload = worker.transcribe(wav, onset_threshold=0.7, minimum_frequency=100.0)

    assert payload["ok"] is True
    assert payload["model"] == "basic_pitch:ICASSP_2022"
    assert payload["sample_rate"] == 22050
    assert len(payload["notes"]) == 2
    assert payload["notes"][0] == {"start": 0.0, "end": 0.5, "pitch": 60, "amplitude": 0.9, "label": ""}
    assert calls["kwargs"]["onset_threshold"] == 0.7
    assert calls["kwargs"]["minimum_frequency"] == 100.0


def test_worker_reports_audio_duration(worker, wav, monkeypatch):
    fake = types.ModuleType("basic_pitch.inference")
    fake.predict = lambda p, **k: ({}, object(), [])
    monkeypatch.setitem(sys.modules, "basic_pitch.inference", fake)

    payload = worker.transcribe(wav)

    assert payload["audio_duration"] == pytest.approx(1.0, abs=0.01)


def test_worker_always_carries_the_approximation_warning(worker, wav, monkeypatch):
    fake = types.ModuleType("basic_pitch.inference")
    fake.predict = lambda p, **k: ({}, object(), [(0.0, 1.0, 60, 1.0, None)])
    monkeypatch.setitem(sys.modules, "basic_pitch.inference", fake)

    payload = worker.transcribe(wav)

    assert payload["warnings"]
    assert "approximation" in payload["warnings"][0]


# --- error handling -------------------------------------------------------


def test_worker_reports_a_dependency_error_when_basic_pitch_is_missing(worker, wav, monkeypatch):
    """ImportError must become a helpful message, not a traceback."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("basic_pitch"):
            raise ImportError("No module named 'basic_pitch'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    monkeypatch.delitem(sys.modules, "basic_pitch.inference", raising=False)

    payload = worker.transcribe(wav)

    assert payload["ok"] is False
    assert payload["kind"] == "dependency"
    assert "basic_pitch" in payload["message"]


def test_worker_reports_a_model_error_when_inference_raises(worker, wav, monkeypatch):
    def boom(path, **kwargs):
        raise RuntimeError("model weights not found")

    fake = types.ModuleType("basic_pitch.inference")
    fake.predict = boom
    monkeypatch.setitem(sys.modules, "basic_pitch.inference", fake)

    payload = worker.transcribe(wav)

    assert payload["ok"] is False
    assert payload["kind"] == "model"
    assert "model weights not found" in payload["message"]


def test_worker_retries_with_positional_only_on_signature_mismatch(worker, wav, monkeypatch):
    """Some versions reject the keyword arguments; the path alone still works."""
    attempts = []

    def picky(path, **kwargs):
        attempts.append(kwargs)
        if kwargs:
            raise TypeError("unexpected keyword argument 'onset_threshold'")
        return ({}, object(), [(0.0, 1.0, 60, 1.0, None)])

    fake = types.ModuleType("basic_pitch.inference")
    fake.predict = picky
    monkeypatch.setitem(sys.modules, "basic_pitch.inference", fake)

    payload = worker.transcribe(wav)

    assert payload["ok"] is True
    assert len(attempts) == 2


def test_worker_survives_a_model_returning_no_notes(worker, wav, monkeypatch):
    fake = types.ModuleType("basic_pitch.inference")
    fake.predict = lambda p, **k: ({}, object(), [])
    monkeypatch.setitem(sys.modules, "basic_pitch.inference", fake)

    payload = worker.transcribe(wav)

    assert payload["ok"] is True
    assert payload["notes"] == []


# --- command line ---------------------------------------------------------


def test_worker_main_writes_json_to_the_output_file(worker, tmp_path, wav, monkeypatch):
    fake = types.ModuleType("basic_pitch.inference")
    fake.predict = lambda p, **k: ({}, object(), [(0.25, 0.75, 64, 0.5, None)])
    monkeypatch.setitem(sys.modules, "basic_pitch.inference", fake)

    out = tmp_path / "nested" / "notes.json"
    code = worker.main(["--input", str(wav), "--output", str(out)])

    assert code == worker.EXIT_OK
    payload = json.loads(out.read_text())
    assert payload["notes"][0]["pitch"] == 64


def test_worker_main_emits_a_marker_line_on_stdout(worker, tmp_path, wav, monkeypatch, capsys):
    fake = types.ModuleType("basic_pitch.inference")
    fake.predict = lambda p, **k: ({}, object(), [(0.0, 1.0, 60, 1.0, None)])
    monkeypatch.setitem(sys.modules, "basic_pitch.inference", fake)

    worker.main(["--input", str(wav), "--output", str(tmp_path / "n.json")])
    printed = capsys.readouterr().out

    assert "__KMB_JSON__" in printed


def test_worker_main_reports_a_missing_input_file(worker, tmp_path):
    code = worker.main(["--input", str(tmp_path / "nope.wav"), "--output", str(tmp_path / "n.json")])

    assert code != worker.EXIT_OK


def test_worker_main_returns_dependency_exit_code(worker, tmp_path, wav, monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("basic_pitch"):
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    monkeypatch.delitem(sys.modules, "basic_pitch.inference", raising=False)

    code = worker.main(["--input", str(wav), "--output", str(tmp_path / "n.json")])

    assert code == worker.EXIT_DEPENDENCY


def test_worker_output_parses_back_into_a_result(worker, tmp_path, wav, monkeypatch):
    """The JSON contract holds end to end: worker output -> TranscriptionResult."""
    from app.transcription.contract import TranscriptionResult

    fake = types.ModuleType("basic_pitch.inference")
    fake.predict = lambda p, **k: ({}, object(), [(0.0, 0.5, 60, 0.9, None), (0.5, 1.0, 72, 0.4, None)])
    monkeypatch.setitem(sys.modules, "basic_pitch.inference", fake)

    out = tmp_path / "notes.json"
    worker.main(["--input", str(wav), "--output", str(out)])
    result = TranscriptionResult.from_dict(json.loads(out.read_text()))

    assert len(result) == 2
    assert [n.pitch for n in result.estimates] == [60, 72]
    assert result.estimates[0].amplitude == pytest.approx(0.9)


def test_worker_file_is_standalone_and_uses_only_stdlib_at_import():
    """It must run under a different interpreter, so no app imports at module level."""
    source = WORKER_PATH.read_text(encoding="utf-8")
    module_level = [
        line for line in source.splitlines()
        if line.startswith(("import ", "from "))
    ]

    for line in module_level:
        assert not line.startswith("from app"), f"worker must not import the app package: {line}"
    assert "from basic_pitch" in source, "basic_pitch must only be imported lazily inside the function"


def test_no_module_under_app_shadows_the_standard_library():
    """Regression guard for a bug that only shows up when the script RUNS.

    Running a script puts its own directory first on sys.path. A file named
    `types.py`, `json.py`, `enum.py` or similar beside the worker therefore
    shadows the stdlib module of that name and breaks the worker before it does
    any work - which is exactly what happened with a `types.py` here.
    """
    app_root = Path(__file__).resolve().parents[1] / "app"
    reserved = set(sys.stdlib_module_names)

    offenders = [
        path.relative_to(app_root)
        for path in app_root.rglob("*.py")
        if path.name != "__init__.py" and path.stem in reserved
    ]
    assert not offenders, f"these module names shadow the stdlib: {offenders}"


def test_worker_runs_as_a_real_subprocess_the_way_the_bridge_does(tmp_path, worker):
    """Execute worker.py exactly as bridge.run_worker does.

    Basic Pitch is not installed here, so the run must fail as a clean
    *dependency* error. A stdlib-shadowing crash instead surfaces as a stray
    ImportError from argparse or enum, which is what this asserts against.
    """
    wav = write_tone_wav(tmp_path / "tone.wav", seconds=0.2)
    output = tmp_path / "notes.json"

    completed = subprocess.run(
        [sys.executable, str(WORKER_PATH), "--input", str(wav), "--output", str(output)],
        capture_output=True,
        text=True,
        cwd=str(WORKER_PATH.parents[2]),
        timeout=180,
    )

    assert output.exists(), f"worker produced no output; stderr:\n{completed.stderr}"
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["ok"] is False
    assert payload["kind"] == "dependency", f"expected a clean dependency error, got: {payload}"
    assert "GenericAlias" not in completed.stderr
    assert completed.returncode == worker.EXIT_DEPENDENCY