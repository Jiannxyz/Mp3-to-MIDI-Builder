"""Tests that availability probing and error messages match reality.

The bug this file exists for: a shallow `import basic_pitch` succeeds while
`basic_pitch.inference` fails, because the package __init__ only probes optional
extras and the real entry point pulls in resampy -> pkg_resources. The app said
"available", then every run failed, and the message blamed basic_pitch when
basic_pitch was installed fine.
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

from app.ai_process import module_available
from app.draft_lyrics import service as draft_service
from app.separate import service as separate_service

ROOT = Path(__file__).resolve().parents[1]
WORKERS = {
    "basic_pitch": ROOT / "app" / "transcription" / "worker.py",
    "demucs": ROOT / "app" / "separate" / "worker.py",
    "faster_whisper": ROOT / "app" / "draft_lyrics" / "worker.py",
}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _break_import(monkeypatch, module_to_break: str, missing_name: str):
    """Make `import module_to_break` fail as if a dependency were absent."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == module_to_break:
            raise ModuleNotFoundError(f"No module named '{missing_name}'", name=missing_name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


# --- probing --------------------------------------------------------------


def test_probe_requires_the_symbol_the_worker_uses():
    """A package can import fine while the submodule the worker needs is broken."""
    package = ROOT / "app" / "probe_fixture"
    package.mkdir(exist_ok=True)
    (package / "__init__.py").write_text("VERSION = 1\n")
    (package / "broken.py").write_text("import a_module_that_does_not_exist\n")
    (package / "fine.py").write_text("class Thing:\n    pass\n")
    (package / "fixtures.txt").write_text("", encoding="utf-8")

    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)

    shallow = subprocess.run(
        [sys.executable, "-c", "import app.probe_fixture"], capture_output=True, text=True, env=env
    )
    assert shallow.returncode == 0, "precondition: the package __init__ must import cleanly"

    assert module_available(Path(sys.executable), "app.probe_fixture", "Thing") is False
    assert module_available(Path(sys.executable), "app.probe_fixture.fine", "Thing") is True


def test_probe_reports_false_for_a_missing_interpreter(tmp_path):
    assert module_available(tmp_path / "nope.exe", "json", "loads") is False


def test_each_service_probes_its_real_entry_point():
    import inspect

    from app.draft_lyrics import bridge as draft_bridge
    from app.separate import bridge as separate_bridge
    from app.transcription import bridge as transcription_bridge

    assert "basic_pitch.inference" in inspect.getsource(transcription_bridge.basic_pitch_available)
    assert "predict" in inspect.getsource(transcription_bridge.basic_pitch_available)
    assert "demucs.api" in inspect.getsource(separate_bridge.demucs_available)
    assert "Separator" in inspect.getsource(separate_service.demucs_available)
    assert "WhisperModel" in inspect.getsource(draft_bridge.whisper_available)


# --- error classification -------------------------------------------------


def test_missing_package_is_reported_as_not_installed(monkeypatch, tmp_path):
    worker = _load(WORKERS["basic_pitch"], "w_missing")
    _break_import(monkeypatch, "basic_pitch.inference", "basic_pitch")

    payload = worker.transcribe(tmp_path / "a.wav")

    assert payload["ok"] is False
    assert "basic_pitch is not installed" in payload["message"]


def test_broken_dependency_is_not_blamed_on_the_package(monkeypatch, tmp_path):
    """The exact user report: basic_pitch fine, resampy/pkg_resources broken."""
    worker = _load(WORKERS["basic_pitch"], "w_broken")
    _break_import(monkeypatch, "basic_pitch.inference", "pkg_resources")

    payload = worker.transcribe(tmp_path / "a.wav")

    assert payload["ok"] is False
    assert payload["kind"] == "dependency"
    assert "basic_pitch is installed" in payload["message"], "must not blame basic_pitch"
    assert "pkg_resources" in payload["message"], "must name the real culprit"
    assert "requirements-ai.txt" in payload["message"]


def test_separation_names_its_broken_dependency(monkeypatch, tmp_path):
    worker = _load(WORKERS["demucs"], "w_demucs")
    _break_import(monkeypatch, "demucs.api", "torch")

    payload = worker.separate(tmp_path / "a.wav", tmp_path / "stems")

    assert "demucs is installed" in payload["message"]
    assert "torch" in payload["message"]


def test_drafting_names_its_broken_dependency(monkeypatch, tmp_path):
    worker = _load(WORKERS["faster_whisper"], "w_draft")
    _break_import(monkeypatch, "faster_whisper", "ctranslate2")

    payload = worker.draft(tmp_path / "a.wav")

    assert "faster_whisper is installed" in payload["message"]
    assert "ctranslate2" in payload["message"]


# --- the three workers must not drift -------------------------------------


def test_all_three_workers_classify_import_errors_identically():
    """They duplicate the classifier so they stay standalone; pin the behaviour."""
    workers = {name: _load(path, f"drift_{name}") for name, path in WORKERS.items()}
    classifiers = {name: w.import_error_message for name, w in workers.items()}

    missing = ModuleNotFoundError("No module named 'pkg_resources'", name="pkg_resources")
    anonymous = ImportError("something went wrong")

    for package, classifier in classifiers.items():
        assert "is installed, but importing it failed" in classifier(package, missing)

        # Each worker must recognise ITS OWN package as the missing one.
        own_missing = ModuleNotFoundError(f"No module named '{package}'", name=package)
        assert "is not installed" in classifier(package, own_missing)

        # A different package counts as a broken dependency, not a missing one.
        other = ModuleNotFoundError("No module named 'something_else'", name="something_else")
        assert "is installed, but importing it failed" in classifier(package, other)

        # An ImportError with no .name falls back to the "not installed" branch.
        assert "is not installed" in classifier(package, anonymous)


def test_workers_still_have_no_app_imports_at_module_level():
    """Standalone workers are what let a different interpreter run them."""
    for path in WORKERS.values():
        source = path.read_text(encoding="utf-8")
        for line in source.splitlines():
            if line.startswith(("import ", "from ")):
                assert not line.startswith("from app"), f"{path.name}: {line}"


# --- it reaches the user --------------------------------------------------


def test_a_broken_dependency_message_survives_to_the_ui_layer(monkeypatch, tmp_path):
    from app.transcription import TranscriptionError, estimate_notes
    from app.transcription.audio_convert import write_tone_wav

    audio = write_tone_wav(tmp_path / "a.wav", seconds=0.2)
    monkeypatch.setattr(
        "app.transcription.bridge.resolve_backend", lambda *a, **k: ("side-venv", Path(sys.executable))
    )

    def boom(*args, **kwargs):
        raise TranscriptionError(
            "basic_pitch is installed, but importing it failed because 'pkg_resources' "
            "could not be imported. Run: pip install -r requirements-ai.txt",
            "dependency",
        )

    monkeypatch.setattr("app.transcription.bridge.run_worker", boom)

    with pytest.raises(TranscriptionError) as error:
        estimate_notes(audio, tmp_path / "work")

    assert "pkg_resources" in str(error.value)
    assert "requirements-ai.txt" in str(error.value)
    assert error.value.hint, "a dependency failure must still carry a hint"
