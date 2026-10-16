"""Milestone 4 tests: stem separation, with Demucs always mocked.

The model is never loaded. These cover the contract, the worker, error mapping
and - importantly - the promise that the user's original audio is untouched.
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

from app.separate import service
from app.separate.contract import (
    DEFAULT_MODEL,
    KNOWN_MODELS,
    SeparationError,
    SeparationResult,
    StemInfo,
    cpu_warning,
)
from app.transcription.audio_convert import write_tone_wav

WORKER_PATH = Path(__file__).resolve().parents[1] / "app" / "separate" / "worker.py"


@pytest.fixture
def worker():
    spec = importlib.util.spec_from_file_location("kmb_sep_worker", WORKER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def wav(tmp_path):
    return write_tone_wav(tmp_path / "song.wav", seconds=1.0)


def fake_demucs(stems, *, samplerate=22050, sources=None):
    """A stand-in for demucs.api that writes real, readable stem WAVs."""
    module = types.ModuleType("demucs.api")

    class FakeSeparator:
        def __init__(self, model=DEFAULT_MODEL, device="cpu", shifts=0, segment=None):
            self.samplerate = samplerate
            self.model_name = model
            self._sources = list(sources or stems)

        def separate_audio_file(self, path):
            assert Path(path).exists(), "the input must still exist when separation reads it"
            return object(), {name: object() for name in self._sources}

    def save_audio(source, dest, samplerate=None, **kwargs):
        write_tone_wav(Path(dest), seconds=0.5, rate=samplerate or 22050, frequency=440.0)

    module.Separator = FakeSeparator
    module.save_audio = save_audio
    return module


# --- contract -------------------------------------------------------------


def test_stem_knows_when_it_is_unsuitable_for_note_tracking():
    assert StemInfo("drums", "x.wav").is_suitable_for_notes is False
    assert "Drums" in StemInfo("drums", "x.wav").suitability_reason
    assert StemInfo("vocals", "x.wav").is_suitable_for_notes is True
    assert StemInfo("vocals", "x.wav").suitability_reason == ""


def test_vocals_are_the_recommended_stem_over_everything_else():
    result = SeparationResult(
        stems=[StemInfo(n, f"{n}.wav") for n in ("drums", "bass", "other", "vocals", "guitar", "piano")]
    )

    assert result.recommended_stem.name == "vocals"
    order = [s.name for s in result.suitable_stems]
    assert order.index("bass") < order.index("guitar")
    assert "drums" not in order


def test_drums_only_result_has_no_recommendation():
    result = SeparationResult(stems=[StemInfo("drums", "drums.wav")])

    assert result.suitable_stems == []
    assert result.recommended_stem is None


def test_result_summary_lists_the_stems():
    result = SeparationResult(stems=[StemInfo("vocals", "v.wav"), StemInfo("drums", "d.wav")])

    assert "2 stems" in result.summary
    assert "vocals, drums" in result.summary


def test_empty_result_says_so():
    assert "No stems" in SeparationResult().summary


def test_result_from_payload_filters_malformed_stems():
    payload = {
        "model": "htdemucs",
        "device": "cpu",
        "audio_duration": 12.0,
        "elapsed_seconds": 40.0,
        "stems": [
            {"name": "vocals", "path": "v.wav", "duration": 12.0},
            {"name": "", "path": "broken.wav"},
            {"path": "no-name.wav"},
            {"name": "bass", "path": "b.wav", "duration": "not-a-number"},
        ],
    }
    result = SeparationResult.from_dict(payload)

    assert result.stem_names == ["vocals"]


def test_result_from_payload_accepts_elapsed_alias():
    result = SeparationResult.from_dict({"stems": [], "elapsed": 7.5})

    assert result.elapsed_seconds == 7.5


def test_stem_lookup_by_name():
    result = SeparationResult(stems=[StemInfo("vocals", "v.wav"), StemInfo("bass", "b.wav")])

    assert result.stem("bass").path == "b.wav"
    assert result.stem("nope") is None


def test_cpu_warning_is_honest_and_mentions_the_audio_length():
    warning = cpu_warning("cpu", 240.0)

    assert warning
    assert "CPU" in warning
    assert "4.0 minutes" in warning


def test_no_cpu_warning_on_gpu():
    assert cpu_warning("cuda", 240.0) is None


def test_known_models_document_their_stems():
    assert "vocals" in KNOWN_MODELS[DEFAULT_MODEL]
    assert KNOWN_MODELS["mdx_extra_q"] == ["no_vocals", "vocals"]


# --- worker ---------------------------------------------------------------


def test_worker_separates_and_writes_one_file_per_stem(worker, tmp_path, wav, monkeypatch):
    fake = fake_demucs(["vocals", "drums", "bass", "other"])
    monkeypatch.setitem(sys.modules, "demucs.api", fake)

    out_dir = tmp_path / "stems"
    payload = worker.separate(wav, out_dir, model="htdemucs", device="cpu")

    assert payload["ok"] is True
    assert {s["name"] for s in payload["stems"]} == {"vocals", "drums", "bass", "other"}
    for stem in payload["stems"]:
        assert Path(stem["path"]).exists(), "each stem must be a real file"
        assert out_dir.resolve() in Path(stem["path"]).resolve().parents


def test_worker_never_writes_to_the_input_file(worker, tmp_path, wav, monkeypatch):
    """The original audio must be untouched: same bytes, same place."""
    before = wav.read_bytes()
    fake = fake_demucs(["vocals", "drums"])
    monkeypatch.setitem(sys.modules, "demucs.api", fake)

    out_dir = tmp_path / "stems"
    worker.separate(wav, out_dir, device="cpu")

    assert wav.exists()
    assert wav.read_bytes() == before
    assert not wav.with_suffix(".other").exists()
    assert sorted(p.name for p in wav.parent.iterdir()) == sorted([wav.name, "stems"])


def test_worker_reports_a_dependency_error_without_demucs(worker, tmp_path, wav, monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("demucs"):
            raise ImportError("No module named 'demucs'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    monkeypatch.delitem(sys.modules, "demucs.api", raising=False)

    payload = worker.separate(wav, tmp_path / "stems")

    assert payload["ok"] is False
    assert payload["kind"] == "dependency"
    assert "demucs" in payload["message"]


def test_worker_reports_a_model_error_when_loading_fails(worker, tmp_path, wav, monkeypatch):
    module = types.ModuleType("demucs.api")

    class Boom:
        def __init__(self, **kwargs):
            raise RuntimeError("model weights unavailable")

    module.Separator = Boom
    module.save_audio = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "demucs.api", module)

    payload = worker.separate(wav, tmp_path / "stems")

    assert payload["ok"] is False
    assert payload["kind"] == "model"
    assert "model weights unavailable" in payload["message"]


def test_worker_reports_a_conversion_error_when_audio_cannot_be_read(worker, tmp_path, wav, monkeypatch):
    class Separator:
        def __init__(self, **kwargs):
            self.samplerate = 22050

        def separate_audio_file(self, path):
            raise FileNotFoundError("FFmpeg is not installed.")

    module = types.ModuleType("demucs.api")
    module.Separator = Separator
    module.save_audio = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "demucs.api", module)

    payload = worker.separate(wav, tmp_path / "stems")

    assert payload["ok"] is False
    assert payload["kind"] == "conversion"
    assert "FFmpeg" in payload["message"]


def test_worker_always_carries_the_separation_caveat(worker, tmp_path, wav, monkeypatch):
    monkeypatch.setitem(sys.modules, "demucs.api", fake_demucs(["vocals"]))

    payload = worker.separate(wav, tmp_path / "stems", device="cpu")

    assert any("NOT clean isolated" in w for w in payload["warnings"])


def test_worker_adds_a_cpu_timing_warning_on_cpu(worker, tmp_path, wav, monkeypatch):
    monkeypatch.setitem(sys.modules, "demucs.api", fake_demucs(["vocals"]))

    payload = worker.separate(wav, tmp_path / "stems", device="cpu")

    assert any("CPU" in w for w in payload["warnings"])


def test_worker_omits_the_cpu_warning_on_gpu(worker, tmp_path, wav, monkeypatch):
    monkeypatch.setitem(sys.modules, "demucs.api", fake_demucs(["vocals"]))

    payload = worker.separate(wav, tmp_path / "stems", device="cuda")

    assert not any("Run on the CPU" in w for w in payload["warnings"])


def test_worker_main_writes_json_and_uses_the_output_dir(worker, tmp_path, wav, monkeypatch):
    monkeypatch.setitem(sys.modules, "demucs.api", fake_demucs(["vocals", "bass"]))

    out_dir = tmp_path / "stems"
    result_file = tmp_path / "result.json"
    code = worker.main([
        "--input", str(wav), "--output-dir", str(out_dir),
        "--output", str(result_file), "--model", "htdemucs",
    ])

    assert code == worker.EXIT_OK
    payload = json.loads(result_file.read_text())
    assert len(payload["stems"]) == 2
    assert out_dir.exists()


def test_worker_main_rejects_a_missing_input(worker, tmp_path):
    code = worker.main([
        "--input", str(tmp_path / "nope.wav"),
        "--output-dir", str(tmp_path / "s"),
        "--output", str(tmp_path / "r.json"),
    ])

    assert code != worker.EXIT_OK


def test_worker_file_does_not_shadow_the_standard_library():
    app_root = Path(__file__).resolve().parents[1] / "app"
    reserved = set(sys.stdlib_module_names)

    offenders = [
        path.name
        for path in app_root.rglob("*.py")
        if path.name != "__init__.py" and path.stem in reserved
    ]
    assert not offenders, f"these module names shadow the stdlib: {offenders}"


def test_separation_worker_runs_as_a_real_subprocess(tmp_path):
    """Run it exactly as the bridge does; demucs is absent, so expect a clean
    dependency error rather than a stdlib ImportError."""
    wav = write_tone_wav(tmp_path / "tone.wav", seconds=0.2)
    out_dir = tmp_path / "stems"
    result_file = tmp_path / "result.json"

    completed = subprocess.run(
        [sys.executable, str(WORKER_PATH), "--input", str(wav),
         "--output-dir", str(out_dir), "--output", str(result_file)],
        capture_output=True, text=True,
        cwd=str(WORKER_PATH.parents[2]), timeout=180,
    )

    assert result_file.exists(), f"no output; stderr:\n{completed.stderr}"
    payload = json.loads(result_file.read_text())
    assert payload["ok"] is False
    assert payload["kind"] == "dependency"
    assert "GenericAlias" not in completed.stderr


# --- service --------------------------------------------------------------


def test_availability_false_without_demucs(tmp_path, monkeypatch):
    monkeypatch.setattr("app.separate.service.demucs_available", lambda *a, **k: False)

    availability = service.check_availability(tmp_path)

    assert not availability.available
    assert availability.backend == "none"
    assert "not installed" in availability.reason


def test_missing_dependency_raises_before_running_anything(tmp_path, wav, monkeypatch):
    monkeypatch.setattr("app.separate.service.resolve_backend", lambda *a, **k: ("none", None))

    with pytest.raises(SeparationError) as error:
        service.separate_stems(wav, tmp_path / "stems")

    assert error.value.kind == "dependency"
    assert error.value.hint


def test_missing_source_file_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr("app.separate.service.resolve_backend", lambda *a, **k: ("side-venv", Path(sys.executable)))

    with pytest.raises(SeparationError) as error:
        service.separate_stems(tmp_path / "nope.wav", tmp_path / "stems")

    assert error.value.kind == "conversion"


def test_output_folder_may_not_be_the_source_folder(tmp_path, wav, monkeypatch):
    """Writing stems beside the original would risk clobbering user files."""
    monkeypatch.setattr("app.separate.service.resolve_backend", lambda *a, **k: ("side-venv", Path(sys.executable)))

    with pytest.raises(SeparationError) as error:
        service.separate_stems(wav, wav.parent)

    assert "same folder" in str(error.value)


def test_processing_failure_is_reported_with_its_kind(tmp_path, wav, monkeypatch):
    from app.ai_process import AiWorkerError

    monkeypatch.setattr("app.separate.service.resolve_backend", lambda *a, **k: ("side-venv", Path(sys.executable)))

    def boom(*args, **kwargs):
        raise AiWorkerError("model", "model download failed")

    monkeypatch.setattr("app.separate.service.run_python_worker", boom)

    with pytest.raises(SeparationError) as error:
        service.separate_stems(wav, tmp_path / "stems")

    assert error.value.kind == "model"
    assert "download" in str(error.value)


def test_empty_separation_result_is_an_error(tmp_path, wav, monkeypatch):
    monkeypatch.setattr("app.separate.service.resolve_backend", lambda *a, **k: ("side-venv", Path(sys.executable)))
    monkeypatch.setattr("app.separate.service.run_python_worker", lambda *a, **k: {"ok": True, "stems": []})

    with pytest.raises(SeparationError) as error:
        service.separate_stems(wav, tmp_path / "stems")

    assert "no stems" in str(error.value)


def test_separation_preserves_the_original_file_end_to_end(tmp_path, wav, monkeypatch):
    """The whole promise, through the service with a fake worker."""
    before = wav.read_bytes()
    monkeypatch.setattr("app.separate.service.resolve_backend", lambda *a, **k: ("side-venv", Path(sys.executable)))

    def fake_run(worker, interpreter, input_path, *, output_path=None, extra_args=None, **kwargs):
        out_dir = Path(extra_args["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        stems = []
        for name in ("vocals", "drums", "bass"):
            target = out_dir / f"{name}.wav"
            write_tone_wav(target, seconds=0.4)
            stems.append({"name": name, "path": str(target), "duration": 0.4})
        output_path.write_text(json.dumps({
            "ok": True, "model": "htdemucs", "device": "cpu",
            "audio_duration": 1.0, "stems": stems,
        }))
        return json.loads(output_path.read_text())

    monkeypatch.setattr("app.separate.service.run_python_worker", fake_run)

    result = service.separate_stems(wav, tmp_path / "stems")

    assert len(result) == 3
    assert result.recommended_stem.name == "vocals"
    assert wav.exists(), "the original must survive"
    assert wav.read_bytes() == before


def test_unknown_model_is_flagged_but_not_fatal(tmp_path, wav, monkeypatch):
    monkeypatch.setattr("app.separate.service.resolve_backend", lambda *a, **k: ("side-venv", Path(sys.executable)))
    monkeypatch.setattr("app.separate.service.run_python_worker", lambda *a, **k: {
        "ok": True, "model": "made-up", "device": "cpu", "audio_duration": 1.0,
        "stems": [{"name": "vocals", "path": "v.wav", "duration": 1.0}],
    })

    result = service.separate_stems(wav, tmp_path / "stems", model="made-up")

    assert any("made-up" in w for w in result.warnings)


def test_stale_stems_are_cleared_before_a_new_run(tmp_path, wav, monkeypatch):
    monkeypatch.setattr("app.separate.service.resolve_backend", lambda *a, **k: ("side-venv", Path(sys.executable)))
    out_dir = tmp_path / "stems"
    out_dir.mkdir()
    stale = out_dir / "vocals.wav"
    stale.write_text("old content")

    def fake_run(worker, interpreter, input_path, *, output_path=None, extra_args=None, **kwargs):
        output_path.write_text(json.dumps({
            "ok": True, "model": "htdemucs", "device": "cpu", "audio_duration": 1.0,
            "stems": [{"name": "vocals", "path": "v.wav", "duration": 1.0}],
        }))
        return json.loads(output_path.read_text())

    monkeypatch.setattr("app.separate.service.run_python_worker", fake_run)
    service.separate_stems(wav, out_dir)

    assert not stale.exists(), "a previous run's stems must not be left to mislead"
