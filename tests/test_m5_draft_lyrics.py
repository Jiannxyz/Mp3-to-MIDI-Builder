"""Milestone 5 tests: lyric drafting, with speech recognition always mocked.

Covers the contract, the worker, error mapping, and - the important one - that
the manual lyric path keeps working when the model is missing or fails.
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

from app.draft_lyrics import service
from app.draft_lyrics.contract import (
    DEFAULT_MODEL,
    DraftError,
    DraftLine,
    DraftWord,
    LyricDraft,
)
from app.timing import LyricRow, rows_from_text, validate_rows
from app.transcription.audio_convert import write_tone_wav

WORKER_PATH = Path(__file__).resolve().parents[1] / "app" / "draft_lyrics" / "worker.py"


@pytest.fixture
def worker():
    spec = importlib.util.spec_from_file_location("kmb_draft_worker", WORKER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def wav(tmp_path):
    return write_tone_wav(tmp_path / "song.wav", seconds=1.0)


def fake_whisper(lines=None, *, language="en", fail_on_transcribe=None):
    module = types.ModuleType("faster_whisper")

    class FakeWhisper:
        def __init__(self, name, device="cpu", compute_type="int8"):
            self.name = name

        def transcribe(self, path, **kwargs):
            if fail_on_transcribe:
                raise RuntimeError(fail_on_transcribe)
            segments = []
            for line in lines or [{"words": [
                {"text": "hello", "start": 0.0, "end": 0.4, "probability": 0.9},
                {"text": "world", "start": 0.4, "end": 0.9, "probability": 0.8},
            ]}]:
                class Segment:
                    def __init__(self, words):
                        self.words = [
                            types.SimpleNamespace(
                                word=w["text"], start=w["start"], end=w["end"], probability=w["probability"]
                            )
                            for w in words
                        ]
                segments.append(Segment(line["words"]))
            return iter(segments), types.SimpleNamespace(language=language)

    module.WhisperModel = FakeWhisper
    return module


# --- contract -------------------------------------------------------------


def test_word_rejects_nonsense_timings():
    assert DraftWord("a", 0.0, 0.5).is_valid
    assert not DraftWord("a", -1.0, 0.5).is_valid
    assert not DraftWord("a", 1.0, 1.0).is_valid
    assert not DraftWord("a", 2.0, 1.0).is_valid
    assert not DraftWord("   ", 0.0, 0.5).is_valid
    assert not DraftWord("a", float("nan"), 0.5).is_valid


def test_draft_text_is_pasteable_lyrics():
    draft = LyricDraft(lines=[
        DraftLine([DraftWord("hello", 0.0, 0.4), DraftWord("world", 0.4, 0.9)]),
        DraftLine([DraftWord("again", 1.0, 1.5)]),
    ])

    assert draft.text == "hello world\nagain"
    assert draft.word_count == 3
    assert "3 words in 2 lines" in draft.summary


def test_draft_timings_are_always_labelled_as_estimates():
    assert LyricDraft().timings_are_estimates is True
    assert LyricDraft(lines=[DraftLine([DraftWord("a", 0.0, 1.0)])]).timings_are_estimates is True


def test_empty_draft_says_so():
    assert "No speech" in LyricDraft().summary


def test_draft_from_payload_drops_unusable_words():
    payload = {"lines": [{"words": [
        {"text": "good", "start": 0.0, "end": 0.5},
        {"text": "", "start": 0.5, "end": 0.9},
        {"text": "backwards", "start": 2.0, "end": 1.0},
        {"text": "ok", "start": "bad", "end": 1.0},
    ]}]}
    draft = LyricDraft.from_dict(payload)

    assert draft.word_count == 1
    assert draft.dropped == 3


def test_draft_from_payload_drops_empty_lines():
    draft = LyricDraft.from_dict({"lines": [{"words": []}, {"words": [{"text": "a", "start": 0, "end": 1}]}]})

    assert len(draft) == 1


def test_draft_omits_empty_lines_from_pasteable_text():
    draft = LyricDraft(lines=[
        DraftLine([DraftWord("a", 0.0, 1.0)]),
        DraftLine([]),
    ])

    assert draft.text == "a"


# --- worker ---------------------------------------------------------------


def test_worker_produces_a_draft_with_word_timings(worker, wav, monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_whisper())

    payload = worker.draft(wav, model_name=DEFAULT_MODEL)

    assert payload["ok"] is True
    assert payload["lines"][0]["words"][0]["text"] == "hello"
    assert payload["lines"][0]["words"][1]["start"] == 0.4
    assert payload["language"] == "en"


def test_worker_always_warns_that_it_is_a_draft(worker, wav, monkeypatch):
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_whisper())

    payload = worker.draft(wav)

    assert any("DRAFT" in w for w in payload["warnings"])
    assert any("singing" in w for w in payload["warnings"])


def test_worker_adds_a_cpu_warning(worker, wav, monkeypatch):
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_whisper())

    payload = worker.draft(wav, device="cpu")

    assert any("CPU" in w for w in payload["warnings"])


def test_worker_reports_a_dependency_error_without_whisper(worker, wav, monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("faster_whisper"):
            raise ImportError("No module named 'faster_whisper'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    monkeypatch.delitem(sys.modules, "faster_whisper", raising=False)

    payload = worker.draft(wav)

    assert payload["ok"] is False
    assert payload["kind"] == "dependency"
    assert "faster_whisper" in payload["message"]


def test_worker_reports_a_model_error_when_loading_fails(worker, wav, monkeypatch):
    module = types.ModuleType("faster_whisper")

    def boom(*args, **kwargs):
        raise RuntimeError("model download failed")

    module.WhisperModel = boom
    monkeypatch.setitem(sys.modules, "faster_whisper", module)

    payload = worker.draft(wav)

    assert payload["ok"] is False
    assert payload["kind"] == "model"
    assert "download failed" in payload["message"]


def test_worker_reports_a_conversion_error_when_audio_fails(worker, wav, monkeypatch):
    module = fake_whisper(fail_on_transcribe="ffmpeg not found")
    monkeypatch.setitem(sys.modules, "faster_whisper", module)

    payload = worker.draft(wav)

    assert payload["ok"] is False
    assert payload["kind"] == "conversion"
    assert "ffmpeg not found" in payload["message"]


def test_worker_survives_silence_with_no_segments(worker, wav, monkeypatch):
    module = types.ModuleType("faster_whisper")

    class Empty:
        def __init__(self, *a, **k):
            pass

        def transcribe(self, path, **kwargs):
            return iter([]), types.SimpleNamespace(language="en")

    module.WhisperModel = Empty
    monkeypatch.setitem(sys.modules, "faster_whisper", module)

    payload = worker.draft(wav)

    assert payload["ok"] is True
    assert payload["lines"] == []


def test_worker_main_writes_json(worker, tmp_path, wav, monkeypatch):
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_whisper())
    out = tmp_path / "nested" / "draft.json"

    code = worker.main(["--input", str(wav), "--output", str(out)])

    assert code == worker.EXIT_OK
    payload = json.loads(out.read_text())
    assert payload["lines"]


def test_draft_worker_runs_as_a_real_subprocess(tmp_path):
    wav = write_tone_wav(tmp_path / "tone.wav", seconds=0.2)
    out = tmp_path / "draft.json"

    completed = subprocess.run(
        [sys.executable, str(WORKER_PATH), "--input", str(wav), "--output", str(out)],
        capture_output=True, text=True,
        cwd=str(WORKER_PATH.parents[2]), timeout=180,
    )

    assert out.exists(), f"no output; stderr:\n{completed.stderr}"
    payload = json.loads(out.read_text())
    assert payload["ok"] is False
    assert payload["kind"] == "dependency"
    assert "GenericAlias" not in completed.stderr


def test_draft_worker_does_not_shadow_the_standard_library():
    app_root = Path(__file__).resolve().parents[1] / "app"
    reserved = set(sys.stdlib_module_names)

    offenders = [
        path.name
        for path in app_root.rglob("*.py")
        if path.name != "__init__.py" and path.stem in reserved
    ]
    assert not offenders


# --- service --------------------------------------------------------------


def test_availability_false_without_whisper(tmp_path, monkeypatch):
    monkeypatch.setattr("app.draft_lyrics.service.whisper_available", lambda *a, **k: False)

    availability = service.check_availability(tmp_path)

    assert not availability.available
    assert "not installed" in availability.reason


def test_missing_dependency_points_the_user_at_the_manual_editor(tmp_path, wav, monkeypatch):
    monkeypatch.setattr("app.draft_lyrics.service.resolve_backend", lambda *a, **k: ("none", None))

    with pytest.raises(DraftError) as error:
        service.draft_lyrics(wav, tmp_path / "work")

    assert error.value.kind == "dependency"
    assert "Step 3" in error.value.hint, "the hint must point at the working manual path"


def test_processing_failure_is_reported_with_its_kind(tmp_path, wav, monkeypatch):
    from app.ai_process import AiWorkerError

    monkeypatch.setattr("app.draft_lyrics.service.resolve_backend", lambda *a, **k: ("side-venv", Path(sys.executable)))

    def boom(*args, **kwargs):
        raise AiWorkerError("model", "model download failed")

    monkeypatch.setattr("app.draft_lyrics.service.run_python_worker", boom)

    with pytest.raises(DraftError) as error:
        service.draft_lyrics(wav, tmp_path / "work")

    assert error.value.kind == "model"


def test_missing_source_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr("app.draft_lyrics.service.resolve_backend", lambda *a, **k: ("side-venv", Path(sys.executable)))

    with pytest.raises(DraftError) as error:
        service.draft_lyrics(tmp_path / "nope.wav", tmp_path / "work")

    assert error.value.kind == "conversion"


def test_draft_never_modifies_the_source_audio(tmp_path, wav, monkeypatch):
    before = wav.read_bytes()
    monkeypatch.setattr("app.draft_lyrics.service.resolve_backend", lambda *a, **k: ("side-venv", Path(sys.executable)))
    monkeypatch.setattr("app.draft_lyrics.service.run_python_worker", lambda *a, **k: {
        "ok": True, "model": "base", "language": "en", "lines": [],
    })

    service.draft_lyrics(wav, tmp_path / "work")

    assert wav.exists()
    assert wav.read_bytes() == before


def test_draft_becomes_editable_rows():
    draft = LyricDraft(lines=[DraftLine([DraftWord("hello", 1.0, 1.4), DraftWord("world", 1.4, 1.9)])])
    rows = service.draft_to_lyric_rows(draft)

    assert [r.text for r in rows] == ["hello", "world"]
    assert rows[0].line == 1
    assert validate_rows(rows) == [], "a draft must be immediately usable in the editor"


def test_draft_rows_get_a_minimum_duration():
    """A zero-length recognised word must not produce an unusable row."""
    draft = LyricDraft(lines=[DraftLine([DraftWord("hi", 1.0, 1.0)])])
    rows = service.draft_to_lyric_rows(draft, unit=0.45)

    assert rows[0].end > rows[0].start


def test_draft_lines_get_distinct_line_numbers():
    draft = LyricDraft(lines=[
        DraftLine([DraftWord("a", 0.0, 0.5)]),
        DraftLine([DraftWord("b", 1.0, 1.5)]),
    ])
    rows = service.draft_to_lyric_rows(draft)

    assert [r.line for r in rows] == [1, 2]


# --- manual fallback ------------------------------------------------------


def test_manual_lyrics_work_with_no_model_installed(tmp_path):
    """The core promise: the editor is fully usable on its own."""
    rows = rows_from_text("hello world\nthis is manual")

    assert validate_rows(rows) == []
    assert [r.text for r in rows][:2] == ["hello", "world"]


def test_manual_lyrics_work_after_a_drafting_failure(tmp_path, wav, monkeypatch):
    from app.ai_process import AiWorkerError

    monkeypatch.setattr("app.draft_lyrics.service.resolve_backend", lambda *a, **k: ("side-venv", Path(sys.executable)))
    monkeypatch.setattr("app.draft_lyrics.service.run_python_worker", lambda *a, **k: (_ for _ in ()).throw(
        AiWorkerError("runtime", "the model crashed")
    ))

    with pytest.raises(DraftError):
        service.draft_lyrics(wav, tmp_path / "work")

    # The user simply types their own lyrics instead; nothing is blocked.
    rows = rows_from_text("my own lyrics here")
    assert validate_rows(rows) == []


def test_pasting_replaces_a_draft_entirely():
    """A draft must be replaceable by hand without leftover rows."""
    draft_rows = service.draft_to_lyric_rows(
        LyricDraft(lines=[DraftLine([DraftWord("wrong guess", 0.0, 1.0)])])
    )
    assert draft_rows[0].text == "wrong guess"

    pasted = rows_from_text("the real words")

    assert [r.text for r in pasted] == ["the", "real", "words"]
    assert not any("wrong" in r.text for r in pasted)
    assert len(pasted) == 3, "the draft row must not linger"


def test_timings_from_a_draft_remain_editable_rows():
    draft = LyricDraft(lines=[DraftLine([DraftWord("a", 0.0, 1.0), DraftWord("b", 1.0, 2.0)])])
    rows = service.draft_to_lyric_rows(draft)

    edited = [LyricRow(line=r.line, text="fixed", start=r.start + 0.25, end=r.end + 0.25) for r in rows]

    assert [r.text for r in edited] == ["fixed", "fixed"]
    assert validate_rows(edited) == []
