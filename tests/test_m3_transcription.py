"""Milestone 3 tests: the note-estimation service, isolated from the model.

Model inference is always mocked. Nothing here downloads or runs TensorFlow, so
the whole file runs in about a second.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.transcription import service
from app.transcription.audio_convert import (
    TARGET_SAMPLE_RATE,
    ensure_wav,
    ffmpeg_path,
    write_tone_wav,
)
from app.transcription.bridge import MARKER, _extract, ai_python, basic_pitch_available, resolve_backend
from app.transcription.contract import (
    KIND_CONVERSION,
    KIND_DEPENDENCY,
    KIND_RUNTIME,
    KIND_TIMEOUT,
    NoteEstimate,
    TranscriptionError,
    TranscriptionResult,
    estimates_from_rows,
    group_estimates,
    pitch_band,
)

# (start, end, pitch, amplitude, pitch_bends) - basic_pitch 0.4.0 shape.
FAKE_NOTE_EVENTS = [
    (0.0, 0.5, 60, 0.9, None),
    (0.5, 1.0, 62, 0.8, None),
    (1.0, 1.5, 64, 0.7, None),
]


# --- generated test audio -------------------------------------------------


def test_tone_wav_is_generated_locally(tmp_path):
    """Short audio is generated, not shipped as a fixture."""
    path = write_tone_wav(tmp_path / "tone.wav", seconds=0.5)

    assert path.exists()
    import wave

    with wave.open(str(path), "rb") as handle:
        assert handle.getframerate() == TARGET_SAMPLE_RATE
        assert handle.getnchannels() == 1
        assert handle.getnframes() == int(0.5 * TARGET_SAMPLE_RATE)


# --- conversion -----------------------------------------------------------


def test_wav_is_passed_through_without_ffmpeg(tmp_path, monkeypatch):
    """A correctly formatted WAV must not need FFmpeg at all."""
    source = write_tone_wav(tmp_path / "tone.wav", seconds=0.5)
    monkeypatch.setattr("app.transcription.audio_convert.ffmpeg_path", lambda: None)

    result = ensure_wav(source, tmp_path / "work")

    assert result.exists()
    assert result.suffix == ".wav"


def test_wrong_sample_rate_is_resampled_and_needs_ffmpeg(tmp_path, monkeypatch):
    source = write_tone_wav(tmp_path / "tone.wav", seconds=0.2, rate=44100)
    monkeypatch.setattr("app.transcription.audio_convert.ffmpeg_path", lambda: None)

    with pytest.raises(TranscriptionError) as error:
        ensure_wav(source, tmp_path / "work")

    assert error.value.kind == KIND_DEPENDENCY
    assert "FFmpeg" in str(error.value)
    assert error.value.hint


def test_mp3_without_ffmpeg_reports_a_dependency_problem(tmp_path, monkeypatch):
    source = tmp_path / "song.mp3"
    source.write_bytes(b"fake mp3 bytes")
    monkeypatch.setattr("app.transcription.audio_convert.ffmpeg_path", lambda: None)

    with pytest.raises(TranscriptionError) as error:
        ensure_wav(source, tmp_path / "work")

    assert error.value.kind == KIND_DEPENDENCY


def test_corrupt_wav_is_reported_as_a_conversion_problem(tmp_path):
    source = tmp_path / "broken.wav"
    source.write_bytes(b"RIFFnot really a wav")

    with pytest.raises(TranscriptionError) as error:
        ensure_wav(source, tmp_path / "work")

    assert error.value.kind == KIND_CONVERSION


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(TranscriptionError):
        ensure_wav(tmp_path / "gone.wav", tmp_path / "work")


def test_conversion_writes_inside_the_workdir(tmp_path, monkeypatch):
    """Temporary conversions must not land next to the user's file."""
    source = write_tone_wav(tmp_path / "tone.wav", seconds=0.3)
    workdir = tmp_path / "work"
    result = ensure_wav(source, workdir)

    assert workdir.resolve() in result.resolve().parents


# --- the result contract --------------------------------------------------


def test_note_estimate_round_trips_through_a_dict():
    original = NoteEstimate(start=1.0, end=1.5, pitch=60, amplitude=0.5, label="piano")
    assert NoteEstimate.from_dict(original.to_dict()) == original


def test_result_from_worker_payload_sorts_and_counts():
    payload = {
        "ok": True,
        "model": "basic_pitch:ICASSP_2022",
        "sample_rate": 22050,
        "audio_duration": 1.5,
        "notes": [
            {"start": 1.0, "end": 1.5, "pitch": 64, "amplitude": 0.7, "label": ""},
            {"start": 0.0, "end": 0.5, "pitch": 60, "amplitude": 0.9, "label": ""},
        ],
    }
    result = TranscriptionResult.from_dict(payload, backend="test")

    assert len(result) == 2
    assert [n.start for n in result.estimates] == [0.0, 1.0]
    assert result.model == "basic_pitch:ICASSP_2022"
    assert result.last_end == 1.5
    assert "2 notes" in result.summary


def test_invalid_notes_are_dropped_and_counted():
    payload = {
        "notes": [
            {"start": 0.0, "end": 0.5, "pitch": 60},  # fine
            {"start": -1.0, "end": 0.5, "pitch": 60},  # negative
            {"start": 1.0, "end": 1.0, "pitch": 60},  # zero length
            {"start": 1.0, "end": 2.0, "pitch": 999},  # impossible pitch
        ]
    }
    result = TranscriptionResult.from_dict(payload)

    assert len(result) == 1
    assert result.dropped == 3


def test_empty_result_says_so():
    assert "No notes" in TranscriptionResult().summary


# --- grouping -------------------------------------------------------------


def test_unlabelled_notes_group_together():
    estimates = [NoteEstimate(0, 1, 60), NoteEstimate(1, 2, 62)]
    groups = group_estimates(estimates, by="label")

    assert list(groups) == ["unlabelled"]
    assert len(groups["unlabelled"]) == 2


def test_labelled_notes_group_by_label():
    estimates = [
        NoteEstimate(0, 1, 60, label="piano"),
        NoteEstimate(1, 2, 62, label="guitar"),
        NoteEstimate(2, 3, 64, label="piano"),
    ]
    groups = group_estimates(estimates, by="label")

    assert set(groups) == {"piano", "guitar"}
    assert len(groups["piano"]) == 2


@pytest.mark.parametrize("pitch,band", [(30, "bass"), (60, "mid"), (80, "treble")])
def test_pitch_bands(pitch, band):
    assert pitch_band(pitch) == band


def test_grouping_by_pitch_splits_bands():
    estimates = [NoteEstimate(0, 1, 36), NoteEstimate(1, 2, 60), NoteEstimate(2, 3, 84)]
    groups = group_estimates(estimates, by="pitch")

    assert set(groups) == {"bass", "mid", "treble"}


def test_unknown_grouping_is_rejected():
    with pytest.raises(ValueError):
        group_estimates([], by="colour")


# --- backend detection ----------------------------------------------------


def test_availability_is_false_when_nothing_is_installed(tmp_path, monkeypatch):
    monkeypatch.setattr("app.transcription.bridge.ai_python", lambda *a, **k: None)
    monkeypatch.setattr("app.transcription.bridge.basic_pitch_available", lambda *a, **k: False)

    availability = service.check_availability(tmp_path)

    assert not availability.available
    assert availability.backend == "none"
    assert "not installed" in availability.reason


def test_side_venv_is_preferred(tmp_path, monkeypatch):
    fake = tmp_path / ".venv-ai" / "Scripts" / "python.exe"
    fake.parent.mkdir(parents=True)
    fake.write_text("")

    assert ai_python(tmp_path / ".venv-ai") == fake
    kind, interpreter = resolve_backend(tmp_path / ".venv-ai")
    assert kind == "side-venv"


def test_falls_back_to_current_interpreter(tmp_path, monkeypatch):
    monkeypatch.setattr("app.transcription.bridge.ai_python", lambda *a, **k: None)
    monkeypatch.setattr("app.transcription.bridge.basic_pitch_available", lambda *a, **k: True)

    kind, interpreter = resolve_backend(tmp_path)

    assert kind == "in-process-venv"
    assert Path(interpreter).name.startswith("python")


def test_missing_side_venv_reports_none(tmp_path):
    assert ai_python(tmp_path / "does-not-exist") is None


def test_basic_pitch_probe_handles_a_missing_interpreter(tmp_path):
    assert basic_pitch_available(tmp_path / "nope.exe") is False


def test_availability_label_is_human_readable(tmp_path, monkeypatch):
    monkeypatch.setattr("app.transcription.bridge.ai_python", lambda *a, **k: None)
    monkeypatch.setattr("app.transcription.bridge.basic_pitch_available", lambda *a, **k: False)

    assert service.check_availability(tmp_path).backend_label == "not installed"


# --- marker parsing -------------------------------------------------------


def test_marker_line_is_extracted_from_noisy_output():
    text = "WARNING: tensorflow\nlots of chatter\n" + MARKER + '{"ok": true}\n'
    assert _extract(text) == {"ok": True}


def test_marker_parsing_tolerates_absent_marker():
    assert _extract("nothing useful here") is None
    assert _extract(MARKER + "not json") is None


# --- orchestration (worker mocked out) ------------------------------------


def test_estimate_notes_converts_then_runs_the_worker(tmp_path, monkeypatch):
    source = write_tone_wav(tmp_path / "tone.wav", seconds=0.5)
    seen = {}

    def fake_run_worker(wav_path, **kwargs):
        seen["wav"] = Path(wav_path)
        seen["params"] = kwargs
        return {
            "ok": True,
            "model": "basic_pitch:ICASSP_2022",
            "sample_rate": 22050,
            "audio_duration": 0.5,
            "notes": [{"start": s, "end": e, "pitch": p, "amplitude": a, "label": ""}
                      for s, e, p, a, _ in FAKE_NOTE_EVENTS],
        }

    monkeypatch.setattr("app.transcription.bridge.run_worker", fake_run_worker)

    result = service.estimate_notes(source, tmp_path / "work", onset_threshold=0.7)

    assert len(result) == 3
    assert seen["wav"].exists(), "the worker must receive a real WAV"
    assert seen["params"]["onset_threshold"] == 0.7
    assert seen["params"]["minimum_frequency"] is None


def test_estimate_notes_propagates_conversion_errors(tmp_path, monkeypatch):
    source = tmp_path / "song.mp3"
    source.write_bytes(b"data")
    monkeypatch.setattr("app.transcription.audio_convert.ffmpeg_path", lambda: None)

    with pytest.raises(TranscriptionError) as error:
        service.estimate_notes(source, tmp_path / "work")

    assert error.value.kind == KIND_DEPENDENCY


def test_worker_failure_is_raised_with_its_kind(tmp_path, monkeypatch):
    source = write_tone_wav(tmp_path / "tone.wav", seconds=0.2)

    def boom(*args, **kwargs):
        raise TranscriptionError("model exploded", "model")

    monkeypatch.setattr("app.transcription.bridge.run_worker", boom)

    with pytest.raises(TranscriptionError) as error:
        service.estimate_notes(source, tmp_path / "work")

    assert error.value.kind == "model"
    assert error.value.hint


# --- bridge error mapping -------------------------------------------------
#
# These drive a real worker script rather than mocking the process layer,
# because the bridge uses Popen (it polls while the model downloads) and a
# mock of subprocess.run would no longer represent it.


def _fake_worker(tmp_path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def test_timeout_is_reported_as_timeout(tmp_path, monkeypatch):
    from app.transcription import bridge

    wav = write_tone_wav(tmp_path / "tone.wav", seconds=0.2)
    worker = _fake_worker(
        tmp_path, "hang.py",
        "import sys, time, pathlib\n"
        "out = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])\n"
        "pathlib.Path(str(out) + '.phase').write_text('downloading-model')\n"
        "time.sleep(30)\n",
    )
    monkeypatch.setattr("app.transcription.bridge.WORKER", worker)
    monkeypatch.setattr("app.transcription.bridge.resolve_backend", lambda *a, **k: ("side-venv", Path(sys.executable)))

    with pytest.raises(TranscriptionError) as error:
        bridge.run_worker(wav, venv_path=tmp_path, timeout=2)

    assert error.value.kind == KIND_TIMEOUT
    assert "Downloading the model" in str(error.value)


def test_no_backend_raises_dependency_error(tmp_path, monkeypatch):
    wav = write_tone_wav(tmp_path / "tone.wav", seconds=0.2)
    monkeypatch.setattr("app.transcription.bridge.resolve_backend", lambda *a, **k: ("none", None))

    from app.transcription import bridge

    with pytest.raises(TranscriptionError) as error:
        bridge.run_worker(wav, venv_path=tmp_path)

    assert error.value.kind == KIND_DEPENDENCY


def test_silent_worker_with_no_output_is_reported(tmp_path, monkeypatch):
    from app.transcription import bridge

    wav = write_tone_wav(tmp_path / "tone.wav", seconds=0.2)
    worker = _fake_worker(
        tmp_path, "silent.py",
        "import sys\n"
        "sys.stderr.write('the interpreter exploded\\n')\n"
        "sys.exit(9)\n",
    )
    monkeypatch.setattr("app.transcription.bridge.WORKER", worker)
    monkeypatch.setattr("app.transcription.bridge.resolve_backend", lambda *a, **k: ("side-venv", Path(sys.executable)))

    with pytest.raises(TranscriptionError) as error:
        bridge.run_worker(wav, venv_path=tmp_path, timeout=60)

    assert "interpreter exploded" in str(error.value)


def test_worker_error_payload_becomes_a_typed_error(tmp_path, monkeypatch):
    from app.transcription import bridge

    wav = write_tone_wav(tmp_path / "tone.wav", seconds=0.2)
    worker = _fake_worker(
        tmp_path, "fails.py",
        "import sys, pathlib, json\n"
        "out = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])\n"
        "out.write_text(json.dumps({'ok': False, 'kind': 'dependency', 'message': 'no basic_pitch'}))\n"
        "sys.exit(3)\n",
    )
    monkeypatch.setattr("app.transcription.bridge.WORKER", worker)
    monkeypatch.setattr("app.transcription.bridge.resolve_backend", lambda *a, **k: ("side-venv", Path(sys.executable)))

    with pytest.raises(TranscriptionError) as error:
        bridge.run_worker(wav, venv_path=tmp_path, timeout=60)

    assert error.value.kind == "dependency"
    assert "no basic_pitch" in str(error.value)


# --- estimates to note rows ----------------------------------------------


@pytest.mark.parametrize(
    "amplitude,expected",
    [(0.0, 1), (0.5, 64), (1.0, 127), (2.0, 127), (-1.0, 1)],
)
def test_amplitude_maps_to_velocity(amplitude, expected):
    assert service.amplitude_to_velocity(amplitude) == expected


def test_estimates_become_note_rows_marked_estimated():
    estimates = [NoteEstimate(start=1.0, end=1.5, pitch=60, amplitude=0.5)]
    rows = service.estimates_to_note_rows(estimates)

    assert len(rows) == 1
    assert rows[0].start == 1.0
    assert rows[0].duration == pytest.approx(0.5)
    assert rows[0].pitch == 60
    assert rows[0].source == "estimated"
    assert rows[0].is_estimated


def test_estimate_rows_have_a_positive_duration_even_for_a_zero_length_note():
    rows = service.estimates_to_note_rows([NoteEstimate(start=1.0, end=1.0, pitch=60)])

    assert rows[0].duration > 0


def test_estimates_from_table_rows_reads_the_estimate_table():
    class Row:
        def __init__(self, start, duration, pitch, amplitude=90, label=""):
            self.start = start
            self.duration = duration
            self.pitch = pitch
            self.amplitude = amplitude
            self.label = label

    rows = [Row(0.0, 0.5, 60), Row(0.5, 0.5, "bad"), Row(1.0, 0.5, 62, amplitude=50)]
    estimates = estimates_from_rows(rows)

    assert [e.pitch for e in estimates] == [60, 62]
    assert estimates[1].amplitude == pytest.approx(0.5)


def test_manual_note_rows_stay_marked_manual():
    from app.notes import NoteRow

    assert not NoteRow().is_estimated
    assert NoteRow(source="manual").source == "manual"


# --- the review table -----------------------------------------------------


def test_assign_groups_returns_time_ordered_pairs():
    estimates = [NoteEstimate(1.0, 1.5, 84), NoteEstimate(0.0, 0.5, 36), NoteEstimate(0.5, 1.0, 60)]
    items = service.assign_groups(estimates, by="pitch")

    assert [note.start for note, _ in items] == [0.0, 0.5, 1.0]
    assert [group for _, group in items] == ["bass", "mid", "treble"]


def test_estimate_frame_round_trip():
    from app.tables import estimate_rows_to_frame, frame_to_estimates

    estimates = [NoteEstimate(0.0, 0.5, 60, amplitude=0.9), NoteEstimate(0.5, 1.0, 72, amplitude=0.4)]
    items = service.assign_groups(estimates, by="pitch")

    frame = estimate_rows_to_frame(items)
    assert list(frame.columns) == [
        "include", "#", "group", "start (s)", "end (s)", "length (s)", "pitch", "amplitude"
    ]
    assert len(frame) == 2

    restored = frame_to_estimates(frame)
    assert [n.pitch for n in restored] == [60, 72]
    assert restored[0].start == 0.0 and restored[1].end == 1.0
    assert restored[0].amplitude == pytest.approx(0.9)


def test_unticked_notes_are_not_carried_into_the_note_table():
    from app.tables import estimate_rows_to_frame, frame_to_estimates

    estimates = [NoteEstimate(0.0, 0.5, 60), NoteEstimate(0.5, 1.0, 62)]
    frame = estimate_rows_to_frame(service.assign_groups(estimates, by="label"))
    frame.loc[1, "include"] = False

    chosen = frame_to_estimates(frame)
    rows = service.estimates_to_note_rows(chosen)

    assert len(rows) == 1
    assert rows[0].pitch == 60


def test_estimate_frame_of_nothing_is_empty_but_well_formed():
    from app.tables import estimate_rows_to_frame

    frame = estimate_rows_to_frame([])

    assert len(frame) == 0
    assert "include" in frame.columns
    assert "pitch" in frame.columns


def test_note_frame_marks_estimated_rows(tmp_path):
    """The source column must survive the round trip so the UI can show it."""
    from app.tables import frame_to_note_rows, note_frame_for

    rows = service.estimates_to_note_rows([NoteEstimate(0.0, 0.5, 60)])
    restored = frame_to_note_rows(note_frame_for(rows))

    assert restored[0].source == "estimated"
    assert restored[0].is_estimated


def test_note_rows_tripping_a_bad_source_string_fall_back_to_manual():
    import pandas as pd

    from app.tables import frame_to_note_rows

    frame = pd.DataFrame([{"channel": 0, "pitch": 60, "start (s)": 0.0, "duration (s)": 1.0,
                           "velocity": 80, "source": "nonsense"}])
    assert frame_to_note_rows(frame)[0].source == "manual"


# --- opt-in: the real model ----------------------------------------------
#
# Everything above mocks inference. These run the actual model and are skipped
# unless the optional AI extras are installed AND KMB_RUN_AI_TESTS=1 is set,
# because they download the model on first use and take seconds, not
# milliseconds. They are the only tests that prove the model path works.


_AI_ENVIRONMENT = Path(__file__).resolve().parents[1] / ".venv-ai"


def _real_model_runnable() -> bool:
    import os

    if os.environ.get("KMB_RUN_AI_TESTS") != "1":
        return False
    try:
        from app.transcription.bridge import ai_python, basic_pitch_available

        interpreter = ai_python() or Path(sys.executable)
        return interpreter.exists() and basic_pitch_available(interpreter)
    except Exception:  # noqa: BLE001
        return False


needs_model = pytest.mark.skipif(
    not _real_model_runnable(),
    reason="needs the optional AI extras and KMB_RUN_AI_TESTS=1 (downloads the model)",
)


@needs_model
def test_real_model_detects_a_known_pitch(tmp_path):
    """A 440 Hz tone is A4 = MIDI 69, so the expected pitch is known exactly."""
    from app.transcription import estimate_notes

    tone = write_tone_wav(tmp_path / "a440.wav", seconds=1.0, frequency=440.0)
    result = estimate_notes(tone, tmp_path / "work")

    assert result.estimates, "the model should detect a sustained 440 Hz tone"
    top = max(result.estimates, key=lambda note: note.duration)
    assert top.pitch == 69, f"expected A4 (MIDI 69), got {top.pitch}"
    assert 0.0 <= top.start < 0.5
    assert top.end > 0.5


@needs_model
def test_real_model_result_converts_to_exportable_note_rows(tmp_path):
    from app.notes import validate_note_rows
    from app.transcription import estimate_notes

    tone = write_tone_wav(tmp_path / "a220.wav", seconds=1.5, frequency=220.0)
    result = estimate_notes(tone, tmp_path / "work")

    rows = service.estimates_to_note_rows(result.estimates)
    assert rows
    assert all(row.is_estimated for row in rows)
    assert validate_note_rows(rows) == []
    # A220 is A3 = MIDI 57.
    assert any(row.pitch == 57 for row in rows), sorted({r.pitch for r in rows})