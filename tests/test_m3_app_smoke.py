"""Milestone 3 tests for the estimate-and-export flow in the real app script.

The model is never loaded: `check_availability` and `estimate_notes` are
replaced, so this exercises the UI wiring (progress, error display, selection,
conversion into the note table) without TensorFlow.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app.transcription.audio_convert import write_tone_wav
from app.transcription.service import Availability
from app.transcription.contract import NoteEstimate, TranscriptionResult

APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"

FAKE_ESTIMATES = [
    NoteEstimate(0.0, 0.45, 36, amplitude=0.9, label=""),
    NoteEstimate(0.45, 0.9, 60, amplitude=0.8, label=""),
    NoteEstimate(0.9, 1.35, 62, amplitude=0.7, label=""),
    NoteEstimate(1.35, 1.8, 84, amplitude=0.6, label=""),
]

AVAILABLE = Availability(True, "side-venv", r"C:\fake\.venv-ai\Scripts\python.exe")


@pytest.fixture
def wav(tmp_path):
    return write_tone_wav(tmp_path / "tone.wav", seconds=2.0)


def _run() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception, [str(error.value) for error in at.exception]
    return at


def _unavailable(monkeypatch):
    monkeypatch.setattr(
        "app.transcription.check_availability",
        lambda: Availability(False, "none", None, "The optional AI extras are not installed."),
    )


def _available(monkeypatch, estimates=FAKE_ESTIMATES):
    monkeypatch.setattr("app.transcription.check_availability", lambda: AVAILABLE)

    def fake_estimate(source, workdir, **kwargs):
        return TranscriptionResult(
            estimates=list(estimates), model="basic_pitch:ICASSP_2022",
            sample_rate=22050, audio_duration=2.0, backend="test",
        )

    monkeypatch.setattr("app.transcription.estimate_notes", fake_estimate)


def _attach_audio(at: AppTest, wav: Path):
    """Attach audio and re-run so the script renders the estimate step."""
    at.session_state["audio_name"] = "tone.wav"
    at.session_state["audio"] = str(wav)
    at.session_state["audio_duration"] = 2.0
    at.run()


def _seed_lyrics(at: AppTest):
    at.text_area(key="lyrics_text").set_value("hello karaoke world")
    at.button(key="split_lyrics").click()
    at.run()


# --- availability ---------------------------------------------------------


def test_app_still_loads_when_estimation_is_unavailable(monkeypatch):
    """Note estimation is optional: the whole app must still work."""
    _unavailable(monkeypatch)

    at = _run()

    assert any("Step 6" in s.value for s in at.subheader)
    assert any("Step 7" in s.value for s in at.subheader)


def test_unavailable_state_explains_itself_and_offers_instructions(monkeypatch):
    _unavailable(monkeypatch)

    at = _run()
    messages = " ".join(m.value for m in at.markdown) + " ".join(i.value for i in at.info)

    assert "not installed" in messages
    assert "requirements-ai.txt" in messages or "AI" in messages


def test_app_always_warns_that_estimates_are_approximate(monkeypatch):
    _unavailable(monkeypatch)
    at = _run()

    warnings = " ".join(w.value for w in at.warning).lower()
    assert "approximation" in warnings
    assert "not a transcription" in warnings


def test_no_estimate_button_without_audio(monkeypatch):
    _available(monkeypatch)

    at = _run()

    # AppTest raises KeyError for a missing keyed widget rather than
    # returning None, so the button's absence is asserted that way.
    with pytest.raises(KeyError):
        at.button(key="estimate")
    assert any("Upload an audio file" in i.value for i in at.info)


# --- the estimation flow --------------------------------------------------


def test_estimating_populates_the_review_table(monkeypatch, tmp_path, wav):
    _available(monkeypatch)
    at = _run()
    _attach_audio(at, wav)

    at.button(key="estimate").click()
    at.run()

    assert not at.exception, [str(error.value) for error in at.exception]
    frame = at.session_state["estimate_frame"]
    assert len(frame) == 4
    assert at.session_state["estimation"].estimates


def test_selected_estimates_are_added_to_the_note_table(monkeypatch, tmp_path, wav):
    _available(monkeypatch)
    at = _run()
    _attach_audio(at, wav)
    _seed_lyrics(at)

    at.button(key="estimate").click()
    at.run()
    at.button(key="use_estimates").click()
    at.run()

    assert not at.exception, [str(error.value) for error in at.exception]
    note_rows = at.session_state["note_rows"]
    assert len(note_rows) == 4
    assert all(row.source == "estimated" for row in note_rows)
    # Lyric timings must be untouched by note estimation.
    assert len(at.session_state["lyric_rows"]) == 3


def test_estimates_do_not_disturb_lyric_timings(monkeypatch, tmp_path, wav):
    """Requirement: lyric timing stays independent of note transcription."""
    _available(monkeypatch)
    at = _run()
    _attach_audio(at, wav)
    _seed_lyrics(at)
    before = [row.start for row in at.session_state["lyric_rows"]]

    at.button(key="estimate").click()
    at.run()
    at.button(key="use_estimates").click()
    at.run()

    assert [row.start for row in at.session_state["lyric_rows"]] == before


def test_estimated_notes_reach_the_exported_midi(monkeypatch, tmp_path, wav):
    _available(monkeypatch)
    at = _run()
    _attach_audio(at, wav)
    _seed_lyrics(at)

    at.text_input(key="song_title").set_value("Estimated Song")
    at.text_input(key="song_artist").set_value("Test Singer")

    at.button(key="estimate").click()
    at.run()
    at.button(key="use_estimates").click()
    at.run()

    at.button(key="export").click()
    at.run()

    assert not at.exception, [str(error.value) for error in at.exception]
    result = at.session_state["last_export"]
    assert result.ok, result.summary

    from app.validate_midi import parse_bytes

    parsed = parse_bytes(result.data)
    note_ons = [e for e in parsed.channel_events() if e.kind == "note_on"]
    assert len(note_ons) == 4
    assert parsed.meta_events(0x05), "lyrics must still be present"


def test_estimating_zero_notes_is_reported_not_crashed(monkeypatch, tmp_path, wav):
    _available(monkeypatch, estimates=[])
    at = _run()
    _attach_audio(at, wav)

    at.button(key="estimate").click()
    at.run()

    assert not at.exception, [str(error.value) for error in at.exception]
    assert len(at.session_state["estimate_frame"]) == 0
    assert at.button(key="use_estimates").disabled


def test_model_failure_is_shown_with_its_hint(monkeypatch, tmp_path, wav):
    from app.transcription.contract import TranscriptionError

    monkeypatch.setattr("app.transcription.check_availability", lambda: AVAILABLE)

    def boom(*args, **kwargs):
        raise TranscriptionError("model weights not found", "model")

    monkeypatch.setattr("app.transcription.estimate_notes", boom)

    at = _run()
    _attach_audio(at, wav)
    at.button(key="estimate").click()
    at.run()

    assert not at.exception, [str(error.value) for error in at.exception]
    errors = " ".join(e.value for e in at.error)
    assert "model weights not found" in errors


def test_dependency_failure_is_shown_with_its_hint(monkeypatch, tmp_path, wav):
    from app.transcription.contract import TranscriptionError

    monkeypatch.setattr("app.transcription.check_availability", lambda: AVAILABLE)

    def boom(*args, **kwargs):
        raise TranscriptionError("basic_pitch is not installed", "dependency")

    monkeypatch.setattr("app.transcription.estimate_notes", boom)

    at = _run()
    _attach_audio(at, wav)
    at.button(key="estimate").click()
    at.run()

    assert not at.exception, [str(error.value) for error in at.exception]
    combined = " ".join(e.value for e in at.error) + " ".join(i.value for i in at.info)
    assert "not installed" in combined
    assert "README" in combined


def test_estimation_is_optional_for_a_lyrics_only_export(monkeypatch):
    """The export must not require estimation to have been run."""
    _unavailable(monkeypatch)
    at = _run()
    _seed_lyrics(at)

    at.button(key="export").click()
    at.run()

    assert not at.exception, [str(error.value) for error in at.exception]
    assert at.session_state["last_export"].ok