"""Milestones 4-5 app tests: separation and draft-lyrics steps in the real script.

Both services are replaced, so no Demucs and no speech model is ever loaded.
The focus is on degradation: the app must stay fully usable when both optional
services are missing or fail.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app.draft_lyrics.contract import DraftLine, DraftWord, LyricDraft
from app.draft_lyrics.service import DraftAvailability
from app.separate.contract import SeparationResult, StemInfo
from app.separate.service import SeparationAvailability
from app.transcription.audio_convert import write_tone_wav

APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"

SEPARATION_OK = SeparationAvailability(True, "side-venv", r"C:\fake\.venv-ai\Scripts\python.exe")
DRAFT_OK = DraftAvailability(True, "side-venv", r"C:\fake\.venv-ai\Scripts\python.exe")

FAKE_STEMS = [StemInfo("vocals", "vocals.wav", 10.0), StemInfo("drums", "drums.wav", 10.0),
              StemInfo("bass", "bass.wav", 10.0), StemInfo("other", "other.wav", 10.0)]


@pytest.fixture
def wav(tmp_path):
    return write_tone_wav(tmp_path / "song.wav", seconds=2.0)


def _run() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception, [str(error.value) for error in at.exception]
    return at


def _offline(monkeypatch):
    monkeypatch.setattr("app.separate.check_availability", lambda: SeparationAvailability(False, "none", None, "not installed"))
    monkeypatch.setattr("app.draft_lyrics.check_availability", lambda: DraftAvailability(False, "none", None, "not installed"))


def _attach_audio(at: AppTest, wav: Path):
    at.session_state["audio_name"] = "song.wav"
    at.session_state["audio"] = str(wav)
    at.session_state["audio_duration"] = 2.0
    at.run()


def _manual_lyrics(at: AppTest, text="hello karaoke world"):
    at.text_area(key="lyrics_text").set_value(text)
    at.button(key="split_lyrics").click()
    at.run()


# --- both optional services missing ---------------------------------------


def test_app_loads_with_both_ai_services_unavailable(monkeypatch):
    _offline(monkeypatch)
    at = _run()

    headings = [s.value for s in at.subheader]
    for step in ("Step 6", "Step 7", "Step 8"):
        assert any(step in h for h in headings), f"{step} missing from {headings}"


def test_manual_flow_completes_with_no_ai_services(monkeypatch):
    _offline(monkeypatch)
    at = _run()

    at.text_input(key="song_title").set_value("Manual Song")
    at.text_input(key="song_artist").set_value("Someone")
    _manual_lyrics(at)
    at.button(key="export").click()
    at.run()

    assert not at.exception, [str(error.value) for error in at.exception]
    assert at.session_state["last_export"].ok


def test_separation_absence_message_points_at_step_7(monkeypatch):
    _offline(monkeypatch)
    at = _run()

    messages = " ".join(i.value for i in at.info)
    assert "Step 7" in messages, "the user must be told they can still estimate from the mix"


def test_draft_absence_message_points_at_the_manual_editor(monkeypatch):
    _offline(monkeypatch)
    at = _run()

    messages = " ".join(i.value for i in at.info) + " ".join(c.value for c in at.caption)
    assert "Split lyrics into rows" in messages or "paste" in messages.lower()


# --- separation -----------------------------------------------------------


def test_separation_produces_stems_and_shows_them(monkeypatch, tmp_path, wav):
    monkeypatch.setattr("app.separate.check_availability", lambda: SEPARATION_OK)
    result = SeparationResult(
        stems=[StemInfo(s.name, str(tmp_path / s.name), 10.0) for s in FAKE_STEMS],
        model="htdemucs", device="cpu", audio_duration=10.0, elapsed_seconds=120.0,
    )
    monkeypatch.setattr("app.separate.separate_stems", lambda *a, **k: result)

    at = _run()
    _attach_audio(at, wav)
    at.button(key="separate").click()
    at.run()

    assert not at.exception, [str(error.value) for error in at.exception]
    body = " ".join(m.value for m in at.markdown)
    for name in ("vocals", "drums", "bass", "other"):
        assert name in body
    assert "not useful for note tracking" in body, "drums must be flagged"


def test_separation_recommends_the_vocals_stem(monkeypatch, tmp_path, wav):
    monkeypatch.setattr("app.separate.check_availability", lambda: SEPARATION_OK)
    result = SeparationResult(
        stems=[StemInfo(s.name, str(tmp_path / s.name), 10.0) for s in FAKE_STEMS],
        model="htdemucs", device="cpu", audio_duration=10.0,
    )
    monkeypatch.setattr("app.separate.separate_stems", lambda *a, **k: result)

    at = _run()
    _attach_audio(at, wav)
    at.button(key="separate").click()
    at.run()

    success = " ".join(s.value for s in at.success)
    assert "vocals" in success


def test_separation_warns_the_stems_are_not_clean(monkeypatch, tmp_path, wav):
    _offline(monkeypatch)
    at = _run()
    warnings = " ".join(w.value for w in at.warning).lower()

    assert "not clean" in warnings or "leaks sound" in warnings
    assert "approximate" in warnings


def test_separation_failure_is_reported_and_app_survives(monkeypatch, tmp_path, wav):
    from app.separate.contract import SeparationError

    monkeypatch.setattr("app.separate.check_availability", lambda: SEPARATION_OK)

    def boom(*a, **k):
        raise SeparationError("model download failed", "model")

    monkeypatch.setattr("app.separate.separate_stems", boom)

    at = _run()
    _attach_audio(at, wav)
    at.button(key="separate").click()
    at.run()

    assert not at.exception, [str(error.value) for error in at.exception]
    assert any("model download failed" in e.value for e in at.error)


def test_separation_never_claims_to_change_the_original(monkeypatch, tmp_path, wav):
    """The promise is shown in the UI when separation is actually offered."""
    monkeypatch.setattr("app.separate.check_availability", lambda: SEPARATION_OK)

    at = _run()
    _attach_audio(at, wav)
    captions = " ".join(c.value for c in at.caption)

    assert "never modified or deleted" in captions


# --- separation feeding estimation ---------------------------------------


def test_stems_become_estimation_sources(monkeypatch, tmp_path, wav):
    from app.transcription.contract import NoteEstimate, TranscriptionResult
    from app.transcription.service import Availability

    monkeypatch.setattr("app.separate.check_availability", lambda: SEPARATION_OK)
    monkeypatch.setattr("app.transcription.check_availability", lambda: Availability(True, "side-venv", "python"))
    monkeypatch.setattr(
        "app.transcription.estimate_notes",
        lambda path, *a, **k: TranscriptionResult(estimates=[NoteEstimate(0.0, 0.5, 60)], model="fake"),
    )

    result = SeparationResult(
        stems=[StemInfo(s.name, str(tmp_path / s.name), 10.0) for s in FAKE_STEMS],
        model="htdemucs", device="cpu", audio_duration=10.0,
    )
    monkeypatch.setattr("app.separate.separate_stems", lambda *a, **k: result)

    at = _run()
    _attach_audio(at, wav)
    at.button(key="separate").click()
    at.run()

    options = at.selectbox(key="estimate_source").options
    assert "Whole song" in options
    assert "vocals" in options
    assert "bass" in options
    assert "drums" not in options, "drums must not be offered as a note source"


# --- draft lyrics ---------------------------------------------------------


def test_draft_fills_the_lyrics_box_and_rows(monkeypatch, wav):
    monkeypatch.setattr("app.draft_lyrics.check_availability", lambda: DRAFT_OK)
    draft = LyricDraft(
        lines=[DraftLine([DraftWord("guessed", 0.0, 0.4), DraftWord("words", 0.4, 0.9)])],
        model="base",
    )
    monkeypatch.setattr("app.draft_lyrics.draft_lyrics", lambda *a, **k: draft)

    at = _run()
    _attach_audio(at, wav)
    at.button(key="make_draft").click()
    at.run()

    assert not at.exception, [str(error.value) for error in at.exception]
    assert at.session_state["lyrics_text"] == "guessed words"
    assert len(at.session_state["lyric_rows"]) == 2


def test_draft_is_labelled_as_a_guess(monkeypatch, wav):
    _offline(monkeypatch)
    at = _run()

    warnings = " ".join(w.value for w in at.warning).lower()
    assert "draft is a guess" in warnings or "trained on speech" in warnings


def test_draft_can_be_discarded_to_use_your_own_words(monkeypatch, wav):
    monkeypatch.setattr("app.draft_lyrics.check_availability", lambda: DRAFT_OK)
    draft = LyricDraft(lines=[DraftLine([DraftWord("wrong", 0.0, 1.0)])], model="base")
    monkeypatch.setattr("app.draft_lyrics.draft_lyrics", lambda *a, **k: draft)

    at = _run()
    _attach_audio(at, wav)
    at.button(key="make_draft").click()
    at.run()
    at.button(key="discard_draft").click()
    at.run()

    assert at.session_state["lyrics_text"] == ""
    assert at.session_state["lyric_rows"] == []


def test_pasting_over_a_draft_replaces_it(monkeypatch, wav):
    monkeypatch.setattr("app.draft_lyrics.check_availability", lambda: DRAFT_OK)
    draft = LyricDraft(lines=[DraftLine([DraftWord("wrongguess", 0.0, 1.0)])], model="base")
    monkeypatch.setattr("app.draft_lyrics.draft_lyrics", lambda *a, **k: draft)

    at = _run()
    _attach_audio(at, wav)
    at.button(key="make_draft").click()
    at.run()

    at.text_area(key="lyrics_text").set_value("the real words")
    at.button(key="split_lyrics").click()
    at.run()

    rows = at.session_state["lyric_rows"]
    assert [r.text for r in rows] == ["the", "real", "words"]


def test_draft_failure_leaves_the_manual_path_usable(monkeypatch, wav):
    from app.draft_lyrics.contract import DraftError

    monkeypatch.setattr("app.draft_lyrics.check_availability", lambda: DRAFT_OK)

    def boom(*a, **k):
        raise DraftError("the model crashed", "runtime")

    monkeypatch.setattr("app.draft_lyrics.draft_lyrics", boom)

    at = _run()
    _attach_audio(at, wav)
    at.button(key="make_draft").click()
    at.run()

    assert not at.exception, [str(error.value) for error in at.exception]
    assert any("crashed" in e.value for e in at.error)

    _manual_lyrics(at, "i typed these myself")
    at.button(key="export").click()
    at.run()
    assert at.session_state["last_export"].ok


def test_empty_draft_is_reported_not_treated_as_success(monkeypatch, wav):
    monkeypatch.setattr("app.draft_lyrics.check_availability", lambda: DRAFT_OK)
    monkeypatch.setattr("app.draft_lyrics.draft_lyrics", lambda *a, **k: LyricDraft(model="base"))

    at = _run()
    _attach_audio(at, wav)
    at.button(key="make_draft").click()
    at.run()

    assert not at.exception, [str(error.value) for error in at.exception]
    assert any("No speech was recognised" in w.value for w in at.warning)


# --- a blocked model download is remembered for the session ---------------


def test_no_block_banner_before_anything_fails(monkeypatch, wav):
    _available_and_working(monkeypatch)

    at = _run()
    _attach_audio(at, wav)

    assert not any("Steps 6 and 7" in e.value for e in at.error)


def test_a_blocked_download_shows_a_banner_at_the_top(monkeypatch, wav):
    """The failing step is far down the page; do not make the user find it twice."""
    _blocked_draft(monkeypatch)

    at = _run()
    _attach_audio(at, wav)
    at.button(key="make_draft").click()
    at.run()
    at.run()

    assert not at.exception, [str(error.value) for error in at.exception]
    banner = [e.value for e in at.error if "Steps 6 and 7" in e.value]
    assert banner, "a persistent banner must appear after a blocked download"
    assert "works without them" in banner[0]


def test_the_banner_persists_after_the_failing_step_is_gone(monkeypatch, wav):
    _blocked_draft(monkeypatch)

    at = _run()
    _attach_audio(at, wav)
    at.button(key="make_draft").click()
    at.run()
    at.run()

    # Replace the failing service and re-render; the warning should remain,
    # because the underlying problem is the network, not the widget.
    _available_and_working(monkeypatch)
    at.run()

    assert any("Steps 6 and 7" in e.value for e in at.error)


def test_a_non_blocked_failure_does_not_set_the_banner(monkeypatch, wav):
    """A crash is not a blocked network; only the latter deserves a banner."""
    from app.draft_lyrics.contract import DraftError

    monkeypatch.setattr("app.draft_lyrics.check_availability", lambda: DRAFT_OK)

    def boom(*a, **k):
        raise DraftError("the model crashed", "runtime")

    monkeypatch.setattr("app.draft_lyrics.draft_lyrics", boom)

    at = _run()
    _attach_audio(at, wav)
    at.button(key="make_draft").click()
    at.run()
    at.run()

    assert not any("Steps 6 and 7" in e.value for e in at.error)


def test_manual_work_still_works_after_a_block(monkeypatch, wav):
    _blocked_draft(monkeypatch)

    at = _run()
    _attach_audio(at, wav)
    at.button(key="make_draft").click()
    at.run()

    _manual_lyrics(at, "i typed these myself")
    at.button(key="export").click()
    at.run()

    assert at.session_state["last_export"].ok


# --- helpers for the block scenarios -------------------------------------


def _available_and_working(monkeypatch):
    from app.draft_lyrics.service import DraftAvailability

    monkeypatch.setattr("app.draft_lyrics.check_availability", lambda: DraftAvailability(True, "side-venv", "py"))
    monkeypatch.setattr(
        "app.draft_lyrics.draft_lyrics",
        lambda *a, **k: LyricDraft(lines=[DraftLine([DraftWord("ok", 0.0, 0.5)])], model="base"),
    )


def _blocked_draft(monkeypatch):
    from app.draft_lyrics.contract import DraftError

    monkeypatch.setattr("app.draft_lyrics.check_availability", lambda: DRAFT_OK)

    def boom(*a, **k):
        raise DraftError(
            "The model download made no progress for 90 seconds and was stopped.",
            "stalled-download",
        )

    monkeypatch.setattr("app.draft_lyrics.draft_lyrics", boom)