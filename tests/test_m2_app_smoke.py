"""Milestone 2 tests that actually EXECUTE the Streamlit app script.

`streamlit run` serving HTTP 200 proves nothing: Streamlit returns a static
shell and only runs the script when a browser connects. These run it in-process
so a crash anywhere in the page - including the note table - fails the suite.

A regression this file exists for: the note table crashed on first load with
"The truth value of a DataFrame is ambiguous", because the frame selection used
`frame or fallback`.
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"


def _run() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception, [str(error.value) for error in at.exception]
    return at


def test_app_loads_without_crashing():
    """Every step renders, including Step 5 with an empty note table."""
    at = _run()

    assert at.title[0].value == "SongHub Karaoke MIDI Builder"
    for heading in ("Step 1", "Step 2", "Step 3", "Step 4", "Step 5", "Step 6"):
        assert any(heading in subheader.value for subheader in at.subheader), heading


def test_export_button_is_present_and_blocked_with_no_lyrics():
    at = _run()

    button = at.button(key="export")
    assert button is not None


def test_pasting_lyrics_and_splitting_creates_rows():
    at = _run()

    at.text_area(key="lyrics_text").set_value("hello world\nsecond line here")
    at.button(key="split_lyrics").click()
    at.run()

    assert not at.exception, [str(error.value) for error in at.exception]
    rows = at.session_state["lyric_rows"]
    assert len(rows) == 5
    assert [row.text for row in rows] == ["hello", "world", "second", "line", "here"]


def test_syllable_mode_creates_more_rows_than_word_mode():
    at = _run()

    at.text_area(key="lyrics_text").set_value("hello")
    at.radio(key="split_mode").set_value("syllables")
    at.button(key="split_lyrics").click()
    at.run()

    assert not at.exception, [str(error.value) for error in at.exception]
    assert len(at.session_state["lyric_rows"]) > 1


def test_loading_the_demo_fills_lyrics_notes_and_titles():
    at = _run()

    at.button(key="load_demo").click()
    at.run()

    assert not at.exception, [str(error.value) for error in at.exception]
    assert at.session_state["lyric_rows"]
    assert at.session_state["note_rows"]
    assert at.session_state["song_title"]
    assert at.session_state["song_artist"]


def test_adding_placeholder_notes_works_from_an_empty_table():
    """Step 5 starts empty; the seeding button must work from there."""
    at = _run()

    at.button(key="seed_notes").click()
    at.run()

    assert not at.exception, [str(error.value) for error in at.exception]
    assert len(at.session_state["note_rows"]) == 4


def test_export_produces_a_downloadable_file():
    at = _run()

    at.text_area(key="lyrics_text").set_value("hello karaoke world")
    at.text_input(key="song_title").set_value("My Song")
    at.text_input(key="song_artist").set_value("My Singer")
    at.button(key="split_lyrics").click()
    at.run()

    at.button(key="seed_notes").click()
    at.run()

    at.button(key="export").click()
    at.run()

    assert not at.exception, [str(error.value) for error in at.exception]
    result = at.session_state["last_export"]
    assert result.ok, result.summary
    assert result.filename == "My Song - My Singer.mid"
    assert at.get("download_button")


def test_lyrics_only_export_still_works():
    """No notes must not block export."""
    at = _run()

    at.text_area(key="lyrics_text").set_value("only lyrics here")
    at.button(key="split_lyrics").click()
    at.run()

    at.button(key="export").click()
    at.run()

    assert not at.exception, [str(error.value) for error in at.exception]
    assert at.session_state["last_export"].ok