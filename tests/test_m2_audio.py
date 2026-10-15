"""Milestone 2 tests: audio uploads, duration probing, and safe temp cleanup."""

from __future__ import annotations

import wave

import pytest

from app.audio import (
    ALLOWED_EXTENSIONS,
    MAX_UPLOAD_BYTES,
    AudioError,
    check_extension,
    check_size,
    format_duration,
    probe_duration,
    safe_cleanup,
    save_upload,
    workspace,
)


# --- validation -----------------------------------------------------------


@pytest.mark.parametrize("name", ["song.mp3", "SONG.MP3", "track.wav", "a.b.Wav"])
def test_allowed_extensions_pass(name):
    assert check_extension(name) in ALLOWED_EXTENSIONS


@pytest.mark.parametrize("name", ["notes.txt", "song.mid", "song", "song.mp4", "song.mp3.exe"])
def test_other_extensions_are_refused_with_the_allowed_list(name):
    with pytest.raises(AudioError) as error:
        check_extension(name)

    message = str(error.value)
    assert "not supported" in message
    assert ".mp3" in message and ".wav" in message


def test_empty_file_is_refused():
    with pytest.raises(AudioError):
        check_size(0)


def test_oversized_file_is_refused_and_the_limit_is_named():
    with pytest.raises(AudioError) as error:
        check_size(MAX_UPLOAD_BYTES + 1)

    assert "50 MB" in str(error.value)


def test_size_exactly_at_the_limit_is_allowed():
    check_size(MAX_UPLOAD_BYTES)


# --- saving ---------------------------------------------------------------


def test_save_upload_writes_the_file(tmp_path):
    path = save_upload(b"fake audio bytes", "song.mp3", tmp_path)

    assert path.exists()
    assert path.name == "song.mp3"
    assert path.read_bytes() == b"fake audio bytes"


def test_save_upload_strips_any_directory_component(tmp_path):
    """An upload name must never escape the workspace directory."""
    path = save_upload(b"data", "..\\..\\escape.wav", tmp_path)

    assert path.parent.resolve() == tmp_path.resolve()
    assert "escape" in path.name


def test_save_upload_refuses_unsupported_type(tmp_path):
    with pytest.raises(AudioError):
        save_upload(b"data", "lyrics.txt", tmp_path)


# --- duration -------------------------------------------------------------


def _write_wav(path, seconds=1.5, rate=8000):
    frames = int(seconds * rate)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * frames)
    return path


def test_wav_duration_is_read_from_the_header(tmp_path):
    path = _write_wav(tmp_path / "a.wav", seconds=2.0)

    assert probe_duration(path) == pytest.approx(2.0, abs=0.01)


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(AudioError) as error:
        probe_duration(tmp_path / "nope.wav")

    assert "not found" in str(error.value)


def test_corrupt_file_is_reported_not_crashed(tmp_path):
    path = tmp_path / "broken.mp3"
    path.write_bytes(b"this is definitely not an mp3 file")

    with pytest.raises(AudioError):
        probe_duration(path)


@pytest.mark.parametrize(
    "seconds,expected",
    [(0, "0:00"), (9, "0:09"), (65, "1:05"), (225, "3:45"), (3725, "1:02:05"), (-1, "unknown")],
)
def test_duration_formatting(seconds, expected):
    assert format_duration(seconds) == expected


# --- temp files and cleanup safety ---------------------------------------


def test_workspace_creates_and_removes_its_directory(tmp_path):
    with workspace(tmp_path) as directory:
        assert directory.exists()
        (directory / "scratch.txt").write_text("x")
        saved = directory

    assert not saved.exists(), "temporary directory must be removed on exit"


def test_workspace_removes_files_even_when_the_body_raises(tmp_path):
    saved = None
    with pytest.raises(RuntimeError):
        with workspace(tmp_path) as directory:
            saved = directory
            raise RuntimeError("boom")

    assert saved is not None and not saved.exists()


def test_safe_cleanup_removes_a_file_inside_the_base(tmp_path):
    target = tmp_path / "inner.txt"
    target.write_text("x")

    assert safe_cleanup(target, tmp_path) is True
    assert not target.exists()


def test_safe_cleanup_removes_a_subdirectory(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "a.txt").write_text("x")

    assert safe_cleanup(nested, tmp_path) is True
    assert not nested.exists()


def test_safe_cleanup_refuses_to_delete_the_base_itself(tmp_path):
    """Asking to remove the workspace root must be a no-op, not a wipe."""
    base = tmp_path / "work"
    base.mkdir()
    (base / "keep.txt").write_text("important")

    assert safe_cleanup(base, base) is False
    assert base.exists()
    assert (base / "keep.txt").exists()


def test_safe_cleanup_removes_a_child_of_the_base(tmp_path):
    base = tmp_path / "work"
    base.mkdir()
    child = base / "scratch"
    child.mkdir()

    assert safe_cleanup(child, base) is True
    assert not child.exists()
    assert base.exists(), "the workspace itself must survive"


def test_safe_cleanup_refuses_to_delete_outside_the_base(tmp_path):
    base = tmp_path / "work"
    base.mkdir()
    victim = tmp_path / "outside.txt"
    victim.write_text("must survive")

    assert safe_cleanup(victim, base) is False
    assert victim.exists(), "cleanup must never reach outside the workspace"


def test_safe_cleanup_refuses_to_delete_a_parent_directory(tmp_path):
    assert safe_cleanup(tmp_path, tmp_path / "work") is False
    assert tmp_path.exists()