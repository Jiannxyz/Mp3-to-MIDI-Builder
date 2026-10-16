#!/usr/bin/env python
"""Basic Pitch worker. THE ONLY module that imports basic_pitch.

Run under whichever interpreter has the AI dependencies installed:

    <python> worker.py --input song.wav --output notes.json

It is deliberately a standalone script using only the standard library at the
top level, so it can be executed by a different interpreter without needing the
`app` package on that interpreter's import path. That is what allows the AI
dependencies to live in a separate environment from the web app.

Result is written as JSON to --output (not stdout) because TensorFlow and
librosa log freely to the console and would corrupt a stdout protocol.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
import wave
from pathlib import Path

MODEL_NAME = "basic_pitch:ICASSP_2022"
SAMPLE_RATE = 22050

EXIT_OK = 0
EXIT_DEPENDENCY = 3
EXIT_MODEL = 4
EXIT_RUNTIME = 5


def import_error_message(package: str, error: BaseException, *, hint: str = "") -> str:
    """Classify an ImportError: package missing, or installed but broken?

    "basic_pitch is not installed" is wrong and unhelpful when basic_pitch is
    fine and resampy cannot import pkg_resources - that sends the user to
    reinstall something they already have. ModuleNotFoundError.name says which
    module actually failed.

    Duplicated in each worker on purpose: a worker must stay a standalone script
    with no `from app` imports. test_dependency_probe.py asserts all three
    behave identically so they cannot drift.
    """
    missing = getattr(error, "name", "") or ""
    if not missing or missing == package or missing.startswith(package + "."):
        return f"{package} is not installed in this environment ({error}). {hint}".strip()
    return (
        f"{package} is installed, but importing it failed because '{missing}' could "
        f"not be imported ({error}). Run: pip install -r requirements-ai.txt"
    )


def _fail(kind: str, message: str, detail: str = "") -> int:
    payload = {"ok": False, "kind": kind, "message": message, "detail": detail}
    _emit(payload)
    return {"dependency": EXIT_DEPENDENCY, "model": EXIT_MODEL}.get(kind, EXIT_RUNTIME)


def _emit(payload: dict) -> None:
    sys.stdout.write("__KMB_JSON__" + json.dumps(payload) + "\n")


def _audio_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            return handle.getnframes() / float(rate) if rate else 0.0
    except Exception:  # noqa: BLE001 - duration is informational only
        return 0.0


def transcribe(
    input_path: Path,
    *,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    minimum_note_length_ms: float = 127.7,
    minimum_frequency: float | None = None,
    maximum_frequency: float | None = None,
) -> dict:
    """Run Basic Pitch and return the JSON payload.

    Imported lazily so that a missing dependency produces a helpful error
    instead of an ImportError traceback at module load.
    """
    try:
        from basic_pitch.inference import predict  # noqa: PLC0415 - deliberately lazy
    except ImportError as error:
        # basic_pitch being installed says nothing about whether it can run; a
        # broken transitive dependency is a different fix.
        return {
            "ok": False,
            "kind": "dependency",
            "message": import_error_message(
                "basic_pitch", error, hint="Install the optional AI extras to enable note estimation."
            ),
            "detail": traceback.format_exc(limit=6),
        }

    try:
        # basic_pitch 0.4.0: predict(audio_path, model_or_model_path=...)
        # returns (model_output, midi_data, note_events) where each note is
        # (start, end, pitch, amplitude, pitch_bends).
        _model_output, _midi_data, note_events = predict(
            input_path,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            minimum_note_length=minimum_note_length_ms,
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency,
        )
    except TypeError:
        # Older/newer signatures differ; retry with just the audio path, which
        # every known version accepts positionally.
        try:
            _model_output, _midi_data, note_events = predict(input_path)
        except Exception as error:  # noqa: BLE001
            return {"ok": False, "kind": "model", "message": str(error), "detail": traceback.format_exc(limit=5)}
    except Exception as error:  # noqa: BLE001 - model download/load failures vary widely
        return {"ok": False, "kind": "model", "message": str(error), "detail": traceback.format_exc(limit=5)}

    notes = []
    for event in note_events or []:
        start, end, pitch, amplitude = event[0], event[1], event[2], event[3]
        notes.append(
            {
                "start": float(start),
                "end": float(end),
                "pitch": int(pitch),
                "amplitude": float(amplitude),
                # basic_pitch 0.4.0 does not report an instrument per note, so
                # this stays empty and notes group under "unlabelled".
                "label": "",
            }
        )

    warnings = [
        "Note estimates come from a monophonic note tracker. They are an "
        "approximation of a full recording, not a transcription of it."
    ]

    return {
        "ok": True,
        "model": MODEL_NAME,
        "sample_rate": SAMPLE_RATE,
        "audio_duration": _audio_duration(input_path),
        "backend": "basic_pitch",
        "warnings": warnings,
        "notes": notes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Basic Pitch worker")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--onset-threshold", type=float, default=0.5)
    parser.add_argument("--frame-threshold", type=float, default=0.3)
    parser.add_argument("--minimum-note-length", type=float, default=127.7)
    parser.add_argument("--minimum-frequency", type=float, default=None)
    parser.add_argument("--maximum-frequency", type=float, default=None)
    args = parser.parse_args(argv)

    if not args.input.exists():
        return _fail("conversion", f"Input not found: {args.input}")

    payload = transcribe(
        args.input,
        onset_threshold=args.onset_threshold,
        frame_threshold=args.frame_threshold,
        minimum_note_length_ms=args.minimum_note_length,
        minimum_frequency=args.minimum_frequency,
        maximum_frequency=args.maximum_frequency,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload), encoding="utf-8")

    _emit({"ok": payload.get("ok", False), "notes": len(payload.get("notes", []))})
    return EXIT_OK if payload.get("ok") else {"dependency": EXIT_DEPENDENCY, "model": EXIT_MODEL}.get(
        payload.get("kind", ""), EXIT_RUNTIME
    )


if __name__ == "__main__":
    raise SystemExit(main())