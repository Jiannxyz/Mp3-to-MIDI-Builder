#!/usr/bin/env python
"""Lyric-draft worker. THE ONLY module that imports faster_whisper.

Run under whichever interpreter has the AI dependencies installed:

    <python> worker.py --input song.wav --output draft.json --model base

Standard library only at module level, so a different interpreter can run it.
No `from app` imports: this must work without this project on sys.path.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

DEFAULT_MODEL = "base"
EXIT_OK = 0
EXIT_DEPENDENCY = 3
EXIT_MODEL = 4
EXIT_RUNTIME = 5

# Mirrors app.ai_process so a timeout can name the stage it was waiting on.
PHASE_DOWNLOADING = "downloading-model"
PHASE_PROCESSING = "processing"


def import_error_message(package: str, error: BaseException, *, hint: str = "") -> str:
    """Classify an ImportError: package missing, or installed but broken?

    Duplicated in each worker on purpose - a worker must stay a standalone
    script with no `from app` imports. test_dependency_probe.py asserts all
    three behave identically so they cannot drift.
    """
    missing = getattr(error, "name", "") or ""
    if not missing or missing == package or missing.startswith(package + "."):
        return f"{package} is not installed in this environment ({error}). {hint}".strip()
    return (
        f"{package} is installed, but importing it failed because '{missing}' could "
        f"not be imported ({error}). Run: pip install -r requirements-ai.txt"
    )

KIND_BY_EXIT = {EXIT_DEPENDENCY: "dependency", EXIT_MODEL: "model", EXIT_RUNTIME: "runtime"}

CAVEAT = (
    "This is a DRAFT produced by speech recognition, which is trained on speech "
    "rather than singing. The words, the spelling and the timings are all "
    "guesses. Replace the text with your own lyrics and correct the timings."
)


def _emit(payload: dict) -> None:
    sys.stdout.write("__KMB_JSON__" + json.dumps(payload) + "\n")


def _phase(output_path: Path, name: str) -> None:
    """Record the current stage so a timeout can name what it was waiting on."""
    try:
        Path(str(output_path) + ".phase").write_text(name, encoding="utf-8")
    except OSError:
        pass


def draft(
    input_path: Path,
    *,
    model_name: str = DEFAULT_MODEL,
    language: str | None = None,
    device: str = "cpu",
    compute_type: str = "int8",
    output_path: Path | None = None,
) -> dict:
    """Run faster-whisper and return a draft payload. Imported lazily on purpose."""
    if output_path is not None:
        _phase(output_path, PHASE_DOWNLOADING)

    try:
        from faster_whisper import WhisperModel  # noqa: PLC0415 - deliberately lazy
    except ImportError as error:
        return {
            "ok": False,
            "kind": "dependency",
            "message": import_error_message("faster_whisper", error),
            "detail": traceback.format_exc(limit=6),
        }

    started = time.monotonic()

    try:
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
    except Exception as error:  # noqa: BLE001 - model download failures vary widely
        return {
            "ok": False,
            "kind": "model",
            "message": f"Could not load the '{model_name}' transcription model: {error}",
            "detail": traceback.format_exc(limit=5),
        }

    if output_path is not None:
        _phase(output_path, PHASE_PROCESSING)

    try:
        segments, info = model.transcribe(
            str(input_path),
            word_timestamps=True,
            language=language or None,
            vad_filter=True,
        )
        lines = []
        for segment in segments:
            words = []
            for word in getattr(segment, "words", None) or []:
                words.append(
                    {
                        "text": str(getattr(word, "word", "")).strip(),
                        "start": float(getattr(word, "start", 0.0) or 0.0),
                        "end": float(getattr(word, "end", 0.0) or 0.0),
                        "confidence": float(getattr(word, "probability", 0.0) or 0.0),
                    }
                )
            if words:
                lines.append({"words": words})
    except Exception as error:  # noqa: BLE001
        return {
            "ok": False,
            "kind": "conversion",
            "message": f"Could not read '{Path(input_path).name}': {error}",
            "detail": traceback.format_exc(limit=5),
        }

    elapsed = time.monotonic() - started
    detected = str(getattr(info, "language", "") or "")

    warnings = [CAVEAT]
    if device == "cpu":
        warnings.append(
            "Drafting ran on the CPU and is much slower than the song's length. "
            "The 'tiny' model is the fastest option."
        )

    return {
        "ok": True,
        "model": model_name,
        "language": detected,
        "device": device,
        "backend": "faster_whisper",
        "elapsed_seconds": elapsed,
        "warnings": warnings,
        "lines": lines,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lyric draft worker")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--language", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    args = parser.parse_args(argv)

    if not args.input.exists():
        payload = {"ok": False, "kind": "conversion", "message": f"Input not found: {args.input}"}
    else:
        payload = draft(
            args.input,
            model_name=args.model,
            language=args.language,
            device=args.device,
            compute_type=args.compute_type,
            output_path=args.output,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload), encoding="utf-8")

    _emit({"ok": payload.get("ok", False), "lines": len(payload.get("lines", []))})
    return EXIT_OK if payload.get("ok") else KIND_BY_EXIT.get(
        {"dependency": EXIT_DEPENDENCY, "model": EXIT_MODEL}.get(payload.get("kind", ""), ""),
        EXIT_RUNTIME,
    )


if __name__ == "__main__":
    raise SystemExit(main())
