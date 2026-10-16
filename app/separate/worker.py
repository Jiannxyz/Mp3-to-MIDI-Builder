#!/usr/bin/env python
"""Demucs stem-separation worker. THE ONLY module that imports demucs.

Run under whichever interpreter has the AI dependencies installed:

    <python> worker.py --input song.wav --output-dir out/ --model htdemucs

Writes one WAV per stem into --output-dir and a JSON result to --output.

The INPUT file is opened read-only by Demucs and is never written to, moved or
deleted. All output goes to a directory the caller supplies.

Uses only the standard library at module level, so a different interpreter can
run it without this project on its import path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import wave
from pathlib import Path

# hf_xet is the default HuggingFace transfer path, and it fails hard on flaky
# links with a message that names nothing useful:
#
#   RuntimeError: Task error: File reconstruction error: CAS Client Error:
#   Request middleware error: error sending request for url
#   (https://us.aws.cdn.hf.co/xorbs/...)
#
# The classic HTTP path supports HTTP Range resume, so a dropped connection costs
# the bytes already written rather than restarting the whole file. Measured on
# this machine: Xet errors outright; the same file over plain HTTP resumes.
# Set before huggingface_hub is imported, and only if the caller has not
# already made a choice.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# Written as `__KMB_JSON__` and mirrored in a .phase file next to the output, so
# app.ai_process can still be the single owner of the worker protocol.
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

DEFAULT_MODEL = "htdemucs"
EXIT_OK = 0
EXIT_DEPENDENCY = 3
EXIT_MODEL = 4
EXIT_RUNTIME = 5

KIND_BY_EXIT = {EXIT_DEPENDENCY: "dependency", EXIT_MODEL: "model", EXIT_RUNTIME: "runtime"}

CAVEAT = (
    "Separated stems are NOT clean isolated instruments. Each one still contains "
    "bleed from the others, so a note estimate from a stem will be approximate."
)


def _emit(payload: dict) -> None:
    sys.stdout.write("__KMB_JSON__" + json.dumps(payload) + "\n")


def _fail(kind: str, message: str, detail: str = "") -> int:
    _emit({"ok": False, "kind": kind, "message": message, "detail": detail})
    return KIND_BY_EXIT.get({"dependency": EXIT_DEPENDENCY, "model": EXIT_MODEL}.get(kind, ""), EXIT_RUNTIME)


def _wav_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            return handle.getnframes() / float(rate) if rate else 0.0
    except Exception:  # noqa: BLE001 - duration is informational only
        return 0.0


# --- preflight: work out whether the weights can actually be fetched ---------
#
# Measured on a slow link: this checkpoint downloads at ~0.04 MB/s, so 80 MB
# needs ~33 minutes, and a mid-transfer ConnectionResetError makes
# huggingface_hub restart the file from zero. Waiting on that looks like a hang
# and never finishes. Measuring first turns an unknowable wait into a specific
# answer.
#
# Demucs 4.x fetches weights from the HuggingFace hub, NOT from
# dl.fbaipublicfiles.com (that is only the legacy fallback repo), so the
# reachability of the AWS directory says nothing about the real download.

HF_NAMESPACE = "adefossez"
PROBE_SECONDS = 6.0
# A first run projected to take longer than this is refused up front, with the
# measured numbers, rather than being started and abandoned an hour later.
MAX_DOWNLOAD_SECONDS = 900


def hf_repo_name(name: str) -> str:
    """Mirrors demucs.hf.hf_repo_name: htdemucs -> HTDemucs, htdemucs_ft -> HTDemucs-ft."""
    if name == "htdemucs":
        return "HTDemucs"
    if name.startswith("htdemucs_"):
        return "HTDemucs-" + name[len("htdemucs_") :]
    return "Demucs-" + name


def required_files(model: str) -> tuple[str, list[str]]:
    """(repo_id, filenames) the hub will be asked for, read from the LOCAL demucs
    package so this needs no network."""
    import demucs  # noqa: PLC0415 - only needed once demucs imports cleanly
    import yaml  # noqa: PLC0415

    repo_id = f"{HF_NAMESPACE}/{hf_repo_name(model)}"
    local = Path(demucs.__file__).parent / "remote" / f"{model}.yaml"
    signatures: list[str] = []
    if local.exists():
        bag = yaml.safe_load(local.read_text(encoding="utf-8")) or {}
        signatures = list(bag.get("models") or [])
    return repo_id, [f"{model}.yaml", *[f"{s}.safetensors" for s in signatures]]


def cached_files(repo_id: str, files: list[str]) -> dict[str, bool]:
    """Which of the required files are already complete in the HF cache."""
    try:
        from huggingface_hub import try_to_load_from_cache  # noqa: PLC0415
    except ImportError:
        return {}

    status: dict[str, bool] = {}
    for name in files:
        try:
            status[name] = try_to_load_from_cache(repo_id, filename=name) is not None
        except Exception:  # noqa: BLE001 - cache probing must never be fatal
            status[name] = False
    return status


def _checkpoint_url(repo_id: str, filename: str) -> str:
    return f"https://huggingface.co/{repo_id}/resolve/main/{filename}"


def measure_download_rate(url: str, *, seconds: float = PROBE_SECONDS) -> tuple[int, float]:
    """Download for a few seconds and report (bytes, rate bytes/sec).

    Returns (0, 0.0) if the URL cannot be reached at all.
    """
    import urllib.request  # noqa: PLC0415 - stdlib, but kept local for clarity

    total = 0
    start = time.monotonic()
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            while time.monotonic() - start < seconds:
                chunk = response.read(256 * 1024)
                if not chunk:
                    break
                total += len(chunk)
    except Exception:  # noqa: BLE001 - reachability is a yes/no question
        if total:
            elapsed = max(time.monotonic() - start, 0.001)
            return total, total / elapsed
        return 0, 0.0

    elapsed = max(time.monotonic() - start, 0.001)
    return total, total / elapsed


def _remote_size(url: str) -> int:
    import urllib.request  # noqa: PLC0415

    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=20) as response:
            return int(response.headers.get("content-length") or 0)
    except Exception:  # noqa: BLE001
        return 0


# --- resumable download ---------------------------------------------------
#
# Why this exists rather than just calling hf_hub_download:
#
# Measured on this machine, huggingface_hub's own transfer sits at 0 bytes for
# many minutes (1s of CPU, nothing written) and, with hf_xet enabled, fails
# outright with "File reconstruction error: CAS Client Error". A plain urllib GET
# of the same URL does deliver bytes. Separately, a mid-transfer
# ConnectionResetError (WinError 10054) makes huggingface_hub start the file over
# from zero, so a 80 MB file on a slow, flaky link can never finish.
#
# This downloads with HTTP Range requests into the same cache layout
# huggingface_hub uses, so once it lands, hf_hub_download finds it and the
# normal code path takes over. Progress survives a dropped connection, which is
# the whole point.

DOWNLOAD_ATTEMPTS = 40
DOWNLOAD_RETRY_WAIT = 3.0
CHUNK_BYTES = 1 << 20
# A read that arrives in a trickle keeps resetting the socket timeout, so the
# deadline has to be about *progress*, not about the socket. Measured: this link
# dribbles a few bytes, then nothing at all, and `read(1MB)` never returns - the
# transfer sat at 4 MB for four minutes without the 30s socket timeout firing.
STALL_SECONDS = 60.0
READ_TIMEOUT = 20
PROGRESS_REPORT_SECONDS = 2.0


def _hub_cache_dir(repo_id: str) -> Path | None:
    try:
        from huggingface_hub.constants import HF_HUB_CACHE  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return None
    return Path(HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}"


def _resolve_target(url: str, repo_id: str) -> tuple[str, int, str]:
    """(etag, size, revision) for a blob, via a HEAD that follows redirects."""
    import urllib.request  # noqa: PLC0415

    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=READ_TIMEOUT) as response:
        headers = response.headers
        etag = (headers.get("etag") or "").strip('"')
        revision = headers.get("x-repo-commit") or _cached_revision(repo_id)
        size = int(headers.get("content-length") or 0)
    return etag, size, revision


def _cached_revision(repo_id: str) -> str:
    """The commit the local cache already points at, or '' if unknown.

    huggingface_hub only looks under snapshots/<commit-sha>/, so writing the file
    under any other name would leave it invisible and it would download again.
    """
    cache_dir = _hub_cache_dir(repo_id)
    if cache_dir is None:
        return ""
    try:
        return (cache_dir / "refs" / "main").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _stream_to_file(url: str, partial: Path, *, total: int = 0, output_path: Path | None) -> int:
    """Append bytes from `url` to `partial`, resuming from its current length.

    Returns the number of bytes on disk afterwards. Raises on a connection that
    delivers nothing at all, so the caller can count attempts.
    """
    import urllib.request  # noqa: PLC0415

    have = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url)
    if have:
        request.add_header("Range", f"bytes={have}-")

    delivered = 0
    last_progress = time.monotonic()
    last_report = 0.0
    with urllib.request.urlopen(request, timeout=READ_TIMEOUT) as response:
        # A server that ignores Range replies 200 with the whole body; appending
        # that to a partial file would corrupt it.
        if have and response.status != 206:
            partial.unlink(missing_ok=True)
            have = 0
        with open(partial, "ab") as handle:
            # read1 returns whatever has arrived instead of filling the whole
            # chunk, so the stall deadline below is actually enforceable.
            reader = getattr(response, "read1", None) or response.read
            while True:
                if time.monotonic() - last_progress > STALL_SECONDS:
                    handle.flush()
                    raise TimeoutError(f"no data for {int(STALL_SECONDS)}s at {have + delivered}B")
                chunk = reader(CHUNK_BYTES)
                if not chunk:
                    break
                handle.write(chunk)
                delivered += len(chunk)
                last_progress = time.monotonic()
                # Throttled: a phase write per chunk would mean a file write per
                # chunk, and asking the network for the size each time would mean
                # a HEAD request per chunk - both dwarf the transfer itself.
                now = time.monotonic()
                if output_path is not None and now - last_report >= PROGRESS_REPORT_SECONDS:
                    last_report = now
                    _phase(output_path, PHASE_DOWNLOADING, f"{have + delivered}/{total}B")
    handle_size = partial.stat().st_size if partial.exists() else 0
    if handle_size == have:
        raise OSError("the connection delivered no data")
    return handle_size


def fetch_weight(repo_id: str, filename: str, *, output_path: Path | None = None) -> dict:
    """Download one model file into the HuggingFace cache, resumably.

    Returns {"ok": bool, "path": str, "message": str}.
    """
    import shutil  # noqa: PLC0415

    cache_dir = _hub_cache_dir(repo_id)
    if cache_dir is None:
        return {"ok": False, "path": "", "message": "The HuggingFace cache location is unknown."}

    url = _checkpoint_url(repo_id, filename)
    try:
        etag, size, revision = _resolve_target(url, repo_id)
    except Exception as error:  # noqa: BLE001
        return {"ok": False, "path": "", "message": f"Could not reach {url}: {error}"}

    snapshot = cache_dir / "snapshots" / revision
    snapshot.mkdir(parents=True, exist_ok=True)
    final = snapshot / filename

    # huggingface_hub resolves "main" through refs/main before looking in
    # snapshots/. Without this the file is on disk but invisible: it would be
    # reported as not cached and downloaded again on every run.
    refs = cache_dir / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "main").write_text(revision, encoding="utf-8")

    if final.exists() and (not size or final.stat().st_size == size):
        return {"ok": True, "path": str(final), "message": "already cached"}

    blobs = cache_dir / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    partial = blobs / f"{etag}.incomplete" if etag else blobs / f"{filename}.incomplete"

    def finish() -> dict:
        blob = blobs / (etag or filename)
        partial.replace(blob)
        # A copy, not a symlink: Windows symlinks need elevation, and a missing
        # snapshot entry would send huggingface_hub straight back to the network.
        shutil.copyfile(blob, final)
        return {"ok": True, "path": str(final), "message": "downloaded"}

    last = ""
    for attempt in range(DOWNLOAD_ATTEMPTS):
        error = ""
        try:
            _stream_to_file(url, partial, total=size, output_path=output_path)
        except Exception as failure:  # noqa: BLE001 - a reset is expected here
            error = str(failure)

        # Measure what is actually on disk, not what the last call returned: a
        # connection dropped on the final chunk still completed the file, and
        # counting only the returned bytes would retry a finished download until
        # the attempts ran out.
        have = partial.stat().st_size if partial.exists() else 0
        if size and have >= size:
            return finish()

        last = error or f"only {have} of {size} bytes arrived"
        if attempt < DOWNLOAD_ATTEMPTS - 1:
            time.sleep(DOWNLOAD_RETRY_WAIT)

    partial.unlink(missing_ok=True)
    return {
        "ok": False,
        "path": "",
        "message": f"Gave up downloading {filename} after {DOWNLOAD_ATTEMPTS} attempts ({last}).",
    }


def clean_stale_partials(repo_id: str) -> dict:
    """Delete zero-byte `.incomplete` blobs for this repo and report what remains.

    A reset mid-transfer leaves an `.incomplete` file behind. Once it has grown
    to zero bytes it can never be resumed, so it is pure waste; one that has
    real bytes is left alone because huggingface_hub can continue from it.
    """
    try:
        from huggingface_hub.constants import HF_HUB_CACHE  # noqa: PLC0415

        blobs = Path(HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}" / "blobs"
    except Exception:  # noqa: BLE001 - housekeeping must never be fatal
        return {"removed": 0, "held_bytes": 0}

    removed = 0
    held = 0
    try:
        for blob in blobs.glob("*.incomplete"):
            size = blob.stat().st_size
            if size == 0:
                blob.unlink(missing_ok=True)
                removed += 1
            else:
                held += size
    except OSError:
        pass
    return {"removed": removed, "held_bytes": held}


def preflight(model: str) -> dict:
    """Decide whether the model can be obtained, before attempting the download.

    Returns a dict with either `cached: True` (nothing to download) or a
    projected download time, plus the reason it cannot be completed.
    """
    try:
        repo_id, files = required_files(model)
    except Exception as error:  # noqa: BLE001 - report rather than crash
        return {"cached": False, "ok": True, "files": [], "note": f"could not plan the download ({error})"}

    if not files:
        return {"cached": False, "ok": True, "files": [], "note": ""}

    status = cached_files(repo_id, files)
    if status and all(status.values()):
        return {
            "repo_id": repo_id,
            "files": files,
            "cached": True,
            "ok": True,
            "note": "The model is already in the HuggingFace cache, so nothing will be downloaded.",
        }

    missing = [name for name, done in status.items() if not done] or files
    cleanup = clean_stale_partials(repo_id)
    weights = max(missing, key=lambda n: n.endswith(".safetensors"), default=missing[0])
    url = _checkpoint_url(repo_id, weights)

    size_bytes = _remote_size(url)
    if not size_bytes:
        return {
            "repo_id": repo_id,
            "files": missing,
            "cached": False,
            "ok": False,
            "reason": "download",
            "cleanup": cleanup,
            "note": f"Could not reach the model weights at {url}.",
        }

    received, rate = measure_download_rate(url)
    if rate <= 0:
        return {
            "repo_id": repo_id,
            "files": missing,
            "cached": False,
            "ok": False,
            "reason": "unreachable",
            "cleanup": cleanup,
            "note": f"Could not download from {url}.",
        }

    projected = size_bytes / rate
    result = {
        "repo_id": repo_id,
        "files": missing,
        "cached": False,
        "ok": projected <= MAX_DOWNLOAD_SECONDS,
        "size_bytes": size_bytes,
        "rate_bytes_per_sec": round(rate),
        "projected_seconds": round(projected),
        "cleanup": cleanup,
        "reason": "too-slow" if projected > MAX_DOWNLOAD_SECONDS else "",
        "fetch": {"repo_id": repo_id, "filename": weights},
    }
    if not result["ok"]:
        result["note"] = (
            f"Your connection downloads this model at {rate / 1048576:.2f} MB/s. "
            f"{weights} is {size_bytes / 1048576:.0f} MB, which would take about "
            f"{projected / 60:.0f} minutes here. The download can be resumed, so it "
            "will pick up where it stopped if you run it again."
        )
    return result


def _phase(output_path: Path, name: str, detail: str = "") -> None:
    """Record the current stage so a timeout can say what it was waiting on.

    Without this, a blocked model download and genuinely slow separation are
    indistinguishable from the outside.
    """
    try:
        Path(str(output_path) + ".phase").write_text(f"{name}|{detail}" if detail else name, encoding="utf-8")
    except OSError:
        pass


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch  # noqa: PLC0415 - only needed for the device probe

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001 - torch missing means CPU anyway
        return "cpu"


def separate(
    input_path: Path,
    output_dir: Path,
    *,
    model: str = DEFAULT_MODEL,
    device: str = "auto",
    shifts: int = 0,
    segment: int | None = None,
    output_path: Path | None = None,
) -> dict:
    """Run Demucs and write one WAV per stem. Imported lazily on purpose."""
    if output_path is not None:
        _phase(output_path, PHASE_DOWNLOADING)

    try:
        from demucs.api import Separator, save_audio  # noqa: PLC0415 - deliberately lazy
    except ImportError as error:
        return {
            "ok": False,
            "kind": "dependency",
            "message": import_error_message("demucs", error),
            "detail": traceback.format_exc(limit=6),
        }

    resolved_device = _resolve_device(device)
    source_duration = _wav_duration(input_path)
    started = time.monotonic()

    plan = preflight(model)
    if plan.get("cached"):
        if output_path is not None:
            _phase(output_path, PHASE_PROCESSING, "cached")
    elif plan.get("ok") is False:
        reason = plan.get("reason", "download")
        return {
            "ok": False,
            "kind": "model",
            "reason": reason,
            "message": (
                f"The '{model}' model could not be obtained. {plan.get('note', '')} "
                "Run this on a faster connection, copy the HuggingFace cache from a "
                "machine that has it, or skip separation - note estimation works "
                "without it."
            ),
            "detail": json.dumps(plan, indent=2),
        }
    else:
        # huggingface_hub's own transfer stalls at zero bytes on flaky links, so
        # fetch the weights resumably first; hf_hub_download then finds them in
        # the cache and the normal path proceeds.
        fetch = plan.get("fetch") or {}
        if fetch:
            fetched = fetch_weight(fetch["repo_id"], fetch["filename"], output_path=output_path)
            if not fetched["ok"]:
                return {
                    "ok": False,
                    "kind": "model",
                    "reason": "download",
                    "message": f"Could not download the '{model}' model. {fetched['message']}",
                    "detail": json.dumps(plan, indent=2),
                }
            if output_path is not None:
                _phase(output_path, PHASE_PROCESSING, fetched["message"])

    try:
        separator = Separator(model=model, device=resolved_device, shifts=shifts, segment=segment)
    except Exception as error:  # noqa: BLE001 - model download/load failures vary widely
        return {
            "ok": False,
            "kind": "model",
            "message": f"Could not load the '{model}' separation model: {error}",
            "detail": traceback.format_exc(limit=5),
        }

    if output_path is not None:
        _phase(output_path, PHASE_PROCESSING)

    try:
        _origin, separated = separator.separate_audio_file(Path(input_path))
    except Exception as error:  # noqa: BLE001
        return {
            "ok": False,
            "kind": "conversion",
            "message": f"Demucs could not read '{input_path.name}': {error}",
            "detail": traceback.format_exc(limit=5),
        }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stems = []
    try:
        for stem_name, source in separated.items():
            destination = output_dir / f"{stem_name}.wav"
            save_audio(source, str(destination), samplerate=separator.samplerate)
            stems.append(
                {"name": str(stem_name), "path": str(destination), "duration": _wav_duration(destination)}
            )
    except Exception as error:  # noqa: BLE001
        return {
            "ok": False,
            "kind": "runtime",
            "message": f"Failed while writing separated stems: {error}",
            "detail": traceback.format_exc(limit=5),
        }

    elapsed = time.monotonic() - started
    warnings = [CAVEAT]

    held = (plan.get("cleanup") or {}).get("held_bytes", 0)
    if held:
        warnings.append(
            f"Resumed from {held / 1048576:.0f} MB of a previous download attempt."
        )
    if plan.get("cached"):
        warnings.append("Used the model already in the HuggingFace cache; nothing was downloaded.")
    if resolved_device == "cpu":
        warnings.append(
            f"Run on the CPU. Separating {source_duration / 60:.1f} minutes of audio "
            f"took {elapsed:.0f} seconds here; on a slower machine expect several "
            "times longer."
        )

    return {
        "ok": True,
        "model": model,
        "device": resolved_device,
        "source": str(input_path),
        "audio_duration": source_duration,
        "backend": "demucs",
        "elapsed_seconds": elapsed,
        "warnings": warnings,
        "stems": stems,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Demucs separation worker")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--shifts", type=int, default=0)
    parser.add_argument("--segment", type=int, default=None)
    args = parser.parse_args(argv)

    if not args.input.exists():
        return _fail("conversion", f"Input not found: {args.input}")

    payload = separate(
        args.input,
        args.output_dir,
        model=args.model,
        device=args.device,
        shifts=args.shifts,
        segment=args.segment,
        output_path=args.output,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload), encoding="utf-8")

    _emit({"ok": payload.get("ok", False), "stems": len(payload.get("stems", []))})
    return EXIT_OK if payload.get("ok") else KIND_BY_EXIT.get(
        {"dependency": EXIT_DEPENDENCY, "model": EXIT_MODEL, "conversion": EXIT_RUNTIME, "runtime": EXIT_RUNTIME}.get(
            payload.get("kind", ""), EXIT_RUNTIME
        ),
        EXIT_RUNTIME,
    )


if __name__ == "__main__":
    raise SystemExit(main())
