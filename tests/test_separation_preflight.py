"""Tests for the Demucs download preflight.

Root cause this guards against: the weights are fetched from the HuggingFace
hub at ~0.04-0.11 MB/s on a slow link, so an 80 MB file needs 12-33 minutes, and
a mid-transfer ConnectionResetError makes huggingface_hub restart from zero. The
symptom is an apparent hang that never finishes, not a clean error.

The preflight measures the real transfer rate and refuses up front when the
download is not worth starting, and reports an exact projection otherwise.

Demucs 4.x loads weights from HuggingFace, NOT dl.fbaipublicfiles.com (that is
only the legacy fallback repo), so probing the AWS directory proves nothing.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

WORKER_PATH = Path(__file__).resolve().parents[1] / "app" / "separate" / "worker.py"


@pytest.fixture
def worker():
    spec = importlib.util.spec_from_file_location("sep_preflight_worker", WORKER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- the repo/filename mapping mirrors demucs -----------------------------


@pytest.mark.parametrize(
    "model,expected",
    [
        ("htdemucs", "HTDemucs"),
        ("htdemucs_ft", "HTDemucs-ft"),
        ("htdemucs_6s", "HTDemucs-6s"),
        ("mdx_extra_q", "Demucs-mdx_extra_q"),
    ],
)
def test_hf_repo_name_matches_demucs(worker, model, expected):
    assert worker.hf_repo_name(model) == expected


def test_required_files_are_derived_from_the_local_package(worker, tmp_path, monkeypatch):
    """The bag definition ships inside the demucs package, so this needs no network."""
    package = tmp_path / "demucs"
    (package / "remote").mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "remote" / "htdemucs.yaml").write_text("models: ['955717e8']\n", encoding="utf-8")

    fake_yaml = types.ModuleType("yaml")
    fake_yaml.safe_load = lambda text: {"models": ["955717e8"]}
    fake_demucs = types.ModuleType("demucs")
    fake_demucs.__file__ = str(package / "__init__.py")
    monkeypatch.setitem(sys.modules, "demucs", fake_demucs)
    monkeypatch.setitem(sys.modules, "yaml", fake_yaml)

    repo_id, files = worker.required_files("htdemucs")

    assert repo_id == "adefossez/HTDemucs"
    assert files == ["htdemucs.yaml", "955717e8.safetensors"]


def test_checkpoint_url_is_the_huggingface_one(worker):
    url = worker._checkpoint_url("adefossez/HTDemucs", "955717e8.safetensors")

    assert url == "https://huggingface.co/adefossez/HTDemucs/resolve/main/955717e8.safetensors"
    assert "dl.fbaipublicfiles.com" not in url, "the AWS repo is only the legacy fallback"


# --- rate measurement ------------------------------------------------------


def _finite_response(chunks=4):
    """A response that ends after `chunks` reads, so the probe loop terminates."""

    class FakeResponse:
        def __init__(self):
            self.calls = 0

        def read(self, n):
            self.calls += 1
            return b"x" * n if self.calls <= chunks else b""

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    return FakeResponse()


def test_rate_is_measured_from_bytes_received(worker, monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _finite_response())

    received, rate = worker.measure_download_rate("http://example/weights", seconds=30)

    assert received == 4 * 256 * 1024
    assert rate > 0


def test_unreachable_url_reports_zero_rate(worker, monkeypatch):
    def boom(*a, **k):
        raise OSError("connection reset by peer")

    monkeypatch.setattr("urllib.request.urlopen", boom)

    received, rate = worker.measure_download_rate("http://example/weights", seconds=30)

    assert (received, rate) == (0, 0.0)


def test_a_reset_mid_transfer_still_reports_what_arrived(worker, monkeypatch):
    """A partial transfer is the exact symptom here: some bytes, then a reset."""

    class ResettingResponse:
        def __init__(self):
            self.calls = 0

        def read(self, n):
            self.calls += 1
            if self.calls > 2:
                raise OSError("connection reset by peer")
            return b"x" * n

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: ResettingResponse())

    received, rate = worker.measure_download_rate("http://example/weights", seconds=30)

    assert received == 2 * 256 * 1024, "bytes already received must not be discarded"
    assert rate > 0


# --- the decision ----------------------------------------------------------


def _plan(worker, monkeypatch, *, cached, size, rate):
    monkeypatch.setattr(
        worker, "required_files", lambda m: ("adefossez/HTDemucs", ["955717e8.safetensors"])
    )
    monkeypatch.setattr(worker, "cached_files", lambda repo, files: {"955717e8.safetensors": cached})
    monkeypatch.setattr(worker, "_remote_size", lambda url: size)
    monkeypatch.setattr(worker, "measure_download_rate", lambda url, **k: (10, rate))
    return worker.preflight("htdemucs")


def test_a_cached_model_skips_the_download_entirely(worker, monkeypatch):
    plan = _plan(worker, monkeypatch, cached=True, size=84_000_000, rate=10)

    assert plan["cached"] is True
    assert plan["ok"] is True
    assert "already" in plan["note"].lower()


def test_an_unreachable_host_is_refused_with_a_reason(worker, monkeypatch):
    monkeypatch.setattr(
        worker, "required_files", lambda m: ("adefossez/HTDemucs", ["955717e8.safetensors"])
    )
    monkeypatch.setattr(worker, "cached_files", lambda repo, files: {"955717e8.safetensors": False})
    monkeypatch.setattr(worker, "_remote_size", lambda url: 0)

    plan = worker.preflight("htdemucs")

    assert plan["ok"] is False
    assert plan["reason"] == "download"
    assert "Could not reach" in plan["note"]


def test_a_slow_link_is_refused_before_starting(worker, monkeypatch):
    """0.04 MB/s on 80 MB is ~33 minutes; starting that is worse than saying so."""
    plan = _plan(worker, monkeypatch, cached=False, size=84_025_440, rate=41_000)

    assert plan["ok"] is False
    assert plan["reason"] == "too-slow"
    assert plan["projected_seconds"] > 1800
    assert "MB/s" in plan["note"]
    assert "minutes" in plan["note"]


def test_a_reasonable_link_is_allowed_with_a_projection(worker, monkeypatch):
    plan = _plan(worker, monkeypatch, cached=False, size=84_025_440, rate=2_000_000)

    assert plan["ok"] is True
    assert plan["projected_seconds"] == pytest.approx(42, abs=2)
    assert plan["size_bytes"] == 84_025_440


def test_preflight_records_the_numbers_needed_to_explain(worker, monkeypatch):
    plan = _plan(worker, monkeypatch, cached=False, size=84_025_440, rate=500_000)

    for key in ("repo_id", "size_bytes", "rate_bytes_per_sec", "projected_seconds"):
        assert key in plan, f"{key} must be reported so a timeout can explain itself"


def test_the_refusal_message_names_a_way_forward(worker, monkeypatch):
    plan = _plan(worker, monkeypatch, cached=False, size=84_025_440, rate=41_000)
    message = f"The 'htdemucs' model could not be obtained. {plan['note']}"

    assert "0.04 MB/s" in message
    assert "minutes" in message
    assert "restart from the beginning" not in message, "downloads now resume"
    assert "resumed" in message


# --- stale partials and the Xet transfer path ------------------------------


def test_xet_is_disabled_before_huggingface_is_imported(worker):
    """hf_xet fails with a 'File reconstruction error' on flaky links; plain HTTP resumes."""
    assert worker.os.environ["HF_HUB_DISABLE_XET"] == "1"


def test_zero_byte_partials_are_removed_but_real_progress_is_kept(worker, tmp_path, monkeypatch):
    """A partial with bytes can be resumed, so deleting it would throw work away."""
    fake_constants = types.ModuleType("huggingface_hub.constants")
    fake_constants.HF_HUB_CACHE = str(tmp_path)
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.ModuleType("huggingface_hub"))
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", fake_constants)

    blobs = tmp_path / "models--adefossez--HTDemucs" / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "a.incomplete").write_bytes(b"")  # worthless: cannot resume zero bytes
    (blobs / "b.incomplete").write_bytes(b"x" * 4096)  # real progress, keep it

    result = worker.clean_stale_partials("adefossez/HTDemucs")

    assert result["removed"] == 1
    assert not (blobs / "a.incomplete").exists()
    assert (blobs / "b.incomplete").exists(), "resumable bytes must not be discarded"
    assert result["held_bytes"] == 4096


def test_cleanup_never_breaks_a_run_when_the_cache_is_unreadable(worker, tmp_path, monkeypatch):
    """Housekeeping is best-effort: a missing cache dir must not fail separation."""
    fake_constants = types.ModuleType("huggingface_hub.constants")
    fake_constants.HF_HUB_CACHE = str(tmp_path / "does" / "not" / "exist")
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.ModuleType("huggingface_hub"))
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", fake_constants)

    assert worker.clean_stale_partials("adefossez/HTDemucs") == {"removed": 0, "held_bytes": 0}


def test_cleanup_survives_huggingface_hub_being_absent(worker):
    result = worker.clean_stale_partials("adefossez/HTDemucs")

    assert result == {"removed": 0, "held_bytes": 0}


def test_the_plan_reports_what_cleanup_found(worker, monkeypatch):
    monkeypatch.setattr(
        worker, "required_files", lambda m: ("adefossez/HTDemucs", ["955717e8.safetensors"])
    )
    monkeypatch.setattr(worker, "cached_files", lambda repo, files: {"955717e8.safetensors": False})
    monkeypatch.setattr(worker, "clean_stale_partials", lambda repo: {"removed": 3, "held_bytes": 4096})
    monkeypatch.setattr(worker, "_remote_size", lambda url: 84_025_440)
    monkeypatch.setattr(worker, "measure_download_rate", lambda url, **k: (10, 2_000_000))

    plan = worker.preflight("htdemucs")

    assert plan["cleanup"] == {"removed": 3, "held_bytes": 4096}


# --- resumable download ----------------------------------------------------


class _FakeHeaders(dict):
    def get(self, key, default=None):
        for name, value in self.items():
            if name.lower() == key.lower():
                return value
        return default


class _RangeResponse:
    """Serves `body` honouring a Range header, like the real CDN does."""

    def __init__(self, body, status=206):
        self._body = body
        self._at = 0
        self.status = status

    def read(self, n):
        chunk = self._body[self._at : self._at + n]
        self._at += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _serve(worker, monkeypatch, body, *, status=206, headers=None):
    """Patch urlopen so HEAD reports `headers` and GET serves `body`.

    status=206 honours the Range header; status=200 ignores it and returns the
    whole body, which is what a non-resumable server does.
    """
    calls = {"gets": 0}

    def fake_urlopen(request, timeout=None):
        method = getattr(request, "get_method", lambda: "GET")()
        if method == "HEAD":
            return _RangeResponse(b"", status=200)
        calls["gets"] += 1
        start = 0
        if status == 206:
            for name, value in getattr(request, "headers", {}).items():
                if name.lower() == "range":
                    start = int(value.split("=")[1].split("-")[0])
        response = _RangeResponse(body[start:], status=status)
        response.headers = _FakeHeaders(headers or {})
        return response

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return calls


def _cache(tmp_path, monkeypatch, repo="adefossez/HTDemucs"):
    fake_constants = types.ModuleType("huggingface_hub.constants")
    fake_constants.HF_HUB_CACHE = str(tmp_path / "hub")
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.ModuleType("huggingface_hub"))
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", fake_constants)
    return tmp_path / "hub"


def test_a_whole_file_downloads_into_the_cache_layout(worker, tmp_path, monkeypatch):
    """The result must land where huggingface_hub looks, or it downloads again."""
    cache = _cache(tmp_path, monkeypatch)
    body = bytes(range(256)) * 2048  # 512 KB
    _serve(worker, monkeypatch, body)

    monkeypatch.setattr(worker, "_resolve_target", lambda url, repo_id: ("etag123", len(body), "rev1"))
    result = worker.fetch_weight("adefossez/HTDemucs", "w.safetensors")

    assert result["ok"] is True
    assert Path(result["path"]).read_bytes() == body
    assert (cache / "models--adefossez--HTDemucs" / "snapshots" / "rev1" / "w.safetensors").exists()


def test_an_already_cached_file_is_not_downloaded_again(worker, tmp_path, monkeypatch):
    cache = _cache(tmp_path, monkeypatch)
    body = b"x" * 1000
    _serve(worker, monkeypatch, body)
    monkeypatch.setattr(worker, "_resolve_target", lambda url, repo_id: ("e", len(body), "rev1"))
    worker.fetch_weight("adefossez/HTDemucs", "w.safetensors")

    calls = _serve(worker, monkeypatch, body)
    again = worker.fetch_weight("adefossez/HTDemucs", "w.safetensors")

    assert again["ok"] is True
    assert calls["gets"] == 0, "a cached model must not touch the network"
    assert "already" in again["message"]


def test_a_reset_mid_file_resumes_instead_of_restarting(worker, tmp_path, monkeypatch):
    """WinError 10054 loses the connection; progress must survive it."""
    cache = _cache(tmp_path, monkeypatch)
    body = bytes(range(256)) * (3 * 4096)  # 3 MB, so more than one chunk is read
    monkeypatch.setattr(worker, "_resolve_target", lambda url, repo_id: ("e", len(body), "rev1"))
    monkeypatch.setattr(worker, "DOWNLOAD_RETRY_WAIT", 0)

    state = {"first": True}

    class Resetting(_RangeResponse):
        def __init__(self, b, status=206):
            super().__init__(b, status)
            self.calls = 0

        def read(self, n):
            self.calls += 1
            if state["first"] and self.calls == 2:
                raise OSError("connection reset by peer")
            return super().read(n)

    ranges = []

    def fake_urlopen(request, timeout=None):
        if getattr(request, "get_method", lambda: "GET")() == "HEAD":
            return _RangeResponse(b"", status=200)
        start = 0
        for name, value in getattr(request, "headers", {}).items():
            if name.lower() == "range":
                start = int(value.split("=")[1].split("-")[0])
        ranges.append(start)
        return Resetting(body[start:])

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = worker.fetch_weight("adefossez/HTDemucs", "w.safetensors")

    assert result["ok"] is True, "a reset must not abandon the download"
    assert Path(result["path"]).read_bytes() == body
    assert ranges[0] == 0, "the first attempt starts at the beginning"
    assert all(start > 0 for start in ranges[1:]), "every retry must resume, never restart"
    assert ranges == sorted(ranges) and len(set(ranges)) == len(ranges), "progress must accumulate"


def test_a_server_that_ignores_range_does_not_corrupt_the_file(worker, tmp_path, monkeypatch):
    """A 200 reply to a Range request is the whole body; appending would double it."""
    _cache(tmp_path, monkeypatch)
    body = bytes(range(256)) * 2048
    monkeypatch.setattr(worker, "_resolve_target", lambda url, repo_id: ("e", len(body), "rev1"))
    monkeypatch.setattr(worker, "DOWNLOAD_RETRY_WAIT", 0)

    # First attempt leaves a partial behind, then the server stops honouring Range.
    blobs = tmp_path / "hub" / "models--adefossez--HTDemucs" / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    (blobs / "e.incomplete").write_bytes(body[:1024])

    _serve(worker, monkeypatch, body, status=200)

    result = worker.fetch_weight("adefossez/HTDemucs", "w.safetensors")

    assert result["ok"] is True
    assert Path(result["path"]).stat().st_size == len(body), "must not be body + partial"


def test_a_dead_host_gives_up_instead_of_looping_forever(worker, tmp_path, monkeypatch):
    _cache(tmp_path, monkeypatch)
    monkeypatch.setattr(worker, "DOWNLOAD_ATTEMPTS", 2)
    monkeypatch.setattr(worker, "DOWNLOAD_RETRY_WAIT", 0)
    monkeypatch.setattr(worker, "_resolve_target", lambda url, repo_id: ("e", 2048, "rev1"))

    def boom(*a, **k):
        raise OSError("no route to host")

    monkeypatch.setattr("urllib.request.urlopen", boom)

    result = worker.fetch_weight("adefossez/HTDemucs", "w.safetensors")

    assert result["ok"] is False
    assert "no route to host" in result["message"]
    assert "2 attempts" in result["message"]


def test_an_unreachable_host_is_reported_before_any_writing(worker, tmp_path, monkeypatch):
    _cache(tmp_path, monkeypatch)

    def boom(*a, **k):
        raise OSError("connection reset")

    monkeypatch.setattr("urllib.request.urlopen", boom)

    result = worker.fetch_weight("adefossez/HTDemucs", "w.safetensors")

    assert result["ok"] is False
    assert "Could not reach" in result["message"]


def test_the_snapshot_uses_the_commit_sha_huggingface_hub_expects(worker, tmp_path, monkeypatch):
    """A file written under any other name is invisible, and it downloads again."""
    cache = _cache(tmp_path, monkeypatch)
    body = b"x" * 64
    _serve(worker, monkeypatch, body)
    monkeypatch.setattr(worker, "_resolve_target", lambda url, repo: ("e", len(body), "deadbeef"))

    worker.fetch_weight("adefossez/HTDemucs", "w.safetensors")

    assert (cache / "models--adefossez--HTDemucs" / "snapshots" / "deadbeef" / "w.safetensors").exists()


def test_a_missing_commit_header_falls_back_to_the_cached_ref(worker, tmp_path, monkeypatch):
    cache = _cache(tmp_path, monkeypatch)
    refs = cache / "models--adefossez--HTDemucs" / "refs"
    refs.mkdir(parents=True)
    (refs / "main").write_text("cafebabe\n", encoding="utf-8")

    assert worker._cached_revision("adefossez/HTDemucs") == "cafebabe"


def test_an_unknown_revision_is_reported_as_empty(worker, tmp_path, monkeypatch):
    _cache(tmp_path, monkeypatch)

    assert worker._cached_revision("adefossez/HTDemucs") == ""


def test_the_ref_is_written_so_huggingface_hub_can_find_the_file(worker, tmp_path, monkeypatch):
    """It resolves "main" via refs/main; without it the file is invisible."""
    cache = _cache(tmp_path, monkeypatch)
    body = b"models: ['955717e8']\n"
    _serve(worker, monkeypatch, body)
    monkeypatch.setattr(worker, "_resolve_target", lambda url, repo: ("etag", len(body), "deadbeef"))

    worker.fetch_weight("adefossez/HTDemucs", "htdemucs.yaml")

    ref = cache / "models--adefossez--HTDemucs" / "refs" / "main"
    assert ref.read_text(encoding="utf-8") == "deadbeef"


def test_a_cached_file_is_found_again_after_fetching_it(worker, tmp_path, monkeypatch):
    """The whole point: fetch it once, and the next run must not re-download."""
    cache = _cache(tmp_path, monkeypatch)
    body = b"models: ['955717e8']\n"
    _serve(worker, monkeypatch, body)
    monkeypatch.setattr(worker, "_resolve_target", lambda url, repo: ("etag", len(body), "deadbeef"))

    worker.fetch_weight("adefossez/HTDemucs", "htdemucs.yaml")

    # This is the lookup preflight() makes, reproduced with huggingface_hub's rule.
    revision = (cache / "models--adefossez--HTDemucs" / "refs" / "main").read_text()
    cached = cache / "models--adefossez--HTDemucs" / "snapshots" / revision / "htdemucs.yaml"
    assert revision == "deadbeef"
    assert cached.is_file(), "the next run must find this without the network"


# --- integration with separate() ------------------------------------------


def test_separation_refuses_a_hopeless_download_before_downloading(worker, tmp_path, monkeypatch):
    def explode(*a, **k):
        raise AssertionError("Separator must not be constructed when the download is hopeless")

    module = types.ModuleType("demucs.api")
    module.Separator = explode
    module.save_audio = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "demucs", types.ModuleType("demucs"))
    monkeypatch.setitem(sys.modules, "demucs.api", module)
    monkeypatch.setattr(
        worker,
        "preflight",
        lambda model: {"ok": False, "reason": "too-slow", "note": "0.04 MB/s would take 33 minutes"},
    )

    payload = worker.separate(tmp_path / "a.wav", tmp_path / "stems", model="htdemucs")

    assert payload["ok"] is False
    assert payload["kind"] == "model"
    assert payload["reason"] == "too-slow"
    assert "0.04 MB/s" in payload["message"]
    assert "note estimation works without it" in payload["message"]


def test_separation_skips_the_probe_when_cached(worker, tmp_path, monkeypatch):
    """A cached model must not trigger any network probe at all."""
    calls = []

    class FakeSeparator:
        samplerate = 22050

        def __init__(self, **kwargs):
            calls.append("Separator")

        def separate_audio_file(self, path):
            return object(), {"vocals": object()}

    module = types.ModuleType("demucs.api")
    module.Separator = FakeSeparator
    module.save_audio = lambda src, dest, samplerate=None, **k: None
    monkeypatch.setitem(sys.modules, "demucs.api", module)
    monkeypatch.setattr(
        worker,
        "preflight",
        lambda model: calls.append("preflight") or {"ok": True, "cached": True, "note": "cached"},
    )

    worker.separate(tmp_path / "a.wav", tmp_path / "stems")

    assert calls[0] == "preflight"
    assert "Separator" in calls


def test_a_feasible_download_fetches_before_loading_the_model(worker, tmp_path, monkeypatch):
    """The weights must land in the cache first, so hf_hub_download finds them."""
    fetched = []

    module = types.ModuleType("demucs.api")
    module.Separator = lambda **k: (_ for _ in ()).throw(RuntimeError("stop here"))
    module.save_audio = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "demucs.api", module)
    monkeypatch.setattr(
        worker,
        "preflight",
        lambda model: {
            "ok": True,
            "cached": False,
            "projected_seconds": 640,
            "rate_bytes_per_sec": 131_000,
            "fetch": {"repo_id": "adefossez/HTDemucs", "filename": "955717e8.safetensors"},
        },
    )
    monkeypatch.setattr(
        worker,
        "fetch_weight",
        lambda repo, name, output_path=None: fetched.append(name) or {"ok": True, "message": "downloaded"},
    )

    output = tmp_path / "result.json"
    worker.separate(tmp_path / "a.wav", tmp_path / "stems", output_path=output)

    assert fetched == ["955717e8.safetensors"]

    from app.ai_process import read_phase

    assert read_phase(output) == "processing", "the fetch finished, so we are past downloading"


def test_a_failed_fetch_stops_before_loading_the_model(worker, tmp_path, monkeypatch):
    def explode(*a, **k):
        raise AssertionError("Separator must not be built without weights")

    module = types.ModuleType("demucs.api")
    module.Separator = explode
    module.save_audio = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "demucs.api", module)
    monkeypatch.setattr(
        worker,
        "preflight",
        lambda model: {
            "ok": True,
            "cached": False,
            "fetch": {"repo_id": "adefossez/HTDemucs", "filename": "955717e8.safetensors"},
        },
    )
    monkeypatch.setattr(
        worker,
        "fetch_weight",
        lambda repo, name, output_path=None: {"ok": False, "message": "gave up after 40 attempts"},
    )

    payload = worker.separate(tmp_path / "a.wav", tmp_path / "stems")

    assert payload["ok"] is False
    assert payload["kind"] == "model"
    assert "gave up after 40 attempts" in payload["message"]


def test_a_trickle_that_never_completes_a_chunk_is_abandoned(worker, tmp_path, monkeypatch):
    """A few bytes per minute keep the socket alive but never finish a 1 MB read.

    This is the measured failure: the transfer sat at 4 MB for four minutes and
    the 20s socket timeout never fired, because a socket timeout only measures
    silence, not progress.
    """
    monkeypatch.setattr(worker, "STALL_SECONDS", 0.01)
    monkeypatch.setattr(worker, "_remote_size", lambda url: 999)

    class Trickling(_RangeResponse):
        """Never returns a full chunk, but never returns empty either."""

        def read1(self, n):
            return b"x"  # a trickle

        read = read1

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Trickling(b""))

    partial = tmp_path / "w.incomplete"
    with pytest.raises(TimeoutError) as failure:
        worker._stream_to_file("http://example/w", partial, output_path=tmp_path / "r.json")

    assert "no data for" in str(failure.value)
    assert partial.exists(), "bytes already received must stay on disk for the resume"


def test_read1_is_used_so_the_deadline_can_fire(worker, tmp_path, monkeypatch):
    """Filling a 1 MB chunk with read() can block past any deadline."""
    monkeypatch.setattr(worker, "_remote_size", lambda url: 4)

    class OnlyRead1:
        status = 206
        headers = {}

        def __init__(self):
            self.body = [b"ab", b"cd", b""]

        def read1(self, n):
            return self.body.pop(0)

        def read(self, n):  # pragma: no cover - must not be used
            raise AssertionError("read() blocks until the chunk is full")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: OnlyRead1())

    partial = tmp_path / "w.incomplete"
    assert worker._stream_to_file("http://example/w", partial, output_path=None) == 4
    assert partial.read_bytes() == b"abcd"


def test_progress_reporting_never_calls_out_to_the_network_per_chunk(worker, tmp_path, monkeypatch):
    """A HEAD request per chunk would cost more than the download itself."""
    body = bytes(range(256)) * 4096  # 1 MB
    monkeypatch.setattr(worker, "_remote_size", lambda url: pytest.fail("the network was touched"))
    monkeypatch.setattr(worker, "PROGRESS_REPORT_SECONDS", 0)
    _serve(worker, monkeypatch, body)

    class Trickle(_RangeResponse):
        def __init__(self, b, status=206):
            super().__init__(b, status)
            self.size = len(b)
            self.at = 0

        def read1(self, n):
            step = 4096
            chunk = self._body[self._at : self._at + step]
            self._at += len(chunk)
            return chunk

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Trickle(body))

    output = tmp_path / "r.json"
    partial = tmp_path / "w.incomplete"
    assert worker._stream_to_file("http://example/w", partial, total=len(body), output_path=output) == len(body)


def test_downloading_reports_byte_progress_in_the_phase_file(worker, tmp_path, monkeypatch):
    """A 12+ minute download must be able to say how far it has got."""
    body = bytes(range(256)) * 2048
    _serve(worker, monkeypatch, body)
    monkeypatch.setattr(worker, "PROGRESS_REPORT_SECONDS", 0)

    output = tmp_path / "result.json"
    partial = tmp_path / "w.incomplete"
    worker._stream_to_file("http://example/w", partial, total=len(body), output_path=output)

    from app.ai_process import read_phase, read_phase_detail

    assert read_phase(output) == "downloading-model"
    assert f"{len(body)}/{len(body)}" in read_phase_detail(output)