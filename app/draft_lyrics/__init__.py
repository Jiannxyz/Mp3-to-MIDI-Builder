"""Optional local lyric drafting (Milestone 5), isolated from the core app.

    service.py    facade used by the UI: availability, orchestration, errors
    bridge.py     talks to a separate Python environment
    worker.py     the ONLY module importing faster_whisper
    contract.py   DraftWord / LyricDraft - the documented contract

This produces a DRAFT, never an authoritative transcription. Speech recognition
is trained on speech, not singing, so words and timings are guesses. The manual
lyric editor is the primary path and is always available, with or without this.

No external API is used and nothing about the audio leaves the machine.
"""

from .bridge import resolve_backend, run_worker
from .contract import (
    DEFAULT_MODEL,
    DRAFT_CAVEAT,
    MODEL_CHOICES,
    DraftError,
    DraftLine,
    DraftWord,
    LyricDraft,
)
from .service import DraftAvailability, check_availability, draft_lyrics, draft_to_lyric_rows

__all__ = [
    "DEFAULT_MODEL",
    "DRAFT_CAVEAT",
    "MODEL_CHOICES",
    "DraftAvailability",
    "DraftError",
    "DraftLine",
    "DraftWord",
    "LyricDraft",
    "check_availability",
    "draft_lyrics",
    "draft_to_lyric_rows",
    "resolve_backend",
    "run_worker",
]
