"""Optional local stem separation (Milestone 4), isolated from the core app.

    service.py    facade used by the UI: availability, orchestration, errors
    bridge.py     talks to a separate Python environment
    worker.py     the ONLY module importing demucs; standalone script
    contract.py   StemInfo / SeparationResult - the documented contract

Separation is optional. Without Demucs the app still exports MIDI; the UI
reports the feature as unavailable instead of failing.

Accuracy note: separated stems are NOT clean isolated instruments. Each stem
still contains bleed from the others, so a note estimate taken from a stem is
still an approximation. Separation makes a *better* input for a note tracker -
one monophonic-ish source instead of a full mix - it does not make it correct.
"""

from .bridge import resolve_backend, run_worker
from .contract import (
    DEFAULT_MODEL,
    KNOWN_MODELS,
    PREFERRED_STEM_ORDER,
    SEPARATION_CAVEAT,
    SeparationError,
    SeparationResult,
    StemInfo,
)
from .service import SeparationAvailability, check_availability, separate_stems

__all__ = [
    "DEFAULT_MODEL",
    "KNOWN_MODELS",
    "PREFERRED_STEM_ORDER",
    "SEPARATION_CAVEAT",
    "SeparationAvailability",
    "SeparationError",
    "SeparationResult",
    "StemInfo",
    "check_availability",
    "resolve_backend",
    "run_worker",
    "separate_stems",
]
