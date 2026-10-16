"""Audio-to-MIDI note estimation (Spotify Basic Pitch), isolated from the core app.

Layering, outermost first:

    service.py    facade used by the UI: availability, orchestration, errors
    bridge.py     talks to a separate Python environment (or this one)
    worker.py     the ONLY code that imports basic_pitch; self-contained script
    audio_convert.py   makes any input file a WAV Basic Pitch can read
    types.py      NoteEstimate / TranscriptionResult - the documented contract

The core app never imports basic_pitch, TensorFlow or librosa. `worker.py` is a
standalone script so it can run under a different interpreter with its own
dependencies, which is what keeps `pytest` fast and free of a 500 MB model.

Accuracy note: Basic Pitch is a monophonic-ish note tracker. Run on a full mix
it produces an APPROXIMATION of what is present, not a faithful arrangement.
Nothing here should be presented as a transcription of the song.
"""

from .service import (
    Availability,
    amplitude_to_velocity,
    assign_groups,
    check_availability,
    estimate_notes,
    estimates_to_note_rows,
)
from .contract import (
    TranscriptionError,
    TranscriptionResult,
    NoteEstimate,
    group_estimates,
    pitch_band,
)

__all__ = [
    "Availability",
    "NoteEstimate",
    "TranscriptionError",
    "TranscriptionResult",
    "amplitude_to_velocity",
    "assign_groups",
    "check_availability",
    "estimate_notes",
    "estimates_to_note_rows",
    "group_estimates",
    "pitch_band",
]