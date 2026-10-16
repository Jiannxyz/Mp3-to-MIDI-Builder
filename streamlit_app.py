"""SongHub Karaoke MIDI Builder - local web app.

Milestone 2: upload audio, enter song details, split lyrics into editable rows,
edit timings and notes by hand, and export a karaoke MIDI for SongHub.

This milestone does NOT derive notes or lyrics from the audio. The note and
timing tables are manual tools and are the fallback whenever automatic
transcription is unavailable or wrong.

Everything runs locally. Audio is written to a temporary folder, used, and
removed. Nothing is uploaded anywhere.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from app.audio import (
    ALLOWED_EXTENSIONS,
    DEFAULT_WORKSPACE,
    AudioError,
    format_duration,
    probe_duration,
    safe_cleanup,
    save_upload,
)
from app.demo import build_demo_song, demo_lyric_rows, demo_note_rows
from app.export_service import export_song_to_bytes
from app.karaoke_dialects import DEFAULT_DIALECT
from app.notes import NoteRow, example_note_rows, rows_to_note_tracks, validate_note_rows
from app.draft_lyrics import (
    MODEL_CHOICES as DRAFT_MODEL_CHOICES,
    DraftError,
    check_availability as draft_availability,
    draft_lyrics,
    draft_to_lyric_rows,
)
from app.separate import (
    DEFAULT_MODEL as SEPARATION_MODEL,
    SeparationError,
    check_availability as separation_availability,
    separate_stems,
)
from app.tables import (
    estimate_rows_to_frame,
    frame_to_estimates,
    frame_to_lyric_rows,
    frame_to_note_rows,
    lyric_frame_for,
    lyrics_preview_frame,
    note_frame_for,
)
from app.timing import LyricRow, group_by_line, respace, rows_from_text, rows_to_lyric_lines, validate_rows
from app.transcription import TranscriptionError, check_availability, estimate_notes
from app.transcription.service import assign_groups, estimates_to_note_rows

st.set_page_config(page_title="SongHub Karaoke MIDI Builder", page_icon="🎤", layout="wide")

TITLE_MAX = 120
DEFAULT_BPM = 120


# ---------------------------------------------------------------- helpers


def workspace() -> Path:
    base = Path(DEFAULT_WORKSPACE)
    base.mkdir(parents=True, exist_ok=True)
    if "workspace" not in st.session_state:
        st.session_state.workspace = Path(tempfile.mkdtemp(prefix="kmb-", dir=base))
    return st.session_state.workspace


def clear_audio() -> None:
    """Remove the temporary audio folder and forget the upload."""
    directory = st.session_state.pop("workspace", None)
    if directory is not None:
        safe_cleanup(Path(directory), Path(DEFAULT_WORKSPACE))
    st.session_state.pop("audio", None)


def reset_editor(state_key: str) -> None:
    """Bump the version so the matching data_editor starts clean."""
    st.session_state[state_key] = st.session_state.get(state_key, 0) + 1


def show_problems(problems: list[str], title: str) -> None:
    if not problems:
        return
    st.error(f"{title} ({len(problems)})")
    for problem in problems[:12]:
        st.write(f"- {problem}")
    if len(problems) > 12:
        st.caption(f"... and {len(problems) - 12} more.")


# ---------------------------------------------------------------- page

st.title("SongHub Karaoke MIDI Builder")
st.caption(
    "Build a karaoke MIDI file with timed lyrics for SongHub Karaoke PC. "
    "Everything runs on this computer."
)

st.info(
    "**Notes and timings are entered by hand in this milestone.** The app does "
    "not listen to the audio or work out an arrangement for you. The note table "
    "and timing table are the tools you use now, and they remain the way you "
    "correct anything a later transcription milestone gets wrong."
)

# A blocked model download is easy to walk into repeatedly, and the failing step
# is far down the page. Remember it for the session and say so up here, rather
# than letting the user rediscover it by clicking each AI step.
if st.session_state.get("ai_blocked"):
    st.error(
        st.session_state["ai_blocked"]
        + "  The steps that need model downloads are Steps 6 and 7. Everything "
        "else - uploading, lyrics, the timing table, the note table and the MIDI "
        "export - works without them."
    )


def note_blocked(error) -> None:
    """Remember a blocked model download so later steps can warn immediately."""
    if getattr(error, "kind", "") == "stalled-download":
        st.session_state["ai_blocked"] = str(error)

with st.expander("Using the demo song instead of your own audio"):
    st.write(
        "If you just want to see how the exporter works, press **Load demo "
        "song**. It uses a short public-domain melody so the timing is easy to "
        "judge by ear."
    )
    if st.button("Load demo song", key="load_demo"):
        demo = build_demo_song()
        st.session_state.lyric_rows = demo_lyric_rows()
        st.session_state.note_rows = demo_note_rows()
        st.session_state.song_title = demo.title
        st.session_state.song_artist = demo.artist
        st.session_state.audio_duration = demo.duration_seconds
        st.session_state.audio_name = "(demo - no audio file)"
        reset_editor("lyric_version")
        reset_editor("note_version")
        st.success("Demo loaded. Scroll down to Step 4.")

# ---- Step 1: audio -------------------------------------------------------

st.divider()
st.subheader("Step 1 - Song audio (optional)")

uploaded = st.file_uploader(
    "Choose an MP3 or WAV file",
    type=sorted(ext.lstrip(".") for ext in ALLOWED_EXTENSIONS),
    help=(
        "Optional in this milestone: the audio is only used for preview and to "
        "warn you when lyrics run past the end. Nothing is uploaded anywhere."
    ),
    key="uploader",
)

if uploaded is not None:
    previous = st.session_state.get("audio_name")
    try:
        directory = workspace()
        path = save_upload(uploaded.getvalue(), uploaded.name, directory)
        duration = probe_duration(path)
    except AudioError as error:
        st.error(str(error))
    else:
        if previous and previous != uploaded.name:
            clear_audio()
            directory = workspace()
            path = save_upload(uploaded.getvalue(), uploaded.name, directory)
            duration = probe_duration(path)

        st.session_state.audio = path
        st.session_state.audio_name = uploaded.name
        st.session_state.audio_duration = duration
        st.success(f"Saved {uploaded.name}")

audio_name = st.session_state.get("audio_name")
audio_duration = st.session_state.get("audio_duration")

if audio_name:
    left, right = st.columns([2, 1])
    with left:
        st.markdown(f"**File:** `{audio_name}`")
        if st.session_state.get("audio"):
            st.audio(
                str(st.session_state["audio"]),
                alt=f"Audio player for {audio_name}",
            )
    with right:
        st.metric("Duration", format_duration(audio_duration))
        if st.button("Remove audio", key="clear_audio"):
            clear_audio()
            st.rerun()

# ---- Step 2: song details ------------------------------------------------

st.divider()
st.subheader("Step 2 - Song title and singer")

title_col, artist_col = st.columns(2)
song_title = title_col.text_input(
    "Song title",
    value=st.session_state.get("song_title", ""),
    max_chars=TITLE_MAX,
    key="song_title",
)
song_artist = artist_col.text_input(
    "Singer",
    value=st.session_state.get("song_artist", ""),
    max_chars=TITLE_MAX,
    key="song_artist",
)
st.caption(
    "SongHub shows this as `[Song title] - [Singer]`. Both fields can be edited "
    "later; the MIDI keeps whatever you type here."
)

# ---- Step 3: lyrics ------------------------------------------------------

st.divider()
st.subheader("Step 3 - Lyrics")

# A draft (or a discard) sets this, then reruns. It is consumed HERE, before the
# text_area is created, because assigning to a widget's key after that widget
# already exists this run is discarded by Streamlit - the widget state manager
# keeps the old value and overwrites it on the next render.
if "pending_lyrics_text" in st.session_state:
    st.session_state["lyrics_text"] = st.session_state.pop("pending_lyrics_text")

lyrics_text = st.text_area(
    "Paste your lyrics",
    value=st.session_state.get("lyrics_text", ""),
    height=220,
    key="lyrics_text",
    help="One line of the song per line. Punctuation is kept as part of the word.",
)

split_col, mode_col = st.columns([1, 2])
split_mode = mode_col.radio(
    "Split each word into",
    options=["words", "syllables"],
    horizontal=True,
    key="split_mode",
    help=(
        "'words' is safer. 'syllables' is a rough guess and will be wrong on "
        "some words - every row stays editable, so you can fix it."
    ),
)

if split_col.button("Split lyrics into rows", type="primary", key="split_lyrics", width="stretch"):
    st.session_state.lyric_rows = rows_from_text(lyrics_text, mode=split_mode)
    reset_editor("lyric_version")
    st.success(f"Created {len(st.session_state.lyric_rows)} rows.")

# --- optional draft generation (Milestone 5) ------------------------------

with st.expander("Generate a draft for me (optional)"):
    st.warning(
        "**A draft is a guess.** Speech recognition is trained on speech, not "
        "singing, so the words, spelling and timings will often be wrong. Treat "
        "it as a starting point, then replace it with your own lyrics - the box "
        "above is always yours to edit."
    )

    draft_status = draft_availability()
    st.caption(f"Speech recognition status: **{draft_status.backend_label}**")

    if not draft_status.available:
        st.info(
            f"{draft_status.reason} Nothing is lost - just type or paste your own "
            "lyrics above and press **Split lyrics into rows**."
        )
        st.code(
            ".venv-ai\\Scripts\\python.exe -m pip install faster-whisper",
            language="powershell",
        )
    elif not st.session_state.get("audio"):
        st.info("Upload an audio file in Step 1 first, then come back to this step.")
    else:
        model_col, lang_col = st.columns(2)
        draft_model = model_col.selectbox(
            "Model",
            options=list(DRAFT_MODEL_CHOICES),
            index=1,
            format_func=lambda choice: DRAFT_MODEL_CHOICES[choice],
            key="draft_model",
        )
        draft_language = lang_col.text_input(
            "Language code",
            value="",
            placeholder="leave blank to detect",
            help="Forcing a language can improve accuracy a little. Leave blank and it will guess.",
        )

        if st.button("Generate draft lyrics", key="make_draft"):
            with st.status("Listening for lyrics...", expanded=True) as status:
                try:
                    st.write("Loading the transcription model (first run downloads it)...")
                    result = draft_lyrics(
                        Path(st.session_state["audio"]),
                        workspace() / "draft",
                        model_name=draft_model,
                        language=draft_language.strip() or None,
                    )
                    if result.lines:
                        st.session_state["lyric_rows"] = draft_to_lyric_rows(result)
                        st.session_state["draft_summary"] = result.summary
                        st.session_state["draft_warnings"] = list(result.warnings)
                        st.session_state["pending_lyrics_text"] = result.text
                        reset_editor("lyric_version")
                        st.rerun()
                    else:
                        status.update(label="Nothing recognised", state="complete")
                        st.warning(
                            "No speech was recognised. That is common with singing - "
                            "please type or paste your own lyrics above."
                        )
                except DraftError as error:
                    status.update(label="Could not draft lyrics", state="error")
                    note_blocked(error)
                    st.error(str(error))
                    if error.hint:
                        st.info(error.hint)
                except Exception as error:  # noqa: BLE001 - the UI must not crash
                    status.update(label="Could not draft lyrics", state="error")
                    note_blocked(error)
                    st.error(f"Lyric drafting failed unexpectedly: {error}")

    if st.session_state.get("lyrics_text") and st.session_state.get("draft_summary"):
        st.info(
            f"The box above holds a DRAFT ({st.session_state['draft_summary']}). "
            "Its timings are ESTIMATED. Correct them in Step 4."
        )
        for warning in st.session_state.get("draft_warnings", []):
            st.caption(warning)
        if st.button("Discard the draft and clear the box", key="discard_draft"):
            st.session_state["lyric_rows"] = []
            st.session_state["pending_lyrics_text"] = ""
            st.session_state.pop("draft_summary", None)
            st.session_state.pop("draft_warnings", None)
            reset_editor("lyric_version")
            st.rerun()

# ---- Step 4: timing ------------------------------------------------------

st.divider()
st.subheader("Step 4 - Check and correct the timings")

st.caption(
    "Every row is one word or syllable with its own times. Change the numbers "
    "directly in the table. Times are in seconds from the start of the song."
)

source_rows = st.session_state.get("lyric_rows", [])
lyric_frame = lyric_frame_for(source_rows)

edited_lyrics = st.data_editor(
    lyric_frame,
    key=f"lyric_editor_{st.session_state.get('lyric_version', 0)}",
    num_rows="dynamic",
    hide_index=True,
    width="stretch",
    column_config={
        "line": st.column_config.NumberColumn("Line", min_value=1, step=1, help="Rows with the same number are sung together."),
        "text": st.column_config.TextColumn("Word", required=True, help="The text shown on screen."),
        "start (s)": st.column_config.NumberColumn("Start (s)", format="%.3f", step=0.01, min_value=0.0),
        "end (s)": st.column_config.NumberColumn("End (s)", format="%.3f", step=0.01, min_value=0.0),
        "word end": st.column_config.CheckboxColumn("Last part of word?", help="Adds a CR after this row so SongHub knows the word is finished."),
    },
)

lyric_rows = frame_to_lyric_rows(edited_lyrics)
st.session_state.lyric_rows = lyric_rows

if st.button("Reset spacing to even timings", key="respace"):
    st.session_state.lyric_rows = respace(lyric_rows)
    reset_editor("lyric_version")
    st.rerun()

lyric_problems = validate_rows(lyric_rows)
show_problems(lyric_problems, "Lyric timings need attention")

if lyric_rows and not lyric_problems:
    st.caption(f"{len(lyric_rows)} rows, ending at {max(r.end for r in lyric_rows):.2f}s.")
    if audio_duration and max(r.end for r in lyric_rows) > audio_duration + 1.0:
        st.warning(
            f"The last lyric ends at {max(r.end for r in lyric_rows):.2f}s but the "
            f"audio is only {audio_duration:.2f}s long. Check the numbers."
        )

# ---- Step 5: notes -------------------------------------------------------

st.divider()
st.subheader("Step 5 - MIDI notes")

st.caption(
    "Type the notes yourself. Pitch 60 is middle C, 0 is the lowest and 127 the "
    "highest. Channel 0 is treated as the vocal melody. "
    "**Leaving this table empty is fine** - the exported file will contain only "
    "lyrics."
)

note_frame = note_frame_for(st.session_state.get("note_rows") or [])

edited_notes = st.data_editor(
    note_frame,
    key=f"note_editor_{st.session_state.get('note_version', 0)}",
    num_rows="dynamic",
    hide_index=True,
    width="stretch",
    column_config={
        "channel": st.column_config.NumberColumn("Channel", min_value=0, max_value=15, step=1),
        "pitch": st.column_config.NumberColumn("Pitch", min_value=0, max_value=127, step=1),
        "start (s)": st.column_config.NumberColumn("Start (s)", format="%.3f", step=0.05, min_value=0.0),
        "duration (s)": st.column_config.NumberColumn("Length (s)", format="%.3f", step=0.05, min_value=0.0),
        "velocity": st.column_config.NumberColumn("Velocity", min_value=1, max_value=127, step=1),
        "source": st.column_config.TextColumn(
            "Source",
            help="'manual' = typed by you. 'estimated' = guessed by the note estimator. Estimated notes are approximations and need checking.",
        ),
    },
)

note_rows = frame_to_note_rows(edited_notes)
st.session_state.note_rows = note_rows

if st.button("Add 4 placeholder notes", key="seed_notes"):
    existing = st.session_state.get("note_rows", [])
    st.session_state.note_rows = existing + example_note_rows(4)
    reset_editor("note_version")
    st.rerun()

note_problems = validate_note_rows(note_rows)
show_problems(note_problems, "Notes need attention")
if not note_rows:
    st.info("No notes yet. That is allowed - the file will have lyrics but no melody.")

estimated_count = sum(1 for row in note_rows if row.is_estimated)
if estimated_count:
    st.caption(
        f"{estimated_count} of {len(note_rows)} notes are ESTIMATES. They are guesses, "
        "not a transcription of the recording."
    )

# ---- Step 6: separate stems ---------------------------------------------

st.divider()
st.subheader("Step 6 - Separate the song into stems (optional)")

st.warning(
    "**Separated stems are not clean instruments.** Each one still leaks sound "
    "from the others, so a note estimate taken from a stem is still approximate. "
    "Separating mainly helps because Basic Pitch copes far better with one "
    "monophonic-ish source than with a whole mix."
)

separation_status = separation_availability()
st.caption(f"Demucs status: **{separation_status.backend_label}**")

if not separation_status.available:
    st.info(
        f"{separation_status.reason} You can still estimate notes from the whole "
        "song in Step 7, and enter them by hand in Step 5."
    )
    st.code(
        ".venv-ai\\Scripts\\python.exe -m pip install -r requirements-ai.txt",
        language="powershell",
    )
elif not st.session_state.get("audio"):
    st.info("Upload an audio file in Step 1 first, then come back to this step.")
else:
    st.caption(
        "Your original file is never modified or deleted - the stems are written "
        "to a separate folder."
    )
    if st.button("Separate into stems", key="separate"):
        with st.status("Separating the song...", expanded=True) as status:
            try:
                st.write("Loading the Demucs model (first run downloads it)...")
                st.write("This is much slower than the song's length on a CPU.")
                result = separate_stems(
                    Path(st.session_state["audio"]),
                    workspace() / "stems",
                    model=SEPARATION_MODEL,
                )
                st.session_state["separation"] = result
                status.update(label="Done", state="complete")
                st.success(result.summary)
            except SeparationError as error:
                status.update(label="Could not separate", state="error")
                note_blocked(error)
                st.error(str(error))
                if error.hint:
                    st.info(error.hint)
            except Exception as error:  # noqa: BLE001 - the UI must not crash
                status.update(label="Could not separate", state="error")
                note_blocked(error)
                st.error(f"Stem separation failed unexpectedly: {error}")

separation = st.session_state.get("separation")
if separation is not None:
    st.markdown(f"**Stems created** (model `{separation.model}`, on {separation.device}):")
    for stem in separation.stems:
        badge = "" if stem.is_suitable_for_notes else " - not useful for note tracking"
        st.markdown(f"- `{stem.name}` - {stem.duration:.1f}s{badge}")

    for warning in separation.warnings:
        st.caption(warning)

    with st.expander("Listen to the stems"):
        for stem in separation.stems:
            st.markdown(f"**{stem.name}**")
            if Path(stem.path).exists():
                st.audio(stem.path, alt=f"Preview of the {stem.name} stem")

    recommended = separation.recommended_stem
    if recommended:
        st.success(
            f"Best stem to estimate a melody from: **{recommended.name}**. "
            "A sung melody is single-line, which is what a note tracker handles best."
        )
    else:
        st.info("No stem here is worth running note tracking on. Use the whole song in Step 7.")

    if st.button("Discard these stems", key="drop_stems"):
        st.session_state.pop("separation", None)
        st.rerun()

# ---- Step 7: estimate notes ---------------------------------------------

st.divider()
st.subheader("Step 7 - Estimate notes from the audio (optional)")

st.warning(
    "**Estimated notes are an approximation, not a transcription.** Basic Pitch is "
    "a note tracker, not an arranger. Run on a complete mix it will report a "
    "simplified sketch of what is present: it will not reproduce the bass line, "
    "chords, drums or a full arrangement, and the timing will be approximate. "
    "Treat the result as a starting point you correct, never as finished work."
)

availability = check_availability()
st.caption(f"Note estimation status: **{availability.backend_label}**")

if not availability.available:
    st.info(
        f"{availability.reason} Everything else in the app still works without it - "
        "you can enter notes by hand in Step 5."
    )
    with st.expander("How to enable note estimation"):
        st.markdown(
            """
Basic Pitch pulls in TensorFlow, which is large and pins the Python version, so
it is kept in a separate environment:

```powershell
py -3.11 -m venv .venv-ai
.venv-ai\\Scripts\\python.exe -m pip install -r requirements-ai.txt
```

Restart the app and this step becomes available. Python 3.10 or 3.11 only.

MP3 and other compressed formats also need **FFmpeg** on PATH. WAV works without it.
"""
        )

has_audio = bool(st.session_state.get("audio"))
if availability.available and has_audio:
    source_options = ["Whole song"]
    source_paths = {st.session_state["audio"]: "Whole song"}
    if separation is not None:
        for stem in separation.suitable_stems:
            label = stem.name
            source_options.append(label)
            source_paths[stem.path] = label

    source_col, param_col1, param_col2 = st.columns([2, 1, 1])
    chosen_source = source_col.selectbox(
        "Estimate from",
        source_options,
        key="estimate_source",
        help="A separated stem usually gives a better melody than the full mix.",
    )
    onset = param_col1.slider("Onset sensitivity", 0.1, 0.9, 0.5, 0.05,
                              help="Lower finds more note starts, so more notes.")
    frame = param_col2.slider("Frame sensitivity", 0.1, 0.9, 0.3, 0.05,
                              help="Lower keeps shorter, quieter notes.")

    chosen_path = next((p for p, label in source_paths.items() if label == chosen_source), None)

    if st.button(f"Estimate notes from {chosen_source.lower()}", type="primary", key="estimate"):
        with st.status("Estimating notes...", expanded=True) as status:
            try:
                st.write("Converting the audio to WAV...")
                st.write("Running the note model (the first run downloads it)...")
                st.session_state.estimation = estimate_notes(
                    Path(chosen_path),
                    workspace() / "estimate",
                    onset_threshold=onset,
                    frame_threshold=frame,
                )
                st.session_state.estimate_version = st.session_state.get("estimate_version", 0) + 1
                status.update(label="Done", state="complete")
                st.success(st.session_state["estimation"].summary)
            except TranscriptionError as error:
                status.update(label="Could not estimate notes", state="error")
                note_blocked(error)
                st.error(str(error))
                if error.hint:
                    st.info(error.hint)
                if error.detail:
                    with st.expander("Technical details"):
                        st.code(error.detail[-1500:], language="text")
            except Exception as error:  # noqa: BLE001 - the UI must not crash
                status.update(label="Could not estimate notes", state="error")
                note_blocked(error)
                st.error(f"Note estimation failed unexpectedly: {error}")
elif availability.available and not has_audio:
    st.info("Upload an audio file in Step 1 first, then come back to this step.")

if st.session_state.get("estimation") is not None:
    estimation = st.session_state["estimation"]
    for warning in estimation.warnings:
        st.caption(warning)

    group_col, use_col = st.columns([2, 1])
    group_by = group_col.radio(
        "Group notes by",
        ["pitch", "label"],
        horizontal=True,
        key="estimate_group_by",
        help="Basic Pitch reports no instrument names, so pitch groups (bass / mid / treble) are usually more useful.",
    )
    use_col.metric("Notes found", len(estimation))

    items = assign_groups(estimation.estimates, by=group_by)
    st.caption("Untick the notes you do not want, then add the rest to Step 5.")

    estimate_frame = st.data_editor(
        estimate_rows_to_frame(items),
        key=f"estimate_editor_{st.session_state.get('estimate_version', 0)}",
        num_rows="fixed",
        hide_index=True,
        width="stretch",
        column_config={
            "include": st.column_config.CheckboxColumn("Use", width="small"),
            "#": st.column_config.NumberColumn("#", width="small"),
            "group": st.column_config.TextColumn("Group"),
            "start (s)": st.column_config.NumberColumn("Start (s)", format="%.3f"),
            "end (s)": st.column_config.NumberColumn("End (s)", format="%.3f"),
            "length (s)": st.column_config.NumberColumn("Length (s)", format="%.3f"),
            "pitch": st.column_config.NumberColumn("Pitch"),
            "amplitude": st.column_config.NumberColumn("Confidence", format="%.2f"),
        },
    )
    st.session_state["estimate_frame"] = estimate_frame

    chosen = frame_to_estimates(estimate_frame)
    st.write(f"**{len(chosen)}** notes selected.")

    if st.button(f"Add {len(chosen)} selected notes to Step 5", key="use_estimates", disabled=not chosen):
        existing = st.session_state.get("note_rows", [])
        st.session_state.note_rows = existing + estimates_to_note_rows(chosen)
        reset_editor("note_version")
        st.rerun()

# ---- Step 8: preview and export -----------------------------------------

st.divider()
st.subheader("Step 8 - Preview and export")

if lyric_rows and not lyric_problems:
    preview_col, player_col = st.columns(2)

    with preview_col:
        st.markdown("**Lyric sequence**")
        st.dataframe(lyrics_preview_frame(lyric_rows), hide_index=True, width="stretch")

    with player_col:
        st.markdown("**Audio**")
        if st.session_state.get("audio"):
            st.audio(str(st.session_state["audio"]), alt=f"Audio player for {audio_name}")
        else:
            st.caption("No audio uploaded. Upload one in Step 1 to listen while checking timings.")

st.divider()

ready = not lyric_problems and not note_problems
if not ready:
    st.info("Fix the problems listed above, then the export button will work.")

if st.button(
    "Generate MIDI file",
    type="primary",
    disabled=not ready,
    key="export",
):
    with st.status("Building the MIDI file...", expanded=True) as status:
        try:
            st.write("Checking the lyric rows and notes...")
            lines = rows_to_lyric_lines(lyric_rows)
            tracks = rows_to_note_tracks(note_rows, tempo_bpm=DEFAULT_BPM)

            st.write("Writing note and lyric events...")
            result = export_song_to_bytes(
                title=song_title.strip() or "Untitled Song",
                artist=song_artist.strip(),
                lyric_lines=lines,
                note_tracks=tracks,
                dialect=DEFAULT_DIALECT,
                bpm=DEFAULT_BPM,
            )

            if result.ok:
                st.write("Re-reading the finished file to check it...")
                status.update(label="Done", state="complete")
                st.success(result.summary)
                st.download_button(
                    "Download MIDI",
                    data=result.data,
                    file_name=result.filename,
                    mime="audio/midi",
                    key="download",
                )
                st.caption(
                    "Before trusting this file, copy it into your SongHub songs "
                    "folder and check the title, lyrics and highlighting. A file "
                    "that opens correctly is not proof that SongHub likes it."
                )
                st.session_state.last_export = result
            else:
                status.update(label="Failed", state="error")
                st.error("The file was written but failed its own checks, so please do not use it.")
                for problem in result.problems:
                    st.write(f"- {problem}")
        except Exception as error:  # noqa: BLE001 - the UI must never crash silently
            status.update(label="Failed", state="error")
            st.error(f"Something went wrong while building the file: {error}")

with st.expander("How to test in SongHub"):
    st.markdown(
        """
1. Download the file and copy it into your SongHub songs folder.
2. Play it, and check: does the title show as `[Song title] - [Singer]`?
3. Do the words appear, and does each one highlight as you sing it?
4. Can you hear the notes you entered?
5. SongHub reveals lyrics *before* the song starts by default. If the text
   appears far too early, change **Reveal lyrics before start** in SongHub's
   settings rather than shifting the timings here.
"""
    )