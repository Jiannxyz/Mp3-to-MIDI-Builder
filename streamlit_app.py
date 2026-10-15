"""Minimal Streamlit page: generate and download the demo karaoke MIDI.

Run:  .venv\\Scripts\\python.exe -m streamlit run streamlit_app.py
"""

from __future__ import annotations

import streamlit as st

from app.demo import SPACING, build_demo_song
from app.export_service import export_song_to_bytes

st.set_page_config(page_title="SongHub Karaoke MIDI Builder", page_icon="🎤", layout="centered")

st.title("SongHub Karaoke MIDI Builder")
st.caption("Milestone 1 - demo export. Audio-to-MIDI comes in a later milestone.")

st.info(
    "This milestone produces a **fixed demo song** so the MIDI structure can be "
    "checked. It does not read your MP3 or WAV yet, and it does not transcribe "
    "anything."
)

st.warning(
    "Generated notes are **estimates of a melody**, not the original recording. "
    "A valid MIDI file is **not** proof that SongHub will display lyrics "
    "correctly - only playing it in SongHub proves that. See README.md."
)

song = build_demo_song()

left, right = st.columns(2)
with left:
    st.subheader("Demo song")
    st.write(f"**Title:** {song.title}")
    st.write(f"**Singer:** {song.artist}")
    st.write(f"**Tempo:** {song.tempo_bpm} BPM")
    st.write(f"**Duration:** {song.duration_seconds:.1f} s")
    st.write(f"**Note tracks:** {', '.join(t.name for t in song.note_tracks)}")
    st.write(f"**Lyric words:** {len(song.all_syllables)} across {len(song.lyric_lines)} lines")

with right:
    st.subheader("Export settings")
    offset_ms = st.slider(
        "Lyric offset (ms)",
        min_value=-1000,
        max_value=1000,
        value=0,
        step=20,
        help=(
            "Shifts the lyric events only; notes never move. "
            "Positive delays the highlight, negative advances it. "
            "SongHub has been observed highlighting text before the word is sung, "
            "so a small positive value may be needed."
        ),
    )
    include_bass = st.checkbox("Include bass track", value=True)

st.divider()

if st.button("Generate demo MIDI", type="primary"):
    tracks = song.note_tracks if include_bass else [song.note_tracks[0]]
    result = export_song_to_bytes(
        title=song.title,
        artist=song.artist,
        lyric_lines=song.lyric_lines,
        note_tracks=tracks,
        bpm=song.tempo_bpm,
        lyric_offset=offset_ms / 1000.0,
    )

    if result.ok:
        st.success(f"Generated and verified: {result.summary}")
        st.download_button(
            "Download MIDI",
            data=result.data,
            file_name=result.filename,
            mime="audio/midi",
        )
        st.caption(
            "Before trusting this file: copy it into your SongHub songs folder and "
            "check that the title, lyrics and in-time highlighting appear."
        )
    else:
        st.error("Export failed structural validation. Do not use this file.")
        for problem in result.problems:
            st.write(f"- {problem}")

with st.expander("How the file is structured"):
    st.markdown(
        f"""
- One lyric meta event (`FF 05`) per word, timed at the moment it is sung.
- `CR` (`0x0D`) after each word, `LF` (`0x0A`) after each line.
- One `set_tempo` event (`FF 51`) at `{song.tempo_bpm}` BPM, `480` ticks per beat.
- Notes are legato at {SPACING:.2f} s per note so playback is easy to follow.
- Delta times are computed from absolute ticks; events are ordered by time
  before conversion.
"""
    )