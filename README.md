# SongHub Karaoke MIDI Builder

A local web app that builds karaoke MIDI files (`.mid`) with musical notes and
timestamped lyrics, for use in **SongHub Karaoke PC**.

**Status: Milestone 2.** You can upload an MP3 or WAV, enter the song title and
singer, paste lyrics, and correct the timing of every word and note by hand in
editable tables. The app then exports a karaoke MIDI for SongHub.

**This milestone does not listen to the audio.** It does not work out notes,
lyrics, or an arrangement for you. The lyric timing table and the MIDI note
table are manual tools — and they remain the way you correct anything a later
transcription milestone gets wrong.

Personal use only. Process only audio and lyrics you are authorized to use.

---

## Requirements

- **Windows**
- **Python 3.11.x** — see below, this is not optional
- No network access is needed at runtime. No audio is uploaded anywhere.

### Why Python 3.11 and not 3.14

A later milestone uses Spotify Basic Pitch, which picks its backend by
environment marker:

```
tensorflow<2.15.1 ; platform_system != "Darwin" and python_version >= "3.11"
onnxruntime       ; platform_system == "Windows"  and python_version <  "3.11"
```

TensorFlow 2.15.0 publishes **cp39 / cp310 / cp311 Windows wheels only** — none
for cp312 or newer. On Python 3.14 `pip install basic-pitch` fails outright.
Only **3.10 and 3.11** work, and 3.10 reaches end of life in October 2026, so
this project is pinned to **3.11**.

Do not "modernise" the virtual environment to a newer Python.

---

## Setup

Run these in PowerShell from the project folder.

```powershell
# 1. Create the virtual environment (must be Python 3.11)
py -3.11 -m venv .venv

# 2. Activate it
.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt
```

If PowerShell blocks activation, either run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or skip activation
and call the interpreter directly: `.venv\Scripts\python.exe ...`

## Run

```powershell
# Start the web app
streamlit run streamlit_app.py

# Run the automated tests
pytest -q
```

Or without activating, using the venv interpreter directly:

```powershell
.venv\Scripts\python.exe -m streamlit run streamlit_app.py
.venv\Scripts\python.exe -m pytest -q
```

Streamlit prints a local URL (usually <http://localhost:8501>).

---

## Using the app

The page is one numbered flow from top to bottom.

1. **Step 1 - Song audio.** Upload an MP3 or WAV (50 MB limit). The filename and
   duration appear, with a player. This is optional: the audio is only used for
   preview and to warn you when lyrics run past the end.
2. **Step 2 - Song title and singer.** SongHub shows these as
   `[Song title] - [Singer]`.
3. **Step 3 - Lyrics.** Paste your lyrics, one line of the song per line, then
   press **Split lyrics into rows**. Choose *words* (safer) or *syllables*
   (a rough guess — it gets "hello" right and "karaoke" wrong; every row stays
   editable).
4. **Step 4 - Check and correct the timings.** One row per word or syllable with
   its own start and end time in seconds. Edit any cell. Errors appear with the
   row number and what to do about them.
5. **Step 5 - MIDI notes.** Type the notes yourself. Pitch 60 is middle C,
   channel 0 is treated as the vocal melody. **Leaving this empty is fine** and
   exports a lyrics-only file.
6. **Step 6 - Preview and export.** The lyric sequence is shown next to the audio
   player so you can check the timing against the music. Press **Generate MIDI
   file**, then **Download MIDI**.

**Load demo song** (under the intro) fills everything with a short public-domain
melody, which is the easiest way to see the whole flow working.

### What the app checks for you

Export is blocked until the tables are usable. It refuses to produce a file when:

- there are no lyrics at all
- a time is blank, not a number, or infinite
- a time is negative (times start at 0)
- a row ends before it starts
- rows are out of time order — it names the two words rather than silently
  reordering them, which would scramble the lyric text

### Temporary files

Uploaded audio is written to `data/workspace/` and removed when you press
**Remove audio** or start the app again. Deletion is restricted to that folder:
the code refuses to delete the workspace itself or anything outside it.

---

## Testing in SongHub — required

> **A file that is valid MIDI does not prove SongHub will display it correctly.**
> Structural validation cannot prove compatibility. Only playing the file in
> SongHub can.

### Manual test procedure

1. Copy the downloaded `.mid` into your SongHub songs folder.
2. Play it in SongHub Karaoke PC.
3. Confirm each of the following:

| # | Check | Expected |
|---|-------|----------|
| 1 | Title shown | `Twinkle Twinkle Little Star - Demo Singer` |
| 2 | Lyrics visible | four lines of text appear |
| 3 | Highlighting | each word highlights as it is sung |
| 4 | Melody audible | the tune is recognisable |
| 5 | Bass audible | a low note every phrase (if included) |
| 6 | Lyric offset | changing the slider visibly moves the highlight |

### Results recorded so far

The demo content has been played in SongHub and confirmed working:

- Title displays as `[Song name] - [Singer]`
- Lyrics display and highlight **in time**
- Melody is audible, recognisable, and plays through
- Bass sounds on every phrase

### "Lyrics appear too early" — check SongHub's settings first

This is **not** a defect in the generated file. SongHub has a player setting:

> **Reveal lyrics before start: 8000 ms** (default; must be greater than the
> 5 s countdown)

SongHub deliberately displays the lyric text several seconds **before** the
music begins. Lower that value in SongHub's settings if you want the text to
appear later.

Four structurally different layouts (`s1`-`s4`, including one that reproduces
the layout of known-good karaoke files exactly) all behaved **identically** in
SongHub, which confirms the perceived earliness comes from the player rather
than the file.

**Leave the lyric offset at 0.** It exists only to correct genuinely mistimed
individual words. A large offset interacts with SongHub's countdown and can make
audio start before the countdown has finished.

---

## How the MIDI is built

- One **lyric meta event (`FF 05`)** per word, timed at the moment it is sung.
  Words are individually timed events, so any word's time can be edited
  without touching the rest.
- **`CR` (`0x0D`)** after each word, **`LF` (`0x0A`)** after each line.
- One **tempo event (`FF 51`)** at 120 BPM, with `480` ticks per beat.
- Absolute ticks are computed from seconds, events are sorted by time, then
  converted to delta times. Delta times in MIDI can never be negative, so the
  ordering step is mandatory, not cosmetic.
- Exported bytes are re-parsed by an **independent** parser
  (`app/validate_midi.py`). Re-reading with Mido would not be a real check,
  because writer and reader would agree on the same wrong bytes.

### A bug worth knowing about

Notes once carried `time=0` and were written only as a *sort key*. Every note
landed on tick 0 and played simultaneously — which sounds like exactly one note
followed by silence. Counts were correct; the timeline was destroyed.

The lesson is now enforced by tests: **assert timings, not just counts**, for
anything a person has to perceive.

---

## Project structure

```
app/
  karaoke_dialects.py   swappable lyric encodings; read this first
  lyrics.py             lyric text, timing model, syllable splitting
  midi_build.py         notes + lyrics -> .mid
  validate_midi.py      independent raw-byte SMF parser
  export_service.py     build -> validate -> bytes (single entry point)
  demo.py               the demo song used by the UI and the tests
  audio.py              upload checks, duration probing, safe temp storage
  timing.py             lyric table rows and their validation
  notes.py              note table rows and their validation
  tables.py             DataFrame <-> row conversion for st.data_editor
  inspect_midi.py       structural report for any .mid file
  make_*_samples.py     generated calibration files for manual testing
streamlit_app.py        the web page
tests/                  158 automated tests
```

`tests/test_m2_app_smoke.py` runs the Streamlit script in-process with
`AppTest`, so a crash anywhere on the page fails the test run rather than
waiting to be discovered in a browser.

### Calibration files

`out/` holds generated samples for manual testing. Regenerate, never hand-edit:

```powershell
python -m app.make_m1_samples       # lyric encoding variants
python -m app.make_note_samples     # note playback variants
python -m app.make_offset_samples   # lyric offset calibration
```

---

## Not in this milestone

- **Audio-to-MIDI transcription** (Basic Pitch) — the note table is manual
- **Vocal/instrument separation** (Demucs)
- **Automatic lyric transcription and timing** — the timing table is manual
- Any upload of your audio to a server: processing is entirely local

None of Demucs, Basic Pitch or any other heavyweight AI library is installed.

---

## Legal

Generated notes are **estimates of a melody**, not the original recording, and a
full arrangement will often be wrong. This project never uploads audio, never
scrapes lyrics, and never downloads commercial karaoke files. Use it only with
material you are authorized to use, for personal use.