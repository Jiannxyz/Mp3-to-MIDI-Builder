# SongHub Karaoke MIDI Builder

A local web app that builds karaoke MIDI files (`.mid`) with musical notes and
timestamped lyrics, for use in **SongHub Karaoke PC**.

**Status: Milestone 1.** This milestone generates a fixed demo song so the MIDI
structure can be inspected and tested. It does **not** read your MP3 or WAV yet,
and it does **not** transcribe audio. Those come in later milestones.

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

1. Click **Generate demo MIDI**.
2. The app builds the file, re-parses the bytes it just produced, and reports
   whether the structure verified.
3. Click **Download MIDI**.

Optional controls:

- **Lyric offset (ms)** — shifts the lyric events only; notes never move.
  Positive delays the highlight, negative advances it.
- **Include bass track** — export the melody alone or melody plus bass.

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
- Melody is audible and plays through

One open question remains: **SongHub has been observed highlighting text
before the word is actually sung.** The lyric offset slider exists to measure
this. While testing, note whether the lead is **constant** or whether the text
falls **progressively further behind** as the song goes on — a growing lead
would mean SongHub is ignoring our tempo map, which an offset cannot fix.

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
  lyrics.py             lyric text and per-word timing model
  midi_build.py         notes + lyrics -> .mid
  validate_midi.py      independent raw-byte SMF parser
  export_service.py     build, validate and serialise (single entry point)
  demo.py               the demo song used by the UI and the tests
  make_*_samples.py     generated calibration files for manual testing
streamlit_app.py        the web page
tests/                  43 automated tests
```

### Calibration files

`out/` holds generated samples for manual testing. Regenerate, never hand-edit:

```powershell
python -m app.make_m1_samples       # lyric encoding variants
python -m app.make_note_samples     # note playback variants
python -m app.make_offset_samples   # lyric offset calibration
```

---

## Not in this milestone

- Reading MP3/WAV, duration display, audio preview
- Audio-to-MIDI transcription (Basic Pitch)
- Vocal/instrument separation (Demucs)
- Automatic lyric transcription
- Uploading, and the timing editor UI

None of Demucs, Basic Pitch or any other heavyweight AI library is installed.

---

## Legal

Generated notes are **estimates of a melody**, not the original recording, and a
full arrangement will often be wrong. This project never uploads audio, never
scrapes lyrics, and never downloads commercial karaoke files. Use it only with
material you are authorized to use, for personal use.