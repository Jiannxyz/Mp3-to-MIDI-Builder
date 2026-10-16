# SongHub Karaoke MIDI Builder

A local web app that builds karaoke MIDI files (`.mid`) with musical notes and
timestamped lyrics, for use in **SongHub Karaoke PC**.

**Status: Milestones 1-5.** Upload an MP3 or WAV, enter the song title and
singer, paste lyrics, correct every word's timing by hand, and optionally
**estimate notes** (Basic Pitch), **separate the song into stems** (Demucs) and
**draft lyrics** (faster-whisper).

**The AI is optional and does not pretend to be a musician.** Every model here
produces a guess. Basic Pitch is a note tracker, not an arranger: on a complete
mix it returns a rough sketch, not a transcription. Speech recognition is trained
on speech, not singing, so drafted words and timings are usually wrong. All of
it is labelled, and all of it is editable.

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
3. **Step 3 - Lyrics.** Paste your own lyrics, one line of the song per line,
   then press **Split lyrics into rows**. Choose *words* (safer) or *syllables*
   (a rough guess — it gets "hello" right and "karaoke" wrong; every row stays
   editable). **This is the primary path and always works.**
   *Generate a draft for me* is an optional extra that pre-fills the same box
   with a guess you can overwrite or discard.
4. **Step 4 - Check and correct the timings.** One row per word or syllable with
   its own start and end time in seconds. Edit any cell. Errors appear with the
   row number and what to do about them.
5. **Step 5 - MIDI notes.** Type the notes yourself. Pitch 60 is middle C,
   channel 0 is treated as the vocal melody. **Leaving this empty is fine** and
   exports a lyrics-only file.
6. **Step 6 - Separate the song into stems (optional).** Splits the audio into
   vocals / bass / drums / other with Demucs. You can listen to each stem, and
   Step 7 will then offer them as sources for note estimation. Your original
   file is never modified.
7. **Step 7 - Estimate notes from the audio (optional).** Estimates notes with
   Basic Pitch, from the whole song or from a separated stem. Shows them in a
   review table so you can untick the ones you do not want before adding them
   to Step 5. See the warnings above.
8. **Step 8 - Preview and export.** The lyric sequence is shown next to the audio
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

## Optional AI extras (note estimation, stem separation, lyric drafting)

Everything above works without these. Each is optional on its own, and the app
tells you which ones are installed rather than failing.

```powershell
py -3.11 -m venv .venv-ai
.venv-ai\Scripts\python.exe -m pip install -r requirements-ai.txt
```

Restart the app. It finds `.venv-ai` automatically.

**Python 3.10 or 3.11 only.** `basic-pitch` picks its backend from a version
marker, and TensorFlow 2.15.0 publishes cp39/cp310/cp311 Windows wheels only —
it cannot install on 3.12 or newer.

| Extra | Pulls in | Used for | Cost |
|---|---|---|---|
| `basic-pitch` | TensorFlow (~500 MB) | Step 7 note estimates | heavy |
| `demucs` | `torch>=2.1` | Step 6 stem separation | **heaviest** |
| `faster-whisper` | CTranslate2, **no torch** | Step 3 lyric drafts | light-ish |

For a CPU-only machine, install a smaller torch build first:

```powershell
.venv-ai\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv-ai\Scripts\python.exe -m pip install demucs
```

> **If Step 7 fails with `No module named 'pkg_resources'`:** `resampy`, a
> basic-pitch dependency, imports `pkg_resources`, which setuptools removed in
> v81. Fix it with
> `.venv-ai\Scripts\python.exe -m pip install "setuptools<81"`.
> The pin is in `requirements-ai.txt`. A `pkg_resources is deprecated` warning
> afterwards is noisy and harmless.

**FFmpeg** is separate and is *not* a Python package. It is needed to read MP3
and other compressed formats for separation or drafting. WAV works without it,
and MIDI export never needs it.

### First run needs a network connection

The three models are obtained differently, which decides whether a feature works
offline:

| Feature | Model download | Works offline afterwards |
|---|---|---|
| Note estimation (Basic Pitch) | **bundled in the pip package** | yes, always |
| Stem separation (Demucs) | ~80 MB from HuggingFace on first use | yes, after |
| Lyric drafting (faster-whisper) | ~75 MB (`tiny`) to ~480 MB (`small`) on first use | yes, after |

A first model download can take a while. The app checks the download speed
*before* starting, so you get one of three answers in a few seconds:

- **Already cached** — nothing is downloaded and no network call is made.
- **Feasible** — it starts, and shows how many bytes of how many have arrived.
- **Refused** — it says your measured speed, the file size, and the time it
  would take, instead of leaving you watching a progress bar that never moves:

  > Your connection downloads this model at 0.04 MB/s. 955717e8.safetensors is
  > 80 MB, which would take about 34 minutes here. The download can be resumed,
  > so it will pick up where it stopped if you run it again.

Stem separation downloads the weights itself with resumable HTTP range requests,
because connections on a slow link get reset part-way through (you may see
`ConnectionResetError: [WinError 10054]` in a terminal). Progress survives a
reset, so an interrupted download continues instead of starting over. If the
transfer stops making progress for a minute, it is resumed too.

If the download is stalled entirely, the app notices after about 90 seconds and
stops, rather than leaving you waiting. You will see:

> The model download made no progress for 90 seconds and was stopped. This is a
> blocked or stalled download, not slow processing. Your network reaches the
> site but not the model files.

Either run that step once on a network that allows the download (the model is
then cached and works offline), copy your `.cache\huggingface\hub` folder from a
machine that has it, or skip the step — the rest of the app works without it.

To check what has been fetched so far:

```powershell
Get-ChildItem "$env:USERPROFILE\.cache\huggingface\hub" -Recurse -File |
  Measure-Object Length -Sum
```

### How good are the results?

Honestly: they are starting points, not answers.

- **Note estimates** — Basic Pitch reports onsets and pitches. It does not
  identify instruments, so a full mix produces a simplified melody line, usually
  without bass or chords. It works far better on **one instrument at a time**
  than on a complete song. Because of this, Step 7 groups notes by **pitch**
  (bass / mid / treble) so you can keep a melody and discard the rest.
- **Separated stems** are **not clean isolated instruments**. Each stem still
  contains bleed from the others, so an estimate from a stem is still
  approximate. Separation helps mainly by giving a note tracker one
  single-line source instead of a mix — it does not make the result correct.
  The app recommends the **vocals** stem when present, because a sung melody is
  monophonic, and refuses to offer **drums**, which are percussive.
- **Drafted lyrics** are the least reliable of the three. Speech recognition is
  trained on speech, not singing, so expect wrong words, wrong spelling and
  roughly-plausible timings. The draft is written into the normal lyrics box so
  you can overwrite it entirely, and its timings are labelled ESTIMATED.

Estimated notes are marked `estimated` in the Step 5 table and never overwrite
your lyric timings. Your original audio is never modified or deleted.

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
  transcription/
    contract.py         NoteEstimate / TranscriptionResult - the model contract
    service.py          facade used by the UI (availability, orchestration)
    bridge.py           runs the worker in a separate interpreter
    worker.py           the ONLY module importing basic_pitch
    audio_convert.py    makes any input a WAV Basic Pitch can read
  separate/
    contract.py         StemInfo / SeparationResult - the model contract
    service.py          facade used by the UI
    worker.py           the ONLY module importing demucs
  draft_lyrics/
    contract.py         DraftWord / LyricDraft - the model contract
    service.py          facade used by the UI
    worker.py           the ONLY module importing faster_whisper
  ai_process.py         shared subprocess protocol for all three services
streamlit_app.py        the web page
tests/                  310 automated tests
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

- **Automatic lyric timing** — drafted timings are estimates and must be
  corrected by hand in Step 4
- Anything that contacts an external service: no lyric lookups, no song
  downloads, no uploads. All three models run on this machine

None of Basic Pitch, Demucs or faster-whisper is installed by default; they are
opt-in in `requirements-ai.txt`.

---

## Legal

Generated notes are **estimates of a melody**, not the original recording, and a
full arrangement will often be wrong. This project never uploads audio, never
scrapes lyrics, and never downloads commercial karaoke files. Use it only with
material you are authorized to use, for personal use.