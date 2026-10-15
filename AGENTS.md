# AGENTS.md

## What this project is

A **local web app (Streamlit) that turns an MP3/WAV song into a karaoke MIDI file**
(`.mid` with notes + timestamped lyrics) for use in **SongHub Karaoke PC**.

Personal use only, on audio and lyrics the user is authorized to use. No audio
is uploaded, no lyrics are scraped, no commercial karaoke files are downloaded.

## Repo state

Git repo on branch `main`, **no remote configured**. Milestone 1 (MIDI builder +
dialect calibration samples) is built and tested. Later milestones are not started.

```
app/karaoke_dialects.py   swappable dialect definitions - read this first
app/lyrics.py             lyric text + timing model
app/midi_build.py         notes + lyrics -> .mid
app/validate_midi.py      INDEPENDENT raw-byte SMF parser (not mido)
app/make_m1_samples.py    generates the 4 calibration .mid files into out/
tests/test_m1.py          22 tests
out/*.mid                 generated samples (untracked; regenerate, don't edit)
```

## Commands

```powershell
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\python.exe -m app.make_m1_samples
```

There is no `requirements.txt` yet — the core venv has only `mido` and `pytest`.
Do not add a dependency manifest, or `pip install` anything, unless asked.

## Python version: 3.11 is mandatory — do NOT "upgrade" to 3.14

`.venv` is **Python 3.11.9**. A 3.14 venv existed here first and was replaced.

`basic-pitch` 0.4.0 selects its backend by environment marker:

```
tensorflow<2.15.1  ;  platform_system != "Darwin" and python_version >= "3.11"
onnxruntime        ;  platform_system == "Windows"  and python_version <  "3.11"
```

TensorFlow 2.15.0 ships **cp39/cp310/cp311 Windows wheels only** — nothing for
cp312+, and none for the latest TF either. So on Windows + 3.14, `pip install
basic-pitch` fails outright. **Python 3.10 and 3.11 are the only working versions.**
Do not "modernise" the venv. Verify wheel availability before choosing any library.

## mido 1.3.3 gotchas (all three have already caused bugs here)

- **`save()` takes no `encoding=`.** Text encoding is `MidiFile.charset`, default
  `latin1`. Set `midi.charset = "utf-8"` or non-Latin lyrics are corrupted.
- **`Message.time` is a DELTA, not absolute.** mido 1.3 has **no public
  `to_abstime`/`to_reltime`** (`mido.midifiles.units` only has `second2tick` /
  `tick2second`). If you build in absolute ticks you must convert explicitly
  before saving, or every timestamp silently accumulates.
- **Format 0 is validated before save.** `mido` raises `type 0 file must have
  exactly 1 track`, so tracks must be merged yourself. Merge in **absolute**
  time, then convert once — merging already-converted deltas scrambles the order.
- Sorting note events: `note_off` must sort **before** `note_on` at the same tick.

## Verified against SongHub Karaoke PC (do not re-litigate)

Tested on real hardware by the user:

- **All four dialects (A/B/C/D) worked.** SongHub is tolerant of format 0/1, of
  lyrics in `FF 05` vs `FF 01`, and of CR/LF separators. Keep the baseline
  `A_fmt1_lyrics_ff05` — it is the most standard, and tolerance is not a reason
  to pick anything stranger. Do not rewrite the builder around a "better" dialect.
- **Title, lyrics, and in-time highlighting all work.** The lyric-event structure
  and timing are confirmed correct. This is the hard part and it is done.
- **RESOLVED: "one note, then silence" was our bug, not a SongHub quirk.** All 23
  notes were being written at tick 0. Root cause below — do not re-diagnose.

## Resolved: "only one note plays" (do not re-diagnose)

Symptom in SongHub: lyrics and highlighting worked perfectly, but the melody was
one sound followed by silence, in every variant including format 0.

Root cause: in `_note_messages` the absolute tick was used **only as a sort key**
and never assigned to the message:

```python
mido.Message("note_on", note=..., velocity=..., channel=...)   # time defaults to 0
```

All 23 notes therefore carried `time=0`, were sorted into a sensible order, then
converted to delta times from a column of zeros. Every note landed on tick 0 and
played simultaneously — which sounds exactly like one note.

Why every automated check missed it: `check_midi` asserted note **counts**
(23 on / 23 off) but never note **times**. Counts were correct; the timeline was
destroyed.

Lessons that generalise:

- **Assert timings, not just counts**, for anything a human has to perceive.
- In mido, a message's `time` *is* its position. Anything used for sorting must
  also be assigned to the message.
- This is the second time only the real player caught what the tests did not.
  Keep `validate_midi.py` byte-level and keep SongHub as the final authority.

`check_midi` now takes `expected_note_ticks` and flags the collapsed-timeline
case explicitly. `test_notes_are_staggered_in_time_not_stacked_on_one_tick` and
`test_note_spacing_matches_the_rhythm` are the regression tests.

`app/make_note_samples.py` generates `n1`-`n4`, which isolate program change,
SMF format, and channel count. They are useful any time note playback is
questioned; regenerate rather than hand-editing `out/`.

## Lyric timing: offset knob, and a question still open

SongHub highlights text **before** the word is sung. `build_midi(lyric_offset=...)`
shifts LYRIC EVENTS ONLY — notes never move, so a positive offset always means
"highlight later". `app/make_offset_samples.py` generates
`o_offset_{0000,0200,0400,0600}ms.mid` to measure the real value by ear.

Still unknown, and it decides the fix:

- **Constant lead** → an offset value solves it; surface an offset control in the UI.
- **Lead that grows during the song** → SongHub is ignoring our tempo map, and an
  offset will only ever be locally correct. Do not ship an offset slider as a fix
  until this is measured.

Two timing traps already hit, both now guarded by tests:

- **`line_gap` must be 0 when words map 1:1 onto notes.** A gap silently shifts
  every word after the first line break away from its note.
- Lyric and note timelines are read independently by SongHub, so they can be
  calibrated separately.

## Karaoke MIDI dialect structure (context)

The SMF spec defines `FF 05` but **not** how to use it; the MIDI Manufacturers
Association publicly acknowledges this gap. Real files disagree (Tune1000 / Soft
Karaoke / Solton).

- The dialect lives in `app/karaoke_dialects.py` as **data, not logic**. Change it
  there, never in the builder.
- `validate_midi.py` parses raw bytes **on purpose**. Re-reading with mido can hide
  a bad encoding, because writer and reader would agree on the same wrong bytes.
- Do not re-run the dialect A/B/C/D calibration; it is settled.

## Legal / product constraints

- Never upload audio, scrape lyrics, or download commercial karaoke files.
- Generated notes are **estimates**, not the original recording, and full
  arrangements will often be wrong. The UI must say so.
- Automatic lyric transcription and timing make mistakes. Manual lyric editing
  and timing correction are mandatory, and timing is **never** to be described
  as accurate without testing it in SongHub.

## Git

Conventional Commits required: `feat(scope):`, `fix(scope):`, `chore:` — lowercase
imperative subject, ~72 chars. Branch naming is not prescribed; follow the prompt.