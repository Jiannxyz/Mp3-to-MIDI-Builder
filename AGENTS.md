# AGENTS.md

## What this project is

A **local web app (Streamlit) that turns an MP3/WAV song into a karaoke MIDI file**
(`.mid` with notes + timestamped lyrics) for use in **SongHub Karaoke PC**.

Personal use only, on audio and lyrics the user is authorized to use. No audio
is uploaded, no lyrics are scraped, no commercial karaoke files are downloaded.

## Repo state

Branch `main`, **no remote configured**. Milestones 1-5 complete: MIDI export
service, Streamlit page (upload, lyrics, optional draft, timing table, note
table, optional stem separation, optional note estimation, export), 310 tests.
Milestone 6+ not started.

```
app/karaoke_dialects.py   swappable dialect definitions; DEFAULT_DIALECT is the one to use
app/lyrics.py             lyric text, timing model, syllable splitting (heuristic)
app/midi_build.py         notes + lyrics -> .mid (absolute ticks -> sorted -> deltas)
app/validate_midi.py      INDEPENDENT raw-byte SMF parser (not mido)
app/export_service.py     build -> validate -> bytes; single entry point for UI and tests
app/demo.py               the fixed demo song (public-domain melody)
app/audio.py              upload checks, duration probing, safe temp workspace
app/timing.py             lyric table rows + validation (pure, no Streamlit)
app/notes.py              note table rows + validation (pure, no Streamlit)
app/tables.py             DataFrame <-> row conversion (pure; keeps streamlit_app.py thin)
app/inspect_midi.py       structural report for any .mid file
app/make_*_samples.py     generated calibration files for manual SongHub testing
app/ai_process.py         SHARED subprocess protocol for all three AI services
app/transcription/        note estimation   (only worker.py imports basic_pitch)
app/separate/             stem separation   (only worker.py imports demucs)
app/draft_lyrics/         lyric drafting    (only worker.py imports faster_whisper)
streamlit_app.py          the web page
tests/test_m4_*.py        separation: contract, worker, errors, original-file safety
tests/test_m5_*.py        drafting: contract, worker, errors, manual fallback
tests/test_m45_app_smoke.py  the real app script, both services mocked/absent
```

## The three AI services: one shape, three models

`ai_process.run_python_worker` is the shared subprocess protocol; each service
adds only its own `contract.py`, `worker.py`, `bridge.py`, `service.py`.

| Service | Model | Heavy dependency | Step |
|---|---|---|---|
| `transcription` | Basic Pitch | TensorFlow ~500 MB | 7 |
| `separate` | Demucs | `torch>=2.1` (heaviest) | 6 |
| `draft_lyrics` | faster-whisper | CTranslate2, **no torch** | 3 |

- **vosk was rejected for drafting**: it publishes no Windows wheels.
  `openai-whisper`/`stable-ts` were rejected because they need torch and give no
  advantage over faster-whisper here.
- **The core app never imports a model library.** Only each `worker.py` does.
- `resolve_backend()` stays a module-level name **in every service's bridge**, so
  tests patch it there. `ai_process` deliberately does not resolve the
  interpreter itself — that was the one design choice needed to keep
  `transcription/bridge.py`'s existing tests passing.

## Probe the real entry point, and name the real culprit

Two bugs from one incident. `import basic_pitch` **succeeds** while
`from basic_pitch.inference import predict` **fails**, because the package
`__init__` only probes optional extras while the real entry point pulls in
`resampy -> pkg_resources` (removed in setuptools 81). So:

- Every `*_available()` probes the **same import the worker performs**, with
  `module_available(python, "basic_pitch.inference", "predict")`. Probing the
  bare package name reported "available" and then every real run failed.
- The workers classify an `ImportError` via `import_error_message`, which uses
  `ModuleNotFoundError.name` to say "basic_pitch is not installed" versus
  "basic_pitch is installed, but 'pkg_resources' could not be imported". The old
  message blamed basic_pitch when basic_pitch was installed fine, which sends
  the user to reinstall something they already have.
- That classifier is **duplicated in all three workers on purpose** so each stays
  standalone (no `from app` at module level). `test_dependency_probe.py` pins
  their behaviour so they cannot drift.
- `requirements-ai.txt` pins `setuptools<81` with the reason.

## Model weights: bundled vs downloaded at run time

This difference decides whether a feature works offline, and it is not obvious:

| Service | Where the model comes from | Offline? |
|---|---|---|
| `transcription` (Basic Pitch) | **Bundled inside the pip wheel** | works offline |
| `separate` (Demucs) | Downloads `htdemucs` from HuggingFace on first run | needs network once |
| `draft_lyrics` (faster-whisper) | Downloads the model on first run | needs network once |

Basic Pitch is verified end to end (440 Hz -> MIDI 69) partly because its model
ships in the package. Demucs and faster-whisper download weights from the
HuggingFace hub on first run.

## Demucs weights: slow and reset, NOT blocked (do not re-diagnose)

An earlier version of this file claimed HuggingFace egress was blocked. **That
was wrong**, and the claim has been measured and disproved. Do not repeat it.

| Question | Answer (measured) |
|---|---|
| Which host? | `huggingface.co`, repo `adefossez/HTDemucs` — **not** `dl.fbaipublicfiles.com` |
| Is it reachable? | Yes. `HEAD` returns 200, 80 102 544 bytes, `accept-ranges: bytes` |
| Does progress happen? | Yes, at **0.04-0.11 MB/s** (30-110 KB/s) |
| Is that enough? | No. 80 MB is **12-33 minutes**, and it resets mid-transfer |

`dl.fbaipublicfiles.com` returns 403, but that URL is Demucs' **legacy fallback
repo**, which 4.x never touches. A 403 there says nothing about the real
download — treating it as the cause was the first wrong turn.

Three separate faults stacked up to look like one hang:

1. **`hf_xet` fails outright.** `hf_hub_download` defaults to Xet, which dies
   with `RuntimeError: Task error: File reconstruction error: CAS Client Error:
   Request middleware error` against `us.aws.cdn.hf.co/xorbs/...`. The worker
   sets `HF_HUB_DISABLE_XET=1` before importing anything.
2. **Plain `huggingface_hub` transfers stall at 0 bytes.** Measured: the process
   held the blob lock for 6+ minutes using 1.1 s of CPU and wrote nothing, while
   a bare `urllib` GET of the same URL moved bytes. That is the 3600 s "hang".
3. **Resets restart the file.** `ConnectionResetError: [WinError 10054]` mid-body
   makes huggingface_hub begin again from zero, and several `.incomplete` blobs
   with different etags accumulate. At 0.04 MB/s a restart never finishes, so
   the symptom is a hang rather than an error.

### The fix: fetch the weights ourselves, resumably

`app/separate/worker.py` downloads the checkpoint with stdlib `urllib` and HTTP
`Range` requests straight into the HuggingFace cache layout, so a dropped
connection costs only the bytes in flight. `hf_hub_download` then finds it and
the normal code path takes over.

- `_stream_to_file` resumes from the partial file's length and **truncates back
  to zero if the server answers 200 instead of 206**, because appending a whole
  body to a partial file would silently corrupt it.
- `fetch_weight` measures what is **on disk** after each attempt, not what the
  last call returned: a connection dropped on the final chunk still completed the
  file, and trusting the return value would retry a finished download until the
  attempts ran out. A regression test caught exactly that.
- `clean_stale_partials` deletes zero-byte `.incomplete` blobs (unresumable, pure
  waste) and **keeps** any that still hold bytes.

### Preflight: measure before waiting

`preflight()` runs before `Separator()` is constructed, so a hopeless download
costs 8 seconds instead of an hour:

- all files cached -> skip entirely, no network probe at all;
- else `HEAD` for the size, then 6 s of real transfer for the rate, and refuse
  with `reason="too-slow"` when the projection exceeds `MAX_DOWNLOAD_SECONDS`.

`MAX_DOWNLOAD_SECONDS` is 900. On the measured 0.106 MB/s link the projection is
~758 s, so it proceeds — barely. A slower moment refuses with the numbers.

**A stalled model download looks exactly like a hang.** `run_python_worker` uses
`Popen` and polls, and while the worker is in the `downloading-model` phase it
watches the HuggingFace cache for growth. No growth for 90 s means blocked, so it
kills the process and reports `kind="stalled-download"`. Verified for real: the
same run that took 3600 s now fails in 96 s with an accurate message. The growth
check only applies during the download phase, so genuinely slow processing still
gets the full timeout.

Workers write their stage to a `<output>.phase` file as `name|detail`, and
`read_phase_detail` surfaces the second half. The downloader uses it to report
`bytes/total` throughout a 12-minute transfer, so a slow run can still explain
itself.

`stdout`/`stderr` go to temp files rather than PIPE, because torch and librosa
log heavily and a full pipe buffer would deadlock the child while we poll it.
The log directory is removed with `shutil.rmtree(ignore_errors=True)`, **not**
`TemporaryDirectory`: a killed child's inherited handles are not released
instantly on Windows and `TemporaryDirectory` raises `PermissionError` during
cleanup, masking the real error.

To confirm a failure is network and not a code bug, read the worker's stderr: if
it reaches the "unauthenticated requests to the HF Hub" warning, torch and
demucs imported fine and only the download is stuck.

**Workers record their stage** in a `<output>.phase` file
(`downloading-model` then `processing`). `ai_process` reads it on timeout, so the
message names the real cause instead of always blaming slow CPU work. Verified
against a real stalled download, the message now says "that download appears to
be stalled. This is not slow processing". Keep the phase writes when editing a
worker; `tests/test_timeout_phases.py` covers both branches.

## Milestones 4-5 rules

- **Never modify or delete the user's audio.** Separation refuses an output
  folder that is the source's own folder, clears stale stems before a run, and
  re-checks the source exists afterwards. Tested end to end.
- **Stems are not clean instruments.** Every surface says so. Demucs output
  still bleeds. Separation helps because a note tracker copes better with one
  monophonic-ish source, not because it is accurate.
- **Drums are never offered as a note source**; vocals are recommended first.
- **Never present drafted lyrics as real.** ASR is trained on speech, not
  singing. The draft lands in the normal lyrics box, is labelled, and can be
  discarded or overwritten. The manual editor must work with the model absent,
  missing or crashing — there are tests for all three.
- CPU runs state their cost up front rather than letting the user assume a hang.

## Programmatic updates to a keyed widget

`st.session_state["x"] = ...` is **discarded** if the widget with key `"x"` was
already created earlier in the same run. The lyrics box therefore writes to
`pending_lyrics_text` and reruns, consuming it at the top of Step 3 **before**
the `st.text_area` is created. `st.rerun()` alone does not fix this.
Same class of bug as the `st.data_editor` feedback loop; the versioned editor
keys (`lyric_editor_{version}`) are the other half of the pattern.

## Milestone 3: how note estimation is isolated

```
streamlit_app.py -> app.transcription.service -> bridge.py -> subprocess
                                                                 |
                                                    .venv-ai python
                                                                 |
                                                         worker.py -> basic_pitch
```

- **The core app never imports basic_pitch, TensorFlow or librosa.** Only
  `app/transcription/worker.py` does. Everything else works without them, and
  `pytest` never touches the model.
- `worker.py` is a **standalone script**: stdlib at module level, `basic_pitch`
  imported lazily inside the function. A test asserts it has no `from app` at
  module level. That is what lets a different interpreter run it.
- Results cross the process boundary as **JSON written to a file**, not stdout,
  because TensorFlow and librosa log freely and would corrupt a stdout protocol.
  A `__KMB_JSON__` marker line is emitted on stdout as a fallback.
- **`requirements-ai.txt` is opt-in.** TensorFlow 2.15 publishes cp39/cp310/cp311
  Windows wheels only, so the AI environment must be Python 3.10 or 3.11 while
  the app itself runs on 3.11.
- `bridge.basic_pitch_available` is **cached per interpreter** because the
  script re-runs on every interaction and the probe spawns a process.

### Basic Pitch API facts (verified from v0.4.0 source, not guessed)

```python
model_output, midi_data, note_events = predict(audio_path, model_or_model_path=..., ...)
# note_events entries: (start, end, pitch, amplitude, pitch_bends)
```

It is **not** `(model, path)` and **not** `(events, labels, pitches)` — an
earlier guess at that signature would have failed at runtime. `worker.py`
retries with the bare positional path if keyword arguments are rejected, for
older/newer versions. v0.4.0 reports **no instrument labels**, so notes group
under `unlabelled`; pitch banding (bass/mid/treble) is offered instead.

### Estimation rules

- **Never present estimates as a transcription.** Basic Pitch is a note tracker;
  on a full mix it returns a rough sketch without bass or chords. It works best
  on one instrument at a time. The UI carries this warning and the worker
  returns one in its payload.
- Estimated rows carry `source="estimated"` so they stay visibly distinct from
  hand-typed notes. **Estimation must never touch lyric rows.**
- Estimation being unavailable is a normal state, not an error: export and the
  whole app must keep working.

## Never name a module after a standard library module

Running a script puts **its own directory first on `sys.path`**. `worker.py` is
run by path, so everything beside it in `app/transcription/` shadows a stdlib
name of the same spelling. `types.py` there broke the worker instantly:
`argparse → re → enum → types` crashed with "cannot import name 'GenericAlias'",
so estimation failed with a confusing error about standard-library internals
instead of the real cause.

It is now `contract.py`. Two tests guard the whole `app/` tree against the
class of mistake, and one runs the worker as a **real subprocess** the way the
bridge does:

- `test_no_module_under_app_shadows_the_standard_library` — compares every
  `app/**/*.py` stem against `sys.stdlib_module_names`
- `test_worker_runs_as_a_real_subprocess_the_way_the_bridge_does` — asserts a
  clean *dependency* error rather than a stray stdlib ImportError

Loading `worker.py` with `importlib` in a test does **not** reproduce this, which
is exactly why it reached the user. Test the way it runs.

## Real model verification (opt-in)

Everything else mocks inference. To exercise the actual model:

```powershell
$env:KMB_RUN_AI_TESTS = "1"
.venv\Scripts\python.exe -m pytest tests/test_m3_transcription.py -k real_model -q
```

Requires `.venv-ai` with `requirements-ai.txt` installed, and downloads the model
on first run. It is slow on CPU — roughly **9 minutes** for both tests, because
TensorFlow loads twice in separate subprocesses. Verified end to end: a
generated 440 Hz tone is detected as MIDI 69 (A4) and a 220 Hz tone as MIDI 57
(A3), both correct, and the estimates convert cleanly to exportable note rows.

## Never trust `streamlit run` + HTTP 200 as an app test

`streamlit run` returns a **static shell** and only executes the script when a
browser opens a websocket. A 200 response proves the server booted, not that the
page renders. Use `streamlit.testing.v1.AppTest` to run the script in-process:

```python
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("streamlit_app.py", default_timeout=30).run()
assert not at.exception, [str(e.value) for e in at.exception]
```

`tests/test_m2_app_smoke.py` does this and exercises every step. Two real bugs
were only findable this way; see below.

## Milestone 2 rules worth keeping

- **Never write `some_frame or other_frame`.** `or` calls `DataFrame.__bool__`,
  which pandas refuses for any frame with more than one column, so this crashed
  Step 5 with "The truth value of a DataFrame is ambiguous". Test the row list
  instead. Use `app.tables.lyric_frame_for` / `note_frame_for`.
- **Check the shape you hand to a table.** `DemoSong.lyric_lines` is nested
  `Syllable` lists and `note_tracks` are `NoteTrack` objects - neither is a table
  row. `demo_lyric_rows()` / `demo_note_rows()` convert them.
- **Logic lives outside `streamlit_app.py`.** Importing that module runs the whole
  UI, so anything worth testing lives in `timing.py` / `notes.py` / `tables.py`.
- **Never assign an `st.data_editor` result back into the session-state key that
  fed it** — that creates a feedback loop where edits vanish on every other
  interaction. Use a `key` and read the returned value. Editor keys carry a
  version counter (`lyric_editor_{version}`) so re-splitting starts clean.
- **A lyrics-only export is valid, not an error.** No notes means no note tracks.
- **`lyrics_in_melody_track=True` is only safe when a note track exists.** With no
  notes there is nothing to interleave with, so the conductor track must carry the
  lyrics. This silently dropped every lyric in a lyrics-only export until a test
  caught it — do not remove the `bool(note_tracks)` guard.
- **Negative lyric times used to be clamped to 0** and exported "successfully".
  `export_service._lyric_structure_problems` now refuses; keep that check.
- `safe_cleanup` refuses to delete the workspace base or anything outside it.
  Tests cover that, because the alternative is deleting a user's files.

## Commands

```powershell
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\python.exe -m streamlit run streamlit_app.py
.venv\Scripts\python.exe -m app.make_m1_samples       # encoding variants
.venv\Scripts\python.exe -m app.make_note_samples     # note playback variants
.venv\Scripts\python.exe -m app.make_offset_samples   # lyric offset calibration
```

`requirements.txt` exists (mido, streamlit, pandas, mutagen, pytest). mutagen
reads MP3 duration from frame headers and needs no FFmpeg. Do not add Demucs /
Basic Pitch / TensorFlow — later milestones, and they force Python 3.11.

## Tracked files that should not be tracked

`git ls-files` currently lists `app/__pycache__/*.pyc` and `out/*.mid`.
`.gitignore` covers them but they were committed first, so ignore rules do not
apply. They need `git rm --cached` (not done here — index changes are the
user's call). Stale `.pyc` files can shadow edited source.

## SongHub highlight lead — NOT a bug in our files (resolved)

SongHub has a player setting: **"Reveal lyrics before start" = 8000 ms by
default** (must exceed the 5 s countdown). The app therefore shows the lyric
text several seconds before the music begins, **by design**. That is what
looked like a "constant early highlight"; it is the player's setting, not a
defect in the MIDI.

Evidence it is not our file: `s1`-`s4` (four structurally different layouts)
all behaved identically in SongHub. A layout that genuinely caused a lead would
have changed the symptom in at least one variant. Separately, the user applied
`lyric_offset=+1000 ms` and playback began *before* the countdown finished,
which shows the offset interacting with SongHub's reveal/countdown scheduling
rather than correcting anything.

**Do not compensate for this with `lyric_offset`.** Keep the offset at 0 for
normal use; it exists only for genuinely mistimed individual words. If the user
wants the text to appear later, the correct lever is SongHub's own setting.

## Reference-file layout — adopted

Ground truth from two known-good karaoke files, both confirmed working:

```
track 0  conductor only: track_name, FF 21 midi_port, marker, set_tempo,
         time/key signature. NO lyrics, NO notes.
track 1  vocal melody (channel 0) WITH lyric events interleaved into it:
           tick=3000  note_on  pitch=71
           tick=3000  LYRIC        <- same tick, immediately after the note
           tick=3027  note_off
         - CR/LF at that SAME tick, not at the word's end
         - every word coincides with a note onset (283/283 and 394/394)
```

These defaults now match it: `lyrics_in_melody_track=True`,
`separator_timing="same_tick"`, `emit_midi_port=True`. This is **fidelity
alignment for other players, not a fix for the early-text symptom** - all four
variants behaved the same in SongHub, so it changed nothing measurable here.

SongHub shows the title as `[Song name] - [Singer]`, read from the `FF 03` track
name. A redundant `FF 01` artist text event was removed; writing one would also
make the conductor track look lyric-bearing.

## SongHub settings that affect playback

- **Reveal lyrics before start: 8000 ms** (default). Must be > Countdown (5 s).
  This is why lyrics are visible before the music starts. Reduce it in SongHub.
- Countdown (5 s) interacts with file timing; a large lyric offset can make
  audio start before the countdown completes.

## Comparing against a known-good file

`app/inspect_midi.py` reports format, PPQ, tempo map (flagging tempo
*changes*), which meta event carries lyrics, track layout, and the **lead** —
the tick gap between first lyric and first note per channel, plus how many
lyric ticks coincide with a note onset.

**Copyright handling:** reference files contain copyrighted lyrics. The analyzer
reports lyric **timing and text length only**; `--show-text` is for files the
user owns and must not be used on commercial files. Never copy reference lyrics
into this project, never commit `reference/` (gitignored), never redistribute.
Structure and timing conventions are the reusable output.

## Non-obvious rules

- **`line_gap` must be 0 when words map 1:1 onto notes.** A gap silently shifts
  every word after the first line break off its note.
- **Out-of-order lyric timings are refused, not sorted.** MIDI deltas cannot be
  negative, so an edit that puts a word before its predecessor would otherwise be
  "fixed" by reordering, which scrambles the text. `export_service` reports it.
- `lyric_offset` shifts lyric events only; notes never move.

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