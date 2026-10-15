"""Inspect any MIDI file's structure, especially a known-good karaoke file.

Run:  .venv\\Scripts\\python.exe -m app.inspect_midi path\\to\\file.mid

Purpose: learn the conventions a WORKING karaoke file uses, so our writer can
match them instead of guessing. Reports header, tempo map, which meta events
carry lyrics, track layout, and the gap between the first lyric and the first
note - that gap is the "lead" that makes SongHub highlight text early.

Copyright note: this reports lyric TIMING and text LENGTH, not lyric content,
so analysing a commercial file does not copy its words into this project.
Pass --show-text only on files you own or wrote yourself.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from .validate_midi import MidiParseError, parse_bytes


def tempo_map(path: Path) -> list[tuple[int, float]]:
    """(tick, BPM) for each tempo change.

    Read with mido on purpose: our independent Event dataclass deliberately
    keeps no meta payloads, so a second pass is the simplest way to decode
    tempo without weakening the validator.
    """
    import mido

    midi = mido.MidiFile(filename=str(path))
    merged = mido.merge_tracks(midi.tracks)
    return [(msg.time, mido.tempo2bpm(msg.tempo)) for msg in merged if msg.type == "set_tempo"]


def describe(path: Path, show_text: bool = False) -> int:
    data = path.read_bytes()
    try:
        parsed = parse_bytes(data)
    except MidiParseError as error:
        print(f"could not parse {path.name}: {error}")
        return 1

    print("=" * 74)
    print(f"{path.name}  ({len(data)} bytes)")
    print("=" * 74)

    # --- header ----------------------------------------------------------
    print(f"format        : {parsed.fmt} ({'single track' if parsed.fmt == 0 else 'multi track'})")
    print(f"tracks        : {len(parsed.tracks)} declared {parsed.declared_tracks}")
    if parsed.smpte_timing:
        print(f"division      : SMPTE 0x{parsed.division:04X} (unusual for karaoke)")
    else:
        print(f"division      : {parsed.division} ticks per quarter note")
    for warning in parsed.warnings:
        print(f"  warning     : {warning}")

    try:
        tempos = tempo_map(path)
    except Exception as error:  # noqa: BLE001 - a third-party file may be odd
        tempos = []
        print(f"  tempo note  : could not read tempo map ({error})")

    if tempos:
        print("\ntempo map")
        for tick, bpm in tempos[:8]:
            print(f"  tick={tick:<8} {bpm:.2f} BPM")
        if len(tempos) > 8:
            print(f"  ... {len(tempos) - 8} more tempo change(s)")
        if len({round(b, 3) for _, b in tempos}) == 1:
            print("  constant tempo")
        else:
            print("  WARNING: tempo changes during the song")

    # --- meta event census ----------------------------------------------
    census: Counter[str] = Counter()
    for track in parsed.tracks:
        for event in track:
            if event.kind == "meta":
                census[event.meta_name] += 1

    print("\nmeta events")
    for name, count in sorted(census.items(), key=lambda kv: -kv[1]):
        marker = ""
        if name in ("lyrics", "text"):
            marker = "   <-- lyric carrier candidate"
        print(f"  {name:<16} {count}{marker}")
    if "lyrics" not in census and "text" not in census:
        print("  (no lyric or text events found - this is not a karaoke file)")

    # --- track layout ----------------------------------------------------
    print("\ntracks")
    for index, track in enumerate(parsed.tracks):
        name = next((e.text for e in track if e.meta_type == 0x03), None)
        ons = [e for e in track if e.kind == "note_on"]
        offs = [e for e in track if e.kind == "note_off"]
        lyric_count = sum(1 for e in track if e.meta_type in (0x05, 0x01) and e.text not in ("\r", "\n"))

        label = f'"{name}"' if name else "(unnamed)"
        channels = sorted({e.channel for e in ons})
        span = f"{track[0].tick}..{track[-1].tick}" if track else "empty"
        print(f"  [{index}] {label}")
        print(f"      events={len(track)} span(ticks)={span} note_on={len(ons)} note_off={len(offs)}")
        print(f"      channels={channels} lyric/text events={lyric_count}")

    # --- lyrics vs notes: the lead measurement ---------------------------
    words = [e for e in parsed.meta_events(0x05) if e.text not in ("\r", "\n")]
    texts = [e for e in parsed.meta_events(0x01) if e.text not in ("\r", "\n")]
    lyric_events = words or texts
    carrier = "FF 05 (lyric)" if words else ("FF 01 (text)" if texts else "none")

    print(f"\nlyric carrier : {carrier}")
    if lyric_events:
        print(f"  first lyric  : tick {lyric_events[0].tick}")
        print(f"  last  lyric  : tick {lyric_events[-1].tick}")
        lengths = [len(e.text or "") for e in lyric_events]
        print(f"  words        : {len(lyric_events)}  avg length {sum(lengths)/len(lengths):.1f} chars")
        if show_text:
            print("  first 8 items:")
            for event in lyric_events[:8]:
                print(f"      tick={event.tick:<8} {event.text!r}")
        else:
            print("  first 8 items (text withheld; use --show-text on your own file):")
            for event in lyric_events[:8]:
                print(f"      tick={event.tick:<8} length={len(event.text or '')}")

    ons = [e for e in parsed.channel_events() if e.kind == "note_on"]
    if ons and lyric_events:
        first_lyric = lyric_events[0].tick
        print("\nlead measurement (negative = lyrics start before notes)")
        for channel in sorted({e.channel for e in ons}):
            channel_ons = [e for e in ons if e.channel == channel]
            delta = channel_ons[0].tick - first_lyric
            print(f"  channel {channel}: first note_on {channel_ons[0].tick} -> lead {delta} ticks")

    if ons and lyric_events:
        # How well do lyric events line up with note onsets overall? If a
        # reference file shows a consistent offset, that offset is the
        # convention the player expects.
        note_ticks = sorted({e.tick for e in ons})
        lyric_ticks = sorted({e.tick for e in lyric_events})
        matches = sum(1 for t in lyric_ticks if t in set(note_ticks))
        print(
            f"\nalignment      : {matches}/{len(lyric_ticks)} lyric ticks coincide "
            f"exactly with a note onset"
        )

    print()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument(
        "--show-text",
        action="store_true",
        help="print lyric text (only for files you own; lyrics are copyrighted)",
    )
    args = parser.parse_args(argv)

    for path in args.paths:
        if not path.exists():
            print(f"missing file: {path}", file=sys.stderr)
            return 1
        describe(path, show_text=args.show_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())