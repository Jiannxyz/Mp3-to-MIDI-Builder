"""Independent Standard MIDI File validator.

Why not just read the file back with mido?
-----------------------------------------
Re-reading with the same library that wrote it can hide a bad encoding: both
sides agree on the same wrong bytes. This module parses the raw file itself, so
"mido can read it" and "the bytes are correct" stay independent claims.

It decodes the structure the karaoke dialects care about: header, track chunks,
running status, and meta events -- in particular FF 05 (lyric) and FF 01 (text).
"""

from __future__ import annotations

from dataclasses import dataclass, field

META_NAMES = {
    0x01: "text",
    0x02: "copyright",
    0x03: "track_name",
    0x04: "instrument_name",
    0x05: "lyrics",
    0x06: "marker",
    0x07: "cue_marker",
    0x20: "channel_prefix",
    0x21: "midi_port",
    0x2F: "end_of_track",
    0x51: "set_tempo",
    0x54: "smpte_offset",
    0x58: "time_signature",
    0x59: "key_signature",
}


class MidiParseError(Exception):
    """Raised when the byte stream is not a well-formed SMF."""


@dataclass(frozen=True)
class Event:
    tick: int
    kind: str  # "meta" | "note_on" | "note_off" | "other"
    meta_type: int | None = None
    text: str | None = None
    channel: int | None = None
    note: int | None = None
    velocity: int | None = None

    @property
    def meta_name(self) -> str | None:
        if self.meta_type is None:
            return None
        return META_NAMES.get(self.meta_type, f"0x{self.meta_type:02X}")


@dataclass
class ParsedMidi:
    fmt: int
    declared_tracks: int
    division: int
    tracks: list[list[Event]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def smpte_timing(self) -> bool:
        return bool(self.division & 0x8000)

    def meta_events(self, meta_type: int) -> list[Event]:
        return [
            event
            for track in self.tracks
            for event in track
            if event.kind == "meta" and event.meta_type == meta_type
        ]

    def channel_events(self) -> list[Event]:
        return [event for track in self.tracks for event in track if event.kind in ("note_on", "note_off")]


def _decode(payload: bytes) -> str:
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        return payload.decode("latin-1")


def _read_vlq(data: bytes, index: int) -> tuple[int, int]:
    """Read a variable-length quantity; return (value, next_index)."""
    value = 0
    for _ in range(4):  # 4 bytes is the SMF maximum
        if index >= len(data):
            raise MidiParseError("truncated variable-length quantity")
        byte = data[index]
        index += 1
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return value, index
    raise MidiParseError("variable-length quantity longer than 4 bytes")


def _parse_track(chunk: bytes, track_index: int) -> list[Event]:
    events: list[Event] = []
    index = 0
    tick = 0
    running_status: int | None = None

    while index < len(chunk):
        delta, index = _read_vlq(chunk, index)
        tick += delta

        if index >= len(chunk):
            raise MidiParseError(f"track {track_index}: ran out of bytes after delta time")

        status = chunk[index]
        if status & 0x80:
            index += 1
            if status < 0xF0:
                running_status = status
        else:
            if running_status is None:
                raise MidiParseError(f"track {track_index}: running status used before any status byte")
            status = running_status

        if status == 0xFF:  # meta event
            if index >= len(chunk):
                raise MidiParseError(f"track {track_index}: truncated meta event")
            meta_type = chunk[index]
            index += 1
            length, index = _read_vlq(chunk, index)
            payload = chunk[index : index + length]
            if len(payload) != length:
                raise MidiParseError(f"track {track_index}: meta event 0x{meta_type:02X} claims {length} bytes, has {len(payload)}")
            index += length

            if meta_type == 0x2F and length != 0:
                raise MidiParseError(f"track {track_index}: end_of_track with non-zero length")

            events.append(
                Event(
                    tick=tick,
                    kind="meta",
                    meta_type=meta_type,
                    text=_decode(payload) if meta_type in (0x01, 0x05) else None,
                )
            )
            if meta_type == 0x2F:
                return events

        elif status in (0xF0, 0xF7):  # sysex
            length, index = _read_vlq(chunk, index)
            index += length
            events.append(Event(tick=tick, kind="other"))

        else:
            message = status & 0xF0
            channel = status & 0x0F
            data_bytes = 1 if message in (0xC0, 0xD0) else 2
            payload = chunk[index : index + data_bytes]
            if len(payload) != data_bytes:
                raise MidiParseError(f"track {track_index}: truncated channel message 0x{status:02X}")
            index += data_bytes

            if message in (0x80, 0x90):
                first, velocity = payload[0], payload[1]
                # A note_on with velocity 0 is a note release; normalise it.
                is_off = message == 0x80 or velocity == 0
                events.append(
                    Event(
                        tick=tick,
                        kind="note_off" if is_off else "note_on",
                        channel=channel,
                        note=first,
                        velocity=velocity,
                    )
                )
            else:
                events.append(Event(tick=tick, kind="other", channel=channel))

    raise MidiParseError(f"track {track_index}: no end_of_track (FF 2F 00) found")


def parse_bytes(data: bytes) -> ParsedMidi:
    if len(data) < 14 or data[:4] != b"MThd":
        raise MidiParseError("missing MThd header")

    header_length = int.from_bytes(data[4:8], "big")
    if header_length < 6:
        raise MidiParseError(f"MThd length {header_length} is too small")

    fmt = int.from_bytes(data[8:10], "big")
    declared_tracks = int.from_bytes(data[10:12], "big")
    division = int.from_bytes(data[12:14], "big")

    parsed = ParsedMidi(fmt=fmt, declared_tracks=declared_tracks, division=division)

    index = 8 + header_length
    while index < len(data):
        if data[index : index + 4] != b"MTrk":
            raise MidiParseError(f"expected MTrk chunk at byte {index}, found {data[index:index+4]!r}")
        chunk_length = int.from_bytes(data[index + 4 : index + 8], "big")
        start = index + 8
        chunk = data[start : start + chunk_length]
        if len(chunk) != chunk_length:
            raise MidiParseError(f"track chunk at byte {index} is truncated")
        parsed.tracks.append(_parse_track(chunk, len(parsed.tracks)))
        index = start + chunk_length

    if not parsed.tracks:
        raise MidiParseError("file contains no track chunks")

    if declared_tracks != len(parsed.tracks):
        parsed.warnings.append(
            f"header declares {declared_tracks} track(s) but {len(parsed.tracks)} track chunk(s) were found"
        )

    return parsed


def parse_file(path) -> ParsedMidi:
    return parse_bytes(open(path, "rb").read())


# --------------------------------------------------------------------------
# Structural checks
# --------------------------------------------------------------------------

SEPARATORS = {"\r", "\n"}


def check_midi(
    parsed: ParsedMidi,
    *,
    expected_format: int,
    expected_units: list[str],
    expected_unit_ticks: list[int],
    expected_notes: int,
    expected_note_ticks: list[int],
    lyric_meta: str,
    expect_word_separator: bool,
    expect_line_separator: bool,
) -> list[str]:
    """Return a list of problems. Empty list means the file passed."""
    problems: list[str] = list(parsed.warnings)
    lyric_ff = 0x05 if lyric_meta == "lyrics" else 0x01

    if parsed.fmt != expected_format:
        problems.append(f"format is {parsed.fmt}, expected {expected_format}")

    if parsed.smpte_timing:
        problems.append("division uses SMPTE timing; karaoke files must use ticks-per-quarter-note")
    elif parsed.division <= 0:
        problems.append(f"invalid division {parsed.division}")

    if expected_format == 0 and len(parsed.tracks) != 1:
        problems.append(f"format 0 must contain exactly 1 track, found {len(parsed.tracks)}")

    # --- lyric events -----------------------------------------------------
    lyric_events = parsed.meta_events(lyric_ff)
    if not lyric_events:
        problems.append(f"no FF {lyric_ff:02X} lyric/text meta events found")
        return problems

    ticks = [event.tick for event in lyric_events]
    if ticks != sorted(ticks):
        problems.append("lyric events are not in non-decreasing time order")

    units = [event.text for event in lyric_events if event.text not in SEPARATORS]
    if units != expected_units:
        problems.append(f"lyric text mismatch: got {units!r}, expected {expected_units!r}")

    unit_ticks = [event.tick for event in lyric_events if event.text not in SEPARATORS]
    if unit_ticks != expected_unit_ticks:
        problems.append(f"lyric tick mismatch: got {unit_ticks}, expected {expected_unit_ticks}")

    cr_count = sum(1 for event in lyric_events if event.text == "\r")
    lf_count = sum(1 for event in lyric_events if event.text == "\n")

    if expect_word_separator and cr_count == 0:
        problems.append("expected CR (0x0D) word separators but found none")
    if not expect_word_separator and cr_count:
        problems.append(f"dialect declares no word separator but file contains {cr_count} CR event(s)")
    if expect_line_separator and lf_count == 0:
        problems.append("expected LF (0x0A) line separators but found none")
    if not expect_line_separator and lf_count:
        problems.append(f"dialect declares no line separator but file contains {lf_count} LF event(s)")

    # --- notes ------------------------------------------------------------
    note_ons = [e for e in parsed.channel_events() if e.kind == "note_on"]
    note_offs = [e for e in parsed.channel_events() if e.kind == "note_off"]

    if len(note_ons) != expected_notes:
        problems.append(f"found {len(note_ons)} note_on events, expected {expected_notes}")
    if len(note_ons) != len(note_offs):
        problems.append(f"unbalanced notes: {len(note_ons)} note_on vs {len(note_offs)} note_off")

    # Note TIMES matter as much as note count. A file whose notes all sit on the
    # same tick passes every count check but plays as one sound, which is exactly
    # how SongHub rendered the first version of these samples.
    on_ticks = [event.tick for event in note_ons]
    # Sorted: format 0 merges tracks, so event order across channels is not
    # preserved. Sorting still catches a collapsed-timeline bug.
    if sorted(on_ticks) != sorted(expected_note_ticks):
        problems.append(
            f"note timing mismatch: got {sorted(on_ticks)[:8]}, expected {sorted(expected_note_ticks)[:8]}"
        )
    if len(set(on_ticks)) == 1 and len(on_ticks) > 1:
        problems.append(
            f"all {len(on_ticks)} notes share tick {on_ticks[0]}; they would play as one simultaneous sound"
        )

    # --- housekeeping -----------------------------------------------------
    for position, track in enumerate(parsed.tracks):
        if not any(e.meta_type == 0x2F for e in track):
            problems.append(f"track {position} has no end_of_track event")

    if not any(e.meta_type == 0x51 for track in parsed.tracks for e in track):
        problems.append("no set_tempo (FF 51) event; players will assume 120 BPM")

    return problems