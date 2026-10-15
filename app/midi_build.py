"""Build karaoke MIDI files.

Notes and lyric events share one absolute tick timeline. Message `time` values
here are absolute ticks; mido converts them to delta times when saving.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mido

from .karaoke_dialects import Dialect
from .lyrics import Syllable

BPM = 120
PPQ = 480
TICKS_PER_SECOND = PPQ * BPM / 60.0  # 960 ticks per second at 120 BPM


def seconds_to_ticks(seconds: float, *, ppq: int = PPQ, bpm: int = BPM) -> int:
    if seconds < 0:
        raise ValueError(f"negative time not allowed: {seconds}")
    return int(round(seconds * ppq * bpm / 60.0))


@dataclass(frozen=True)
class NoteEvent:
    start: float  # seconds
    duration: float  # seconds
    pitch: int  # 0-127
    velocity: int = 80  # 1-127
    channel: int = 0  # 0-15
    name: str = "notes"


@dataclass
class NoteTrack:
    name: str
    notes: list[NoteEvent] = field(default_factory=list)
    program: int | None = None  # FF 08 program change (0-127); None = omit
    channel_prefix: bool = False  # emit FF 20 channel prefix before the notes


def _lyric_messages(
    line: list[Syllable],
    dialect: Dialect,
    *,
    ppq: int,
    bpm: int,
    lyric_offset: float = 0.0,
) -> list[mido.MetaMessage]:
    messages: list[mido.MetaMessage] = []

    def tick(seconds: float) -> int:
        # lyric_offset shifts ONLY the lyric events. Positive = lyrics are
        # written later, which delays the highlight. Notes are untouched, so
        # this is the calibration knob for "the text highlights too early".
        return seconds_to_ticks(max(0.0, seconds + lyric_offset), ppq=ppq, bpm=bpm)

    for syllable in line:
        messages.append(
            mido.MetaMessage(
                dialect.lyric_meta,
                text=syllable.text,
                time=tick(syllable.start),
            )
        )
        # Separator rides at the END of the unit it terminates.
        if dialect.word_separator and syllable.is_word_end:
            messages.append(
                mido.MetaMessage(
                    dialect.lyric_meta,
                    text=dialect.word_separator,
                    time=tick(syllable.end),
                )
            )

    if dialect.line_separator and line:
        last = line[-1]
        messages.append(
            mido.MetaMessage(
                dialect.lyric_meta,
                text=dialect.line_separator,
                time=tick(last.end),
            )
        )

    return messages


def _note_messages(track: NoteTrack, *, ppq: int, bpm: int) -> list:
    """Full message list for a note track: name, optional prefix/program, notes."""
    messages: list = [mido.MetaMessage("track_name", name=track.name, time=0)]

    if not track.notes:
        return messages

    channel = track.notes[0].channel

    if track.channel_prefix:
        # FF 20 names the channel the following events belong to. Several
        # karaoke players consult it when routing tracks.
        messages.append(mido.MetaMessage("channel_prefix", channel=channel, time=0))

    if track.program is not None:
        # Without FF 08 the player falls back to its default patch, which in a
        # soundfont-based player can sound thin or wrong for the part.
        messages.append(mido.Message("program_change", program=track.program, channel=channel, time=0))

    timed: list[tuple[int, mido.Message]] = []

    for note in track.notes:
        if note.pitch < 0 or note.pitch > 127:
            raise ValueError(f"pitch out of MIDI range: {note.pitch}")
        if not 0 <= note.channel <= 15:
            raise ValueError(f"channel out of range: {note.channel}")
        if note.duration <= 0:
            raise ValueError("note duration must be positive")

        on = seconds_to_ticks(note.start, ppq=ppq, bpm=bpm)
        off = seconds_to_ticks(note.start + note.duration, ppq=ppq, bpm=bpm)
        # The absolute tick MUST be set on the message itself. Keeping it only
        # as a sort key silently collapsed every note onto tick 0, which sounded
        # in SongHub as a single note followed by silence.
        timed.append(
            (
                on,
                mido.Message(
                    "note_on", note=note.pitch, velocity=note.velocity, channel=note.channel, time=on
                ),
            )
        )
        timed.append(
            (
                off,
                mido.Message("note_off", note=note.pitch, velocity=0, channel=note.channel, time=off),
            )
        )

    # Sort by tick; note_off sorts before note_on at the same tick so repeated
    # pitches release cleanly instead of stacking up.
    timed.sort(key=lambda item: (item[0], item[1].type == "note_on"))

    messages.extend(message for _, message in timed)
    return messages


def _delta_times(messages: list) -> list:
    """Convert absolute-tick messages into MIDI delta times.

    mido 1.3 stores each message's `time` as a DELTA and exposes no public
    absolute<->relative helper, so the conversion is done explicitly here.
    Working in absolute ticks up to this point keeps the builder readable.
    """
    converted = []
    previous = 0
    for message in messages:
        converted.append(message.copy(time=max(0, message.time - previous)))
        previous = message.time
    return converted


def build_midi(
    *,
    title: str,
    artist: str,
    lines: list[list[Syllable]],
    note_tracks: list[NoteTrack],
    dialect: Dialect,
    bpm: int = BPM,
    ppq: int = PPQ,
    lyric_offset: float = 0.0,
) -> mido.MidiFile:
    """Assemble notes + timed lyrics into a karaoke MIDI file.

    For format 0 the tracks are still appended separately; mido merges them
    into a single track on save, which is exactly what format 0 requires.
    """
    midi = mido.MidiFile(type=dialect.smf_format, ticks_per_beat=ppq)
    # mido names this "charset" and defaults to latin1, which would corrupt
    # non-Latin karaoke lyrics. utf-8 is widely accepted by current players.
    midi.charset = "utf-8"

    meta_messages: list[mido.Message | mido.MetaMessage] = []

    if dialect.banner:
        meta_messages.append(mido.MetaMessage("text", text=dialect.banner, time=0))

    display = f"{title} - {artist}" if artist else (title or "Untitled")
    meta_messages.append(mido.MetaMessage("track_name", name=display, time=0))
    meta_messages.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    meta_messages.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    meta_messages.append(mido.MetaMessage("key_signature", key="C", time=0))

    if artist and dialect.lyric_meta != "text":
        # Only safe when lyrics live in FF 05. In a dialect that carries lyrics
        # as FF 01 Text, an artist name written as FF 01 would be displayed and
        # highlighted as if it were a sung lyric.
        meta_messages.append(mido.MetaMessage("text", text=artist, time=0))

    for line in lines:
        meta_messages.extend(
            _lyric_messages(line, dialect, ppq=ppq, bpm=bpm, lyric_offset=lyric_offset)
        )

    absolute_tracks: list[list] = [meta_messages]

    for track in note_tracks:
        absolute_tracks.append(_note_messages(track, ppq=ppq, bpm=bpm))

    if dialect.smf_format == 0:
        # Format 0 is a single-track file, and mido checks len(tracks) == 1
        # before saving. Merge in ABSOLUTE time first, then convert once;
        # sorting already-converted delta times would scramble the timeline.
        # sort() is stable, so events sharing a tick keep metadata-first order.
        combined: list = []
        for messages in absolute_tracks:
            combined.extend(messages)
        combined.sort(key=lambda message: message.time)
        midi.tracks = [mido.MidiTrack(_delta_times(combined))]
    else:
        for messages in absolute_tracks:
            midi.tracks.append(mido.MidiTrack(_delta_times(messages)))

    return midi


def save_midi(midi: mido.MidiFile, path) -> None:
    # Text encoding is a property of the MidiFile (see midi.charset), not of save().
    midi.save(filename=str(path))