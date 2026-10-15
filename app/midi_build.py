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
    separator_timing: str = "word_end",
) -> list[mido.MetaMessage]:
    messages: list[mido.MetaMessage] = []

    def tick(seconds: float) -> int:
        # lyric_offset shifts ONLY the lyric events. Positive = lyrics are
        # written later, which delays the highlight. Notes are untouched, so
        # this is the calibration knob for "the text highlights too early".
        return seconds_to_ticks(max(0.0, seconds + lyric_offset), ppq=ppq, bpm=bpm)

    def separator_time(syllable: Syllable) -> float:
        # Working karaoke files put CR/LF at the SAME tick as the word they
        # terminate, not at the word's end. Both observed in real SongHub files.
        return syllable.start if separator_timing == "same_tick" else syllable.end

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
                    time=tick(separator_time(syllable)),
                )
            )

    if dialect.line_separator and line:
        last = line[-1]
        messages.append(
            mido.MetaMessage(
                dialect.lyric_meta,
                text=dialect.line_separator,
                time=tick(separator_time(last)),
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
    lyrics_in_melody_track: bool = True,
    separator_timing: str = "same_tick",
    emit_midi_port: bool = True,
) -> mido.MidiFile:
    """Assemble notes + timed lyrics into a karaoke MIDI file.

    `lyrics_in_melody_track` reproduces the layout of real working karaoke
    files: the lyric events live in the SAME track as the vocal melody notes,
    each written at the same tick as its note. Track 0 is a conductor track
    carrying only tempo/signature information. SongHub may schedule separate
    lyric and melody tracks with a fixed skew, which shows up as text
    highlighting ahead of the singing.

    `separator_timing` controls where CR/LF go: "word_end" (default) or
    "same_tick", which is what the reference files do.

    `emit_midi_port` adds FF 21 to each track, as the reference files do.

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

    # The title SongHub shows comes from the track name above, formatted as
    # "[Song] - [Singer]", so no extra FF 01 text is written here. An FF 01 in
    # the conductor track would also make that track look lyric-bearing, which
    # the working karaoke files avoid.

    lyric_messages: list[mido.MetaMessage] = []
    for line in lines:
        lyric_messages.extend(
            _lyric_messages(
                line,
                dialect,
                ppq=ppq,
                bpm=bpm,
                lyric_offset=lyric_offset,
                separator_timing=separator_timing,
            )
        )

    # Lyrics are merged into the melody track only when there IS a melody track.
    # With no notes at all there is nothing to interleave with, so the conductor
    # track must carry them instead - otherwise a lyrics-only export silently
    # drops every lyric.
    merge_lyrics_into_melody = lyrics_in_melody_track and bool(note_tracks)

    if not merge_lyrics_into_melody:
        meta_messages.extend(lyric_messages)

    # Guarantee non-decreasing absolute time no matter how the caller ordered
    # things. sort() is stable, so messages sharing a tick keep their insertion
    # order (track name, tempo, time/key signature, then lyrics).
    meta_messages.sort(key=lambda message: message.time)

    absolute_tracks: list[list] = [meta_messages]

    for index, track in enumerate(note_tracks):
        messages = _note_messages(track, ppq=ppq, bpm=bpm)

        if merge_lyrics_into_melody and index == 0:
            # Interleave lyrics into the melody track. Stable sort keeps
            # note_on ahead of the lyric written at the same tick, which is the
            # order the reference karaoke files use.
            messages = messages + lyric_messages
            messages.sort(key=lambda message: message.time)

        if emit_midi_port:
            messages.insert(
                1, mido.MetaMessage("midi_port", port=index + 1, time=0)
            )

        absolute_tracks.append(messages)

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