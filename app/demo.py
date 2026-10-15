"""The demo song used by the Streamlit page and the tests.

Content is "Twinkle Twinkle Little Star" ("Ah vous dirai-je, Maman"), a
traditional public-domain melody. It is used precisely because it is
unmistakable by ear, so a human can tell whether notes and highlighting line up.

Each lyric word carries its OWN start/end time, which is what makes the timing
editable in Milestone 2 without changing the file format.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .lyrics import Syllable, parse_lyrics
from .midi_build import NoteEvent, NoteTrack
from .notes import NoteRow
from .timing import LyricRow

TITLE = "Twinkle Twinkle Little Star"
ARTIST = "Demo Singer"
TEMPO_BPM = 120
SPACING = 0.45  # seconds between consecutive notes/words
VELOCITY = 100

# 23 notes, one per lyric word.
MELODY = [
    60, 60, 67, 67, 69, 69, 67,  # C C G G A A G
    65, 65, 64, 64, 62, 62, 60,  # F F E E D D C
    67, 67, 65, 65, 64, 64, 62,  # G G F F E E D
    67, 67,                      # G G
]

LYRIC_TEXT = """twinkle twinkle little star
how I wonder what you are
up above all the world
so high like a diamond in the sky"""


@dataclass
class DemoSong:
    title: str
    artist: str
    tempo_bpm: int
    lyric_lines: list[list[Syllable]] = field(default_factory=list)
    note_tracks: list[NoteTrack] = field(default_factory=list)

    @property
    def all_syllables(self) -> list[Syllable]:
        return [syl for line in self.lyric_lines for syl in line]

    @property
    def duration_seconds(self) -> float:
        ends = [syl.end for syl in self.all_syllables]
        ends += [note.start + note.duration for track in self.note_tracks for note in track.notes]
        return max(ends) if ends else 0.0


def build_demo_song() -> DemoSong:
    """A short demo: 23 notes, 23 timed lyric words, one tempo event.

    `line_gap=0` is required: each word must sit at i * SPACING so it lines up
    with melody note i. A gap would shift every word after the first line break
    off its note and look like a highlighting timing fault.
    """
    lyric_lines = parse_lyrics(LYRIC_TEXT, start=0.0, unit_seconds=SPACING, line_gap=0.0)

    melody = NoteTrack(
        name="melody",
        program=0,  # acoustic grand piano
        channel_prefix=True,
        notes=[
            NoteEvent(start=i * SPACING, duration=SPACING, pitch=pitch, velocity=VELOCITY, channel=0)
            for i, pitch in enumerate(MELODY)
        ],
    )

    phrase = 7 * SPACING
    bass = NoteTrack(
        name="bass",
        program=32,  # acoustic bass
        channel_prefix=True,
        notes=[
            NoteEvent(start=i * phrase, duration=phrase * 0.95, pitch=pitch, velocity=90, channel=1)
            for i, pitch in enumerate([36, 41, 36, 36])  # C, F, C, C
        ],
    )

    return DemoSong(
        title=TITLE,
        artist=ARTIST,
        tempo_bpm=TEMPO_BPM,
        lyric_lines=lyric_lines,
        note_tracks=[melody, bass],
    )


def demo_lyric_rows() -> list[LyricRow]:
    """The demo song as editable table rows.

    `DemoSong.lyric_lines` is nested lists of `Syllable`, which is NOT the shape
    the timing table consumes. Converting here keeps the demo out of
    `streamlit_app.py`.
    """
    song = build_demo_song()
    rows: list[LyricRow] = []
    for line_number, line in enumerate(song.lyric_lines, start=1):
        for syllable in line:
            rows.append(
                LyricRow(
                    line=line_number,
                    text=syllable.text,
                    start=syllable.start,
                    end=syllable.end,
                    word_end=syllable.is_word_end,
                )
            )
    return rows


def demo_note_rows() -> list[NoteRow]:
    """The demo song's notes as editable table rows."""
    song = build_demo_song()
    return [
        NoteRow(
            channel=note.channel,
            pitch=note.pitch,
            start=note.start,
            duration=note.duration,
            velocity=note.velocity,
        )
        for track in song.note_tracks
        for note in track.notes
    ]