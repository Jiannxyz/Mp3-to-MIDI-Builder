"""Karaoke MIDI dialects.

Why this module exists
----------------------
The Standard MIDI File spec defines a Lyric meta event (FF 05) but does NOT
define how karaoke software should use it. The MIDI Manufacturers Association
has publicly acknowledged this gap. Real-world karaoke files disagree:

* Tune1000 style  - syllables in FF 05; word break = FF 05 with 0x0D (CR);
                    line end = FF 05 with 0x0A (LF); expects format 0.
* Soft Karaoke    - lyrics in FF 01 Text events, banner "@KMIDI KARAOKE FILE".
* Solton          - FF 05 with "<" / "%" tag prefixes.

So "the file opens in a MIDI library" proves nothing. The dialect is therefore
data, not hardcoded logic, so that once the target player (SongHub) is tested
we set the winning dialect here and everything else just works.

Reference: midi.org community FAQ "How many MIDI Karaoke formats exist?",
and Mixage Software midikit karaoke format documentation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Dialect:
    """One karaoke MIDI encoding style."""

    name: str
    smf_format: int  # 0 = single merged track, 1 = multi-track
    lyric_meta: str  # mido meta name: "lyrics" (FF 05) or "text" (FF 01)
    word_separator: str | None  # emitted after the last unit of a word
    line_separator: str | None  # emitted after the last unit of a line
    banner: str | None  # optional leading text meta event
    description: str

    @property
    def lyric_ff(self) -> str:
        return "FF 05" if self.lyric_meta == "lyrics" else "FF 01"


# Baseline + three variants that each change exactly ONE dimension from the
# baseline, so a single SongHub test session identifies the cause of any
# failure instead of leaving several variables confounded.
DIALECTS: dict[str, Dialect] = {
    "A_fmt1_lyrics_ff05": Dialect(
        name="A_fmt1_lyrics_ff05",
        smf_format=1,
        lyric_meta="lyrics",
        word_separator="\r",
        line_separator="\n",
        banner=None,
        description="Baseline. Format 1, lyrics in FF 05, CR between words, LF between lines.",
    ),
    "B_fmt0_lyrics_ff05": Dialect(
        name="B_fmt0_lyrics_ff05",
        smf_format=0,
        lyric_meta="lyrics",
        word_separator="\r",
        line_separator="\n",
        banner=None,
        description="Like A but single merged track (format 0), which Tune1000 expects.",
    ),
    "C_fmt1_text_ff01": Dialect(
        name="C_fmt1_text_ff01",
        smf_format=1,
        lyric_meta="text",
        word_separator="\r",
        line_separator="\n",
        banner=None,
        description="Like A but lyrics carried in Text events FF 01 (Soft Karaoke style).",
    ),
    "D_fmt1_lyrics_nosep": Dialect(
        name="D_fmt1_lyrics_nosep",
        smf_format=1,
        lyric_meta="lyrics",
        word_separator=None,
        line_separator=None,
        banner=None,
        description="Like A but no CR/LF events at all, in case SongHub renders them literally.",
    ),
}

BASELINE = "A_fmt1_lyrics_ff05"