"""Vocal transcription: audio -> timed lyric words and lines (faster-whisper).

Words carry start times so they can be attached to melody notes; lines keep
segment-level timing for song-structure analysis.

When the user supplies the lyrics as text (Whisper mishears sung words often),
their words replace the transcription but still need times. Two strategies:
  1. align_user_lyrics — match the user's words against Whisper's timed words
     and inherit the timestamps; unmatched words interpolate between matches.
  2. lyrics_from_onsets — no usable transcription at all: place one word per
     melody-note onset, in order.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional


@dataclass
class TimedWord:
    start: float
    end: float
    text: str


@dataclass
class LyricLine:
    start: float
    end: float
    text: str


@dataclass
class Lyrics:
    words: list[TimedWord] = field(default_factory=list)
    lines: list[LyricLine] = field(default_factory=list)
    language: str = ""

    def __bool__(self) -> bool:
        return bool(self.words)


@lru_cache(maxsize=1)
def _load_whisper(model_size: str):
    from faster_whisper import WhisperModel

    # int8 on CPU keeps memory modest and runs fine on Apple Silicon.
    return WhisperModel(model_size, device="cpu", compute_type="int8")


# Hallucination guards. Silero VAD is tuned for speech and rejects singing
# over accompaniment wholesale (a full vocal track can come back empty), so
# VAD stays off and we filter per segment instead. no_speech_prob is useless
# here — real sung lines routinely score 0.95+ on it — but hallucinated text
# on instrumental stretches tends to be very short ("Zither Harp", watermark
# credits) or have terrible decoder confidence.
MIN_SEGMENT_SECONDS = 1.5
MIN_AVG_LOGPROB = -1.2


def transcribe_lyrics(wav_path: Path, model_size: str = "small") -> Lyrics:
    """Transcribe sung/spoken words with word-level timestamps."""
    model = _load_whisper(model_size)
    segments, info = model.transcribe(
        str(wav_path),
        word_timestamps=True,
        vad_filter=False,
        beam_size=5,
        condition_on_previous_text=False,  # avoids repetition loops on music
    )

    lyrics = Lyrics(language=info.language or "")
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        if segment.end - segment.start < MIN_SEGMENT_SECONDS:
            continue
        if segment.avg_logprob < MIN_AVG_LOGPROB:
            continue
        lyrics.lines.append(LyricLine(start=float(segment.start), end=float(segment.end), text=text))
        for word in segment.words or []:
            cleaned = word.word.strip()
            if cleaned:
                lyrics.words.append(TimedWord(start=float(word.start), end=float(word.end), text=cleaned))
    return lyrics


# --- user-provided lyrics ----------------------------------------------------

# Lyrics pasted from the web carry structural annotations that must never be
# engraved as sung words: "[Verse 1]", "[Chorus: Artist]", "{x2}", a bare
# "Pre-Chorus:" heading line, or a mangled one like "Verse 1]" (a stray bracket
# from a bad copy-paste). Bracketed spans are annotation by convention and
# always dropped; parenthesized text is real singing ("(ooh)") unless it names
# a section or a repeat count. Full-width CJK brackets/parens are handled too.
_BRACKETED = re.compile(r"\[[^\]]*\]|\{[^}]*\}|【[^】]*】|［[^］]*］")
_SECTION = (
    r"(?:intro|outro|verse|chorus|pre[-\s]?chorus|post[-\s]?chorus|bridge|hook"
    r"|refrain|interlude|instrumental|solo|breakdown|break|drop|coda|vamp"
    r"|ad[-\s]?libs?|spoken|skit"
    r"|主歌|副歌|前奏|间奏|間奏|尾奏|尾声|尾聲|桥段|橋段|导唱|預唱)"
)
_REPEAT = r"(?:repeat[^)）]*|[x×]\s*\d+|\d+\s*[x×])"
_PAREN_TAG = re.compile(
    rf"[(（]\s*(?:{_SECTION}(?:\s*\d+)?(?:\s*[:：][^)）]*)?|{_REPEAT})\s*[)）]",
    re.IGNORECASE,
)
# Whole-line heading, tested against a copy with annotation punctuation stripped
# so stray brackets/asterisks/dashes ("Verse 1]", "**Chorus**") don't hide it.
_HEADING_WORD = re.compile(rf"^{_SECTION}\s*\d*$", re.IGNORECASE)
_HEADING_COLON = re.compile(rf"^{_SECTION}\s*\d*\s*[:：]", re.IGNORECASE)


def strip_lyric_tags(line: str) -> str:
    """Remove structural annotations from one lyric line ('' if it was only a tag)."""
    line = _BRACKETED.sub(" ", line)
    line = _PAREN_TAG.sub(" ", line)
    if _HEADING_COLON.match(line.strip()):
        return ""
    probe = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", line)).strip()
    if _HEADING_WORD.match(probe):
        return ""
    return line


# CJK scripts don't use spaces; each character is a sung syllable, so split
# them into single-character tokens on both sides of the alignment.
_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯豈-﫿]")

MIN_ANCHORS = 3            # alignment needs at least this many matched words...
MIN_ANCHOR_FRACTION = 0.2  # ...and at least this share of the user's words
FALLBACK_WORD_STEP = 0.4   # seconds per word when extrapolating past the anchors
WORD_FALLBACK_DURATION = 0.3


def _split_token(token: str) -> list[str]:
    """'hello' -> ['hello']; '你好there' -> ['你', '好', 'there']."""
    if not _CJK.search(token):
        return [token]
    out: list[str] = []
    buf = ""
    for ch in token:
        if _CJK.match(ch):
            if buf:
                out.append(buf)
                buf = ""
            out.append(ch)
        else:
            buf += ch
    if buf:
        out.append(buf)
    return out


def _clean(token: str) -> str:
    """Normalized form used for matching: lowercase, letters/digits only."""
    return re.sub(r"[^\w]+", "", token.lower())


def parse_lyric_lines(text: str) -> list[list[str]]:
    """User text -> display tokens per non-empty line (section tags dropped,
    CJK split per character, pure-punctuation tokens dropped)."""
    lines = []
    for raw_line in text.splitlines():
        raw_line = strip_lyric_tags(raw_line)
        tokens = [t for word in raw_line.split() for t in _split_token(word) if _clean(t)]
        if tokens:
            lines.append(tokens)
    return lines


def align_user_lyrics(text: str, reference: Lyrics) -> Optional[Lyrics]:
    """Time the user's lyrics by aligning them to a Whisper transcription.

    Matched words (anchors) inherit the reference timestamps; words between
    anchors interpolate linearly. Words outside the outermost anchors spread
    across the reference's remaining span (ref_start before the first anchor,
    ref_end after the last) so a long tail of unmatched words isn't crammed
    against the anchor at a fixed step — that compression is what dragged every
    late section mark up to the start of the score. Falls back to
    FALLBACK_WORD_STEP stepping when the reference doesn't extend past the
    anchors. Returns None when too few words match to trust the timing (caller
    should fall back to lyrics_from_onsets).
    """
    token_lines = parse_lyric_lines(text)
    flat = [t for line in token_lines for t in line]
    if not flat or not reference.words:
        return None

    # Split reference words the same way so CJK sequences compare per character.
    ref_tokens: list[TimedWord] = []
    for word in reference.words:
        parts = [p for p in _split_token(word.text) if _clean(p)]
        span = (word.end - word.start) / max(len(parts), 1)
        for i, part in enumerate(parts):
            ref_tokens.append(TimedWord(start=word.start + i * span, end=word.start + (i + 1) * span, text=part))

    user_norm = [_clean(t) for t in flat]
    ref_norm = [_clean(w.text) for w in ref_tokens]
    matcher = difflib.SequenceMatcher(None, user_norm, ref_norm, autojunk=False)
    anchors: dict[int, TimedWord] = {}
    for a, b, size in matcher.get_matching_blocks():
        for k in range(size):
            anchors[a + k] = ref_tokens[b + k]
    if len(anchors) < max(MIN_ANCHORS, int(MIN_ANCHOR_FRACTION * len(flat))):
        return None

    starts: list[float] = [0.0] * len(flat)
    ends: list[float] = [0.0] * len(flat)
    indices = sorted(anchors)
    for i in indices:
        starts[i] = anchors[i].start
        ends[i] = max(anchors[i].end, anchors[i].start)
    first, last = indices[0], indices[-1]
    ref_start = min(w.start for w in ref_tokens)
    ref_end = max(w.end for w in ref_tokens)

    # Before the first anchor: if the reference starts meaningfully earlier,
    # spread the lead-in words across [ref_start, first anchor]; otherwise keep
    # the tight 0.4 s/word stepping backwards from the anchor.
    if first > 0 and starts[first] - ref_start > FALLBACK_WORD_STEP * first:
        step = (starts[first] - ref_start) / first
        for i in range(first):
            starts[i] = max(ref_start + i * step, 0.0)
    else:
        for i in range(first - 1, -1, -1):
            starts[i] = max(starts[i + 1] - FALLBACK_WORD_STEP, 0.0)

    # Past the last anchor: same idea toward [last anchor, ref_end]. This is the
    # common case for long songs where Whisper only anchors the first minute —
    # fixed stepping would cram the whole tail (and its section marks) up front.
    n_tail = len(flat) - 1 - last
    if n_tail > 0 and ref_end - starts[last] > FALLBACK_WORD_STEP * n_tail:
        step = (ref_end - starts[last]) / n_tail
        for k in range(1, n_tail + 1):
            starts[last + k] = starts[last] + k * step
    else:
        for k in range(1, n_tail + 1):
            starts[last + k] = starts[last + k - 1] + FALLBACK_WORD_STEP

    for i in range(first + 1, last):  # between two anchors: linear in word index
        if i in anchors:
            continue
        prev_i = max(j for j in indices if j < i)
        next_i = min(j for j in indices if j > i)
        frac = (i - prev_i) / (next_i - prev_i)
        starts[i] = starts[prev_i] + frac * (starts[next_i] - starts[prev_i])
    for i in range(len(flat)):
        if i not in anchors:
            ends[i] = starts[i] + WORD_FALLBACK_DURATION

    lyrics = Lyrics(language=reference.language)
    pos = 0
    for tokens in token_lines:
        line_start, line_end = starts[pos], ends[pos + len(tokens) - 1]
        for token in tokens:
            lyrics.words.append(TimedWord(start=starts[pos], end=ends[pos], text=token))
            pos += 1
        lyrics.lines.append(LyricLine(start=line_start, end=max(line_end, line_start), text=" ".join(tokens)))
    return lyrics


def lyrics_from_onsets(text: str, onsets: list[float]) -> Lyrics:
    """Last-resort timing: the i-th lyric word lands on the i-th melody onset.

    Words beyond the available onsets are dropped (the score reports the
    attached-word count, so the shortfall is visible).
    """
    times = sorted(set(onsets))
    lyrics = Lyrics()
    pos = 0
    for tokens in parse_lyric_lines(text):
        placed: list[str] = []
        line_start = None
        for token in tokens:
            if pos >= len(times):
                break
            start = times[pos]
            end = times[pos + 1] if pos + 1 < len(times) else start + WORD_FALLBACK_DURATION
            lyrics.words.append(TimedWord(start=start, end=end, text=token))
            placed.append(token)
            line_start = start if line_start is None else line_start
            pos += 1
        if placed:
            lyrics.lines.append(LyricLine(start=line_start, end=lyrics.words[-1].end, text=" ".join(placed)))
        if pos >= len(times):
            break
    return lyrics
