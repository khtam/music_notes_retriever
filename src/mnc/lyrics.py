"""Vocal transcription: audio -> timed lyric words and lines (faster-whisper).

Words carry start times so they can be attached to melody notes; lines keep
segment-level timing for song-structure analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


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
