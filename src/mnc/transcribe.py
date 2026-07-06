"""Audio -> note events, using Spotify's Basic Pitch model.

A note event is (start_seconds, end_seconds, midi_pitch, amplitude).
Tempo is estimated separately with librosa's beat tracker so the score
generator can quantize onsets to a musical grid.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

# Piano range: A0..C8. Constrain the model so rumble and cymbal hiss don't
# become impossible ledger-line notes.
PIANO_MIN_HZ = 27.5
PIANO_MAX_HZ = 4186.0


@dataclass
class NoteEvent:
    start: float
    end: float
    pitch: int
    amplitude: float


@lru_cache(maxsize=1)
def _load_model():
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import Model

    return Model(ICASSP_2022_MODEL_PATH)


def estimate_tempo(wav_path: Path) -> float:
    """Beat-track the audio and fold the tempo into a playable 65-190 BPM band."""
    import librosa

    y, sr = librosa.load(str(wav_path), sr=None, mono=True)
    if not np.any(y):
        return 120.0
    tempo, _beats = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])
    if bpm <= 0 or not np.isfinite(bpm):
        return 120.0
    while bpm < 65:
        bpm *= 2
    while bpm > 190:
        bpm /= 2
    return bpm


def transcribe(
    wav_path: Path,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    min_note_length_ms: float = 120.0,
) -> list[NoteEvent]:
    from basic_pitch.inference import predict

    # basic-pitch's CoreML path prints per-window debug lines; swallow them.
    with contextlib.redirect_stdout(io.StringIO()):
        _model_output, _midi, note_events = predict(
            str(wav_path),
            model_or_model_path=_load_model(),
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            minimum_note_length=min_note_length_ms,
            minimum_frequency=PIANO_MIN_HZ,
            maximum_frequency=PIANO_MAX_HZ,
            melodia_trick=True,
        )
    events = [
        NoteEvent(start=float(s), end=float(e), pitch=int(p), amplitude=float(a))
        for s, e, p, a, _bends in note_events
    ]
    events.sort(key=lambda n: (n.start, n.pitch))
    return events
