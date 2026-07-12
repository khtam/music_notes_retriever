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

# Tempo estimation: librosa's beat tracker frequently locks onto a
# subharmonic/harmonic of the true beat (typically 2x on slow ballads with
# strong subdivided percussion). Fold into a playable band first, then use
# the Fourier tempogram to decide whether the fold-band tempo is itself an
# octave error and should be halved.
TEMPO_FOLD_MIN = 65.0
TEMPO_FOLD_MAX = 190.0
HALVING_MIN_BPM = 100.0    # only consider halving results at or above this
SUBHARMONIC_RATIO = 0.4    # halve when strength(bpm/2) >= ratio * strength(bpm)


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
    return _estimate_tempo_from_signal(y, sr)


def _estimate_tempo_from_signal(y: np.ndarray, sr: float, hop_length: int = 512) -> float:
    import librosa

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempo, _beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    bpm = float(np.atleast_1d(tempo)[0])
    if bpm <= 0 or not np.isfinite(bpm):
        return 120.0

    # Fold into the playable band *before* the octave decider: halving only
    # ever fires at >=100 BPM, so the result never drops back below the
    # fold-min and needs re-folding (which would undo the halving).
    while bpm < TEMPO_FOLD_MIN:
        bpm *= 2
    while bpm > TEMPO_FOLD_MAX:
        bpm /= 2

    strength = _fourier_tempo_strength(onset_env, sr, hop_length)
    return _resolve_tempo_octave(bpm, strength)


def _fourier_tempo_strength(onset_env: np.ndarray, sr: float, hop_length: int = 512, win_length: int = 384):
    """Return a callable bpm -> mean |Fourier tempogram| at the nearest tempo bin."""
    import librosa

    tg = np.abs(
        librosa.feature.fourier_tempogram(
            onset_envelope=onset_env, sr=sr, hop_length=hop_length, win_length=win_length
        )
    )
    freqs = librosa.fourier_tempo_frequencies(sr=sr, hop_length=hop_length, win_length=win_length)
    profile = tg.mean(axis=1)

    def strength(bpm: float) -> float:
        return float(profile[int(np.argmin(np.abs(freqs - bpm)))])

    return strength


def _resolve_tempo_octave(
    bpm: float,
    strength,
    min_bpm: float = HALVING_MIN_BPM,
    ratio: float = SUBHARMONIC_RATIO,
) -> float:
    """Halve a fast beat-tracked tempo when the tempogram shows a strong
    subharmonic at half the rate (a common octave error on slow songs with
    busy subdivided percussion). `strength` is any Callable[[float], float]
    so this is testable without audio."""
    if bpm >= min_bpm:
        s_full = strength(bpm)
        s_half = strength(bpm / 2.0)
        if s_full > 0 and s_half >= ratio * s_full:
            return bpm / 2.0
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
