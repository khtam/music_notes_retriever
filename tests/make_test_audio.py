"""Synthesize a small two-hand piano piece as a WAV for end-to-end testing.

Right hand: "Twinkle Twinkle Little Star" opening phrase.
Left hand: whole/half-note bass roots an octave-plus below.
Rendered as decaying harmonics so it resembles a plucked/struck timbre.
"""

from pathlib import Path

import numpy as np
import soundfile as sf

SR = 22050
BPM = 100
BEAT = 60.0 / BPM

# (midi_pitch, start_beat, duration_beats)
MELODY = [
    (60, 0, 1), (60, 1, 1), (67, 2, 1), (67, 3, 1),
    (69, 4, 1), (69, 5, 1), (67, 6, 2),
    (65, 8, 1), (65, 9, 1), (64, 10, 1), (64, 11, 1),
    (62, 12, 1), (62, 13, 1), (60, 14, 2),
]
BASS = [
    (36, 0, 2), (43, 2, 2), (41, 4, 2), (43, 6, 2),
    (38, 8, 2), (36, 10, 2), (43, 12, 2), (36, 14, 2),
]


def midi_to_hz(m: int) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


def tone(pitch: int, dur_s: float, amp: float) -> np.ndarray:
    t = np.arange(int(dur_s * SR)) / SR
    wave = np.zeros_like(t)
    for harmonic, weight in ((1, 1.0), (2, 0.4), (3, 0.2), (4, 0.1)):
        wave += weight * np.sin(2 * np.pi * midi_to_hz(pitch) * harmonic * t)
    envelope = np.minimum(t / 0.01, 1.0) * np.exp(-2.5 * t)
    return amp * envelope * wave


def render(notes, amp) -> np.ndarray:
    total = max(s + d for _, s, d in notes) * BEAT + 1.0
    buf = np.zeros(int(total * SR))
    for pitch, start_beat, dur_beats in notes:
        start = int(start_beat * BEAT * SR)
        chunk = tone(pitch, dur_beats * BEAT, amp)
        buf[start:start + len(chunk)] += chunk
    return buf


def main() -> Path:
    mix = render(MELODY, 0.5) + render(BASS, 0.35)
    mix /= np.abs(mix).max() * 1.1
    out = Path(__file__).parent / "twinkle.wav"
    sf.write(out, mix, SR)
    print(f"wrote {out} ({len(mix)/SR:.1f}s)")
    return out


if __name__ == "__main__":
    main()
