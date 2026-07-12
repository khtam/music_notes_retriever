"""Unit tests for tempo octave resolution. Run with:

    .venv/bin/python -m unittest discover tests
"""

import unittest
from pathlib import Path

from mnc.transcribe import HALVING_MIN_BPM, SUBHARMONIC_RATIO, _resolve_tempo_octave

TWINKLE_WAV = Path(__file__).parent / "twinkle.wav"


class TestResolveTempoOctave(unittest.TestCase):
    def test_strong_subharmonic_halves(self):
        # Measured on the real "Surrender" ballad: beat_track locks onto 129.2
        # BPM (2x the true 64.6) but the Fourier tempogram shows the true
        # tempo's bin is nearly as strong as the doubled one.
        strength = {129.19921875: 22.92, 64.599609375: 14.53}.get
        self.assertAlmostEqual(_resolve_tempo_octave(129.19921875, strength), 64.599609375)

    def test_weak_subharmonic_keeps_tempo(self):
        # Measured on the synthetic 100 BPM twinkle test track: the half-tempo
        # bin is much weaker, so beat_track's answer is trusted as-is.
        strength = {99.38401442307692: 86.91, 49.69200721153846: 17.26}.get
        self.assertAlmostEqual(_resolve_tempo_octave(99.38401442307692, strength), 99.38401442307692)

    def test_slow_tempo_never_consulted(self):
        # Below the halving gate, the tempogram is never even sampled at bpm/2.
        strength = {80.0: 1.0, 40.0: 1000.0}.get
        self.assertEqual(_resolve_tempo_octave(80.0, strength), 80.0)

    def test_exact_ratio_boundary_halves(self):
        full, half = 120.0, 60.0
        strength = {full: 10.0, half: SUBHARMONIC_RATIO * 10.0}.get
        self.assertEqual(_resolve_tempo_octave(full, strength), half)

    def test_just_below_ratio_keeps_tempo(self):
        full, half = 120.0, 60.0
        strength = {full: 10.0, half: (SUBHARMONIC_RATIO - 0.01) * 10.0}.get
        self.assertEqual(_resolve_tempo_octave(full, strength), full)

    def test_just_below_min_bpm_gate_keeps_tempo(self):
        bpm = HALVING_MIN_BPM - 0.01
        strength = {bpm: 1.0, bpm / 2: 1000.0}.get
        self.assertEqual(_resolve_tempo_octave(bpm, strength), bpm)

    def test_zero_strength_does_not_divide_by_zero(self):
        strength = {150.0: 0.0, 75.0: 0.0}.get
        self.assertEqual(_resolve_tempo_octave(150.0, strength), 150.0)


@unittest.skipUnless(TWINKLE_WAV.exists(), "tests/twinkle.wav not generated; run tests/make_test_audio.py")
class TestEstimateTempoTwinkle(unittest.TestCase):
    def test_estimate_tempo_near_true_bpm(self):
        # tests/make_test_audio.py synthesizes twinkle.wav at 100 BPM. This
        # guards against the octave decider itself introducing a regression
        # (the verify skill's manual workflow depends on this staying ~100).
        from mnc.transcribe import estimate_tempo

        bpm = estimate_tempo(TWINKLE_WAV)
        self.assertTrue(90 <= bpm <= 110, f"expected ~100 BPM, got {bpm}")


if __name__ == "__main__":
    unittest.main()
