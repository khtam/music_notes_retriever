"""Unit tests for chord-symbol detection. Run with:

    .venv/bin/python -m unittest discover tests
"""

import unittest

from mnc.chords import (
    DetectedChord,
    _best_chord,
    _measure_weights,
    chord_figure,
    detect_chords,
)
from mnc.score import build_score
from mnc.transcribe import NoteEvent


class TestBestChord(unittest.TestCase):
    def test_major_triad(self):
        # C E G, all equal weight -> C major, root=0, quality=''
        weights = {0: 4.0, 4: 4.0, 7: 4.0}
        root_pc, quality, confidence = _best_chord(weights, bass_pc=0)
        self.assertEqual((root_pc, quality), (0, ""))
        self.assertGreater(confidence, 0.9)

    def test_minor_triad(self):
        # A C E -> A minor
        weights = {9: 4.0, 0: 4.0, 4: 4.0}
        root_pc, quality, _ = _best_chord(weights, bass_pc=9)
        self.assertEqual((root_pc, quality), (9, "m"))

    def test_dominant_seventh_needs_real_seventh_weight(self):
        # C E G with a strong Bb -> C7, not plain C
        strong = {0: 4.0, 4: 4.0, 7: 4.0, 10: 3.5}
        root_pc, quality, _ = _best_chord(strong, bass_pc=0)
        self.assertEqual((root_pc, quality), (0, "7"))

    def test_weak_incidental_tone_does_not_flip_to_seventh(self):
        # C E G with only a trace of Bb (a passing tone / transcription
        # artifact) should still read as a plain C triad, not C7. This
        # guards against the "superset always wins" scoring bug where any
        # nonzero extra weight used to flip the match to the larger template.
        weak = {0: 4.0, 4: 4.0, 7: 4.0, 10: 0.2}
        root_pc, quality, _ = _best_chord(weak, bass_pc=0)
        self.assertEqual((root_pc, quality), (0, ""))

    def test_sus4(self):
        # D G A -> Dsus4
        weights = {2: 4.0, 7: 4.0, 9: 4.0}
        root_pc, quality, _ = _best_chord(weights, bass_pc=2)
        self.assertEqual((root_pc, quality), (2, "sus4"))

    def test_maj7_and_m7_spelling(self):
        dmaj7 = {2: 3.0, 6: 3.0, 9: 3.0, 1: 2.8}  # D F# A C#
        root_pc, quality, _ = _best_chord(dmaj7, bass_pc=2)
        self.assertEqual((root_pc, quality), (2, "maj7"))

        fsm7 = {6: 3.0, 9: 3.0, 1: 3.0, 4: 2.8}  # F# A C# E
        root_pc, quality, _ = _best_chord(fsm7, bass_pc=6)
        self.assertEqual((root_pc, quality), (6, "m7"))

    def test_bass_bonus_does_not_override_clear_pitch_content(self):
        # First-inversion C major (bass = E) should still resolve to root C,
        # not be dragged toward E, since C's triad template explains 100%
        # of the pitch mass while any E-rooted template would not.
        weights = {0: 4.0, 4: 4.0, 7: 4.0}
        root_pc, quality, _ = _best_chord(weights, bass_pc=4)
        self.assertEqual((root_pc, quality), (0, ""))

    def test_chromatic_mush_returns_none(self):
        weights = {pc: 1.0 for pc in range(12)}
        self.assertIsNone(_best_chord(weights, bass_pc=None))

    def test_below_min_weight_returns_none(self):
        self.assertIsNone(_best_chord({0: 0.5}, bass_pc=0))

    def test_empty_weights_returns_none(self):
        self.assertIsNone(_best_chord({}, bass_pc=None))


class TestMeasureWeights(unittest.TestCase):
    def test_note_within_one_measure(self):
        # (onset, dur, pitch) at 4 beats/measure
        quantized = [(0.0, 2.0, 60), (2.0, 1.0, 64)]
        result = _measure_weights(quantized, measure_beats=4.0)
        self.assertEqual(len(result), 1)
        weights, bass_pc = result[0]
        self.assertAlmostEqual(weights[0], 2.0)   # pc 0 = C
        self.assertAlmostEqual(weights[4], 1.0)   # pc 4 = E
        self.assertEqual(bass_pc, 0)              # 60 < 64

    def test_note_spanning_barline_splits_weight(self):
        # A note from beat 2 to beat 6 (dur 4) spans measure 0 (beats 2-4,
        # i.e. 2 beats of overlap) and measure 1 (beats 4-6, 2 beats).
        quantized = [(2.0, 4.0, 60)]
        result = _measure_weights(quantized, measure_beats=4.0)
        self.assertEqual(len(result), 2)
        w0, bass0 = result[0]
        w1, bass1 = result[1]
        self.assertAlmostEqual(w0[0], 2.0)
        self.assertAlmostEqual(w1[0], 2.0)
        self.assertEqual(bass0, 0)
        self.assertEqual(bass1, 0)

    def test_empty_input(self):
        self.assertEqual(_measure_weights([]), [])

    def test_bass_is_lowest_pitch_not_first_note(self):
        quantized = [(0.0, 1.0, 67), (0.0, 1.0, 55)]  # G4 then G3
        _weights, bass_pc = _measure_weights(quantized, measure_beats=4.0)[0]
        self.assertEqual(bass_pc, 55 % 12)


class TestDetectChords(unittest.TestCase):
    def _chord_notes(self, root_pcs_per_measure, base_octave=60, measure_beats=4.0):
        """Build a flat quantized list: one triad (root, root+4, root+7) held
        for the full measure, per entry in root_pcs_per_measure."""
        quantized = []
        for i, root_pc in enumerate(root_pcs_per_measure):
            onset = i * measure_beats
            for offset in (0, 4, 7):
                pitch = base_octave + root_pc + offset
                quantized.append((onset, measure_beats, pitch))
        return quantized

    def test_repeated_chord_emits_once(self):
        quantized = self._chord_notes([0, 0, 0])  # C, C, C
        chords = detect_chords(quantized)
        self.assertEqual(chords, [DetectedChord(offset_beats=0.0, root_pc=0, quality="")])

    def test_change_emits_new_symbol(self):
        quantized = self._chord_notes([0, 0, 9])  # C, C, A(minor triad shape reused as major here)
        chords = detect_chords(quantized)
        self.assertEqual(len(chords), 2)
        self.assertEqual(chords[0].offset_beats, 0.0)
        self.assertEqual(chords[1].offset_beats, 8.0)

    def test_low_confidence_measure_does_not_reset_run(self):
        quantized = self._chord_notes([0, 0])  # two clean C measures
        # Add a chromatic-mush third measure (low confidence -> skipped),
        # then a fourth measure back to C: should not re-emit C.
        mush_onset = 8.0
        for pc in range(12):
            quantized.append((mush_onset, 4.0, 60 + pc))
        quantized += self._chord_notes([0], measure_beats=4.0)
        # shift the appended 4th measure to the correct onset (12.0)
        quantized = [
            (onset if onset != 0.0 or i < 6 else onset + 12.0, dur, pitch)
            for i, (onset, dur, pitch) in enumerate(quantized)
        ]
        chords = detect_chords(quantized)
        # Only the initial C at 0.0 should be emitted; the mush measure is
        # skipped and the trailing C (same chord) doesn't re-trigger.
        self.assertEqual(chords, [DetectedChord(offset_beats=0.0, root_pc=0, quality="")])

    def test_empty_input_returns_empty(self):
        self.assertEqual(detect_chords([]), [])


class TestChordFigure(unittest.TestCase):
    def test_sharp_spelling(self):
        self.assertEqual(chord_figure(1, "m7", prefer_sharps=True), "C#m7")
        self.assertEqual(chord_figure(6, "maj7", prefer_sharps=True), "F#maj7")

    def test_flat_spelling(self):
        self.assertEqual(chord_figure(10, "", prefer_sharps=False), "B-")
        self.assertEqual(chord_figure(1, "m7", prefer_sharps=False), "D-m7")

    def test_major_suffix_is_empty(self):
        self.assertEqual(chord_figure(0, "", prefer_sharps=True), "C")


class TestBuildScoreChordIntegration(unittest.TestCase):
    """build_score wiring: no key pollution, correct on/off toggle."""

    def _events(self):
        # Two measures at 120 BPM (2s each): C major triad, then A minor triad.
        return [
            NoteEvent(0.0, 2.0, 60, 0.8), NoteEvent(0.0, 2.0, 64, 0.8), NoteEvent(0.0, 2.0, 67, 0.8),
            NoteEvent(2.0, 4.0, 57, 0.8), NoteEvent(2.0, 4.0, 60, 0.8), NoteEvent(2.0, 4.0, 64, 0.8),
        ]

    def test_chords_true_adds_symbols(self):
        score, _key, _summary, _n = build_score(self._events(), 120.0, title="T", chords=True)
        symbols = list(score.recurse().getElementsByClass("ChordSymbol"))
        self.assertGreaterEqual(len(symbols), 1)

    def test_chords_false_adds_no_symbols(self):
        score, _key, _summary, _n = build_score(self._events(), 120.0, title="T", chords=False)
        symbols = list(score.recurse().getElementsByClass("ChordSymbol"))
        self.assertEqual(len(symbols), 0)

    def test_chords_do_not_change_detected_key(self):
        _score_off, key_off, _s1, _n1 = build_score(self._events(), 120.0, title="T", chords=False)
        _score_on, key_on, _s2, _n2 = build_score(self._events(), 120.0, title="T", chords=True)
        self.assertEqual(key_off, key_on)


if __name__ == "__main__":
    unittest.main()
