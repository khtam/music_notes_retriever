"""Unit tests for the pure logic (no model inference). Run with:

    .venv/bin/python -m unittest discover tests
"""

import unittest

from mnc.cli import parse_pitch
from mnc.pipeline import slugify
from mnc.score import _quantize, _to_chord_sequence
from mnc.transcribe import NoteEvent


class TestParsePitch(unittest.TestCase):
    def test_midi_number(self):
        self.assertEqual(parse_pitch("60"), 60)

    def test_note_names(self):
        self.assertEqual(parse_pitch("C4"), 60)
        self.assertEqual(parse_pitch("A4"), 69)
        self.assertEqual(parse_pitch("F#3"), 54)
        self.assertEqual(parse_pitch("Bb2"), 46)

    def test_out_of_range(self):
        import argparse

        with self.assertRaises(argparse.ArgumentTypeError):
            parse_pitch("C9")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_pitch("xyz")


class TestQuantize(unittest.TestCase):
    def test_snaps_to_grid_and_shifts_to_zero(self):
        events = [
            NoteEvent(start=0.51, end=1.02, pitch=60, amplitude=0.5),
            NoteEvent(start=1.49, end=2.55, pitch=64, amplitude=0.5),
        ]
        # 120 BPM -> beat = 0.5 s
        result = _quantize(events, tempo_bpm=120.0)
        self.assertEqual(result, [(0.0, 1.0, 60), (2.0, 2.0, 64)])

    def test_minimum_duration_is_one_grid_step(self):
        events = [NoteEvent(start=0.0, end=0.01, pitch=60, amplitude=0.5)]
        result = _quantize(events, tempo_bpm=120.0)
        self.assertEqual(result[0][1], 0.25)


class TestChordSequence(unittest.TestCase):
    def test_same_onset_becomes_chord(self):
        seq = _to_chord_sequence([(0.0, 1.0, 60), (0.0, 2.0, 64), (0.0, 1.0, 67)])
        self.assertEqual(seq, [(0.0, 2.0, [60, 64, 67])])

    def test_overlap_truncated_to_next_attack(self):
        seq = _to_chord_sequence([(0.0, 4.0, 60), (1.0, 1.0, 62)])
        self.assertEqual(seq[0], (0.0, 1.0, [60]))

    def test_gaps_preserved(self):
        seq = _to_chord_sequence([(0.0, 1.0, 60), (3.0, 1.0, 62)])
        self.assertEqual(seq, [(0.0, 1.0, [60]), (3.0, 1.0, [62])])

    def test_duplicate_pitches_merged(self):
        seq = _to_chord_sequence([(0.0, 1.0, 60), (0.0, 1.5, 60)])
        self.assertEqual(seq, [(0.0, 1.5, [60])])


class TestSlugify(unittest.TestCase):
    def test_strips_punctuation(self):
        self.assertEqual(slugify("My Song (Official Video!)"), "My_Song_Official_Video")

    def test_empty_fallback(self):
        self.assertEqual(slugify("???"), "transcription")


if __name__ == "__main__":
    unittest.main()
