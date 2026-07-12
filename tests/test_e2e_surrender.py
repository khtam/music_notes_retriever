"""End-to-end: transcribe the local Surrender mp3 and check the score against
the reference sheet surrender_notes.jpeg.

Both files are untracked, machine-local reference material (not committed to
the repo), so this test class is skipped wherever they aren't present.

Reference sheet (surrender_notes.jpeg), read by hand:
  - Key: A major (3 sharps)
  - Time signature: 4/4
  - Tempo marking: quarter note = 64
  - Grand staff piano arrangement (treble + bass), no lyrics printed
  - Chord symbols across the piece: D, A, Bm7, E7, F#m7, Esus4, E, F#m,
    Dmaj7, C#m7, A11 (roots: D, A, B, E, F#, C# -> pitch classes {2,9,11,4,6,1})
  - ~29 measures shown on page 1; the source track is ~330s long, so at
    quarter=64 in 4/4 the full song is roughly 88 measures.

This is a note-for-note-imperfect automatic transcription of a full mix, not
a hand-engraved arrangement, so the test asserts musical properties derived
from the sheet (key, tempo, meter, measure count, chord vocabulary) rather
than exact note matching. Run with:

    .venv/bin/python -m unittest discover tests
"""

import unittest
from pathlib import Path

MP3 = Path(__file__).resolve().parents[1] / "【降服 ⧸ Surrender】歌詞MV - 約書亞樂團 ft. ZEcho｜男key.mp3"

# A major diatonic pitch classes: A B C# D E F# G#
A_MAJOR_PCS = {9, 11, 1, 2, 4, 6, 8}
# Roots of the chord symbols printed on the reference sheet: D A B E F# C#
REFERENCE_ROOT_PCS = {2, 9, 11, 4, 6, 1}


@unittest.skipUnless(MP3.exists(), "surrender reference mp3 not present (local-only test material)")
class TestSurrenderEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from music21 import converter

        from mnc.pipeline import Options, run

        out_dir = Path(__file__).parent / "out" / "e2e_surrender"
        cls.info = run(
            str(MP3),
            out_dir,
            Options(lyrics=False, structure=False),  # sheet has no lyrics; auto tempo
        )
        cls.score = converter.parse(str(cls.info.musicxml_path))
        cls.n_measures = len(cls.score.parts[0].getElementsByClass("Measure"))
        cls.chord_symbols = list(cls.score.recurse().getElementsByClass("ChordSymbol"))

    def test_key_is_a_major(self):
        self.assertEqual(self.info.key_name, "A major")
        signatures = list(self.score.recurse().getElementsByClass("KeySignature"))
        self.assertTrue(signatures, "no KeySignature found in the score")
        self.assertEqual(signatures[0].sharps, 3)

    def test_tempo_in_ballad_range(self):
        # Sheet marks quarter=64; the old (pre-fix) pipeline reported 129
        # (an exact octave error), so this range pins the tempo fix.
        self.assertTrue(58 <= self.info.tempo_bpm <= 72, f"tempo {self.info.tempo_bpm} not in [58, 72]")

    def test_time_signature_is_4_4(self):
        signatures = list(self.score.recurse().getElementsByClass("TimeSignature"))
        self.assertTrue(signatures, "no TimeSignature found in the score")
        self.assertEqual(signatures[0].ratioString, "4/4")

    def test_measure_count_matches_song_length(self):
        # ~330s at quarter=64 in 4/4 -> ~88 measures. A 129 BPM regression
        # would roughly double this to ~176, so this independently pins the
        # tempo fix even if test_tempo_in_ballad_range were loosened later.
        self.assertTrue(70 <= self.n_measures <= 106, f"measure count {self.n_measures} not in [70, 106]")

    def test_grand_staff_two_parts(self):
        self.assertEqual(len(self.score.parts), 2)
        treble_clefs = list(self.score.parts[0].recurse().getElementsByClass("TrebleClef"))
        bass_clefs = list(self.score.parts[1].recurse().getElementsByClass("BassClef"))
        self.assertTrue(treble_clefs, "right-hand part has no TrebleClef")
        self.assertTrue(bass_clefs, "left-hand part has no BassClef")

    def test_chord_symbols_present_and_plausible(self):
        self.assertGreaterEqual(
            len(self.chord_symbols), max(1, self.n_measures // 8),
            f"only {len(self.chord_symbols)} chord symbols across {self.n_measures} measures",
        )
        roots = [cs.root().pitchClass for cs in self.chord_symbols]
        diatonic_fraction = sum(1 for r in roots if r in A_MAJOR_PCS) / len(roots)
        self.assertGreaterEqual(
            diatonic_fraction, 0.6,
            f"only {diatonic_fraction:.0%} of chord roots are diatonic to A major",
        )
        overlap = set(roots) & REFERENCE_ROOT_PCS
        self.assertGreaterEqual(
            len(overlap), 3,
            f"detected chord roots {set(roots)} overlap reference roots {REFERENCE_ROOT_PCS} in only {overlap}",
        )
        self.assertEqual(self.info.n_chord_symbols, len(self.chord_symbols))

    def test_outputs_exist_and_are_substantial(self):
        self.assertTrue(self.info.musicxml_path.exists())
        self.assertTrue(self.info.midi_path.exists())
        self.assertGreater(self.info.n_notes, 500)
        self.assertTrue(250 <= self.info.duration_seconds <= 400)


if __name__ == "__main__":
    unittest.main()
