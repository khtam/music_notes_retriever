"""Unit tests for the pure logic (no model inference). Run with:

    .venv/bin/python -m unittest discover tests
"""

import unittest

from mnc.cli import parse_pitch
from mnc.llm import LLMError, resolve_provider
from mnc.lyrics import (
    Lyrics,
    LyricLine,
    TimedWord,
    align_user_lyrics,
    lyrics_from_onsets,
    parse_lyric_lines,
    strip_lyric_tags,
)
from mnc.pipeline import slugify
from mnc.score import _make_remap, _plan_repeats, _quantize, _to_chord_sequence, build_score
from mnc.structure import Section, analyze_heuristic, analyze_structure
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
        result, base = _quantize(events, tempo_bpm=120.0)
        self.assertEqual(result, [(0.0, 1.0, 60), (2.0, 2.0, 64)])
        self.assertEqual(base, 1.0)  # first onset was on beat 1

    def test_minimum_duration_is_one_grid_step(self):
        events = [NoteEvent(start=0.0, end=0.01, pitch=60, amplitude=0.5)]
        result, _ = _quantize(events, tempo_bpm=120.0)
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


def _phrase(start_beat: float, pitches: list[int]) -> list[tuple[float, float, int]]:
    """One quarter note per beat starting at start_beat."""
    return [(start_beat + i, 1.0, p) for i, p in enumerate(pitches)]


class TestRepeatPlanning(unittest.TestCase):
    def test_identical_chorus_is_cut(self):
        verse = _phrase(0.0, [60, 62, 64, 65] * 4)
        chorus1 = _phrase(16.0, [67, 69, 71, 72] * 4)
        chorus2 = _phrase(32.0, [67, 69, 71, 72] * 4)
        quantized = verse + chorus1 + chorus2
        ranges = [("Verse 1", 0.0, 16.0), ("Chorus", 16.0, 32.0), ("Chorus", 32.0, 48.0)]
        kept, cut = _plan_repeats(quantized, ranges)
        self.assertEqual([r[0] for r in kept], ["Verse 1", "Chorus"])
        self.assertEqual(cut, [("Chorus", 32.0, 48.0)])

    def test_same_label_different_notes_is_kept(self):
        chorus1 = _phrase(0.0, [67, 69, 71, 72] * 4)
        chorus2 = _phrase(16.0, [40, 42, 44, 45] * 4)  # totally different material
        ranges = [("Chorus", 0.0, 16.0), ("Chorus", 16.0, 32.0)]
        kept, cut = _plan_repeats(chorus1 + chorus2, ranges)
        self.assertEqual(len(kept), 2)
        self.assertEqual(cut, [])

    def test_remap_shifts_past_cut_and_drops_inside(self):
        remap = _make_remap([(16.0, 32.0)])
        self.assertEqual(remap(8.0), 8.0)
        self.assertIsNone(remap(20.0))
        self.assertEqual(remap(32.0), 16.0)
        self.assertEqual(remap(40.0), 24.0)


class TestStructureHeuristic(unittest.TestCase):
    def _lines(self, texts_with_times):
        return [LyricLine(start=s, end=e, text=t) for s, e, t in texts_with_times]

    def test_repeated_lines_become_chorus(self):
        lines = self._lines([
            (0, 4, "walking down the road one day"),
            (4, 8, "thinking about the things you say"),
            (8, 12, "oh oh this is the chorus"),
            (12, 16, "sing it loud this is the chorus"),
            (16, 20, "second verse is different now"),
            (20, 24, "telling you another story somehow"),
            (24, 28, "oh oh this is the chorus"),
            (28, 32, "sing it loud this is the chorus"),
        ])
        sections = analyze_heuristic(lines, duration=32.0)
        self.assertEqual(
            [s.label for s in sections],
            ["Verse 1", "Chorus", "Verse 2", "Chorus"],
        )
        self.assertEqual(sections[0].start, 0.0)
        self.assertEqual(sections[-1].end, 32.0)

    def test_too_few_lines_yields_no_structure(self):
        lines = self._lines([(0, 4, "la la la"), (4, 8, "la la la")])
        sections, method = analyze_structure(lines, duration=8.0)
        self.assertEqual(sections, [])
        self.assertEqual(method, "none")


class _FakeLLM:
    """Stands in for a provider; returns a canned structure answer."""

    name = "fake"

    def generate_json(self, system, prompt, schema):
        return {
            "sections": [
                {"label": "Verse 1", "first_line": 0, "last_line": 1},
                {"label": "Chorus", "first_line": 2, "last_line": 3},
            ]
        }


class TestStructureLLM(unittest.TestCase):
    def test_llm_path_is_used_when_available(self):
        lines = [
            LyricLine(start=i * 5.0, end=i * 5.0 + 4.0, text=f"line {i}")
            for i in range(4)
        ]
        sections, method = analyze_structure(lines, duration=20.0, llm=_FakeLLM())
        self.assertEqual(method, "llm:fake")
        self.assertEqual([s.label for s in sections], ["Verse 1", "Chorus"])
        self.assertEqual(sections[0].start, 0.0)
        self.assertEqual(sections[1].end, 20.0)


def _timed(*specs) -> Lyrics:
    """specs: (start, text) with 0.4 s words."""
    return Lyrics(words=[TimedWord(start=s, end=s + 0.4, text=t) for s, t in specs])


class TestParseLyricLines(unittest.TestCase):
    def test_splits_lines_and_words(self):
        self.assertEqual(
            parse_lyric_lines("Hello there\n\nsecond line!\n"),
            [["Hello", "there"], ["second", "line!"]],
        )

    def test_cjk_splits_per_character(self):
        self.assertEqual(parse_lyric_lines("你好 world"), [["你", "好", "world"]])

    def test_punctuation_only_tokens_dropped(self):
        self.assertEqual(parse_lyric_lines("la — la"), [["la", "la"]])

    def test_section_tag_lines_dropped(self):
        text = "[Verse 1]\nhello world\n[Chorus: Some Artist]\nsing along\n"
        self.assertEqual(parse_lyric_lines(text), [["hello", "world"], ["sing", "along"]])


class TestStripLyricTags(unittest.TestCase):
    def test_bracketed_tags_removed_anywhere(self):
        self.assertEqual(strip_lyric_tags("[Verse 1]").strip(), "")
        self.assertEqual(strip_lyric_tags("{Chorus}").strip(), "")
        self.assertEqual(strip_lyric_tags("la la la [x2]").strip(), "la la la")

    def test_paren_section_and_repeat_tags_removed(self):
        self.assertEqual(strip_lyric_tags("(Chorus)").strip(), "")
        self.assertEqual(strip_lyric_tags("(Verse 2)").strip(), "")
        self.assertEqual(strip_lyric_tags("(Pre-Chorus: Artist)").strip(), "")
        self.assertEqual(strip_lyric_tags("la la la (x2)").strip(), "la la la")
        self.assertEqual(strip_lyric_tags("la la la (repeat 3 times)").strip(), "la la la")

    def test_backing_vocals_in_parens_kept(self):
        self.assertEqual(strip_lyric_tags("hold me close (ooh ooh)"), "hold me close (ooh ooh)")

    def test_bare_heading_lines_dropped(self):
        self.assertEqual(strip_lyric_tags("Chorus:"), "")
        self.assertEqual(strip_lyric_tags("VERSE 2"), "")
        self.assertEqual(strip_lyric_tags("Bridge"), "")

    def test_lyric_line_using_section_word_kept(self):
        line = "standing on the bridge at midnight"
        self.assertEqual(strip_lyric_tags(line), line)

    def test_unbalanced_bracket_headings_dropped(self):
        # The bug the user hit: a stray bracket from a bad copy-paste hid the tag.
        self.assertEqual(strip_lyric_tags("Verse 1]"), "")
        self.assertEqual(strip_lyric_tags("[Pre Chorus"), "")
        self.assertEqual(strip_lyric_tags("**Chorus**"), "")
        self.assertEqual(strip_lyric_tags("Verse 1: Some Artist]"), "")

    def test_cjk_section_tags_dropped(self):
        self.assertEqual(strip_lyric_tags("【副歌】").strip(), "")
        self.assertEqual(strip_lyric_tags("（副歌）").strip(), "")
        self.assertEqual(strip_lyric_tags("副歌："), "")
        self.assertEqual(strip_lyric_tags("主歌 1"), "")

    def test_cjk_lyric_line_kept(self):
        line = "我来到祢面前"
        self.assertEqual(strip_lyric_tags(line), line)


class TestResolveProvider(unittest.TestCase):
    def _clear_env(self):
        import os

        for var in ("MNC_LLM_PROVIDER", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL"):
            os.environ.pop(var, None)

    def setUp(self):
        import os

        self._saved = {
            var: os.environ.get(var)
            for var in ("MNC_LLM_PROVIDER", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL")
        }
        self._clear_env()

    def tearDown(self):
        import os

        for var, value in self._saved.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value

    def test_off_switch_wins(self):
        self.assertIsNone(resolve_provider("none", api_key="sk-ant-xyz"))

    def test_explicit_provider(self):
        self.assertEqual(resolve_provider("anthropic"), "anthropic")
        self.assertEqual(resolve_provider("openai", api_key="sk-ant-xyz"), "openai")

    def test_key_prefix_sniffing(self):
        self.assertEqual(resolve_provider(api_key="sk-ant-xyz"), "anthropic")
        self.assertEqual(resolve_provider(api_key="sk-proj-xyz"), "openai")

    def test_no_provider_no_key_means_heuristic(self):
        self.assertIsNone(resolve_provider())

    def test_unknown_provider_raises(self):
        with self.assertRaises(LLMError):
            resolve_provider("gemini")


class TestAlignUserLyrics(unittest.TestCase):
    def test_matched_words_inherit_reference_times(self):
        reference = _timed((1.0, "twinkle"), (2.0, "twinkle"), (3.0, "little"), (4.0, "star"))
        lyrics = align_user_lyrics("Twinkle, twinkle\nlittle star", reference)
        self.assertIsNotNone(lyrics)
        self.assertEqual([w.text for w in lyrics.words], ["Twinkle,", "twinkle", "little", "star"])
        self.assertEqual([w.start for w in lyrics.words], [1.0, 2.0, 3.0, 4.0])
        # User line breaks define the lines (better structure input than Whisper).
        self.assertEqual([l.text for l in lyrics.lines], ["Twinkle, twinkle", "little star"])
        self.assertEqual(lyrics.lines[1].start, 3.0)

    def test_misheard_word_interpolates_between_anchors(self):
        # Whisper heard "littlest are" where the real lyric is "little star".
        reference = _timed((1.0, "twinkle"), (2.0, "twinkle"), (3.0, "littlest"),
                           (4.0, "are"), (5.0, "how"), (6.0, "I"), (7.0, "wonder"))
        lyrics = align_user_lyrics("twinkle twinkle little star how I wonder", reference)
        self.assertIsNotNone(lyrics)
        starts = [w.start for w in lyrics.words]
        self.assertEqual(starts[0], 1.0)
        self.assertEqual(starts[4], 5.0)  # "how" re-anchors
        self.assertTrue(2.0 < starts[2] < starts[3] < 5.0)  # interpolated, in order
        self.assertEqual(starts, sorted(starts))

    def test_too_few_matches_returns_none(self):
        reference = _timed((1.0, "completely"), (2.0, "unrelated"), (3.0, "transcription"))
        self.assertIsNone(align_user_lyrics("some real lyrics that never matched", reference))

    def test_empty_reference_returns_none(self):
        self.assertIsNone(align_user_lyrics("la la la", Lyrics()))

    def test_words_past_last_anchor_are_extrapolated(self):
        reference = _timed((1.0, "one"), (2.0, "two"), (3.0, "three"))
        lyrics = align_user_lyrics("one two three four five", reference)
        starts = [w.start for w in lyrics.words]
        self.assertEqual(starts[:3], [1.0, 2.0, 3.0])
        self.assertEqual(starts, sorted(starts))
        self.assertGreater(starts[3], 3.0)

    def test_long_unmatched_tail_spreads_across_reference(self):
        # Whisper anchors only the first few words but keeps singing to 100 s.
        # The user's many trailing words must spread toward the reference end,
        # not bunch up 0.4 s apart right after the last anchor (the bug that
        # dragged every late section mark to the start of the score).
        ref_specs = [(float(i + 1), f"m{i}") for i in range(15)]   # matched, 1–15 s
        ref_specs += [(20.0 + i, f"x{i}") for i in range(80)]      # sung tail to ~99 s
        reference = _timed(*ref_specs)
        user = " ".join(f"m{i}" for i in range(15)) + " " + " ".join(f"w{i}" for i in range(40))
        lyrics = align_user_lyrics(user, reference)
        self.assertIsNotNone(lyrics)
        starts = [w.start for w in lyrics.words]
        self.assertEqual(starts, sorted(starts))          # monotonic
        # Old behavior put the last word at 15 + 40*0.4 = 31 s. Spreading uses
        # the reference span (ends ~99.4 s), so the tail must reach far past that.
        self.assertGreater(starts[-1], 60.0)
        self.assertLessEqual(starts[-1], reference.words[-1].end + 0.01)


class TestLyricsFromOnsets(unittest.TestCase):
    def test_one_word_per_onset(self):
        lyrics = lyrics_from_onsets("la la\nla", [0.0, 1.0, 2.0, 3.0])
        self.assertEqual([w.start for w in lyrics.words], [0.0, 1.0, 2.0])
        self.assertEqual(len(lyrics.lines), 2)

    def test_extra_words_dropped_when_onsets_run_out(self):
        lyrics = lyrics_from_onsets("one two three four", [0.0, 1.0])
        self.assertEqual([w.text for w in lyrics.words], ["one", "two"])

    def test_no_onsets_yields_empty_lyrics(self):
        self.assertFalse(lyrics_from_onsets("la la", []))


class TestBuildScoreExtras(unittest.TestCase):
    def _events(self, beat_specs, tempo=120.0):
        beat = 60.0 / tempo
        return [
            NoteEvent(start=b * beat, end=(b + 0.8) * beat, pitch=p, amplitude=0.5)
            for b, p in beat_specs
        ]

    def test_lyrics_attach_to_melody_notes(self):
        events = self._events([(0, 72), (1, 74), (2, 76)])
        lyrics = Lyrics(words=[
            TimedWord(start=0.02, end=0.4, text="Twin"),
            TimedWord(start=0.52, end=0.9, text="kle"),
            TimedWord(start=1.03, end=1.4, text="star"),
        ])
        score, _, _, n_words = build_score(events, tempo_bpm=120.0, lyrics=lyrics)
        self.assertEqual(n_words, 3)
        attached = [n.lyric for n in score.parts[0].flatten().notes if n.lyric]
        self.assertEqual(attached, ["Twin", "kle", "star"])

    def test_repeated_chorus_collapses_measures(self):
        # 120 BPM: a beat is 0.5 s, a measure is 2 s. Verse mm. 1-4,
        # chorus mm. 5-8, then the same chorus again (mm. 9-12 uncollapsed).
        verse = [(i, [60, 62, 64, 65][i % 4]) for i in range(16)]
        chorus = [(16 + i, [67, 69, 71, 72][i % 4]) for i in range(16)]
        chorus_again = [(32 + i, [67, 69, 71, 72][i % 4]) for i in range(16)]
        events = self._events(verse + chorus + chorus_again)
        sections = [
            Section("Verse 1", 0.0, 8.0),
            Section("Chorus", 8.0, 16.0),
            Section("Chorus", 16.0, 24.0),
        ]
        score, _, summary, _ = build_score(events, tempo_bpm=120.0, sections=sections)
        n_measures = len(score.parts[0].getElementsByClass("Measure"))
        self.assertEqual(n_measures, 8)  # 12 measures of music engraved as 8
        self.assertEqual(len(summary), 3)
        self.assertIn("repeat of mm. 5", summary[2])

        # Without dedup all 12 measures are engraved.
        full, _, full_summary, _ = build_score(
            events, tempo_bpm=120.0, sections=sections, dedup=False
        )
        self.assertEqual(len(full.parts[0].getElementsByClass("Measure")), 12)
        self.assertEqual(len(full_summary), 3)


class TestAccidentals(unittest.TestCase):
    def _events(self, beat_specs, tempo=120.0):
        beat = 60.0 / tempo
        return [
            NoteEvent(start=b * beat, end=(b + 0.8) * beat, pitch=p, amplitude=0.5)
            for b, p in beat_specs
        ]

    def _displayed_accidentals(self, score):
        return [
            p.accidental.name
            for part in score.parts
            for n in part.flatten().notes
            for p in n.pitches
            if p.accidental is not None and p.accidental.displayStatus
        ]

    def test_diatonic_melody_has_no_naturals(self):
        # All white keys, spanning octaves (like "Twinkle Twinkle"): a
        # diatonic melody should carry zero displayed accidentals.
        events = self._events([
            (0, 60), (1, 60), (2, 67), (3, 67),
            (4, 69), (5, 69), (6, 67),
        ])
        score, _, _, _ = build_score(events, tempo_bpm=120.0)
        self.assertEqual(self._displayed_accidentals(score), [])

    def test_necessary_natural_still_shown(self):
        # F#4 followed by F-natural4 in the same measure: the cancelling
        # natural must still be engraved even though spurious ones are not.
        events = self._events([(0, 66), (1, 65), (2, 60), (3, 72)])
        score, _, _, _ = build_score(events, tempo_bpm=120.0)
        accidentals = self._displayed_accidentals(score)
        self.assertIn("sharp", accidentals)
        self.assertIn("natural", accidentals)
        self.assertEqual(accidentals.count("natural"), 1)


if __name__ == "__main__":
    unittest.main()
