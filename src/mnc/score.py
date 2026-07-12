"""Note events -> engraved piano score (music21), exported as MusicXML + MIDI.

Pipeline: quantize onsets/durations to a sixteenth-note grid at the detected
tempo, split notes across treble/bass staves at a configurable pitch, merge
simultaneous notes into chords, then let music21 build measures, ties, rests,
and accidentals.

Optionally the score also carries:
  * lyrics — timed words attached to the nearest right-hand (melody) note;
  * sections — rehearsal marks (Verse 1, Chorus, ...) at section starts;
  * repeat collapsing — a later section whose label matches an earlier one
    and whose notes are near-identical is cut and replaced with a
    "Chorus: play mm. X–Y" direction, so choruses aren't engraved twice.
"""

from __future__ import annotations

import bisect
import contextlib
import io
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .chords import chord_figure, detect_chords
from .transcribe import NoteEvent

if TYPE_CHECKING:
    from .lyrics import Lyrics
    from .structure import Section

GRID = 0.25          # quantization grid in beats (a sixteenth note in 4/4)
MAX_DUR_BEATS = 8.0  # cap runaway sustains at two bars
DEFAULT_SPLIT_MIDI = 60  # middle C: >= goes to treble, < goes to bass
MEASURE_BEATS = 4.0  # everything is engraved in 4/4

LYRIC_TOLERANCE_BEATS = 1.0   # max onset distance when pairing a word to a note
REPEAT_SIMILARITY = 0.4       # min note-content Jaccard to collapse a repeat


@dataclass
class ScoreInfo:
    title: str
    tempo_bpm: float
    key_name: str
    n_notes: int
    duration_seconds: float
    musicxml_path: Path
    midi_path: Path
    sections: list[str] = field(default_factory=list)
    structure_method: str = ""
    n_lyric_words: int = 0
    lyrics_language: str = ""
    lyrics_source: str = ""  # "transcribed" | "aligned to vocals" | "mapped to melody notes"
    n_chord_symbols: int = 0


def _quantize(
    events: list[NoteEvent], tempo_bpm: float
) -> tuple[list[tuple[float, float, int]], float]:
    """Return ((onset_beats, duration_beats, midi_pitch) snapped to GRID, base)
    where base is the beat shift applied so the first onset lands on beat 0."""
    beat = 60.0 / tempo_bpm
    quantized = []
    for ev in events:
        onset = round(ev.start / beat / GRID) * GRID
        offset = round(ev.end / beat / GRID) * GRID
        dur = min(max(offset - onset, GRID), MAX_DUR_BEATS)
        quantized.append((onset, dur, ev.pitch))
    base = 0.0
    if quantized:
        base = min(o for o, _, _ in quantized)
        quantized = [(o - base, d, p) for o, d, p in quantized]
    return quantized, base


def _to_chord_sequence(events: list[tuple[float, float, int]]) -> list[tuple[float, float, list[int]]]:
    """Group same-onset notes into chords and truncate durations so nothing
    overlaps the next attack (single-voice engraving keeps scores readable)."""
    by_onset: dict[float, list[tuple[float, int]]] = defaultdict(list)
    for onset, dur, pitch in events:
        by_onset[onset].append((dur, pitch))
    onsets = sorted(by_onset)
    sequence = []
    for i, onset in enumerate(onsets):
        group = by_onset[onset]
        pitches = sorted({p for _, p in group})
        dur = max(d for d, _ in group)
        if i + 1 < len(onsets):
            dur = min(dur, onsets[i + 1] - onset)
        dur = max(dur, GRID)
        sequence.append((onset, dur, pitches))
    return sequence


# --- song structure: section marks and repeat collapsing -------------------

def _snap_to_measure(beats: float) -> float:
    return round(beats / MEASURE_BEATS) * MEASURE_BEATS


def _section_beat_ranges(
    sections: list["Section"], tempo_bpm: float, base_beats: float
) -> list[tuple[str, float, float]]:
    """Section time ranges -> (label, start_beat, end_beat) snapped to barlines."""
    beat = 60.0 / tempo_bpm
    ranges = []
    for sec in sections:
        start = max(_snap_to_measure(sec.start / beat - base_beats), 0.0)
        end = _snap_to_measure(sec.end / beat - base_beats)
        if end - start >= MEASURE_BEATS:
            ranges.append((sec.label.strip() or "Section", start, end))
    return ranges


def _note_cells(quantized: list[tuple[float, float, int]], start: float, end: float) -> set:
    """Content fingerprint of a span: (grid step within span, pitch) pairs."""
    return {
        (int(round((onset - start) / GRID)), pitch)
        for onset, _, pitch in quantized
        if start <= onset < end
    }


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _plan_repeats(
    quantized: list[tuple[float, float, int]],
    ranges: list[tuple[str, float, float]],
) -> tuple[list[tuple[str, float, float]], list[tuple[str, float, float]]]:
    """Split sections into (kept, cut). A section is cut when an earlier
    section carries the same label and near-identical note content."""
    first_seen: dict[str, tuple[float, float]] = {}
    kept, cut = [], []
    for label, start, end in ranges:
        key = label.lower()
        if key in first_seen:
            f_start, f_end = first_seen[key]
            similar_length = abs((end - start) - (f_end - f_start)) <= 0.3 * max(end - start, f_end - f_start)
            similarity = _jaccard(
                _note_cells(quantized, start, end),
                _note_cells(quantized, f_start, f_end),
            )
            if similar_length and similarity >= REPEAT_SIMILARITY:
                cut.append((label, start, end))
                continue
        else:
            first_seen[key] = (start, end)
        kept.append((label, start, end))
    return kept, cut


def _make_remap(cut_spans: list[tuple[float, float]]):
    """Map a pre-cut beat position to its post-cut position.

    Positions inside a removed [start, end) span map to None; with
    boundary=True they map to the junction where the cut was made instead
    (for placing section marks, which must never vanish).
    """
    spans = sorted(cut_spans)

    def remap(beats: float, boundary: bool = False) -> Optional[float]:
        shift = 0.0
        for start, end in spans:
            if beats >= end:
                shift += end - start
            elif beats >= start:
                return start - shift if boundary else None
        return beats - shift

    return remap


def _measure_number(beats: float) -> int:
    return int(beats // MEASURE_BEATS) + 1


# --- score assembly ---------------------------------------------------------

def build_score(
    events: list[NoteEvent],
    tempo_bpm: float,
    title: str = "Transcription",
    split_midi: int = DEFAULT_SPLIT_MIDI,
    lyrics: Optional["Lyrics"] = None,
    sections: Optional[list["Section"]] = None,
    dedup: bool = True,
    chords: bool = True,
):
    """Build a two-staff piano music21 Score.

    Returns (score, key_name, section_summary, n_lyric_words) where
    section_summary lists the engraved structure, e.g.
    ["Verse 1 — mm. 1–8", "Chorus — mm. 9–16", "Chorus (repeat of mm. 9–16, not re-engraved)"].
    """
    from music21 import (
        chord,
        clef,
        expressions,
        harmony,
        instrument,
        key,
        layout,
        metadata,
        meter,
        note,
        stream,
        tempo,
    )

    right = stream.Part(id="RH")
    left = stream.Part(id="LH")
    right.partName = left.partName = "Piano"
    right.insert(0, instrument.Piano())
    left.insert(0, instrument.Piano())
    right.insert(0, clef.TrebleClef())
    left.insert(0, clef.BassClef())
    for part in (right, left):
        part.insert(0, meter.TimeSignature("4/4"))
    right.insert(0, tempo.MetronomeMark(number=round(tempo_bpm)))

    quantized, base_beats = _quantize(events, tempo_bpm)
    beat_seconds = 60.0 / tempo_bpm

    # Plan the structure: which sections stay, which collapse into repeats.
    ranges = _section_beat_ranges(sections or [], tempo_bpm, base_beats)
    kept_sections, cut_sections = _plan_repeats(quantized, ranges) if dedup else (ranges, [])
    remap = _make_remap([(s, e) for _, s, e in cut_sections])
    first_range = {label.lower(): (s, e) for label, s, e in kept_sections}

    # Apply the cuts: drop notes inside removed spans, truncate sustains that
    # cross into one, and shift everything after a cut left to close the gap.
    cut_spans = sorted((s, e) for _, s, e in cut_sections)
    remapped: list[tuple[float, float, int]] = []
    for onset, dur, pitch in quantized:
        for span_start, _ in cut_spans:
            if onset < span_start < onset + dur:
                dur = span_start - onset
        new_onset = remap(onset)
        if new_onset is not None:
            remapped.append((new_onset, dur, pitch))
    quantized = remapped

    hands = {
        right: _to_chord_sequence([e for e in quantized if e[2] >= split_midi]),
        left: _to_chord_sequence([e for e in quantized if e[2] < split_midi]),
    }
    total_beats = max(
        (onset + dur for seq in hands.values() for onset, dur, _ in seq),
        default=4.0,
    )
    melody_onsets: list[float] = []
    melody_elements: list = []
    for part, sequence in hands.items():
        for onset, dur, pitches in sequence:
            if len(pitches) == 1:
                element = note.Note(pitches[0])
            else:
                element = chord.Chord(pitches)
            # music21 attaches an explicit natural accidental to every white-key
            # pitch built from a MIDI int. Dropping it lets makeAccidentals show
            # a natural only when it actually cancels a prior sharp/flat in the
            # bar, instead of on every diatonic note.
            for p in element.pitches:
                if p.accidental is not None and p.accidental.name == "natural":
                    p.accidental = None
            element.quarterLength = dur
            part.insert(onset, element)
            if part is right:
                melody_onsets.append(onset)
                melody_elements.append(element)
        if not sequence:  # keep an empty staff engravable
            rest = note.Rest()
            rest.quarterLength = min(total_beats, 4.0)
            part.insert(0, rest)

    # Lyrics: attach each word to the nearest melody note (post-cut timeline).
    n_lyric_words = 0
    if lyrics and melody_onsets:
        for word in lyrics.words:
            beats = remap(word.start / beat_seconds - base_beats)
            if beats is None:  # word fell inside a collapsed repeat
                continue
            i = bisect.bisect_left(melody_onsets, beats)
            best = None
            for j in (i - 1, i):
                if 0 <= j < len(melody_onsets) and abs(melody_onsets[j] - beats) <= LYRIC_TOLERANCE_BEATS:
                    if best is None or abs(melody_onsets[j] - beats) < abs(melody_onsets[best] - beats):
                        best = j
            if best is None:
                continue
            element = melody_elements[best]
            element.lyric = f"{element.lyric} {word.text}" if element.lyric else word.text
            n_lyric_words += 1

    # Section marks: rehearsal marks for kept sections, a "play mm. X-Y"
    # direction where a repeated section was cut. Summary stays in song order.
    summary_entries: list[tuple[float, str]] = []
    for label, start, end in kept_sections:
        offset = remap(start, boundary=True)
        if offset >= total_beats:
            continue
        right.insert(offset, expressions.RehearsalMark(label))
        m1 = _measure_number(offset)
        m2 = max(_measure_number(remap(end, boundary=True)) - 1, m1)
        summary_entries.append((start, f"{label} — mm. {m1}–{m2}"))
    for label, start, end in cut_sections:
        f_start, f_end = first_range[label.lower()]
        m1 = _measure_number(remap(f_start, boundary=True))
        m2 = max(_measure_number(remap(f_end, boundary=True)) - 1, m1)
        # Place the direction at the junction; a cut at the very end of the
        # song goes just inside the final measure instead of adding an empty one.
        offset = min(remap(start, boundary=True), max(total_beats - GRID, 0.0))
        direction = expressions.TextExpression(f"{label}: play mm. {m1}–{m2}")
        direction.style.fontWeight = "bold"
        right.insert(offset, direction)
        summary_entries.append((start, f"{label} — repeat of mm. {m1}–{m2} (not re-engraved)"))
    section_summary = [text for _, text in sorted(summary_entries)]

    score = stream.Score()
    score.insert(0, metadata.Metadata(title=title, composer="Music Notes Creator"))
    score.insert(0, right)
    score.insert(0, left)

    try:
        detected = score.analyze("key")
        key_name = f"{detected.tonic.name} {detected.mode}"
        signature = key.KeySignature(detected.sharps)
    except Exception:
        key_name = "C major"
        signature = key.KeySignature(0)
    for part in (right, left):
        part.insert(0, signature)

    if chords and quantized:
        prefer_sharps = signature.sharps >= 0
        for dc in detect_chords(quantized):
            if dc.offset_beats < total_beats:
                figure = chord_figure(dc.root_pc, dc.quality, prefer_sharps)
                right.insert(dc.offset_beats, harmony.ChordSymbol(figure))

    brace = layout.StaffGroup([right, left], name="Piano", symbol="brace")
    brace.barTogether = True
    score.insert(0, brace)

    # music21 warns to stderr about every beam pair it has to repair, which on
    # dense transcribed rhythms floods the terminal; the repairs themselves are fine.
    with contextlib.redirect_stderr(io.StringIO()):
        notated = score.makeNotation(inPlace=False)
    return notated, key_name, section_summary, n_lyric_words


def _strip_chord_symbols(score) -> None:
    """Remove ChordSymbol objects in place. music21's MIDI export doesn't skip
    Harmony objects (they're a Chord subclass), so left in place they'd add
    audible blips to the .mid; MusicXML export already ignores them
    (writeAsChord=False), so this only needs to run before the MIDI write."""
    for cs in list(score.recurse().getElementsByClass("ChordSymbol")):
        cs.activeSite.remove(cs)


def export_score(
    events: list[NoteEvent],
    tempo_bpm: float,
    output_dir: Path,
    basename: str,
    title: str,
    split_midi: int = DEFAULT_SPLIT_MIDI,
    lyrics: Optional["Lyrics"] = None,
    sections: Optional[list["Section"]] = None,
    dedup: bool = True,
    structure_method: str = "",
    lyrics_source: str = "",
    chords: bool = True,
) -> ScoreInfo:
    output_dir.mkdir(parents=True, exist_ok=True)
    score, key_name, section_summary, n_lyric_words = build_score(
        events,
        tempo_bpm,
        title=title,
        split_midi=split_midi,
        lyrics=lyrics,
        sections=sections,
        dedup=dedup,
        chords=chords,
    )

    n_chord_symbols = len(score.recurse().getElementsByClass("ChordSymbol"))

    musicxml_path = output_dir / f"{basename}.musicxml"
    midi_path = output_dir / f"{basename}.mid"
    score.write("musicxml", fp=str(musicxml_path))
    _strip_chord_symbols(score)
    score.write("midi", fp=str(midi_path))

    duration = max((ev.end for ev in events), default=0.0)
    return ScoreInfo(
        title=title,
        tempo_bpm=tempo_bpm,
        key_name=key_name,
        n_notes=len(events),
        duration_seconds=duration,
        musicxml_path=musicxml_path,
        midi_path=midi_path,
        sections=section_summary,
        structure_method=structure_method,
        n_lyric_words=n_lyric_words,
        lyrics_language=(lyrics.language if lyrics else ""),
        lyrics_source=(lyrics_source if n_lyric_words else ""),
        n_chord_symbols=n_chord_symbols,
    )
