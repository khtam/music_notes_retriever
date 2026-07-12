"""Quantized note events -> chord symbols, via per-measure template matching.

Each measure's sounding pitch classes are weighted by duration and matched
against a small vocabulary of triad/seventh/sus templates. Consecutive
measures carrying the same chord collapse into a single symbol at the first
measure (a "change points only" reading, the way a lead sheet is written).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

MEASURE_BEATS = 4.0

# Figure suffix -> pitch-class offsets from the root (music21 harmony figures).
QUALITIES: dict[str, tuple[int, ...]] = {
    "": (0, 4, 7),
    "m": (0, 3, 7),
    "7": (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11),
    "m7": (0, 3, 7, 10),
    "sus4": (0, 5, 7),
}

BASS_BONUS = 0.25    # bonus (as a fraction of measure weight) when the bass note is the root
MIN_WEIGHT = 1.0     # beats of pitch mass required before a measure can be judged
CONF_MIN = 0.30       # minimum (present-outside)/total to accept a chord for a measure

SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_NAMES = ["C", "D-", "D", "E-", "E", "F", "G-", "G", "A-", "A", "B-", "B"]


@dataclass(frozen=True)
class DetectedChord:
    offset_beats: float
    root_pc: int
    quality: str


def _measure_weights(
    quantized: list[tuple[float, float, int]], measure_beats: float = MEASURE_BEATS
) -> list[tuple[dict[int, float], Optional[int]]]:
    """Per measure: {pitch_class: duration_weight}, plus the pitch class of the
    lowest sounding pitch (bass). Notes that span a barline contribute their
    clipped overlap to each measure they touch."""
    if not quantized:
        return []
    last_end = max(onset + dur for onset, dur, _ in quantized)
    n_measures = max(int(last_end // measure_beats) + 1, 1)
    weights: list[dict[int, float]] = [dict() for _ in range(n_measures)]
    lowest: list[Optional[int]] = [None] * n_measures

    for onset, dur, pitch in quantized:
        end = onset + dur
        first_m = int(onset // measure_beats)
        last_m = int((end - 1e-9) // measure_beats)
        pc = pitch % 12
        for m in range(first_m, last_m + 1):
            m_start = m * measure_beats
            m_end = m_start + measure_beats
            overlap = min(end, m_end) - max(onset, m_start)
            if overlap <= 0:
                continue
            weights[m][pc] = weights[m].get(pc, 0.0) + overlap
            if lowest[m] is None or pitch < lowest[m]:
                lowest[m] = pitch

    bass_pcs = [(p % 12) if p is not None else None for p in lowest]
    return list(zip(weights, bass_pcs))


def _best_chord(weights: dict[int, float], bass_pc: Optional[int]) -> Optional[tuple[int, str, float]]:
    """Pick the (root, quality) whose template best explains the measure's
    pitch-class weights, using a size-normalized match score so that a
    larger template (e.g. a seventh chord) only wins over a smaller one
    (its triad) when the extra tone(s) carry real weight -- not merely
    because explaining more of the pitch mass looks good in isolation.
    Confidence for the accept/reject gate is reported separately, as the
    plain (matched - unmatched) fraction of the measure's total weight."""
    total = sum(weights.values())
    if total < MIN_WEIGHT:
        return None

    best: Optional[tuple[int, str, float, float]] = None  # (root, quality, rank_score, confidence)
    for root_pc in range(12):
        bonus = BASS_BONUS * total if bass_pc is not None and bass_pc == root_pc else 0.0
        for quality, offsets in QUALITIES.items():
            template = {(root_pc + o) % 12 for o in offsets}
            present = sum(w for pc, w in weights.items() if pc in template)
            outside = total - present
            confidence = (present - outside) / total
            rank_score = (present + bonus - outside) / (len(template) ** 0.5)
            if best is None or rank_score > best[2]:
                best = (root_pc, quality, rank_score, confidence)

    if best is None or best[3] < CONF_MIN:
        return None
    root_pc, quality, _rank_score, confidence = best
    return root_pc, quality, confidence


def detect_chords(
    quantized: list[tuple[float, float, int]], measure_beats: float = MEASURE_BEATS
) -> list[DetectedChord]:
    """Detect one chord symbol per harmonic change, skipping low-confidence
    measures without resetting the currently active chord."""
    per_measure = _measure_weights(quantized, measure_beats)
    chords: list[DetectedChord] = []
    current: Optional[tuple[int, str]] = None
    for m, (weights, bass_pc) in enumerate(per_measure):
        result = _best_chord(weights, bass_pc)
        if result is None:
            continue
        root_pc, quality, _conf = result
        if current is None or (root_pc, quality) != current:
            chords.append(DetectedChord(offset_beats=m * measure_beats, root_pc=root_pc, quality=quality))
            current = (root_pc, quality)
    return chords


def chord_figure(root_pc: int, quality: str, prefer_sharps: bool) -> str:
    """music21 harmony figure string, e.g. 'C#m7' or 'D-m7'."""
    names = SHARP_NAMES if prefer_sharps else FLAT_NAMES
    return f"{names[root_pc % 12]}{quality}"
