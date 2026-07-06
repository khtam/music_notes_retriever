"""Note events -> engraved piano score (music21), exported as MusicXML + MIDI.

Pipeline: quantize onsets/durations to a sixteenth-note grid at the detected
tempo, split notes across treble/bass staves at a configurable pitch, merge
simultaneous notes into chords, then let music21 build measures, ties, rests,
and accidentals.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .transcribe import NoteEvent

GRID = 0.25          # quantization grid in beats (a sixteenth note in 4/4)
MAX_DUR_BEATS = 8.0  # cap runaway sustains at two bars
DEFAULT_SPLIT_MIDI = 60  # middle C: >= goes to treble, < goes to bass


@dataclass
class ScoreInfo:
    title: str
    tempo_bpm: float
    key_name: str
    n_notes: int
    duration_seconds: float
    musicxml_path: Path
    midi_path: Path


def _quantize(events: list[NoteEvent], tempo_bpm: float) -> list[tuple[float, float, int]]:
    """Return (onset_beats, duration_beats, midi_pitch), snapped to GRID,
    shifted so the first onset lands on beat 0."""
    beat = 60.0 / tempo_bpm
    quantized = []
    for ev in events:
        onset = round(ev.start / beat / GRID) * GRID
        offset = round(ev.end / beat / GRID) * GRID
        dur = min(max(offset - onset, GRID), MAX_DUR_BEATS)
        quantized.append((onset, dur, ev.pitch))
    if quantized:
        base = min(o for o, _, _ in quantized)
        quantized = [(o - base, d, p) for o, d, p in quantized]
    return quantized


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


def build_score(
    events: list[NoteEvent],
    tempo_bpm: float,
    title: str = "Transcription",
    split_midi: int = DEFAULT_SPLIT_MIDI,
):
    """Build a two-staff piano music21 Score. Returns (score, key_name)."""
    from music21 import (
        chord,
        clef,
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

    quantized = _quantize(events, tempo_bpm)
    hands = {
        right: _to_chord_sequence([e for e in quantized if e[2] >= split_midi]),
        left: _to_chord_sequence([e for e in quantized if e[2] < split_midi]),
    }
    total_beats = max(
        (onset + dur for seq in hands.values() for onset, dur, _ in seq),
        default=4.0,
    )
    for part, sequence in hands.items():
        for onset, dur, pitches in sequence:
            if len(pitches) == 1:
                element = note.Note(pitches[0])
            else:
                element = chord.Chord(pitches)
            element.quarterLength = dur
            part.insert(onset, element)
        if not sequence:  # keep an empty staff engravable
            rest = note.Rest()
            rest.quarterLength = min(total_beats, 4.0)
            part.insert(0, rest)

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

    brace = layout.StaffGroup([right, left], name="Piano", symbol="brace")
    brace.barTogether = True
    score.insert(0, brace)

    notated = score.makeNotation(inPlace=False)
    return notated, key_name


def export_score(
    events: list[NoteEvent],
    tempo_bpm: float,
    output_dir: Path,
    basename: str,
    title: str,
    split_midi: int = DEFAULT_SPLIT_MIDI,
) -> ScoreInfo:
    output_dir.mkdir(parents=True, exist_ok=True)
    score, key_name = build_score(events, tempo_bpm, title=title, split_midi=split_midi)

    musicxml_path = output_dir / f"{basename}.musicxml"
    midi_path = output_dir / f"{basename}.mid"
    score.write("musicxml", fp=str(musicxml_path))
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
    )
