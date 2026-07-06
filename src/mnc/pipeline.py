"""End-to-end orchestration: source -> audio -> notes -> engraved score."""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .audio_input import prepare_audio
from .score import DEFAULT_SPLIT_MIDI, ScoreInfo, export_score
from .transcribe import estimate_tempo, transcribe

ProgressCallback = Callable[[str], None]


@dataclass
class Options:
    split_midi: int = DEFAULT_SPLIT_MIDI
    tempo_override: Optional[float] = None
    min_note_length_ms: float = 120.0
    onset_threshold: float = 0.5
    frame_threshold: float = 0.3
    title: Optional[str] = None


def slugify(text: str) -> str:
    slug = re.sub(r"[^\w\-]+", "_", text.strip()).strip("_")
    return slug[:80] or "transcription"


def run(
    source: str,
    output_dir: Path,
    options: Options | None = None,
    progress: ProgressCallback | None = None,
) -> ScoreInfo:
    options = options or Options()
    report = progress or (lambda stage: None)

    with tempfile.TemporaryDirectory(prefix="mnc-") as tmp:
        report("Fetching audio")
        wav_path, source_title = prepare_audio(source, Path(tmp))
        title = options.title or source_title

        if options.tempo_override:
            tempo_bpm = float(options.tempo_override)
        else:
            report("Estimating tempo")
            tempo_bpm = estimate_tempo(wav_path)

        report("Transcribing notes (this is the slow part)")
        events = transcribe(
            wav_path,
            onset_threshold=options.onset_threshold,
            frame_threshold=options.frame_threshold,
            min_note_length_ms=options.min_note_length_ms,
        )
        if not events:
            raise RuntimeError(
                "No notes were detected in this audio. Try material with "
                "clearer pitched content, or lower --min-note-length."
            )

        report("Engraving score")
        info = export_score(
            events,
            tempo_bpm,
            output_dir=output_dir,
            basename=slugify(title),
            title=title,
            split_midi=options.split_midi,
        )
    report("Done")
    return info
