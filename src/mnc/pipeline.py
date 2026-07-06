"""End-to-end orchestration: source -> audio -> notes + lyrics + structure -> score."""

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
    lyrics: bool = True                 # transcribe vocals and print words under the melody
    lyrics_text: Optional[str] = None   # user-supplied lyrics; replaces Whisper's words
    structure: bool = True              # label verse/chorus/bridge sections
    dedup_repeats: bool = True          # collapse near-identical repeated sections
    whisper_model: str = "small"        # faster-whisper size: tiny/base/small/medium/large-v3
    llm_provider: Optional[str] = None  # anthropic | openai | none; None = auto-detect from env
    llm_model: Optional[str] = None


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

        # Vocals are optional extras: if Whisper or the LLM fails, the score
        # still ships — just without words or section labels.
        lyrics = None
        lyrics_source = ""
        sections: list = []
        structure_method = ""
        user_text = (options.lyrics_text or "").strip()
        want_lyrics = options.lyrics or bool(user_text)
        if user_text:
            # The user's words are the truth; Whisper only supplies timing.
            from .lyrics import align_user_lyrics, lyrics_from_onsets, transcribe_lyrics

            report("Transcribing vocals to time your lyrics")
            reference = None
            try:
                reference = transcribe_lyrics(wav_path, model_size=options.whisper_model)
            except Exception as exc:
                report(f"Vocal timing pass failed ({exc}); mapping lyrics to melody notes")
            if reference:
                lyrics = align_user_lyrics(user_text, reference)
                if lyrics:
                    lyrics_source = "aligned to vocals"
            if lyrics is None:
                onsets = [ev.start for ev in events if ev.pitch >= options.split_midi]
                lyrics = lyrics_from_onsets(user_text, onsets)
                lyrics_source = "mapped to melody notes"
        elif options.lyrics or options.structure:
            report("Transcribing lyrics")
            try:
                from .lyrics import transcribe_lyrics

                lyrics = transcribe_lyrics(wav_path, model_size=options.whisper_model)
                lyrics_source = "transcribed"
            except Exception as exc:
                report(f"Lyric transcription failed ({exc}); continuing without lyrics")
                lyrics = None
        if options.structure and lyrics and lyrics.lines:
            report("Analyzing song structure")
            from .llm import LLMError, get_llm_client
            from .structure import analyze_structure

            try:
                llm = get_llm_client(options.llm_provider, options.llm_model)
            except LLMError as exc:
                report(f"LLM unavailable ({exc}); using heuristic structure analysis")
                llm = None
            duration = max(
                max((ev.end for ev in events), default=0.0),
                lyrics.lines[-1].end,
            )
            sections, structure_method = analyze_structure(lyrics.lines, duration, llm)

        report("Engraving score")
        info = export_score(
            events,
            tempo_bpm,
            output_dir=output_dir,
            basename=slugify(title),
            title=title,
            split_midi=options.split_midi,
            lyrics=lyrics if want_lyrics else None,
            sections=sections,
            dedup=options.dedup_repeats,
            structure_method=structure_method,
            lyrics_source=lyrics_source if want_lyrics else "",
        )
    report("Done")
    return info
