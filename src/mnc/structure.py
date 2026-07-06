"""Song-structure analysis: label sections (verse, chorus, bridge, ...) in time.

Two strategies, best available wins:
  1. LLM: given timestamped lyric lines, a language model labels the sections.
     Models are far better than signal processing at "this is the chorus".
  2. Heuristic: cluster repeated lyric lines; repeated clusters become
     choruses, unique stretches become verses. Used when no LLM is configured
     or the LLM call fails.

Sections come back as labeled time ranges; the score builder turns them into
rehearsal marks and (optionally) deduplicated repeats.
"""

from __future__ import annotations

import difflib
import re
from collections import Counter
from dataclasses import dataclass

from .llm import LLMClient, LLMError
from .lyrics import LyricLine

MIN_SECTION_SECONDS = 4.0
FUZZY_MATCH = 0.7  # SequenceMatcher ratio at which two lyric lines count as the same line


@dataclass
class Section:
    label: str
    start: float
    end: float


_SECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "description": "e.g. Intro, Verse 1, Pre-Chorus, Chorus, Verse 2, Bridge, Outro/Coda",
                    },
                    "first_line": {"type": "integer"},
                    "last_line": {"type": "integer"},
                },
                "required": ["label", "first_line", "last_line"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["sections"],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = (
    "You are a musicologist annotating song structure for sheet-music engraving. "
    "Given timestamped lyric lines, group them into musical sections (Intro, "
    "Verse 1, Pre-Chorus, Chorus, Verse 2, Bridge, Outro/Coda, ...). Repeats of "
    "the same material must reuse the same base label (every chorus is labeled "
    "'Chorus'; verses are numbered 'Verse 1', 'Verse 2'). Every lyric line "
    "belongs to exactly one section; sections are contiguous, non-overlapping "
    "ranges of line indices in the given order."
)


def _fmt_time(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}:{secs:02d}"


def _normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def _sections_from_line_ranges(
    ranges: list[tuple[str, int, int]], lines: list[LyricLine], duration: float
) -> list[Section]:
    """Convert (label, first_line, last_line) into contiguous time ranges."""
    sections = []
    for label, first, last in ranges:
        if not (0 <= first <= last < len(lines)):
            continue
        sections.append(Section(label=str(label).strip() or "Section", start=lines[first].start, end=lines[last].end))
    sections.sort(key=lambda s: s.start)
    # Make contiguous: each section runs until the next begins; pad the edges.
    for i, section in enumerate(sections):
        section.end = sections[i + 1].start if i + 1 < len(sections) else max(duration, section.end)
    if sections:
        sections[0].start = 0.0
    return [s for s in sections if s.end - s.start >= MIN_SECTION_SECONDS]


def analyze_with_llm(llm: LLMClient, lines: list[LyricLine], duration: float) -> list[Section]:
    numbered = "\n".join(
        f"{i:3d} [{_fmt_time(line.start)}-{_fmt_time(line.end)}] {line.text}"
        for i, line in enumerate(lines)
    )
    prompt = (
        f"Song duration: {_fmt_time(duration)}. Transcribed lyric lines "
        f"(index, time range, text):\n\n{numbered}\n\n"
        "Label the song's sections."
    )
    data = llm.generate_json(_SYSTEM_PROMPT, prompt, _SECTION_SCHEMA)
    ranges = [
        (item["label"], int(item["first_line"]), int(item["last_line"]))
        for item in data.get("sections", [])
    ]
    if not ranges:
        raise LLMError("LLM returned no sections")
    return _sections_from_line_ranges(ranges, lines, duration)


def _cluster_lines(lines: list[LyricLine]) -> list[int]:
    """Assign each line a cluster id; fuzzy-equal lines share a cluster.

    Whisper never transcribes a repeated chorus identically twice (especially
    sung/CJK material), so exact matching misses most repeats — fuzzy ratio
    matching against each cluster's first representative catches them.
    """
    representatives: list[str] = []
    assignments: list[int] = []
    for line in lines:
        norm = _normalize(line.text)
        cluster = None
        for ci, rep in enumerate(representatives):
            if difflib.SequenceMatcher(None, norm, rep).ratio() >= FUZZY_MATCH:
                cluster = ci
                break
        if cluster is None:
            representatives.append(norm)
            cluster = len(representatives) - 1
        assignments.append(cluster)
    return assignments


def analyze_heuristic(lines: list[LyricLine], duration: float) -> list[Section]:
    """No-LLM fallback: repeated lyric lines mark choruses, the rest are verses.

    Lines are fuzzy-clustered; a line whose cluster occurs 2+ times is
    'repeated'. Contiguous runs of repeated lines become Chorus; other runs
    become numbered Verses. Single-line runs are transcription noise and get
    absorbed into the preceding section.
    """
    if not lines:
        return []

    clusters = _cluster_lines(lines)
    counts = Counter(clusters)
    repeated = [counts[c] >= 2 for c in clusters]

    runs: list[list[int]] = [[0]]
    for i in range(1, len(lines)):
        gap = lines[i].start - lines[i - 1].end
        if repeated[i] != repeated[runs[-1][0]] or gap > 8.0:
            runs.append([i])
        else:
            runs[-1].append(i)
    merged: list[list[int]] = []
    for run in runs:
        if merged and len(run) < 2:
            merged[-1].extend(run)
        else:
            merged.append(run)

    ranges: list[tuple[str, int, int]] = []
    verse_number = 0
    for run in merged:
        if repeated[run[0]]:
            label = "Chorus"
        else:
            verse_number += 1
            label = f"Verse {verse_number}"
        ranges.append((label, run[0], run[-1]))

    return _sections_from_line_ranges(ranges, lines, duration)


def analyze_structure(
    lines: list[LyricLine],
    duration: float,
    llm: LLMClient | None = None,
) -> tuple[list[Section], str]:
    """Return (sections, method). Empty list when there is nothing to segment."""
    if len(lines) < 4:  # too little text to infer structure from
        return [], "none"
    if llm is not None:
        try:
            sections = analyze_with_llm(llm, lines, duration)
            if sections:
                return sections, f"llm:{llm.name}"
        except (LLMError, ValueError, KeyError, TypeError):
            pass  # fall through to the heuristic
    return analyze_heuristic(lines, duration), "heuristic"
