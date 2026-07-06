# Progress / Handoff

_Last updated: 2026-07-05_

## Project state

Music Notes Creator transcribes audio/video/YouTube into a two-staff piano
score (MusicXML + MIDI) with lyrics under the melody and verse/chorus section
labels. Working features as of now:

- Note transcription (Basic Pitch), tempo/key estimation, quantization,
  treble/bass split, repeat collapsing ("Chorus: play mm. X–Y").
- Lyrics via faster-whisper (no VAD — it rejects singing; per-segment
  hallucination filters instead).
- Song structure via LLM (Anthropic/OpenAI-compatible) with a
  repeated-line heuristic fallback.
- Web app (FastAPI + OSMD) and CLI (`mnc transcribe`, `mnc serve`).

## Just completed: user-provided lyrics (this session)

**Why:** Whisper's lyric identification is often inaccurate on sung material.
Users can now supply the real lyrics as text; the pipeline maps them onto the
score in place of Whisper's words.

**Design** (core logic in `src/mnc/lyrics.py`):

1. `align_user_lyrics(text, reference)` — Whisper still runs, but only as a
   *timing reference*. User words are aligned to Whisper's timed words with
   `difflib.SequenceMatcher`; matched words (anchors) inherit timestamps,
   words between anchors interpolate linearly by word index, words outside
   the outermost anchors are spaced 0.4 s apart. Returns `None` when fewer
   than `max(3, 20%)` of the user's words match (timing untrustworthy).
2. `lyrics_from_onsets(text, onsets)` — fallback when alignment fails or
   Whisper errors/finds nothing: word *i* lands on the *i*-th melody-note
   onset (notes ≥ split point); overflow words are dropped.
3. CJK handling: `_split_token` splits CJK runs into single characters (one
   sung syllable each) on **both** sides of the alignment, so spaceless
   scripts still match.
4. User line breaks define `LyricLine`s → cleaner input for structure
   analysis than Whisper's segmentation.

**Plumbing:**

- `pipeline.py` — `Options.lyrics_text`; providing text implies lyrics on
  even with `--no-lyrics`; tracks `lyrics_source` ("transcribed" /
  "aligned to vocals" / "mapped to melody notes").
- `score.py` — `ScoreInfo.lyrics_source`, threaded through `export_score`.
- `cli.py` — `--lyrics-file PATH` (reads UTF-8, errors on missing/empty);
  result line prints source, e.g. `lyrics: 10 words (aligned to vocals)`.
- `web/app.py` — `lyrics_text` form field on `POST /api/jobs`; `Job` exposes
  `lyrics_source`.
- `web/static/` — "Lyrics (optional, recommended)" textarea above the
  options; `app.js` submits it and shows the source in the result meta;
  textarea styling in `style.css`.
- `README.md` — feature documented (How it works §3 + CLI flags).

**Verification done:**

- 30/30 unit tests pass (`.venv/bin/python -m unittest discover tests`),
  including new `TestParseLyricLines`, `TestAlignUserLyrics`,
  `TestLyricsFromOnsets`.
- End-to-end CLI run on `tests/twinkle.wav` (synthetic instrumental) with a
  lyrics file → 10 words mapped to melody notes, words in order in the
  MusicXML.
- End-to-end web API run via FastAPI `TestClient` (upload + `lyrics_text`)
  → job completes, MusicXML downloads.

**Not yet verified:** the *aligned-to-vocals* path on real sung audio — the
test WAV is instrumental, so only unit tests cover alignment. Recommended
next step: run one real vocal track with pasted lyrics and eyeball the
word placement.

## Possible follow-ups

- Syllabification: split multi-syllable words across tied/melisma notes
  (currently one word per note).
- Show a warning in the UI when the melody-note fallback fired (timing is
  approximate there).
- Expose the anchor-fraction threshold as an option if real-world tracks
  fall back too eagerly.

## Environment notes

- uv-managed Python 3.11 venv at `.venv`; `setuptools<81` pin (see memory).
- Tests: `.venv/bin/python -m unittest discover tests`.
- Generated test audio: `.venv/bin/python tests/make_test_audio.py` →
  `tests/twinkle.wav` (gitignored).
