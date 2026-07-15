# Progress / Handoff

_Last updated: 2026-07-15_

## Project state

Music Notes Creator transcribes audio/video/YouTube into a two-staff piano
score (MusicXML + MIDI) with lyrics under the melody, verse/chorus section
labels, and chord symbols. Working features as of now:

- Note transcription (Basic Pitch), tempo estimation (with octave-error
  correction), key detection, quantization, treble/bass split, repeat
  collapsing ("Chorus: play mm. X–Y").
- **Chord symbol detection**: per-measure template matching over the
  quantized note grid, engraved as music21 `ChordSymbol`s in the treble
  part and surfaced in the web UI as a count alongside lyrics/section info.
  See "Prior session (2026-07-12)" below for the detection design.
- Lyrics via faster-whisper (no VAD — it rejects singing; per-segment
  hallucination filters instead), or **user-provided lyrics text** aligned
  onto Whisper's timing (`align_user_lyrics`) or, on alignment failure,
  mapped one word per melody-note onset (`lyrics_from_onsets`), both in
  `src/mnc/lyrics.py`. `strip_lyric_tags` filters structural annotations
  ("[Verse 1]", "{x2}", full-width CJK section brackets/words) out of pasted
  lyrics before tokenizing.
- Song structure via LLM with a repeated-line heuristic fallback — every LLM
  failure mode (bad key, unreachable endpoint, malformed response) degrades
  to the heuristic labeler rather than failing the job. LLM structure
  analysis supports **12 providers** through a single registry
  (`PROVIDERS` in `src/mnc/llm.py`): Anthropic native, plus 11 OpenAI-compat
  providers (OpenAI, Google/Gemini, xAI, DeepSeek, Qwen, Moonshot, Zhipu,
  Groq, OpenRouter, local Ollama/LM Studio, and any custom endpoint) that
  all reuse `OpenAIClient` with a provider-specific `base_url`.
- Engraving strips spurious `natural` accidentals that music21 10.5.0
  attaches to white-key pitches built from raw MIDI ints (`src/mnc/score.py`,
  `build_score`'s note/chord loop): a genuine cancelling natural (e.g. F♯
  then F♮ in the same measure) still displays correctly.
- Web app (FastAPI + OSMD) and CLI (`mnc transcribe`, `mnc serve`).
- Project skills: `handoff`, `commit-push`, `verify` (build/drive recipe for
  end-to-end checks), and `download-video` under `.claude/skills/`.
- **First real-song end-to-end regression test**: `tests/test_e2e_surrender.py`
  transcribes a real 330s track and checks the score against a hand-read
  reference sheet — see below.

## Prior session (2026-07-12): tempo octave fix, chord symbols, e2e test

**Why:** The user pointed at a concrete quality gap — transcribing
`【降服 ⧸ Surrender】…男key.mp3` (a 330s ballad) produced a score that didn't
match a professional reference arrangement (`surrender_notes.jpeg`, provided
by the user, untracked). Reading the reference sheet by hand: **A major
(3♯), 4/4, ♩=64**, chord symbols (D, A, Bm7, E7, F♯m7, Esus4, E, F♯m, Dmaj7,
C♯m7, A11), grand staff, no lyrics. The baseline pipeline run (pre-fix) got
the key right (A major) but the tempo was **129 BPM — an exact octave
double** of the true 64, and there was no chord-symbol feature at all.
Note-for-note matching of a professional arrangement isn't achievable by
automatic transcription, so the fix targets the two things that *are*
fixable — tempo and chord symbols — and the new test asserts musical
properties derived from the sheet rather than exact notes (confirmed scope
with the user via `AskUserQuestion` before implementing).

**Design — tempo octave fix** (`src/mnc/transcribe.py`): librosa's
`beat.beat_track` frequently locks onto a harmonic/subharmonic of the true
beat. Measured on the real track: `beat_track` returns 129.2 (2× true 64.6);
a Fourier tempogram (`librosa.feature.fourier_tempogram`) shows the
strength at 64.6 BPM is 0.63× the strength at 129.2 — a strong subharmonic
signal. On the synthetic 100 BPM `tests/twinkle.wav` fixture, that ratio is
only 0.20 — no octave error, `beat_track`'s 99.4 is trusted as-is. The fix
folds into the existing 65–190 BPM playable band **before** applying the
octave decision (critical ordering: halving only fires at ≥100 BPM, so the
result is never re-folded upward and the old `while bpm < 65: bpm *= 2`
can't undo the halving).

**Design — chord symbols** (`src/mnc/chords.py`, new file, pure logic, no
music21/librosa imports): per 4-beat measure, build a duration-weighted
pitch-class histogram (notes spanning a barline split their weight across
both measures) plus the bass pitch class. Score each of 12 roots × 6
qualities (maj/min/7/maj7/m7/sus4) with a **size-normalized** match:
`(present − outside) / sqrt(template_size)`, plus a bass-note bonus when the
bass matches the candidate root. The normalization matters — an earlier
version used a flat linear penalty (`present − MISS_PENALTY·outside`) which
had a real bug: it always preferred the larger template (e.g. a 7th chord
over its triad) whenever *any* nonzero weight fell outside the smaller
template, even a single stray transcription artifact. The sqrt-normalized
version requires the extra chord tone to carry real weight (empirically,
>~15% of the triad's weight) before upgrading to a seventh — caught by a
dedicated unit test (`test_weak_incidental_tone_does_not_flip_to_seventh`).
`detect_chords()` emits one symbol per harmonic **change** (not one per
measure); low-confidence measures are skipped without resetting the
currently-active chord.

**Plumbing, file by file:**

- `src/mnc/transcribe.py` — `estimate_tempo` now delegates to
  `_estimate_tempo_from_signal(y, sr)`; new `_fourier_tempo_strength(...)`
  returns a `bpm -> strength` callable; new `_resolve_tempo_octave(bpm,
  strength, min_bpm=100, ratio=0.4)` is the pure, unit-testable decision
  function (takes any `Callable[[float], float]`, so tests use dict-backed
  fakes instead of real audio). New constants: `TEMPO_FOLD_MIN/MAX`,
  `HALVING_MIN_BPM=100.0`, `SUBHARMONIC_RATIO=0.4`.
- `src/mnc/chords.py` (new) — `QUALITIES` template dict, `_measure_weights`,
  `_best_chord`, `detect_chords` -> `list[DetectedChord(offset_beats,
  root_pc, quality)]`, `chord_figure(root_pc, quality, prefer_sharps)` for
  music21 figure strings (`"C#m7"` sharps / `"D-m7"` flats — music21 uses
  `-` not `b`).
- `src/mnc/score.py` — `build_score(..., chords: bool = True)`: inserts
  `harmony.ChordSymbol` objects between key-signature selection and
  `StaffGroup` construction (spelling needs `signature.sharps`; inserted
  *after* `score.analyze("key")` so zero-duration chord symbols can't
  pollute key detection — verified empirically, not just in theory).
  `export_score(..., chords: bool = True)` counts symbols into new
  `ScoreInfo.n_chord_symbols: int = 0`, and calls new
  `_strip_chord_symbols(score)` **between** the MusicXML write and the MIDI
  write — required because music21's MIDI exporter does *not* skip Harmony
  objects (`ChordSymbol` is a `Chord` subclass) the way MusicXML export
  does; left in place they'd add audible blips to the `.mid`. Verified: a
  6-note test score with 2 chord symbols produces exactly 6 `note_on`
  events in the exported MIDI, not 6 + chord-symbol pitches.
- `src/mnc/pipeline.py` — `Options.chords: bool = True`, passed through to
  `export_score`.
- `src/mnc/cli.py` — new `--no-chords` flag; summary print adds
  `chords: N symbols` when nonzero.
- `src/mnc/web/app.py` — `Job.n_chord_symbols: int = 0` added and populated
  from `info.n_chord_symbols` in `_run_job` (backend plumbing only in this
  session — the frontend display was added in the 2026-07-15 session below).
- `tests/test_tempo.py` (new, 8 tests) — `TestResolveTempoOctave` (pure,
  dict-backed strength fakes covering the real measured ratios, the
  min-bpm gate, the exact-ratio boundary, and a divide-by-zero guard) +
  `TestEstimateTempoTwinkle` (real audio, skipped if `twinkle.wav` absent).
- `tests/test_chords.py` (new, 24 tests) — `TestBestChord` (major/minor
  triads, sus4, maj7/m7 spelling, the weak-vs-strong-seventh distinction,
  bass-bonus not overriding clear pitch content, chromatic mush -> `None`),
  `TestMeasureWeights` (barline-spanning notes, bass-pc-is-lowest-not-first),
  `TestDetectChords` (change-only emission, low-confidence measures don't
  reset the run), `TestChordFigure`, `TestBuildScoreChordIntegration`
  (chords on/off, no key pollution).
- `tests/test_e2e_surrender.py` (new, 7 tests) — `skipUnless` the mp3
  exists (both reference files are untracked, local-only material — CI and
  other machines skip this class automatically). `setUpClass` runs the full
  pipeline once (`Options(lyrics=False, structure=False)` — reference sheet
  has no lyrics, so this also avoids a Whisper pass) and parses the
  resulting MusicXML with `music21.converter`. Assertions, all
  jpeg-derived: key = "A major" / 3 sharps; tempo in [58, 72] BPM (catches
  the old 129 BPM regression); 4/4; measure count in [70, 106] (~89
  measures at 65 BPM for a 330s song — also independently catches a tempo
  regression, since 129 BPM would roughly double it to ~176); 2 parts with
  correct clefs; chord symbols present (≥ `n_measures // 8`), ≥60% diatonic
  to A major, and root-set overlap ≥3 with the reference sheet's chord
  roots {A, D, E, B, F♯, C♯}; output files exist and are substantial.
  Thresholds were set from one calibration run with generous margin (see
  Verification below), not tightened to the exact observed numbers.

**Verification done:**

- Full suite: `.venv/bin/python -m unittest discover tests` → **93/93
  pass** (was 54; +8 tempo, +24 chords, +7 e2e = 39 new tests; zero
  regressions in the pre-existing 54).
- Isolated tempo check: `estimate_tempo()` on the real mp3 (via `to_wav` +
  direct call) returns **64.6 BPM** (truth: 64); on `tests/twinkle.wav`
  returns **99.4 BPM** (truth: 100) — confirmed the fix doesn't regress the
  existing synthetic-fixture behavior the `verify` skill depends on.
  Directly probed the raw `librosa` tempo APIs and the Fourier-tempogram
  strength ratios on both files before implementing, to ground the
  threshold choices in real numbers rather than guesses.
  ```
  surrender: beat_track=129.2  strength(64.6)/strength(129.2)=0.63
  twinkle:   beat_track=99.4   strength(49.7)/strength(99.4)=0.20
  ```
- Chord scoring bug caught and fixed *during* this session: wrote the
  linear-penalty version first, hand-verified it with synthetic weight
  dicts, found it always preferred supersets (7th over triad) on any
  nonzero stray weight, redesigned with sqrt-normalization, re-verified all
  the same synthetic cases plus the fix-specific case
  (`test_weak_incidental_tone_does_not_flip_to_seventh`).
- Full calibration run on the real mp3 (`mnc transcribe ... --no-lyrics
  --no-structure`) after all fixes landed:
  **tempo 65 BPM, key A major, 2605 notes, 89 measures, 75 chord
  symbols, 98.7% diatonic chord roots, 6/6 root overlap with the reference
  sheet's chord vocabulary** ({A,D,E,B,F♯,C♯} all present). All e2e
  thresholds were set below/above these measured numbers with real margin
  (e.g. measure-count band [70,106] around an observed 89; chord-count
  floor `n//8`≈11 against an observed 75).
- `build_score`-level integration checks (synthetic 2-measure C→Am example):
  `chords=True` adds ≥1 `ChordSymbol`; `chords=False` adds zero; detected
  key identical either way; MusicXML `<harmony>` count matches
  `n_chord_symbols`; MIDI `note_on` count unaffected by chord symbols
  (stripped correctly before the MIDI write).
- Confirmed via reading the installed music21 10.5.0 source
  (`harmony.py`, `m21ToXml.py`, `midi/translate.py`) rather than assuming:
  `ChordSymbol.__init__` defaults to zero duration and `writeAsChord=False`;
  MusicXML export ignores the realized pitches for `writeAsChord=False`
  symbols (emits `<harmony>` instead); **MIDI export does not check
  `writeAsChord`** and would emit the chord's pitches as notes if not
  stripped first — this was the reason `_strip_chord_symbols` exists.

**Not yet verified:**

- **Generalization to other songs**: all chord-detection thresholds
  (`CONF_MIN=0.30`, `MIN_WEIGHT=1.0`, `BASS_BONUS=0.25`) were calibrated
  against this one real track. Not verified against a second real-world
  song with different instrumentation/density. If chord output looks off
  on other tracks, recalibrate `CONF_MIN` first (raise it if too many
  spurious low-confidence chords appear; lower it if legitimate chords are
  being dropped).
- Carried over from prior sessions, still open: a second real (non-Surrender)
  chromatic track to further sanity-check the natural-accidental stripping
  in MuseScore/OSMD; the *aligned-to-vocals* lyrics path on real sung audio
  (only tested on the instrumental `twinkle.wav` and, this session, with
  `lyrics=False` on Surrender); a real-key smoke test against one of the
  11 non-Anthropic LLM providers with a *valid* key (all provider testing
  to date used deliberately invalid keys to exercise fallback paths).

## Just completed: chord-symbol/lyric-fallback UI surfacing + verification (2026-07-15)

**Why:** The prior session's handoff flagged the OSMD chord-symbol rendering
as never visually confirmed (highest-priority open item), the `--no-chords`
CLI flag as never invoked end-to-end, and `n_chord_symbols` as backend-only
with no UI surface. The user picked "small UI polish" + "verification pass"
from the follow-up list (deferred: syllabification, exposing the
anchor-fraction threshold).

**Changes (`src/mnc/web/static/`):**
- `app.js` — `renderResult()`: appends `"N chord symbols"` to the meta line
  when `job.n_chord_symbols` is nonzero; shows a new `#result-warning`
  element when `job.lyrics_source === "mapped to melody notes"` (the
  onset-fallback path, set in `pipeline.py`), explaining that timing is
  approximate because vocal alignment failed.
- `index.html` — added `<p id="result-warning" class="warning hidden">`
  next to the existing `#result-structure` line.
- `style.css` — new `.warning` rule (amber `#8a5a00`, `⚠` prefix via
  `::before`), sibling to the existing `.hint` rule.
- No backend changes — `n_chord_symbols` and `lyrics_source` were already in
  `Job.public()` (`src/mnc/web/app.py`).

**Verification done:**
- Full suite: `.venv/bin/python -m unittest discover tests` → 93/93, no
  regressions (frontend-only change).
- `--no-chords` end-to-end: `mnc transcribe tests/twinkle.wav --no-chords
  --no-lyrics --no-structure` → summary omits the `chords:` line; output
  MusicXML has 0 `<harmony>` elements. Control run without the flag →
  `chords: 4 symbols` printed, 4 `<harmony>` elements in the XML.
- Browser pass (`mnc serve --port 8765`, driven via claude-in-chrome per the
  `verify` skill's DataTransfer recipe): submitted `twinkle.wav` with pasted
  lyrics and `llm=false`, which deterministically lands on
  `lyrics_source: "mapped to melody notes"`. Single screenshot confirmed all
  three things at once: "4 chord symbols" in the meta line, the amber
  warning banner, and OSMD rendering "C" / "Dsus4" / "Dm7" chord symbols
  above the treble staff.
- **Surrender OSMD chord rendering** (the original unverified item): loaded
  the already-generated `tests/out/e2e_surrender/*.musicxml` (75 `<harmony>`
  elements, all 7 root pitch classes present) directly through OSMD in a
  fresh tab (bypassing a 330 s re-transcription). Confirmed Dmaj7, F♯m7,
  Bm7, A, Amaj7, Asus4 render correctly above the treble staff at ♩=65,
  A major (3♯), grand staff — matches the reference sheet's key/tempo and
  overlaps its chord vocabulary.
- Cleaned up: temp files copied into `static/` for the browser test, and the
  test job directory under `~/.cache/music-notes-creator/jobs/`, were
  removed after verification; the dev server was stopped.

**Not yet verified:**

- **Warning-hidden path**: only screenshot-tested the case where
  `#result-warning` becomes visible (`lyrics_source: "mapped to melody
  notes"`). Never submitted a job with `lyrics_source: "aligned to vocals"`
  or `"transcribed"` through the actual UI to confirm the element stays
  hidden/empty in the normal case — the hide branch is a two-line mirror of
  the existing `#result-structure` pattern, so low risk, but unobserved.
- **Zero-chord-count path in the browser**: `--no-chords` was verified via
  the CLI (see above) but never submitted as a browser job, so the meta line
  omitting "N chord symbols" when `n_chord_symbols` is 0 was checked by code
  inspection (`if (job.n_chord_symbols)`) rather than a live screenshot.
- **Surrender chord rendering bypassed the job pipeline**: the 75-chord
  visual check loaded `tests/out/e2e_surrender/*.musicxml` directly into a
  fresh OSMD instance via `javascript_tool`, not through a real `/api/jobs`
  submission + `renderResult()` — so the meta-line chord count and warning
  banner were never seen together on the real Surrender track in the actual
  UI (would require a ~330s live transcription job to close this gap).

## Possible follow-ups

- Second real-song calibration pass for the chord-detection thresholds.
- Real-key smoke test against a non-Anthropic LLM provider (carried over).
- Real-track accidental sanity check on a second chromatic song (carried
  over).
- Aligned-to-vocals lyrics on a real vocal track (carried over).
- Syllabification: split multi-syllable words across tied/melisma notes
  (currently one word per note).
- Expose the anchor-fraction threshold as an option if real-world tracks
  fall back too eagerly.

## Environment notes

- uv-managed Python 3.11 venv at `.venv`; `setuptools<81` pin (see memory).
- Tests: `.venv/bin/python -m unittest discover tests` (93 tests). The new
  `tests/test_e2e_surrender.py` class auto-skips unless the untracked
  reference mp3 (`【降服 ⧸ Surrender】歌詞MV - 約書亞樂團 ft. ZEcho｜男key.mp3`)
  is present at the repo root — it and `surrender_notes.jpeg` are
  user-provided local reference material, not committed.
- music21 version in this venv: 10.5.0 (accidental-stripping *and* the new
  chord-symbol MIDI-strip behavior in `score.py` are version-specific;
  re-check both if music21 is upgraded, especially whether a future version
  makes MIDI export respect `writeAsChord`).
- openai SDK version in this venv: 2.44.0.
- Generated test audio: `.venv/bin/python tests/make_test_audio.py` →
  `tests/twinkle.wav` (gitignored, 100 BPM). Generated score fixtures under
  `tests/out/` are gitignored, including the new `tests/out/e2e_surrender/`.
- No real API keys for any LLM provider are configured in this environment.
