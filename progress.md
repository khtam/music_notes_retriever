# Progress / Handoff

_Last updated: 2026-07-06_

## Project state

Music Notes Creator transcribes audio/video/YouTube into a two-staff piano
score (MusicXML + MIDI) with lyrics under the melody and verse/chorus section
labels. Working features as of now:

- Note transcription (Basic Pitch), tempo/key estimation, quantization,
  treble/bass split, repeat collapsing ("Chorus: play mm. X–Y").
- Lyrics via faster-whisper (no VAD — it rejects singing; per-segment
  hallucination filters instead), or **user-provided lyrics text** aligned
  onto Whisper's timing (`align_user_lyrics`) or, on alignment failure,
  mapped one word per melody-note onset (`lyrics_from_onsets`), both in
  `src/mnc/lyrics.py`. `strip_lyric_tags` filters structural annotations
  ("[Verse 1]", "{x2}", full-width CJK section brackets/words) out of pasted
  lyrics before tokenizing.
- Song structure via LLM (Anthropic/OpenAI-compatible) with a
  repeated-line heuristic fallback; the web UI now exposes a live on/off
  toggle, provider choice, and per-request API key (`src/mnc/llm.py`,
  `web/static/app.js`) — every LLM failure mode degrades to the heuristic
  labeler rather than failing the job.
- Web app (FastAPI + OSMD) and CLI (`mnc transcribe`, `mnc serve`).
- Engraving no longer prints spurious natural (♮) signs on plain diatonic
  notes — see "Just completed" below.
- Project skills: `handoff`, `commit-push`, and `verify` (build/drive recipe
  for end-to-end checks) under `.claude/skills/`.

## Just completed: fix spurious natural accidentals (this session)

**Why:** Generated scores showed natural signs on notes that don't need
them. The committed test fixture proved it: "Twinkle Twinkle" is pure
C-major (every pitch `<alter>0</alter>`, i.e. all white keys), yet the
exported MusicXML carried ~20 `<accidental>natural</accidental>` tags.

**Root cause (music21 10.5.0):** `note.Note(midi_int)` / `chord.Chord([...])`
attach an *explicit* `natural` accidental object to every white-key pitch
built from a raw MIDI integer (black keys correctly get `sharp` instead).
`score.makeNotation()` then displays several of those as cautionary/
octave-displacement naturals. Confirmed via a standalone music21 repro: a
7-chord C-major bar produced 14 raw accidental objects for 14 pitches, and
tweaking `makeAccidentals`'s cautionary-flag kwargs did **not** remove them
— the naturals originate at pitch construction, not from the notation pass.

**Fix** (`src/mnc/score.py`, in `build_score`'s note/chord-construction
loop, ~line 255): after building each `note.Note`/`chord.Chord` element and
before inserting it into the part, strip any `natural` accidental from its
pitches (`element.pitches`), leaving sharps/flats untouched:

```python
for p in element.pitches:
    if p.accidental is not None and p.accidental.name == "natural":
        p.accidental = None
```

`makeAccidentals` (invoked inside `makeNotation`) then re-derives display
status from scratch, showing a natural only when it genuinely cancels a
prior sharp/flat within the same measure.

**Plumbing:** single change in `src/mnc/score.py`. No API or call-site
changes elsewhere.

**Verification done:**

- Standalone music21 repro (not committed) confirmed the mechanism: pure
  C-major bar went from 6 displayed naturals → 0 after stripping; an F♯
  followed by F♮ in the same bar still correctly showed sharp-then-natural;
  octave jumps (C4→C5) showed no spurious natural.
- Added `TestAccidentals` to `tests/test_units.py` (2 tests, using
  `build_score` directly with hand-built `NoteEvent`s, no inference
  needed): `test_diatonic_melody_has_no_naturals` (asserts zero displayed
  accidentals on an all-white-key melody spanning octaves) and
  `test_necessary_natural_still_shown` (F♯4 → F4 → C4 → C5 in one measure;
  asserts exactly one `sharp` and one `natural` are displayed).
- Full suite: `.venv/bin/python -m unittest discover tests` → 47/47 pass
  (was 45; +2 new).
- End-to-end: regenerated `tests/out/Twinkle_Test.musicxml` via
  `.venv/bin/mnc transcribe tests/twinkle.wav --no-lyrics --output-dir
  tests/out --title "Twinkle Test"` (real Basic Pitch inference, not just
  the unit test). `<accidental>natural</accidental>` count went from 20 → 0.
  `tests/out/` is gitignored, so this fixture is a local artifact only, not
  committed.

**Not yet verified:** a real (non-synthetic) song with genuine chromatic
content, to eyeball that legitimate sharps/flats and their in-measure
cancelling naturals still render correctly in MuseScore/OSMD (the F♯→F♮
unit test covers the logic but not visual engraving). Recommended next
step: run one real track with accidentals through `mnc serve` or the CLI
and open the MusicXML in a viewer.

**Not yet verified (carried over):** the *aligned-to-vocals* lyrics path on
real sung audio — `tests/twinkle.wav` is instrumental, so only unit tests
cover word alignment. Recommended next step: run one real vocal track with
pasted lyrics and eyeball word placement.

## Possible follow-ups

- Real-track accidental sanity check (see above) — highest priority since
  it's the only unverified part of this session's fix.
- Syllabification: split multi-syllable words across tied/melisma notes
  (currently one word per note).
- Show a warning in the UI when the melody-note fallback fired (timing is
  approximate there).
- Expose the anchor-fraction threshold as an option if real-world tracks
  fall back too eagerly.

## Environment notes

- uv-managed Python 3.11 venv at `.venv`; `setuptools<81` pin (see memory).
- Tests: `.venv/bin/python -m unittest discover tests` (47 tests).
- music21 version in this venv: 10.5.0 (accidental behavior above is
  version-specific; re-check if music21 is upgraded).
- Generated test audio: `.venv/bin/python tests/make_test_audio.py` →
  `tests/twinkle.wav` (gitignored). Generated score fixtures under
  `tests/out/` are also gitignored.
