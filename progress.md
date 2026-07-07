# Progress / Handoff

_Last updated: 2026-07-07_

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
- Song structure via LLM with a repeated-line heuristic fallback — every LLM
  failure mode (bad key, unreachable endpoint, malformed response) degrades
  to the heuristic labeler rather than failing the job. As of this session
  the LLM side supports **12 providers** through a single registry — see
  "Just completed" below.
- Engraving strips spurious `natural` accidentals that music21 10.5.0
  attaches to white-key pitches built from raw MIDI ints (`src/mnc/score.py`,
  `build_score`'s note/chord loop): a genuine cancelling natural (e.g. F♯
  then F♮ in the same measure) still displays correctly.
- Web app (FastAPI + OSMD) and CLI (`mnc transcribe`, `mnc serve`).
- Project skills: `handoff`, `commit-push`, and `verify` (build/drive recipe
  for end-to-end checks) under `.claude/skills/`.

## Just completed: multi-provider LLM selector (this session)

**Why:** The LLM-assisted structure analysis only supported Anthropic and
OpenAI. The user wanted to pick from a broad set of providers — major
players plus low-cost/Chinese options — and supply whatever that provider
needs (API key, and for some, a model or custom endpoint) directly in the
web form.

**Design:** almost every non-Anthropic provider (Google, DeepSeek, Alibaba
Qwen, Moonshot Kimi, Zhipu GLM, xAI Grok, Groq, OpenRouter, local
Ollama/LM Studio, any custom endpoint) speaks the OpenAI chat-completions
API, so they all reuse the existing `OpenAIClient` with a provider-specific
`base_url` — **zero new dependencies** (`anthropic` + `openai` extras
unchanged). Only Anthropic keeps its native client. A single
`ProviderSpec` registry (`PROVIDERS` dict) in `src/mnc/llm.py` is the source
of truth for both backend client construction and the frontend dropdown, so
adding a 13th provider later is one registry entry, not a change in three
places.

**Plumbing, file by file:**

- `src/mnc/llm.py` — new `ProviderSpec` dataclass + `PROVIDERS` registry
  (12 entries, `group` = major/regional/local). `resolve_provider()` now
  validates against registry ids (was hardcoded to `anthropic`/`openai`) and
  auto-detects from any provider's `key_env` vars, not just
  `ANTHROPIC_API_KEY`/`OPENAI_API_KEY`. `get_llm_client()` gained a
  `base_url` param; it raises `LLMError` up front if a regional provider has
  no key, or if `custom` has no base URL (rather than silently falling
  through to `api.openai.com`). `OpenAIClient.generate_json` now retries
  once without `response_format={"type":"json_object"}` on a
  `BadRequestError`, since some compat servers reject that param — the
  existing tolerant `_extract_json` handles the plain-text response.
- `src/mnc/pipeline.py` — `Options.llm_base_url: Optional[str]` added and
  threaded into `get_llm_client(...)` at the structure-analysis call site.
- `src/mnc/web/app.py` — `create_job` gained `llm_model` / `llm_base_url`
  form fields (threaded into `Options`, never stored on `Job`); new
  `GET /api/providers` returns `[spec.public() for spec in PROVIDERS.values()]`
  (no secrets — id/label/group/default_model/base_url/key_prefix/docs_url/
  needs_key/editable_base_url).
- `src/mnc/web/static/index.html` — the old 2-option `<select>` is now
  populated at runtime; added `#llm-model` and `#llm-base-url-row` (hidden
  unless the provider is local/custom) plus an `#llm-docs-link`.
- `src/mnc/web/static/app.js` — `loadProviders()` fetches `/api/providers`
  and builds `<optgroup>`s (Major players / Low-cost & regional / Local &
  custom); `syncProviderFields()` updates the key placeholder, docs link,
  base-URL visibility, and model placeholder on provider change; submit
  handler appends `llm_model`/`llm_base_url` when non-empty.
- `src/mnc/cli.py` — added `--llm-api-key` and `--llm-base-url` flags
  (previously only `--llm`/`--llm-model` existed); `--llm` help text lists
  all registry ids.
- `README.md` — provider env-var list expanded (`DEEPSEEK_API_KEY`,
  `DASHSCOPE_API_KEY`, `MOONSHOT_API_KEY`, `ZHIPU_API_KEY`, `GROQ_API_KEY`,
  `OPENROUTER_API_KEY`, `GEMINI_API_KEY`, `XAI_API_KEY`) plus
  `MNC_LLM_BASE_URL` and the new CLI flags.

**Provider registry contents** (id — client — base_url):
`anthropic` (native), `openai` (SDK default), `google`
(`generativelanguage.googleapis.com/v1beta/openai/`), `xai` (`api.x.ai/v1`),
`deepseek` (`api.deepseek.com`), `qwen`
(`dashscope-intl.aliyuncs.com/compatible-mode/v1`), `moonshot`
(`api.moonshot.ai/v1`), `zhipu` (`open.bigmodel.cn/api/paas/v4`), `groq`
(`api.groq.com/openai/v1`), `openrouter` (`openrouter.ai/api/v1`), `local`
(`http://localhost:11434/v1`, `needs_key=False`, editable), `custom`
(no default base_url, editable, required).

**Verification done:**

- Full suite: `.venv/bin/python -m unittest discover tests` → **54/54 pass**
  (was 47; +7 new: `test_unknown_provider_raises` updated to use a name
  outside the registry since `google`/`gemini` is now valid;
  `test_registry_providers_resolve_to_themselves`,
  `test_env_key_auto_detects_regional_provider` added to
  `TestResolveProvider`; new `TestProviderRegistry` class with
  `test_every_spec_is_well_formed`,
  `test_get_llm_client_builds_openai_client_with_provider_base_url`,
  `test_get_llm_client_local_needs_no_key`,
  `test_get_llm_client_regional_without_key_raises`,
  `test_get_llm_client_custom_without_base_url_raises`).
- Live end-to-end job via `mnc serve --port 8765` + `curl` against
  `tests/twinkle.wav`: submitted with `llm_provider=deepseek` and a bogus
  key; job completed successfully (no crash) — though with this short
  synthetic fixture `structure_method` landed on `"none"` (too few lyric
  lines from the melody-onset mapping to trigger structure analysis at all,
  a fixture limitation, not a code issue).
- Directly exercised the real failure path with network access: called
  `get_llm_client("deepseek", api_key="sk-invalid-test-key-xxxx")` then
  `analyze_structure(...)` with 6 synthetic lyric lines — the invalid key
  hit the real `api.deepseek.com` endpoint, raised inside
  `analyze_with_llm`, and `analyze_structure` correctly fell back to
  `method="heuristic"`.
- Browser-driven UI check (`claude-in-chrome`): confirmed the "Transcription
  options" panel expands, the LLM provider `<select>` is populated with all
  12 options in the correct 3 optgroups (verified via accessibility tree),
  and via `javascript_tool` dispatching `change` events: selecting
  `deepseek` sets the key placeholder to `sk-...`, the model placeholder to
  `deepseek-chat`, and shows a "Get a key →" link to
  `platform.deepseek.com/api_keys` while keeping Base URL hidden; selecting
  `local` reveals Base URL prefilled with `http://localhost:11434/v1` and
  disables the API key field; selecting `custom` reveals an empty, enabled
  Base URL field and a required key field.
- Cleaned up: killed the dev server, removed the two test job directories
  under `~/.cache/music-notes-creator/jobs/`.

**Not yet verified:** a real, successful LLM call against one of the new
non-Anthropic/OpenAI providers with a *valid* key (all testing above used
invalid keys to exercise the fallback path, since no real keys for these
services are available in this environment). Recommended next step: if the
user has a real key for any of DeepSeek/Qwen/Moonshot/Zhipu/Groq/
OpenRouter/Gemini/xAI, run one job with it and confirm
`structure_method` reports `llm:openai` (the `OpenAIClient.name` is
`"openai"` regardless of which compat provider is behind it) and that
section labels look sensible. Also not verified: a real local server
(Ollama/LM Studio) end-to-end, since none was running in this environment.

**Not yet verified (carried over from prior session):** a real
(non-synthetic) song with genuine chromatic content, to eyeball that
legitimate sharps/flats and their in-measure cancelling naturals still
render correctly in MuseScore/OSMD. Also carried over: the *aligned-to-vocals*
lyrics path on real sung audio (`tests/twinkle.wav` is instrumental).

## Possible follow-ups

- Real-key smoke test against at least one new provider (see above) —
  highest priority since it's the only unverified part of this session's
  work.
- Real-track accidental sanity check (carried over, see above).
- Aligned-to-vocals lyrics on a real vocal track (carried over, see above).
- Syllabification: split multi-syllable words across tied/melisma notes
  (currently one word per note).
- Show a warning in the UI when the melody-note lyric fallback fired (timing
  is approximate there).
- Expose the anchor-fraction threshold as an option if real-world tracks
  fall back too eagerly.
- Consider trimming the provider list if 12 options feels like too much
  choice in the dropdown — grouping via `<optgroup>` already mitigates this,
  but it's worth watching real usage.

## Environment notes

- uv-managed Python 3.11 venv at `.venv`; `setuptools<81` pin (see memory).
- Tests: `.venv/bin/python -m unittest discover tests` (54 tests).
- music21 version in this venv: 10.5.0 (accidental-stripping behavior in
  `score.py` is version-specific; re-check if music21 is upgraded).
- openai SDK version in this venv: 2.44.0 (`openai.BadRequestError` used in
  `llm.py`'s retry-without-`response_format` path; confirmed present at
  this version).
- Generated test audio: `.venv/bin/python tests/make_test_audio.py` →
  `tests/twinkle.wav` (gitignored). Generated score fixtures under
  `tests/out/` are also gitignored.
- No real API keys for any LLM provider are configured in this environment;
  all provider-specific testing this session used deliberately invalid keys
  to exercise error/fallback paths, plus unit-level assertions on client
  construction (model, base_url) without making real completion calls.
