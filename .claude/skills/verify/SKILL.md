---
name: verify
description: Build/launch/drive recipe for verifying Music Notes Creator changes end-to-end (web app + pipeline).
---

# Verifying Music Notes Creator

## Launch

```bash
.venv/bin/mnc serve --port 8765   # web app (background it); FastAPI + static UI
```

Unit tests (pure logic only, no inference): `.venv/bin/python -m unittest discover tests`

## Test audio

`tests/twinkle.wav` (regenerate with `.venv/bin/python tests/make_test_audio.py`).
~10 s synthetic two-hand piano; melody notes are all >= C4 so user lyrics map
onto them via the onset fallback. Whisper finds no vocals in it — user-lyrics
jobs land on `lyrics_source: "mapped to melody notes"`.

## Drive the API

```bash
curl -s :8765/api/jobs -F "file=@tests/twinkle.wav" -F "lyrics_text=<lyrics.txt" \
     -F "llm=false" -F "title=Test"          # -> {"id": ...}
curl -s :8765/api/jobs/<id>                  # poll; done in ~15 s (models cached)
curl -s :8765/api/jobs/<id>/musicxml         # lyric words live in <text> elements
```

Useful assertions: `n_lyric_words`, `structure_method` (`heuristic` vs
`llm:<provider>`), `lyrics_source`; grep the MusicXML `<text>` elements for
lyric content. LLM failures (bad key/provider) must degrade to
`structure_method: "heuristic"`, never a failed job.

## Drive the browser UI

File inputs can't be filled by path; temporarily `cp` the wav into
`src/mnc/web/static/`, then set it via JS:

```js
const blob = await fetch('/twinkle-test.wav').then(r => r.blob());
const dt = new DataTransfer();
dt.items.add(new File([blob], 'twinkle.wav', {type: 'audio/wav'}));
document.querySelector('#file').files = dt.files;
```

Remove the copied wav afterwards. Job artifacts accumulate in
`~/.cache/music-notes-creator/jobs/<id>` — delete test ones when done.

## Gotchas

- The options `<details>` panel is collapsed by default; click the summary
  before looking for option fields in the accessibility tree.
- New inputs need their `type` listed in the shared selector in
  `style.css` (`input[type="url"], input[type="text"], ...`) or they render
  unstyled.
