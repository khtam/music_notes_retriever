# Music Notes Creator

Turn songs into piano sheet music. Give it an audio file, a video file, or a
YouTube URL and it produces a two-staff piano score (treble + bass) that
musicians can read and play — with the lyrics printed under the melody and
the song's structure (verse/chorus/bridge) labeled — along with MusicXML and
MIDI downloads.

## How it works

1. **Audio extraction** — local audio is used as-is; video files have their
   audio track extracted with ffmpeg; YouTube URLs are fetched with `yt-dlp`.
2. **Transcription** — [Spotify's Basic Pitch](https://github.com/spotify/basic-pitch)
   neural network converts the audio into note events (pitch, onset, duration).
3. **Lyrics** — [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
   transcribes the vocals with word-level timestamps; each word is attached
   to the nearest melody note so it prints under the treble staff.
4. **Song structure** — the timed lyric lines are segmented into sections
   (Verse 1, Chorus, Bridge, ...). If an LLM is configured (Anthropic or any
   OpenAI-compatible provider) it does the labeling; otherwise a repeated-line
   heuristic takes over. Sections become rehearsal marks on the score, and a
   later section whose label and notes match an earlier one is not engraved
   twice — it collapses into a "Chorus: play mm. 5–12" direction.
5. **Score generation** — tempo and key are estimated, notes are quantized to
   a beat grid, split across treble/bass staves around middle C, and engraved
   into a piano grand staff with `music21`.
6. **Rendering** — the web UI renders the resulting MusicXML in your browser
   using OpenSheetMusicDisplay; you can also download MusicXML (for
   MuseScore/Finale/Sibelius) and MIDI.

## Setup

Requires Python 3.10–3.11 (ffmpeg is bundled via `imageio-ffmpeg`, no system
install needed).

```bash
git clone https://github.com/khtam/music_notes_retriever.git
cd music_notes_retriever

# with uv (recommended; installs Python 3.11 if you don't have it)
uv python install 3.11
uv venv --python 3.11 .venv
uv pip install -e .            # or -e '.[llm]' for LLM structure analysis

# or with plain pip (needs an existing Python 3.10/3.11)
python3.11 -m venv .venv
.venv/bin/pip install -e .     # or -e '.[llm]'
```

### Optional: LLM-powered structure analysis

Everything works without an LLM (a lyric-repetition heuristic labels the
sections), but a language model does a noticeably better job of telling
verses from choruses. Install the `[llm]` extra and set one of:

```bash
export ANTHROPIC_API_KEY=...           # uses Claude (claude-opus-4-8)
# or
export OPENAI_API_KEY=...              # uses OpenAI (gpt-4o-mini)
# or any OpenAI-compatible server (Ollama, vLLM, LM Studio, ...):
export OPENAI_BASE_URL=http://localhost:11434/v1
export MNC_LLM_MODEL=llama3.1          # model name on that server
```

The provider is auto-detected from whichever key is present; override with
`MNC_LLM_PROVIDER=anthropic|openai|none` and `MNC_LLM_MODEL=<model>`, or the
`--llm` / `--llm-model` CLI flags. With nothing configured, the heuristic
fallback is used automatically.

## Usage

### Web app

```bash
.venv/bin/mnc serve            # then open http://127.0.0.1:8000
```

Paste a YouTube URL or upload an audio/video file, wait for transcription,
and the piano score appears on the page.

### Command line

```bash
.venv/bin/mnc transcribe song.mp3                  # audio file
.venv/bin/mnc transcribe video.mp4                 # video file
.venv/bin/mnc transcribe "https://youtu.be/..."    # YouTube
```

Outputs `<name>.musicxml` and `<name>.mid` next to the input (or into
`--output-dir`). Useful flags:

- `--split-point C4` — pitch that divides right hand (treble) from left hand (bass)
- `--tempo 120` — override the estimated tempo (BPM)
- `--min-note-length 120` — drop notes shorter than this many milliseconds
- `--title "My Song"` — title printed on the score
- `--no-lyrics` / `--no-structure` / `--no-dedup` — turn off lyric
  transcription, section labeling, or repeat collapsing
- `--whisper-model medium` — larger Whisper model for better lyrics (default: small)
- `--llm anthropic|openai|none`, `--llm-model <name>` — structure-analysis LLM

## Project structure

```
src/mnc/
  audio_input.py   # file/video/YouTube -> normalized mono WAV (ffmpeg, yt-dlp)
  transcribe.py    # WAV -> note events (Basic Pitch) + tempo estimate (librosa)
  lyrics.py        # WAV -> timed lyric words/lines (faster-whisper)
  structure.py     # lyric lines -> labeled sections (LLM or fuzzy heuristic)
  llm.py           # provider-agnostic LLM client (Anthropic / OpenAI-compatible)
  score.py         # notes + lyrics + sections -> two-staff piano score (music21)
  pipeline.py      # end-to-end orchestration
  cli.py           # `mnc transcribe` / `mnc serve`
  web/
    app.py         # FastAPI: job queue + REST API
    static/        # browser UI, renders MusicXML via OpenSheetMusicDisplay
tests/
  test_units.py        # unit tests (quantization, structure, lyrics, dedup)
  make_test_audio.py   # synthesizes a known two-hand piece for e2e testing
```

Run the tests with:

```bash
.venv/bin/python -m unittest discover tests
```

For an end-to-end check, synthesize the test piece and transcribe it — the
treble staff should read the "Twinkle Twinkle" melody:

```bash
.venv/bin/python tests/make_test_audio.py
.venv/bin/mnc transcribe tests/twinkle.wav --tempo 100
```

## Notes on accuracy

Automatic music transcription is an unsolved problem: results are best for
solo piano or clearly pitched melodic material, and get noisier for dense
mixes with drums and vocals. Lyric transcription of singing is similarly
approximate (Whisper is speech-trained), and heuristic section labels are a
rough sketch — an LLM does markedly better. The generated score is a strong
starting point for a human arranger, not a finished engraving. Opening the
MusicXML in MuseScore (free) to touch up rhythms is the recommended workflow.

## Legal

Only transcribe audio you have the right to use. YouTube downloads are
subject to YouTube's Terms of Service; use this tool for your own recordings,
public-domain works, or content you are licensed to arrange.
