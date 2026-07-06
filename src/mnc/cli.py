"""Command-line interface: `mnc transcribe <source>` and `mnc serve`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .pipeline import Options, run

NOTE_NAMES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def parse_pitch(text: str) -> int:
    """Accept either a MIDI number ('60') or a note name ('C4', 'F#3')."""
    text = text.strip()
    if text.isdigit():
        midi = int(text)
    else:
        name = text[0].upper()
        rest = text[1:]
        alter = 0
        while rest and rest[0] in "#b":
            alter += 1 if rest[0] == "#" else -1
            rest = rest[1:]
        try:
            octave = int(rest)
            midi = 12 * (octave + 1) + NOTE_NAMES[name] + alter
        except (ValueError, KeyError):
            raise argparse.ArgumentTypeError(
                f"Cannot parse pitch {text!r}; use a MIDI number (60) or note name (C4)"
            )
    if not 21 <= midi <= 108:
        raise argparse.ArgumentTypeError(f"Pitch {text!r} is outside the piano range (A0-C8)")
    return midi


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mnc",
        description="Transcribe songs from audio/video files or YouTube into piano sheet music.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("transcribe", help="Transcribe a file or YouTube URL to sheet music")
    t.add_argument("source", help="Audio file, video file, or YouTube URL")
    t.add_argument("--output-dir", type=Path, default=None,
                   help="Directory for the .musicxml/.mid output (default: alongside input, or ./output for URLs)")
    t.add_argument("--split-point", type=parse_pitch, default=60, metavar="PITCH",
                   help="Treble/bass split pitch, e.g. C4 or 60 (default: C4)")
    t.add_argument("--tempo", type=float, default=None, help="Override the estimated tempo (BPM)")
    t.add_argument("--min-note-length", type=float, default=120.0, metavar="MS",
                   help="Discard notes shorter than this many milliseconds (default: 120)")
    t.add_argument("--onset-threshold", type=float, default=0.5,
                   help="Note-onset sensitivity, 0-1; lower catches more notes (default: 0.5)")
    t.add_argument("--title", default=None, help="Title printed on the score")
    t.add_argument("--no-lyrics", action="store_true",
                   help="Skip vocal transcription (no words under the melody)")
    t.add_argument("--lyrics-file", type=Path, default=None, metavar="PATH",
                   help="Text file with the song's lyrics (one line per sung line). "
                        "Used instead of Whisper's transcription and aligned to the audio.")
    t.add_argument("--no-structure", action="store_true",
                   help="Skip verse/chorus section labeling")
    t.add_argument("--no-dedup", action="store_true",
                   help="Engrave repeated sections in full instead of collapsing them")
    t.add_argument("--whisper-model", default="small", metavar="SIZE",
                   choices=["tiny", "base", "small", "medium", "large-v3"],
                   help="faster-whisper model for lyrics: tiny/base/small/medium/large-v3 (default: small)")
    t.add_argument("--llm", default=None, metavar="PROVIDER",
                   help="LLM for structure analysis: anthropic, openai, or none "
                        "(default: auto-detect from ANTHROPIC_API_KEY / OPENAI_API_KEY)")
    t.add_argument("--llm-model", default=None,
                   help="Model override for the chosen LLM provider")

    s = sub.add_parser("serve", help="Run the web app")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)

    if args.command == "serve":
        import uvicorn

        uvicorn.run("mnc.web.app:app", host=args.host, port=args.port)
        return 0

    from .audio_input import is_url

    lyrics_text = None
    if args.lyrics_file is not None:
        try:
            lyrics_text = args.lyrics_file.expanduser().read_text(encoding="utf-8")
        except OSError as exc:
            print(f"error: cannot read lyrics file: {exc}", file=sys.stderr)
            return 1
        if not lyrics_text.strip():
            print(f"error: lyrics file {args.lyrics_file} is empty", file=sys.stderr)
            return 1

    if args.output_dir is not None:
        output_dir = args.output_dir
    elif is_url(args.source):
        output_dir = Path.cwd() / "output"
    else:
        output_dir = Path(args.source).expanduser().resolve().parent

    options = Options(
        split_midi=args.split_point,
        tempo_override=args.tempo,
        min_note_length_ms=args.min_note_length,
        onset_threshold=args.onset_threshold,
        title=args.title,
        lyrics=not args.no_lyrics,
        lyrics_text=lyrics_text,
        structure=not args.no_structure,
        dedup_repeats=not args.no_dedup,
        whisper_model=args.whisper_model,
        llm_provider=args.llm,
        llm_model=args.llm_model,
    )
    try:
        info = run(args.source, output_dir, options, progress=lambda stage: print(f"* {stage}"))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"\n{info.title}\n"
        f"  notes: {info.n_notes} | tempo: {info.tempo_bpm:.0f} BPM | "
        f"key: {info.key_name} | length: {info.duration_seconds:.0f}s"
    )
    if info.n_lyric_words:
        details = ", ".join(x for x in (info.lyrics_language, info.lyrics_source) if x)
        print(f"  lyrics: {info.n_lyric_words} words" + (f" ({details})" if details else ""))
    if info.sections:
        method = f" [{info.structure_method}]" if info.structure_method else ""
        print(f"  structure{method}:")
        for line in info.sections:
            print(f"    - {line}")
    print(
        f"  sheet music: {info.musicxml_path}\n"
        f"  midi:        {info.midi_path}\n\n"
        f"Open the .musicxml in MuseScore (free) to view/print, or run "
        f"`mnc serve` to view in the browser."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
