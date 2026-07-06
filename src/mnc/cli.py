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

    s = sub.add_parser("serve", help="Run the web app")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)

    if args.command == "serve":
        import uvicorn

        uvicorn.run("mnc.web.app:app", host=args.host, port=args.port)
        return 0

    from .audio_input import is_url

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
    )
    try:
        info = run(args.source, output_dir, options, progress=lambda stage: print(f"* {stage}"))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"\n{info.title}\n"
        f"  notes: {info.n_notes} | tempo: {info.tempo_bpm:.0f} BPM | "
        f"key: {info.key_name} | length: {info.duration_seconds:.0f}s\n"
        f"  sheet music: {info.musicxml_path}\n"
        f"  midi:        {info.midi_path}\n\n"
        f"Open the .musicxml in MuseScore (free) to view/print, or run "
        f"`mnc serve` to view in the browser."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
