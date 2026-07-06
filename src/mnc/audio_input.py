"""Turn any supported source (audio file, video file, YouTube URL) into a WAV.

Everything is normalized to a 22050 Hz mono WAV, which is Basic Pitch's
native sample rate, so downstream code never has to care about formats.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".aiff", ".aif"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".ts", ".flv"}

_URL_RE = re.compile(r"^https?://", re.IGNORECASE)

TARGET_SAMPLE_RATE = 22050


def is_url(source: str) -> bool:
    return bool(_URL_RE.match(source.strip()))


def get_ffmpeg() -> str:
    """Prefer a system ffmpeg; fall back to the binary bundled with imageio-ffmpeg."""
    system = shutil.which("ffmpeg")
    if system:
        return system
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def to_wav(source: Path, out_wav: Path) -> Path:
    """Extract/convert the audio track of any media file to mono WAV."""
    cmd = [
        get_ffmpeg(),
        "-y",
        "-i", str(source),
        "-vn",
        "-ac", "1",
        "-ar", str(TARGET_SAMPLE_RATE),
        "-acodec", "pcm_s16le",
        str(out_wav),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-8:])
        raise RuntimeError(f"ffmpeg could not extract audio from {source.name}:\n{tail}")
    return out_wav


def download_youtube(url: str, workdir: Path) -> tuple[Path, str]:
    """Download the best audio-only stream for a YouTube (or yt-dlp supported) URL.

    Returns (downloaded_file, video_title). We request a single pre-merged
    stream so ffprobe/format-merging is never needed.
    """
    import yt_dlp

    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(workdir / "source.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "noprogress": True,
        "no_warnings": True,
        "ffmpeg_location": str(Path(get_ffmpeg()).parent),
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if info is None:
            raise RuntimeError(f"Could not fetch media info for {url}")
        if "entries" in info:  # playlist guard, take first entry
            info = next(e for e in info["entries"] if e)
        path = Path(ydl.prepare_filename(info))
    title = info.get("title") or "YouTube audio"
    return path, title


def prepare_audio(source: str, workdir: Path) -> tuple[Path, str]:
    """Resolve any source into (wav_path, title)."""
    workdir.mkdir(parents=True, exist_ok=True)
    wav = workdir / "audio.wav"

    if is_url(source):
        media, title = download_youtube(source, workdir)
        return to_wav(media, wav), title

    path = Path(source).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    ext = path.suffix.lower()
    if ext not in AUDIO_EXTENSIONS | VIDEO_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type {ext!r}. Supported: "
            + ", ".join(sorted(AUDIO_EXTENSIONS | VIDEO_EXTENSIONS))
        )
    return to_wav(path, wav), path.stem
