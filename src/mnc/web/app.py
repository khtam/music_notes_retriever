"""FastAPI web app: submit a YouTube URL or upload a file, poll job status,
then fetch the rendered MusicXML/MIDI.

Jobs run on a single worker thread (model inference is CPU/ANE-bound and
memory-hungry, so serializing keeps the machine responsive). State lives in
memory; artifacts live under ~/.cache/music-notes-creator/jobs/<id>.
"""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..audio_input import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS, is_url
from ..cli import parse_pitch
from ..pipeline import Options, run

JOBS_ROOT = Path.home() / ".cache" / "music-notes-creator" / "jobs"
STATIC_DIR = Path(__file__).parent / "static"

MAX_UPLOAD_BYTES = 512 * 1024 * 1024


@dataclass
class Job:
    id: str
    status: str = "queued"  # queued | running | done | error
    stage: str = "Queued"
    title: str = ""
    error: Optional[str] = None
    tempo_bpm: Optional[float] = None
    key_name: Optional[str] = None
    n_notes: Optional[int] = None
    duration_seconds: Optional[float] = None
    sections: list[str] = field(default_factory=list)
    structure_method: str = ""
    n_lyric_words: int = 0
    lyrics_language: str = ""
    lyrics_source: str = ""
    musicxml_path: Optional[str] = field(default=None, repr=False)
    midi_path: Optional[str] = field(default=None, repr=False)

    def public(self) -> dict:
        d = asdict(self)
        d["has_result"] = self.status == "done"
        d.pop("musicxml_path")
        d.pop("midi_path")
        return d


app = FastAPI(title="Music Notes Creator")
jobs: dict[str, Job] = {}
jobs_lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=1)


def _set(job: Job, **fields) -> None:
    with jobs_lock:
        for k, v in fields.items():
            setattr(job, k, v)


def _run_job(job: Job, source: str, options: Options) -> None:
    _set(job, status="running", stage="Starting")
    try:
        info = run(
            source,
            output_dir=JOBS_ROOT / job.id,
            options=options,
            progress=lambda stage: _set(job, stage=stage),
        )
        _set(
            job,
            status="done",
            stage="Done",
            title=info.title,
            tempo_bpm=info.tempo_bpm,
            key_name=info.key_name,
            n_notes=info.n_notes,
            duration_seconds=info.duration_seconds,
            sections=info.sections,
            structure_method=info.structure_method,
            n_lyric_words=info.n_lyric_words,
            lyrics_language=info.lyrics_language,
            lyrics_source=info.lyrics_source,
            musicxml_path=str(info.musicxml_path),
            midi_path=str(info.midi_path),
        )
    except Exception as exc:  # surfaced to the UI, so keep the message readable
        _set(job, status="error", stage="Failed", error=str(exc))


@app.post("/api/jobs")
async def create_job(
    url: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    split_point: str = Form("C4"),
    tempo: Optional[float] = Form(None),
    min_note_length: float = Form(120.0),
    onset_threshold: float = Form(0.5),
    title: Optional[str] = Form(None),
    lyrics: bool = Form(True),
    lyrics_text: Optional[str] = Form(None),
    structure: bool = Form(True),
    dedup: bool = Form(True),
    llm: bool = Form(True),
    llm_provider: Optional[str] = Form(None),
    llm_api_key: Optional[str] = Form(None),
):
    if not url and not file:
        raise HTTPException(400, "Provide a YouTube URL or upload a file.")
    if url and not is_url(url):
        raise HTTPException(400, "That doesn't look like a URL (must start with http:// or https://).")

    try:
        split_midi = parse_pitch(split_point)
    except Exception as exc:
        raise HTTPException(400, f"Bad split point: {exc}")

    job = Job(id=uuid.uuid4().hex[:12])
    job_dir = JOBS_ROOT / job.id
    job_dir.mkdir(parents=True, exist_ok=True)

    if url:
        source = url.strip()
    else:
        assert file is not None
        ext = Path(file.filename or "upload").suffix.lower()
        if ext not in AUDIO_EXTENSIONS | VIDEO_EXTENSIONS:
            raise HTTPException(400, f"Unsupported file type {ext!r}.")
        upload_path = job_dir / f"upload{ext}"
        size = 0
        with upload_path.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "File too large (512 MB max).")
                out.write(chunk)
        source = str(upload_path)
        if not title:
            title = Path(file.filename or "").stem or None

    options = Options(
        split_midi=split_midi,
        tempo_override=tempo,
        min_note_length_ms=min_note_length,
        onset_threshold=onset_threshold,
        title=title,
        lyrics=lyrics,
        lyrics_text=lyrics_text,
        structure=structure,
        dedup_repeats=dedup,
        # The key rides in Options only — Job (and the status endpoint) never sees it.
        llm_provider="none" if not llm else (llm_provider or "").strip() or None,
        llm_api_key=(llm_api_key or "").strip() or None,
    )
    with jobs_lock:
        jobs[job.id] = job
    executor.submit(_run_job, job, source, options)
    return JSONResponse(job.public(), status_code=202)


def _get_job(job_id: str) -> Job:
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "No such job")
    return job


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    return _get_job(job_id).public()


@app.get("/api/jobs/{job_id}/musicxml")
def job_musicxml(job_id: str):
    job = _get_job(job_id)
    if job.status != "done" or not job.musicxml_path:
        raise HTTPException(409, "Job is not finished")
    return FileResponse(job.musicxml_path, media_type="application/vnd.recordare.musicxml+xml",
                        filename=Path(job.musicxml_path).name)


@app.get("/api/jobs/{job_id}/midi")
def job_midi(job_id: str):
    job = _get_job(job_id)
    if job.status != "done" or not job.midi_path:
        raise HTTPException(409, "Job is not finished")
    return FileResponse(job.midi_path, media_type="audio/midi",
                        filename=Path(job.midi_path).name)


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
