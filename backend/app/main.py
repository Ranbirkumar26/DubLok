from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.config import Settings, get_settings
from app.constants import ALLOWED_ARTIFACTS, JOB_STAGES, LANGUAGE_OPTIONS
from app.db import Database
from app.errors import AppError, ValidationError
from app.media import command_exists, probe_video
from app.schemas import (
    HealthResponse,
    JobActionOptions,
    JobCreate,
    JobResponse,
    ResultResponse,
    SourceResponse,
    TranscriptPayload,
    UrlInput,
)
from app.security import validate_public_video_url
from app.storage import (
    assert_inside,
    ensure_storage,
    new_id,
    public_artifact_name,
    save_upload,
)

app = FastAPI(title="Local Zero-Cost Video Dubbing Studio")


@app.on_event("startup")
def startup() -> None:
    settings = get_settings()
    ensure_storage(settings)
    Database(settings).init()


def settings_dep() -> Settings:
    return get_settings()


def db_dep(settings: Settings = Depends(settings_dep)) -> Database:
    return Database(settings)


app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def health(settings: Settings = Depends(settings_dep)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        ffmpeg=command_exists("ffmpeg"),
        ffprobe=command_exists("ffprobe"),
        engine_mode=settings.engine_mode,
    )


@app.get("/api/languages")
def languages() -> dict[str, Any]:
    return {"languages": LANGUAGE_OPTIONS}


@app.get("/api/stages")
def stages() -> dict[str, Any]:
    return {"stages": JOB_STAGES}


@app.post("/api/video/upload", response_model=SourceResponse)
async def upload_video(
    file: UploadFile = File(...),
    settings: Settings = Depends(settings_dep),
    db: Database = Depends(db_dep),
) -> SourceResponse:
    source_id = new_id("src")
    try:
        path = await save_upload(file, settings, source_id)
        metadata = probe_video(path, settings)
        source = db.create_source(
            source_id=source_id,
            input_type="upload",
            original_name=file.filename,
            local_path=str(path),
            url=None,
            url_kind=None,
            mime_type=file.content_type,
            status="ready",
            metadata=metadata,
        )
        return _source_response(source)
    except AppError as exc:
        raise HTTPException(status_code=400, detail=exc.message) from exc


@app.post("/api/video/url", response_model=SourceResponse)
def submit_url(
    payload: UrlInput,
    settings: Settings = Depends(settings_dep),
    db: Database = Depends(db_dep),
) -> SourceResponse:
    try:
        normalized_url, kind = validate_public_video_url(str(payload.url), settings)
        source_id = new_id("src")
        source = db.create_source(
            source_id=source_id,
            input_type="url",
            original_name=None,
            local_path=None,
            url=normalized_url,
            url_kind=kind,
            mime_type=None,
            status="pending_download",
        )
        return _source_response(
            source,
            message="URL accepted. The worker will download and validate it during analysis.",
        )
    except AppError as exc:
        raise HTTPException(status_code=400, detail=exc.message) from exc


@app.post("/api/jobs", response_model=JobResponse)
def create_job(
    payload: JobCreate,
    db: Database = Depends(db_dep),
) -> JobResponse:
    source = db.get_source(payload.source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    job = db.create_job(job_id=new_id("job"), source_id=payload.source_id, payload=payload.model_dump())
    return _job_response(job, source)


@app.get("/api/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str, db: Database = Depends(db_dep)) -> JobResponse:
    job, source = _job_and_source(job_id, db)
    return _job_response(job, source)


@app.get("/api/jobs/{job_id}/transcript", response_model=TranscriptPayload)
def get_transcript(
    job_id: str,
    kind: Literal["original", "translated"] = Query(default="original"),
    db: Database = Depends(db_dep),
) -> TranscriptPayload:
    job, _ = _job_and_source(job_id, db)
    payload = job.get("translated_transcript") if kind == "translated" else job.get("transcript")
    if not payload:
        raise HTTPException(status_code=404, detail=f"{kind.title()} transcript not available")
    return TranscriptPayload.model_validate(payload)


@app.put("/api/jobs/{job_id}/transcript", response_model=JobResponse)
def update_transcript(
    job_id: str,
    payload: TranscriptPayload,
    kind: Literal["original", "translated"] = Query(default="translated"),
    db: Database = Depends(db_dep),
) -> JobResponse:
    job, source = _job_and_source(job_id, db)
    field = "translated_transcript" if kind == "translated" else "transcript"
    updated = db.update_job(job["id"], **{field: payload.model_dump()})
    return _job_response(updated, source)


@app.post("/api/jobs/{job_id}/translate", response_model=JobResponse)
def enqueue_translation(
    job_id: str,
    options: JobActionOptions | None = None,
    db: Database = Depends(db_dep),
) -> JobResponse:
    job, source = _job_and_source(job_id, db)
    if not job.get("transcript"):
        raise HTTPException(status_code=409, detail="Analysis must complete before translation.")
    updates = _option_updates(options)
    updates.update(status="queued_translation", current_stage="queued translation", progress=max(job["progress"], 40))
    updated = db.update_job(job_id, **updates)
    return _job_response(updated, source)


@app.post("/api/jobs/{job_id}/generate-audio", response_model=JobResponse)
def enqueue_audio(
    job_id: str,
    options: JobActionOptions | None = None,
    db: Database = Depends(db_dep),
) -> JobResponse:
    job, source = _job_and_source(job_id, db)
    if not job.get("translated_transcript"):
        raise HTTPException(status_code=409, detail="Translation must be ready before audio generation.")
    updates = _option_updates(options)
    updates.update(status="queued_audio", current_stage="queued audio", progress=max(job["progress"], 60))
    updated = db.update_job(job_id, **updates)
    return _job_response(updated, source)


@app.post("/api/jobs/{job_id}/render", response_model=JobResponse)
def enqueue_render(job_id: str, db: Database = Depends(db_dep)) -> JobResponse:
    job, source = _job_and_source(job_id, db)
    if not job.get("artifacts", {}).get("final_audio"):
        raise HTTPException(status_code=409, detail="Audio must be ready before rendering.")
    updated = db.update_job(
        job_id,
        status="queued_render",
        current_stage="queued render",
        progress=max(job["progress"], 92),
    )
    return _job_response(updated, source)


@app.get("/api/jobs/{job_id}/result", response_model=ResultResponse)
def get_result(job_id: str, db: Database = Depends(db_dep)) -> ResultResponse:
    job, _ = _job_and_source(job_id, db)
    artifacts = _public_artifacts(job)
    return ResultResponse(
        job_id=job["id"],
        status=job["status"],
        video_url=artifacts.get("video", {}).get("url"),
        srt_url=artifacts.get("srt", {}).get("url"),
        vtt_url=artifacts.get("vtt", {}).get("url"),
        original_transcript_url=artifacts.get("original_transcript", {}).get("url"),
        translated_transcript_url=artifacts.get("translated_transcript", {}).get("url"),
        artifacts=artifacts,
    )


@app.get("/api/jobs/{job_id}/download/{artifact}")
def download_artifact(
    job_id: str,
    artifact: str,
    settings: Settings = Depends(settings_dep),
    db: Database = Depends(db_dep),
) -> FileResponse:
    if artifact not in ALLOWED_ARTIFACTS:
        raise HTTPException(status_code=404, detail="Unknown artifact")
    job, _ = _job_and_source(job_id, db)
    path_value = (job.get("artifacts") or {}).get(artifact)
    if not path_value:
        raise HTTPException(status_code=404, detail="Artifact not available")
    path = assert_inside(settings.storage_root, Path(path_value))
    if not path.exists():
        raise HTTPException(status_code=404, detail="Artifact file missing")
    return FileResponse(path, filename=path.name)


@app.get("/api/media/source/{source_id}")
def preview_source(
    source_id: str,
    settings: Settings = Depends(settings_dep),
    db: Database = Depends(db_dep),
) -> FileResponse:
    source = db.get_source(source_id)
    if not source or not source.get("local_path"):
        raise HTTPException(status_code=404, detail="Source preview not available")
    path = assert_inside(settings.storage_root, Path(source["local_path"]))
    if not path.exists():
        raise HTTPException(status_code=404, detail="Source file missing")
    return FileResponse(path, filename=path.name)


def _job_and_source(job_id: str, db: Database) -> tuple[dict[str, Any], dict[str, Any]]:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    source = db.get_source(job["source_id"])
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    return job, source


def _source_response(source: dict[str, Any], message: str | None = None) -> SourceResponse:
    preview = f"/api/media/source/{source['id']}" if source.get("local_path") else None
    return SourceResponse(
        source_id=source["id"],
        input_type=source["input_type"],
        status=source["status"],
        metadata=source.get("metadata") or None,
        preview_url=preview,
        message=message,
    )


def _job_response(job: dict[str, Any], source: dict[str, Any]) -> JobResponse:
    payload = {
        **job,
        "source": _source_response(source).model_dump(),
        "metadata": (job.get("artifacts") or {}).get("metadata") or source.get("metadata"),
        "artifacts": _public_artifacts(job),
    }
    return JobResponse.model_validate(payload)


def _public_artifacts(job: dict[str, Any]) -> dict[str, Any]:
    public: dict[str, Any] = {}
    for key, path in (job.get("artifacts") or {}).items():
        if key == "metadata":
            continue
        if key in ALLOWED_ARTIFACTS and path:
            public[key] = {
                "name": public_artifact_name(path),
                "url": f"/api/jobs/{job['id']}/download/{key}",
            }
    return public


def _option_updates(options: JobActionOptions | None) -> dict[str, Any]:
    if not options:
        return {}
    raw = options.model_dump(exclude_none=True)
    updates: dict[str, Any] = {}
    for key, value in raw.items():
        updates[key] = value
    return updates
