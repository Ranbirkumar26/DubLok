from __future__ import annotations

import re
import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from app.config import Settings
from app.constants import ALLOWED_VIDEO_EXTENSIONS
from app.errors import ValidationError

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def safe_filename(filename: str) -> str:
    name = Path(filename or "video").name
    cleaned = _SAFE_NAME.sub("_", name).strip("._")
    return cleaned or "video"


def validate_video_extension(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_VIDEO_EXTENSIONS))
        raise ValidationError(f"Unsupported video format '{ext or 'unknown'}'. Use {allowed}.")
    return ext


def ensure_storage(settings: Settings) -> None:
    settings.storage_root.mkdir(parents=True, exist_ok=True)
    for child in ("sources", "jobs", "tmp", "models"):
        (settings.storage_root / child).mkdir(parents=True, exist_ok=True)


def source_dir(settings: Settings, source_id: str) -> Path:
    path = settings.storage_root / "sources" / source_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def job_dir(settings: Settings, job_id: str) -> Path:
    path = settings.storage_root / "jobs" / job_id
    path.mkdir(parents=True, exist_ok=True)
    for child in ("input", "audio", "tts", "subtitles", "output", "tmp"):
        (path / child).mkdir(parents=True, exist_ok=True)
    return path


def assert_inside(root: Path, candidate: Path) -> Path:
    resolved_root = root.resolve()
    resolved_candidate = candidate.resolve()
    if resolved_root != resolved_candidate and resolved_root not in resolved_candidate.parents:
        raise ValidationError("Unsafe path resolution blocked")
    return resolved_candidate


async def save_upload(upload: UploadFile, settings: Settings, source_id: str) -> Path:
    original_name = safe_filename(upload.filename or "upload.mp4")
    validate_video_extension(original_name)
    target = source_dir(settings, source_id) / original_name
    target = assert_inside(settings.storage_root, target)

    size = 0
    with target.open("wb") as handle:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > settings.max_upload_bytes:
                target.unlink(missing_ok=True)
                raise ValidationError(
                    f"Upload exceeds the configured {settings.max_upload_mb} MB limit."
                )
            handle.write(chunk)
    return target


def copy_to_job_input(settings: Settings, source_path: Path, job_id: str) -> Path:
    validate_video_extension(source_path.name)
    target = job_dir(settings, job_id) / "input" / source_path.name
    target = assert_inside(settings.storage_root, target)
    shutil.copy2(source_path, target)
    return target


def public_artifact_name(path: str | None) -> str | None:
    if not path:
        return None
    return Path(path).name
