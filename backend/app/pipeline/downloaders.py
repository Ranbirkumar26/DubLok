from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from app.config import Settings
from app.errors import DependencyMissingError, ProcessingError
from app.media import run_command
from app.storage import validate_video_extension


def download_public_video(url: str, output_dir: Path, settings: Settings) -> Path:
    command = yt_dlp_command()
    if not command:
        raise DependencyMissingError(
            "yt-dlp is required for public URL inputs. Install backend requirements and retry."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    template = output_dir / "source.%(ext)s"
    run_command(
        [
            *command,
            "--no-playlist",
            "--restrict-filenames",
            "--merge-output-format",
            "mp4",
            "--max-filesize",
            f"{settings.max_upload_mb}M",
            "-f",
            "bv*[height<=720][ext=mp4]+ba[ext=m4a]/bv*[height<=720]+ba/b[height<=720]/b",
            "-o",
            str(template),
            url,
        ],
        timeout=settings.job_timeout_seconds,
    )
    candidates = sorted(output_dir.glob("source.*"))
    for candidate in candidates:
        if candidate.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm"}:
            validate_video_extension(candidate.name)
            return candidate
    raise ProcessingError("The public video download finished but no supported video file was produced.")


def yt_dlp_command() -> list[str] | None:
    from shutil import which

    executable = which("yt-dlp")
    if executable:
        return [executable]
    if importlib.util.find_spec("yt_dlp") is not None:
        return [sys.executable, "-m", "yt_dlp"]
    return None
