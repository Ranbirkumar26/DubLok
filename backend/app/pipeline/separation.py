from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.errors import ProcessingError
from app.media import run_command


class BackgroundSeparator:
    def __init__(self, settings: Settings):
        self.settings = settings

    def preserve_background(self, audio_path: Path, output_dir: Path) -> Path | None:
        if self.settings.engine_mode == "demo":
            return None
        try:
            import demucs.separate  # noqa: F401
        except ImportError:
            return None
        try:
            run_command(
                [
                    "python",
                    "-m",
                    "demucs.separate",
                    "--two-stems",
                    "vocals",
                    "-o",
                    str(output_dir),
                    str(audio_path),
                ],
                timeout=self.settings.job_timeout_seconds,
            )
        except ProcessingError:
            return None
        candidates = list(output_dir.rglob("no_vocals.*"))
        return candidates[0] if candidates else None
