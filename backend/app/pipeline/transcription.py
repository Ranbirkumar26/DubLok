from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.errors import DependencyMissingError, ProcessingError
from app.pipeline.types import Segment, transcript_payload

WHISPER_MODEL_BY_PRESET = {
    "fast": "tiny",
    "balanced": "small",
    "quality": "medium",
}


class Transcriber:
    def __init__(self, settings: Settings, model_preset: str):
        self.settings = settings
        self.model_preset = model_preset

    def transcribe(
        self,
        audio_path: Path,
        *,
        source_language_override: str | None,
        duration: float,
    ) -> dict:
        if self.settings.engine_mode == "demo":
            return self._demo_transcript(duration, source_language_override)
        return self._faster_whisper(audio_path, source_language_override)

    def _demo_transcript(self, duration: float, language: str | None) -> dict:
        end = min(max(duration, 1.0), 6.0)
        segments = [
            Segment(
                id="seg_0001",
                speaker="Speaker 1",
                start=0.0,
                end=end,
                text="Demo transcript segment. Install local models for real transcription.",
            )
        ]
        return transcript_payload(
            segments=segments,
            detected_language=language or "en",
            confidence=0.99,
        )

    def _faster_whisper(self, audio_path: Path, language: str | None) -> dict:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise DependencyMissingError(
                "faster-whisper is not installed. Run `pip install -r backend/requirements-ml.txt`."
            ) from exc

        model_size = WHISPER_MODEL_BY_PRESET.get(self.model_preset, "small")
        device = "cpu" if self.settings.device == "cpu" else "auto"
        compute_type = "int8" if device == "cpu" else "float16"
        try:
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
            raw_segments, info = model.transcribe(
                str(audio_path),
                language=language,
                vad_filter=True,
                word_timestamps=True,
                beam_size=5,
            )
        except Exception as exc:  # pragma: no cover - depends on local model/runtime
            raise ProcessingError(f"Transcription failed: {exc}") from exc

        segments: list[Segment] = []
        for index, raw in enumerate(raw_segments, start=1):
            text = (raw.text or "").strip()
            if not text:
                continue
            segments.append(
                Segment(
                    id=f"seg_{index:04d}",
                    speaker="Speaker 1",
                    start=float(raw.start),
                    end=float(raw.end),
                    text=text,
                )
            )
        if not segments:
            raise ProcessingError("No detectable speech was found in the audio.")
        return transcript_payload(
            segments=segments,
            detected_language=getattr(info, "language", None),
            confidence=getattr(info, "language_probability", None),
        )
