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

WHISPER_FALLBACKS = {
    "medium": ["small", "tiny"],
    "small": ["tiny"],
    "tiny": [],
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
        self.settings.apply_model_environment()
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise DependencyMissingError(
                "faster-whisper is not installed. Run `pip install -r backend/requirements-ml.txt`."
            ) from exc

        model_size = WHISPER_MODEL_BY_PRESET.get(self.model_preset, "small")
        device = "cpu" if self.settings.device == "cpu" else "auto"
        compute_type = "int8" if device == "cpu" else "float16"
        model_errors: list[str] = []
        model_names = [model_size, *WHISPER_FALLBACKS.get(model_size, [])]
        cached_models = [
            (candidate, cached_path)
            for candidate in model_names
            if (cached_path := self._cached_whisper_model_path(candidate))
        ]
        load_candidates: list[tuple[str, str | Path, bool]]
        if cached_models:
            load_candidates = [
                (candidate, cached_path, True)
                for candidate, cached_path in cached_models
            ]
        else:
            load_candidates = [
                (candidate, candidate, False)
                for candidate in model_names
            ]

        model = None
        for label, model_ref, local_files_only in load_candidates:
            try:
                model = WhisperModel(
                    str(model_ref),
                    device=device,
                    compute_type=compute_type,
                    local_files_only=local_files_only,
                    use_auth_token=self.settings.huggingface_hub_token or None,
                )
                break
            except Exception as exc:  # pragma: no cover - depends on local model/runtime
                model_errors.append(f"{label}: {exc}")
                if label == load_candidates[-1][0]:
                    raise ProcessingError(
                        "Transcription failed while loading Whisper models. "
                        "Tried " + "; ".join(model_errors)
                    ) from exc

        try:
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

    def _cached_whisper_model_path(self, model_size: str) -> Path | None:
        snapshots_dir = (
            self.settings.hf_home
            / "hub"
            / f"models--Systran--faster-whisper-{model_size}"
            / "snapshots"
        )
        if not snapshots_dir.is_dir():
            return None

        candidates: list[Path] = []
        ref_path = snapshots_dir.parent / "refs" / "main"
        if ref_path.is_file():
            ref = ref_path.read_text(encoding="utf-8").strip()
            if ref:
                candidates.append(snapshots_dir / ref)
        candidates.extend(
            sorted(
                (path for path in snapshots_dir.iterdir() if path.is_dir()),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        )

        required_files = ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt")
        for candidate in candidates:
            if all((candidate / filename).exists() for filename in required_files):
                return candidate
        return None
