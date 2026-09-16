from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator

from app.constants import LanguageCode


class SourceResponse(BaseModel):
    source_id: str
    input_type: Literal["upload", "url"]
    status: str
    metadata: dict[str, Any] | None = None
    preview_url: str | None = None
    message: str | None = None


class UrlInput(BaseModel):
    url: HttpUrl


class JobCreate(BaseModel):
    source_id: str
    target_language: LanguageCode = "hi"
    source_language_override: LanguageCode | None = None
    model_preset: Literal["fast", "balanced", "quality"] = "balanced"
    audio_mode: Literal["full_replacement", "preserve_background"] = "full_replacement"
    subtitle_mode: Literal["none", "original", "translated", "dual"] = "translated"
    speech_speed: float = Field(default=1.0, ge=0.5, le=2.0)
    original_volume: float = Field(default=0.2, ge=0.0, le=1.5)
    translated_volume: float = Field(default=1.0, ge=0.0, le=2.0)
    voice_map: dict[str, str] = Field(default_factory=dict)


class JobActionOptions(BaseModel):
    target_language: LanguageCode | None = None
    model_preset: Literal["fast", "balanced", "quality"] | None = None
    audio_mode: Literal["full_replacement", "preserve_background"] | None = None
    subtitle_mode: Literal["none", "original", "translated", "dual"] | None = None
    speech_speed: float | None = Field(default=None, ge=0.5, le=2.0)
    original_volume: float | None = Field(default=None, ge=0.0, le=1.5)
    translated_volume: float | None = Field(default=None, ge=0.0, le=2.0)
    voice_map: dict[str, str] | None = None


class TranscriptSegment(BaseModel):
    id: str
    speaker: str = "Speaker 1"
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str
    translated_text: str | None = None

    @field_validator("end")
    @classmethod
    def end_must_be_after_start(cls, value: float, info: Any) -> float:
        start = info.data.get("start")
        if start is not None and value < start:
            raise ValueError("segment end must be after start")
        return value


class TranscriptPayload(BaseModel):
    detected_language: LanguageCode | str | None = None
    confidence: float | None = None
    segments: list[TranscriptSegment]


class JobResponse(BaseModel):
    id: str
    source_id: str
    status: str
    current_stage: str | None
    progress: float
    error: str | None
    target_language: LanguageCode
    source_language_override: LanguageCode | None
    model_preset: str
    audio_mode: str
    subtitle_mode: str
    speech_speed: float
    original_volume: float
    translated_volume: float
    voice_map: dict[str, str]
    source: SourceResponse | None = None
    metadata: dict[str, Any] | None = None
    transcript: TranscriptPayload | None = None
    translated_transcript: TranscriptPayload | None = None
    artifacts: dict[str, Any]
    stage_log: list[dict[str, Any]]
    created_at: str
    updated_at: str


class ResultResponse(BaseModel):
    job_id: str
    status: str
    video_url: str | None = None
    srt_url: str | None = None
    vtt_url: str | None = None
    original_transcript_url: str | None = None
    translated_transcript_url: str | None = None
    artifacts: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    ffmpeg: bool
    ffprobe: bool
    engine_mode: str
