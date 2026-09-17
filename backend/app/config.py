from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    app_name: str = "Local Video Dubbing Studio"
    app_env: str = "development"
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"]
    )

    storage_root: Path = Path("./storage")
    database_url: str = "sqlite:///./storage/app.db"
    worker_poll_seconds: float = 2.0
    job_timeout_seconds: int = 7200
    stage_max_retries: int = 1

    max_upload_mb: int = 750
    max_duration_seconds: int = 900
    allowed_url_hosts: list[str] = Field(
        default_factory=lambda: [
            "youtube.com",
            "www.youtube.com",
            "youtu.be",
            "drive.google.com",
        ]
    )

    engine_mode: Literal["production", "demo"] = "production"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    model_preset: Literal["fast", "balanced", "quality"] = "balanced"
    hf_home: Path = Path("./storage/models/huggingface")
    hf_hub_disable_xet: str | None = "1"
    huggingface_hub_token: str | None = None

    background_mode: Literal["full_replacement", "preserve_background"] = (
        "full_replacement"
    )
    original_volume: float = 0.20
    translated_volume: float = 1.00
    speech_speed: float = 1.00

    @field_validator("cors_origins", "allowed_url_hosts", mode="before")
    @classmethod
    def split_csv(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @property
    def sqlite_path(self) -> Path:
        prefix = "sqlite:///"
        if not self.database_url.startswith(prefix):
            raise ValueError("Only sqlite:/// DATABASE_URL values are supported")
        return Path(self.database_url[len(prefix) :])

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def apply_model_environment(self) -> None:
        os.environ.setdefault("HF_HOME", str(self.hf_home))
        if self.hf_hub_disable_xet:
            os.environ.setdefault("HF_HUB_DISABLE_XET", self.hf_hub_disable_xet)


@lru_cache
def get_settings() -> Settings:
    return Settings()
