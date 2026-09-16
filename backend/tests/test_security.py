from __future__ import annotations

import pytest

from app.config import Settings
from app.errors import ValidationError
from app.security import normalize_google_drive_url, validate_public_video_url


def settings() -> Settings:
    return Settings(
        allowed_url_hosts=["youtube.com", "www.youtube.com", "youtu.be", "drive.google.com"]
    )


def test_accepts_youtube_url() -> None:
    url, kind = validate_public_video_url("https://www.youtube.com/watch?v=abc123", settings())
    assert kind == "youtube"
    assert url.endswith("abc123")


def test_rejects_unlisted_host() -> None:
    with pytest.raises(ValidationError):
        validate_public_video_url("https://example.com/video.mp4", settings())


def test_rejects_private_ip_url() -> None:
    with pytest.raises(ValidationError):
        validate_public_video_url("http://127.0.0.1/video.mp4", settings())


def test_normalizes_google_drive_file_url() -> None:
    url = normalize_google_drive_url("https://drive.google.com/file/d/FILE_ID/view?usp=sharing")
    assert url == "https://drive.google.com/uc?id=FILE_ID&export=download"
