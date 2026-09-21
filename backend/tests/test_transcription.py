from __future__ import annotations

import sys
import types
from types import SimpleNamespace

from app.config import Settings
from app.pipeline.transcription import Transcriber


def test_balanced_transcription_falls_back_to_cached_tiny(monkeypatch, tmp_path) -> None:
    attempts: list[str] = []

    class FakeWhisperModel:
        def __init__(self, model_size: str, **_kwargs) -> None:
            attempts.append(model_size)
            if model_size == "small":
                raise RuntimeError("offline")

        def transcribe(self, *_args, **_kwargs):
            return [
                SimpleNamespace(text="Hello world", start=0.0, end=1.0),
            ], SimpleNamespace(language="en", language_probability=0.99)

    fake_module = types.SimpleNamespace(WhisperModel=FakeWhisperModel)
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)

    settings = Settings(engine_mode="production", storage_root=tmp_path, hf_home=tmp_path / "hf")
    payload = Transcriber(settings, "balanced").transcribe(
        tmp_path / "audio.wav",
        source_language_override=None,
        duration=1.0,
    )

    assert attempts == ["small", "tiny"]
    assert payload["detected_language"] == "en"
    assert payload["segments"][0]["text"] == "Hello world"


def test_balanced_transcription_prefers_cached_fallback(monkeypatch, tmp_path) -> None:
    attempts: list[str] = []
    cached_tiny = tmp_path / "tiny"

    class FakeWhisperModel:
        def __init__(self, model_ref: str, **_kwargs) -> None:
            attempts.append(model_ref)

        def transcribe(self, *_args, **_kwargs):
            return [
                SimpleNamespace(text="Hello world", start=0.0, end=1.0),
            ], SimpleNamespace(language="en", language_probability=0.99)

    fake_module = types.SimpleNamespace(WhisperModel=FakeWhisperModel)
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)
    monkeypatch.setattr(
        Transcriber,
        "_cached_whisper_model_path",
        lambda _self, model_size: cached_tiny if model_size == "tiny" else None,
    )

    settings = Settings(engine_mode="production", storage_root=tmp_path, hf_home=tmp_path / "hf")
    payload = Transcriber(settings, "balanced").transcribe(
        tmp_path / "audio.wav",
        source_language_override=None,
        duration=1.0,
    )

    assert attempts == [str(cached_tiny)]
    assert payload["segments"][0]["text"] == "Hello world"
