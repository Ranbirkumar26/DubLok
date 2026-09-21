from __future__ import annotations

from app.config import Settings
from app.pipeline.tts import TTSEngine


def test_hindi_tts_uses_recommended_default_voice(tmp_path) -> None:
    engine = TTSEngine(Settings(storage_root=tmp_path))

    description = engine._voice_description("hi", None)

    assert "Divya's voice" in description
    assert "very clear audio" in description


def test_tts_accepts_full_custom_voice_description(tmp_path) -> None:
    engine = TTSEngine(Settings(storage_root=tmp_path))
    custom = "Rohit's voice is expressive and close-sounding."

    assert engine._voice_description("hi", custom) == custom
