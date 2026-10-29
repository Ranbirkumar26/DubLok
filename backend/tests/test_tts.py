from __future__ import annotations

from app.config import Settings
from app.pipeline.tts import TTSEngine, gender_from_pitch


def test_hindi_tts_uses_recommended_default_voice(tmp_path) -> None:
    engine = TTSEngine(Settings(storage_root=tmp_path))

    description = engine._voice_description("hi", None)

    assert "Divya's voice" in description
    assert "very clear audio" in description


def test_hindi_tts_picks_male_voice_for_low_pitched_speaker(tmp_path) -> None:
    engine = TTSEngine(Settings(storage_root=tmp_path))

    description = engine._voice_description("hi", None, gender_from_pitch(120.0))

    assert "Rohit's voice" in description
    assert gender_from_pitch(190.0) == "female"
    assert gender_from_pitch(None) is None


def test_tts_loads_model_once_per_device(monkeypatch, tmp_path) -> None:
    import sys
    import types

    loads: list[str] = []

    class FakeModel:
        config = types.SimpleNamespace(text_encoder=types.SimpleNamespace(_name_or_path="t5"))

        @classmethod
        def from_pretrained(cls, name, token=None):
            loads.append(name)
            return cls()

        def to(self, _device):
            return self

    fake_parler = types.SimpleNamespace(ParlerTTSForConditionalGeneration=FakeModel)
    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda name, token=None: name)
    )
    monkeypatch.setitem(sys.modules, "parler_tts", fake_parler)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(TTSEngine, "_loaded", {})

    engine = TTSEngine(Settings(storage_root=tmp_path))
    engine._load("cpu")
    TTSEngine(Settings(storage_root=tmp_path))._load("cpu")

    assert loads == ["ai4bharat/indic-parler-tts"]


def test_tts_accepts_full_custom_voice_description(tmp_path) -> None:
    engine = TTSEngine(Settings(storage_root=tmp_path))
    custom = "Rohit's voice is expressive and close-sounding."

    assert engine._voice_description("hi", custom) == custom


def test_plausible_speech_rate_rejects_looping_and_truncated_takes() -> None:
    from app.pipeline.tts import plausible_speech_rate

    text = "आप एक पल है जहां आप आगे जा सकते हैं"  # 29 characters without spaces
    assert plausible_speech_rate(text, 3.0)
    assert not plausible_speech_rate(text, 9.0)  # looped or stuttering take
    assert not plausible_speech_rate(text, 0.8)  # cut-off take


def test_adds_stutter_flags_unscripted_repeats_only() -> None:
    from app.pipeline.tts import adds_stutter

    script = "आप एक पल है जहां आप आगे जा सकते हैं"
    assert adds_stutter(script, "आप एक पल है, जहाँ आप आप आगे जा सकते हैं")
    assert not adds_stutter(script, "आप एक पल है, जहाँ आप आगे जा सकते हैं")
    assert not adds_stutter("वह धीरे धीरे चला", "वह धीरे धीरे चला")  # scripted reduplication
    assert not adds_stutter(script, "")  # no check available
