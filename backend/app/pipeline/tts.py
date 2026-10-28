from __future__ import annotations

import wave
from math import pi, sin
from pathlib import Path

from app.config import Settings
from app.constants import SUPPORTED_LANGUAGES
from app.errors import DependencyMissingError, ProcessingError

RECOMMENDED_TTS_SPEAKERS = {
    "hi": {"female": "Divya", "male": "Rohit"},
    "en": {"female": "Mary", "male": "Thoma"},
    "ta": {"female": "Jaya"},
    "te": {"female": "Lalitha", "male": "Prakash"},
    "kn": {"female": "Anu", "male": "Suresh"},
    "ml": {"female": "Anjali", "male": "Harish"},
    "bn": {"female": "Aditi", "male": "Arjun"},
    "mr": {"female": "Sunita", "male": "Sanjay"},
}

# Median F0 split between typical adult male and female speech.
FEMALE_PITCH_HZ = 165.0

MODEL_NAME = "ai4bharat/indic-parler-tts"


def gender_from_pitch(median_pitch_hz: float | None) -> str | None:
    if not median_pitch_hz:
        return None
    return "female" if median_pitch_hz >= FEMALE_PITCH_HZ else "male"


class TTSEngine:
    # Loaded once per worker process; loading Parler-TTS took ~50 s per segment.
    _loaded: dict[tuple[str, str], tuple[object, object, object]] = {}

    def __init__(self, settings: Settings):
        self.settings = settings

    def synthesize(
        self,
        *,
        text: str,
        language: str,
        voice: str | None,
        output_path: Path,
        gender: str | None = None,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.settings.engine_mode == "demo":
            self._demo_tone(text, output_path)
            return
        self._indic_parler(
            text=text, language=language, voice=voice, gender=gender, output_path=output_path
        )

    def _load(self, device: str) -> tuple[object, object, object]:
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        key = (MODEL_NAME, device)
        if key not in self._loaded:
            token = self.settings.huggingface_hub_token or None
            model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL_NAME, token=token).to(device)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=token)
            description_tokenizer = AutoTokenizer.from_pretrained(
                model.config.text_encoder._name_or_path, token=token
            )
            TTSEngine._loaded[key] = (model, tokenizer, description_tokenizer)
        return self._loaded[key]

    def _indic_parler(
        self,
        *,
        text: str,
        language: str,
        voice: str | None,
        gender: str | None,
        output_path: Path,
    ) -> None:
        self.settings.apply_model_environment()
        try:
            import soundfile as sf
            import torch
            import parler_tts  # noqa: F401
        except ImportError as exc:
            raise DependencyMissingError(
                "Indic Parler-TTS dependencies are missing. Run `pip install -r backend/requirements-ml.txt`."
            ) from exc

        prompt = self._voice_description(language, voice, gender)
        device = "cuda" if self.settings.device != "cpu" and torch.cuda.is_available() else "cpu"
        try:
            model, tokenizer, description_tokenizer = self._load(device)
            inputs = description_tokenizer(prompt, return_tensors="pt").to(device)
            prompt_inputs = tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                generation = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    prompt_input_ids=prompt_inputs.input_ids,
                    prompt_attention_mask=prompt_inputs.attention_mask,
                )
            audio_arr = generation.cpu().numpy().squeeze()
            sf.write(output_path, audio_arr, model.config.sampling_rate)
        except Exception as exc:  # pragma: no cover - depends on local model/runtime
            raise ProcessingError(
                "Indic Parler-TTS failed. Confirm Hugging Face model access, token, "
                f"and local runtime dependencies: {exc}"
            ) from exc

    def _voice_description(
        self, language: str, voice: str | None, gender: str | None = None
    ) -> str:
        requested_voice = (voice or "").strip()
        if requested_voice and (" " in requested_voice or "." in requested_voice):
            return requested_voice

        speakers = RECOMMENDED_TTS_SPEAKERS.get(language, {})
        speaker = requested_voice or speakers.get(gender or "female")
        if speaker:
            return (
                f"{speaker}'s voice is clear and natural, with a moderate pace and pitch. "
                "The recording is very clear audio, close-sounding, and has no background noise."
            )

        language_name = SUPPORTED_LANGUAGES.get(language, SUPPORTED_LANGUAGES["en"]).name
        who = f"{gender} " if gender else ""
        return (
            f"A native {language_name} {who}speaker delivers clear, natural speech at a moderate pace "
            "and pitch. The recording is very clear audio, close-sounding, and has no background noise."
        )

    def _demo_tone(self, text: str, output_path: Path) -> None:
        sample_rate = 22050
        seconds = min(max(len(text) / 18.0, 0.7), 4.0)
        frames = int(sample_rate * seconds)
        with wave.open(str(output_path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            for i in range(frames):
                sample = int(6000 * sin(2 * pi * 220 * i / sample_rate))
                handle.writeframesraw(sample.to_bytes(2, "little", signed=True))
