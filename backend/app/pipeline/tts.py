from __future__ import annotations

import wave
from math import pi, sin
from pathlib import Path

from app.config import Settings
from app.constants import SUPPORTED_LANGUAGES
from app.errors import DependencyMissingError, ProcessingError


class TTSEngine:
    def __init__(self, settings: Settings):
        self.settings = settings

    def synthesize(
        self,
        *,
        text: str,
        language: str,
        voice: str | None,
        output_path: Path,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.settings.engine_mode == "demo":
            self._demo_tone(text, output_path)
            return
        self._indic_parler(text=text, language=language, voice=voice, output_path=output_path)

    def _indic_parler(
        self,
        *,
        text: str,
        language: str,
        voice: str | None,
        output_path: Path,
    ) -> None:
        self.settings.apply_model_environment()
        try:
            import soundfile as sf
            import torch
            from parler_tts import ParlerTTSForConditionalGeneration
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise DependencyMissingError(
                "Indic Parler-TTS dependencies are missing. Run `pip install -r backend/requirements-ml.txt`."
            ) from exc

        model_name = "ai4bharat/indic-parler-tts"
        language_name = SUPPORTED_LANGUAGES.get(language, SUPPORTED_LANGUAGES["en"]).name
        speaker = voice or "a clear neutral speaker"
        prompt = f"{speaker} speaking {language_name} with natural conversational pacing."
        device = "cuda" if self.settings.device != "cpu" and torch.cuda.is_available() else "cpu"
        try:
            model = ParlerTTSForConditionalGeneration.from_pretrained(
                model_name,
                token=self.settings.huggingface_hub_token or None,
            ).to(device)
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=self.settings.huggingface_hub_token or None,
            )
            description_tokenizer = AutoTokenizer.from_pretrained(
                model.config.text_encoder._name_or_path,
                token=self.settings.huggingface_hub_token or None,
            )
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
