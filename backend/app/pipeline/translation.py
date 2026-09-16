from __future__ import annotations

from app.config import Settings
from app.constants import SUPPORTED_LANGUAGES
from app.errors import DependencyMissingError, ProcessingError, ValidationError
from app.pipeline.types import Segment, transcript_payload

MODEL_BY_DIRECTION = {
    "en_to_indic": "ai4bharat/indictrans2-en-indic-1B",
    "indic_to_en": "ai4bharat/indictrans2-indic-en-1B",
    "indic_to_indic": "ai4bharat/indictrans2-indic-indic-1B",
}


class Translator:
    def __init__(self, settings: Settings):
        self.settings = settings

    def translate_payload(self, payload: dict, target_language: str) -> dict:
        source_language = payload.get("detected_language") or "en"
        if source_language not in SUPPORTED_LANGUAGES:
            raise ValidationError(
                "Unsupported source language for v1. Supported sources are English, "
                "Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali, and Marathi."
            )
        if target_language not in SUPPORTED_LANGUAGES:
            raise ValidationError("Unsupported target language.")
        segments = [
            Segment.from_dict(segment)
            for segment in payload.get("segments", [])
        ]
        translated_segments = self.translate_segments(segments, source_language, target_language)
        return transcript_payload(
            segments=translated_segments,
            detected_language=source_language,
            confidence=payload.get("confidence"),
        )

    def translate_segments(
        self,
        segments: list[Segment],
        source_language: str,
        target_language: str,
    ) -> list[Segment]:
        if source_language == target_language:
            for segment in segments:
                segment.translated_text = segment.text
            return segments
        if self.settings.engine_mode == "demo":
            target_name = SUPPORTED_LANGUAGES[target_language].name
            for segment in segments:
                segment.translated_text = f"[{target_name}] {segment.text}"
            return segments
        translations = self._indictrans2(
            [segment.text for segment in segments],
            source_language,
            target_language,
        )
        for segment, translated in zip(segments, translations):
            segment.translated_text = translated
        return segments

    def _indictrans2(
        self,
        texts: list[str],
        source_language: str,
        target_language: str,
    ) -> list[str]:
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise DependencyMissingError(
                "IndicTrans2 dependencies are missing. Run `pip install -r backend/requirements-ml.txt`."
            ) from exc

        direction = self._direction(source_language, target_language)
        model_name = MODEL_BY_DIRECTION[direction]
        src_tag = SUPPORTED_LANGUAGES[source_language].indic_tag
        tgt_tag = SUPPORTED_LANGUAGES[target_language].indic_tag
        device = self._torch_device(torch)

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                token=self.settings.huggingface_hub_token or None,
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                token=self.settings.huggingface_hub_token or None,
            ).to(device)
            model.eval()
            prepared = [f"{src_tag} {tgt_tag} {text}" for text in texts]
            encoded = tokenizer(
                prepared,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                output = model.generate(
                    **encoded,
                    max_new_tokens=256,
                    num_beams=5,
                    length_penalty=1.0,
                )
            return [
                item.strip()
                for item in tokenizer.batch_decode(output, skip_special_tokens=True)
            ]
        except Exception as exc:  # pragma: no cover - depends on local model/runtime
            raise ProcessingError(
                "IndicTrans2 translation failed. Confirm model access and local dependencies: "
                f"{exc}"
            ) from exc

    @staticmethod
    def _direction(source_language: str, target_language: str) -> str:
        if source_language == "en" and target_language != "en":
            return "en_to_indic"
        if source_language != "en" and target_language == "en":
            return "indic_to_en"
        return "indic_to_indic"

    def _torch_device(self, torch_module: object) -> str:
        if self.settings.device == "cpu":
            return "cpu"
        cuda = getattr(torch_module, "cuda", None)
        if self.settings.device == "cuda" or (cuda and cuda.is_available()):
            return "cuda"
        return "cpu"
