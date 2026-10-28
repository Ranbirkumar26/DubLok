from __future__ import annotations

import re
from pathlib import Path

from app.config import Settings
from app.constants import SUPPORTED_LANGUAGES
from app.errors import DependencyMissingError, ProcessingError, ValidationError
from app.pipeline.types import Segment, transcript_payload

MODEL_BY_DIRECTION = {
    "en_to_indic": "ai4bharat/indictrans2-en-indic-1B",
    "indic_to_en": "ai4bharat/indictrans2-indic-en-1B",
    "indic_to_indic": "ai4bharat/indictrans2-indic-indic-1B",
}

NLLB_FALLBACK_MODEL = "facebook/nllb-200-distilled-600M"

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.?!।])\s+")


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_BOUNDARY.split(text) if part.strip()]


PAIR_FALLBACK_MODELS = {
    ("en", "hi"): "Helsinki-NLP/opus-mt-en-hi",
    ("hi", "en"): "Helsinki-NLP/opus-mt-hi-en",
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
        # Seq2seq MT drops sentences from multi-sentence inputs, so translate one
        # sentence at a time and join the results back per segment.
        sentence_lists = [split_sentences(segment.text) for segment in segments]
        texts = [sentence for sentences in sentence_lists for sentence in sentences]
        translations = self._translate_texts(texts, source_language, target_language)
        position = 0
        for segment, sentences in zip(segments, sentence_lists):
            parts = translations[position : position + len(sentences)]
            position += len(sentences)
            segment.translated_text = " ".join(part for part in parts if part).strip()
        return segments

    def _translate_texts(
        self,
        texts: list[str],
        source_language: str,
        target_language: str,
    ) -> list[str]:
        if not texts:
            return []
        model_name = MODEL_BY_DIRECTION[self._direction(source_language, target_language)]
        if not self._model_is_cached(model_name):
            return self._fallback_translate(texts, source_language, target_language)
        try:
            return self._indictrans2(texts, source_language, target_language)
        except ProcessingError as indic_error:
            try:
                return self._fallback_translate(texts, source_language, target_language)
            except ProcessingError as fallback_error:
                raise ProcessingError(
                    f"{indic_error.message}\nLocal fallback also failed: "
                    f"{fallback_error.message}"
                ) from fallback_error

    def _indictrans2(
        self,
        texts: list[str],
        source_language: str,
        target_language: str,
    ) -> list[str]:
        self.settings.apply_model_environment()
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

    def _nllb(
        self,
        texts: list[str],
        source_language: str,
        target_language: str,
    ) -> list[str]:
        self.settings.apply_model_environment()
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise DependencyMissingError(
                "NLLB fallback dependencies are missing. Run `pip install -r backend/requirements-ml.txt`."
            ) from exc

        source_tag = SUPPORTED_LANGUAGES[source_language].indic_tag
        target_tag = SUPPORTED_LANGUAGES[target_language].indic_tag
        device = self._torch_device(torch)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                NLLB_FALLBACK_MODEL,
                token=self.settings.huggingface_hub_token or None,
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(
                NLLB_FALLBACK_MODEL,
                token=self.settings.huggingface_hub_token or None,
            ).to(device)
            tokenizer.src_lang = source_tag
            encoded = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_tag)
            with torch.no_grad():
                output = model.generate(
                    **encoded,
                    forced_bos_token_id=forced_bos_token_id,
                    max_new_tokens=256,
                    num_beams=4,
                )
            return [
                item.strip()
                for item in tokenizer.batch_decode(output, skip_special_tokens=True)
            ]
        except Exception as exc:  # pragma: no cover - depends on local model/runtime
            raise ProcessingError(f"NLLB fallback translation failed: {exc}") from exc

    def _fallback_translate(
        self,
        texts: list[str],
        source_language: str,
        target_language: str,
    ) -> list[str]:
        pair_model = PAIR_FALLBACK_MODELS.get((source_language, target_language))
        if pair_model:
            try:
                return self._plain_seq2seq(texts, pair_model)
            except ProcessingError:
                pass
        return self._nllb(texts, source_language, target_language)

    def _plain_seq2seq(self, texts: list[str], model_name: str) -> list[str]:
        self.settings.apply_model_environment()
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise DependencyMissingError(
                "Translation fallback dependencies are missing. Run `pip install -r backend/requirements-ml.txt`."
            ) from exc

        device = self._torch_device(torch)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=self.settings.huggingface_hub_token or None,
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                token=self.settings.huggingface_hub_token or None,
            ).to(device)
            encoded = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                output = model.generate(**encoded, max_new_tokens=256, num_beams=4)
            return [
                item.strip()
                for item in tokenizer.batch_decode(output, skip_special_tokens=True)
            ]
        except Exception as exc:  # pragma: no cover - depends on local model/runtime
            raise ProcessingError(f"Pair fallback translation failed: {exc}") from exc

    def _model_is_cached(self, model_name: str) -> bool:
        cache_dir = Path(self.settings.hf_home) / "hub" / f"models--{model_name.replace('/', '--')}"
        snapshots_dir = cache_dir / "snapshots"
        return snapshots_dir.exists() and any(snapshots_dir.iterdir())

    def _torch_device(self, torch_module: object) -> str:
        if self.settings.device == "cpu":
            return "cpu"
        cuda = getattr(torch_module, "cuda", None)
        if self.settings.device == "cuda" or (cuda and cuda.is_available()):
            return "cuda"
        return "cpu"
