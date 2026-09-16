from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


LanguageCode = Literal["hi", "en", "ta", "te", "kn", "ml", "bn", "mr"]


@dataclass(frozen=True)
class Language:
    code: LanguageCode
    name: str
    whisper_code: str
    indic_tag: str


SUPPORTED_LANGUAGES: dict[str, Language] = {
    "hi": Language("hi", "Hindi", "hi", "hin_Deva"),
    "en": Language("en", "English", "en", "eng_Latn"),
    "ta": Language("ta", "Tamil", "ta", "tam_Taml"),
    "te": Language("te", "Telugu", "te", "tel_Telu"),
    "kn": Language("kn", "Kannada", "kn", "kan_Knda"),
    "ml": Language("ml", "Malayalam", "ml", "mal_Mlym"),
    "bn": Language("bn", "Bengali", "bn", "ben_Beng"),
    "mr": Language("mr", "Marathi", "mr", "mar_Deva"),
}

LANGUAGE_OPTIONS = [
    {"code": language.code, "name": language.name}
    for language in SUPPORTED_LANGUAGES.values()
]

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}
ALLOWED_ARTIFACTS = {"video", "srt", "vtt", "original_transcript", "translated_transcript"}

JOB_STAGES = [
    "downloading video",
    "extracting audio",
    "detecting language",
    "transcribing",
    "detecting speakers",
    "separating speech/background",
    "translating",
    "generating translated speech",
    "synchronizing audio",
    "mixing audio",
    "generating subtitles",
    "rendering video",
    "verifying output timing",
    "complete",
]
