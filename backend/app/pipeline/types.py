from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Segment:
    id: str
    speaker: str
    start: float
    end: float
    text: str
    translated_text: str | None = None
    overlap: bool = False
    words: list[dict[str, Any]] | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Segment":
        return cls(
            id=str(data["id"]),
            speaker=str(data.get("speaker") or "Speaker 1"),
            start=float(data["start"]),
            end=float(data["end"]),
            text=str(data.get("text") or ""),
            translated_text=(
                None
                if data.get("translated_text") is None
                else str(data.get("translated_text"))
            ),
            overlap=bool(data.get("overlap", False)),
            words=data.get("words") or None,
        )


SENTENCE_END = (".", "?", "!", "।", "…")

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.?!।])\s+")


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_BOUNDARY.split(text) if part.strip()]


def speech_pieces(text: str, *, min_chars: int = 12) -> list[str]:
    """Sentences to synthesise separately; fragments too short for TTS join their neighbour."""
    pieces: list[str] = []
    for sentence in split_sentences(text):
        if pieces and (len(sentence) < min_chars or len(pieces[-1]) < min_chars):
            pieces[-1] = f"{pieces[-1]} {sentence}"
        else:
            pieces.append(sentence)
    return pieces or ([text.strip()] if text.strip() else [])


_REPEATED_WORD = re.compile(r"(?<!\S)(\S+)(?:\s+\1(?!\S)){2,}")


def collapse_repeated_words(text: str) -> str:
    """Collapse a word repeated three or more times ("आप आप आप") to one; MT and TTS loop on
    some inputs. Genuine Hindi reduplication ("धीरे धीरे") has two copies and is kept."""
    return _REPEATED_WORD.sub(r"\1", text)


def merge_sentence_fragments(
    segments: list[Segment],
    *,
    max_gap: float = 1.5,
    max_duration: float = 20.0,
) -> list[Segment]:
    """Join same-speaker ASR segments that split a sentence, so each is translated whole."""
    merged: list[Segment] = []
    for segment in segments:
        previous = merged[-1] if merged else None
        if (
            previous
            and previous.speaker == segment.speaker
            and not previous.text.rstrip().endswith(SENTENCE_END)
            and segment.start - previous.end <= max_gap
            and segment.end - previous.start <= max_duration
        ):
            previous.end = segment.end
            previous.overlap = previous.overlap or segment.overlap
            previous.words = None
            previous.text = f"{previous.text.rstrip()} {segment.text.lstrip()}"
            continue
        merged.append(Segment.from_dict(segment.to_dict()))
    for index, segment in enumerate(merged, start=1):
        segment.id = f"seg_{index:04d}"
    return merged


def transcript_payload(
    *,
    segments: list[Segment],
    detected_language: str | None,
    confidence: float | None,
) -> dict[str, Any]:
    return {
        "detected_language": detected_language,
        "confidence": confidence,
        "segments": [segment.to_dict() for segment in segments],
    }


def segments_from_payload(payload: dict[str, Any] | None) -> list[Segment]:
    if not payload:
        return []
    return [Segment.from_dict(item) for item in payload.get("segments") or []]
