from __future__ import annotations

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
        )


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
