from __future__ import annotations

from pathlib import Path

from app.pipeline.types import Segment


def _srt_timestamp(seconds: float) -> str:
    ms_total = max(0, int(round(seconds * 1000)))
    hours, rem = divmod(ms_total, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{ms:03}"


def _vtt_timestamp(seconds: float) -> str:
    return _srt_timestamp(seconds).replace(",", ".")


def subtitle_text(segment: Segment, mode: str) -> str:
    translated = segment.translated_text or ""
    if mode == "original":
        return segment.text
    if mode == "translated":
        return translated
    if mode == "dual":
        return f"{segment.text}\n{translated}".strip()
    return ""


def write_srt(segments: list[Segment], path: Path, mode: str) -> None:
    lines: list[str] = []
    index = 1
    for segment in segments:
        text = subtitle_text(segment, mode).strip()
        if not text:
            continue
        lines.extend(
            [
                str(index),
                f"{_srt_timestamp(segment.start)} --> {_srt_timestamp(segment.end)}",
                text,
                "",
            ]
        )
        index += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_vtt(segments: list[Segment], path: Path, mode: str) -> None:
    lines = ["WEBVTT", ""]
    for segment in segments:
        text = subtitle_text(segment, mode).strip()
        if not text:
            continue
        lines.extend(
            [
                f"{_vtt_timestamp(segment.start)} --> {_vtt_timestamp(segment.end)}",
                text,
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
