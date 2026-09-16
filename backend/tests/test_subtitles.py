from __future__ import annotations

from app.pipeline.subtitles import write_srt, write_vtt
from app.pipeline.types import Segment


def segments() -> list[Segment]:
    return [
        Segment(
            id="seg_0001",
            speaker="Speaker 1",
            start=1.2,
            end=3.45,
            text="Hello",
            translated_text="Namaste",
        )
    ]


def test_write_srt(tmp_path) -> None:
    path = tmp_path / "out.srt"
    write_srt(segments(), path, "dual")
    content = path.read_text(encoding="utf-8")
    assert "00:00:01,200 --> 00:00:03,450" in content
    assert "Hello\nNamaste" in content


def test_write_vtt(tmp_path) -> None:
    path = tmp_path / "out.vtt"
    write_vtt(segments(), path, "translated")
    content = path.read_text(encoding="utf-8")
    assert content.startswith("WEBVTT")
    assert "00:00:01.200 --> 00:00:03.450" in content
    assert "Namaste" in content
