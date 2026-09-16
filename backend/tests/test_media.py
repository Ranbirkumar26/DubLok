from __future__ import annotations

import subprocess

import pytest

from app import media
from app.media import atempo_chain


def test_atempo_chain_keeps_values_in_ffmpeg_range() -> None:
    chain = atempo_chain(8.0)
    assert chain == "atempo=2.000000,atempo=2.000000,atempo=2.000000"


def test_atempo_chain_handles_slowdown() -> None:
    chain = atempo_chain(0.125)
    assert chain == "atempo=0.500000,atempo=0.500000,atempo=0.500000"


def test_render_video_maps_generated_audio_only(monkeypatch, tmp_path) -> None:
    calls: list[list[str]] = []

    monkeypatch.setattr(media, "require_command", lambda _command: None)
    monkeypatch.setattr(
        media,
        "run_command",
        lambda args, timeout=3600: calls.append(args)
        or subprocess.CompletedProcess(args, 0, "", ""),
    )

    media.render_video(tmp_path / "input.mp4", tmp_path / "dub.wav", tmp_path / "out.mp4", 4.0)

    maps = [calls[0][index + 1] for index, arg in enumerate(calls[0]) if arg == "-map"]
    assert maps == ["0:v:0", "1:a:0"]
    assert "0:a:0" not in maps


def test_fit_audio_to_segment_trims_to_target(monkeypatch, tmp_path) -> None:
    calls: list[list[str]] = []

    monkeypatch.setattr(media, "audio_duration", lambda _path: 2.0)
    monkeypatch.setattr(
        media,
        "run_command",
        lambda args, timeout=3600: calls.append(args)
        or subprocess.CompletedProcess(args, 0, "", ""),
    )

    media.fit_audio_to_segment(tmp_path / "raw.wav", tmp_path / "fit.wav", 1.25)

    filter_chain = calls[0][calls[0].index("-af") + 1]
    assert "atrim=0:1.250" in filter_chain


def test_verify_render_timing_rejects_long_audio(monkeypatch, tmp_path) -> None:
    audio_path = tmp_path / "final.wav"
    output_path = tmp_path / "out.mp4"

    def fake_duration(path):
        return 6.0 if path == audio_path else 5.0

    monkeypatch.setattr(media, "audio_duration", fake_duration)

    with pytest.raises(media.ProcessingError, match="longer than the source video"):
        media.verify_render_timing(5.0, audio_path, output_path, tolerance=0.25)
