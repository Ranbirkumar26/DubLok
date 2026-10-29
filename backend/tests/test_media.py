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


def _fit_filter(monkeypatch, tmp_path, speech: float, target: float, **kwargs) -> str:
    calls: list[list[str]] = []
    monkeypatch.setattr(media, "audio_duration", lambda _path: speech)
    monkeypatch.setattr(
        media,
        "run_command",
        lambda args, timeout=3600: calls.append(args)
        or subprocess.CompletedProcess(args, 0, "", ""),
    )
    media.fit_audio_to_segment(tmp_path / "raw.wav", tmp_path / "fit.wav", target, **kwargs)
    return calls[0][calls[0].index("-af") + 1]


def test_fit_audio_speeds_up_long_speech_to_window(monkeypatch, tmp_path) -> None:
    chain = _fit_filter(monkeypatch, tmp_path, 1.5, 1.25)

    assert "atempo=1.200000" in chain
    assert "atrim=0:1.250" in chain
    assert "loudnorm" not in chain  # per-segment loudnorm caused level jumps


def test_fit_audio_only_slows_short_speech_mildly(monkeypatch, tmp_path) -> None:
    chain = _fit_filter(monkeypatch, tmp_path, 3.6, 7.74)

    assert f"atempo={media.MIN_TEMPO:.6f}" in chain  # not 0.46x
    assert "atrim=0:7.740" in chain


def test_fit_audio_spills_into_gap_before_speeding_up(monkeypatch, tmp_path) -> None:
    chain = _fit_filter(monkeypatch, tmp_path, 4.0, 3.3, max_duration=4.04)

    assert "atempo=1.000000" in chain
    assert "atrim=0:4.000" in chain


def test_fit_audio_caps_speedup_and_fades(monkeypatch, tmp_path) -> None:
    chain = _fit_filter(monkeypatch, tmp_path, 10.0, 3.0, gain_db=4.5)

    assert f"atempo={media.MAX_SPEEDUP:.6f}" in chain
    assert "atrim=0:3.000" in chain
    assert "volume=4.50dB" in chain
    assert "afade=t=out:st=2.970" in chain


def test_match_pitch_to_source_shifts_generated_audio(monkeypatch, tmp_path) -> None:
    calls: list[list[str]] = []
    source_path = tmp_path / "source.wav"
    raw_path = tmp_path / "raw.wav"
    pitch_path = tmp_path / "pitch.wav"

    def fake_pitch(path, **_kwargs):
        return 132.0 if path == source_path else 110.0

    monkeypatch.setattr(media, "estimate_median_pitch_hz", fake_pitch)
    monkeypatch.setattr(media, "audio_sample_rate", lambda _path: 44100)
    monkeypatch.setattr(media, "ffmpeg_has_filter", lambda _name: False)
    monkeypatch.setattr(
        media,
        "run_command",
        lambda args, timeout=3600: calls.append(args)
        or subprocess.CompletedProcess(args, 0, "", ""),
    )

    result = media.match_pitch_to_source(
        source_path,
        raw_path,
        pitch_path,
        source_start=1.0,
        source_duration=2.0,
    )

    assert result == pitch_path
    filter_chain = calls[0][calls[0].index("-af") + 1]
    assert "asetrate=52920" in filter_chain
    assert "atempo=0.833333" in filter_chain


def test_pitch_shift_prefers_formant_preserving_rubberband(monkeypatch, tmp_path) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(media, "ffmpeg_has_filter", lambda name: name == "rubberband")
    monkeypatch.setattr(
        media,
        "run_command",
        lambda args, timeout=3600: calls.append(args)
        or subprocess.CompletedProcess(args, 0, "", ""),
    )

    media.pitch_shift_audio(tmp_path / "in.wav", tmp_path / "out.wav", 1.2)

    assert calls[0][calls[0].index("-af") + 1] == "rubberband=pitch=1.200000:formant=preserved"


def test_match_pitch_to_source_skips_when_pitch_is_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(media, "estimate_median_pitch_hz", lambda *_args, **_kwargs: None)

    raw_path = tmp_path / "raw.wav"
    result = media.match_pitch_to_source(
        tmp_path / "source.wav",
        raw_path,
        tmp_path / "pitch.wav",
        source_start=0.0,
        source_duration=1.0,
    )

    assert result == raw_path


def test_overlay_segments_pads_to_total_duration(monkeypatch, tmp_path) -> None:
    calls: list[list[str]] = []

    monkeypatch.setattr(
        media,
        "run_command",
        lambda args, timeout=3600: calls.append(args)
        or subprocess.CompletedProcess(args, 0, "", ""),
    )

    media.overlay_segments([(tmp_path / "seg.wav", 0.0)], tmp_path / "dub.wav", 4.0, 1.0)

    filter_complex = calls[0][calls[0].index("-filter_complex") + 1]
    assert "apad,atrim=0:4.000" in filter_complex
    # amix's default 1/N scaling made the dub quiet and drift louder over time.
    assert "normalize=0" in filter_complex


def test_mix_tracks_pads_full_replacement_to_duration(monkeypatch, tmp_path) -> None:
    calls: list[list[str]] = []

    monkeypatch.setattr(
        media,
        "run_command",
        lambda args, timeout=3600: calls.append(args)
        or subprocess.CompletedProcess(args, 0, "", ""),
    )

    media.mix_tracks(
        None,
        tmp_path / "dub.wav",
        tmp_path / "final.wav",
        duration=4.0,
        original_volume=0.0,
        translated_volume=1.0,
    )

    filter_chain = calls[0][calls[0].index("-af") + 1]
    assert "apad,atrim=0:4.000" in filter_chain


def test_verify_render_timing_rejects_long_audio(monkeypatch, tmp_path) -> None:
    audio_path = tmp_path / "final.wav"
    output_path = tmp_path / "out.mp4"

    def fake_duration(path):
        return 6.0 if path == audio_path else 5.0

    monkeypatch.setattr(media, "audio_duration", fake_duration)

    with pytest.raises(media.ProcessingError, match="does not match"):
        media.verify_render_timing(5.0, audio_path, output_path, tolerance=0.25)


def test_verify_render_timing_rejects_short_audio(monkeypatch, tmp_path) -> None:
    audio_path = tmp_path / "final.wav"
    output_path = tmp_path / "out.mp4"

    def fake_duration(path):
        return 4.0 if path == audio_path else 5.0

    monkeypatch.setattr(media, "audio_duration", fake_duration)

    with pytest.raises(media.ProcessingError, match="does not match"):
        media.verify_render_timing(5.0, audio_path, output_path, tolerance=0.25)


def test_normalize_loudness_compares_mono_without_downmix_gain(monkeypatch, tmp_path) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(
        media,
        "measure_loudness",
        lambda _path: {
            "input_i": "-21.0", "input_tp": "-2.0", "input_lra": "5.0",
            "input_thresh": "-31.0", "target_offset": "0.1",
        },
    )
    monkeypatch.setattr(
        media,
        "run_command",
        lambda args, timeout=3600: calls.append(args)
        or subprocess.CompletedProcess(args, 0, "", ""),
    )

    media.normalize_loudness(tmp_path / "mix.wav", tmp_path / "final.wav", -17.2)

    chain = calls[0][calls[0].index("-af") + 1]
    assert chain.startswith("pan=mono|c0=c0,loudnorm=I=-17.2")
    assert "measured_I=-21.0" in chain and "linear=true" in chain
    assert chain.endswith("pan=stereo|c0=c0|c1=c0")


def test_plan_piece_slots_spreads_slack_between_sentences() -> None:
    # 3 s + 1 s of speech in a 10 s window: slowed to MIN_TEMPO, rest becomes pauses.
    slots = media.plan_piece_slots([3.0, 1.0], 5.0, 10.0, 10.0)

    assert slots[0][0] == 5.0
    assert abs(slots[1][0] - (5.0 + slots[0][1])) < 1e-9
    assert abs(sum(slot for _, slot in slots) - 10.0) < 1e-9
    assert slots[0][1] > 3.0 / media.MIN_TEMPO  # includes its share of the pauses


def test_plan_piece_slots_speeds_up_overrun_and_single_piece_is_unchanged() -> None:
    fast = media.plan_piece_slots([6.0, 6.0], 0.0, 6.0, 6.0)
    assert abs(sum(slot for _, slot in fast) - 12.0 / media.MAX_SPEEDUP) < 1e-9
    assert media.plan_piece_slots([4.0], 2.0, 4.0, 4.0) == [(2.0, 4.0)]


def test_trim_silence_keeps_onset_padding(monkeypatch, tmp_path) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(media, "run_command", lambda args, timeout=3600: calls.append(args))

    media.trim_silence(tmp_path / "a.wav", tmp_path / "b.wav")

    assert f"start_silence={media.ONSET_PAD_SECONDS}" in calls[0][calls[0].index("-af") + 1]
