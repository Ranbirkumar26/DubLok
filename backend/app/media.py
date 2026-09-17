from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

from app.config import Settings
from app.errors import DependencyMissingError, ProcessingError, ValidationError


def command_exists(command: str) -> bool:
    return shutil.which(command) is not None


def require_command(command: str) -> None:
    if not command_exists(command):
        raise DependencyMissingError(
            f"{command} is required and was not found on PATH. Install FFmpeg and try again."
        )


def run_command(args: list[str], *, timeout: int = 3600) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            args,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise ProcessingError(f"Command timed out: {' '.join(args[:3])}") from exc
    if completed.returncode != 0:
        stderr = (completed.stderr or completed.stdout or "").strip()
        raise ProcessingError(stderr[-1000:] or f"Command failed: {' '.join(args)}")
    return completed


def ffprobe_json(path: Path) -> dict[str, Any]:
    require_command("ffprobe")
    result = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        timeout=60,
    )
    return json.loads(result.stdout)


def probe_video(path: Path, settings: Settings) -> dict[str, Any]:
    data = ffprobe_json(path)
    fmt = data.get("format") or {}
    streams = data.get("streams") or []
    duration = float(fmt.get("duration") or 0)
    if duration <= 0:
        raise ValidationError("The video duration could not be detected.")
    if duration > settings.max_duration_seconds:
        raise ValidationError(
            f"Video duration {duration:.1f}s exceeds the configured "
            f"{settings.max_duration_seconds}s limit."
        )
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), {})
    fps = _parse_fps(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate"))
    return {
        "duration": duration,
        "size_bytes": int(fmt.get("size") or 0),
        "format": fmt.get("format_name"),
        "bit_rate": int(fmt.get("bit_rate") or 0),
        "video": {
            "width": video_stream.get("width"),
            "height": video_stream.get("height"),
            "fps": fps,
            "codec": video_stream.get("codec_name"),
            "pix_fmt": video_stream.get("pix_fmt"),
        },
        "audio": {
            "codec": audio_stream.get("codec_name"),
            "sample_rate": int(audio_stream.get("sample_rate") or 0),
            "channels": audio_stream.get("channels"),
            "channel_layout": audio_stream.get("channel_layout"),
        },
    }


def _parse_fps(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    if "/" in value:
        num, den = value.split("/", 1)
        den_float = float(den)
        return float(num) / den_float if den_float else None
    return float(value)


def extract_audio(video_path: Path, wav_path: Path) -> None:
    require_command("ffmpeg")
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            str(wav_path),
        ]
    )


def audio_duration(path: Path) -> float:
    data = ffprobe_json(path)
    return float((data.get("format") or {}).get("duration") or 0)


def create_silence(path: Path, duration: float, sample_rate: int = 44100) -> None:
    require_command("ffmpeg")
    path.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=r={sample_rate}:cl=mono",
            "-t",
            f"{max(duration, 0.1):.3f}",
            "-q:a",
            "9",
            "-acodec",
            "pcm_s16le",
            str(path),
        ]
    )


def atempo_chain(ratio: float) -> str:
    if ratio <= 0:
        return "atempo=1.0"
    parts: list[float] = []
    remaining = ratio
    while remaining > 2.0:
        parts.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        parts.append(0.5)
        remaining /= 0.5
    parts.append(remaining)
    return ",".join(f"atempo={part:.6f}" for part in parts)


def fit_audio_to_segment(input_path: Path, output_path: Path, target_duration: float) -> None:
    source_duration = audio_duration(input_path)
    if source_duration <= 0:
        create_silence(output_path, target_duration)
        return
    ratio = source_duration / max(target_duration, 0.05)
    filter_chain = (
        f"{atempo_chain(ratio)},"
        f"apad,atrim=0:{target_duration:.3f},asetpts=N/SR/TB"
    )
    run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-af",
            filter_chain,
            "-ar",
            "44100",
            "-ac",
            "1",
            str(output_path),
        ]
    )


def overlay_segments(
    segment_paths: list[tuple[Path, float]],
    output_path: Path,
    total_duration: float,
    volume: float,
) -> None:
    if not segment_paths:
        create_silence(output_path, total_duration)
        return

    inputs: list[str] = []
    filters: list[str] = []
    mix_inputs: list[str] = []
    for index, (path, start) in enumerate(segment_paths):
        inputs.extend(["-i", str(path)])
        label = f"a{index}"
        delay_ms = max(0, int(round(start * 1000)))
        filters.append(
            f"[{index}:a]adelay={delay_ms}|{delay_ms},volume={volume:.4f}[{label}]"
        )
        mix_inputs.append(f"[{label}]")
    filters.append(
        f"{''.join(mix_inputs)}amix=inputs={len(segment_paths)}:duration=longest,"
        f"apad,atrim=0:{total_duration:.3f},asetpts=N/SR/TB[out]"
    )
    run_command(
        [
            "ffmpeg",
            "-y",
            *inputs,
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[out]",
            "-ar",
            "44100",
            "-ac",
            "2",
            str(output_path),
        ]
    )


def mix_tracks(
    background_path: Path | None,
    dubbed_path: Path,
    output_path: Path,
    *,
    duration: float,
    original_volume: float,
    translated_volume: float,
) -> None:
    if background_path is None:
        run_command(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(dubbed_path),
                "-af",
                f"volume={translated_volume:.4f},apad,atrim=0:{duration:.3f},asetpts=N/SR/TB",
                "-ar",
                "44100",
                "-ac",
                "2",
                str(output_path),
            ]
        )
        return

    run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(background_path),
            "-i",
            str(dubbed_path),
            "-filter_complex",
            (
                f"[0:a]volume={original_volume:.4f},apad,atrim=0:{duration:.3f},asetpts=N/SR/TB[bg];"
                f"[1:a]volume={translated_volume:.4f},apad,atrim=0:{duration:.3f},asetpts=N/SR/TB[dub];"
                "[bg][dub]amix=inputs=2:duration=longest:normalize=0,"
                f"atrim=0:{duration:.3f},asetpts=N/SR/TB[out]"
            ),
            "-map",
            "[out]",
            "-ar",
            "44100",
            "-ac",
            "2",
            str(output_path),
        ]
    )


def render_video(video_path: Path, audio_path: Path, output_path: Path, duration: float) -> None:
    require_command("ffmpeg")
    try:
        run_command(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-i",
                str(audio_path),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-t",
                f"{duration:.3f}",
                str(output_path),
            ]
        )
    except ProcessingError:
        run_command(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-i",
                str(audio_path),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-preset",
                "veryfast",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-t",
                f"{duration:.3f}",
                str(output_path),
            ]
        )


def verify_render_timing(
    source_duration: float,
    final_audio_path: Path,
    output_video_path: Path,
    *,
    tolerance: float = 0.5,
) -> None:
    final_audio_duration = audio_duration(final_audio_path)
    output_duration = audio_duration(output_video_path)
    if abs(final_audio_duration - source_duration) > tolerance:
        raise ProcessingError(
            "Generated audio duration does not match the source video "
            f"({final_audio_duration:.2f}s vs {source_duration:.2f}s)."
        )
    if abs(output_duration - source_duration) > tolerance:
        raise ProcessingError(
            "Rendered video duration does not match the source video "
            f"({output_duration:.2f}s vs {source_duration:.2f}s)."
        )


def clamp_float(value: float, low: float, high: float) -> float:
    if math.isnan(value):
        return low
    return min(max(value, low), high)
