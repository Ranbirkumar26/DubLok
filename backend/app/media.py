from __future__ import annotations

import json
import math
import shutil
import subprocess
from functools import lru_cache
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


def audio_sample_rate(path: Path) -> int:
    data = ffprobe_json(path)
    audio_stream = next(
        (stream for stream in data.get("streams", []) if stream.get("codec_type") == "audio"),
        {},
    )
    return int(audio_stream.get("sample_rate") or 44100)


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


MAX_SPEEDUP = 1.6
# Mild slow-down is inaudible and fills more of the window; 0.46x sounded robotic.
MIN_TEMPO = 0.85
SILENCE_THRESHOLD = "-50dB"
# Quiet plosive onsets (e.g. the "k" of "kya") sit near the threshold; keep a little lead-in.
ONSET_PAD_SECONDS = 0.05
# Indic Parler-TTS output is bass-heavy (~65% energy in 100-300 Hz vs ~26% in real
# speech). Tuning knob: this EQ moved the spectral centroid from ~1300 to ~1900 Hz.
TTS_EQ = "equalizer=f=180:t=q:w=1:g=-5,equalizer=f=2500:t=q:w=1.5:g=4"


def trim_silence(input_path: Path, output_path: Path) -> None:
    """Drop leading/trailing silence and shorten internal pauses to 0.15 s."""
    run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-af",
            (
                f"silenceremove=start_periods=1:start_threshold={SILENCE_THRESHOLD}:"
                f"start_silence={ONSET_PAD_SECONDS}:"
                f"stop_periods=-1:stop_duration=0.3:stop_threshold={SILENCE_THRESHOLD}:"
                "stop_silence=0.15"
            ),
            "-ar",
            "44100",
            "-ac",
            "1",
            str(output_path),
        ]
    )


def plan_piece_slots(
    naturals: list[float], start: float, duration: float, available: float
) -> list[tuple[float, float]]:
    """Place the sentences of one segment as (start, slot) pairs with one shared pace.

    All sentences are slowed (>= MIN_TEMPO) to fill the window, or sped up (<= MAX_SPEEDUP)
    when they overrun it; leftover time becomes pauses between sentences, not dead air at the end.
    ``available`` is the window plus the gap before the next segment, usable only when overrunning.
    """
    total = sum(naturals)
    if total <= 0:
        return [(start, duration)]
    if total < duration:
        tempo = max(total / duration, MIN_TEMPO)
    elif total > max(duration, available):
        tempo = min(total / available, MAX_SPEEDUP)
    else:
        tempo = 1.0
    slack = max(0.0, duration - total / tempo)
    slots: list[tuple[float, float]] = []
    cursor = start
    for natural in naturals:
        slot = natural / tempo + slack * natural / total
        slots.append((cursor, slot))
        cursor += slot
    return slots


def fit_audio_to_segment(
    input_path: Path,
    output_path: Path,
    target_duration: float,
    *,
    max_duration: float | None = None,
    base_tempo: float = 1.0,
    gain_db: float = 0.0,
) -> None:
    """Place speech in its window without slowing it down.

    Speech shorter than the window keeps natural speed and is padded. Longer speech
    may spill into the gap before the next segment (``max_duration``), and is only
    sped up (capped at MAX_SPEEDUP) when it would overrun that gap.
    """
    source_duration = audio_duration(input_path)
    if source_duration <= 0:
        create_silence(output_path, target_duration)
        return
    available = max(target_duration, max_duration or target_duration)
    tempo = max(base_tempo, 1.0)
    if source_duration / tempo < target_duration:
        tempo = max(source_duration / target_duration, MIN_TEMPO)
    elif source_duration / tempo > available:
        tempo = min(source_duration / available, MAX_SPEEDUP)
    out_duration = min(max(source_duration / tempo, target_duration), available)
    fade_out_start = max(0.0, out_duration - 0.03)
    filter_chain = (
        f"{atempo_chain(tempo)},"
        "highpass=f=70,"
        f"{TTS_EQ},"
        f"volume={gain_db:.2f}dB,"
        "afade=t=in:d=0.005,"
        f"apad,atrim=0:{out_duration:.3f},"
        f"afade=t=out:st={fade_out_start:.3f}:d=0.03,asetpts=N/SR/TB"
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


def estimate_median_pitch_hz(
    audio_path: Path,
    *,
    start_seconds: float = 0.0,
    duration_seconds: float | None = None,
) -> float | None:
    try:
        import librosa
        import numpy as np
    except ImportError:
        return None

    try:
        samples, sample_rate = librosa.load(
            str(audio_path),
            sr=16000,
            mono=True,
            offset=max(0.0, start_seconds),
            duration=duration_seconds,
        )
    except Exception:
        return None
    if samples.size < int(sample_rate * 0.08):
        return None

    try:
        f0, voiced, _ = librosa.pyin(
            samples,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C6"),
            sr=sample_rate,
        )
    except Exception:
        return None
    voiced_pitch = f0[voiced & np.isfinite(f0)]
    if voiced_pitch.size == 0:
        return None
    return float(np.median(voiced_pitch))


def active_speech_db(
    audio_path: Path,
    *,
    start_seconds: float = 0.0,
    duration_seconds: float | None = None,
) -> float | None:
    """Mean RMS (dBFS) of frames within 30 dB of the loudest frame, i.e. speech only."""
    try:
        import librosa
        import numpy as np
    except ImportError:
        return None
    try:
        samples, _ = librosa.load(
            str(audio_path),
            sr=16000,
            mono=True,
            offset=max(0.0, start_seconds),
            duration=duration_seconds,
        )
    except Exception:
        return None
    if samples.size < 1600:
        return None
    rms_db = 20 * np.log10(librosa.feature.rms(y=samples, frame_length=480, hop_length=160)[0] + 1e-6)
    active = rms_db[rms_db > rms_db.max() - 30]
    return float(active.mean()) if active.size else None


def measure_loudness(path: Path) -> dict[str, str] | None:
    """First loudnorm pass on channel 0; returns ffmpeg's measured_* values or None.

    Uses `pan=mono|c0=c0` (no gain): the mono source and the dual-mono dub then compare
    like for like. aformat/-ac downmix adds +3 dB to dual-mono, which left the dub ~3.5 dB
    under the source.
    """
    try:
        result = run_command(
            [
                "ffmpeg", "-hide_banner", "-i", str(path),
                "-af", "pan=mono|c0=c0,loudnorm=print_format=json",
                "-f", "null", "-",
            ]
        )
    except ProcessingError:
        return None
    text = result.stderr
    start = text.rfind("{")
    if start < 0:
        return None
    try:
        data = json.loads(text[start : text.rfind("}") + 1])
    except json.JSONDecodeError:
        return None
    if not math.isfinite(float(data.get("input_i", "-inf"))):
        return None
    return data


def normalize_loudness(input_path: Path, output_path: Path, target_lufs: float) -> None:
    """Two-pass EBU R128 normalisation of the whole track (linear gain, no pumping)."""
    target = clamp_float(target_lufs, -24.0, -14.0)
    measured = measure_loudness(input_path)
    loudnorm = f"loudnorm=I={target:.1f}:TP=-1.5:LRA=11"
    if measured:
        loudnorm += (
            f":measured_I={measured['input_i']}:measured_TP={measured['input_tp']}"
            f":measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}"
            f":offset={measured['target_offset']}:linear=true"
        )
    run_command(
        [
            "ffmpeg", "-y", "-i", str(input_path),
            "-af", f"pan=mono|c0=c0,{loudnorm},pan=stereo|c0=c0|c1=c0",
            "-ar", "44100", "-ac", "2", str(output_path),
        ]
    )


def match_pitch_to_source(
    source_audio_path: Path,
    generated_audio_path: Path,
    output_path: Path,
    *,
    source_start: float,
    source_duration: float,
) -> Path:
    source_pitch = estimate_median_pitch_hz(
        source_audio_path,
        start_seconds=source_start,
        duration_seconds=source_duration,
    )
    generated_pitch = estimate_median_pitch_hz(generated_audio_path)
    if not source_pitch or not generated_pitch:
        return generated_audio_path

    ratio = clamp_float(source_pitch / generated_pitch, 0.75, 1.35)
    if abs(ratio - 1.0) < 0.03:
        return generated_audio_path

    pitch_shift_audio(generated_audio_path, output_path, ratio)
    return output_path


@lru_cache
def ffmpeg_has_filter(name: str) -> bool:
    try:
        result = run_command(["ffmpeg", "-hide_banner", "-filters"], timeout=30)
    except (ProcessingError, OSError):
        return False
    return any(line.split()[1:2] == [name] for line in result.stdout.splitlines() if line.strip())


def pitch_shift_audio(input_path: Path, output_path: Path, ratio: float) -> None:
    ratio = clamp_float(ratio, 0.75, 1.35)
    if ffmpeg_has_filter("rubberband"):
        # Formant-preserving shift keeps timbre; asetrate moves formants (boomy/chipmunk).
        filter_chain = f"rubberband=pitch={ratio:.6f}:formant=preserved"
    else:
        sample_rate = audio_sample_rate(input_path)
        shifted_rate = max(8000, int(round(sample_rate * ratio)))
        filter_chain = (
            f"asetrate={shifted_rate},"
            f"aresample={sample_rate},"
            f"{atempo_chain(sample_rate / shifted_rate)}"
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
        # normalize=0: segments do not overlap, and amix's default 1/N scaling made the
        # dub quiet and louder as earlier inputs ended.
        f"{''.join(mix_inputs)}amix=inputs={len(segment_paths)}:duration=longest:normalize=0,"
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
