from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from app.config import Settings
from app.db import Database
from app.errors import AppError, ProcessingError, ValidationError
from app.media import (
    extract_audio,
    fit_audio_to_segment,
    mix_tracks,
    overlay_segments,
    probe_video,
    render_video,
    verify_render_timing,
)
from app.pipeline.diarization import Diarizer
from app.pipeline.downloaders import download_public_video
from app.pipeline.separation import BackgroundSeparator
from app.pipeline.subtitles import write_srt, write_vtt
from app.pipeline.transcription import Transcriber
from app.pipeline.translation import Translator
from app.pipeline.tts import TTSEngine
from app.pipeline.types import Segment, segments_from_payload, transcript_payload
from app.storage import copy_to_job_input, job_dir


class PipelineRunner:
    def __init__(self, settings: Settings, db: Database):
        self.settings = settings
        self.db = db

    def run(self, job: dict[str, Any]) -> None:
        status = job["status"]
        try:
            if status == "running_analysis":
                self.run_analysis(job)
            elif status == "running_translation":
                self.run_translation(job)
            elif status == "running_audio":
                self.run_audio(job)
            elif status == "running_render":
                self.run_render(job)
        except AppError as exc:
            self.fail(job, exc.message)
        except Exception as exc:  # pragma: no cover - final safety net
            self.fail(job, f"Unexpected processing failure: {exc}")

    def run_analysis(self, job: dict[str, Any]) -> None:
        work_dir = job_dir(self.settings, job["id"])
        source = self._source(job)

        if source["input_type"] == "url" and not source.get("local_path"):
            input_video = self._stage(
                job,
                "downloading video",
                5,
                lambda: download_public_video(
                    source["url"],
                    work_dir / "input",
                    self.settings,
                ),
            )
            self.db.update_source(source["id"], local_path=str(input_video), status="stored")
        else:
            local_path = source.get("local_path")
            if not local_path:
                raise ValidationError("The source video is missing.")
            input_video = self._stage(
                job,
                "downloading video",
                5,
                lambda: copy_to_job_input(self.settings, Path(local_path), job["id"]),
            )

        metadata = self._stage(
            job,
            "extracting metadata",
            10,
            lambda: probe_video(input_video, self.settings),
        )
        self.db.update_source(source["id"], metadata=metadata, status="ready")

        audio_path = work_dir / "audio" / "source_16k.wav"
        self._stage(job, "extracting audio", 15, lambda: extract_audio(input_video, audio_path))

        def transcribe() -> dict[str, Any]:
            return Transcriber(self.settings, job["model_preset"]).transcribe(
                audio_path,
                source_language_override=job.get("source_language_override"),
                duration=float(metadata["duration"]),
            )

        transcript = self._stage(job, "transcribing", 25, transcribe)
        if job.get("source_language_override"):
            transcript["detected_language"] = job["source_language_override"]
            transcript["confidence"] = 1.0

        def diarize() -> dict[str, Any]:
            segments = segments_from_payload(transcript)
            assigned = Diarizer(self.settings).assign_speakers(audio_path, segments)
            return transcript_payload(
                segments=assigned,
                detected_language=transcript.get("detected_language"),
                confidence=transcript.get("confidence"),
            )

        transcript = self._stage(job, "detecting speakers", 35, diarize)
        artifacts = dict(job.get("artifacts") or {})
        artifacts.update(
            {
                "input_video": str(input_video),
                "source_audio": str(audio_path),
                "metadata": metadata,
            }
        )
        self.db.update_job(
            job["id"],
            status="analysis_complete",
            current_stage="analysis complete",
            progress=40,
            transcript=transcript,
            artifacts=artifacts,
        )

    def run_translation(self, job: dict[str, Any]) -> None:
        if not job.get("transcript"):
            raise ValidationError("Analyze the video before translating.")

        translated = self._stage(
            job,
            "translating",
            55,
            lambda: Translator(self.settings).translate_payload(
                job["transcript"],
                job["target_language"],
            ),
        )
        self.db.update_job(
            job["id"],
            status="translation_ready",
            current_stage="translation ready",
            progress=60,
            translated_transcript=translated,
        )

    def run_audio(self, job: dict[str, Any]) -> None:
        translated_payload = job.get("translated_transcript")
        if not translated_payload:
            raise ValidationError("Translate or edit the transcript before generating audio.")

        artifacts = dict(job.get("artifacts") or {})
        metadata = artifacts.get("metadata") or {}
        duration = float(metadata.get("duration") or 0)
        if duration <= 0:
            raise ProcessingError("Cannot generate audio without valid media duration.")
        work_dir = job_dir(self.settings, job["id"])
        segments = segments_from_payload(translated_payload)
        tts = TTSEngine(self.settings)

        fitted_paths: list[tuple[Path, float]] = []
        for index, segment in enumerate(segments, start=1):
            text = (segment.translated_text or segment.text).strip()
            if not text:
                continue
            raw_path = work_dir / "tts" / f"{segment.id}_raw.wav"
            fit_path = work_dir / "tts" / f"{segment.id}_fit.wav"
            voice = job["voice_map"].get(segment.speaker)
            self._stage(
                job,
                f"generating translated speech {index}/{len(segments)}",
                60 + min(20, index / max(len(segments), 1) * 20),
                lambda text=text, raw_path=raw_path, voice=voice: tts.synthesize(
                    text=text,
                    language=job["target_language"],
                    voice=voice,
                    output_path=raw_path,
                ),
            )
            target_duration = min(
                segment.duration,
                max(0.05, segment.duration / max(float(job["speech_speed"]), 0.5)),
            )
            self._stage(
                job,
                "synchronizing audio",
                82,
                lambda raw_path=raw_path, fit_path=fit_path, target_duration=target_duration: (
                    fit_audio_to_segment(raw_path, fit_path, target_duration)
                ),
            )
            fitted_paths.append((fit_path, segment.start))

        dubbed_track = work_dir / "audio" / "dubbed_segments.wav"
        self._stage(
            job,
            "mixing audio",
            86,
            lambda: overlay_segments(
                fitted_paths,
                dubbed_track,
                duration,
                volume=float(job["translated_volume"]),
            ),
        )

        background_path = None
        if job["audio_mode"] == "preserve_background":
            source_audio = Path(artifacts.get("source_audio") or "")
            background_path = self._stage(
                job,
                "separating speech/background",
                88,
                lambda: BackgroundSeparator(self.settings).preserve_background(
                    source_audio,
                    work_dir / "tmp" / "demucs",
                ),
            )

        final_audio = work_dir / "audio" / "final_mix.wav"
        self._stage(
            job,
            "mixing audio",
            90,
            lambda: mix_tracks(
                background_path,
                dubbed_track,
                final_audio,
                duration=duration,
                original_volume=float(job["original_volume"]),
                translated_volume=1.0,
            ),
        )

        subtitle_artifacts = self._write_subtitles(job, segments, work_dir)
        artifacts.update(
            {
                "dubbed_audio": str(dubbed_track),
                "final_audio": str(final_audio),
                **subtitle_artifacts,
            }
        )
        self.db.update_job(
            job["id"],
            status="audio_ready",
            current_stage="audio ready",
            progress=92,
            artifacts=artifacts,
        )

    def run_render(self, job: dict[str, Any]) -> None:
        artifacts = dict(job.get("artifacts") or {})
        metadata = artifacts.get("metadata") or {}
        duration = float(metadata.get("duration") or 0)
        input_video = Path(artifacts.get("input_video") or "")
        final_audio = Path(artifacts.get("final_audio") or "")
        if not input_video.exists() or not final_audio.exists():
            raise ValidationError("Generate audio before rendering the final video.")
        output_path = job_dir(self.settings, job["id"]) / "output" / (
            f"translated_video_{job['target_language']}.mp4"
        )
        self._stage(
            job,
            "rendering video",
            96,
            lambda: render_video(input_video, final_audio, output_path, duration),
        )
        self._stage(
            job,
            "verifying output timing",
            98,
            lambda: verify_render_timing(duration, final_audio, output_path),
        )

        artifacts["video"] = str(output_path)
        self._write_transcript_json(job, artifacts)
        self.db.update_job(
            job["id"],
            status="completed",
            current_stage="complete",
            progress=100,
            artifacts=artifacts,
        )

    def fail(self, job: dict[str, Any], message: str) -> None:
        self.db.update_job(
            job["id"],
            status="failed",
            current_stage="failed",
            error=message,
        )

    def _stage(
        self,
        job: dict[str, Any],
        name: str,
        progress: float,
        fn: Callable[[], Any],
    ) -> Any:
        fresh = self.db.update_job(job["id"], current_stage=name, progress=progress)
        start = time.perf_counter()
        try:
            result = fn()
        except Exception as exc:
            self.db.append_stage_log(
                fresh,
                stage=name,
                status="failed",
                message=str(exc),
                seconds=time.perf_counter() - start,
            )
            raise
        self.db.append_stage_log(
            fresh,
            stage=name,
            status="complete",
            seconds=time.perf_counter() - start,
        )
        return result

    def _source(self, job: dict[str, Any]) -> dict[str, Any]:
        source = self.db.get_source(job["source_id"])
        if not source:
            raise ValidationError("The source video no longer exists.")
        return source

    def _write_subtitles(
        self,
        job: dict[str, Any],
        segments: list[Segment],
        work_dir: Path,
    ) -> dict[str, str]:
        if job["subtitle_mode"] == "none":
            return {}
        srt_path = work_dir / "subtitles" / f"subtitles_{job['target_language']}.srt"
        vtt_path = work_dir / "subtitles" / f"subtitles_{job['target_language']}.vtt"
        write_srt(segments, srt_path, job["subtitle_mode"])
        write_vtt(segments, vtt_path, job["subtitle_mode"])
        return {"srt": str(srt_path), "vtt": str(vtt_path)}

    def _write_transcript_json(self, job: dict[str, Any], artifacts: dict[str, Any]) -> None:
        work_dir = job_dir(self.settings, job["id"])
        original_path = work_dir / "output" / "original_transcript.json"
        translated_path = work_dir / "output" / "translated_transcript.json"
        original_path.write_text(
            json.dumps(job.get("transcript") or {}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        translated_path.write_text(
            json.dumps(job.get("translated_transcript") or {}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        artifacts["original_transcript"] = str(original_path)
        artifacts["translated_transcript"] = str(translated_path)
