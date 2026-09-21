from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import requests


DEFAULT_OPTIONS: dict[str, Any] = {
    "target_language": "hi",
    "model_preset": "balanced",
    "audio_mode": "full_replacement",
    "subtitle_mode": "translated",
    "speech_speed": 1.0,
    "original_volume": 0.2,
    "translated_volume": 1.0,
    "voice_map": {},
}


def request_json(method: str, base_url: str, path: str, **kwargs: Any) -> dict[str, Any]:
    response = requests.request(method, f"{base_url}{path}", timeout=180, **kwargs)
    if response.status_code >= 400:
        raise RuntimeError(f"{method} {path} failed: {response.status_code} {response.text}")
    return response.json()


def wait_for_status(
    base_url: str,
    job_id: str,
    expected_statuses: set[str],
) -> dict[str, Any]:
    last_line = ""
    while True:
        job = request_json("GET", base_url, f"/api/jobs/{job_id}")
        line = f"{job['status']} | {job.get('current_stage')} | {job.get('progress')}%"
        if line != last_line:
            print(line, flush=True)
            last_line = line
        if job["status"] == "failed":
            raise RuntimeError(job.get("error") or "Job failed")
        if job["status"] in expected_statuses:
            return job
        time.sleep(3)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the dashboard upload workflow end to end.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--target-language", default=DEFAULT_OPTIONS["target_language"])
    parser.add_argument("--model-preset", default=DEFAULT_OPTIONS["model_preset"])
    parser.add_argument(
        "--reviewed-translations",
        type=Path,
        help="JSON list of reviewed translated segment text, used after analysis.",
    )
    args = parser.parse_args()

    video_path = args.video.expanduser().resolve()
    if not video_path.exists():
        raise RuntimeError(f"Video does not exist: {video_path}")

    options = {
        **DEFAULT_OPTIONS,
        "target_language": args.target_language,
        "model_preset": args.model_preset,
    }

    print(f"Uploading {video_path.name}...", flush=True)
    with video_path.open("rb") as handle:
        response = requests.post(
            f"{args.base_url}/api/video/upload",
            files={"file": (video_path.name, handle, "video/mp4")},
            timeout=180,
        )
    response.raise_for_status()
    source = response.json()
    print(f"source_id={source['source_id']}", flush=True)

    job = request_json(
        "POST",
        args.base_url,
        "/api/jobs",
        json={"source_id": source["source_id"], **options},
    )
    job_id = job["id"]
    print(f"job_id={job_id}", flush=True)

    job = wait_for_status(args.base_url, job_id, {"analysis_complete"})
    transcript = job.get("transcript") or {}
    detected_language = transcript.get("detected_language")
    print(f"detected={detected_language}", flush=True)
    print(f"segments={len(transcript.get('segments') or [])}", flush=True)

    if detected_language == options["target_language"]:
        print("same_language_stop=true", flush=True)
        return 2

    request_json(
        "PUT",
        args.base_url,
        f"/api/jobs/{job_id}/transcript?kind=original",
        json=transcript,
    )
    if args.reviewed_translations:
        reviewed_path = args.reviewed_translations.expanduser().resolve()
        translated_texts = json.loads(reviewed_path.read_text(encoding="utf-8"))
        segments = transcript.get("segments") or []
        if not isinstance(translated_texts, list) or len(translated_texts) != len(segments):
            raise RuntimeError(
                "Reviewed translations must be a JSON list matching the segment count."
            )
        translated = {
            **transcript,
            "segments": [
                {
                    **segment,
                    "translated_text": str(translated_texts[index]).strip(),
                }
                for index, segment in enumerate(segments)
            ],
        }
        job = request_json(
            "PUT",
            args.base_url,
            f"/api/jobs/{job_id}/transcript?kind=translated",
            json=translated,
        )
    else:
        request_json("POST", args.base_url, f"/api/jobs/{job_id}/translate", json=options)
        job = wait_for_status(args.base_url, job_id, {"translation_ready"})
        translated = job.get("translated_transcript") or {}
    print(f"translated_segments={len(translated.get('segments') or [])}", flush=True)

    request_json(
        "PUT",
        args.base_url,
        f"/api/jobs/{job_id}/transcript?kind=translated",
        json=translated,
    )
    request_json("POST", args.base_url, f"/api/jobs/{job_id}/generate-audio", json=options)
    job = wait_for_status(args.base_url, job_id, {"audio_ready"})
    print(f"audio_artifact={job.get('artifacts', {}).get('final_audio', {}).get('url')}", flush=True)

    request_json("POST", args.base_url, f"/api/jobs/{job_id}/render")
    job = wait_for_status(args.base_url, job_id, {"completed"})
    print(
        json.dumps(
            {
                "job_id": job_id,
                "status": job["status"],
                "duration": (job.get("metadata") or {}).get("duration"),
                "artifacts": job["artifacts"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"dry run failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
