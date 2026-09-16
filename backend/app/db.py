from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from app.config import Settings


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=_json_default)


def loads(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    return json.loads(value)


class Database:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.path = settings.sqlite_path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sources (
                    id TEXT PRIMARY KEY,
                    input_type TEXT NOT NULL,
                    original_name TEXT,
                    local_path TEXT,
                    url TEXT,
                    url_kind TEXT,
                    mime_type TEXT,
                    status TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL REFERENCES sources(id),
                    status TEXT NOT NULL,
                    current_stage TEXT,
                    progress REAL NOT NULL DEFAULT 0,
                    error TEXT,
                    target_language TEXT NOT NULL,
                    source_language_override TEXT,
                    model_preset TEXT NOT NULL,
                    audio_mode TEXT NOT NULL,
                    subtitle_mode TEXT NOT NULL,
                    speech_speed REAL NOT NULL,
                    original_volume REAL NOT NULL,
                    translated_volume REAL NOT NULL,
                    voice_map_json TEXT NOT NULL DEFAULT '{}',
                    transcript_json TEXT,
                    translated_transcript_json TEXT,
                    artifacts_json TEXT NOT NULL DEFAULT '{}',
                    stage_log_json TEXT NOT NULL DEFAULT '[]',
                    config_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_jobs_status_updated
                    ON jobs(status, updated_at);
                """
            )

    def create_source(
        self,
        *,
        source_id: str,
        input_type: str,
        original_name: str | None,
        local_path: str | None,
        url: str | None,
        url_kind: str | None,
        mime_type: str | None,
        status: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = utc_now()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO sources (
                    id, input_type, original_name, local_path, url, url_kind, mime_type,
                    status, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_id,
                    input_type,
                    original_name,
                    local_path,
                    url,
                    url_kind,
                    mime_type,
                    status,
                    dumps(metadata or {}),
                    now,
                    now,
                ),
            )
        return self.get_source(source_id)

    def get_source(self, source_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM sources WHERE id = ?", (source_id,)).fetchone()
        return self._source_from_row(row) if row else None

    def update_source(self, source_id: str, **fields: Any) -> dict[str, Any]:
        if not fields:
            source = self.get_source(source_id)
            if not source:
                raise KeyError(source_id)
            return source
        fields["updated_at"] = utc_now()
        values: list[Any] = []
        parts: list[str] = []
        for key, value in fields.items():
            if key == "metadata":
                key = "metadata_json"
                value = dumps(value)
            parts.append(f"{key} = ?")
            values.append(value)
        values.append(source_id)
        with self.connect() as conn:
            conn.execute(f"UPDATE sources SET {', '.join(parts)} WHERE id = ?", values)
        source = self.get_source(source_id)
        if not source:
            raise KeyError(source_id)
        return source

    def create_job(self, *, job_id: str, source_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        now = utc_now()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    id, source_id, status, current_stage, progress, target_language,
                    source_language_override, model_preset, audio_mode, subtitle_mode,
                    speech_speed, original_volume, translated_volume, voice_map_json,
                    artifacts_json, stage_log_json, config_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    source_id,
                    "queued_analysis",
                    "queued analysis",
                    0,
                    payload["target_language"],
                    payload.get("source_language_override"),
                    payload["model_preset"],
                    payload["audio_mode"],
                    payload["subtitle_mode"],
                    payload["speech_speed"],
                    payload["original_volume"],
                    payload["translated_volume"],
                    dumps(payload.get("voice_map") or {}),
                    dumps({}),
                    dumps([]),
                    dumps(payload),
                    now,
                    now,
                ),
            )
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._job_from_row(row) if row else None

    def update_job(self, job_id: str, **fields: Any) -> dict[str, Any]:
        if not fields:
            job = self.get_job(job_id)
            if not job:
                raise KeyError(job_id)
            return job
        fields["updated_at"] = utc_now()
        values: list[Any] = []
        parts: list[str] = []
        for key, value in fields.items():
            if key in {
                "voice_map",
                "transcript",
                "translated_transcript",
                "artifacts",
                "stage_log",
                "config",
            }:
                key = f"{key}_json"
                value = dumps(value)
            parts.append(f"{key} = ?")
            values.append(value)
        values.append(job_id)
        with self.connect() as conn:
            conn.execute(f"UPDATE jobs SET {', '.join(parts)} WHERE id = ?", values)
        job = self.get_job(job_id)
        if not job:
            raise KeyError(job_id)
        return job

    def append_stage_log(
        self,
        job: dict[str, Any],
        *,
        stage: str,
        status: str,
        message: str | None = None,
        seconds: float | None = None,
    ) -> dict[str, Any]:
        log = list(job.get("stage_log") or [])
        entry = {
            "stage": stage,
            "status": status,
            "message": message,
            "seconds": seconds,
            "at": utc_now(),
        }
        log.append(entry)
        return self.update_job(job["id"], stage_log=log)

    def acquire_next_job(self) -> dict[str, Any] | None:
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT * FROM jobs
                WHERE status IN ('queued_analysis', 'queued_translation', 'queued_audio', 'queued_render')
                ORDER BY updated_at ASC
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                return None
            status = row["status"]
            running = status.replace("queued_", "running_", 1)
            stage = running.replace("running_", "", 1).replace("_", " ")
            conn.execute(
                """
                UPDATE jobs
                SET status = ?, current_stage = ?, updated_at = ?
                WHERE id = ?
                """,
                (running, stage, utc_now(), row["id"]),
            )
        return self.get_job(row["id"])

    def _source_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "input_type": row["input_type"],
            "original_name": row["original_name"],
            "local_path": row["local_path"],
            "url": row["url"],
            "url_kind": row["url_kind"],
            "mime_type": row["mime_type"],
            "status": row["status"],
            "metadata": loads(row["metadata_json"], {}),
            "error": row["error"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _job_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "source_id": row["source_id"],
            "status": row["status"],
            "current_stage": row["current_stage"],
            "progress": row["progress"],
            "error": row["error"],
            "target_language": row["target_language"],
            "source_language_override": row["source_language_override"],
            "model_preset": row["model_preset"],
            "audio_mode": row["audio_mode"],
            "subtitle_mode": row["subtitle_mode"],
            "speech_speed": row["speech_speed"],
            "original_volume": row["original_volume"],
            "translated_volume": row["translated_volume"],
            "voice_map": loads(row["voice_map_json"], {}),
            "transcript": loads(row["transcript_json"], None),
            "translated_transcript": loads(row["translated_transcript_json"], None),
            "artifacts": loads(row["artifacts_json"], {}),
            "stage_log": loads(row["stage_log_json"], []),
            "config": loads(row["config_json"], {}),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
