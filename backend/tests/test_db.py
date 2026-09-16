from __future__ import annotations

from app.db import Database
from app.storage import new_id


def test_job_queue_acquires_queued_job(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    from app.config import Settings

    settings = Settings(database_url=f"sqlite:///{db_path}", storage_root=tmp_path)
    db = Database(settings)
    db.init()
    source_id = new_id("src")
    db.create_source(
        source_id=source_id,
        input_type="upload",
        original_name="video.mp4",
        local_path=str(tmp_path / "video.mp4"),
        url=None,
        url_kind=None,
        mime_type="video/mp4",
        status="ready",
    )
    job = db.create_job(
        job_id=new_id("job"),
        source_id=source_id,
        payload={
            "target_language": "hi",
            "source_language_override": None,
            "model_preset": "fast",
            "audio_mode": "full_replacement",
            "subtitle_mode": "translated",
            "speech_speed": 1.0,
            "original_volume": 0.2,
            "translated_volume": 1.0,
            "voice_map": {},
        },
    )
    acquired = db.acquire_next_job()
    assert acquired is not None
    assert acquired["id"] == job["id"]
    assert acquired["status"] == "running_analysis"
