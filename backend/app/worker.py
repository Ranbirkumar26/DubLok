from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

from app.config import get_settings
from app.db import Database
from app.pipeline.runner import PipelineRunner
from app.storage import ensure_storage

_SHOULD_STOP = False


def _stop(_signum: int, _frame: object) -> None:
    global _SHOULD_STOP
    _SHOULD_STOP = True


def _acquire_single_instance_lock(lock_path: Path) -> object:
    """Hold an OS lock for the process lifetime; a second worker exits instead of
    competing for jobs (possibly from another Python without the ML packages)."""
    handle = lock_path.open("a+")
    try:
        if sys.platform == "win32":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise SystemExit(f"Another DubLok worker already holds {lock_path}; exiting.")
    return handle


def run_worker(once: bool = False) -> None:
    settings = get_settings()
    ensure_storage(settings)
    lock = _acquire_single_instance_lock(settings.storage_root / "worker.lock")  # noqa: F841
    print(f"DubLok worker using {sys.executable}", flush=True)
    db = Database(settings)
    db.init()
    runner = PipelineRunner(settings, db)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    while not _SHOULD_STOP:
        job = db.acquire_next_job()
        if job:
            runner.run(job)
            if once:
                return
        elif once:
            return
        else:
            time.sleep(settings.worker_poll_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local video dubbing worker.")
    parser.add_argument("--once", action="store_true", help="Process one queued job and exit.")
    args = parser.parse_args()
    run_worker(once=args.once)
    return 0


if __name__ == "__main__":
    sys.exit(main())
