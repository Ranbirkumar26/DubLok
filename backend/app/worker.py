from __future__ import annotations

import argparse
import signal
import sys
import time

from app.config import get_settings
from app.db import Database
from app.pipeline.runner import PipelineRunner
from app.storage import ensure_storage

_SHOULD_STOP = False


def _stop(_signum: int, _frame: object) -> None:
    global _SHOULD_STOP
    _SHOULD_STOP = True


def run_worker(once: bool = False) -> None:
    settings = get_settings()
    ensure_storage(settings)
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
