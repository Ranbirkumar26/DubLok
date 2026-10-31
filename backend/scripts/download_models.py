"""Download every model the production pipeline uses into HF_HOME.

Run once after install so the first dubbing job does not stall on downloads:

    python backend/scripts/download_models.py

Gated repos (Indic Parler-TTS, pyannote) need HUGGINGFACE_HUB_TOKEN in .env and the
model terms accepted on Hugging Face. A gated failure is reported, not fatal, so the
open models still download.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPOS = [
    # (repo id, gated, purpose)
    ("Systran/faster-whisper-small", False, "ASR (balanced preset)"),
    ("Systran/faster-whisper-tiny", False, "ASR fallback"),
    ("facebook/nllb-200-distilled-1.3B", False, "Translation (preferred)"),
    ("facebook/nllb-200-distilled-600M", False, "Translation fallback"),
    ("speechbrain/spkrec-ecapa-voxceleb", False, "Speaker embeddings (diarization fallback)"),
    ("ai4bharat/indic-parler-tts", True, "Text to speech"),
    ("pyannote/speaker-diarization-community-1", True, "Overlap-aware diarization"),
]


def _env_values(path: Path = Path(".env")) -> dict[str, str]:
    values: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def main() -> int:
    env = _env_values()
    os.environ.setdefault("HF_HOME", env.get("HF_HOME", "./storage/models/huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", env.get("HF_HUB_DISABLE_XET", "1"))
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or env.get("HUGGINGFACE_HUB_TOKEN") or None
    from huggingface_hub import snapshot_download

    failed = []
    for repo, gated, purpose in REPOS:
        print(f"{repo} ({purpose}) ...", flush=True)
        try:
            snapshot_download(repo_id=repo, token=token)
            print("  ok")
        except Exception as exc:  # network, 403 for gated terms, missing token
            failed.append(repo)
            hint = " Accept the model terms and set HUGGINGFACE_HUB_TOKEN." if gated else ""
            print(f"  FAILED: {type(exc).__name__}: {str(exc).splitlines()[0]}.{hint}")
    if failed:
        print("\nNot downloaded: " + ", ".join(failed))
        return 1
    print("\nAll models downloaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
