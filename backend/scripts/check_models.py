from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path


REQUIRED_COMMANDS = ["ffmpeg", "ffprobe"]
PRODUCTION_PACKAGES = [
    ("faster_whisper", "ASR"),
    ("transformers", "Translation/TTS"),
    ("torch", "Local model runtime"),
    ("torchaudio", "Diarization runtime"),
    ("soundfile", "TTS audio writing"),
    ("speechbrain", "Speaker diarization"),
    ("demucs", "Background preservation"),
    ("parler_tts", "Indic Parler-TTS"),
]


def _env_file_values(path: Path = Path(".env")) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _setting(name: str, default: str = "") -> str:
    return os.environ.get(name) or _env_file_values().get(name) or default


def _torch_device_status() -> str | None:
    if importlib.util.find_spec("torch") is None:
        return None
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on local runtime
        return f"torch import failed: {exc}"
    if torch.cuda.is_available():
        return f"cuda: {torch.cuda.get_device_name(0)}"
    return "cpu fallback"


def main() -> int:
    ok = True
    mode = _setting("ENGINE_MODE", "production").lower()
    production = mode == "production"

    print(f"Engine mode: {mode}")
    print("Command checks")
    for command in REQUIRED_COMMANDS:
        found = shutil.which(command)
        print(f"  {command}: {'ok' if found else 'missing'}")
        ok = ok and bool(found)

    print("\nPython package checks")
    missing_packages: list[str] = []
    for package, purpose in PRODUCTION_PACKAGES:
        found = importlib.util.find_spec(package) is not None
        print(f"  {package}: {'ok' if found else 'missing'} ({purpose})")
        if production and not found:
            missing_packages.append(package)

    device_status = _torch_device_status()
    if device_status:
        print(f"\nTorch device: {device_status}")

    if production and not _setting("HUGGINGFACE_HUB_TOKEN"):
        print("\nWarning: HUGGINGFACE_HUB_TOKEN is empty. Gated free models may fail to download.")

    if not ok:
        print("\nInstall FFmpeg/FFprobe before running production jobs.")
        return 1
    if missing_packages:
        print(
            "\nInstall missing production Python packages with "
            "`pip install -r backend/requirements-ml.txt` before production jobs."
        )
        return 1
    print("\nCore media dependencies are available.")
    if production:
        print("Production model package checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
