# DubLok

DubLok is a local-first video dubbing dashboard. It lets you upload a local video or submit a supported public URL, extract speech, review the transcript, translate it, generate replacement speech, and render a final MP4 where the original audio is suppressed and the generated voice is used instead.

The project is designed to be reproducible without paid APIs. All inference runs locally on your machine through open-source packages and locally downloaded models.

## What You Get

- Upload-first React/Vite dashboard for local video files.
- Secondary public URL input for YouTube and Google Drive links.
- FastAPI backend with SQLite job storage.
- Separate Python worker process for long-running media and ML stages.
- FFmpeg/FFprobe media handling.
- Local ASR/language detection through faster-whisper.
- Local translation through IndicTrans2.
- Local TTS through Indic Parler-TTS.
- Segment-level audio fitting so generated speech is stretched, trimmed, or padded to the source timestamps.
- Full Replacement mode by default: the final video maps original video plus generated audio only.
- Downloadable final MP4, generated audio, SRT, VTT, original transcript, and translated transcript.

## Cost And Privacy

DubLok does not use OpenAI, Google Cloud, AWS, Azure, ElevenLabs, Deepgram, AssemblyAI, or any paid hosted inference service.

| Item | Required? | Cost | Notes |
| --- | --- | --- | --- |
| Paid transcription API | No | INR 0 | Not used |
| Paid translation API | No | INR 0 | Not used |
| Paid TTS API | No | INR 0 | Not used |
| Hugging Face account | Sometimes | INR 0 | Needed only to accept free gated model terms and download model files |
| YouTube / Google Drive public URL | Optional | INR 0 | Private/authenticated links are rejected |

Model files are downloaded to local disk. Video files, transcripts, audio, and rendered outputs stay under `storage/` unless you move them yourself.

## Local Inference Components

| Stage | Default component | Runtime note |
| --- | --- | --- |
| Media probe/extract/render | FFmpeg / FFprobe | Required command-line tools on `PATH`. |
| Public URL download | yt-dlp | Used only for supported public URLs; site support can change. |
| ASR and language detection | faster-whisper | Local Whisper inference through CTranslate2. |
| VAD | Silero VAD | Local speech region detection. |
| Speaker labels | SpeechBrain ECAPA embeddings | Best-effort clustering with single-speaker fallback. |
| Translation | AI4Bharat IndicTrans2 | Local model; gated Hugging Face access may be required. |
| TTS | AI4Bharat Indic Parler-TTS | Local model; gated Hugging Face access may be required. |
| Background preservation | Demucs | Optional/best-effort; Full Replacement does not require original audio mixing. |
| Timing and muxing | FFmpeg filters | Segment-level stretch, trim, pad, overlay, and final MP4 mux. |

## Supported Languages

DubLok v1 targets English plus these Indic languages:

| Language | Code |
| --- | --- |
| English | `en` |
| Hindi | `hi` |
| Tamil | `ta` |
| Telugu | `te` |
| Kannada | `kn` |
| Malayalam | `ml` |
| Bengali | `bn` |
| Marathi | `mr` |

If the detected source language already matches the selected target language, the dashboard stops after analysis and shows `Already in the requested language`. You can still choose `Replace voice anyway` to regenerate same-language speech explicitly.

## Repository Layout

```text
.
|-- backend/
|   |-- app/
|   |   |-- main.py             FastAPI routes
|   |   |-- worker.py           queue worker entry point
|   |   |-- db.py               SQLite schema and stage state
|   |   |-- media.py            FFmpeg helpers, audio fitting, render logic
|   |   `-- pipeline/           ASR, translation, TTS, diarization, subtitles
|   |-- scripts/
|   |   |-- check_models.py     production preflight
|   |   `-- dashboard_dry_run.py
|   `-- tests/
|-- frontend/
|   |-- src/
|   |   |-- App.tsx             dashboard flow
|   |   |-- api.ts              API client
|   |   `-- styles.css
|   `-- package.json
|-- scripts/
|   `-- start-production.ps1    Windows production startup helper
|-- storage/                    runtime data, ignored by git
|-- .env.example
|-- docker-compose.yml
`-- README.md
```

## Prerequisites

Required:

- Git.
- Python 3.11 or newer.
- Node.js 20 or newer.
- `pnpm`.
- FFmpeg and FFprobe available on `PATH`.

Recommended:

- Windows 10/11 or Linux.
- NVIDIA GPU with CUDA-compatible PyTorch for faster inference.
- 8 GB+ VRAM for smoother production runs.
- 16 GB+ system RAM.

CPU fallback works, but transcription, translation, and TTS will be much slower.

## Fresh Clone Setup On Windows

Open PowerShell:

```powershell
git clone https://github.com/Ranbirkumar26/DubLok.git
cd DubLok
Copy-Item .env.example .env
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r backend\requirements.txt
pip install -r backend\requirements-ml.txt
cd frontend
pnpm install
cd ..
```

If you have an NVIDIA GPU, the production helper can install the CUDA PyTorch wheel for this project virtualenv:

```powershell
.\scripts\start-production.ps1 -Install
```

That command also installs Python packages and frontend dependencies if they are missing.

## Fresh Clone Setup On Linux

```bash
git clone https://github.com/Ranbirkumar26/DubLok.git
cd DubLok
cp .env.example .env
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
pip install -r backend/requirements-ml.txt
cd frontend
pnpm install
cd ..
```

Install FFmpeg with your package manager if it is not already present:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

## Environment Configuration

Start from `.env.example`:

```env
ENGINE_MODE=production
DEVICE=auto
MODEL_PRESET=balanced
HF_HOME=./storage/models/huggingface
HF_HUB_DISABLE_XET=1
BACKGROUND_MODE=full_replacement
TRANSLATED_VOLUME=1.00
```

Important values:

| Variable | Recommended value | Meaning |
| --- | --- | --- |
| `ENGINE_MODE` | `production` | Real local inference. Use `demo` only for UI testing. |
| `DEVICE` | `auto` | Uses CUDA when available, otherwise CPU. |
| `MODEL_PRESET` | `fast`, `balanced`, or `quality` | Controls default model size/quality tradeoff. |
| `HF_HOME` | `./storage/models/huggingface` | Local model cache path. |
| `HUGGINGFACE_HUB_TOKEN` | your free read token | Required only for gated models after accepting terms. |
| `BACKGROUND_MODE` | `full_replacement` | Default mode; original audio is replaced. |
| `MAX_UPLOAD_MB` | `750` | Upload size limit. |
| `MAX_DURATION_SECONDS` | `900` | Video duration limit. |

Do not commit `.env`. It can contain your Hugging Face token and is ignored by git.

## Hugging Face Model Access

Some AI4Bharat models are gated. Gated does not mean paid; it means your free Hugging Face account must accept the model terms before download.

Use this flow:

1. Create or sign in to a free Hugging Face account.
2. Visit the required model pages and accept the terms for your account.
3. Create a read token from Hugging Face settings.
4. Put it in `.env`:

```env
HUGGINGFACE_HUB_TOKEN=hf_your_token_here
```

5. Run the production preflight:

```powershell
.\.venv\Scripts\python.exe backend\scripts\check_models.py
```

If model access is not accepted for that account, translation or TTS may fail with a `403` or gated-model error. Accept the terms, keep the token in `.env`, and retry.

## Production Startup On Windows

Use the helper:

```powershell
.\scripts\start-production.ps1
```

It does the following:

- Creates `.venv` if missing.
- Creates `.env` from `.env.example` if missing.
- Loads environment variables from `.env`.
- Runs `backend/scripts/check_models.py`.
- Stops old DubLok API, worker, and Vite processes.
- Starts API, worker, and frontend in hidden PowerShell processes.
- Writes logs to `storage/logs/`.
- Confirms API health.

Open:

```text
http://127.0.0.1:5173/
```

Logs:

```text
storage/logs/api.log
storage/logs/worker.log
storage/logs/frontend.log
```

## Manual Startup

Use three terminals.

Terminal 1, API:

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH="backend"
uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload
```

Terminal 2, worker:

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH="backend"
python -m app.worker
```

Terminal 3, frontend:

```powershell
cd frontend
pnpm dev --host 127.0.0.1 --port 5173
```

On Linux, replace the activation and environment syntax:

```bash
. .venv/bin/activate
export PYTHONPATH=backend
uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload
```

```bash
. .venv/bin/activate
export PYTHONPATH=backend
python -m app.worker
```

```bash
cd frontend
pnpm dev --host 127.0.0.1 --port 5173
```

## Dashboard Workflow

1. Open `http://127.0.0.1:5173/`.
2. Select a local video file.
3. Choose target language and model/settings.
4. Start upload.
5. Wait for analysis to complete.
6. Review detected metadata, language, transcript segments, and stage log.
7. If the transcript is wrong, mark `Transcript flagged`, edit the text, save it, then mark it fixed.
8. Click `Continue to final video`.
9. The frontend runs translation, audio generation, and render in sequence.
10. Preview or download:
    - final merged MP4,
    - generated audio,
    - SRT,
    - VTT,
    - original transcript,
    - translated transcript.

For URL input, paste a supported public URL in the secondary URL panel. Private links, authenticated links, cookies, and OAuth are intentionally not supported in v1.

## Audio Replacement Behavior

Full Replacement is the default. In this mode:

- original audio is not mixed into the final output,
- the render maps original video stream `0:v:0`,
- the render maps generated audio stream `1:a:0`,
- the final MP4 should contain one video stream and one generated audio stream.

Each generated TTS segment is fitted to its source transcript segment duration, overlaid at the original start timestamp, and trimmed to the full source video duration.

## Demo Mode

Demo mode is only for dashboard development:

```env
ENGINE_MODE=demo
```

It uses placeholder transcript/translation text and generated tones. It is not real dubbing. For real transcription, translation, TTS, and render verification, use:

```env
ENGINE_MODE=production
```

## Verification Checklist

Run these before considering a setup reproducible.

Backend tests:

```powershell
.\.venv\Scripts\python.exe -m pytest backend\tests
```

Frontend build:

```powershell
cd frontend
pnpm build
cd ..
```

Production preflight:

```powershell
.\.venv\Scripts\python.exe backend\scripts\check_models.py
```

API health after startup:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

Verify an output MP4:

```powershell
ffprobe -hide_banner -v error -show_streams -show_format storage\jobs\<job_id>\output\translated_video_hi.mp4
```

Expected output properties:

- one video stream,
- one audio stream,
- audio codec present,
- no original audio stream retained,
- final video duration matches source duration within a small tolerance.

## Dry Run Script

The dashboard dry-run script exercises the same API flow the frontend uses.

Example with a local video:

```powershell
.\.venv\Scripts\python.exe backend\scripts\dashboard_dry_run.py `
  --input "C:\path\to\video.mp4" `
  --target-language hi
```

If gated translation access is unavailable, you can test the review/edit path with a JSON file containing reviewed translated segments:

```powershell
.\.venv\Scripts\python.exe backend\scripts\dashboard_dry_run.py `
  --input "C:\path\to\video.mp4" `
  --target-language hi `
  --reviewed-translations storage\tmp\reviewed_hi.json
```

The script fails if translated segment text is blank before audio generation.

## API Overview

```text
POST   /api/video/upload
POST   /api/video/url
POST   /api/jobs
GET    /api/jobs/{job_id}
GET    /api/jobs/{job_id}/transcript
PUT    /api/jobs/{job_id}/transcript
POST   /api/jobs/{job_id}/translate
POST   /api/jobs/{job_id}/generate-audio
POST   /api/jobs/{job_id}/render
GET    /api/jobs/{job_id}/result
GET    /api/jobs/{job_id}/download/{artifact}
```

Download artifacts include:

```text
video
final_audio
srt
vtt
original_transcript
translated_transcript
```

## Docker

```bash
docker compose up --build
```

Docker starts the API, worker, and frontend. For GPU acceleration inside Docker, configure the NVIDIA container runtime on the host. Without GPU passthrough, inference runs on CPU.

Docker is useful for repeatable service startup, but the local `.venv` setup is the recommended path for Windows GPU development.

## Troubleshooting

`403` or gated-model error:

- Accept the model terms on Hugging Face using the same account that created your token.
- Put the token in `.env` as `HUGGINGFACE_HUB_TOKEN`.
- Restart the stack.

`ffmpeg` or `ffprobe` missing:

- Install FFmpeg.
- Confirm both commands work from a new terminal:

```powershell
ffmpeg -version
ffprobe -version
```

Dashboard says demo mode:

- Set `ENGINE_MODE=production` in `.env`.
- Restart API and worker.

CUDA not used:

- Confirm `nvidia-smi` works.
- Re-run:

```powershell
.\scripts\start-production.ps1 -Install
```

Translation fails but ASR works:

- Most likely gated IndicTrans2 access is not accepted for the current Hugging Face token.
- Use the manual translated transcript review path to continue, or accept model terms and retry.

Generated media is missing:

- Check `storage/logs/worker.log`.
- Confirm the worker is running.
- Confirm the job did not fail during translation or TTS.

URL download fails:

- Confirm the link is public.
- Use local upload for the most reliable path.
- Site support can change because public URL download depends on `yt-dlp`.

## What Is Ignored By Git

These paths are intentionally not pushed:

```text
.env
.venv/
storage/
frontend/node_modules/
frontend/dist/
__pycache__/
.pytest_cache/
```

This keeps secrets, model caches, generated audio/video, SQLite data, and build artifacts out of GitHub.

## Known Limitations

- First production run can be slow because model files must download.
- Free gated Hugging Face model access must be accepted manually.
- CPU inference is much slower than GPU inference.
- Diarization and background preservation are best-effort.
- Public URL support is best-effort and may change as video sites change.
- Same-language dubbing requires explicit user confirmation.

## Reproducible Success Criteria

A fresh clone is considered working when all of these pass:

1. `.env` exists and `ENGINE_MODE=production`.
2. `ffmpeg -version` and `ffprobe -version` work.
3. `backend/scripts/check_models.py` passes.
4. `python -m pytest backend/tests` passes.
5. `pnpm build` passes in `frontend/`.
6. `scripts/start-production.ps1` starts API, worker, and frontend.
7. Dashboard upload reaches transcript review.
8. `Continue to final video` produces a downloadable MP4.
9. `ffprobe` shows one video stream and one generated audio stream.
10. Final video duration matches source duration within tolerance.
