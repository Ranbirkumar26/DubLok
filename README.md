# DubLok

A local-first web application for translating and dubbing videos without paid APIs. The stack is FastAPI, SQLite, a polling worker process, React/Vite, FFmpeg, yt-dlp, and local open-source ML models.

The workspace was empty when this project was scaffolded, so the application is implemented as a greenfield monorepo.

## Cost

This project does not require paid APIs or subscriptions.

All core processing is performed using free/open-source software and locally running models.

Required costs:
₹0

## External Services

| Service | Required? | Cost | Purpose |
| --- | --- | --- | --- |
| Paid transcription/translation/TTS APIs | No | ₹0 | Not used |
| Hugging Face account | Optional for gated models | ₹0 | Accept free model terms and download local model files |
| YouTube / Google Drive public URLs | Optional input source | ₹0 | User-supplied public video input |

Private/authenticated links are rejected in v1. The app does not accept cookies, OAuth, cloud credentials, or paid hosted inference keys.

## Local Model Choices

| Stage | Default choice | License/cost note |
| --- | --- | --- |
| Media processing | FFmpeg / FFprobe | Free software; LGPL by default, GPL if built with GPL components |
| Public video download | yt-dlp | Unlicense; used only for supported public URLs |
| ASR + language detection | faster-whisper / Whisper models | MIT ecosystem, local CTranslate2 inference |
| VAD | Silero VAD | MIT, local inference |
| Speaker diarization | SpeechBrain embeddings + clustering | Apache-2.0 toolkit; best-effort, single-speaker fallback |
| Translation | AI4Bharat IndicTrans2 | MIT code/model family; local model downloads |
| TTS | AI4Bharat Indic Parler-TTS | Apache-2.0; free gated model access may require accepting terms |
| Background preservation | Demucs | MIT; best-effort vocal/background separation |
| Alignment | FFmpeg atempo/apad/atrim | Local segment-level retiming |

Guaranteed v1 source-language scope: English plus Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali, and Marathi. Output languages are the same eight languages.

## Repository Layout

```text
backend/
  app/
    main.py                 FastAPI app and API routes
    db.py                   SQLite schema and queue persistence
    worker.py               Separate polling worker process
    pipeline/               Media, ML, alignment, mixing, subtitles
frontend/
  src/                      React/Vite UI
storage/                    Runtime data, ignored by git
```

## Prerequisites

- Python 3.11 or newer
- Node.js 20 or newer
- FFmpeg and FFprobe on `PATH`
- Optional but recommended: NVIDIA GPU with CUDA-compatible PyTorch
- Optional: Docker Desktop

This machine already has FFmpeg, Python, Node.js, pnpm, Docker, and an RTX 4060 Laptop GPU available.

## Windows Setup

```powershell
Copy-Item .env.example .env
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt

# Optional heavy local-model stack. This downloads packages only, not every model file.
pip install -r backend\requirements-ml.txt

cd frontend
pnpm install
cd ..
```

Run the API:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --app-dir backend --reload
```

Run the worker in a second terminal:

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH="backend"
python -m app.worker
```

Run the frontend in a third terminal:

```powershell
cd frontend
pnpm dev
```

Open `http://localhost:5173`.

For production dubbing on Windows, use the project virtualenv and startup helper:

```powershell
Copy-Item .env.example .env
# Add HUGGINGFACE_HUB_TOKEN to .env after accepting free gated model terms.
.\scripts\start-production.ps1 -Install
```

After the first install, start production mode with:

```powershell
.\scripts\start-production.ps1
```

The script refuses to start production jobs while FFmpeg/FFprobe or required local model packages are missing.
When `nvidia-smi` is available, `-Install` also replaces CPU PyTorch with the CUDA 12.8 Windows wheel so `DEVICE=auto` can use the NVIDIA GPU.

## Linux Setup

```bash
cp .env.example .env
python3 -m venv .venv
. .venv/bin/activate
pip install -r backend/requirements.txt
pip install -r backend/requirements-ml.txt

cd frontend
pnpm install
cd ..
```

Run:

```bash
. .venv/bin/activate
uvicorn app.main:app --app-dir backend --reload
PYTHONPATH=backend python -m app.worker
```

## Model Downloads

The first production run downloads model files into `storage/models/huggingface` or your configured `HF_HOME`.

For gated Hugging Face models:

1. Create a free Hugging Face account.
2. Accept the relevant model terms.
3. Set `HUGGINGFACE_HUB_TOKEN` in `.env`.
4. Run `.\.venv\Scripts\python.exe backend/scripts/check_models.py`.

This is still local inference and does not add API or per-minute charges.

## Demo Mode

For UI development without model downloads, set:

```env
ENGINE_MODE=demo
```

Demo mode creates deterministic placeholder transcripts, translated text markers, and tone-based speech. It is for testing the app flow only and is not a production dubbing engine.

## API

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

The worker processes queued stages asynchronously. The frontend polls job status and never blocks a request while video processing runs.

## Security Notes

- Only allowlisted public URL hosts are accepted.
- Local/private network URLs are rejected.
- File names are normalized and stored under generated IDs.
- Internal filesystem paths are never returned to the frontend.
- Upload size and duration are configurable.
- Unsupported formats and inaccessible/private links fail with user-facing messages.
- Temporary files stay inside the per-job storage directory and can be cleaned safely.

## Testing

```bash
python -m pytest backend/tests
```

The test suite avoids heavyweight model downloads and uses deterministic fixtures for validators, queue state, subtitles, and alignment helpers.

## Docker

```bash
docker compose up --build
```

The Docker setup starts the API, worker, and frontend. GPU-enabled model acceleration still requires host-specific NVIDIA container runtime configuration.
