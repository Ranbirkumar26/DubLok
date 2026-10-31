# DubLok

DubLok is a local-first video dubbing studio. Give it a video, and it transcribes the speech, tells the speakers apart (including when they talk over each other), translates the text, speaks it back in a target language with a matching voice pitch, and renders a new MP4.

Everything runs on your own machine with open-source models. There are no paid APIs and no cloud inference. Authors: Ranbir Kumar, Yash Malhotra, Khushali Jagrut Gandhi (SNLP project).

## Contents

1. [How it works](#how-it-works)
2. [What you get](#what-you-get)
3. [Measured results](#measured-results)
4. [Requirements](#requirements)
5. [Quick start](#quick-start)
6. [Model downloads and Hugging Face access](#model-downloads-and-hugging-face-access)
7. [Configuration](#configuration)
8. [Running the app](#running-the-app)
9. [Using the dashboard](#using-the-dashboard)
10. [Overlapping speakers](#overlapping-speakers)
11. [Verifying your setup](#verifying-your-setup)
12. [Docker](#docker)
13. [Security and privacy](#security-and-privacy)
14. [Troubleshooting](#troubleshooting)
15. [Known limitations](#known-limitations)
16. [Repository layout](#repository-layout)

## How it works

```text
video in
  |
  v
1. Extract audio (FFmpeg)
2. Transcribe (faster-whisper + Silero VAD, word timestamps)
3. Diarize (pyannote community-1, overlap aware; falls back to SpeechBrain ECAPA)
  |
  v   [you can review and edit the transcript here]
4. Translate, one sentence at a time (NLLB-200 1.3B; IndicTrans2 if installed)
5. Speak (Indic Parler-TTS, one voice description per speaker)
6. Match pitch (FFmpeg rubberband, each line moves toward its source speaker's pitch)
7. Fit timing (shared tempo, trim or pad to the source timestamps)
8. Mix and normalise loudness (EBU R128)
9. Render MP4 (original video stream + new audio)
```

The work is split into a FastAPI web server, a SQLite job queue, and a separate worker process that runs the long ML stages. A React/Vite dashboard drives it.

## What you get

- Upload a local video, or paste a public YouTube or Google Drive link.
- Speech recognition with language detection.
- Speaker labels, with overlapped speech detected and flagged (see [Overlapping speakers](#overlapping-speakers)).
- Sentence-level translation. Translating sentence by sentence avoids the dropped sentences that multi-sentence inputs cause in sequence-to-sequence models.
- Replacement speech in the target language, with a distinct voice per speaker.
- Pitch matching and a shared speaking pace, so the dub keeps the original's rise and fall.
- Loudness evened out across the dub.
- Downloads: final MP4, generated audio, SRT, VTT, original transcript, translated transcript.
- Transcript review: edit any segment before translation, and re-run from there.

Target languages: Hindi (`hi`), Tamil (`ta`), Telugu (`te`), Kannada (`kn`), Malayalam (`ml`), Bengali (`bn`), Marathi (`mr`), and English (`en`) as a source. If the detected source language already matches the target, the dashboard stops after analysis; choose "Replace voice anyway" to regenerate same-language speech.

## Measured results

Mean of 4 short test clips against a baseline pipeline (same Whisper small and NLLB-1.3B on both sides), measured on the development machine (RTX 4060 Laptop, 8 GB). Treat them as indicative, not a benchmark.

| Metric | Baseline | DubLok |
| --- | --- | --- |
| Loudness deviation from source | 5.4 dB | 0.7 dB |
| Per-segment level error | 9.1 dB | 1.2 dB |
| Leading silence | 0.18 s | 0.03 s |
| Trailing silence | 0.32 s | 0.09 s |
| Round-trip character error rate (CER) | 43.3% | 35.1% |
| Total pipeline time | 512 s | 415 s (19% faster) |
| Median pitch error | 4.6% | 5.2% (slightly worse) |

On a separate clip of two people talking over each other, the overlap-aware diarizer found 4 overlap bursts, told the 2 speakers apart, and flagged 3 of 4 dub segments as overlapped.

## Requirements

Required:

- Git.
- Python 3.11 or newer (tested on 3.12).
- Node.js 20 or newer (tested on 24) and `pnpm` 10 or newer (pnpm 9 rejects the workspace file). Install with `npm install -g pnpm@10`.
- FFmpeg and FFprobe on `PATH`. Use a build that includes the `rubberband` filter (standard Windows "full" builds and most Linux packages do). Pitch matching uses it for formant-preserving shifts and falls back to a simpler resample method without it, which sounds worse.
- About 15 GB of free disk for model files.
- A free Hugging Face account (see below).

Recommended:

- NVIDIA GPU with 8 GB or more of VRAM. CPU works but transcription, translation and speech are much slower.
- 16 GB or more of RAM.

## Quick start

Windows (PowerShell):

```powershell
git clone https://github.com/Ranbirkumar26/DubLok.git
cd DubLok
Copy-Item .env.example .env
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -c backend\constraints.txt -r backend\requirements.txt -r backend\requirements-ml.txt
cd frontend
pnpm install --frozen-lockfile
cd ..
```

Linux or macOS:

```bash
git clone https://github.com/Ranbirkumar26/DubLok.git
cd DubLok
cp .env.example .env
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -c backend/constraints.txt -r backend/requirements.txt -r backend/requirements-ml.txt
cd frontend && pnpm install --frozen-lockfile && cd ..
sudo apt-get install -y ffmpeg   # or your package manager
```

PyTorch: `requirements-ml.txt` installs the default `torch` wheel for your platform. For an NVIDIA GPU on Windows or Linux, install the CUDA wheel instead (this is what the project was tested with):

```bash
pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0+cu128 torchaudio==2.11.0+cu128
```

On Windows, `.\scripts\start-production.ps1 -Install` does all of the above, including the CUDA wheel when `nvidia-smi` is present.

Then set up model access (next section), download the models, and run the preflight:

```bash
python backend/scripts/download_models.py
PYTHONPATH=backend python backend/scripts/check_models.py
```

On Windows use `$env:PYTHONPATH="backend"` before the second command.

## Model downloads and Hugging Face access

Some models are gated. Gated does not mean paid: your free Hugging Face account has to accept the model's terms once before downloads work.

1. Create or sign in to a free account at huggingface.co.
2. Open each page below and accept the terms:
   - `ai4bharat/indic-parler-tts` (speech, required)
   - `pyannote/speaker-diarization-community-1` (overlap-aware diarization)
   - If a pyannote download still returns 403, also accept `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0`.
   - `ai4bharat/indictrans2-en-indic-1B` (optional; only if you want IndicTrans2 instead of NLLB)
3. Create a read token under Settings, then Access Tokens.
4. Put it in your local `.env` (never commit this file):

```env
HUGGINGFACE_HUB_TOKEN=your_token_here
```

5. Run `python backend/scripts/download_models.py`. It downloads every model into `HF_HOME` (default `./storage/models/huggingface`) and tells you which gated ones still need terms accepted.

Models used:

| Stage | Model | Gated |
| --- | --- | --- |
| Speech to text | `Systran/faster-whisper-small` (balanced), `-tiny`, `-medium` by preset | No |
| Voice activity | Silero VAD (bundled in faster-whisper) | No |
| Diarization | `pyannote/speaker-diarization-community-1` | Yes |
| Diarization fallback | `speechbrain/spkrec-ecapa-voxceleb` | No |
| Translation | `facebook/nllb-200-distilled-1.3B`, then `-600M` | No |
| Translation (optional) | `ai4bharat/indictrans2-*-1B`, used first when downloaded | Yes |
| Text to speech | `ai4bharat/indic-parler-tts` | Yes |
| Background separation | Demucs (optional, only for the preserve-background mode) | No |

Translation uses IndicTrans2 when its files are already cached locally, otherwise NLLB-200 1.3B, then 600M. If a gated model fails, the pipeline falls back to the next one instead of stopping, except for text to speech, which has no fallback.

## Configuration

All settings live in `.env` (copied from `.env.example`). The ones you are most likely to change:

| Variable | Default | Meaning |
| --- | --- | --- |
| `ENGINE_MODE` | `production` | `production` runs the real models. `demo` uses placeholders for UI development only. |
| `DEVICE` | `auto` | `auto` uses CUDA when available, otherwise CPU. |
| `MODEL_PRESET` | `balanced` | `fast` (tiny Whisper), `balanced` (small), `quality` (medium). |
| `HF_HOME` | `./storage/models/huggingface` | Where model files are cached. |
| `HUGGINGFACE_HUB_TOKEN` | empty | Hugging Face read token for gated models. |
| `NUM_SPEAKERS` | empty | Number of speakers in the video. See below. |
| `BACKGROUND_MODE` | `full_replacement` | Original audio is replaced (default) or mixed in as background. |
| `ORIGINAL_VOLUME` / `TRANSLATED_VOLUME` | `0.20` / `1.00` | Mix levels when background is preserved. |
| `SPEECH_SPEED` | `1.00` | Global speaking-rate multiplier. |
| `MAX_UPLOAD_MB` | `750` | Upload size limit. |
| `MAX_DURATION_SECONDS` | `900` | Video length limit. |
| `ALLOWED_URL_HOSTS` | YouTube and Google Drive | Hosts accepted for URL input. |
| `STORAGE_ROOT`, `DATABASE_URL` | `./storage`, SQLite file | Where jobs and the database live. |

`NUM_SPEAKERS`: when left blank, pyannote decides how many speakers there are, and on overlapped speech it tends to under-count (it can report 1 speaker for a 2-person conversation). If you know the count, set it (for example `NUM_SPEAKERS=2`) for much better labels.

## Running the app

Windows, one command (creates the venv if missing, runs the preflight, starts the API, worker and frontend in hidden windows, and writes logs to `storage/logs/`):

```powershell
.\scripts\start-production.ps1
```

Manual start, three terminals from the repository root:

```bash
# 1. API
. .venv/bin/activate && export PYTHONPATH=backend
uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000

# 2. Worker (runs the ML stages)
. .venv/bin/activate && export PYTHONPATH=backend
python -m app.worker

# 3. Frontend
cd frontend && pnpm dev --host 127.0.0.1 --port 5173
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1` and set `$env:PYTHONPATH="backend"`.

Open http://127.0.0.1:5173/. The API health check is http://127.0.0.1:8000/api/health.

The worker must be running. The API only queues jobs.

## Using the dashboard

1. Choose a video file and a target language, then start the upload.
2. Wait for analysis. You will see the detected language, speakers and transcript.
3. If a line is wrong, flag it, edit the text, save, and mark it fixed.
4. Click "Continue to final video". Translation, speech generation and rendering run in sequence.
5. Preview or download the final MP4, the generated audio, SRT and VTT subtitles, and both transcripts.

You can also drive the same flow from a script, which is how the pipeline is exercised end to end without the browser:

```bash
PYTHONPATH=backend python backend/scripts/dashboard_dry_run.py --input path/to/video.mp4 --target-language hi
```

API summary:

```text
POST  /api/video/upload              upload a file
POST  /api/video/url                 submit a public URL
POST  /api/jobs                      create a job
GET   /api/jobs/{id}                 job status and progress
GET   /api/jobs/{id}/transcript      ?kind=original|translated
PUT   /api/jobs/{id}/transcript      save an edited transcript
POST  /api/jobs/{id}/translate       queue translation
POST  /api/jobs/{id}/generate-audio  queue speech generation
POST  /api/jobs/{id}/render          queue the final render
GET   /api/jobs/{id}/result          result summary
GET   /api/jobs/{id}/download/{artifact}   video | final_audio | srt | vtt | original_transcript | translated_transcript
```

## Overlapping speakers

Most diarizers give each stretch of time one speaker, so two people talking at once break them. DubLok uses pyannote community-1, which outputs speaker turns that can overlap in time:

- Each Whisper word is assigned to the speaker who is talking at that word's midpoint.
- A segment is split wherever the speaker changes, so one blended sentence becomes one segment per speaker.
- A segment where more than one speaker was active gets an `overlap` flag, which is saved with the transcript and exposed by the API.
- The dub mixer overlays segments rather than concatenating them, so overlapped lines play on top of each other as in the source.
- If pyannote cannot run (missing access, no GPU memory, any error), the pipeline falls back to SpeechBrain ECAPA clustering, then to a pause-based heuristic. A job never fails because of diarization.

What it does not do: Whisper hears the overlap as one blended stream, so the quieter speaker's words inside an overlap are usually not recovered, and a very short interjection can take the louder speaker's label. Recovering them needs source separation, which is not built in.

## Verifying your setup

Run these on a fresh clone before trusting it:

```bash
PYTHONPATH=backend python -m pytest backend/tests -q       # 56 tests, no models needed
PYTHONPATH=backend python backend/scripts/check_models.py  # ffmpeg, packages, device
cd frontend && pnpm build                                  # type-check and production build
curl http://127.0.0.1:8000/api/health                      # after starting the API
```

The tests do not download models and do not need a GPU.

After a real dub, check the output file:

```bash
ffprobe -hide_banner -v error -show_streams -show_format storage/jobs/<job_id>/output/translated_video_hi.mp4
```

Expect one video stream, one audio stream, and a duration within a small tolerance of the source.

A fresh clone is working when: `.env` exists with `ENGINE_MODE=production`, `ffmpeg -version` works, `check_models.py` passes, the tests pass, `pnpm build` passes, an uploaded video reaches transcript review, and "Continue to final video" produces a downloadable MP4.

## Docker

```bash
cp .env.example .env
docker compose up --build
```

This starts the API (port 8000), the worker, and the frontend (port 5173). The image installs the ML packages, but model files are not baked in: they download into the mounted `./storage` on first use, so run the model download step (or let the first job do it). For GPU use inside Docker you need the NVIDIA container runtime on the host; without it everything runs on CPU. The local virtualenv setup is the recommended path on Windows with a GPU.

## Security and privacy

- Nothing leaves your machine except model downloads from Hugging Face and optional public-URL downloads.
- Videos, transcripts, audio and the database stay under `storage/`, which is git-ignored.
- Secrets live only in `.env`, which is git-ignored. `.env.example` is the only env file in the repository and contains no secrets. Never paste a token into any tracked file.
- `.gitignore` also excludes media files, model caches, databases, logs, local tool settings and scratch folders, so a stray `git add .` does not publish them.
- Before pushing, a quick check: `git status` should show no `.env`, no `storage/`, and no media files.
- If you ever commit a token by mistake, revoke it on Hugging Face immediately, then remove it from history. Deleting it in a later commit does not remove it from the repository's history.
- URL input accepts only the hosts in `ALLOWED_URL_HOSTS`, and rejects private or authenticated links. Upload size and video duration limits are enforced.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `403` or "gated" error | Accept the model terms on Hugging Face with the same account that made the token, set `HUGGINGFACE_HUB_TOKEN` in `.env`, restart the API and worker. |
| `ffmpeg` or `ffprobe` missing | Install FFmpeg and check `ffmpeg -version` in a new terminal. |
| Dubbed voice sounds boomy or chipmunk-like | Your FFmpeg lacks the `rubberband` filter, so the lower-quality pitch fallback is in use. Install a full FFmpeg build. |
| Dashboard says demo mode | Set `ENGINE_MODE=production` in `.env` and restart. |
| CUDA not used | Check `nvidia-smi`, then install the CUDA PyTorch wheel (see Quick start). |
| Job stuck in "queued" | The worker is not running. Start `python -m app.worker`. |
| Diarization reports 1 speaker for several people | Set `NUM_SPEAKERS` in `.env`. |
| Diarization silently uses the fallback | Check `storage/logs/worker.log` for a 403 on `pyannote/speaker-diarization-community-1`. |
| Translation fails but transcription works | Check the worker log. Pre-download NLLB with `download_models.py`. |
| URL download fails | The link must be public. Local upload is the most reliable path. |
| A model downloads slowly or fails halfway | Re-run `download_models.py`. It resumes. |

Logs are in `storage/logs/` (`api.log`, `worker.log`, `frontend.log`) when started with the Windows helper.

## Known limitations

- The first production run is slow because models download.
- Gated Hugging Face access must be accepted by hand.
- CPU inference is much slower than GPU.
- The quieter speaker's words inside an overlap are not recovered (see above), and the speaker count needs `NUM_SPEAKERS` on overlapped audio.
- Pitch matching on overlapped source audio is approximate, because the source pitch itself is blended.
- Translation quality is that of the open NLLB and IndicTrans2 models. Names and unusual words can come out untranslated or odd.
- URL download depends on yt-dlp and site support can change.
- Dependency versions in `backend/constraints.txt` are the ones tested. Newer versions usually work but are not verified.

## Repository layout

```text
.
|-- backend/
|   |-- app/
|   |   |-- main.py            FastAPI routes
|   |   |-- worker.py          queue worker entry point
|   |   |-- db.py              SQLite schema and job state
|   |   |-- config.py          settings read from .env
|   |   |-- media.py           FFmpeg helpers, mixing, loudness, render
|   |   `-- pipeline/          transcription, diarization, translation, tts, subtitles
|   |-- scripts/
|   |   |-- check_models.py    preflight check
|   |   |-- download_models.py model downloader
|   |   `-- dashboard_dry_run.py
|   |-- tests/                 pytest suite (no models required)
|   |-- requirements.txt       web server dependencies
|   |-- requirements-ml.txt    model dependencies
|   `-- constraints.txt        tested versions
|-- frontend/                  React + Vite dashboard
|-- scripts/start-production.ps1   Windows startup helper
|-- docker-compose.yml
|-- .env.example               template for your local .env
`-- storage/                   runtime data (ignored by git)
```
