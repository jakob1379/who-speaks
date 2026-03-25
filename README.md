# Who Speaks

Streamlit app for local transcription, speaker diarization, thumbnails, and optional face overlays on uploaded audio or video.

## What it does

- local audio/video upload
- `ffmpeg` audio extraction
- `faster-whisper` transcription
- SpeechBrain speaker diarization
- transcript chunks labeled `Person A/B/C`
- OpenCV thumbnails
- optional face detection and landmarks on sampled frames

## System dependencies

- `python312`
- `uv`
- `ffmpeg`
- on Linux, OpenCV/MediaPipe runtime libs such as `glib`, `libGL`, `libice`, `libsm`, `libx11`, `libxext`, and `libxrender`

`flake.nix` provides those. Python package dependencies come from `pyproject.toml` via `uv`.

## Run

With Nix:

```bash
nix develop -c uv run who-speaks
```

Without Nix, install the system dependencies above, then:

```bash
uv sync --python 3.12
uv run who-speaks
```

If no GPU is available, the app runs on CPU.

## Smoke test

```bash
uv run pytest tests/test_sample_smoke.py -m sample_smoke --run-sample-smoke
```

Uses `samples/george-siemens-interview-90s.webm` by default and may download model weights on first run.

## Repo layout

- `st_who_speaks/streamlit_app.py` — packaged Streamlit entrypoint
- `st_who_speaks/app.py` — UI
- `st_who_speaks/pipeline.py` — media processing pipeline
- `st_who_speaks/models.py` — result models
