# Who Speaks

Streamlit app for local transcription, speaker diarization, thumbnails, and optional face overlays on uploaded audio or video.

![preview of the streamlit application](https://raw.githubusercontent.com/jakob1379/who-speaks/refs/heads/main/preview.png)

## What it does

- local audio/video upload
- `ffmpeg` audio extraction
- `faster-whisper` transcription
- SpeechBrain speaker diarization
- transcript chunks labeled `Person A/B/C`
- OpenCV thumbnails
- optional face detection and landmarks on sampled frames

Face overlays are visual-only. They do not identify which diarized speaker is on screen.

## System dependencies

- `uv`
- `ffmpeg`
- on Linux, OpenCV/MediaPipe runtime libs such as `glib`, `libGL`, `libice`, `libsm`, `libx11`, `libxext`, and `libxrender`

> [!NOTE]
> `flake.nix` provides the required system libraries. Python package dependencies come from `pyproject.toml` via `uv`.

## Run

With Nix:

```bash
nix develop -c uv run who-speaks
```

Without Nix, install the system dependencies above, then:

```bash
uv run who-speaks
```

If no GPU is available, the app runs on CPU.

## Smoke test

```bash
uv run pytest tests/test_sample_smoke.py -m sample_smoke --run-sample-smoke
```

> [!IMPORTANT]
> The smoke test uses `samples/george-siemens-interview-90s.webm` by default and may download model weights on first run.

## Repo layout

- `st_who_speaks/streamlit_app.py` — packaged Streamlit entrypoint
- `st_who_speaks/app.py` — UI
- `st_who_speaks/pipeline.py` — media processing pipeline
- `st_who_speaks/models.py` — result models
