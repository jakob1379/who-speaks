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
> `flake.nix` provides the required system libraries. Python package dependencies come from `pyproject.toml` and are locked in `uv.lock`.

## Dependency policy

`mediapipe` is the OpenCV provider for this app: it brings `opencv-contrib-python`, which satisfies the `cv2` imports used for frame extraction, face detection, and overlays. Do not add a second direct OpenCV wheel unless the provider decision is revisited and verified with `uv tree`.

Runtime ML/media dependencies use compatible upper bounds in `pyproject.toml`; `uv.lock` is the tested compatibility boundary. After changing dependency constraints, run:

```bash
uv lock
uv tree --package st-who-speaks --depth 2
```

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

## Release check

Run the real-model smoke test before tagging or publishing a release from an environment with `ffmpeg`, `ffprobe`, and warmed Whisper/SpeechBrain model caches:

```bash
RUN_SAMPLE_SMOKE=1 uv run pytest tests/test_sample_smoke.py -m sample_smoke --run-sample-smoke
```

Keep this check out of the default local test path because first-run model downloads are slow and network-dependent.

## Repo layout

- `st_who_speaks/cli.py` — `who-speaks` command that launches Streamlit
- `st_who_speaks/streamlit_app.py` — packaged Streamlit entrypoint
- `st_who_speaks/ui/app.py` — Streamlit UI orchestration and result rendering
- `st_who_speaks/ui/media.py` — media player and subtitle helpers
- `st_who_speaks/pipeline.py` — high-level `process_media` orchestration and result assembly
- `st_who_speaks/frame_assets.py` — frame asset collection, face enrichment, and wireframe video rendering
- `st_who_speaks/media_io.py` — generic ffmpeg/ffprobe calls, bounded subprocess helpers, frame extraction, thumbnailing, and media duration helpers
- `st_who_speaks/face_detection.py` — OpenCV/MediaPipe detector loading, face boxes, landmarks, and overlays
- `st_who_speaks/transcription.py` — faster-whisper model loading and word/segment extraction
- `st_who_speaks/diarization.py` — SpeechBrain embeddings, waveform slicing, and speaker clustering
- `st_who_speaks/transcript.py` — transcript chunk assembly and speaker labels
- `st_who_speaks/runtime.py` — CPU/GPU execution settings and acceleration detection
- `st_who_speaks/dependency_compat.py` — compatibility shims for third-party model dependencies
- `st_who_speaks/colors.py` — speaker color assignment and display names
- `st_who_speaks/logging.py` — structlog configuration
- `st_who_speaks/models.py` — shared result, transcript, face, and metadata dataclasses
