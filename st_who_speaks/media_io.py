from __future__ import annotations

import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from st_who_speaks.dependency_compat import cv2
from st_who_speaks.logging import get_logger
from st_who_speaks.models import TranscriptChunk

logger = get_logger(__name__)

FFMPEG_BINARY = "ffmpeg"
FFMPEG_TIMEOUT_SECONDS = 600
FFPROBE_TIMEOUT_SECONDS = 30
SECONDS_TO_MILLISECONDS = 1000


@dataclass(frozen=True, slots=True)
class ThumbnailExtractionResult:
    thumbnails: dict[int, bytes]
    sampled_frames: dict[int, np.ndarray]


def extract_audio(media_path: str, audio_path: str) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is not available. Install it through flake.nix or your OS package manager."
        )
    if not media_has_audio_stream(media_path):
        raise RuntimeError(
            "The uploaded media file does not contain an audio stream. "
            "Transcription and diarization require speech audio."
        )
    command = [
        FFMPEG_BINARY,
        "-y",
        "-i",
        media_path,
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        audio_path,
    ]
    completed = run_bounded_subprocess(
        command,
        timeout_seconds=FFMPEG_TIMEOUT_SECONDS,
        operation="ffmpeg audio extraction",
    )
    if completed.returncode != 0:
        logger.error(
            "audio extraction failed",
            media_path=media_path,
            stderr=completed.stderr.strip(),
        )
        raise RuntimeError(
            summarize_ffmpeg_error(completed.stderr)
            or "ffmpeg failed to extract audio."
        )


def _read_capture_dimensions(capture: Any) -> tuple[int, int]:
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    return width, height


def prepare_video_for_frame_processing(
    media_path: str, working_dir: Path, *, max_width: int = 1920, max_height: int = 1080
) -> str:
    capture = cv2.VideoCapture(media_path)
    try:
        if not capture.isOpened():
            return media_path

        width, height = _read_capture_dimensions(capture)
        if width <= 0 or height <= 0 or (width <= max_width and height <= max_height):
            return media_path
    finally:
        capture.release()

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is not available. Install it through flake.nix or your OS package manager."
        )

    target = working_dir / "frame-source.mp4"
    command = [
        FFMPEG_BINARY,
        "-y",
        "-i",
        media_path,
        "-vf",
        f"scale={max_width}:{max_height}:force_original_aspect_ratio=decrease",
        "-an",
        str(target),
    ]
    completed = run_bounded_subprocess(
        command,
        timeout_seconds=FFMPEG_TIMEOUT_SECONDS,
        operation="ffmpeg frame-source downsampling",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            completed.stderr.strip() or "ffmpeg failed to downsample the video."
        )
    return str(target)


def run_bounded_subprocess(
    command: list[str], *, timeout_seconds: int, operation: str
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"{operation} timed out after {timeout_seconds}s.") from error


def media_has_audio_stream(media_path: str) -> bool:
    if shutil.which("ffprobe") is None:
        return True
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "csv=p=0",
        media_path,
    ]
    try:
        completed = run_bounded_subprocess(
            command,
            timeout_seconds=FFPROBE_TIMEOUT_SECONDS,
            operation="ffprobe audio stream detection",
        )
    except RuntimeError as error:
        logger.warning(
            "audio stream probe timed out",
            media_path=media_path,
            error=str(error),
        )
        return True
    return completed.returncode == 0 and bool(completed.stdout.strip())


def summarize_ffmpeg_error(stderr: str) -> str:
    normalized = stderr.strip()
    if "Output file does not contain any stream" in normalized:
        return (
            "The uploaded media file does not contain an audio stream. "
            "Transcription and diarization require speech audio."
        )
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    if not lines:
        return ""
    return lines[-1]


def extract_thumbnails(
    media_path: str,
    chunks: list[TranscriptChunk],
    *,
    max_thumbnails: int,
    generate_thumbnails: bool,
    include_frames: bool,
) -> ThumbnailExtractionResult:
    if not chunks or max_thumbnails <= 0:
        return ThumbnailExtractionResult({}, {})

    capture = cv2.VideoCapture(media_path)
    try:
        if not capture.isOpened():
            return ThumbnailExtractionResult({}, {})
        return _extract_thumbnails_from_capture(
            capture,
            chunks=chunks,
            max_thumbnails=max_thumbnails,
            generate_thumbnails=generate_thumbnails,
            include_frames=include_frames,
        )
    finally:
        capture.release()


def read_frame_at_timestamp(capture: Any, timestamp: float) -> np.ndarray | None:
    capture.set(cv2.CAP_PROP_POS_MSEC, max(timestamp, 0.0) * SECONDS_TO_MILLISECONDS)
    success, frame = capture.read()
    if not success:
        return None
    return frame


def encode_thumbnail(frame: np.ndarray) -> bytes | None:
    encoded, buffer = cv2.imencode(".jpg", frame)
    if not encoded or buffer is None:
        return None
    return buffer.tobytes()


def _extract_thumbnails_from_capture(
    capture: Any,
    *,
    chunks: list[TranscriptChunk],
    max_thumbnails: int,
    generate_thumbnails: bool,
    include_frames: bool,
) -> ThumbnailExtractionResult:
    selected_indices = sampled_indices(len(chunks), max_thumbnails)
    thumbnails: dict[int, bytes] = {}
    sampled_frames: dict[int, np.ndarray] = {}
    for index in selected_indices:
        timestamp = chunks[index].thumbnail_timestamp or chunks[index].start
        frame = read_frame_at_timestamp(capture, timestamp)
        if frame is None:
            continue
        if include_frames:
            sampled_frames[index] = frame.copy()
        if generate_thumbnails:
            thumbnail_bytes = encode_thumbnail(frame)
            if thumbnail_bytes is not None:
                thumbnails[index] = thumbnail_bytes
        del frame
    return ThumbnailExtractionResult(thumbnails=thumbnails, sampled_frames=sampled_frames)


def sampled_indices(total_items: int, limit: int) -> list[int]:
    if total_items <= limit:
        return list(range(total_items))
    step = total_items / limit
    return sorted(
        {min(total_items - 1, math.floor(index * step)) for index in range(limit)}
    )


def infer_duration(media_path: str, chunks: list[TranscriptChunk]) -> float:
    capture = cv2.VideoCapture(media_path)
    try:
        if capture.isOpened():
            fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
            frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
            if fps > 0 and frame_count > 0:
                return round(frame_count / fps, 2)
    finally:
        capture.release()
    if chunks:
        return round(chunks[-1].end, 2)
    return 0.0
