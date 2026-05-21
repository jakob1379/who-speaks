from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from st_who_speaks.dependency_compat import cv2
from st_who_speaks.face_detection import (
    FACE_OVERLAY_COLOR_HEX,
    detect_and_annotate_faces,
    detect_and_annotate_faces_with_frame,
    load_face_detector,
)
from st_who_speaks.logging import get_logger
from st_who_speaks.media_io import (
    FFMPEG_BINARY,
    FFMPEG_TIMEOUT_SECONDS,
    extract_thumbnails,
    prepare_video_for_frame_processing,
)
from st_who_speaks.models import FaceDetectionFrame, TranscriptChunk

FFMPEG_RAW_PIX_FMT = "bgr24"
FFMPEG_OUTPUT_PIX_FMT = "yuv420p"
TEXT_ENCODING = "utf-8"
ProgressCallback = Callable[[str, float], None]
logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class _WireframeProcessResult:
    return_code: int
    stderr: str
    total_faces_detected: int


@dataclass(frozen=True, slots=True)
class FrameAssetCollectionContext:
    media_path: str
    working_path: Path
    chunks: list[TranscriptChunk]
    generate_thumbnails: bool
    enable_face_detection: bool
    max_thumbnails: int
    progress_callback: ProgressCallback | None


@dataclass(frozen=True, slots=True)
class FrameAssetCollectionResult:
    thumbnails: dict[int, bytes]
    face_thumbnails: dict[int, bytes]
    face_counts_by_chunk: dict[int, int]
    face_detections: dict[int, FaceDetectionFrame]
    chunks: list[TranscriptChunk]
    wireframe_video_bytes: bytes | None
    wireframe_faces_detected: int
    face_detector_available: bool
    diagnostics: dict[str, str]


@dataclass(frozen=True, slots=True)
class FaceAssetEnrichmentResult:
    chunks: list[TranscriptChunk]
    face_thumbnails: dict[int, bytes]
    face_counts_by_chunk: dict[int, int]
    face_detections: dict[int, FaceDetectionFrame]
    wireframe_video_bytes: bytes | None
    wireframe_faces_detected: int
    face_detector_available: bool
    diagnostics: dict[str, str]


def collect_frame_assets(
    context: FrameAssetCollectionContext,
) -> FrameAssetCollectionResult:
    if not (context.generate_thumbnails or context.enable_face_detection):
        return FrameAssetCollectionResult(
            thumbnails={},
            face_thumbnails={},
            face_counts_by_chunk={},
            face_detections={},
            chunks=context.chunks,
            wireframe_video_bytes=None,
            wireframe_faces_detected=0,
            face_detector_available=False,
            diagnostics={},
        )

    frame_media_path = prepare_video_for_frame_processing(
        context.media_path, context.working_path
    )
    extracted_frames = extract_thumbnails(
        frame_media_path,
        context.chunks,
        max_thumbnails=context.max_thumbnails,
        generate_thumbnails=context.generate_thumbnails,
        include_frames=context.enable_face_detection,
    )
    enrichment = _build_face_asset_enrichment(
        context,
        frame_media_path=frame_media_path,
        sampled_frames=extracted_frames.sampled_frames,
    )

    return FrameAssetCollectionResult(
        thumbnails=extracted_frames.thumbnails,
        face_thumbnails=enrichment.face_thumbnails,
        face_counts_by_chunk=enrichment.face_counts_by_chunk,
        face_detections=enrichment.face_detections,
        chunks=enrichment.chunks,
        wireframe_video_bytes=enrichment.wireframe_video_bytes,
        wireframe_faces_detected=enrichment.wireframe_faces_detected,
        face_detector_available=enrichment.face_detector_available,
        diagnostics=enrichment.diagnostics,
    )


def _build_face_asset_enrichment(
    context: FrameAssetCollectionContext,
    *,
    frame_media_path: str,
    sampled_frames: dict[int, np.ndarray],
) -> FaceAssetEnrichmentResult:
    if not context.enable_face_detection:
        return FaceAssetEnrichmentResult(
            chunks=context.chunks,
            face_thumbnails={},
            face_counts_by_chunk={},
            face_detections={},
            wireframe_video_bytes=None,
            wireframe_faces_detected=0,
            face_detector_available=False,
            diagnostics={},
        )

    face_detector_available = load_face_detector() is not None
    face_thumbnails: dict[int, bytes] = {}
    face_counts_by_chunk: dict[int, int] = {}
    face_detections: dict[int, FaceDetectionFrame] = {}
    for index, frame in sampled_frames.items():
        face_detection = detect_and_annotate_faces(
            frame,
            color_hex=FACE_OVERLAY_COLOR_HEX,
        )
        face_counts_by_chunk[index] = face_detection.face_count
        face_detections[index] = replace(
            face_detection,
            color_hex=face_detection.color_hex or FACE_OVERLAY_COLOR_HEX,
        )
        if face_detection.annotated_image is not None:
            face_thumbnails[index] = face_detection.annotated_image

    chunks = [
        replace(chunk, face_count=face_counts_by_chunk.get(index))
        for index, chunk in enumerate(context.chunks)
    ]
    _notify(context.progress_callback, "Rendering wireframe video", 0.97)
    diagnostics: dict[str, str] = {}
    try:
        wireframe_video_bytes, wireframe_faces_detected = build_wireframe_video(
            frame_media_path,
            original_media_path=context.media_path,
            working_dir=context.working_path,
        )
    except RuntimeError as error:
        wireframe_video_bytes = None
        wireframe_faces_detected = 0
        diagnostics["wireframe_video"] = (
            f"Wireframe video generation failed: {error}"
        )
        logger.warning(
            "wireframe video generation failed",
            media_path=context.media_path,
            error=str(error),
        )
    return FaceAssetEnrichmentResult(
        chunks=chunks,
        face_thumbnails=face_thumbnails,
        face_counts_by_chunk=face_counts_by_chunk,
        face_detections=face_detections,
        wireframe_video_bytes=wireframe_video_bytes,
        wireframe_faces_detected=wireframe_faces_detected,
        face_detector_available=face_detector_available,
        diagnostics=diagnostics,
    )


def _read_wireframe_capture_dimensions(capture: Any) -> tuple[int, int]:
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    return width, height


def _read_stderr_file(stderr_file: Any) -> str:
    stderr_file.seek(0)
    stderr_bytes = stderr_file.read()
    return stderr_bytes.decode(TEXT_ENCODING, errors="ignore")


def _write_annotated_frame(stdin: Any, annotated: np.ndarray) -> bool:
    try:
        stdin.write(annotated.tobytes())
    except BrokenPipeError:
        return False
    return True


def _render_wireframe_video_frames(
    capture: Any,
    process: Any,
) -> int:
    total_faces_detected = 0
    stdin = process.stdin
    if stdin is None:
        return 0

    while True:
        success, frame = capture.read()
        if not success:
            break
        face_detection, annotated = detect_and_annotate_faces_with_frame(
            frame,
            color_hex=FACE_OVERLAY_COLOR_HEX,
        )
        total_faces_detected += face_detection.face_count
        if not _write_annotated_frame(stdin, annotated):
            break
    return total_faces_detected


def _wireframe_video_command(
    *,
    width: int,
    height: int,
    fps: float,
    original_media_path: str,
    target: Path,
) -> list[str]:
    return [
        FFMPEG_BINARY,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        FFMPEG_RAW_PIX_FMT,
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "-",
        "-i",
        original_media_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "libx264",
        "-pix_fmt",
        FFMPEG_OUTPUT_PIX_FMT,
        "-preset",
        "veryfast",
        "-c:a",
        "aac",
        "-shortest",
        str(target),
    ]


def _run_wireframe_ffmpeg_process(
    command: list[str],
    capture: Any,
) -> _WireframeProcessResult:
    with tempfile.TemporaryFile() as stderr_file:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stderr=stderr_file,
        )
        total_faces_detected = 0
        stdin = process.stdin
        if stdin is not None:
            with stdin:
                total_faces_detected = _render_wireframe_video_frames(
                    capture,
                    process,
                )
        try:
            return_code = process.wait(timeout=FFMPEG_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as error:
            process.kill()
            process.wait()
            stderr = _read_stderr_file(stderr_file).strip()
            raise RuntimeError(
                f"ffmpeg wireframe rendering timed out after {FFMPEG_TIMEOUT_SECONDS}s."
                f" {stderr}"
            ) from error
        return _WireframeProcessResult(
            return_code=return_code,
            stderr=_read_stderr_file(stderr_file),
            total_faces_detected=total_faces_detected,
        )


def _build_wireframe_video_from_capture(
    capture: Any,
    *,
    original_media_path: str,
    working_dir: Path,
) -> tuple[bytes | None, int]:
    width, height = _read_wireframe_capture_dimensions(capture)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if width <= 0 or height <= 0:
        return None, 0
    if not math.isfinite(fps) or fps <= 0:
        fps = 24.0

    target = working_dir / "wireframe-overlay.mp4"
    process_result = _run_wireframe_ffmpeg_process(
        _wireframe_video_command(
            width=width,
            height=height,
            fps=fps,
            original_media_path=original_media_path,
            target=target,
        ),
        capture,
    )
    if process_result.return_code != 0:
        raise RuntimeError(
            process_result.stderr.strip()
            or "ffmpeg failed to build the wireframe video."
        )
    if target.exists() and process_result.total_faces_detected > 0:
        return target.read_bytes(), process_result.total_faces_detected
    return None, process_result.total_faces_detected


def build_wireframe_video(
    media_path: str,
    *,
    original_media_path: str,
    working_dir: Path,
) -> tuple[bytes | None, int]:
    if shutil.which("ffmpeg") is None or load_face_detector() is None:
        return None, 0

    capture = cv2.VideoCapture(media_path)
    try:
        if not capture.isOpened():
            return None, 0
        return _build_wireframe_video_from_capture(
            capture,
            original_media_path=original_media_path,
            working_dir=working_dir,
        )
    finally:
        capture.release()


def _notify(
    progress_callback: ProgressCallback | None, label: str, progress: float
) -> None:
    if progress_callback is not None:
        progress_callback(label, progress)
