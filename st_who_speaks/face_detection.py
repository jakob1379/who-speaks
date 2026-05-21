from __future__ import annotations

import os
import tempfile
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from st_who_speaks.colors import SPEAKER_COLOR_PALETTE
from st_who_speaks.dependency_compat import (
    PIPELINE_TEMP_DIR_NAME,
    _import_optional_module,
    cv2,
    is_opencv_available,
)
from st_who_speaks.logging import get_logger
from st_who_speaks.models import FaceBox, FaceDetectionFrame, LandmarkPoint

logger = get_logger(__name__)

HAARCASCADE_FILENAME = "haarcascade_frontalface_default.xml"
HAARCASCADE_DOWNLOAD_SCHEME = "https"
HAARCASCADE_DOWNLOAD_HOST = ".".join(("raw", "githubusercontent", "com"))
HAARCASCADE_DOWNLOAD_PATH = "opencv/opencv/master/data/haarcascades"
HAARCASCADE_DOWNLOAD_URL = (
    f"{HAARCASCADE_DOWNLOAD_SCHEME}://"
    f"{HAARCASCADE_DOWNLOAD_HOST}/{HAARCASCADE_DOWNLOAD_PATH}/{HAARCASCADE_FILENAME}"
)
MEDIAPIPE_FACE_MESH_MODULE = "mediapipe.solutions.face_mesh"
FACE_OVERLAY_COLOR_HEX = SPEAKER_COLOR_PALETTE[0]

APPROXIMATE_LANDMARK_CONNECTIONS = [
    (0, 6),
    (6, 2),
    (2, 7),
    (7, 1),
    (0, 8),
    (8, 3),
    (3, 9),
    (9, 1),
    (2, 4),
    (4, 5),
    (5, 3),
]

def hex_to_bgr(color_hex: str) -> tuple[int, int, int]:
    normalized = color_hex.lstrip("#")
    red = int(normalized[0:2], 16)
    green = int(normalized[2:4], 16)
    blue = int(normalized[4:6], 16)
    return blue, green, red


def load_face_landmarker() -> Any | None:
    face_mesh_module = _import_optional_module(MEDIAPIPE_FACE_MESH_MODULE)
    if face_mesh_module is None:
        return None
    return face_mesh_module.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )


def load_face_mesh_connections() -> list[tuple[int, int]]:
    face_mesh_module = _import_optional_module(MEDIAPIPE_FACE_MESH_MODULE)
    if face_mesh_module is None:
        return APPROXIMATE_LANDMARK_CONNECTIONS
    connections = set(getattr(face_mesh_module, "FACEMESH_TESSELATION", set()))
    connections.update(getattr(face_mesh_module, "FACEMESH_CONTOURS", set()))
    connections.update(getattr(face_mesh_module, "FACEMESH_IRISES", set()))
    if not connections:
        return APPROXIMATE_LANDMARK_CONNECTIONS
    return [(int(start), int(end)) for start, end in sorted(connections)]


def approximate_landmarks_from_box(box: FaceBox) -> list[LandmarkPoint]:
    left = box.x
    top = box.y
    center_x = box.x + box.width // 2
    return [
        LandmarkPoint(x=int(left + box.width * 0.30), y=int(top + box.height * 0.35)),
        LandmarkPoint(x=int(left + box.width * 0.70), y=int(top + box.height * 0.35)),
        LandmarkPoint(x=int(center_x), y=int(top + box.height * 0.52)),
        LandmarkPoint(x=int(left + box.width * 0.36), y=int(top + box.height * 0.72)),
        LandmarkPoint(x=int(left + box.width * 0.64), y=int(top + box.height * 0.72)),
        LandmarkPoint(x=int(center_x), y=int(top + box.height * 0.90)),
        LandmarkPoint(x=int(left + box.width * 0.18), y=int(top + box.height * 0.46)),
        LandmarkPoint(x=int(left + box.width * 0.82), y=int(top + box.height * 0.46)),
        LandmarkPoint(x=int(left + box.width * 0.25), y=int(top + box.height * 0.58)),
        LandmarkPoint(x=int(left + box.width * 0.75), y=int(top + box.height * 0.58)),
    ]


def landmarks_from_crop(
    face_mesh: Any | None, crop: np.ndarray, *, offset_x: int, offset_y: int
) -> list[LandmarkPoint]:
    if face_mesh is None or crop.size == 0:
        return []

    rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_crop)
    if not getattr(results, "multi_face_landmarks", None):
        return []

    crop_height, crop_width = crop.shape[:2]
    points: list[LandmarkPoint] = []
    for landmark in results.multi_face_landmarks[0].landmark:
        x = int(round(landmark.x * crop_width)) + offset_x
        y = int(round(landmark.y * crop_height)) + offset_y
        points.append(LandmarkPoint(x=x, y=y))
    return points


def resolve_haarcascade_path() -> Path | None:
    if not is_opencv_available():
        return None
    candidates: list[Path] = []
    cv2_data = getattr(cv2, "data", None)
    haarcascades_path = getattr(cv2_data, "haarcascades", None)
    if haarcascades_path:
        candidates.append(Path(str(haarcascades_path)) / HAARCASCADE_FILENAME)
    cv2_file = getattr(cv2, "__file__", None)
    if cv2_file is not None:
        candidates.append(
            Path(cv2_file).resolve().parent / "data" / HAARCASCADE_FILENAME
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    download_target = (
        Path(tempfile.gettempdir())
        / PIPELINE_TEMP_DIR_NAME
        / "haarcascades"
        / HAARCASCADE_FILENAME
    )
    download_target.parent.mkdir(parents=True, exist_ok=True)
    if not download_haarcascade(download_target):
        return None
    return download_target if download_target.exists() else None


def download_haarcascade(download_target: Path) -> bool:
    temporary_path: Path | None = None
    try:
        with urllib.request.urlopen(HAARCASCADE_DOWNLOAD_URL, timeout=30) as response:
            descriptor, temporary_name = tempfile.mkstemp(dir=download_target.parent)
            temporary_path = Path(temporary_name)
            with os.fdopen(descriptor, "wb") as target:
                target.write(response.read())
            temporary_path.replace(download_target)
    except (OSError, urllib.error.URLError) as error:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        logger.warning(
            "failed to download Haar cascade face detector",
            url=HAARCASCADE_DOWNLOAD_URL,
            download_target=str(download_target),
            error=str(error),
        )
        return False
    return True


@lru_cache(maxsize=1)
def load_face_detector() -> Any | None:
    cascade_path = resolve_haarcascade_path()
    if cascade_path is None:
        return None
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        return None
    return detector


def resolve_landmark_connections(points: list[LandmarkPoint]) -> list[tuple[int, int]]:
    connections = load_face_mesh_connections()
    if len(points) < 50:
        connections = APPROXIMATE_LANDMARK_CONNECTIONS
    return [
        (start, end)
        for start, end in connections
        if start < len(points) and end < len(points)
    ]


def draw_landmark_wireframe(
    annotated: np.ndarray,
    points: list[LandmarkPoint],
    overlay_color: tuple[int, int, int],
) -> None:
    for start, end in resolve_landmark_connections(points):
        start_point = points[start]
        end_point = points[end]
        cv2.line(
            annotated,
            (start_point.x, start_point.y),
            (end_point.x, end_point.y),
            overlay_color,
            1,
        )


def annotate_face_box(
    annotated: np.ndarray,
    box: FaceBox,
    points: list[LandmarkPoint],
    overlay_color: tuple[int, int, int],
) -> None:
    cv2.rectangle(
        annotated,
        (box.x, box.y),
        (box.x + box.width, box.y + box.height),
        overlay_color,
        2,
    )
    draw_landmark_wireframe(annotated, points, overlay_color)
    draw_landmark_points(annotated, points, overlay_color)


def draw_landmark_points(
    annotated: np.ndarray,
    points: list[LandmarkPoint],
    overlay_color: tuple[int, int, int],
) -> None:
    for point in points:
        cv2.circle(annotated, (point.x, point.y), 1, overlay_color, -1)


def detect_and_annotate_faces_with_frame(
    frame: np.ndarray,
    *,
    color_hex: str | None = None,
) -> tuple[FaceDetectionFrame, np.ndarray]:
    detector = load_face_detector()
    resolved_color_hex = color_hex or FACE_OVERLAY_COLOR_HEX
    annotated = frame.copy()
    if detector is None:
        return _empty_face_detection_frame(
            annotated=annotated,
            resolved_color_hex=resolved_color_hex,
        )
    boxes, landmarks = _annotate_detected_faces(
        frame,
        detector=detector,
        annotated=annotated,
        resolved_color_hex=resolved_color_hex,
    )
    return (
        FaceDetectionFrame(
            face_count=len(boxes),
            boxes=boxes,
            landmarks=landmarks,
            annotated_image=None,
            color_hex=resolved_color_hex,
        ),
        annotated,
    )


def _empty_face_detection_frame(
    *,
    annotated: np.ndarray,
    resolved_color_hex: str,
) -> tuple[FaceDetectionFrame, np.ndarray]:
    return (
        FaceDetectionFrame(
            face_count=0,
            boxes=[],
            landmarks=[],
            annotated_image=None,
            color_hex=resolved_color_hex,
        ),
        annotated,
    )


def _annotate_detected_faces(
    frame: np.ndarray,
    *,
    detector: Any,
    annotated: np.ndarray,
    resolved_color_hex: str,
) -> tuple[list[FaceBox], list[list[LandmarkPoint]]]:
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        grayscale,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )
    boxes: list[FaceBox] = []
    landmarks: list[list[LandmarkPoint]] = []
    face_mesh = load_face_landmarker()
    overlay_color = hex_to_bgr(resolved_color_hex)
    for x, y, width, height in faces:
        box = FaceBox(x=int(x), y=int(y), width=int(width), height=int(height))
        boxes.append(box)
        crop = frame[box.y : box.y + box.height, box.x : box.x + box.width]
        points = landmarks_from_crop(
            face_mesh,
            crop,
            offset_x=box.x,
            offset_y=box.y,
        )
        if not points:
            points = approximate_landmarks_from_box(box)
        landmarks.append(points)
        annotate_face_box(annotated, box, points, overlay_color)
    return boxes, landmarks


def detect_and_annotate_faces(
    frame: np.ndarray,
    *,
    color_hex: str | None = None,
) -> FaceDetectionFrame:
    face_detection, annotated = detect_and_annotate_faces_with_frame(
        frame,
        color_hex=color_hex,
    )
    encoded, buffer = cv2.imencode(".jpg", annotated)
    return FaceDetectionFrame(
        face_count=face_detection.face_count,
        boxes=face_detection.boxes,
        landmarks=face_detection.landmarks,
        annotated_image=buffer.tobytes() if encoded else None,
        color_hex=face_detection.color_hex,
    )
