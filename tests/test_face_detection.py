from __future__ import annotations

import numpy as np

from st_who_speaks.face_detection import detect_and_annotate_faces
from st_who_speaks.models import FaceBox, LandmarkPoint


def test_detect_and_annotate_faces_returns_boxes_and_image(monkeypatch) -> None:
    class FakeDetector:
        def detectMultiScale(self, *_args, **_kwargs):
            return np.array([[10, 15, 30, 40], [50, 60, 20, 25]])

    class FakeLandmark:
        def __init__(self, x: float, y: float):
            self.x = x
            self.y = y

    class FakeFaceLandmarks:
        def __init__(self):
            self.landmark = [FakeLandmark(0.2, 0.3), FakeLandmark(0.8, 0.7)]

    class FakeLandmarker:
        def process(self, *_args, **_kwargs):
            return type("Result", (), {"multi_face_landmarks": [FakeFaceLandmarks()]})()

    monkeypatch.setattr(
        "st_who_speaks.face_detection.load_face_detector",
        lambda: FakeDetector(),
    )
    monkeypatch.setattr(
        "st_who_speaks.face_detection.load_face_landmarker",
        lambda: FakeLandmarker(),
    )

    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    result = detect_and_annotate_faces(
        frame,
        color_hex="#ef4444",
    )

    assert result.face_count == 2
    assert result.boxes == [
        FaceBox(x=10, y=15, width=30, height=40),
        FaceBox(x=50, y=60, width=20, height=25),
    ]
    assert result.color_hex == "#ef4444"
    assert result.landmarks == [
        [LandmarkPoint(x=16, y=27), LandmarkPoint(x=34, y=43)],
        [LandmarkPoint(x=54, y=68), LandmarkPoint(x=66, y=78)],
    ]
    assert result.annotated_image is not None
