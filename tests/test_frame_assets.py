from __future__ import annotations

from pathlib import Path

import numpy as np

from st_who_speaks.frame_assets import (
    FrameAssetCollectionContext,
    FrameAssetCollectionResult,
    collect_frame_assets,
)
from st_who_speaks.media_io import ThumbnailExtractionResult
from st_who_speaks.models import FaceDetectionFrame, TranscriptChunk


def test_collect_frame_assets_returns_empty_when_frame_work_disabled(
    monkeypatch, tmp_path: Path
) -> None:
    chunks = [
        TranscriptChunk(
            speaker="Person A",
            start=0.0,
            end=1.0,
            text="hello",
            confidence=0.9,
            word_count=1,
        )
    ]
    monkeypatch.setattr(
        "st_who_speaks.frame_assets.prepare_video_for_frame_processing",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unused")),
    )
    monkeypatch.setattr(
        "st_who_speaks.frame_assets.extract_thumbnails",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unused")),
    )
    monkeypatch.setattr(
        "st_who_speaks.frame_assets.build_wireframe_video",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unused")),
    )
    monkeypatch.setattr(
        "st_who_speaks.frame_assets.load_face_detector",
        lambda: (_ for _ in ()).throw(AssertionError("unused")),
    )

    result = collect_frame_assets(
        FrameAssetCollectionContext(
            media_path="clip.webm",
            working_path=tmp_path,
            chunks=chunks,
            generate_thumbnails=False,
            enable_face_detection=False,
            max_thumbnails=6,
            progress_callback=None,
        )
    )

    assert result == FrameAssetCollectionResult(
        thumbnails={},
        face_thumbnails={},
        face_counts_by_chunk={},
        face_detections={},
            chunks=chunks,
            wireframe_video_bytes=None,
            wireframe_faces_detected=0,
            face_detector_available=False,
            diagnostics={},
        )


def test_collect_frame_assets_thumbnail_only_skips_wireframe(
    monkeypatch, tmp_path: Path
) -> None:
    chunks = [
        TranscriptChunk(
            speaker="Person A",
            start=0.0,
            end=1.0,
            text="hello",
            confidence=0.9,
            word_count=1,
        )
    ]
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "st_who_speaks.frame_assets.prepare_video_for_frame_processing",
        lambda media_path, working_path: captured.update(
            {"media_path": media_path, "working_path": working_path}
        )
        or "prepared.webm",
    )

    def fake_extract_thumbnails(media_path, received_chunks, **kwargs):
        captured["extract_media_path"] = media_path
        captured["chunks"] = received_chunks
        captured["kwargs"] = kwargs
        return ThumbnailExtractionResult(
            thumbnails={0: b"thumb"},
            sampled_frames={},
        )

    monkeypatch.setattr(
        "st_who_speaks.frame_assets.extract_thumbnails",
        fake_extract_thumbnails,
    )
    monkeypatch.setattr(
        "st_who_speaks.frame_assets.build_wireframe_video",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unused")),
    )
    monkeypatch.setattr(
        "st_who_speaks.frame_assets.load_face_detector",
        lambda: (_ for _ in ()).throw(AssertionError("unused")),
    )

    result = collect_frame_assets(
        FrameAssetCollectionContext(
            media_path="clip.webm",
            working_path=tmp_path,
            chunks=chunks,
            generate_thumbnails=True,
            enable_face_detection=False,
            max_thumbnails=6,
            progress_callback=None,
        )
    )

    assert captured["media_path"] == "clip.webm"
    assert captured["working_path"] == tmp_path
    assert captured["extract_media_path"] == "prepared.webm"
    assert captured["chunks"] == chunks
    assert captured["kwargs"] == {
        "max_thumbnails": 6,
        "generate_thumbnails": True,
        "include_frames": False,
    }
    assert result.thumbnails == {0: b"thumb"}
    assert result.face_detections == {}
    assert result.chunks == chunks
    assert result.wireframe_video_bytes is None
    assert result.face_detector_available is False
    assert result.diagnostics == {}


def test_collect_frame_assets_enriches_faces_and_wireframe(
    monkeypatch, tmp_path: Path
) -> None:
    chunks = [
        TranscriptChunk(
            speaker="Person A",
            start=0.0,
            end=1.0,
            text="hello",
            confidence=0.9,
            word_count=1,
        )
    ]
    progress_events: list[tuple[str, float]] = []
    monkeypatch.setattr(
        "st_who_speaks.frame_assets.prepare_video_for_frame_processing",
        lambda *_args, **_kwargs: "prepared.webm",
    )
    monkeypatch.setattr(
        "st_who_speaks.frame_assets.extract_thumbnails",
        lambda *_args, **_kwargs: ThumbnailExtractionResult(
            thumbnails={0: b"thumb"},
            sampled_frames={0: np.zeros((4, 4, 3), dtype=np.uint8)},
        ),
    )
    monkeypatch.setattr(
        "st_who_speaks.frame_assets.detect_and_annotate_faces",
        lambda *_args, **_kwargs: FaceDetectionFrame(
            face_count=2,
            annotated_image=b"face-thumb",
        ),
    )
    monkeypatch.setattr("st_who_speaks.frame_assets.load_face_detector", lambda: object())
    monkeypatch.setattr(
        "st_who_speaks.frame_assets.build_wireframe_video",
        lambda media_path, **kwargs: (
            b"wireframe",
            3,
        )
        if media_path == "prepared.webm"
        and kwargs == {
            "original_media_path": "clip.webm",
            "working_dir": tmp_path,
        }
        else (_ for _ in ()).throw(AssertionError("unexpected wireframe args")),
    )

    result = collect_frame_assets(
        FrameAssetCollectionContext(
            media_path="clip.webm",
            working_path=tmp_path,
            chunks=chunks,
            generate_thumbnails=True,
            enable_face_detection=True,
            max_thumbnails=6,
            progress_callback=lambda label, progress: progress_events.append(
                (label, progress)
            ),
        )
    )

    assert result.thumbnails == {0: b"thumb"}
    assert result.face_thumbnails == {0: b"face-thumb"}
    assert result.face_counts_by_chunk == {0: 2}
    assert result.face_detections[0].color_hex == "#ef4444"
    assert result.chunks[0].face_count == 2
    assert result.wireframe_video_bytes == b"wireframe"
    assert result.wireframe_faces_detected == 3
    assert result.face_detector_available is True
    assert result.diagnostics == {}
    assert progress_events == [("Rendering wireframe video", 0.97)]


def test_collect_frame_assets_preserves_faces_when_wireframe_video_fails(
    monkeypatch, tmp_path: Path
) -> None:
    chunks = [
        TranscriptChunk(
            speaker="Person A",
            start=0.0,
            end=1.0,
            text="hello",
            confidence=0.9,
            word_count=1,
        )
    ]
    monkeypatch.setattr(
        "st_who_speaks.frame_assets.prepare_video_for_frame_processing",
        lambda *_args, **_kwargs: "prepared.webm",
    )
    monkeypatch.setattr(
        "st_who_speaks.frame_assets.extract_thumbnails",
        lambda *_args, **_kwargs: ThumbnailExtractionResult(
            thumbnails={0: b"thumb"},
            sampled_frames={0: np.zeros((4, 4, 3), dtype=np.uint8)},
        ),
    )
    monkeypatch.setattr(
        "st_who_speaks.frame_assets.detect_and_annotate_faces",
        lambda *_args, **_kwargs: FaceDetectionFrame(
            face_count=1,
            annotated_image=b"face-thumb",
        ),
    )
    monkeypatch.setattr("st_who_speaks.frame_assets.load_face_detector", lambda: object())
    monkeypatch.setattr(
        "st_who_speaks.frame_assets.build_wireframe_video",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("ffmpeg exited 1")
        ),
    )

    result = collect_frame_assets(
        FrameAssetCollectionContext(
            media_path="clip.webm",
            working_path=tmp_path,
            chunks=chunks,
            generate_thumbnails=True,
            enable_face_detection=True,
            max_thumbnails=6,
            progress_callback=None,
        )
    )

    assert result.thumbnails == {0: b"thumb"}
    assert result.face_thumbnails == {0: b"face-thumb"}
    assert result.face_counts_by_chunk == {0: 1}
    assert result.chunks[0].face_count == 1
    assert result.wireframe_video_bytes is None
    assert result.wireframe_faces_detected == 0
    assert result.face_detector_available is True
    assert result.diagnostics == {
        "wireframe_video": "Wireframe video generation failed: ffmpeg exited 1"
    }
