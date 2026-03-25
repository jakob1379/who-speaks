from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from st_who_speaks.pipeline import ProcessMediaOptions, process_media


@pytest.mark.sample_smoke
def test_sample_media_pipeline_smoke(sample_media_path: Path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.fail(
            "Sample smoke test requires ffmpeg on PATH. "
            "Install it locally or run inside the repo dev environment."
        )
    if shutil.which("ffprobe") is None:
        pytest.fail(
            "Sample smoke test requires ffprobe on PATH. "
            "Install it locally or run inside the repo dev environment."
        )

    progress_events: list[tuple[str, float]] = []

    result = process_media(
        str(sample_media_path),
        ProcessMediaOptions(
            use_hardware_acceleration=False,
            media_label=sample_media_path.name,
            whisper_model_size="tiny",
            transcription_device="cpu",
            transcription_compute_type="int8",
            embedding_device="cpu",
            hardware_acceleration_enabled=False,
            min_speakers=1,
            max_speakers=4,
            generate_thumbnails=True,
            enable_face_detection=True,
            max_thumbnails=4,
            generate_wireframe_video=True,
        ),
        progress_callback=lambda label, progress: progress_events.append(
            (label, progress)
        ),
    )

    assert result.media_path == sample_media_path.name
    assert result.duration > 0, "sample media produced no duration"
    assert result.speakers, (
        "sample media produced no speakers; check diarization caches"
    )
    assert result.chunks, (
        "sample media produced no transcript chunks; check Whisper/model caches"
    )
    assert len(result.speakers) >= 2, (
        "sample interview should yield at least two speakers; interviewer and interviewee are being merged"
    )
    assert len({chunk.speaker for chunk in result.chunks}) >= 2, (
        "sample transcript chunks should contain at least two distinct speaker labels"
    )
    assert result.metadata["chunk_count"] == len(result.chunks)
    assert result.metadata["speaker_count"] == len(result.speakers)
    assert progress_events and progress_events[-1] == ("Done", 1.0)

    first_chunk = result.chunks[0]
    assert first_chunk.text.strip(), "first transcript chunk is empty"
    assert first_chunk.end >= first_chunk.start
    assert result.metadata["diarization_mode"] == "local-clustering"
    assert result.metadata["opencv_available"] is True, (
        f"OpenCV failed to import in the sample smoke environment: {result.metadata.get('opencv_import_error')}"
    )
    assert result.metadata["face_detector_available"] is True, (
        "Face detector was unavailable for the sample smoke run; OpenCV/cascade runtime is broken"
    )
    assert result.metadata["wireframe_video_available"] is True, (
        "Sample video should produce a processed wireframe video"
    )
    assert result.metadata["wireframe_faces_detected"] > 0, (
        "Sample video should produce detectable faces for the wireframe overlay"
    )
    assert result.wireframe_video_bytes is not None, (
        "Wireframe video bytes were not generated for the checked-in sample"
    )
    assert result.face_thumbnails, (
        "Sample video should produce annotated face thumbnails"
    )
    assert result.face_detections, "Sample video should produce face detections"
    assert any(
        detection.face_count > 0 for detection in result.face_detections.values()
    ), "Sample video should contain at least one detected face"
    assert any(detection.landmarks for detection in result.face_detections.values()), (
        "Sample video should contain facial landmark points"
    )
