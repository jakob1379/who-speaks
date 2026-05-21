from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

from st_who_speaks.frame_assets import FrameAssetCollectionResult
from st_who_speaks.media_io import ThumbnailExtractionResult
from st_who_speaks.models import (
    FaceBox,
    FaceDetectionFrame,
    SpeakerTurn,
    TranscriptionSegment,
    WordToken,
)
from st_who_speaks.pipeline import ProcessMediaOptions, process_media
from st_who_speaks.runtime import ExecutionSettings


CUDA_EXECUTION = ExecutionSettings(
    hardware_acceleration_enabled=True,
    backend_label="NVIDIA CUDA",
    summary="CUDA available",
    transcription_device="cuda",
    transcription_compute_type="float16",
    embedding_device="cuda",
)

CPU_EXECUTION = ExecutionSettings(
    hardware_acceleration_enabled=False,
    backend_label=None,
    summary="CPU only",
    transcription_device="cpu",
    transcription_compute_type="int8",
    embedding_device="cpu",
)


def test_process_media_with_local_diarization_and_face_data(
    monkeypatch, tmp_path: Path
) -> None:
    progress_events: list[tuple[str, float]] = []

    monkeypatch.setattr(
        "st_who_speaks.pipeline.extract_audio",
        lambda media_path, audio_path: Path(audio_path).write_bytes(b"RIFF"),
    )
    monkeypatch.setattr(
        "st_who_speaks.pipeline.transcribe_audio_data",
        lambda *_args, **_kwargs: (
            (
                WordToken("Hello", 0.0, 0.4, 0.9, "Unknown"),
                WordToken("world", 0.5, 1.0, 0.8, "Unknown"),
            ),
            (TranscriptionSegment("Hello world", 0.0, 1.0, 0.9),),
        ),
    )
    monkeypatch.setattr(
        "st_who_speaks.pipeline.diarize_audio",
        lambda *_args, **_kwargs: [
            SpeakerTurn("speaker_0", "Person A", 0.0, 0.6),
        ],
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
            boxes=[FaceBox(1, 2, 3, 4)],
            annotated_image=b"face-thumb",
        ),
    )
    monkeypatch.setattr("st_who_speaks.frame_assets.load_face_detector", lambda: object())
    monkeypatch.setattr(
        "st_who_speaks.pipeline.is_opencv_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "st_who_speaks.frame_assets.build_wireframe_video",
        lambda *_args, **_kwargs: (b"wireframe-video", 4),
    )
    monkeypatch.setattr("st_who_speaks.pipeline.infer_duration", lambda *_args: 12.5)

    media_path = tmp_path / "clip.webm"
    media_path.write_bytes(b"video")

    result = process_media(
        str(media_path),
        ProcessMediaOptions(
            execution_settings=CUDA_EXECUTION,
            media_label="clip.webm",
            whisper_model_size="tiny",
            min_speakers=1,
            max_speakers=4,
            generate_thumbnails=True,
            enable_face_detection=True,
            max_thumbnails=6,
        ),
        progress_callback=lambda label, progress: progress_events.append(
            (label, progress)
        ),
    )

    assert result.duration == 12.5
    assert result.speakers == ["Person A"]
    assert len(result.chunks) == 1
    assert result.chunks[0].text == "Hello world"
    assert result.chunks[0].face_count == 2
    assert result.metadata.face_detection_enabled is True
    assert result.metadata.total_faces_detected == 2
    assert result.metadata.speaker_colors == {"Person A": "#ef4444"}
    assert result.metadata.hardware_acceleration_enabled is True
    assert result.metadata.transcription_device == "cuda"
    assert result.metadata.embedding_device == "cuda"
    assert result.metadata.diarization_warning is None
    assert result.metadata.diagnostics == {}
    assert result.metadata.opencv_available is True
    assert result.metadata.face_detector_available is True
    assert result.metadata.generate_wireframe_video is True
    assert result.metadata.wireframe_video_available is True
    assert result.metadata.wireframe_faces_detected == 4
    assert result.face_detections[0].face_count == 2
    assert result.face_detections[0].color_hex == "#ef4444"
    assert result.wireframe_video_bytes == b"wireframe-video"
    assert progress_events[-1] == ("Done", 1.0)


def test_process_media_local_diarization_failure_degrades_gracefully(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "st_who_speaks.pipeline.extract_audio",
        lambda media_path, audio_path: Path(audio_path).write_bytes(b"RIFF"),
    )
    monkeypatch.setattr(
        "st_who_speaks.pipeline.transcribe_audio_data",
        lambda *_args, **_kwargs: (
            (
                WordToken("A", 0.0, 0.2, 0.9, "Unknown"),
                WordToken("test", 0.3, 0.6, 0.9, "Unknown"),
            ),
            (TranscriptionSegment("A test", 0.0, 0.6, 0.9),),
        ),
    )
    monkeypatch.setattr(
        "st_who_speaks.pipeline.diarize_audio",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bad diarizer")),
    )
    monkeypatch.setattr(
        "st_who_speaks.frame_assets.extract_thumbnails",
        lambda *_args, **_kwargs: ThumbnailExtractionResult({}, {}),
    )
    monkeypatch.setattr("st_who_speaks.pipeline.infer_duration", lambda *_args: 1.0)

    media_path = tmp_path / "clip.webm"
    media_path.write_bytes(b"video")

    result = process_media(
        str(media_path),
        ProcessMediaOptions(
            execution_settings=CPU_EXECUTION,
            media_label="clip.webm",
            whisper_model_size="tiny",
            min_speakers=1,
            max_speakers=2,
            generate_thumbnails=False,
            enable_face_detection=False,
            max_thumbnails=6,
        ),
    )

    assert result.speakers == ["Person A"]
    assert result.speaker_turns[0].label == "Person A"
    assert result.metadata.diarization_warning is not None
    assert result.metadata.diarization_warning.startswith(
        "Local speaker clustering failed. Falling back to a single-speaker transcript."
    )
    assert "bad diarizer" in result.metadata.diarization_warning
    assert result.metadata.diagnostics == {}


def test_process_media_thumbnail_only_does_not_probe_face_detector(
    monkeypatch, tmp_path: Path
) -> None:
    def fake_collect_frame_assets(context):
        assert context.generate_thumbnails is True
        assert context.enable_face_detection is False
        return FrameAssetCollectionResult(
            thumbnails={0: b"thumb"},
            face_thumbnails={},
            face_counts_by_chunk={},
            face_detections={},
            chunks=context.chunks,
            wireframe_video_bytes=None,
            wireframe_faces_detected=0,
            face_detector_available=False,
            diagnostics={},
        )

    monkeypatch.setattr(
        "st_who_speaks.pipeline.extract_audio",
        lambda media_path, audio_path: Path(audio_path).write_bytes(b"RIFF"),
    )
    monkeypatch.setattr(
        "st_who_speaks.pipeline.transcribe_audio_data",
        lambda *_args, **_kwargs: (
            (WordToken("Hello", 0.0, 0.5, 0.9, "Unknown"),),
            (TranscriptionSegment("Hello", 0.0, 0.5, 0.9),),
        ),
    )
    monkeypatch.setattr(
        "st_who_speaks.pipeline.diarize_audio",
        lambda *_args, **_kwargs: [
            SpeakerTurn("speaker_0", "Person A", 0.0, 0.5),
        ],
    )
    monkeypatch.setattr(
        "st_who_speaks.pipeline.collect_frame_assets",
        fake_collect_frame_assets,
    )
    monkeypatch.setattr("st_who_speaks.pipeline.infer_duration", lambda *_args: 0.5)

    media_path = tmp_path / "clip.webm"
    media_path.write_bytes(b"video")

    result = process_media(
        str(media_path),
        ProcessMediaOptions(
            execution_settings=CPU_EXECUTION,
            media_label="clip.webm",
            whisper_model_size="tiny",
            min_speakers=1,
            max_speakers=2,
            generate_thumbnails=True,
            enable_face_detection=False,
            max_thumbnails=6,
        ),
    )

    assert result.thumbnails == {0: b"thumb"}
    assert result.metadata.face_detection_enabled is False
    assert result.metadata.face_detector_available is False
    assert result.metadata.generate_wireframe_video is False


def test_process_media_exercises_real_media_boundaries_with_stubbed_models(
    monkeypatch, tmp_path: Path
) -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise AssertionError("ffmpeg and ffprobe are required for media boundary tests")

    media_path = tmp_path / "tiny.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x64:rate=2:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(media_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    monkeypatch.setattr(
        "st_who_speaks.pipeline.transcribe_audio_data",
        lambda *_args, **_kwargs: (
            (
                WordToken("Boundary", 0.0, 0.4, 0.9, "Unknown"),
                WordToken("test", 0.4, 0.8, 0.9, "Unknown"),
            ),
            (TranscriptionSegment("Boundary test", 0.0, 0.8, 0.9),),
        ),
    )
    monkeypatch.setattr(
        "st_who_speaks.pipeline.diarize_audio",
        lambda *_args, **_kwargs: [
            SpeakerTurn("speaker_0", "Person A", 0.0, 0.8)
        ],
    )
    result = process_media(
        str(media_path),
        ProcessMediaOptions(
            execution_settings=CPU_EXECUTION,
            media_label="Friendly clip",
            whisper_model_size="tiny",
            min_speakers=1,
            max_speakers=2,
            generate_thumbnails=True,
            enable_face_detection=False,
            max_thumbnails=2,
        ),
    )

    assert result.display_name == "Friendly clip"
    assert result.media_identity == media_path.name
    assert result.chunks[0].text == "Boundary test"
    assert result.metadata.chunk_count == 1
    assert result.metadata.speaker_count == 1
    assert result.thumbnails
    assert result.metadata.opencv_available is True
    assert result.metadata.diarization_mode == "local-clustering"
