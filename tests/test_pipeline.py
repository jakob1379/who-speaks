from __future__ import annotations

import builtins
import tempfile
import sys
import types
from pathlib import Path

import numpy as np

from st_who_speaks.models import (
    FaceBox,
    FaceDetectionFrame,
    LandmarkPoint,
    SpeakerTurn,
    TranscriptionSegment,
    WordToken,
)
from st_who_speaks.pipeline import (
    _ensure_huggingface_hub_compatibility,
    build_transcript_chunks,
    detect_and_annotate_faces,
    media_has_audio_stream,
    load_speaker_embedding_model,
    ProcessMediaOptions,
    process_media,
    summarize_ffmpeg_error,
)


def test_build_transcript_chunks_assigns_speakers_and_splits_on_gap() -> None:
    words = [
        WordToken("Hello", 0.0, 0.5, 0.9, "Unknown"),
        WordToken("there", 0.5, 1.0, 0.8, "Unknown"),
        WordToken("General", 3.0, 3.4, 0.7, "Unknown"),
        WordToken("Kenobi", 3.4, 4.0, 0.6, "Unknown"),
    ]
    turns = [
        SpeakerTurn("spk_0", "Person A", 0.0, 1.2),
        SpeakerTurn("spk_1", "Person B", 2.8, 4.2),
    ]

    chunks, speakers = build_transcript_chunks(words, turns)

    assert speakers == ["Person A", "Person B"]
    assert len(chunks) == 2
    assert chunks[0].speaker == "Person A"
    assert chunks[0].text == "Hello there"
    assert chunks[1].speaker == "Person B"
    assert chunks[1].text == "General Kenobi"


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
        "st_who_speaks.pipeline.load_face_detector",
        lambda: FakeDetector(),
    )
    monkeypatch.setattr(
        "st_who_speaks.pipeline.load_face_landmarker",
        lambda: FakeLandmarker(),
    )

    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    result = detect_and_annotate_faces(
        frame,
        speaker_label="Person A",
        color_hex="#ef4444",
    )

    assert result.face_count == 2
    assert result.boxes == [
        FaceBox(x=10, y=15, width=30, height=40),
        FaceBox(x=50, y=60, width=20, height=25),
    ]
    assert result.speaker_label == "Person A"
    assert result.color_hex == "#ef4444"
    assert result.landmarks == [
        [LandmarkPoint(x=16, y=27), LandmarkPoint(x=34, y=43)],
        [LandmarkPoint(x=54, y=68), LandmarkPoint(x=66, y=78)],
    ]
    assert result.annotated_image is not None


def test_process_media_with_local_diarization_and_face_data(
    monkeypatch, tmp_path: Path
) -> None:
    progress_events: list[tuple[str, float]] = []

    monkeypatch.setattr(
        "st_who_speaks.pipeline.resolve_execution_settings",
        lambda use_hardware_acceleration: {
            "hardware_acceleration_enabled": use_hardware_acceleration,
            "summary": "CUDA available",
            "backend_label": "NVIDIA CUDA",
            "transcription_device": "cuda",
            "transcription_compute_type": "float16",
            "embedding_device": "cuda",
        },
    )
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
        "st_who_speaks.pipeline.extract_thumbnails",
        lambda *_args, **_kwargs: (
            {0: b"thumb"},
            {0: b"face-thumb"},
            {0: 2},
            {
                0: FaceDetectionFrame(
                    face_count=2,
                    boxes=[FaceBox(1, 2, 3, 4)],
                    annotated_image=b"face-thumb",
                )
            },
        ),
    )
    monkeypatch.setattr(
        "st_who_speaks.pipeline.load_face_detector",
        lambda: object(),
    )
    monkeypatch.setattr(
        "st_who_speaks.pipeline.is_opencv_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "st_who_speaks.pipeline.build_wireframe_video",
        lambda *_args, **_kwargs: (b"wireframe-video", 4),
    )
    monkeypatch.setattr("st_who_speaks.pipeline.infer_duration", lambda *_args: 12.5)

    media_path = tmp_path / "clip.webm"
    media_path.write_bytes(b"video")

    result = process_media(
        str(media_path),
        ProcessMediaOptions(
            use_hardware_acceleration=True,
            media_label="clip.webm",
            whisper_model_size="tiny",
            transcription_device="cuda",
            transcription_compute_type="float16",
            embedding_device="cuda",
            hardware_acceleration_enabled=True,
            min_speakers=1,
            max_speakers=4,
            generate_thumbnails=True,
            enable_face_detection=True,
            max_thumbnails=6,
            generate_wireframe_video=True,
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
    assert result.metadata["face_detection_enabled"] is True
    assert result.metadata["total_faces_detected"] == 2
    assert result.metadata["speaker_colors"] == {"Person A": "#ef4444"}
    assert result.metadata["hardware_acceleration_enabled"] is True
    assert result.metadata["transcription_device"] == "cuda"
    assert result.metadata["embedding_device"] == "cuda"
    assert result.metadata["diarization_warning"] is None
    assert result.metadata["opencv_available"] is True
    assert result.metadata["face_detector_available"] is True
    assert result.metadata["generate_wireframe_video"] is True
    assert result.metadata["wireframe_video_available"] is True
    assert result.metadata["wireframe_faces_detected"] == 4
    assert result.face_detections[0].face_count == 2
    assert result.face_detections[0].speaker_label == "Person A"
    assert result.face_detections[0].color_hex == "#ef4444"
    assert result.wireframe_video_bytes == b"wireframe-video"
    assert progress_events[-1] == ("Done", 1.0)


def test_process_media_local_diarization_failure_degrades_gracefully(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "st_who_speaks.pipeline.resolve_execution_settings",
        lambda use_hardware_acceleration: {
            "hardware_acceleration_enabled": use_hardware_acceleration,
            "summary": "CPU only",
            "backend_label": None,
            "transcription_device": "cpu",
            "transcription_compute_type": "int8",
            "embedding_device": "cpu",
        },
    )
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
        "st_who_speaks.pipeline.extract_thumbnails",
        lambda *_args, **_kwargs: ({}, {}, {}, {}),
    )
    monkeypatch.setattr("st_who_speaks.pipeline.infer_duration", lambda *_args: 1.0)

    media_path = tmp_path / "clip.webm"
    media_path.write_bytes(b"video")

    result = process_media(
        str(media_path),
        ProcessMediaOptions(
            use_hardware_acceleration=False,
            media_label="clip.webm",
            whisper_model_size="tiny",
            transcription_device="cpu",
            transcription_compute_type="int8",
            embedding_device="cpu",
            hardware_acceleration_enabled=False,
            min_speakers=1,
            max_speakers=2,
            generate_thumbnails=False,
            enable_face_detection=False,
            max_thumbnails=6,
        ),
    )

    assert result.speakers == ["Person A"]
    assert result.speaker_turns[0].label == "Person A"
    assert "Local speaker clustering failed" in result.metadata["diarization_warning"]


def test_load_speaker_embedding_model_patches_torchaudio_before_import(
    monkeypatch,
) -> None:
    load_speaker_embedding_model.cache_clear()

    fake_torchaudio = types.ModuleType("torchaudio")
    monkeypatch.setitem(sys.modules, "torchaudio", fake_torchaudio)

    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "speechbrain.inference.classifiers":
            assert hasattr(fake_torchaudio, "list_audio_backends")
            speechbrain_pkg = sys.modules.setdefault(
                "speechbrain", types.ModuleType("speechbrain")
            )
            inference_pkg = sys.modules.setdefault(
                "speechbrain.inference", types.ModuleType("speechbrain.inference")
            )
            classifiers_pkg = types.ModuleType("speechbrain.inference.classifiers")

            class FakeEncoderClassifier:
                @classmethod
                def from_hparams(cls, **kwargs):
                    return {"kwargs": kwargs}

            setattr(classifiers_pkg, "EncoderClassifier", FakeEncoderClassifier)
            setattr(speechbrain_pkg, "inference", inference_pkg)
            setattr(inference_pkg, "classifiers", classifiers_pkg)
            sys.modules[name] = classifiers_pkg
            return classifiers_pkg
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    model = load_speaker_embedding_model("cpu")

    assert hasattr(fake_torchaudio, "list_audio_backends")
    assert fake_torchaudio.list_audio_backends() == ["soundfile", "sox_io"]
    assert model["kwargs"]["run_opts"] == {"device": "cpu"}
    assert model["kwargs"]["source"] == "speechbrain/spkrec-ecapa-voxceleb"


def test_huggingface_hub_compatibility_maps_use_auth_token(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class RemoteEntryNotFoundError(Exception):
        pass

    def original_hf_hub_download(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        if kwargs.get("filename") == "custom.py":
            raise RemoteEntryNotFoundError("missing custom.py")
        return "ok"

    fake_huggingface_hub = types.ModuleType("huggingface_hub")
    fake_file_download = types.SimpleNamespace(hf_hub_download=original_hf_hub_download)
    fake_hub_any = fake_huggingface_hub
    setattr(fake_hub_any, "hf_hub_download", original_hf_hub_download)
    setattr(fake_hub_any, "file_download", fake_file_download)

    monkeypatch.setattr(
        "st_who_speaks.pipeline.importlib.import_module",
        lambda name: fake_huggingface_hub if name == "huggingface_hub" else None,
    )

    _ensure_huggingface_hub_compatibility()
    stub_path = getattr(fake_huggingface_hub, "hf_hub_download")(
        "repo", filename="custom.py", use_auth_token="token"
    )
    assert stub_path == str(
        Path(tempfile.gettempdir())
        / "st-who-speaks"
        / "speechbrain-compat"
        / "custom.py"
    )

    getattr(fake_huggingface_hub, "hf_hub_download")(
        "repo", filename="hyperparams.yaml", use_auth_token="token"
    )

    assert captured["args"] == ("repo",)
    assert captured["kwargs"] == {"filename": "hyperparams.yaml", "token": "token"}


def test_media_has_audio_stream_uses_ffprobe(monkeypatch, tmp_path: Path) -> None:
    media_path = tmp_path / "clip.mp4"
    media_path.write_bytes(b"video")

    class CompletedProcess:
        def __init__(self, returncode: int, stdout: str):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = ""

    monkeypatch.setattr(
        "st_who_speaks.pipeline.shutil.which",
        lambda name: "/usr/bin/ffprobe" if name == "ffprobe" else None,
    )
    monkeypatch.setattr(
        "st_who_speaks.pipeline.subprocess.run",
        lambda *args, **kwargs: CompletedProcess(0, "audio\n"),
    )

    assert media_has_audio_stream(str(media_path)) is True


def test_summarize_ffmpeg_error_for_missing_audio_stream() -> None:
    stderr = "Output file does not contain any stream\nError opening output file /tmp/audio.wav"
    assert "does not contain an audio stream" in summarize_ffmpeg_error(stderr)
