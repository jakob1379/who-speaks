from __future__ import annotations

import altair as alt
from types import SimpleNamespace

from st_who_speaks import app as app_module
from st_who_speaks.app import (
    build_speaker_color_legend,
    SUPPORTED_UPLOAD_TYPES,
    describe_color_hex,
    build_timeline_chart,
    format_face_count,
    format_face_overlay_summary,
    format_seconds,
    include_chunk,
    render_media_player,
)
from st_who_speaks.models import (
    FaceDetectionFrame,
    LandmarkPoint,
    ProcessingResult,
    SpeakerTurn,
    TranscriptChunk,
)
from st_who_speaks.runtime import AccelerationStatus, resolve_execution_settings


def test_cli_launches_streamlit(monkeypatch) -> None:
    calls: list[list[str]] = []

    class CompletedProcess:
        returncode = 0

    monkeypatch.setattr(
        "st_who_speaks.cli.subprocess.run",
        lambda command, check=False: calls.append(command) or CompletedProcess(),
    )

    from st_who_speaks.cli import main

    try:
        main(["--server.headless", "true"])
    except SystemExit as error:
        assert error.code == 0

    assert calls
    assert calls[0][1:4] == ["-m", "streamlit", "run"]
    assert calls[0][-2:] == ["--server.headless", "true"]


def test_run_pipeline_uses_face_detection_toggle_for_wireframe(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class ProgressStub:
        def progress(self, *_args, **_kwargs) -> None:
            return None

        def empty(self) -> None:
            return None

    class EmptyStub:
        def caption(self, *_args, **_kwargs) -> None:
            return None

        def empty(self) -> None:
            return None

    def fake_process_media(*_args, **kwargs):
        captured.update(kwargs)
        return ProcessingResult(
            media_path="clip.webm",
            audio_path=None,
            duration=1.0,
            speakers=["Person A"],
            chunks=[],
            speaker_turns=[],
            metadata={"processing_time_seconds": 0.1},
        )

    monkeypatch.setattr(
        app_module,
        "resolve_execution_settings",
        lambda _enabled: {
            "hardware_acceleration_enabled": False,
            "transcription_device": "cpu",
            "transcription_compute_type": "int8",
            "embedding_device": "cpu",
        },
    )
    monkeypatch.setattr("st_who_speaks.pipeline.process_media", fake_process_media)
    monkeypatch.setattr(
        app_module.st, "progress", lambda *_args, **_kwargs: ProgressStub()
    )
    monkeypatch.setattr(app_module.st, "empty", lambda: EmptyStub())
    monkeypatch.setattr(app_module.st, "rerun", lambda: None)
    monkeypatch.setattr(app_module.st, "error", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        app_module.st,
        "session_state",
        {
            "hardware_acceleration": False,
            "whisper_model_size": "tiny",
            "min_speakers": 1,
            "max_speakers": 4,
            "generate_thumbnails": True,
            "enable_face_detection": True,
            "max_thumbnails": 6,
        },
    )

    app_module.run_pipeline(
        SimpleNamespace(name="clip.webm", getvalue=lambda: b"video-bytes")
    )

    assert captured["enable_face_detection"] is True
    assert captured["generate_wireframe_video"] is True


def test_format_seconds() -> None:
    assert format_seconds(65.7) == "01:05"
    assert format_seconds(3661) == "01:01:01"


def test_supported_upload_types_include_web_video_formats() -> None:
    assert "webm" in SUPPORTED_UPLOAD_TYPES
    assert "ogv" in SUPPORTED_UPLOAD_TYPES


def test_format_face_count() -> None:
    assert format_face_count(None, detection_enabled=False) == "face detection off"
    assert format_face_count(None, detection_enabled=True) == "not sampled"
    assert format_face_count(3, detection_enabled=True) == "3 detections"


def test_speaker_color_legend_and_overlay_summary_are_explicit() -> None:
    legend = build_speaker_color_legend(["Person A", "Person B", "Person C"])
    detection = FaceDetectionFrame(
        face_count=2,
        boxes=[],
        landmarks=[[LandmarkPoint(1, 2), LandmarkPoint(3, 4)]],
        speaker_label="Person A",
        color_hex="#ef4444",
    )
    chunk = TranscriptChunk(
        speaker="Person A",
        start=12.0,
        end=18.0,
        text="hello",
        confidence=0.9,
        word_count=1,
        face_count=2,
    )

    assert [entry.color_hex for entry in legend] == ["#ef4444", "#22c55e", "#3b82f6"]
    assert describe_color_hex("#ef4444") == "red (#ef4444)"
    summary = format_face_overlay_summary(chunk, detection, detection_enabled=True)
    assert "speaker color red (#ef4444)" in summary
    assert "landmarks 1 sets / 2 points" in summary


def test_include_chunk_filters_on_speaker_search_and_word_count() -> None:
    chunk = TranscriptChunk(
        speaker="Person B",
        start=0.0,
        end=1.0,
        text="This is a transcript chunk",
        confidence=0.8,
        word_count=5,
    )

    assert include_chunk(chunk, "All", "", 1) is True
    assert include_chunk(chunk, "Person B", "transcript", 5) is True
    assert include_chunk(chunk, "Person A", "", 1) is False
    assert include_chunk(chunk, "All", "missing", 1) is False
    assert include_chunk(chunk, "All", "", 6) is False


def test_render_media_player_routes_audio_and_video(monkeypatch) -> None:
    events: list[tuple[str, bytes, int, str | None, str | None, str | None]] = []

    chunks = [
        TranscriptChunk(
            speaker="Person A",
            start=1.0,
            end=3.5,
            text="Hello there.",
            confidence=0.9,
            word_count=2,
        )
    ]

    monkeypatch.setattr(
        "st_who_speaks.app.st.audio",
        lambda payload, start_time=0: events.append(
            ("audio", payload, start_time, None, None, None)
        ),
    )
    monkeypatch.setattr(
        "st_who_speaks.app.st.video",
        lambda payload, start_time=0, subtitles=None, format=None, width=None: (
            events.append(("video", payload, start_time, subtitles, format, width))
        ),
    )

    render_media_player(b"audio-bytes", "clip.wav", chunks=chunks, start_time=7)
    render_media_player(b"video-bytes", "clip.webm", chunks=chunks, start_time=9)

    assert events == [
        ("audio", b"audio-bytes", 7, None, None, None),
        (
            "video",
            b"video-bytes",
            9,
            "WEBVTT\n\n00:00:01.000 --> 00:00:03.500\nPerson A: Hello there.",
            "video/webm",
            "stretch",
        ),
    ]


def test_build_timeline_chart_returns_altair_chart() -> None:
    result = ProcessingResult(
        media_path="clip.webm",
        audio_path="clip.wav",
        duration=12.0,
        speakers=["Person A", "Person B"],
        chunks=[],
        speaker_turns=[
            SpeakerTurn("spk_0", "Person A", 0.0, 4.0),
            SpeakerTurn("spk_1", "Person B", 4.0, 8.0),
        ],
        metadata={},
    )

    chart = build_timeline_chart(result, selected_time=2.0)

    assert isinstance(chart, alt.LayerChart)


def test_build_webvtt_subtitles_prefixes_speaker_labels() -> None:
    subtitles = app_module.build_webvtt_subtitles(
        [
            TranscriptChunk(
                speaker="Person A",
                start=1.234,
                end=4.0,
                text="Hello\nthere",
                confidence=0.9,
                word_count=2,
            ),
            TranscriptChunk(
                speaker="Person B",
                start=4.0,
                end=6.75,
                text="Hi.",
                confidence=0.8,
                word_count=1,
            ),
        ]
    )

    assert subtitles == (
        "WEBVTT\n\n"
        "00:00:01.234 --> 00:00:04.000\n"
        "Person A: Hello there\n\n"
        "00:00:04.000 --> 00:00:06.750\n"
        "Person B: Hi."
    )


def test_format_subtitle_timestamp_uses_webvtt_shape() -> None:
    assert app_module.format_subtitle_timestamp(0.0) == "00:00:00.000"
    assert app_module.format_subtitle_timestamp(3661.042) == "01:01:01.042"


def test_resolve_execution_settings_respects_available_hardware(monkeypatch) -> None:
    monkeypatch.setattr(
        "st_who_speaks.runtime.detect_acceleration_status",
        lambda: AccelerationStatus(
            hardware_available=True,
            torch_gpu_available=True,
            whisper_gpu_available=False,
            backend_label="NVIDIA CUDA",
            summary="CUDA is available for embeddings only.",
            host_gpu_label="NVIDIA CUDA",
        ),
    )

    enabled = resolve_execution_settings(True)
    disabled = resolve_execution_settings(False)

    assert enabled["hardware_acceleration_enabled"] is True
    assert enabled["embedding_device"] == "cuda"
    assert enabled["transcription_device"] == "cpu"
    assert disabled["hardware_acceleration_enabled"] is False
    assert disabled["embedding_device"] == "cpu"


def test_detect_acceleration_status_supports_rocm_embeddings(monkeypatch) -> None:
    from st_who_speaks import runtime

    runtime.detect_acceleration_status.cache_clear()
    monkeypatch.setattr(
        "st_who_speaks.runtime.detect_host_gpu_label", lambda: "AMD ROCm"
    )
    monkeypatch.setattr(
        "st_who_speaks.runtime.detect_torch_backend_label", lambda: "AMD ROCm"
    )
    monkeypatch.setattr("st_who_speaks.runtime.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr(
        "st_who_speaks.runtime.ctranslate2.get_cuda_device_count", lambda: 0
    )

    status = runtime.detect_acceleration_status()

    assert status.hardware_available is True
    assert status.torch_gpu_available is True
    assert status.whisper_gpu_available is False
    assert status.backend_label == "AMD ROCm"
    assert "Whisper transcription will use CPU" in status.summary

    runtime.detect_acceleration_status.cache_clear()


def test_detect_acceleration_status_reports_amd_host_without_rocm_env(
    monkeypatch,
) -> None:
    from st_who_speaks import runtime

    runtime.detect_acceleration_status.cache_clear()
    monkeypatch.setattr(
        "st_who_speaks.runtime.detect_host_gpu_label", lambda: "AMD ROCm"
    )
    monkeypatch.setattr(
        "st_who_speaks.runtime.detect_torch_backend_label", lambda: None
    )
    monkeypatch.setattr("st_who_speaks.runtime.torch.cuda.is_available", lambda: False)

    status = runtime.detect_acceleration_status()

    assert status.hardware_available is False
    assert status.host_gpu_label == "AMD ROCm"
    assert "not ROCm-enabled" in status.summary

    runtime.detect_acceleration_status.cache_clear()
