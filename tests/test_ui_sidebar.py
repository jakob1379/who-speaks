from __future__ import annotations

from typing import Any

from st_who_speaks.runtime import ExecutionSettings
from st_who_speaks.ui import app as app_module
from st_who_speaks.ui.sidebar import (
    SUPPORTED_UPLOAD_TYPES,
    build_process_media_options,
    render_sidebar_model_section,
)


def test_build_process_media_options_maps_ui_session_state() -> None:
    execution = ExecutionSettings(
        hardware_acceleration_enabled=False,
        backend_label=None,
        summary="CPU only",
        transcription_device="cpu",
        transcription_compute_type="int8",
        embedding_device="cpu",
    )

    options = build_process_media_options(
        "clip.webm",
        execution,
        {
            "whisper_model_size": "tiny",
            "min_speakers": 1,
            "max_speakers": 4,
            "generate_thumbnails": True,
            "enable_face_detection": True,
            "max_thumbnails": 6,
        },
    )

    assert options.execution_settings is execution
    assert options.media_label == "clip.webm"
    assert options.whisper_model_size == "tiny"
    assert options.min_speakers == 1
    assert options.max_speakers == 4
    assert options.generate_thumbnails is True
    assert options.enable_face_detection is True
    assert options.max_thumbnails == 6
    assert not hasattr(options, "generate_wireframe_video")


def test_supported_upload_types_include_web_video_formats() -> None:
    assert "webm" in SUPPORTED_UPLOAD_TYPES
    assert "ogv" in SUPPORTED_UPLOAD_TYPES


def test_render_sidebar_model_section_explains_whisper_sizes(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    monkeypatch.setattr(app_module.st, "header", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        app_module.st,
        "selectbox",
        lambda label, **kwargs: captured.update({"label": label, **kwargs}),
    )

    render_sidebar_model_section()

    assert captured["label"] == "Whisper model"
    assert captured["options"] == ["tiny", "base", "small", "medium"]
    assert "39M params" in captured["format_func"]("tiny")
    assert "244M params" in captured["format_func"]("small")
    assert "Larger Whisper models are slower" in captured["help"]
