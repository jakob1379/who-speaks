from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from st_who_speaks.ui import app as app_module
from st_who_speaks.models import (
    ProcessingMetadata,
    ProcessingResult,
)
from st_who_speaks.runtime import ExecutionSettings

def test_cli_launches_streamlit(monkeypatch) -> None:
    calls: list[tuple[str, list[str]]] = []

    monkeypatch.setattr(
        "st_who_speaks.cli.os.execv",
        lambda executable, command: calls.append((executable, command))
        or (_ for _ in ()).throw(SystemExit(0)),
    )

    from st_who_speaks.cli import main

    try:
        main(["--server.headless", "true"])
    except SystemExit as error:
        assert error.code == 0

    assert calls
    executable, command = calls[0]
    assert executable == command[0]
    assert command[1:4] == ["-m", "streamlit", "run"]
    assert command[-2:] == ["--server.headless", "true"]

def test_streamlit_entrypoint_configures_warning_filters(monkeypatch) -> None:
    from st_who_speaks import streamlit_app

    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    monkeypatch.setattr(
        streamlit_app.warnings,
        "filterwarnings",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    streamlit_app.configure_warning_filters()

    assert len(calls) == 2
    speechbrain_args, speechbrain_kwargs = calls[0]
    assert speechbrain_args == ("ignore",)
    assert speechbrain_kwargs["category"] is UserWarning
    assert "speechbrain" in str(speechbrain_kwargs["message"])

    torch_amp_args, torch_amp_kwargs = calls[1]
    assert torch_amp_args == ("ignore",)
    assert torch_amp_kwargs["category"] is FutureWarning
    assert "custom_fwd" in str(torch_amp_kwargs["message"])

def test_streamlit_entrypoint_configures_filters_before_ui(monkeypatch) -> None:
    from st_who_speaks import streamlit_app

    events: list[str] = []

    monkeypatch.setattr(
        streamlit_app,
        "configure_logging",
        lambda: events.append("logging"),
    )
    monkeypatch.setattr(
        streamlit_app,
        "configure_warning_filters",
        lambda: events.append("filters"),
    )
    monkeypatch.setattr(
        "st_who_speaks.ui.app.main",
        lambda: events.append("app"),
    )

    streamlit_app.main()

    assert events == ["logging", "filters", "app"]

def test_run_pipeline_updates_streamlit_state(monkeypatch) -> None:
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

    def fake_process_media(_media_path, options, _progress_callback):
        captured["media_path"] = _media_path
        captured["options"] = options
        return ProcessingResult(
            display_name="clip.webm",
            media_identity="clip.webm",
            audio_path=None,
            duration=1.0,
            speakers=["Person A"],
            chunks=[],
            speaker_turns=[],
            metadata=ProcessingMetadata(processing_time_seconds=0.1),
        )

    monkeypatch.setattr(
        app_module,
        "resolve_execution_settings",
        lambda *, use_hardware_acceleration: ExecutionSettings(
            hardware_acceleration_enabled=False,
            backend_label=None,
            summary="CPU only",
            transcription_device="cpu",
            transcription_compute_type="int8",
            embedding_device="cpu",
        ),
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

    assert Path(captured["media_path"]).name == "input.webm"
    assert (
        app_module.st.session_state[app_module.SESSION_VIDEO_BYTES_KEY]
        == b"video-bytes"
    )
    assert (
        app_module.st.session_state[app_module.SESSION_RESULT_KEY].display_name
        == "clip.webm"
    )
    assert app_module.st.session_state[app_module.SESSION_SHOW_WIREFRAME_VIDEO_KEY] is False
