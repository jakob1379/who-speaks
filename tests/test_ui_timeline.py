from __future__ import annotations

import altair as alt
from typing import Any

from st_who_speaks.models import ProcessingResult, SpeakerTurn
from st_who_speaks.ui import app as app_module
from st_who_speaks.ui.app import render_timeline_controls
from st_who_speaks.ui.timeline import build_timeline_chart


def test_build_timeline_chart_returns_altair_chart() -> None:
    result = ProcessingResult(
        display_name="clip.webm",
        media_identity="clip.webm",
        audio_path="clip.wav",
        duration=12.0,
        speakers=["Person A", "Person B"],
        chunks=[],
        speaker_turns=[
            SpeakerTurn("spk_0", "Person A", 0.0, 4.0),
            SpeakerTurn("spk_1", "Person B", 4.0, 8.0),
        ],
    )

    chart = build_timeline_chart(result, selected_time=2.0)

    assert isinstance(chart, alt.LayerChart)


def test_render_timeline_controls_uses_session_key_without_default(monkeypatch) -> None:
    captured: dict[str, Any] = {}
    result = ProcessingResult(
        display_name="clip.webm",
        media_identity="clip.webm",
        audio_path="clip.wav",
        duration=12.0,
        speakers=["Person A"],
        chunks=[],
        speaker_turns=[],
    )

    monkeypatch.setattr(app_module.st, "altair_chart", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        app_module.st,
        "slider",
        lambda label, **kwargs: captured.update({"label": label, **kwargs}) or 3,
    )
    monkeypatch.setattr(app_module.st, "caption", lambda *_args, **_kwargs: None)

    render_timeline_controls(result, selected_time=3)

    assert captured["label"] == "Playback position"
    assert captured["key"] == app_module.SESSION_SELECTED_TIME_SECONDS_KEY
    assert "value" not in captured
