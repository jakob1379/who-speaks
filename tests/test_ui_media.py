from __future__ import annotations

from st_who_speaks.models import TranscriptChunk
from st_who_speaks.ui.media import (
    build_webvtt_subtitles,
    format_subtitle_timestamp,
    render_media_player,
)


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
        "st_who_speaks.ui.media.st.audio",
        lambda payload, start_time=0: events.append(
            ("audio", payload, start_time, None, None, None)
        ),
    )
    monkeypatch.setattr(
        "st_who_speaks.ui.media.st.video",
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


def test_build_webvtt_subtitles_prefixes_speaker_labels() -> None:
    subtitles = build_webvtt_subtitles(
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
    assert format_subtitle_timestamp(0.0) == "00:00:00.000"
    assert format_subtitle_timestamp(3661.042) == "01:01:01.042"
