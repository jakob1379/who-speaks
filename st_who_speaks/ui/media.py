from __future__ import annotations

import os

import streamlit as st

from st_who_speaks.models import TranscriptChunk

VIDEO_MP4_MIME_TYPE = "video/mp4"
STRETCH_WIDTH = "stretch"


def render_media_player(
    payload: bytes,
    media_path: str,
    chunks: list[TranscriptChunk],
    *,
    start_time: int,
    video_format: str | None = None,
) -> None:
    suffix = os.path.splitext(media_path)[1].lower()
    if suffix in {".mp3", ".wav"}:
        st.audio(payload, start_time=start_time)
        return
    resolved_video_format = video_format or resolve_video_format(media_path)
    subtitles = build_webvtt_subtitles(chunks)
    if subtitles is None:
        st.video(
            payload,
            start_time=start_time,
            format=resolved_video_format,
            width=STRETCH_WIDTH,
        )
        return
    st.video(
        payload,
        start_time=start_time,
        format=resolved_video_format,
        subtitles=subtitles,
        width=STRETCH_WIDTH,
    )


def is_video_media(media_path: str) -> bool:
    return os.path.splitext(media_path)[1].lower() not in {".mp3", ".wav"}


def resolve_video_format(media_path: str) -> str:
    suffix = os.path.splitext(media_path)[1].lower()
    return {
        ".mp4": VIDEO_MP4_MIME_TYPE,
        ".m4v": VIDEO_MP4_MIME_TYPE,
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
        ".webm": "video/webm",
        ".mkv": "video/x-matroska",
        ".ogv": "video/ogg",
    }.get(suffix, VIDEO_MP4_MIME_TYPE)


def build_webvtt_subtitles(chunks: list[TranscriptChunk]) -> str | None:
    cues: list[str] = []
    for chunk in chunks:
        start = format_subtitle_timestamp(chunk.start)
        end = format_subtitle_timestamp(max(chunk.end, chunk.start + 0.001))
        text = collapse_subtitle_text(chunk.text)
        if not text:
            continue
        cues.append(f"{start} --> {end}\n{chunk.speaker}: {text}")
    if not cues:
        return None
    return "WEBVTT\n\n" + "\n\n".join(cues)


def collapse_subtitle_text(value: str) -> str:
    return " ".join(value.split())


def format_subtitle_timestamp(value: float) -> str:
    total_milliseconds = max(int(round(value * 1000)), 0)
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
