from __future__ import annotations

import math
import os
from dataclasses import asdict
import tempfile
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from st_who_speaks.models import (
    FaceDetectionFrame,
    ProcessingResult,
    SpeakerTurn,
    SpeakerColor,
    TranscriptChunk,
)
from st_who_speaks.runtime import detect_acceleration_status, resolve_execution_settings

SUPPORTED_UPLOAD_TYPES = [
    "mp4",
    "mov",
    "m4v",
    "avi",
    "webm",
    "mkv",
    "mp3",
    "wav",
    "ogv",
]

SPEAKER_COLOR_NAMES = {
    "#ef4444": "red",
    "#22c55e": "green",
    "#3b82f6": "blue",
    "#f59e0b": "amber",
    "#a855f7": "violet",
    "#14b8a6": "teal",
    "#f97316": "orange",
    "#eab308": "yellow",
}

SPEAKER_COLOR_PALETTE = list(SPEAKER_COLOR_NAMES.keys())


def build_speaker_color_map(speakers: list[str]) -> dict[str, str]:
    return {
        speaker: SPEAKER_COLOR_PALETTE[index % len(SPEAKER_COLOR_PALETTE)]
        for index, speaker in enumerate(speakers)
    }


def build_speaker_color_legend(speakers: list[str]) -> list[SpeakerColor]:
    color_map = build_speaker_color_map(speakers)
    return [
        SpeakerColor(label=speaker, color_hex=color_map[speaker])
        for speaker in speakers
    ]


def describe_color_hex(color_hex: str | None) -> str:
    if color_hex is None:
        return "unassigned"
    color_name = SPEAKER_COLOR_NAMES.get(color_hex.lower(), "custom")
    return f"{color_name} ({color_hex})"


def format_face_overlay_summary(
    chunk: TranscriptChunk,
    detection: FaceDetectionFrame | None,
    *,
    detection_enabled: bool,
) -> str:
    parts = [
        f"{chunk.speaker}",
        f"{format_seconds(chunk.start)} → {format_seconds(chunk.end)}",
        format_face_count(
            detection.face_count if detection is not None else None,
            detection_enabled=detection_enabled,
        ),
    ]
    if detection is not None and detection.speaker_label:
        parts.append(f"speaker color {describe_color_hex(detection.color_hex)}")
    if detection is not None and detection.landmarks:
        landmark_points = sum(len(points) for points in detection.landmarks)
        parts.append(
            f"landmarks {len(detection.landmarks)} sets / {landmark_points} points"
        )
    return " · ".join(parts)


def main() -> None:
    st.set_page_config(page_title="Who Speaks", page_icon="🎙️", layout="wide")
    st.title("Who Speaks")
    st.caption(
        "Upload a local video or audio file, transcribe it, run open local diarization, optionally detect faces in sampled frames, and inspect the timeline chunk-by-chunk."
    )

    state_defaults = {
        "result": None,
        "video_bytes": None,
        "selected_chunk_index": 0,
        "selected_time_seconds": 0,
        "show_wireframe_video": False,
        "last_uploaded_name": None,
    }
    for key, value in state_defaults.items():
        st.session_state.setdefault(key, value)

    uploaded_file, run_requested = render_sidebar()
    if (
        uploaded_file is not None
        and uploaded_file.name != st.session_state["last_uploaded_name"]
    ):
        st.session_state["result"] = None
        st.session_state["video_bytes"] = uploaded_file.getvalue()
        st.session_state["selected_chunk_index"] = 0
        st.session_state["selected_time_seconds"] = 0
        st.session_state["show_wireframe_video"] = False
        st.session_state["last_uploaded_name"] = uploaded_file.name

    if st.session_state["result"] is None:
        st.info("Upload a media file and run the pipeline.")
    else:
        render_result(st.session_state["result"], st.session_state["video_bytes"])

    if uploaded_file is not None and run_requested:
        run_pipeline(uploaded_file)


def render_sidebar():
    with st.sidebar:
        st.header("Input")
        uploaded_file = st.file_uploader(
            "Video or audio file",
            type=SUPPORTED_UPLOAD_TYPES,
        )

        st.header("Models")
        st.selectbox(
            "Whisper model",
            options=["tiny", "base", "small", "medium"],
            index=2,
            key="whisper_model_size",
        )

        acceleration_status = detect_acceleration_status()
        st.header("Local diarization")
        st.toggle(
            "Hardware acceleration",
            value=acceleration_status.hardware_available,
            disabled=not acceleration_status.hardware_available,
            key="hardware_acceleration",
            help=acceleration_status.summary,
        )
        if not acceleration_status.hardware_available:
            st.caption(acceleration_status.summary)
        else:
            st.caption(
                f"{acceleration_status.summary} · Toggle on to use it, off to stay on CPU."
            )
        st.number_input(
            "Min speakers", min_value=1, max_value=10, value=1, key="min_speakers"
        )
        st.number_input(
            "Max speakers", min_value=1, max_value=10, value=4, key="max_speakers"
        )

        st.header("Frames")
        st.toggle(
            "Extract OpenCV thumbnails",
            value=True,
            key="generate_thumbnails",
        )
        st.toggle(
            "Enable face detection + wireframe",
            value=True,
            key="enable_face_detection",
            help="Runs OpenCV face detection on sampled frames and generates a second processed video with landmark wireframes. Use the player toggle to switch between the original and processed videos.",
        )
        st.slider(
            "Max thumbnails",
            min_value=6,
            max_value=48,
            value=18,
            step=6,
            key="max_thumbnails",
        )

        run_requested = st.button("Run pipeline", type="primary")

    return uploaded_file, run_requested


def run_pipeline(uploaded_file) -> None:
    from st_who_speaks.pipeline import process_media

    video_bytes = uploaded_file.getvalue()
    execution = resolve_execution_settings(st.session_state["hardware_acceleration"])

    progress_bar = st.progress(0.0, text="Queued")
    status_box = st.empty()

    def update_progress(label: str, progress: float) -> None:
        progress_bar.progress(progress, text=label)
        status_box.caption(label)

    with tempfile.TemporaryDirectory(prefix="st-who-speaks-upload-") as upload_dir:
        suffix = Path(uploaded_file.name).suffix or ".mp4"
        media_path = Path(upload_dir) / f"input{suffix}"
        media_path.write_bytes(video_bytes)
        enable_face_detection = bool(st.session_state["enable_face_detection"])

        try:
            result = process_media(
                str(media_path),
                use_hardware_acceleration=bool(
                    st.session_state["hardware_acceleration"]
                ),
                media_label=uploaded_file.name,
                whisper_model_size=st.session_state["whisper_model_size"],
                transcription_device=execution["transcription_device"],
                transcription_compute_type=execution["transcription_compute_type"],
                embedding_device=execution["embedding_device"],
                hardware_acceleration_enabled=execution[
                    "hardware_acceleration_enabled"
                ],
                min_speakers=int(st.session_state["min_speakers"]),
                max_speakers=int(st.session_state["max_speakers"]),
                generate_thumbnails=st.session_state["generate_thumbnails"],
                enable_face_detection=enable_face_detection,
                generate_wireframe_video=enable_face_detection,
                max_thumbnails=int(st.session_state["max_thumbnails"]),
                progress_callback=update_progress,
            )
        except Exception as error:
            progress_bar.empty()
            status_box.empty()
            st.error(str(error))
            return

    progress_bar.empty()
    status_box.empty()
    st.session_state["result"] = result
    st.session_state["video_bytes"] = video_bytes
    st.session_state["selected_chunk_index"] = 0
    st.session_state["selected_time_seconds"] = 0
    st.session_state["show_wireframe_video"] = result.wireframe_video_bytes is not None
    st.rerun()


def render_result(result: ProcessingResult, video_bytes: bytes | None) -> None:
    render_metrics(result)
    st.caption(
        "OpenCV face detection + facial landmarks run on sampled frames only and do not identify people."
    )

    selected_time = pick_selected_time(result)
    selected_chunk_match = pick_active_chunk_for_time(result, float(selected_time))
    selected_index = (
        selected_chunk_match[0] if selected_chunk_match is not None else None
    )
    selected_chunk = (
        selected_chunk_match[1] if selected_chunk_match is not None else None
    )
    if selected_index is not None:
        st.session_state["selected_chunk_index"] = selected_index
    left_column, right_column = st.columns([1.4, 1.0], gap="large")
    with left_column:
        st.subheader("Media")
        show_wireframe_video = False
        if (
            result.metadata.get("generate_wireframe_video")
            and result.wireframe_video_bytes is None
        ):
            if not result.metadata.get("opencv_available", True):
                st.warning(
                    "Wireframe video could not be generated because OpenCV is not available in this runtime. "
                    f"Import error: {result.metadata.get('opencv_import_error') or 'unknown error'}"
                )
            elif not result.metadata.get("face_detector_available", False):
                st.warning(
                    "Wireframe video could not be generated because the Haar cascade face detector is unavailable, even after trying to download the XML."
                )
            else:
                st.warning(
                    "Wireframe video was requested, but no face overlays were produced. The detector ran and found zero faces in the processed frames."
                )
        if result.wireframe_video_bytes is not None and is_video_media(
            result.media_path
        ):
            show_wireframe_video = st.toggle(
                "Show facial wireframe overlay",
                key="show_wireframe_video",
                help="Switch between the original upload and the processed wireframe overlay video.",
            )
        media_payload = (
            result.wireframe_video_bytes
            if show_wireframe_video and result.wireframe_video_bytes is not None
            else video_bytes
        )
        if is_video_media(result.media_path):
            st.caption(
                "Showing processed wireframe video"
                if show_wireframe_video and result.wireframe_video_bytes is not None
                else "Showing original video"
            )
        if media_payload is not None:
            render_media_player(
                media_payload,
                result.media_path,
                result.chunks,
                start_time=selected_time,
                video_format="video/mp4" if show_wireframe_video else None,
            )
        render_timeline_controls(result, selected_time)

    with right_column:
        if selected_chunk is not None and selected_index is not None:
            st.subheader("Current segment")
            render_selected_chunk(result, selected_index, selected_chunk)

    transcript_tab, frames_tab, faces_tab, debug_tab = st.tabs(
        ["Transcript", "Frames", "Faces", "Debug"]
    )
    with transcript_tab:
        render_transcript_tab(result)
    with frames_tab:
        render_frames_tab(result)
    with faces_tab:
        render_faces_tab(result)
    with debug_tab:
        render_debug_tab(result)


def render_metrics(result: ProcessingResult) -> None:
    column_one, column_two, column_three, column_four, column_five = st.columns(5)
    column_one.metric("Duration", format_seconds(result.duration))
    column_two.metric("Speakers", str(len(result.speakers)))
    column_three.metric("Chunks", str(len(result.chunks)))
    column_four.metric(
        "Face detections", str(result.metadata.get("total_faces_detected", 0))
    )
    column_five.metric(
        "Processing time", f"{result.metadata['processing_time_seconds']}s"
    )
    if result.metadata.get("diarization_warning"):
        st.warning(result.metadata["diarization_warning"])


def pick_selected_chunk(result: ProcessingResult) -> TranscriptChunk | None:
    if not result.chunks:
        return None
    index = min(st.session_state.get("selected_chunk_index", 0), len(result.chunks) - 1)
    st.session_state["selected_chunk_index"] = index
    return result.chunks[index]


def pick_selected_time(result: ProcessingResult) -> int:
    maximum = max(int(math.ceil(result.duration)), 0)
    selected_time = min(st.session_state.get("selected_time_seconds", 0), maximum)
    st.session_state["selected_time_seconds"] = selected_time
    return selected_time


def pick_active_chunk_for_time(
    result: ProcessingResult, selected_time: float
) -> tuple[int, TranscriptChunk] | None:
    for index, chunk in enumerate(result.chunks):
        if chunk.start <= selected_time < chunk.end:
            return index, chunk
    if selected_time >= result.duration:
        if not result.chunks:
            return None
        last_index = len(result.chunks) - 1
        return last_index, result.chunks[last_index]
    return None


def pick_active_speaker_turn(
    result: ProcessingResult, selected_time: float
) -> SpeakerTurn | None:
    for turn in result.speaker_turns:
        if turn.start <= selected_time < turn.end:
            return turn
    if result.speaker_turns and selected_time >= result.duration:
        return result.speaker_turns[-1]
    return None


def render_selected_chunk(
    result: ProcessingResult, index: int, chunk: TranscriptChunk
) -> None:
    with st.container(border=True):
        preview = result.face_detections.get(index)
        has_face_preview = preview is not None and preview.annotated_image is not None
        if has_face_preview and preview is not None:
            preview_image = preview.annotated_image
        else:
            preview_image = result.face_thumbnails.get(index) or result.thumbnails.get(
                index
            )
        if preview_image is not None:
            detection = preview or result.face_detections.get(index)
            preview_label = (
                "Landmark-aware face overlay"
                if has_face_preview and preview is not None and preview.landmarks
                else "Speaker-colored face overlay"
                if has_face_preview
                else "Frame preview"
            )
            caption = (
                f"{format_face_overlay_summary(chunk, detection, detection_enabled=result.metadata.get('face_detection_enabled', False))} · {preview_label}"
                if detection is not None
                else f"{format_face_count(chunk.face_count, detection_enabled=result.metadata.get('face_detection_enabled', False))} · {preview_label}"
            )
            st.image(
                preview_image,
                width="stretch",
                caption=caption,
            )
        st.markdown(
            f"**{chunk.speaker}** · {format_seconds(chunk.start)} → {format_seconds(chunk.end)}"
        )
        st.write(chunk.text)
        if chunk.confidence is not None:
            st.caption(
                f"Average confidence: {chunk.confidence:.3f} · {chunk.word_count} words"
            )
        else:
            st.caption(f"{chunk.word_count} words")


def render_timeline_controls(result: ProcessingResult, selected_time: int) -> None:
    timeline = build_timeline_chart(result, float(selected_time))
    st.altair_chart(timeline, width="stretch")
    maximum = max(int(math.ceil(result.duration)), 0)
    selected_time = st.slider(
        "Playback position",
        min_value=0,
        max_value=maximum,
        value=selected_time,
        step=1,
        format="%d s",
        key="selected_time_seconds",
    )
    active_turn = pick_active_speaker_turn(result, float(selected_time))
    if active_turn is None:
        st.caption(f"No active speaker at {format_seconds(selected_time)}.")
        return
    st.caption(
        f"{active_turn.label} speaking at {format_seconds(selected_time)} · {format_seconds(active_turn.start)} → {format_seconds(active_turn.end)}"
    )


def build_timeline_chart(result: ProcessingResult, selected_time: float) -> alt.Chart:
    speaker_colors = build_speaker_color_map(result.speakers)
    turns_frame = pd.DataFrame(
        [
            {
                "speaker": turn.label,
                "start": turn.start,
                "end": turn.end,
                "duration": turn.duration,
                "track": "Speaker activity",
            }
            for turn in result.speaker_turns
        ]
    )
    if turns_frame.empty:
        turns_frame = pd.DataFrame(
            [
                {
                    "speaker": "No speaker data",
                    "start": 0.0,
                    "end": max(result.duration, 0.0),
                    "duration": 0.0,
                    "track": "Speaker activity",
                }
            ]
        )
    duration_limit = max(result.duration, 1.0)
    bars = (
        alt.Chart(turns_frame)
        .mark_bar(size=28, cornerRadius=6)
        .encode(
            x=alt.X(
                "start:Q",
                title="Seconds",
                scale=alt.Scale(domain=[0, duration_limit]),
            ),
            x2="end:Q",
            y=alt.Y("track:N", title=None, axis=None),
            color=(
                alt.Color(
                    "speaker:N",
                    scale=alt.Scale(
                        domain=list(speaker_colors.keys()),
                        range=list(speaker_colors.values()),
                    ),
                    legend=None,
                )
                if speaker_colors
                else alt.Color("speaker:N", legend=None)
            ),
            tooltip=[
                alt.Tooltip("speaker:N", title="Speaker"),
                alt.Tooltip("start:Q", title="Start", format=".2f"),
                alt.Tooltip("end:Q", title="End", format=".2f"),
                alt.Tooltip("duration:Q", title="Duration", format=".2f"),
            ],
        )
    )
    indicator_frame = pd.DataFrame(
        [{"selected_time": min(max(selected_time, 0.0), duration_limit)}]
    )
    indicator = (
        alt.Chart(indicator_frame)
        .mark_rule(
            color="#111827",
            strokeWidth=3,
        )
        .encode(x=alt.X("selected_time:Q", scale=alt.Scale(domain=[0, duration_limit])))
    )
    return (bars + indicator).properties(height=72).configure_view(strokeWidth=0)


def render_transcript_tab(result: ProcessingResult) -> None:
    filter_column, search_column, merge_column = st.columns([1, 1, 1])
    with filter_column:
        speaker_filter = st.selectbox(
            "Speaker", options=["All", *result.speakers], index=0
        )
    with search_column:
        search_term = st.text_input("Search transcript")
    with merge_column:
        minimum_words = st.slider("Minimum words", min_value=1, max_value=20, value=1)

    filtered_chunks = {
        index
        for index, chunk in enumerate(result.chunks)
        if include_chunk(chunk, speaker_filter, search_term, minimum_words)
    }
    st.caption(f"Showing {len(filtered_chunks)} of {len(result.chunks)} chunks")

    for index, chunk in enumerate(result.chunks):
        if index not in filtered_chunks:
            continue
        with st.container(border=True):
            st.markdown(
                f"**{chunk.speaker}** · {format_seconds(chunk.start)} → {format_seconds(chunk.end)}"
            )
            st.write(chunk.text)
            footer = f"{chunk.word_count} words · {format_face_count(chunk.face_count, detection_enabled=result.metadata.get('face_detection_enabled', False))}"
            if chunk.confidence is not None:
                footer = f"{footer} · confidence {chunk.confidence:.3f}"
            st.caption(footer)
            thumbnail = result.face_thumbnails.get(index) or result.thumbnails.get(
                index
            )
            if thumbnail:
                st.image(thumbnail, width="stretch")


def include_chunk(
    chunk: TranscriptChunk, speaker_filter: str, search_term: str, minimum_words: int
) -> bool:
    if speaker_filter != "All" and chunk.speaker != speaker_filter:
        return False
    if chunk.word_count < minimum_words:
        return False
    if search_term and search_term.lower() not in chunk.text.lower():
        return False
    return True


def render_frames_tab(result: ProcessingResult) -> None:
    frame_previews = result.thumbnails or result.face_thumbnails
    if not frame_previews:
        st.info(
            "No frame previews are available. If face detection was enabled, OpenCV likely couldn't read sampled frames from this upload. Otherwise, enable thumbnails or wireframe processing and rerun."
        )
        return

    items = list(frame_previews.items())
    for row_start in range(0, len(items), 3):
        row = items[row_start : row_start + 3]
        columns = st.columns(len(row))
        for column, (index, image_bytes) in zip(columns, row, strict=False):
            chunk = result.chunks[index]
            detection = result.face_detections.get(index)
            with column:
                st.image(
                    image_bytes,
                    width="stretch",
                    caption=f"{chunk.speaker} · {format_seconds(chunk.start)} · speaker color {describe_color_hex(detection.color_hex if detection is not None else None)} · {format_face_count(chunk.face_count, detection_enabled=result.metadata.get('face_detection_enabled', False))}",
                )


def render_faces_tab(result: ProcessingResult) -> None:
    if not result.face_detections:
        if result.metadata.get("face_detection_enabled"):
            st.info(
                "Face detection was enabled, but no sampled video frames were successfully processed. The usual cause is an unreadable video stream for OpenCV, not the toggle."
            )
        else:
            st.info(
                "No face detection results available. Enable face detection in the sidebar before running the pipeline."
            )
        return

    items = list(result.face_detections.items())
    st.caption(
        f"Showing {len(items)} sampled frames with {result.metadata.get('total_faces_detected', 0)} total faces detected. Overlays use diarized speaker colors and facial landmarks when available."
    )
    for row_start in range(0, len(items), 3):
        row = items[row_start : row_start + 3]
        columns = st.columns(len(row))
        for column, (index, detection) in zip(columns, row, strict=False):
            chunk = result.chunks[index]
            with column:
                preview = detection.annotated_image or result.face_thumbnails.get(index)
                if preview:
                    st.image(
                        preview,
                        width="stretch",
                        caption=format_face_overlay_summary(
                            chunk, detection, detection_enabled=True
                        ),
                    )
                else:
                    st.caption(
                        format_face_overlay_summary(
                            chunk, detection, detection_enabled=True
                        )
                    )
                if detection.boxes:
                    st.caption(
                        ", ".join(
                            f"{box.x},{box.y} {box.width}x{box.height}"
                            for box in detection.boxes
                        )
                    )


def render_debug_tab(result: ProcessingResult) -> None:
    st.json(
        {
            "speakers": result.speakers,
            "speaker_turns": [asdict(turn) for turn in result.speaker_turns],
            "chunks": [asdict(chunk) for chunk in result.chunks],
            "face_detections": [
                {
                    "frame_index": index,
                    "face_count": detection.face_count,
                    "speaker_label": detection.speaker_label,
                    "color_hex": detection.color_hex,
                    "boxes": [asdict(box) for box in detection.boxes],
                    "landmarks": [
                        [asdict(point) for point in face_landmarks]
                        for face_landmarks in detection.landmarks
                    ],
                }
                for index, detection in result.face_detections.items()
            ],
            "metadata": result.metadata,
        }
    )


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
            width="stretch",
        )
        return
    st.video(
        payload,
        start_time=start_time,
        format=resolved_video_format,
        subtitles=subtitles,
        width="stretch",
    )


def is_video_media(media_path: str) -> bool:
    return os.path.splitext(media_path)[1].lower() not in {".mp3", ".wav"}


def resolve_video_format(media_path: str) -> str:
    suffix = os.path.splitext(media_path)[1].lower()
    return {
        ".mp4": "video/mp4",
        ".m4v": "video/mp4",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
        ".webm": "video/webm",
        ".mkv": "video/x-matroska",
        ".ogv": "video/ogg",
    }.get(suffix, "video/mp4")


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


def format_seconds(value: float) -> str:
    total_seconds = max(int(value), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def format_face_count(value: int | None, *, detection_enabled: bool) -> str:
    if value is None:
        return "not sampled" if detection_enabled else "face detection off"
    return f"{value} detections"
