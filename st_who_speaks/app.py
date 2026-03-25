from __future__ import annotations

import math
import os
from dataclasses import asdict
import tempfile
from pathlib import Path
from typing import Any

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
from st_who_speaks.runtime import detect_acceleration_status

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

WHISPER_MODEL_OPTIONS = ["tiny", "base", "small", "medium"]
WHISPER_MODEL_METRICS = {
    "tiny": "39M params · fastest · lowest accuracy",
    "base": "74M params · fast · light upgrade",
    "small": "244M params · balanced speed/accuracy",
    "medium": "769M params · slowest · strongest accuracy",
}
WHISPER_MODEL_HELP = (
    "Larger Whisper models are slower and use more memory, but usually improve "
    "transcription accuracy. Tiny is quickest; medium is the most accurate of "
    "these four."
)

SESSION_RESULT_KEY = "result"
SESSION_VIDEO_BYTES_KEY = "video_bytes"
SESSION_SELECTED_CHUNK_INDEX_KEY = "selected_chunk_index"
SESSION_SELECTED_TIME_SECONDS_KEY = "selected_time_seconds"
SESSION_SHOW_WIREFRAME_VIDEO_KEY = "show_wireframe_video"
SESSION_LAST_UPLOADED_NAME_KEY = "last_uploaded_name"
SESSION_HARDWARE_ACCELERATION_KEY = "hardware_acceleration"

VIDEO_MP4_MIME_TYPE = "video/mp4"
STRETCH_WIDTH = "stretch"
FACE_DETECTION_ENABLED_KEY = "face_detection_enabled"
SPEAKER_N_FIELD = "speaker:N"


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


def face_detection_enabled(result: ProcessingResult) -> bool:
    return bool(result.metadata.get(FACE_DETECTION_ENABLED_KEY, False))


def format_whisper_model_option(model_size: str) -> str:
    return f"{model_size} — {WHISPER_MODEL_METRICS[model_size]}"


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


def initialize_session_state() -> None:
    state_defaults = {
        SESSION_RESULT_KEY: None,
        SESSION_VIDEO_BYTES_KEY: None,
        SESSION_SELECTED_CHUNK_INDEX_KEY: 0,
        SESSION_SELECTED_TIME_SECONDS_KEY: 0,
        SESSION_SHOW_WIREFRAME_VIDEO_KEY: False,
        SESSION_LAST_UPLOADED_NAME_KEY: None,
    }
    for key, value in state_defaults.items():
        st.session_state.setdefault(key, value)


def reset_uploaded_file_state(uploaded_file) -> None:
    st.session_state[SESSION_RESULT_KEY] = None
    st.session_state[SESSION_VIDEO_BYTES_KEY] = uploaded_file.getvalue()
    st.session_state[SESSION_SELECTED_CHUNK_INDEX_KEY] = 0
    st.session_state[SESSION_SELECTED_TIME_SECONDS_KEY] = 0
    st.session_state[SESSION_SHOW_WIREFRAME_VIDEO_KEY] = False
    st.session_state[SESSION_LAST_UPLOADED_NAME_KEY] = uploaded_file.name


def build_execution_settings(*, use_hardware_acceleration: bool) -> dict[str, Any]:
    status = detect_acceleration_status()
    acceleration_enabled = use_hardware_acceleration and status.hardware_available
    transcription_device = (
        "cuda" if acceleration_enabled and status.whisper_gpu_available else "cpu"
    )
    embedding_device = (
        "cuda" if acceleration_enabled and status.torch_gpu_available else "cpu"
    )
    return {
        "hardware_acceleration_enabled": acceleration_enabled,
        "transcription_device": transcription_device,
        "transcription_compute_type": "float16"
        if transcription_device == "cuda"
        else "int8",
        "embedding_device": embedding_device,
    }


def resolve_execution_settings(use_hardware_acceleration: bool) -> dict[str, Any]:
    return build_execution_settings(use_hardware_acceleration=use_hardware_acceleration)


def main() -> None:
    st.set_page_config(page_title="Who Speaks", page_icon="🎙️", layout="wide")
    st.title("Who Speaks")
    st.caption(
        "Upload a local video or audio file, transcribe it, run open local diarization, optionally detect faces in sampled frames, and inspect the timeline chunk-by-chunk."
    )

    initialize_session_state()

    uploaded_file, run_requested = render_sidebar()
    if (
        uploaded_file is not None
        and uploaded_file.name != st.session_state[SESSION_LAST_UPLOADED_NAME_KEY]
    ):
        reset_uploaded_file_state(uploaded_file)

    if st.session_state[SESSION_RESULT_KEY] is None:
        st.info("Upload a media file and run the pipeline.")
    else:
        render_result(
            st.session_state[SESSION_RESULT_KEY],
            st.session_state[SESSION_VIDEO_BYTES_KEY],
        )

    if uploaded_file is not None and run_requested:
        run_pipeline(uploaded_file)


def render_sidebar():
    with st.sidebar:
        uploaded_file = render_sidebar_input_section()
        render_sidebar_model_section()
        render_sidebar_local_diarization_section()
        render_sidebar_frame_section()
        run_requested = st.button("Run pipeline", type="primary")

    return uploaded_file, run_requested


def render_sidebar_input_section():
    st.header("Input")
    return st.file_uploader(
        "Video or audio file",
        type=SUPPORTED_UPLOAD_TYPES,
    )


def render_sidebar_model_section() -> None:
    st.header("Models")
    st.selectbox(
        "Whisper model",
        options=WHISPER_MODEL_OPTIONS,
        index=2,
        key="whisper_model_size",
        format_func=format_whisper_model_option,
        help=WHISPER_MODEL_HELP,
    )


def render_sidebar_local_diarization_section() -> None:
    acceleration_status = detect_acceleration_status()
    st.header("Local diarization")
    st.toggle(
        "Hardware acceleration",
        value=acceleration_status.hardware_available,
        disabled=not acceleration_status.hardware_available,
        key=SESSION_HARDWARE_ACCELERATION_KEY,
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


def render_sidebar_frame_section() -> None:
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


def run_pipeline(uploaded_file) -> None:
    video_bytes = uploaded_file.getvalue()
    execution = resolve_execution_settings(
        use_hardware_acceleration=st.session_state[SESSION_HARDWARE_ACCELERATION_KEY]
    )

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
            result = process_uploaded_media(
                str(media_path),
                media_label=uploaded_file.name,
                execution=execution,
                enable_face_detection=enable_face_detection,
                progress_callback=update_progress,
            )
        except Exception as error:
            progress_bar.empty()
            status_box.empty()
            st.error(str(error))
            return

    progress_bar.empty()
    status_box.empty()
    st.session_state[SESSION_RESULT_KEY] = result
    st.session_state[SESSION_VIDEO_BYTES_KEY] = video_bytes
    st.session_state[SESSION_SELECTED_CHUNK_INDEX_KEY] = 0
    st.session_state[SESSION_SELECTED_TIME_SECONDS_KEY] = 0
    st.session_state[SESSION_SHOW_WIREFRAME_VIDEO_KEY] = (
        result.wireframe_video_bytes is not None
    )
    st.rerun()


def process_uploaded_media(
    media_path: str,
    *,
    media_label: str,
    execution: dict[str, Any],
    enable_face_detection: bool,
    progress_callback,
) -> ProcessingResult:
    from st_who_speaks.pipeline import ProcessMediaOptions, process_media

    options = ProcessMediaOptions(
        use_hardware_acceleration=bool(
            st.session_state[SESSION_HARDWARE_ACCELERATION_KEY]
        ),
        media_label=media_label,
        whisper_model_size=st.session_state["whisper_model_size"],
        transcription_device=execution["transcription_device"],
        transcription_compute_type=execution["transcription_compute_type"],
        embedding_device=execution["embedding_device"],
        hardware_acceleration_enabled=execution["hardware_acceleration_enabled"],
        min_speakers=int(st.session_state["min_speakers"]),
        max_speakers=int(st.session_state["max_speakers"]),
        generate_thumbnails=st.session_state["generate_thumbnails"],
        enable_face_detection=enable_face_detection,
        max_thumbnails=int(st.session_state["max_thumbnails"]),
        generate_wireframe_video=enable_face_detection,
    )
    return process_media(media_path, options, progress_callback)


def render_result(result: ProcessingResult, video_bytes: bytes | None) -> None:
    render_metrics(result)
    st.caption(
        "OpenCV face detection + facial landmarks run on sampled frames only and do not identify people."
    )

    selected_time = pick_selected_time(result)
    selected_index, selected_chunk = selected_chunk_for_time(result, selected_time)
    if selected_index is not None:
        st.session_state[SESSION_SELECTED_CHUNK_INDEX_KEY] = selected_index
    left_column, right_column = st.columns([1.4, 1.0], gap="large")
    with left_column:
        render_result_media(result, video_bytes, selected_time)

    with right_column:
        render_result_selected_chunk(result, selected_index, selected_chunk)

    render_result_tabs(result)


def selected_chunk_for_time(
    result: ProcessingResult, selected_time: int
) -> tuple[int | None, TranscriptChunk | None]:
    selected_chunk_match = pick_active_chunk_for_time(result, float(selected_time))
    if selected_chunk_match is None:
        return None, None
    return selected_chunk_match[0], selected_chunk_match[1]


def render_result_media(
    result: ProcessingResult, video_bytes: bytes | None, selected_time: int
) -> None:
    st.subheader("Media")
    show_wireframe_video = False
    render_wireframe_video_warning(result)
    if result.wireframe_video_bytes is not None and is_video_media(result.media_path):
        show_wireframe_video = st.toggle(
            "Show facial wireframe overlay",
            key=SESSION_SHOW_WIREFRAME_VIDEO_KEY,
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
            video_format=VIDEO_MP4_MIME_TYPE if show_wireframe_video else None,
        )
    render_timeline_controls(result, selected_time)


def render_wireframe_video_warning(result: ProcessingResult) -> None:
    if (
        not result.metadata.get("generate_wireframe_video")
        or result.wireframe_video_bytes is not None
    ):
        return
    if not result.metadata.get("opencv_available", True):
        st.warning(
            "Wireframe video could not be generated because OpenCV is not available in this runtime. "
            f"Import error: {result.metadata.get('opencv_import_error') or 'unknown error'}"
        )
        return
    if not result.metadata.get("face_detector_available", False):
        st.warning(
            "Wireframe video could not be generated because the Haar cascade face detector is unavailable, even after trying to download the XML."
        )
        return
    st.warning(
        "Wireframe video was requested, but no face overlays were produced. The detector ran and found zero faces in the processed frames."
    )


def render_result_selected_chunk(
    result: ProcessingResult,
    selected_index: int | None,
    selected_chunk: TranscriptChunk | None,
) -> None:
    if selected_chunk is None or selected_index is None:
        return
    st.subheader("Current segment")
    render_selected_chunk(result, selected_index, selected_chunk)


def render_result_tabs(result: ProcessingResult) -> None:
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


def pick_selected_time(result: ProcessingResult) -> int:
    maximum = max(int(math.ceil(result.duration)), 0)
    selected_time = min(
        st.session_state.get(SESSION_SELECTED_TIME_SECONDS_KEY, 0), maximum
    )
    st.session_state[SESSION_SELECTED_TIME_SECONDS_KEY] = selected_time
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
        preview, preview_image = get_selected_chunk_preview(result, index)
        if preview_image is not None:
            st.image(
                preview_image,
                width=STRETCH_WIDTH,
                caption=build_selected_chunk_caption(result, chunk, preview),
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


def get_selected_chunk_preview(
    result: ProcessingResult, index: int
) -> tuple[FaceDetectionFrame | None, bytes | None]:
    preview = result.face_detections.get(index)
    has_face_preview = preview is not None and preview.annotated_image is not None
    if has_face_preview and preview is not None:
        return preview, preview.annotated_image
    return preview, result.face_thumbnails.get(index) or result.thumbnails.get(index)


def build_selected_chunk_caption(
    result: ProcessingResult, chunk: TranscriptChunk, preview: FaceDetectionFrame | None
) -> str:
    detection_enabled = face_detection_enabled(result)
    if preview is None:
        return format_face_count(chunk.face_count, detection_enabled=detection_enabled)
    preview_label = (
        "Landmark-aware face overlay"
        if preview.landmarks
        else "Speaker-colored face overlay"
        if preview.annotated_image is not None
        else "Frame preview"
    )
    return f"{format_face_overlay_summary(chunk, preview, detection_enabled=detection_enabled)} · {preview_label}"


def render_timeline_controls(result: ProcessingResult, selected_time: int) -> None:
    timeline = build_timeline_chart(result, float(selected_time))
    st.altair_chart(timeline, width=STRETCH_WIDTH)
    maximum = max(int(math.ceil(result.duration)), 0)
    selected_time = st.slider(
        "Playback position",
        min_value=0,
        max_value=maximum,
        value=selected_time,
        step=1,
        format="%d s",
        key=SESSION_SELECTED_TIME_SECONDS_KEY,
    )
    active_turn = pick_active_speaker_turn(result, float(selected_time))
    if active_turn is None:
        st.caption(f"No active speaker at {format_seconds(selected_time)}.")
        return
    st.caption(
        f"{active_turn.label} speaking at {format_seconds(selected_time)} · {format_seconds(active_turn.start)} → {format_seconds(active_turn.end)}"
    )


def build_timeline_chart(result: ProcessingResult, selected_time: float) -> alt.Chart:
    turns_frame = build_timeline_turns_frame(result)
    duration_limit = max(result.duration, 1.0)
    speaker_colors = build_speaker_color_map(result.speakers)
    return (
        (
            build_timeline_bars(turns_frame, speaker_colors, duration_limit)
            + build_timeline_indicator(selected_time, duration_limit)
        )
        .properties(height=72)
        .configure_view(strokeWidth=0)
    )


def build_timeline_turns_frame(result: ProcessingResult) -> pd.DataFrame:
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
    if not turns_frame.empty:
        return turns_frame
    return pd.DataFrame(
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


def build_timeline_bars(
    turns_frame: pd.DataFrame, speaker_colors: dict[str, str], duration_limit: float
) -> alt.Chart:
    color_encoding = (
        alt.Color(
            SPEAKER_N_FIELD,
            scale=alt.Scale(
                domain=list(speaker_colors.keys()),
                range=list(speaker_colors.values()),
            ),
            legend=None,
        )
        if speaker_colors
        else alt.Color(SPEAKER_N_FIELD, legend=None)
    )
    return (
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
            color=color_encoding,
            tooltip=[
                alt.Tooltip(SPEAKER_N_FIELD, title="Speaker"),
                alt.Tooltip("start:Q", title="Start", format=".2f"),
                alt.Tooltip("end:Q", title="End", format=".2f"),
                alt.Tooltip("duration:Q", title="Duration", format=".2f"),
            ],
        )
    )


def build_timeline_indicator(selected_time: float, duration_limit: float) -> alt.Chart:
    indicator_frame = pd.DataFrame(
        [{"selected_time": min(max(selected_time, 0.0), duration_limit)}]
    )
    return (
        alt.Chart(indicator_frame)
        .mark_rule(color="#111827", strokeWidth=3)
        .encode(x=alt.X("selected_time:Q", scale=alt.Scale(domain=[0, duration_limit])))
    )


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
            footer = f"{chunk.word_count} words · {format_face_count(chunk.face_count, detection_enabled=face_detection_enabled(result))}"
            if chunk.confidence is not None:
                footer = f"{footer} · confidence {chunk.confidence:.3f}"
            st.caption(footer)
            thumbnail = result.face_thumbnails.get(index) or result.thumbnails.get(
                index
            )
            if thumbnail:
                st.image(thumbnail, width=STRETCH_WIDTH)


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
        render_frame_preview_row(result, items[row_start : row_start + 3])


def render_frame_preview_row(
    result: ProcessingResult, row: list[tuple[int, bytes]]
) -> None:
    columns = st.columns(len(row))
    for column, (index, image_bytes) in zip(columns, row, strict=False):
        chunk = result.chunks[index]
        detection = result.face_detections.get(index)
        with column:
            st.image(
                image_bytes,
                width=STRETCH_WIDTH,
                caption=f"{chunk.speaker} · {format_seconds(chunk.start)} · speaker color {describe_color_hex(detection.color_hex if detection is not None else None)} · {format_face_count(chunk.face_count, detection_enabled=face_detection_enabled(result))}",
            )


def render_faces_tab(result: ProcessingResult) -> None:
    if not result.face_detections:
        if face_detection_enabled(result):
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
        render_face_detection_row(result, items[row_start : row_start + 3])


def render_face_detection_row(
    result: ProcessingResult, row: list[tuple[int, FaceDetectionFrame]]
) -> None:
    columns = st.columns(len(row))
    for column, (index, detection) in zip(columns, row, strict=False):
        chunk = result.chunks[index]
        with column:
            preview = detection.annotated_image or result.face_thumbnails.get(index)
            if preview:
                st.image(
                    preview,
                    width=STRETCH_WIDTH,
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
    _ = build_speaker_color_legend(result.speakers)
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
