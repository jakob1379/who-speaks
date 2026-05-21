from __future__ import annotations

from dataclasses import asdict
import math
import os
import tempfile
from pathlib import Path
from typing import Callable, TypeVar

import streamlit as st

from st_who_speaks.colors import describe_color_hex
from st_who_speaks.models import (
    FaceDetectionFrame,
    ProcessingResult,
    SpeakerTurn,
    TranscriptChunk,
)
from st_who_speaks.runtime import resolve_execution_settings
from st_who_speaks.ui.media import (
    STRETCH_WIDTH,
    VIDEO_MP4_MIME_TYPE,
    is_video_media,
    render_media_player,
)
from st_who_speaks.ui.sidebar import (
    SESSION_HARDWARE_ACCELERATION_KEY,
    UploadedMedia,
    build_process_media_options,
    render_sidebar,
)
from st_who_speaks.ui.formatting import (
    face_detection_enabled,
    format_face_count,
    format_face_overlay_summary,
    format_seconds,
    include_chunk,
)
from st_who_speaks.ui.timeline import build_timeline_chart

SESSION_RESULT_KEY = "result"
SESSION_VIDEO_BYTES_KEY = "video_bytes"
SESSION_SELECTED_CHUNK_INDEX_KEY = "selected_chunk_index"
SESSION_SELECTED_TIME_SECONDS_KEY = "selected_time_seconds"
SESSION_SHOW_WIREFRAME_VIDEO_KEY = "show_wireframe_video"
SESSION_LAST_UPLOADED_NAME_KEY = "last_uploaded_name"

PreviewItem = TypeVar("PreviewItem")


__all__ = ["main"]


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


def reset_uploaded_file_state(uploaded_file: UploadedMedia) -> None:
    st.session_state[SESSION_RESULT_KEY] = None
    st.session_state[SESSION_VIDEO_BYTES_KEY] = uploaded_file.getvalue()
    st.session_state[SESSION_SELECTED_CHUNK_INDEX_KEY] = 0
    st.session_state[SESSION_SELECTED_TIME_SECONDS_KEY] = 0
    st.session_state[SESSION_SHOW_WIREFRAME_VIDEO_KEY] = False
    st.session_state[SESSION_LAST_UPLOADED_NAME_KEY] = uploaded_file.name


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


def run_pipeline(uploaded_file: UploadedMedia) -> None:
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
        descriptor, temporary_name = tempfile.mkstemp(dir=upload_dir)
        with os.fdopen(descriptor, "wb") as target:
            target.write(video_bytes)
        Path(temporary_name).replace(media_path)
        options = build_process_media_options(
            uploaded_file.name,
            execution,
            st.session_state,
        )

        try:
            from st_who_speaks.pipeline import process_media

            result = process_media(
                str(media_path),
                options,
                update_progress,
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

def render_result(result: ProcessingResult, video_bytes: bytes | None) -> None:
    render_metrics(result)
    st.caption(
        "OpenCV face detection + facial landmarks run on sampled frames only and do not identify people."
    )

    selected_time = pick_selected_time(result)
    selected_index, selected_chunk = selected_chunk_for_time(result, selected_time)
    if selected_index is not None:
        st.session_state[SESSION_SELECTED_CHUNK_INDEX_KEY] = selected_index
    render_result_media(result, video_bytes, selected_time)
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
    if result.wireframe_video_bytes is not None and is_video_media(result.media_identity):
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
    if is_video_media(result.media_identity):
        st.caption(
            "Showing processed wireframe video"
            if show_wireframe_video and result.wireframe_video_bytes is not None
            else "Showing original video"
        )
    if media_payload is not None:
        render_media_player(
            media_payload,
            result.media_identity,
            result.chunks,
            start_time=selected_time,
            video_format=VIDEO_MP4_MIME_TYPE if show_wireframe_video else None,
        )
    render_timeline_controls(result, selected_time)


def render_wireframe_video_warning(result: ProcessingResult) -> None:
    if (
        not result.metadata.generate_wireframe_video
        or result.wireframe_video_bytes is not None
    ):
        return
    if not result.metadata.opencv_available:
        st.warning(
            "Wireframe video could not be generated because OpenCV is not available in this runtime. "
            f"Import error: {result.metadata.opencv_import_error or 'unknown error'}"
        )
        return
    if not result.metadata.face_detector_available:
        st.warning(
            "Wireframe video could not be generated because the Haar cascade face detector is unavailable, even after trying to download the XML."
        )
        return
    if warning := result.metadata.diagnostics.get("wireframe_video"):
        st.warning(warning)
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
        "Face detections", str(result.metadata.total_faces_detected)
    )
    column_five.metric(
        "Processing time", f"{result.metadata.processing_time_seconds}s"
    )
    if result.metadata.diarization_warning:
        st.warning(result.metadata.diarization_warning)


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
        face_detection, preview_image_bytes = resolve_selected_chunk_preview_assets(
            result, index
        )
        if preview_image_bytes is not None:
            st.image(
                preview_image_bytes,
                width=STRETCH_WIDTH,
                caption=build_selected_chunk_caption(result, chunk, face_detection),
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


def resolve_selected_chunk_preview_assets(
    result: ProcessingResult, index: int
) -> tuple[FaceDetectionFrame | None, bytes | None]:
    face_detection = result.face_detections.get(index)
    has_face_preview = (
        face_detection is not None and face_detection.annotated_image is not None
    )
    if has_face_preview and face_detection is not None:
        return face_detection, face_detection.annotated_image
    return face_detection, result.face_thumbnails.get(index) or result.thumbnails.get(
        index
    )


def build_selected_chunk_caption(
    result: ProcessingResult,
    chunk: TranscriptChunk,
    face_detection: FaceDetectionFrame | None,
) -> str:
    detection_enabled = face_detection_enabled(result)
    if face_detection is None:
        return format_face_count(chunk.face_count, detection_enabled=detection_enabled)
    detection_label = (
        "Landmark-aware face overlay"
        if face_detection.landmarks
        else "Face overlay"
        if face_detection.annotated_image is not None
        else "Frame preview"
    )
    return f"{format_face_overlay_summary(chunk, face_detection, detection_enabled=detection_enabled)} · {detection_label}"


def render_timeline_controls(result: ProcessingResult, selected_time: int) -> None:
    timeline = build_timeline_chart(result, float(selected_time))
    st.altair_chart(timeline, width=STRETCH_WIDTH)
    maximum = max(int(math.ceil(result.duration)), 0)
    selected_time = st.slider(
        "Playback position",
        min_value=0,
        max_value=maximum,
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
    def render_item(item: tuple[int, bytes]) -> None:
        index, image_bytes = item
        chunk = result.chunks[index]
        detection = result.face_detections.get(index)
        st.image(
            image_bytes,
            width=STRETCH_WIDTH,
            caption=f"transcript speaker {chunk.speaker} · {format_seconds(chunk.start)} · overlay color {describe_color_hex(detection.color_hex if detection is not None else None)} · {format_face_count(chunk.face_count, detection_enabled=face_detection_enabled(result))}",
        )

    render_preview_grid(row, render_item)


def render_preview_grid(
    row: list[PreviewItem], render_item: Callable[[PreviewItem], None]
) -> None:
    columns = st.columns(len(row))
    for column, item in zip(columns, row, strict=False):
        with column:
            render_item(item)


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
        f"Showing {len(items)} sampled frames with {result.metadata.total_faces_detected} total faces detected. Face overlays are visual-only and stay independent from transcript speaker labels."
    )
    for row_start in range(0, len(items), 3):
        render_face_detection_row(result, items[row_start : row_start + 3])


def render_face_detection_row(
    result: ProcessingResult, row: list[tuple[int, FaceDetectionFrame]]
) -> None:
    def render_item(item: tuple[int, FaceDetectionFrame]) -> None:
        index, detection = item
        chunk = result.chunks[index]
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
                format_face_overlay_summary(chunk, detection, detection_enabled=True)
            )
        if detection.boxes:
            st.caption(
                ", ".join(
                    f"{box.x},{box.y} {box.width}x{box.height}"
                    for box in detection.boxes
                )
            )

    render_preview_grid(row, render_item)


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
                    "color_hex": detection.color_hex,
                    "boxes": [asdict(box) for box in detection.boxes],
                    "landmarks": [
                        [asdict(point) for point in face_landmarks]
                        for face_landmarks in detection.landmarks
                    ],
                }
                for index, detection in result.face_detections.items()
            ],
            "metadata": asdict(result.metadata),
        }
    )
