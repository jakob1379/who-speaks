from __future__ import annotations

from typing import Mapping, Protocol, TYPE_CHECKING

import streamlit as st

from st_who_speaks.runtime import (
    ExecutionSettings,
    detect_acceleration_status,
)

if TYPE_CHECKING:
    from st_who_speaks.pipeline import ProcessMediaOptions

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

SESSION_HARDWARE_ACCELERATION_KEY = "hardware_acceleration"


class UploadedMedia(Protocol):
    name: str

    def getvalue(self) -> bytes: ...


def format_whisper_model_option(model_size: str) -> str:
    return f"{model_size} — {WHISPER_MODEL_METRICS[model_size]}"


def render_sidebar() -> tuple[UploadedMedia | None, bool]:
    with st.sidebar:
        uploaded_file = render_sidebar_input_section()
        render_sidebar_model_section()
        render_sidebar_local_diarization_section()
        render_sidebar_frame_section()
        run_requested = st.button("Run pipeline", type="primary")

    return uploaded_file, run_requested


def render_sidebar_input_section() -> UploadedMedia | None:
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


def build_process_media_options(
    uploaded_name: str,
    execution_settings: ExecutionSettings,
    session_state: Mapping[str, object],
) -> ProcessMediaOptions:
    from st_who_speaks.pipeline import ProcessMediaOptions

    return ProcessMediaOptions(
        execution_settings=execution_settings,
        media_label=uploaded_name,
        whisper_model_size=str(session_state["whisper_model_size"]),
        min_speakers=int(session_state["min_speakers"]),
        max_speakers=int(session_state["max_speakers"]),
        generate_thumbnails=bool(session_state["generate_thumbnails"]),
        enable_face_detection=bool(session_state["enable_face_detection"]),
        max_thumbnails=int(session_state["max_thumbnails"]),
    )
