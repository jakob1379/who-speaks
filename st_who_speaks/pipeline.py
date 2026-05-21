from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from st_who_speaks.colors import build_speaker_color_map
from st_who_speaks.dependency_compat import (
    PIPELINE_TEMP_DIR_NAME,
    get_opencv_import_error_message,
    is_opencv_available,
)
from st_who_speaks.diarization import diarize_audio
from st_who_speaks.frame_assets import (
    FrameAssetCollectionContext,
    collect_frame_assets,
)
from st_who_speaks.logging import get_logger
from st_who_speaks.media_io import (
    extract_audio,
    infer_duration,
)
from st_who_speaks.models import (
    ProcessingMetadata,
    ProcessingResult,
    SpeakerTurn,
    TranscriptionSegment,
    TranscriptChunk,
    WordToken,
)
from st_who_speaks.runtime import ExecutionSettings
from st_who_speaks.transcript import (
    build_transcript_chunks,
)
from st_who_speaks.transcription import transcribe_audio_data

logger = get_logger(__name__)

PIPELINE_TEMP_DIR_PREFIX = f"{PIPELINE_TEMP_DIR_NAME}-"

ProgressCallback = Callable[[str, float], None]


@dataclass(frozen=True, slots=True)
class ProcessMediaOptions:
    execution_settings: ExecutionSettings
    media_label: str | None
    whisper_model_size: str
    min_speakers: int | None
    max_speakers: int | None
    generate_thumbnails: bool
    enable_face_detection: bool
    max_thumbnails: int


def _build_processing_metadata(
    *,
    processing_time: float,
    diarization_warning: str | None,
    options: ProcessMediaOptions,
    chunks: list[TranscriptChunk],
    speakers: list[str],
    speaker_colors: dict[str, str],
    face_counts_by_chunk: dict[int, int],
    total_faces_detected: int,
    face_detector_available: bool,
    wireframe_video_bytes: bytes | None,
    wireframe_faces_detected: int,
    diagnostics: dict[str, str],
) -> ProcessingMetadata:
    execution_settings = options.execution_settings
    return ProcessingMetadata(
        processing_time_seconds=round(processing_time, 2),
        diarization_warning=diarization_warning,
        whisper_model=options.whisper_model_size,
        hardware_acceleration_enabled=execution_settings.hardware_acceleration_enabled,
        hardware_acceleration_summary=execution_settings.summary,
        backend_label=execution_settings.backend_label,
        transcription_device=execution_settings.transcription_device,
        transcription_compute_type=execution_settings.transcription_compute_type,
        embedding_device=execution_settings.embedding_device,
        chunk_count=len(chunks),
        speaker_count=len(speakers),
        speaker_colors=speaker_colors,
        total_faces_detected=total_faces_detected,
        face_counts_by_chunk=face_counts_by_chunk,
        face_detection_enabled=options.enable_face_detection,
        opencv_available=is_opencv_available(),
        opencv_import_error=get_opencv_import_error_message(),
        face_detector_available=face_detector_available,
        generate_thumbnails=options.generate_thumbnails,
        generate_wireframe_video=options.enable_face_detection,
        wireframe_faces_detected=wireframe_faces_detected,
        wireframe_video_available=wireframe_video_bytes is not None,
        diarization_mode="local-clustering",
        diagnostics=diagnostics,
    )


def process_media(
    media_path: str,
    options: ProcessMediaOptions,
    progress_callback: ProgressCallback | None = None,
) -> ProcessingResult:
    started_at = time.perf_counter()
    execution_settings = options.execution_settings

    logger.info(
        "processing media",
        media_path=media_path,
        media_label=options.media_label,
        whisper_model_size=options.whisper_model_size,
        hardware_acceleration_enabled=execution_settings.hardware_acceleration_enabled,
        transcription_device=execution_settings.transcription_device,
        embedding_device=execution_settings.embedding_device,
    )

    with tempfile.TemporaryDirectory(prefix=PIPELINE_TEMP_DIR_PREFIX) as working_dir:
        working_path = Path(working_dir)
        audio_path = working_path / "audio.wav"

        _notify(progress_callback, "Extracting audio", 0.1)
        extract_audio(media_path, str(audio_path))

        _notify(progress_callback, "Transcribing audio", 0.35)
        transcript_words, transcript_segments = transcribe_audio_data(
            str(audio_path),
            model_size=options.whisper_model_size,
            device=execution_settings.transcription_device,
            compute_type=execution_settings.transcription_compute_type,
        )

        _notify(progress_callback, "Running local speaker clustering", 0.6)
        speaker_turns, diarization_warning = run_local_speaker_clustering(
            str(audio_path),
            transcript_segments,
            options,
        )

        _notify(progress_callback, "Building transcript chunks", 0.8)
        chunks, speakers, speaker_turns = (
            build_transcript_chunks_with_fallback_speaker_turns(
                list(transcript_words), speaker_turns
            )
        )

        speaker_colors = build_speaker_color_map(speakers)
        should_process_faces = options.enable_face_detection

        _notify(progress_callback, "Extracting frame thumbnails", 0.92)
        frame_assets = collect_frame_assets(
            FrameAssetCollectionContext(
                media_path=media_path,
                working_path=working_path,
                chunks=chunks,
                generate_thumbnails=options.generate_thumbnails,
                enable_face_detection=should_process_faces,
                max_thumbnails=options.max_thumbnails,
                progress_callback=progress_callback,
            )
        )
        chunks = frame_assets.chunks

        total_faces_detected = sum(frame_assets.face_counts_by_chunk.values())

        duration = infer_duration(media_path, chunks)
        processing_time = time.perf_counter() - started_at

        _notify(progress_callback, "Done", 1.0)
        return ProcessingResult(
            display_name=options.media_label or Path(media_path).name,
            media_identity=Path(media_path).name,
            audio_path=None,
            duration=duration,
            speakers=speakers,
            chunks=chunks,
            speaker_turns=speaker_turns,
            thumbnails=frame_assets.thumbnails,
            face_thumbnails=frame_assets.face_thumbnails,
            face_detections=frame_assets.face_detections,
            wireframe_video_bytes=frame_assets.wireframe_video_bytes,
            metadata=_build_processing_metadata(
                processing_time=processing_time,
                diarization_warning=diarization_warning,
                options=options,
                chunks=chunks,
                speakers=speakers,
                speaker_colors=speaker_colors,
                face_counts_by_chunk=frame_assets.face_counts_by_chunk,
                total_faces_detected=total_faces_detected,
                face_detector_available=frame_assets.face_detector_available,
                wireframe_video_bytes=frame_assets.wireframe_video_bytes,
                wireframe_faces_detected=frame_assets.wireframe_faces_detected,
                diagnostics=frame_assets.diagnostics,
            ),
        )


def run_local_speaker_clustering(
    audio_path: str,
    transcript_segments: tuple[TranscriptionSegment, ...],
    options: ProcessMediaOptions,
) -> tuple[list[SpeakerTurn], str | None]:
    try:
        return (
            diarize_audio(
                audio_path,
                transcript_segments=list(transcript_segments),
                device=options.execution_settings.embedding_device,
                min_speakers=options.min_speakers,
                max_speakers=options.max_speakers,
            ),
            None,
        )
    except Exception as error:
        logger.exception("speaker clustering failed")
        return [], (
            f"Local speaker clustering failed. Falling back to a single-speaker transcript. {error}"
        )


def build_transcript_chunks_with_fallback_speaker_turns(
    words: list[WordToken], speaker_turns: list[SpeakerTurn]
) -> tuple[list[TranscriptChunk], list[str], list[SpeakerTurn]]:
    chunks, speakers = build_transcript_chunks(words, speaker_turns)
    if not speakers:
        speakers = ["Person A"]
    if not speaker_turns:
        speaker_turns = [
            SpeakerTurn(
                raw_speaker="speaker_0",
                label=speakers[0],
                start=chunks[0].start if chunks else 0.0,
                end=chunks[-1].end if chunks else 0.0,
            )
        ]
    return chunks, speakers, speaker_turns


def _notify(
    progress_callback: ProgressCallback | None, label: str, progress: float
) -> None:
    if progress_callback is not None:
        progress_callback(label, progress)
