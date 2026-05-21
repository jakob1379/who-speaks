from __future__ import annotations

from st_who_speaks.colors import build_speaker_color_map, describe_color_hex
from st_who_speaks.models import (
    FaceDetectionFrame,
    ProcessingResult,
    SpeakerColor,
    TranscriptChunk,
)


def build_speaker_color_legend(speakers: list[str]) -> list[SpeakerColor]:
    color_map = build_speaker_color_map(speakers)
    return [
        SpeakerColor(label=speaker, color_hex=color_map[speaker])
        for speaker in speakers
    ]


def face_detection_enabled(result: ProcessingResult) -> bool:
    return result.metadata.face_detection_enabled


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


def format_face_overlay_summary(
    chunk: TranscriptChunk,
    detection: FaceDetectionFrame | None,
    *,
    detection_enabled: bool,
) -> str:
    parts = [
        f"transcript speaker {chunk.speaker}",
        f"{format_seconds(chunk.start)} → {format_seconds(chunk.end)}",
        format_face_count(
            detection.face_count if detection is not None else None,
            detection_enabled=detection_enabled,
        ),
    ]
    if detection is not None and detection.color_hex:
        parts.append(f"overlay color {describe_color_hex(detection.color_hex)}")
    if detection is not None and detection.landmarks:
        landmark_points = sum(len(points) for points in detection.landmarks)
        parts.append(
            f"landmarks {len(detection.landmarks)} sets / {landmark_points} points"
        )
    return " · ".join(parts)


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
