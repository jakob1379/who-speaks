from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SpeakerTurn:
    raw_speaker: str
    label: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(self.end - self.start, 0.0)


@dataclass(slots=True)
class WordToken:
    text: str
    start: float
    end: float
    probability: float | None
    speaker: str


@dataclass(slots=True)
class TranscriptionSegment:
    text: str
    start: float
    end: float
    confidence: float | None


@dataclass(slots=True)
class TranscriptChunk:
    speaker: str
    start: float
    end: float
    text: str
    confidence: float | None
    word_count: int
    thumbnail_timestamp: float | None = None
    face_count: int | None = None

    @property
    def duration(self) -> float:
        return max(self.end - self.start, 0.0)


@dataclass(slots=True)
class FaceBox:
    x: int
    y: int
    width: int
    height: int


@dataclass(slots=True)
class LandmarkPoint:
    x: int
    y: int


@dataclass(slots=True)
class FaceDetectionFrame:
    face_count: int
    boxes: list[FaceBox] = field(default_factory=list)
    landmarks: list[list[LandmarkPoint]] = field(default_factory=list)
    annotated_image: bytes | None = None
    color_hex: str | None = None


@dataclass(slots=True)
class SpeakerColor:
    label: str
    color_hex: str


@dataclass(slots=True)
class ProcessingResult:
    media_path: str
    audio_path: str | None
    duration: float
    speakers: list[str]
    chunks: list[TranscriptChunk]
    speaker_turns: list[SpeakerTurn]
    thumbnails: dict[int, bytes] = field(default_factory=dict)
    face_thumbnails: dict[int, bytes] = field(default_factory=dict)
    face_detections: dict[int, FaceDetectionFrame] = field(default_factory=dict)
    wireframe_video_bytes: bytes | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
