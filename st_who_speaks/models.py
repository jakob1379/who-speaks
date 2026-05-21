from __future__ import annotations

from dataclasses import dataclass, field


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


@dataclass(frozen=True, slots=True)
class ProcessingMetadata:
    processing_time_seconds: float = 0.0
    diarization_warning: str | None = None
    whisper_model: str | None = None
    hardware_acceleration_enabled: bool = False
    hardware_acceleration_summary: str | None = None
    backend_label: str | None = None
    transcription_device: str | None = None
    transcription_compute_type: str | None = None
    embedding_device: str | None = None
    chunk_count: int = 0
    speaker_count: int = 0
    speaker_colors: dict[str, str] = field(default_factory=dict)
    total_faces_detected: int = 0
    face_counts_by_chunk: dict[int, int] = field(default_factory=dict)
    face_detection_enabled: bool = False
    opencv_available: bool = True
    opencv_import_error: str | None = None
    face_detector_available: bool = False
    generate_thumbnails: bool = False
    generate_wireframe_video: bool = False
    wireframe_faces_detected: int = 0
    wireframe_video_available: bool = False
    diarization_mode: str = "local-clustering"
    diagnostics: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ProcessingResult:
    display_name: str
    media_identity: str
    audio_path: str | None
    duration: float
    speakers: list[str]
    chunks: list[TranscriptChunk]
    speaker_turns: list[SpeakerTurn]
    thumbnails: dict[int, bytes] = field(default_factory=dict)
    face_thumbnails: dict[int, bytes] = field(default_factory=dict)
    face_detections: dict[int, FaceDetectionFrame] = field(default_factory=dict)
    wireframe_video_bytes: bytes | None = None
    metadata: ProcessingMetadata = field(default_factory=ProcessingMetadata)
