from __future__ import annotations

import math
import re
import shutil
import subprocess
import tempfile
import time
import wave
import importlib
import inspect
import warnings
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from faster_whisper import WhisperModel
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from st_who_speaks.models import (
    FaceBox,
    FaceDetectionFrame,
    LandmarkPoint,
    ProcessingResult,
    TranscriptionSegment,
    SpeakerTurn,
    TranscriptChunk,
    WordToken,
)
from st_who_speaks.logging import get_logger
from st_who_speaks.runtime import resolve_execution_settings

logger = get_logger(__name__)

HAARCASCADE_FILENAME = "haarcascade_frontalface_default.xml"
HAARCASCADE_DOWNLOAD_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/"
    f"{HAARCASCADE_FILENAME}"
)
OPENCV_IMPORT_ERROR: Exception | None = None


def _ensure_torchaudio_compatibility() -> None:
    try:
        torchaudio = importlib.import_module("torchaudio")
    except Exception:
        return

    if not hasattr(torchaudio, "get_audio_backend"):
        setattr(torchaudio, "get_audio_backend", lambda: None)
    if not hasattr(torchaudio, "set_audio_backend"):
        setattr(torchaudio, "set_audio_backend", lambda *_args, **_kwargs: None)
    if hasattr(torchaudio, "list_audio_backends"):
        return

    def list_audio_backends() -> list[str]:
        backends: list[str] = []
        get_audio_backend = getattr(torchaudio, "get_audio_backend", None)
        if callable(get_audio_backend):
            try:
                backend = get_audio_backend()
            except Exception:
                backend = None
            if backend:
                backends.append(str(backend))
        if not backends:
            backends.extend(["soundfile", "sox_io"])
        return backends

    setattr(torchaudio, "list_audio_backends", list_audio_backends)


def _ensure_huggingface_hub_compatibility() -> None:
    try:
        huggingface_hub = importlib.import_module("huggingface_hub")
    except Exception:
        return

    hf_hub_download = getattr(huggingface_hub, "hf_hub_download", None)
    if not callable(hf_hub_download):
        return

    try:
        signature = inspect.signature(hf_hub_download)
    except (TypeError, ValueError):
        return

    if "use_auth_token" in signature.parameters:
        return

    def compat_hf_hub_download(*args, **kwargs):
        filename = kwargs.get("filename")
        if filename is None and len(args) >= 2:
            filename = args[1]
        if "use_auth_token" in kwargs and "token" not in kwargs:
            kwargs["token"] = kwargs.pop("use_auth_token")
        else:
            kwargs.pop("use_auth_token", None)
        try:
            return hf_hub_download(*args, **kwargs)
        except Exception as error:
            if filename == "custom.py" and error.__class__.__name__ in {
                "RemoteEntryNotFoundError",
                "EntryNotFoundError",
                "HTTPStatusError",
            }:
                empty_custom_module = (
                    Path(tempfile.gettempdir())
                    / "st-who-speaks"
                    / "speechbrain-compat"
                    / "custom.py"
                )
                empty_custom_module.parent.mkdir(parents=True, exist_ok=True)
                if not empty_custom_module.exists():
                    empty_custom_module.write_text("", encoding="utf-8")
                return str(empty_custom_module)
            raise

    setattr(huggingface_hub, "hf_hub_download", compat_hf_hub_download)

    file_download = getattr(huggingface_hub, "file_download", None)
    if file_download is not None and hasattr(file_download, "hf_hub_download"):
        setattr(file_download, "hf_hub_download", compat_hf_hub_download)


try:
    import cv2
except Exception as error:
    OPENCV_IMPORT_ERROR = error

    class _FallbackVideoCapture:
        def __init__(self, *_args, **_kwargs):
            self._opened = False

        def isOpened(self) -> bool:
            return False

        def get(self, *_args, **_kwargs):
            return 0.0

        def release(self) -> None:
            return None

        def read(self):
            return False, None

        def set(self, *_args, **_kwargs):
            return False

    class _FallbackCascadeClassifier:
        def __init__(self, *_args, **_kwargs):
            self._empty = True

        def empty(self) -> bool:
            return self._empty

        def detectMultiScale(self, *_args, **_kwargs):
            return []

    class _FallbackCv2:
        CAP_PROP_FRAME_WIDTH = 3
        CAP_PROP_FRAME_HEIGHT = 4
        CAP_PROP_POS_MSEC = 0
        CAP_PROP_FPS = 5
        CAP_PROP_FRAME_COUNT = 7
        COLOR_BGR2GRAY = 0
        COLOR_BGR2RGB = 1
        FONT_HERSHEY_SIMPLEX = 0
        LINE_AA = 16
        VideoCapture = _FallbackVideoCapture
        CascadeClassifier = _FallbackCascadeClassifier

        @staticmethod
        def cvtColor(frame, *_args, **_kwargs):
            return frame

        @staticmethod
        def rectangle(*_args, **_kwargs):
            return None

        @staticmethod
        def circle(*_args, **_kwargs):
            return None

        @staticmethod
        def line(*_args, **_kwargs):
            return None

        @staticmethod
        def putText(*_args, **_kwargs):
            return None

        @staticmethod
        def imencode(*_args, **_kwargs):
            return True, np.frombuffer(b"fallback", dtype=np.uint8)

    cv2: Any = _FallbackCv2()

ProgressCallback = Callable[[str, float], None]

SPEAKER_COLOR_PALETTE = [
    "#ef4444",
    "#22c55e",
    "#3b82f6",
    "#f59e0b",
    "#a855f7",
    "#14b8a6",
    "#ec4899",
    "#84cc16",
]

APPROXIMATE_LANDMARK_CONNECTIONS = [
    (0, 6),
    (6, 2),
    (2, 7),
    (7, 1),
    (0, 8),
    (8, 3),
    (3, 9),
    (9, 1),
    (2, 4),
    (4, 5),
    (5, 3),
]


def build_speaker_color_map(speakers: list[str]) -> dict[str, str]:
    return {
        speaker: SPEAKER_COLOR_PALETTE[index % len(SPEAKER_COLOR_PALETTE)]
        for index, speaker in enumerate(speakers)
    }


def hex_to_bgr(color_hex: str) -> tuple[int, int, int]:
    normalized = color_hex.lstrip("#")
    red = int(normalized[0:2], 16)
    green = int(normalized[2:4], 16)
    blue = int(normalized[4:6], 16)
    return blue, green, red


@lru_cache(maxsize=1)
def load_face_landmarker() -> Any | None:
    try:
        face_mesh_module: Any = importlib.import_module("mediapipe.solutions.face_mesh")
    except Exception:
        return None
    return face_mesh_module.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )


@lru_cache(maxsize=1)
def load_face_mesh_connections() -> list[tuple[int, int]]:
    try:
        face_mesh_module: Any = importlib.import_module("mediapipe.solutions.face_mesh")
    except Exception:
        return APPROXIMATE_LANDMARK_CONNECTIONS
    connections = set(getattr(face_mesh_module, "FACEMESH_TESSELATION", set()))
    connections.update(getattr(face_mesh_module, "FACEMESH_CONTOURS", set()))
    connections.update(getattr(face_mesh_module, "FACEMESH_IRISES", set()))
    if not connections:
        return APPROXIMATE_LANDMARK_CONNECTIONS
    return [(int(start), int(end)) for start, end in sorted(connections)]


def approximate_landmarks_from_box(box: FaceBox) -> list[LandmarkPoint]:
    left = box.x
    top = box.y
    center_x = box.x + box.width // 2
    return [
        LandmarkPoint(x=int(left + box.width * 0.30), y=int(top + box.height * 0.35)),
        LandmarkPoint(x=int(left + box.width * 0.70), y=int(top + box.height * 0.35)),
        LandmarkPoint(x=int(center_x), y=int(top + box.height * 0.52)),
        LandmarkPoint(x=int(left + box.width * 0.36), y=int(top + box.height * 0.72)),
        LandmarkPoint(x=int(left + box.width * 0.64), y=int(top + box.height * 0.72)),
        LandmarkPoint(x=int(center_x), y=int(top + box.height * 0.90)),
        LandmarkPoint(x=int(left + box.width * 0.18), y=int(top + box.height * 0.46)),
        LandmarkPoint(x=int(left + box.width * 0.82), y=int(top + box.height * 0.46)),
        LandmarkPoint(x=int(left + box.width * 0.25), y=int(top + box.height * 0.58)),
        LandmarkPoint(x=int(left + box.width * 0.75), y=int(top + box.height * 0.58)),
    ]


def landmarks_from_crop(
    face_mesh: Any | None, crop: np.ndarray, *, offset_x: int, offset_y: int
) -> list[LandmarkPoint]:
    if face_mesh is None or crop.size == 0:
        return []

    rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_crop)
    if not getattr(results, "multi_face_landmarks", None):
        return []

    crop_height, crop_width = crop.shape[:2]
    points: list[LandmarkPoint] = []
    for landmark in results.multi_face_landmarks[0].landmark:
        x = int(round(landmark.x * crop_width)) + offset_x
        y = int(round(landmark.y * crop_height)) + offset_y
        points.append(LandmarkPoint(x=x, y=y))
    return points


def process_media(
    media_path: str,
    *,
    use_hardware_acceleration: bool,
    media_label: str | None,
    whisper_model_size: str,
    transcription_device: str,
    transcription_compute_type: str,
    embedding_device: str,
    hardware_acceleration_enabled: bool,
    min_speakers: int | None,
    max_speakers: int | None,
    generate_thumbnails: bool,
    enable_face_detection: bool,
    max_thumbnails: int,
    generate_wireframe_video: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> ProcessingResult:
    started_at = time.perf_counter()
    execution = resolve_execution_settings(use_hardware_acceleration)

    logger.info(
        "processing media",
        media_path=media_path,
        media_label=media_label,
        whisper_model_size=whisper_model_size,
        use_hardware_acceleration=use_hardware_acceleration,
        transcription_device=transcription_device,
        embedding_device=embedding_device,
    )

    with tempfile.TemporaryDirectory(prefix="st-who-speaks-") as working_dir:
        working_path = Path(working_dir)
        audio_path = working_path / "audio.wav"

        _notify(progress_callback, "Extracting audio", 0.1)
        extract_audio(media_path, str(audio_path))

        _notify(progress_callback, "Transcribing audio", 0.35)
        transcript_words, transcript_segments = transcribe_audio_data(
            str(audio_path),
            model_size=whisper_model_size,
            device=transcription_device,
            compute_type=transcription_compute_type,
        )

        diarization_warning = None
        try:
            _notify(progress_callback, "Running local speaker clustering", 0.6)
            speaker_turns = diarize_audio(
                str(audio_path),
                transcript_segments=list(transcript_segments),
                device=embedding_device,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
        except Exception as error:
            logger.exception("speaker clustering failed")
            diarization_warning = f"Local speaker clustering failed. Falling back to a single-speaker transcript. {error}"
            speaker_turns = []

        _notify(progress_callback, "Building transcript chunks", 0.8)
        chunks, speakers = build_transcript_chunks(
            list(transcript_words), speaker_turns
        )
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

        speaker_colors = build_speaker_color_map(speakers)
        face_detector_available = False
        if enable_face_detection or generate_wireframe_video:
            face_detector_available = load_face_detector() is not None

        _notify(progress_callback, "Extracting frame thumbnails", 0.92)
        thumbnails = {}
        face_thumbnails = {}
        face_counts_by_chunk: dict[int, int] = {}
        face_detections: dict[int, FaceDetectionFrame] = {}
        wireframe_video_bytes: bytes | None = None
        wireframe_faces_detected = 0
        if generate_thumbnails or enable_face_detection:
            frame_media_path = prepare_video_for_frame_processing(
                media_path, working_path
            )
            (
                thumbnails,
                face_thumbnails,
                face_counts_by_chunk,
                face_detections,
            ) = extract_thumbnails(
                frame_media_path,
                chunks,
                max_thumbnails=max_thumbnails,
                generate_thumbnails=generate_thumbnails,
                enable_face_detection=enable_face_detection,
                speaker_colors=speaker_colors,
            )
            if enable_face_detection:
                face_detections = {
                    index: replace(
                        face_detection,
                        speaker_label=chunks[index].speaker,
                        color_hex=speaker_colors.get(chunks[index].speaker),
                    )
                    for index, face_detection in face_detections.items()
                }
            if enable_face_detection:
                chunks = [
                    replace(chunk, face_count=face_counts_by_chunk.get(index))
                    for index, chunk in enumerate(chunks)
                ]
            if enable_face_detection and generate_wireframe_video:
                _notify(progress_callback, "Rendering wireframe video", 0.97)
                wireframe_video_bytes, wireframe_faces_detected = build_wireframe_video(
                    frame_media_path,
                    original_media_path=media_path,
                    speaker_turns=speaker_turns,
                    speaker_colors=speaker_colors,
                    working_dir=working_path,
                )

        total_faces_detected = sum(face_counts_by_chunk.values())

        duration = infer_duration(media_path, chunks)
        processing_time = time.perf_counter() - started_at

        _notify(progress_callback, "Done", 1.0)
        return ProcessingResult(
            media_path=media_label or Path(media_path).name,
            audio_path=None,
            duration=duration,
            speakers=speakers,
            chunks=chunks,
            speaker_turns=speaker_turns,
            thumbnails=thumbnails,
            face_thumbnails=face_thumbnails,
            face_detections=face_detections,
            wireframe_video_bytes=wireframe_video_bytes,
            metadata={
                "processing_time_seconds": round(processing_time, 2),
                "diarization_warning": diarization_warning,
                "whisper_model": whisper_model_size,
                "hardware_acceleration_enabled": hardware_acceleration_enabled,
                "hardware_acceleration_summary": execution["summary"],
                "backend_label": execution["backend_label"],
                "transcription_device": transcription_device,
                "transcription_compute_type": transcription_compute_type,
                "embedding_device": embedding_device,
                "chunk_count": len(chunks),
                "speaker_count": len(speakers),
                "speaker_colors": speaker_colors,
                "total_faces_detected": total_faces_detected,
                "face_counts_by_chunk": face_counts_by_chunk,
                "face_detection_enabled": enable_face_detection,
                "opencv_available": is_opencv_available(),
                "opencv_import_error": get_opencv_import_error_message(),
                "face_detector_available": face_detector_available,
                "generate_thumbnails": generate_thumbnails,
                "generate_wireframe_video": generate_wireframe_video,
                "wireframe_faces_detected": wireframe_faces_detected,
                "wireframe_video_available": wireframe_video_bytes is not None,
                "diarization_mode": "local-clustering",
            },
        )


def extract_audio(media_path: str, audio_path: str) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is not available. Install it through flake.nix or your OS package manager."
        )
    if not media_has_audio_stream(media_path):
        raise RuntimeError(
            "The uploaded media file does not contain an audio stream. "
            "Transcription and diarization require speech audio."
        )
    command = [
        "ffmpeg",
        "-y",
        "-i",
        media_path,
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        audio_path,
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        logger.error(
            "audio extraction failed",
            media_path=media_path,
            stderr=completed.stderr.strip(),
        )
        raise RuntimeError(
            summarize_ffmpeg_error(completed.stderr)
            or "ffmpeg failed to extract audio."
        )


@lru_cache(maxsize=6)
def load_whisper_model(model_size: str, device: str, compute_type: str) -> WhisperModel:
    return WhisperModel(model_size, device=device, compute_type=compute_type)


@lru_cache(maxsize=8)
def transcribe_audio_data(
    audio_path: str,
    *,
    model_size: str,
    device: str,
    compute_type: str,
) -> tuple[tuple[WordToken, ...], tuple[TranscriptionSegment, ...]]:
    model = load_whisper_model(model_size, device, compute_type)
    segments, _ = model.transcribe(
        audio_path,
        beam_size=5,
        best_of=5,
        word_timestamps=True,
        vad_filter=True,
        condition_on_previous_text=False,
    )

    words: list[WordToken] = []
    transcript_segments: list[TranscriptionSegment] = []
    for segment in segments:
        segment_words = list(segment.words or [])
        if segment_words:
            for word in segment_words:
                if word.start is None or word.end is None:
                    continue
                cleaned = word.word.strip()
                if not cleaned:
                    continue
                words.append(
                    WordToken(
                        text=cleaned,
                        start=float(word.start),
                        end=float(word.end),
                        probability=float(word.probability)
                        if word.probability is not None
                        else None,
                        speaker="Unknown",
                    )
                )

        text = (segment.text or "").strip()
        if not text:
            continue
        transcript_segments.append(
            TranscriptionSegment(
                text=text,
                start=float(segment.start),
                end=float(segment.end),
                confidence=float(segment.avg_logprob)
                if segment.avg_logprob is not None
                else None,
            )
        )

    return tuple(words), tuple(transcript_segments)


def transcribe_audio(
    audio_path: str,
    *,
    model_size: str,
    device: str,
    compute_type: str,
) -> list[WordToken]:
    words, _ = transcribe_audio_data(
        audio_path,
        model_size=model_size,
        device=device,
        compute_type=compute_type,
    )
    return list(words)


def diarize_audio(
    audio_path: str,
    *,
    transcript_segments: list[TranscriptionSegment],
    device: str,
    min_speakers: int | None,
    max_speakers: int | None,
) -> list[SpeakerTurn]:
    if not transcript_segments:
        return []

    waveform = load_waveform(audio_path)
    embedding_model = load_speaker_embedding_model(device)
    embeddings: list[np.ndarray] = []
    usable_segments: list[TranscriptionSegment] = []
    for segment in transcript_segments:
        window = slice_waveform(waveform, segment.start, segment.end)
        if window.numel() == 0:
            continue
        embeddings.append(extract_speaker_embedding(embedding_model, window))
        usable_segments.append(segment)

    if not embeddings:
        return []

    cluster_labels = cluster_speakers(
        embeddings,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    if len(set(cluster_labels)) <= 1:
        segment = usable_segments[0]
        return [
            SpeakerTurn(
                raw_speaker="speaker_0",
                label="Person A",
                start=segment.start,
                end=segment.end,
            )
        ]

    cluster_order: list[int] = []
    for cluster_label in cluster_labels:
        if cluster_label not in cluster_order:
            cluster_order.append(cluster_label)
    cluster_to_label = {
        cluster_label: f"Person {chr(65 + index)}"
        for index, cluster_label in enumerate(cluster_order)
    }

    turns = [
        SpeakerTurn(
            raw_speaker=f"speaker_{cluster_label}",
            label=cluster_to_label[cluster_label],
            start=segment.start,
            end=segment.end,
        )
        for segment, cluster_label in zip(usable_segments, cluster_labels, strict=False)
    ]
    return merge_adjacent_turns(turns)


@lru_cache(maxsize=2)
def load_speaker_embedding_model(device: str) -> Any:
    _ensure_torchaudio_compatibility()
    _ensure_huggingface_hub_compatibility()

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=re.escape(
                "`torch.cuda.amp.custom_fwd(args...)` is deprecated. Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead."
            ),
            category=FutureWarning,
        )
        from speechbrain.inference.classifiers import EncoderClassifier

    savedir = Path(tempfile.gettempdir()) / "st-who-speaks" / f"speechbrain-{device}"
    savedir.mkdir(parents=True, exist_ok=True)
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(savedir),
        run_opts={"device": device},
    )


def slice_waveform(
    waveform: dict[str, Any], start_seconds: float, end_seconds: float
) -> torch.Tensor:
    sample_rate = int(waveform["sample_rate"])
    tensor = waveform["waveform"]
    start_index = max(0, int(start_seconds * sample_rate))
    end_index = max(start_index + 1, int(end_seconds * sample_rate))
    end_index = min(end_index, tensor.shape[-1])
    return tensor[:, start_index:end_index]


def extract_speaker_embedding(model: Any, waveform: torch.Tensor) -> np.ndarray:
    signal = waveform.to(dtype=torch.float32)
    minimum_samples = 16000
    if signal.shape[-1] < minimum_samples:
        signal = torch.nn.functional.pad(
            signal, (0, minimum_samples - signal.shape[-1])
        )
    with torch.no_grad():
        embedding = model.encode_batch(signal)
    return np.asarray(embedding.squeeze().detach().cpu(), dtype=np.float32)


def cluster_speakers(
    embeddings: list[np.ndarray],
    *,
    min_speakers: int | None,
    max_speakers: int | None,
) -> list[int]:
    if not embeddings:
        return []
    if len(embeddings) == 1:
        return [0]

    normalized = normalize(np.vstack(embeddings))
    n_samples = len(embeddings)
    lower_bound = max(1, int(min_speakers or 1))
    upper_bound = int(max_speakers or n_samples)
    upper_bound = min(upper_bound, n_samples)
    if lower_bound > upper_bound:
        lower_bound = upper_bound

    candidate_clusters = list(range(lower_bound, upper_bound + 1))
    if not candidate_clusters:
        candidate_clusters = [1]

    best_labels: list[int] | None = None
    best_score = float("-inf")
    for cluster_count in candidate_clusters:
        labels = KMeans(
            n_clusters=cluster_count, n_init="auto", random_state=0
        ).fit_predict(normalized)
        if len(set(labels)) <= 1 or cluster_count >= n_samples:
            score = float("-inf")
        else:
            try:
                score = silhouette_score(normalized, labels, metric="cosine")
            except Exception:
                score = float("-inf")
        if score > best_score:
            best_score = score
            best_labels = [int(label) for label in labels]

    return best_labels or [0] * n_samples


def merge_adjacent_turns(turns: list[SpeakerTurn]) -> list[SpeakerTurn]:
    if not turns:
        return []

    ordered = sorted(turns, key=lambda turn: turn.start)
    merged: list[SpeakerTurn] = [ordered[0]]
    for turn in ordered[1:]:
        previous = merged[-1]
        if (
            previous.raw_speaker == turn.raw_speaker
            and turn.start <= previous.end + 0.05
        ):
            merged[-1] = SpeakerTurn(
                raw_speaker=previous.raw_speaker,
                label=previous.label,
                start=previous.start,
                end=max(previous.end, turn.end),
            )
            continue
        merged.append(turn)
    return merged


def pick_speaker_turn_at_time(
    speaker_turns: list[SpeakerTurn], timestamp: float
) -> SpeakerTurn | None:
    if not speaker_turns:
        return None
    for turn in speaker_turns:
        if turn.start <= timestamp < turn.end:
            return turn
    if timestamp >= speaker_turns[-1].end:
        return speaker_turns[-1]
    return min(
        speaker_turns,
        key=lambda turn: min(abs(timestamp - turn.start), abs(timestamp - turn.end)),
    )


def build_transcript_chunks(
    words: list[WordToken],
    speaker_turns: list[SpeakerTurn],
    *,
    max_gap_seconds: float = 1.2,
) -> tuple[list[TranscriptChunk], list[str]]:
    if not words:
        return [], []

    annotated_words = [
        WordToken(
            text=word.text,
            start=word.start,
            end=word.end,
            probability=word.probability,
            speaker=resolve_speaker(word.start, word.end, speaker_turns),
        )
        for word in words
    ]

    first_seen: dict[str, float] = {}
    for word in annotated_words:
        first_seen.setdefault(word.speaker, word.start)
    ordered_speakers = [
        speaker for speaker, _ in sorted(first_seen.items(), key=lambda item: item[1])
    ]

    chunks: list[TranscriptChunk] = []
    current_group: list[WordToken] = []
    for word in annotated_words:
        if not current_group:
            current_group = [word]
            continue

        previous = current_group[-1]
        same_speaker = previous.speaker == word.speaker
        close_in_time = (word.start - previous.end) <= max_gap_seconds
        if same_speaker and close_in_time:
            current_group.append(word)
            continue

        chunks.append(group_to_chunk(current_group))
        current_group = [word]

    if current_group:
        chunks.append(group_to_chunk(current_group))

    return chunks, ordered_speakers


def resolve_speaker(start: float, end: float, speaker_turns: list[SpeakerTurn]) -> str:
    if not speaker_turns:
        return "Person A"

    overlap_scores: Counter[str] = Counter()
    midpoint = (start + end) / 2
    for turn in speaker_turns:
        overlap = max(0.0, min(end, turn.end) - max(start, turn.start))
        if overlap > 0:
            overlap_scores[turn.label] += overlap
        elif turn.start <= midpoint <= turn.end:
            overlap_scores[turn.label] += 0.001

    if overlap_scores:
        return overlap_scores.most_common(1)[0][0]

    nearest_turn = min(
        speaker_turns,
        key=lambda turn: min(abs(start - turn.end), abs(end - turn.start)),
    )
    return nearest_turn.label


def group_to_chunk(group: list[WordToken]) -> TranscriptChunk:
    text = normalize_text([word.text for word in group])
    probabilities = [word.probability for word in group if word.probability is not None]
    confidence = (
        round(sum(probabilities) / len(probabilities), 3) if probabilities else None
    )
    return TranscriptChunk(
        speaker=group[0].speaker,
        start=group[0].start,
        end=group[-1].end,
        text=text,
        confidence=confidence,
        word_count=len(group),
        thumbnail_timestamp=(group[0].start + group[-1].end) / 2,
    )


def normalize_text(parts: list[str]) -> str:
    text = " ".join(part.strip() for part in parts if part.strip())
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    return text.strip()


def load_waveform(audio_path: str) -> dict[str, Any]:
    with wave.open(audio_path, "rb") as source:
        sample_rate = source.getframerate()
        channel_count = source.getnchannels()
        frames = source.readframes(source.getnframes())

    waveform = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channel_count > 1:
        waveform = waveform.reshape(-1, channel_count).T
    else:
        waveform = waveform[None, :]

    return {
        "waveform": torch.from_numpy(waveform.copy()),
        "sample_rate": sample_rate,
    }


def prepare_video_for_frame_processing(
    media_path: str, working_dir: Path, *, max_width: int = 1920, max_height: int = 1080
) -> str:
    capture = cv2.VideoCapture(media_path)
    if not capture.isOpened():
        return media_path

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    if width <= 0 or height <= 0 or (width <= max_width and height <= max_height):
        return media_path

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is not available. Install it through flake.nix or your OS package manager."
        )

    target = working_dir / "frame-source.mp4"
    command = [
        "ffmpeg",
        "-y",
        "-i",
        media_path,
        "-vf",
        f"scale={max_width}:{max_height}:force_original_aspect_ratio=decrease",
        "-an",
        str(target),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            completed.stderr.strip() or "ffmpeg failed to downsample the video."
        )
    return str(target)


def media_has_audio_stream(media_path: str) -> bool:
    if shutil.which("ffprobe") is None:
        return True
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "csv=p=0",
        media_path,
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    return completed.returncode == 0 and bool(completed.stdout.strip())


def summarize_ffmpeg_error(stderr: str) -> str:
    normalized = stderr.strip()
    if "Output file does not contain any stream" in normalized:
        return (
            "The uploaded media file does not contain an audio stream. "
            "Transcription and diarization require speech audio."
        )
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    if not lines:
        return ""
    return lines[-1]


def build_wireframe_video(
    media_path: str,
    *,
    original_media_path: str,
    speaker_turns: list[SpeakerTurn],
    speaker_colors: dict[str, str],
    working_dir: Path,
) -> tuple[bytes | None, int]:
    if shutil.which("ffmpeg") is None:
        return None, 0

    if load_face_detector() is None:
        return None, 0

    capture = cv2.VideoCapture(media_path)
    if not capture.isOpened():
        return None, 0

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if width <= 0 or height <= 0:
        capture.release()
        return None, 0
    if not math.isfinite(fps) or fps <= 0:
        fps = 24.0

    target = working_dir / "wireframe-overlay.mp4"
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "-",
        "-i",
        original_media_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-c:a",
        "aac",
        "-shortest",
        str(target),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_index = 0
    total_faces_detected = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            timestamp = frame_index / fps
            speaker_turn = pick_speaker_turn_at_time(speaker_turns, timestamp)
            speaker_label = speaker_turn.label if speaker_turn is not None else None
            color_hex = speaker_colors.get(speaker_label) if speaker_label else None
            face_detection, annotated = detect_and_annotate_faces_with_frame(
                frame,
                speaker_label=speaker_label,
                color_hex=color_hex,
            )
            total_faces_detected += face_detection.face_count
            if process.stdin is None:
                break
            try:
                process.stdin.write(annotated.tobytes())
            except BrokenPipeError:
                break
            frame_index += 1
    finally:
        capture.release()
        if process.stdin is not None:
            process.stdin.close()
    stderr = (
        process.stderr.read().decode("utf-8", errors="ignore") if process.stderr else ""
    )
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            stderr.strip() or "ffmpeg failed to build the wireframe video."
        )
    if not target.exists() or total_faces_detected <= 0:
        return None, total_faces_detected
    return target.read_bytes(), total_faces_detected


def extract_thumbnails(
    media_path: str,
    chunks: list[TranscriptChunk],
    *,
    max_thumbnails: int,
    generate_thumbnails: bool,
    enable_face_detection: bool,
    speaker_colors: dict[str, str] | None = None,
) -> tuple[
    dict[int, bytes],
    dict[int, bytes],
    dict[int, int],
    dict[int, FaceDetectionFrame],
]:
    if not chunks or max_thumbnails <= 0:
        return {}, {}, {}, {}

    capture = cv2.VideoCapture(media_path)
    if not capture.isOpened():
        return {}, {}, {}, {}

    selected_indices = sampled_indices(len(chunks), max_thumbnails)
    thumbnails: dict[int, bytes] = {}
    face_thumbnails: dict[int, bytes] = {}
    face_counts_by_chunk: dict[int, int] = {}
    face_detections: dict[int, FaceDetectionFrame] = {}
    try:
        for index in selected_indices:
            timestamp = chunks[index].thumbnail_timestamp or chunks[index].start
            capture.set(cv2.CAP_PROP_POS_MSEC, max(timestamp, 0.0) * 1000)
            success, frame = capture.read()
            if not success:
                continue
            if generate_thumbnails:
                encoded, buffer = cv2.imencode(".jpg", frame)
            else:
                encoded, buffer = False, None
            if encoded and buffer is not None:
                thumbnails[index] = buffer.tobytes()
            if enable_face_detection:
                speaker_label = chunks[index].speaker
                color_hex = (speaker_colors or {}).get(speaker_label)
                face_detection = detect_and_annotate_faces(
                    frame,
                    speaker_label=speaker_label,
                    color_hex=color_hex,
                )
                face_counts_by_chunk[index] = face_detection.face_count
                face_detections[index] = face_detection
                if face_detection.annotated_image is not None:
                    face_thumbnails[index] = face_detection.annotated_image
    finally:
        capture.release()

    return thumbnails, face_thumbnails, face_counts_by_chunk, face_detections


def is_opencv_available() -> bool:
    return OPENCV_IMPORT_ERROR is None


def get_opencv_import_error_message() -> str | None:
    if OPENCV_IMPORT_ERROR is None:
        return None
    return str(OPENCV_IMPORT_ERROR)


def resolve_haarcascade_path() -> Path | None:
    if not is_opencv_available():
        return None
    candidates: list[Path] = []
    cv2_data = getattr(cv2, "data", None)
    haarcascades_path = getattr(cv2_data, "haarcascades", None)
    if haarcascades_path:
        candidates.append(Path(str(haarcascades_path)) / HAARCASCADE_FILENAME)
    cv2_file = getattr(cv2, "__file__", None)
    if cv2_file is not None:
        candidates.append(
            Path(cv2_file).resolve().parent / "data" / HAARCASCADE_FILENAME
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    download_target = (
        Path(tempfile.gettempdir())
        / "st-who-speaks"
        / "haarcascades"
        / HAARCASCADE_FILENAME
    )
    download_target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(HAARCASCADE_DOWNLOAD_URL, timeout=30) as response:
            download_target.write_bytes(response.read())
    except (OSError, urllib.error.URLError):
        return None
    return download_target if download_target.exists() else None


@lru_cache(maxsize=1)
def load_face_detector() -> cv2.CascadeClassifier | None:
    cascade_path = resolve_haarcascade_path()
    if cascade_path is None:
        return None
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        return None
    return detector


def resolve_landmark_connections(points: list[LandmarkPoint]) -> list[tuple[int, int]]:
    connections = load_face_mesh_connections()
    if len(points) < 50:
        connections = APPROXIMATE_LANDMARK_CONNECTIONS
    return [
        (start, end)
        for start, end in connections
        if start < len(points) and end < len(points)
    ]


def draw_landmark_wireframe(
    annotated: np.ndarray,
    points: list[LandmarkPoint],
    overlay_color: tuple[int, int, int],
) -> None:
    for start, end in resolve_landmark_connections(points):
        start_point = points[start]
        end_point = points[end]
        cv2.line(
            annotated,
            (start_point.x, start_point.y),
            (end_point.x, end_point.y),
            overlay_color,
            1,
        )


def detect_and_annotate_faces_with_frame(
    frame: np.ndarray,
    *,
    speaker_label: str | None = None,
    color_hex: str | None = None,
) -> tuple[FaceDetectionFrame, np.ndarray]:
    detector = load_face_detector()
    resolved_color_hex = color_hex or "#ef4444"
    annotated = frame.copy()
    if detector is None:
        return (
            FaceDetectionFrame(
                face_count=0,
                boxes=[],
                landmarks=[],
                annotated_image=None,
                speaker_label=speaker_label,
                color_hex=resolved_color_hex,
            ),
            annotated,
        )
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        grayscale,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )
    boxes: list[FaceBox] = []
    landmarks: list[list[LandmarkPoint]] = []
    face_mesh = load_face_landmarker()
    overlay_color = hex_to_bgr(resolved_color_hex)
    for x, y, width, height in faces:
        box = FaceBox(x=int(x), y=int(y), width=int(width), height=int(height))
        boxes.append(box)
        crop = frame[box.y : box.y + box.height, box.x : box.x + box.width]
        points = landmarks_from_crop(
            face_mesh,
            crop,
            offset_x=box.x,
            offset_y=box.y,
        )
        if not points:
            points = approximate_landmarks_from_box(box)
        landmarks.append(points)

        cv2.rectangle(
            annotated,
            (box.x, box.y),
            (box.x + box.width, box.y + box.height),
            overlay_color,
            2,
        )
        draw_landmark_wireframe(annotated, points, overlay_color)
        for point in points:
            cv2.circle(annotated, (point.x, point.y), 1, overlay_color, -1)
    return (
        FaceDetectionFrame(
            face_count=len(boxes),
            boxes=boxes,
            landmarks=landmarks,
            annotated_image=None,
            speaker_label=speaker_label,
            color_hex=resolved_color_hex,
        ),
        annotated,
    )


def detect_and_annotate_faces(
    frame: np.ndarray,
    *,
    speaker_label: str | None = None,
    color_hex: str | None = None,
) -> FaceDetectionFrame:
    face_detection, annotated = detect_and_annotate_faces_with_frame(
        frame,
        speaker_label=speaker_label,
        color_hex=color_hex,
    )
    encoded, buffer = cv2.imencode(".jpg", annotated)
    return FaceDetectionFrame(
        face_count=face_detection.face_count,
        boxes=face_detection.boxes,
        landmarks=face_detection.landmarks,
        annotated_image=buffer.tobytes() if encoded else None,
        speaker_label=face_detection.speaker_label,
        color_hex=face_detection.color_hex,
    )


def sampled_indices(total_items: int, limit: int) -> list[int]:
    if total_items <= limit:
        return list(range(total_items))
    step = total_items / limit
    return sorted(
        {min(total_items - 1, math.floor(index * step)) for index in range(limit)}
    )


def infer_duration(media_path: str, chunks: list[TranscriptChunk]) -> float:
    capture = cv2.VideoCapture(media_path)
    if capture.isOpened():
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        capture.release()
        if fps > 0 and frame_count > 0:
            return round(frame_count / fps, 2)
    if chunks:
        return round(chunks[-1].end, 2)
    return 0.0


def _notify(
    progress_callback: ProgressCallback | None, label: str, progress: float
) -> None:
    if progress_callback is not None:
        progress_callback(label, progress)
