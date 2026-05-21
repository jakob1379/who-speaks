from __future__ import annotations

import re
import tempfile
import warnings
import wave
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from st_who_speaks.dependency_compat import (
    PIPELINE_TEMP_DIR_NAME,
    _ensure_huggingface_hub_compatibility,
    _ensure_torchaudio_compatibility,
)
from st_who_speaks.logging import get_logger
from st_who_speaks.models import SpeakerTurn, TranscriptionSegment
from st_who_speaks.transcript import DEFAULT_SINGLE_SPEAKER_LABEL

WARNINGS_IGNORE = "ignore"
SPEAKER_EMBEDDING_MINIMUM_SAMPLES = 16000
logger = get_logger(__name__)


@dataclass(frozen=True)
class AudioWaveform:
    waveform: torch.Tensor
    sample_rate: int


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
    embeddings, usable_segments = collect_diarization_embeddings(
        waveform,
        embedding_model,
        transcript_segments,
    )

    if not embeddings:
        return []

    cluster_labels = cluster_speakers(
        embeddings,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    if len(set(cluster_labels)) <= 1:
        return [single_speaker_turn(usable_segments[0])]

    return merge_adjacent_turns(
        cluster_labels_to_turns(usable_segments, cluster_labels)
    )


def collect_diarization_embeddings(
    waveform: AudioWaveform,
    embedding_model: Any,
    transcript_segments: list[TranscriptionSegment],
) -> tuple[list[np.ndarray], list[TranscriptionSegment]]:
    embeddings: list[np.ndarray] = []
    usable_segments: list[TranscriptionSegment] = []
    for segment in transcript_segments:
        window = slice_waveform(waveform, segment.start, segment.end)
        if window.numel() == 0:
            continue
        embeddings.append(extract_speaker_embedding(embedding_model, window))
        usable_segments.append(segment)
    return embeddings, usable_segments


def single_speaker_turn(segment: TranscriptionSegment) -> SpeakerTurn:
    return SpeakerTurn(
        raw_speaker="speaker_0",
        label=DEFAULT_SINGLE_SPEAKER_LABEL,
        start=segment.start,
        end=segment.end,
    )


def cluster_labels_to_turns(
    segments: list[TranscriptionSegment], cluster_labels: list[int]
) -> list[SpeakerTurn]:
    cluster_order: list[int] = []
    for cluster_label in cluster_labels:
        if cluster_label not in cluster_order:
            cluster_order.append(cluster_label)
    cluster_to_label = {
        cluster_label: f"Person {chr(65 + index)}"
        for index, cluster_label in enumerate(cluster_order)
    }
    return [
        SpeakerTurn(
            raw_speaker=f"speaker_{cluster_label}",
            label=cluster_to_label[cluster_label],
            start=segment.start,
            end=segment.end,
        )
        for segment, cluster_label in zip(segments, cluster_labels, strict=True)
    ]


@lru_cache(maxsize=2)
def load_speaker_embedding_model(device: str) -> Any:
    _ensure_torchaudio_compatibility()
    _ensure_huggingface_hub_compatibility()

    with warnings.catch_warnings():
        # SpeechBrain 1.0.x emits this through its pretrained classifier path with
        # current PyTorch; remove once SpeechBrain updates the torch.amp call site.
        warnings.filterwarnings(
            WARNINGS_IGNORE,
            message=re.escape(
                "`torch.cuda.amp.custom_fwd(args...)` is deprecated. Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead."
            ),
            category=FutureWarning,
        )
        from speechbrain.inference.classifiers import EncoderClassifier

    savedir = (
        Path(tempfile.gettempdir()) / PIPELINE_TEMP_DIR_NAME / f"speechbrain-{device}"
    )
    savedir.mkdir(parents=True, exist_ok=True)
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(savedir),
        run_opts={"device": device},
    )


def slice_waveform(
    waveform: AudioWaveform, start_seconds: float, end_seconds: float
) -> torch.Tensor:
    tensor = waveform.waveform
    start_index = max(0, int(start_seconds * waveform.sample_rate))
    end_index = max(start_index + 1, int(end_seconds * waveform.sample_rate))
    end_index = min(end_index, tensor.shape[-1])
    return tensor[:, start_index:end_index]


def extract_speaker_embedding(model: Any, waveform: torch.Tensor) -> np.ndarray:
    signal = waveform.to(dtype=torch.float32)
    if signal.shape[-1] < SPEAKER_EMBEDDING_MINIMUM_SAMPLES:
        signal = torch.nn.functional.pad(
            signal, (0, SPEAKER_EMBEDDING_MINIMUM_SAMPLES - signal.shape[-1])
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
    candidate_clusters = speaker_cluster_candidates(
        min_speakers, max_speakers, n_samples
    )
    if n_samples == 2:
        if 2 in candidate_clusters and not np.allclose(normalized[0], normalized[1]):
            return [0, 1]
        return [0, 0]

    best_labels: list[int] | None = None
    best_score = float("-inf")
    for cluster_count in candidate_clusters:
        labels = speaker_cluster_labels(normalized, cluster_count)
        try:
            score = speaker_cluster_score(normalized, labels, cluster_count, n_samples)
        except ValueError:
            logger.info(
                "skipping invalid speaker cluster candidate",
                cluster_count=cluster_count,
            )
            continue
        if score > best_score:
            best_score = score
            best_labels = [int(label) for label in labels]

    return best_labels or [0] * n_samples


def speaker_cluster_candidates(
    min_speakers: int | None, max_speakers: int | None, n_samples: int
) -> list[int]:
    lower_bound = max(1, int(min_speakers or 1))
    upper_bound = int(max_speakers or n_samples)
    upper_bound = min(upper_bound, n_samples)
    if lower_bound > upper_bound:
        lower_bound = upper_bound
    candidate_clusters = list(range(lower_bound, upper_bound + 1))
    return candidate_clusters or [1]


def speaker_cluster_labels(normalized: np.ndarray, cluster_count: int) -> np.ndarray:
    return KMeans(n_clusters=cluster_count, n_init="auto", random_state=0).fit_predict(
        normalized
    )


def speaker_cluster_score(
    normalized: np.ndarray, labels: np.ndarray, cluster_count: int, n_samples: int
) -> float:
    if len(set(labels)) <= 1 or cluster_count >= n_samples:
        return float("-inf")
    return silhouette_score(normalized, labels, metric="cosine")


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


def load_waveform(audio_path: str) -> AudioWaveform:
    with wave.open(audio_path, "rb") as source:
        sample_rate = source.getframerate()
        channel_count = source.getnchannels()
        frames = source.readframes(source.getnframes())

    waveform = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channel_count > 1:
        waveform = waveform.reshape(-1, channel_count).T
    else:
        waveform = waveform[None, :]

    return AudioWaveform(
        waveform=torch.from_numpy(waveform.copy()),
        sample_rate=sample_rate,
    )
