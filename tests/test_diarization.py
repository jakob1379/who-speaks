from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import torch

from st_who_speaks import diarization
from st_who_speaks.diarization import (
    AudioWaveform,
    cluster_labels_to_turns,
    cluster_speakers,
    collect_diarization_embeddings,
    diarize_audio,
    load_waveform,
    merge_adjacent_turns,
    speaker_cluster_candidates,
    slice_waveform,
)
from st_who_speaks.models import SpeakerTurn, TranscriptionSegment


def test_load_and_slice_waveform_uses_typed_audio_boundary(tmp_path: Path) -> None:
    audio_path = tmp_path / "clip.wav"
    samples = np.array([0, 8192, 16384, 32767], dtype=np.int16)
    with wave.open(str(audio_path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(4)
        target.writeframes(samples.tobytes())

    waveform = load_waveform(str(audio_path))
    window = slice_waveform(waveform, start_seconds=0.25, end_seconds=0.75)

    assert isinstance(waveform, AudioWaveform)
    assert waveform.sample_rate == 4
    assert waveform.waveform.shape == (1, 4)
    assert window.shape == (1, 2)
    assert np.allclose(window.numpy(), [[0.25, 0.5]], atol=0.001)


def test_diarize_audio_returns_empty_without_segments(monkeypatch) -> None:
    monkeypatch.setattr(
        "st_who_speaks.diarization.load_waveform",
        lambda _audio_path: (_ for _ in ()).throw(AssertionError("unused")),
    )

    assert (
        diarize_audio(
            "audio.wav",
            transcript_segments=[],
            device="cpu",
            min_speakers=1,
            max_speakers=2,
        )
        == []
    )


def test_collect_diarization_embeddings_skips_empty_waveform_windows(
    monkeypatch,
) -> None:
    extracted_shapes: list[tuple[int, ...]] = []

    def fake_extract_speaker_embedding(_model, waveform) -> np.ndarray:
        extracted_shapes.append(tuple(waveform.shape))
        return np.array([1.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(
        "st_who_speaks.diarization.extract_speaker_embedding",
        fake_extract_speaker_embedding,
    )

    embeddings, segments = collect_diarization_embeddings(
        AudioWaveform(waveform=torch.ones((1, 10)), sample_rate=10),
        embedding_model=object(),
        transcript_segments=[
            TranscriptionSegment("inside", 0.0, 0.5, 0.9),
            TranscriptionSegment("outside", 2.0, 2.5, 0.8),
        ],
    )

    assert len(embeddings) == 1
    assert [segment.text for segment in segments] == ["inside"]
    assert extracted_shapes == [(1, 5)]


def test_speaker_cluster_candidates_clamp_to_sample_count() -> None:
    assert speaker_cluster_candidates(min_speakers=3, max_speakers=8, n_samples=2) == [
        2
    ]
    assert speaker_cluster_candidates(min_speakers=None, max_speakers=None, n_samples=3) == [
        1,
        2,
        3,
    ]


def test_cluster_speakers_uses_best_scoring_candidate(monkeypatch) -> None:
    seen_cluster_counts: list[int] = []

    def fake_labels(_normalized, cluster_count: int) -> np.ndarray:
        seen_cluster_counts.append(cluster_count)
        if cluster_count == 2:
            return np.array([0, 0, 1])
        return np.array([0, 1, 2])

    def fake_score(_normalized, _labels, cluster_count: int, _n_samples: int) -> float:
        return 10.0 if cluster_count == 2 else 1.0

    monkeypatch.setattr(diarization, "speaker_cluster_labels", fake_labels)
    monkeypatch.setattr(diarization, "speaker_cluster_score", fake_score)

    labels = cluster_speakers(
        [
            np.array([1.0, 0.0]),
            np.array([0.9, 0.1]),
            np.array([0.0, 1.0]),
        ],
        min_speakers=2,
        max_speakers=3,
    )

    assert labels == [0, 0, 1]
    assert seen_cluster_counts == [2, 3]


def test_cluster_speakers_falls_back_to_single_label_when_scores_are_unusable(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        diarization,
        "speaker_cluster_labels",
        lambda _normalized, _cluster_count: np.array([0, 1, 2]),
    )
    monkeypatch.setattr(
        diarization,
        "speaker_cluster_score",
        lambda *_args, **_kwargs: float("-inf"),
    )

    assert cluster_speakers(
        [
            np.array([1.0, 0.0]),
            np.array([0.5, 0.5]),
            np.array([0.0, 1.0]),
        ],
        min_speakers=1,
        max_speakers=2,
    ) == [0, 0, 0]


def test_cluster_speakers_splits_distinct_two_sample_embeddings() -> None:
    assert cluster_speakers(
        [np.array([1.0, 0.0]), np.array([0.0, 1.0])],
        min_speakers=1,
        max_speakers=2,
    ) == [0, 1]


def test_cluster_speakers_collapses_near_identical_two_sample_embeddings() -> None:
    assert cluster_speakers(
        [np.array([1.0, 0.0]), np.array([1.0, 1e-10])],
        min_speakers=1,
        max_speakers=2,
    ) == [0, 0]


def test_cluster_speakers_skips_invalid_candidate_score(monkeypatch) -> None:
    def fake_labels(_normalized, cluster_count: int) -> np.ndarray:
        if cluster_count == 2:
            return np.array([0, 0, 1])
        return np.array([0, 1, 2])

    def fake_score(_normalized, _labels, cluster_count: int, _n_samples: int) -> float:
        if cluster_count == 2:
            raise ValueError("candidate cannot be scored")
        return 5.0

    monkeypatch.setattr(diarization, "speaker_cluster_labels", fake_labels)
    monkeypatch.setattr(diarization, "speaker_cluster_score", fake_score)

    assert cluster_speakers(
        [
            np.array([1.0, 0.0]),
            np.array([0.9, 0.1]),
            np.array([0.0, 1.0]),
        ],
        min_speakers=2,
        max_speakers=3,
    ) == [0, 1, 2]


def test_cluster_speakers_preserves_unexpected_scoring_errors(monkeypatch) -> None:
    monkeypatch.setattr(
        diarization,
        "speaker_cluster_labels",
        lambda _normalized, _cluster_count: np.array([0, 0, 1]),
    )
    monkeypatch.setattr(
        diarization,
        "speaker_cluster_score",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("scoring backend crashed")
        ),
    )

    try:
        cluster_speakers(
            [
                np.array([1.0, 0.0]),
                np.array([0.9, 0.1]),
                np.array([0.0, 1.0]),
            ],
            min_speakers=2,
            max_speakers=2,
        )
    except RuntimeError as error:
        assert str(error) == "scoring backend crashed"
    else:
        raise AssertionError("unexpected scoring errors should propagate")


def test_cluster_labels_to_turns_and_merge_adjacent_turns_are_deterministic() -> None:
    turns = cluster_labels_to_turns(
        [
            TranscriptionSegment("first", 0.0, 1.0, 0.9),
            TranscriptionSegment("second", 1.02, 2.0, 0.9),
            TranscriptionSegment("third", 2.5, 3.0, 0.9),
        ],
        [3, 3, 7],
    )
    merged = merge_adjacent_turns(turns)

    assert [(turn.raw_speaker, turn.label, turn.start, turn.end) for turn in turns] == [
        ("speaker_3", "Person A", 0.0, 1.0),
        ("speaker_3", "Person A", 1.02, 2.0),
        ("speaker_7", "Person B", 2.5, 3.0),
    ]
    assert [
        (turn.raw_speaker, turn.label, turn.start, turn.end) for turn in merged
    ] == [
        ("speaker_3", "Person A", 0.0, 2.0),
        ("speaker_7", "Person B", 2.5, 3.0),
    ]


def test_cluster_labels_to_turns_rejects_segment_label_mismatches() -> None:
    segments = [
        TranscriptionSegment("first", 0.0, 1.0, 0.9),
        TranscriptionSegment("second", 1.0, 2.0, 0.9),
    ]

    for received_segments, received_labels in (
        (segments, [0]),
        (segments[:1], [0, 1]),
    ):
        try:
            cluster_labels_to_turns(received_segments, received_labels)
        except ValueError:
            continue
        raise AssertionError("segment/label mismatches should fail")


def test_diarize_audio_returns_single_speaker_turn_when_clustering_collapses(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "st_who_speaks.diarization.load_waveform",
        lambda _audio_path: AudioWaveform(
            waveform=torch.ones((1, 20)),
            sample_rate=10,
        ),
    )
    monkeypatch.setattr(
        "st_who_speaks.diarization.load_speaker_embedding_model",
        lambda _device: object(),
    )
    monkeypatch.setattr(
        "st_who_speaks.diarization.extract_speaker_embedding",
        lambda _model, waveform: np.array([float(waveform.shape[-1]), 0.0]),
    )
    monkeypatch.setattr(
        "st_who_speaks.diarization.cluster_speakers",
        lambda *_args, **_kwargs: [0, 0],
    )

    turns = diarize_audio(
        "audio.wav",
        transcript_segments=[
            TranscriptionSegment("first", 0.0, 0.5, 0.9),
            TranscriptionSegment("second", 0.5, 1.0, 0.8),
        ],
        device="cpu",
        min_speakers=1,
        max_speakers=2,
    )

    assert turns == [SpeakerTurn("speaker_0", "Person A", 0.0, 0.5)]
