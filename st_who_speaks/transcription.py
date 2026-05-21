from __future__ import annotations

from functools import lru_cache
from typing import Any

from faster_whisper import WhisperModel

from st_who_speaks.models import TranscriptionSegment, WordToken


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
        words.extend(_segment_words(segment))
        transcription_segment = _segment_to_transcription_segment(segment)
        if transcription_segment is not None:
            transcript_segments.append(transcription_segment)

    return tuple(words), tuple(transcript_segments)


def _segment_words(segment: Any) -> list[WordToken]:
    words: list[WordToken] = []
    for word in segment.words or []:
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
    return words


def _segment_to_transcription_segment(
    segment: Any,
) -> TranscriptionSegment | None:
    text = (segment.text or "").strip()
    if not text:
        return None
    return TranscriptionSegment(
        text=text,
        start=float(segment.start),
        end=float(segment.end),
        confidence=float(segment.avg_logprob)
        if segment.avg_logprob is not None
        else None,
    )
