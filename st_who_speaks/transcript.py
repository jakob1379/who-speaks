from __future__ import annotations

import re
from collections import Counter

from st_who_speaks.models import SpeakerTurn, TranscriptChunk, WordToken

DEFAULT_SINGLE_SPEAKER_LABEL = "Person A"


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
        return DEFAULT_SINGLE_SPEAKER_LABEL

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
