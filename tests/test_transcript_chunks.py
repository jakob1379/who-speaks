from __future__ import annotations

from st_who_speaks.models import SpeakerTurn, WordToken
from st_who_speaks.transcript import build_transcript_chunks


def test_build_transcript_chunks_assigns_speakers_and_splits_on_gap() -> None:
    words = [
        WordToken("Hello", 0.0, 0.5, 0.9, "Unknown"),
        WordToken("there", 0.5, 1.0, 0.8, "Unknown"),
        WordToken("General", 3.0, 3.4, 0.7, "Unknown"),
        WordToken("Kenobi", 3.4, 4.0, 0.6, "Unknown"),
    ]
    turns = [
        SpeakerTurn("spk_0", "Person A", 0.0, 1.2),
        SpeakerTurn("spk_1", "Person B", 2.8, 4.2),
    ]

    chunks, speakers = build_transcript_chunks(words, turns)

    assert speakers == ["Person A", "Person B"]
    assert len(chunks) == 2
    assert chunks[0].speaker == "Person A"
    assert chunks[0].text == "Hello there"
    assert chunks[1].speaker == "Person B"
    assert chunks[1].text == "General Kenobi"
