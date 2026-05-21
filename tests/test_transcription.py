from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from st_who_speaks import transcription
from st_who_speaks.models import TranscriptionSegment, WordToken


def test_transcribe_audio_data_converts_segments_and_words(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class FakeModel:
        def transcribe(self, audio_path: str, **kwargs):
            captured["audio_path"] = audio_path
            captured["kwargs"] = kwargs
            return (
                [
                    SimpleNamespace(
                        text=" Hello world ",
                        start=1,
                        end=3.5,
                        avg_logprob=-0.25,
                        words=[
                            SimpleNamespace(
                                word=" Hello ",
                                start=1,
                                end=1.4,
                                probability=0.9,
                            ),
                            SimpleNamespace(
                                word="",
                                start=1.4,
                                end=1.5,
                                probability=0.1,
                            ),
                            SimpleNamespace(
                                word="world",
                                start=1.6,
                                end=2.0,
                                probability=None,
                            ),
                        ],
                    )
                ],
                object(),
            )

    transcription.transcribe_audio_data.cache_clear()
    monkeypatch.setattr(
        transcription,
        "load_whisper_model",
        lambda model_size, device, compute_type: captured.update(
            {
                "model_size": model_size,
                "device": device,
                "compute_type": compute_type,
            }
        )
        or FakeModel(),
    )

    words, segments = transcription.transcribe_audio_data(
        "audio.wav",
        model_size="small",
        device="cuda",
        compute_type="float16",
    )

    assert captured == {
        "model_size": "small",
        "device": "cuda",
        "compute_type": "float16",
        "audio_path": "audio.wav",
        "kwargs": {
            "beam_size": 5,
            "best_of": 5,
            "word_timestamps": True,
            "vad_filter": True,
            "condition_on_previous_text": False,
        },
    }
    assert words == (
        WordToken("Hello", 1.0, 1.4, 0.9, "Unknown"),
        WordToken("world", 1.6, 2.0, None, "Unknown"),
    )
    assert segments == (
        TranscriptionSegment("Hello world", 1.0, 3.5, -0.25),
    )


def test_transcribe_audio_data_skips_incomplete_words_and_blank_segments(
    monkeypatch,
) -> None:
    class FakeModel:
        def transcribe(self, *_args, **_kwargs):
            return (
                [
                    SimpleNamespace(
                        text="   ",
                        start=0,
                        end=1,
                        avg_logprob=None,
                        words=[
                            SimpleNamespace(
                                word="missing-start",
                                start=None,
                                end=0.4,
                                probability=0.5,
                            ),
                            SimpleNamespace(
                                word="missing-end",
                                start=0.5,
                                end=None,
                                probability=0.5,
                            ),
                        ],
                    )
                ],
                object(),
            )

    transcription.transcribe_audio_data.cache_clear()
    monkeypatch.setattr(
        transcription,
        "load_whisper_model",
        lambda *_args: FakeModel(),
    )

    words, segments = transcription.transcribe_audio_data(
        "empty.wav",
        model_size="tiny",
        device="cpu",
        compute_type="int8",
    )

    assert words == ()
    assert segments == ()
