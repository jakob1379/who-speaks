from __future__ import annotations

from pathlib import Path

from st_who_speaks.media_io import (
    ThumbnailExtractionResult,
    extract_thumbnails,
    infer_duration,
    media_has_audio_stream,
    summarize_ffmpeg_error,
)
from st_who_speaks.models import TranscriptChunk


def test_media_has_audio_stream_uses_ffprobe(monkeypatch, tmp_path: Path) -> None:
    media_path = tmp_path / "clip.mp4"
    media_path.write_bytes(b"video")

    class CompletedProcess:
        def __init__(self, returncode: int, stdout: str):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = ""

    monkeypatch.setattr(
        "st_who_speaks.media_io.shutil.which",
        lambda name: "/usr/bin/ffprobe" if name == "ffprobe" else None,
    )
    monkeypatch.setattr(
        "st_who_speaks.media_io.subprocess.run",
        lambda *args, **kwargs: CompletedProcess(0, "audio\n"),
    )

    assert media_has_audio_stream(str(media_path)) is True


def test_summarize_ffmpeg_error_for_missing_audio_stream() -> None:
    stderr = "Output file does not contain any stream\nError opening output file /tmp/audio.wav"
    assert "does not contain an audio stream" in summarize_ffmpeg_error(stderr)


def test_extract_thumbnails_releases_unopened_capture(monkeypatch) -> None:
    releases: list[bool] = []

    class UnopenedCapture:
        def isOpened(self) -> bool:
            return False

        def release(self) -> None:
            releases.append(True)

    monkeypatch.setattr(
        "st_who_speaks.media_io.cv2.VideoCapture",
        lambda _media_path: UnopenedCapture(),
    )

    result = extract_thumbnails(
        "missing.webm",
        [
            TranscriptChunk(
                speaker="Person A",
                start=1.0,
                end=3.5,
                text="fallback",
                confidence=None,
                word_count=1,
            )
        ],
        max_thumbnails=1,
        generate_thumbnails=True,
        include_frames=False,
    )

    assert result == ThumbnailExtractionResult({}, {})
    assert releases == [True]


def test_infer_duration_releases_unopened_capture(monkeypatch) -> None:
    releases: list[bool] = []

    class UnopenedCapture:
        def isOpened(self) -> bool:
            return False

        def release(self) -> None:
            releases.append(True)

    monkeypatch.setattr(
        "st_who_speaks.media_io.cv2.VideoCapture",
        lambda _media_path: UnopenedCapture(),
    )

    duration = infer_duration(
        "missing.webm",
        [
            TranscriptChunk(
                speaker="Person A",
                start=1.0,
                end=3.5,
                text="fallback",
                confidence=None,
                word_count=1,
            )
        ],
    )

    assert duration == 3.5
    assert releases == [True]
