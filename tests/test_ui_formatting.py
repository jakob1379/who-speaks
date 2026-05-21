from __future__ import annotations

from st_who_speaks.colors import (
    SPEAKER_COLOR_PALETTE,
    build_speaker_color_map,
    describe_color_hex as describe_palette_color_hex,
)
from st_who_speaks.models import FaceDetectionFrame, LandmarkPoint, TranscriptChunk
from st_who_speaks.ui.formatting import (
    build_speaker_color_legend,
    format_face_count,
    format_face_overlay_summary,
    format_seconds,
    include_chunk,
)


def test_format_seconds() -> None:
    assert format_seconds(65.7) == "01:05"
    assert format_seconds(3661) == "01:01:01"


def test_format_face_count() -> None:
    assert format_face_count(None, detection_enabled=False) == "face detection off"
    assert format_face_count(None, detection_enabled=True) == "not sampled"
    assert format_face_count(3, detection_enabled=True) == "3 detections"


def test_color_helpers_cycle_palette_and_describe_known_values() -> None:
    speakers = [f"Person {index}" for index in range(len(SPEAKER_COLOR_PALETTE) + 1)]

    color_map = build_speaker_color_map(speakers)

    assert color_map["Person 0"] == SPEAKER_COLOR_PALETTE[0]
    assert color_map[f"Person {len(SPEAKER_COLOR_PALETTE)}"] == SPEAKER_COLOR_PALETTE[0]
    assert describe_palette_color_hex("#ef4444") == "red (#ef4444)"
    assert describe_palette_color_hex("#ABCDEF") == "custom (#ABCDEF)"
    assert describe_palette_color_hex(None) == "unassigned"


def test_speaker_color_legend_and_overlay_summary_are_explicit() -> None:
    legend = build_speaker_color_legend(["Person A", "Person B", "Person C"])
    detection = FaceDetectionFrame(
        face_count=2,
        boxes=[],
        landmarks=[[LandmarkPoint(1, 2), LandmarkPoint(3, 4)]],
        color_hex="#ef4444",
    )
    chunk = TranscriptChunk(
        speaker="Person A",
        start=12.0,
        end=18.0,
        text="hello",
        confidence=0.9,
        word_count=1,
        face_count=2,
    )

    assert [entry.color_hex for entry in legend] == ["#ef4444", "#22c55e", "#3b82f6"]
    assert describe_palette_color_hex("#ef4444") == "red (#ef4444)"
    summary = format_face_overlay_summary(chunk, detection, detection_enabled=True)
    assert "transcript speaker Person A" in summary
    assert "overlay color red (#ef4444)" in summary
    assert "landmarks 1 sets / 2 points" in summary


def test_include_chunk_filters_on_speaker_search_and_word_count() -> None:
    chunk = TranscriptChunk(
        speaker="Person B",
        start=0.0,
        end=1.0,
        text="This is a transcript chunk",
        confidence=0.8,
        word_count=5,
    )

    assert include_chunk(chunk, "All", "", 1) is True
    assert include_chunk(chunk, "Person B", "transcript", 5) is True
    assert include_chunk(chunk, "Person A", "", 1) is False
    assert include_chunk(chunk, "All", "missing", 1) is False
    assert include_chunk(chunk, "All", "", 6) is False
