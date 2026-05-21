from __future__ import annotations

SPEAKER_COLOR_NAMES = {
    "#ef4444": "red",
    "#22c55e": "green",
    "#3b82f6": "blue",
    "#f59e0b": "amber",
    "#a855f7": "violet",
    "#14b8a6": "teal",
    "#f97316": "orange",
    "#eab308": "yellow",
}

SPEAKER_COLOR_PALETTE = list(SPEAKER_COLOR_NAMES.keys())


def build_speaker_color_map(speakers: list[str]) -> dict[str, str]:
    return {
        speaker: SPEAKER_COLOR_PALETTE[index % len(SPEAKER_COLOR_PALETTE)]
        for index, speaker in enumerate(speakers)
    }


def describe_color_hex(color_hex: str | None) -> str:
    if color_hex is None:
        return "unassigned"
    color_name = SPEAKER_COLOR_NAMES.get(color_hex.lower(), "custom")
    return f"{color_name} ({color_hex})"
