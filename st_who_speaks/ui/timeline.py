from __future__ import annotations

import altair as alt
import pandas as pd

from st_who_speaks.colors import build_speaker_color_map
from st_who_speaks.models import ProcessingResult

SPEAKER_N_FIELD = "speaker:N"


def build_timeline_chart(result: ProcessingResult, selected_time: float) -> alt.Chart:
    turns_frame = build_timeline_turns_frame(result)
    duration_limit = max(result.duration, 1.0)
    speaker_colors = build_speaker_color_map(result.speakers)
    return (
        (
            build_timeline_bars(turns_frame, speaker_colors, duration_limit)
            + build_timeline_indicator(selected_time, duration_limit)
        )
        .properties(height=72)
        .configure_view(strokeWidth=0)
    )


def build_timeline_turns_frame(result: ProcessingResult) -> pd.DataFrame:
    turns_frame = pd.DataFrame(
        [
            {
                "speaker": turn.label,
                "start": turn.start,
                "end": turn.end,
                "duration": turn.duration,
                "track": "Speaker activity",
            }
            for turn in result.speaker_turns
        ]
    )
    if not turns_frame.empty:
        return turns_frame
    return pd.DataFrame(
        [
            {
                "speaker": "No speaker data",
                "start": 0.0,
                "end": max(result.duration, 0.0),
                "duration": 0.0,
                "track": "Speaker activity",
            }
        ]
    )


def build_timeline_bars(
    turns_frame: pd.DataFrame, speaker_colors: dict[str, str], duration_limit: float
) -> alt.Chart:
    color_encoding = (
        alt.Color(
            SPEAKER_N_FIELD,
            scale=alt.Scale(
                domain=list(speaker_colors.keys()),
                range=list(speaker_colors.values()),
            ),
            legend=None,
        )
        if speaker_colors
        else alt.Color(SPEAKER_N_FIELD, legend=None)
    )
    return (
        alt.Chart(turns_frame)
        .mark_bar(size=28, cornerRadius=6)
        .encode(
            x=alt.X(
                "start:Q",
                title="Seconds",
                scale=alt.Scale(domain=[0, duration_limit]),
            ),
            x2="end:Q",
            y=alt.Y("track:N", title=None, axis=None),
            color=color_encoding,
            tooltip=[
                alt.Tooltip(SPEAKER_N_FIELD, title="Speaker"),
                alt.Tooltip("start:Q", title="Start", format=".2f"),
                alt.Tooltip("end:Q", title="End", format=".2f"),
                alt.Tooltip("duration:Q", title="Duration", format=".2f"),
            ],
        )
    )


def build_timeline_indicator(selected_time: float, duration_limit: float) -> alt.Chart:
    indicator_frame = pd.DataFrame(
        [{"selected_time": min(max(selected_time, 0.0), duration_limit)}]
    )
    return (
        alt.Chart(indicator_frame)
        .mark_rule(color="#111827", strokeWidth=3)
        .encode(x=alt.X("selected_time:Q", scale=alt.Scale(domain=[0, duration_limit])))
    )
