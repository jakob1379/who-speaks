from __future__ import annotations

import os
from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("sample-smoke")
    group.addoption(
        "--run-sample-smoke",
        action="store_true",
        default=False,
        help="Run sample media smoke tests that exercise real models.",
    )
    group.addoption(
        "--sample-media",
        action="store",
        default="samples/george-siemens-interview-90s.webm",
        help="Path to the checked-in sample media used by smoke tests.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "sample_smoke: sample media smoke test that may download model caches",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    enabled = (
        config.getoption("--run-sample-smoke") or os.getenv("RUN_SAMPLE_SMOKE") == "1"
    )
    if enabled:
        return

    skip_sample_smoke = pytest.mark.skip(
        reason="sample smoke test skipped; pass --run-sample-smoke or set RUN_SAMPLE_SMOKE=1"
    )
    for item in items:
        if "sample_smoke" in item.keywords:
            item.add_marker(skip_sample_smoke)


@pytest.fixture
def sample_media_path(request: pytest.FixtureRequest) -> Path:
    raw_path = Path(str(request.config.getoption("--sample-media")))
    path = (
        raw_path if raw_path.is_absolute() else Path(request.config.rootpath) / raw_path
    )
    if not path.exists():
        pytest.fail(
            f"Sample media not found: {path}. Expected a checked-in file such as "
            "samples/george-siemens-interview-90s.webm."
        )
    return path
