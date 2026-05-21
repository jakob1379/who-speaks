from __future__ import annotations

from typing import Any

from st_who_speaks import logging as logging_module


def test_configure_logging_is_idempotent(monkeypatch) -> None:
    basic_config_calls: list[dict[str, Any]] = []
    structlog_configure_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(logging_module, "_CONFIGURED", False)
    monkeypatch.setattr(
        logging_module.logging,
        "basicConfig",
        lambda **kwargs: basic_config_calls.append(kwargs),
    )
    monkeypatch.setattr(
        logging_module.structlog,
        "configure",
        lambda **kwargs: structlog_configure_calls.append(kwargs),
    )

    logging_module.configure_logging()
    logging_module.configure_logging()

    assert basic_config_calls == [
        {
            "level": logging_module.logging.INFO,
            "format": "%(message)s",
            "stream": logging_module.sys.stdout,
        }
    ]
    assert len(structlog_configure_calls) == 1
    structlog_kwargs = structlog_configure_calls[0]
    assert len(structlog_kwargs["processors"]) == 5
    assert structlog_kwargs["cache_logger_on_first_use"] is True
    assert logging_module._CONFIGURED is True


def test_get_logger_delegates_without_configuring(monkeypatch) -> None:
    events: list[tuple[str, str]] = []
    logger = object()

    monkeypatch.setattr(
        logging_module,
        "configure_logging",
        lambda: (_ for _ in ()).throw(AssertionError("configure should not run")),
    )
    monkeypatch.setattr(
        logging_module.structlog,
        "get_logger",
        lambda name: events.append(("get_logger", name)) or logger,
    )

    assert logging_module.get_logger("st_who_speaks.pipeline") is logger
    assert events == [
        ("get_logger", "st_who_speaks.pipeline"),
    ]
