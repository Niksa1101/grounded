from __future__ import annotations

import json
import logging
from collections.abc import Iterator

import pytest

from grounded.infra.logging import JsonFormatter, configure_logging


@pytest.fixture
def restore_root_logger() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


@pytest.mark.usefixtures("restore_root_logger")
def test_configure_logging_is_idempotent_and_keeps_foreign_handlers() -> None:
    # e.g. pytest's caplog handler, which tests of the app rely on
    foreign = logging.NullHandler()
    logging.getLogger().addHandler(foreign)

    configure_logging("INFO")
    configure_logging("DEBUG")

    root = logging.getLogger()
    assert foreign in root.handlers
    assert sum(isinstance(h.formatter, JsonFormatter) for h in root.handlers) == 1
    assert root.level == logging.DEBUG


def test_json_formatter_keeps_extras_and_drops_uvicorn_color_message() -> None:
    record = logging.LogRecord("uvicorn.error", logging.INFO, "", 0, "Started %s", ("x",), None)
    record.request_id = "r-1"
    record.color_message = "Started [36m%s[0m"
    payload = json.loads(JsonFormatter().format(record))
    assert payload["msg"] == "Started x"
    assert payload["request_id"] == "r-1"
    assert "color_message" not in payload
