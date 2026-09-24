"""JSON-lines logging on stdlib ``logging`` (Tech.md §14).

Fields passed via ``extra=`` become top-level keys, so call sites log structured data instead of
formatting it into the message. Never pass question text or raw IPs here (AGENTS.md §6.13).
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

# Attributes every LogRecord has; anything else on the record came from ``extra=``. uvicorn adds
# ``color_message`` (the message with ANSI color codes), which is noise in JSON.
_STANDARD_ATTRS = frozenset(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__.keys()
    | {"message", "asctime", "color_message"}
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


_HANDLER_NAME = "grounded-json"


def configure_logging(level: str) -> None:
    """Route the root logger to stderr as JSON. Idempotent, so app factories can call it freely.

    Only our own handler is replaced; handlers installed by others (pytest's caplog, a hosting
    platform's collector) stay attached.
    """
    handler = logging.StreamHandler(sys.stderr)
    handler.set_name(_HANDLER_NAME)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    for existing in [h for h in root.handlers if h.get_name() == _HANDLER_NAME]:
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)
