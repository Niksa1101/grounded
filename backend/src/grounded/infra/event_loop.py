"""The event loop async psycopg can run on, in one place.

psycopg's async mode cannot run on the Proactor loop that ``asyncio`` picks by default on Windows,
so everything that owns a loop (the CLI's ``asyncio.run``, the promptfoo provider's loop thread)
asks here instead of repeating the platform check. ``grounded serve`` passes uvicorn the same choice
as a string (``loop="asyncio:SelectorEventLoop"``), because uvicorn builds its own loop.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable


def loop_factory() -> Callable[[], asyncio.AbstractEventLoop] | None:
    """For ``asyncio.run(..., loop_factory=...)``: the Selector loop on Windows, else the default
    (``None``)."""
    return asyncio.SelectorEventLoop if sys.platform == "win32" else None


def new_event_loop() -> asyncio.AbstractEventLoop:
    """A new loop psycopg async can use, for code that keeps a loop alive across calls."""
    factory = loop_factory()
    return factory() if factory is not None else asyncio.new_event_loop()
