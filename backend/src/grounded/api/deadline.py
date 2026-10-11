"""The overall request deadline of ``POST /v1/ask`` (Tech §6, §18, PRD D53, 5.08).

``REQUEST_DEADLINE_S`` (25 s) sits between the stage timeouts and the platform's ``maxDuration``
(30 s), so that a slow request ends with our own 504 ``deadline_exceeded`` (JSON, ``request_id``,
a ``request_logs`` row) instead of the platform's ``FUNCTION_INVOCATION_TIMEOUT``.

``request_deadline`` is ``asyncio.timeout`` with one extra rule: only *its own* expiry becomes
``DeadlineExceededError``. A ``TimeoutError`` raised by the work inside (a stage that timed out on
its own) is not the deadline and passes through unchanged, so the error handlers keep telling the
two apart. The work is cancelled where it stands, which is the point: an LLM call in flight is
abandoned (its connection is closed) and a database call is cancelled; the connection pool discards
a connection that is returned mid-query. What cancellation leaves behind is documented in Tech §12:
a budget unit reserved before a call that then timed out stays spent.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager


class DeadlineExceededError(Exception):
    """The request outlived ``REQUEST_DEADLINE_S``; ``api/errors.py`` turns it into a 504."""

    def __init__(self, deadline_s: float) -> None:
        super().__init__(f"request deadline of {deadline_s:g}s exceeded")
        self.deadline_s = deadline_s


@asynccontextmanager
async def request_deadline(seconds: float | None) -> AsyncGenerator[None]:
    """Bound the ``async with`` body by ``seconds``; ``None`` means no deadline (eval mode)."""
    if seconds is None:
        yield
        return
    scope = asyncio.timeout(seconds)
    try:
        async with scope:
            yield
    except TimeoutError as exc:
        if scope.expired():
            raise DeadlineExceededError(seconds) from exc
        raise
