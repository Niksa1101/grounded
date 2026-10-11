"""The body-size limit of ``POST /v1/ask`` (Tech.md §12), applied before the JSON is parsed.

The question rules (3-500 characters after trimming, no control characters) live in ``AskRequest``;
this module is the one that cannot: a body that is too big must be refused *before* it is parsed,
or before it is even read in full. ``read_limited_body`` answers from the ``Content-Length`` header
when there is one (nothing is read) and otherwise stops reading at the first byte over the limit,
so a chunked upload cannot make the server buffer more than ``MAX_BODY_BYTES`` + one chunk.

It returns a *new* ``Request`` that replays the bytes it read, because a stream can be read once and
FastAPI reads the body again to parse it. Only the ``receive`` callable is replaced: the scope, and
with it ``request.state`` (the trace, the request id), is the same object.
"""

from __future__ import annotations

from fastapi import Request
from starlette.types import Message

from grounded.schemas.api import MAX_BODY_BYTES


class InputRejectedError(Exception):
    """The request breaks an input limit before any validation model could see it (-> 422
    ``bad_request``, like an invalid field). ``message`` names the rule, never the input."""


async def read_limited_body(request: Request, limit: int = MAX_BODY_BYTES) -> Request:
    """``request`` with its body already read (at most ``limit`` bytes), or raises
    ``InputRejectedError``."""
    too_big = InputRejectedError(f"Invalid request. The body must be at most {limit} bytes.")
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdecimal() and int(declared) > limit:
        raise too_big
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise too_big
        chunks.append(chunk)
    body = b"".join(chunks)
    delivered = False

    async def replay() -> Message:
        nonlocal delivered
        if delivered:
            return {"type": "http.disconnect"}
        delivered = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(request.scope, replay)
