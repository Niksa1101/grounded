"""``POST /v1/ask`` (Tech.md §13). Thin on purpose: validation happens in ``AskRequest``, the work
in ``AskPipeline``, error translation in ``api/errors.py``.

Every request writes one ``request_logs`` row (3.12). ``AskRoute`` wraps the route so the trace
starts before the body is parsed (a malformed body is a ``bad_request`` row too) and logs the
success; the error handlers log the failures, each exactly once (``RequestTrace.written``).

The proxy secret is checked first of all (``api/security.py``), before the trace starts: a caller
that fails it gets a 401 and no ``request_logs`` row, and one that passes it gives the trace its
``source`` and ``ip_hash``.

The order after that is Tech §1's "validate -> rate-limit -> cache lookup": the body-size limit
runs in ``AskRoute`` before the JSON is parsed (``api/input_limits.py``), ``AskRequest`` validates
the question, and only then does the endpoint count the request against the rate limit
(``infra/ratelimit.py``) and hand it to the pipeline, whose first lookup is the answer cache. So a
malformed request costs no counter write, and a cache hit still counts. The limit has to be checked
in the endpoint, not in a dependency: FastAPI runs every dependency before it reports a body that
failed validation, which would count invalid requests.

The deadline (``api/deadline.py``, 5.08) starts right after authentication and bounds everything
that follows: the body read, validation, the rate-limit upsert, the cache lookup, the budget check
and reserve, the stages and the response. A database call that hangs is as good a reason for a 504
as a slow model, and the platform's clock (``maxDuration``) counts all of it. The failure
write-up of a timed-out request happens *after* the deadline, in the error handler, so the row is
still written. Eval mode has no deadline: it does not go through this route, and its LLM timeout
(``EVAL_LLM_TIMEOUT_S``, 40 s) is longer than the deadline on purpose (Tech §15.6).
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

from fastapi import APIRouter, Request, Response
from fastapi.routing import APIRoute

from grounded.api.deadline import request_deadline
from grounded.api.deps import (
    PipelineDep,
    RateLimiterDep,
    RequestIdDep,
    TraceDep,
    get_request_logger,
    get_settings_dep,
    request_id_of,
)
from grounded.api.input_limits import read_limited_body
from grounded.api.security import authenticate
from grounded.infra.ratelimit import RateLimitedError
from grounded.schemas.api import AskRequest, AskResponse, ErrorResponse


class AskRoute(APIRoute):
    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        handle = super().get_route_handler()

        async def logged(request: Request) -> Response:
            client = authenticate(request)  # raises ProxyAuthError -> 401 (no trace, no row)
            request_logger = get_request_logger(request)
            trace = request_logger.start(
                request_id_of(request), source=client.source, ip_hash=client.ip_hash
            )
            request.state.trace = trace
            settings = get_settings_dep(request)
            deadline_s = None if settings.app_env == "eval" else settings.request_deadline_s
            # The router starts an LLM attempt only if the time left holds its whole timeout
            # (Tech §10); the pipeline reads the time left from the trace, on its clock.
            trace.deadline_s = deadline_s
            async with request_deadline(deadline_s):  # DeadlineExceededError -> 504, logged
                request = await read_limited_body(request)  # InputRejectedError -> 422, logged
                response = await handle(request)  # a failure raises and is logged by its handler
            if trace.status is None:
                # The pipeline sets it whenever it returns an answer, so this is a bug. Raised (not
                # asserted, which ``python -O`` would strip) so the internal-error handler logs it
                # and writes the row instead of a row with no outcome.
                raise RuntimeError("the pipeline returned an answer without setting trace.status")
            await request_logger.write(
                trace, outcome=trace.status, http_status=response.status_code
            )
            return response

        return logged


router = APIRouter(prefix="/v1", tags=["ask"], route_class=AskRoute)


@router.post(
    "/ask",
    responses={
        401: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
)
async def ask(
    body: AskRequest,
    pipeline: PipelineDep,
    request_id: RequestIdDep,
    trace: TraceDep,
    limiter: RateLimiterDep,
) -> AskResponse:
    trace.question = body.question
    if limiter is not None:  # None in eval mode: the limiter is not consulted (Tech §15.6)
        decision = await limiter.check(trace.ip_hash)
        if not decision.allowed:
            raise RateLimitedError(decision)
    return await pipeline.ask(body.question, request_id, trace=trace)
