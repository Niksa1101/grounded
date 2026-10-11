"""The shared error shape and the handlers that produce it (Tech.md §13).

Every error is ``{"error": {"code", "message", "retry_after_s"}, "request_id"}``. Messages are
written for the caller: no stack traces, no internals, and no echo of the question text.
Exceptions that already carry a meaning (provider errors, a missing index) are translated here, so
routes stay thin and the pipeline never imports anything from the HTTP layer.

Every handler goes through ``_fail``, which also writes the ``request_logs`` row of a failed
``/v1/ask`` request (3.12): this is the one place that knows both the HTTP status and the error
code, so the log cannot disagree with the response. Requests of other routes have no trace and are
not logged.
"""

from __future__ import annotations

import logging
import math
from typing import Any, cast

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from grounded.api.deadline import DeadlineExceededError
from grounded.api.deps import get_request_logger, request_id_of
from grounded.api.input_limits import InputRejectedError
from grounded.api.security import ProxyAuthError
from grounded.infra.budget import BudgetExhaustedError
from grounded.infra.hashing import question_hash
from grounded.infra.provider_errors import (
    ProviderBadOutput,
    ProviderRateLimited,
    ProviderTimeout,
    ProviderUnavailable,
)
from grounded.infra.ratelimit import RateLimitedError
from grounded.ingest.embed import EmbedderUnavailableError
from grounded.observability.request_log import Outcome, RequestTrace
from grounded.retrieval.index import NoActiveIndexError
from grounded.schemas.api import ErrorBody, ErrorCode, ErrorResponse

logger = logging.getLogger(__name__)


def error_response(
    request: Request,
    http_status: int,
    code: ErrorCode,
    message: str,
    *,
    retry_after_s: float | None = None,
) -> JSONResponse:
    retry_after = None if retry_after_s is None else max(1, math.ceil(retry_after_s))
    body = ErrorResponse(
        error=ErrorBody(code=code, message=message, retry_after_s=retry_after),
        request_id=request_id_of(request),
    )
    headers = None if retry_after is None else {"Retry-After": str(retry_after)}
    return JSONResponse(body.model_dump(mode="json"), status_code=http_status, headers=headers)


async def _fail(
    request: Request,
    http_status: int,
    code: ErrorCode,
    message: str,
    *,
    retry_after_s: float | None = None,
    outcome: Outcome | None = None,
) -> JSONResponse:
    trace: RequestTrace | None = getattr(request.state, "trace", None)
    if trace is not None:
        # Every ErrorCode but one is also a request_logs outcome (DB.md §4); the exception passes
        # ``outcome`` (see ``_deadline_exceeded``).
        await get_request_logger(request).write(
            trace,
            outcome=outcome or cast(Outcome, code),
            http_status=http_status,
            error_code=code,
        )
    return error_response(request, http_status, code, message, retry_after_s=retry_after_s)


async def _bad_request(request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, RequestValidationError):  # registered for that type only
        return await _internal_error(request, exc)
    trace: RequestTrace | None = getattr(request.state, "trace", None)
    if trace is not None:
        # The rejected question is not stored, but its hash is: dedup stats see it all the same.
        body: object = exc.body
        if isinstance(body, dict):
            text: object = cast("dict[str, object]", body).get("question")
            if isinstance(text, str):
                trace.question_hash = question_hash(text)
    # Location and rule only. The default detail echoes the rejected input, i.e. the question.
    problems = "; ".join(
        f"{'.'.join(str(part) for part in err['loc'] if part != 'body')}: {err['msg']}"
        for err in exc.errors()
    )
    return await _fail(
        request,
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        "bad_request",
        f"Invalid request. {problems}",
    )


async def _unauthorized(request: Request, exc: Exception) -> JSONResponse:
    # Raised before the request trace exists (api/security.py), so ``_fail`` writes no row. The
    # body does not say which check failed; the log does (a deploy with a wrong secret is the
    # usual cause, and the reason is the first thing to look at).
    reason = exc.reason if isinstance(exc, ProxyAuthError) else "unknown"
    logger.warning(
        "proxy auth rejected", extra={"request_id": str(request_id_of(request)), "reason": reason}
    )
    return await _fail(
        request, status.HTTP_401_UNAUTHORIZED, "unauthorized", "Missing or invalid credentials."
    )


async def _input_rejected(request: Request, exc: Exception) -> JSONResponse:
    # A body over the size limit (api/input_limits.py): the same code and status as an invalid
    # field. Like a body that failed validation, it keeps no question (there is none to read).
    return await _fail(request, status.HTTP_422_UNPROCESSABLE_CONTENT, "bad_request", str(exc))


async def _rate_limited(request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, RateLimitedError):  # registered for that type only
        return await _internal_error(request, exc)
    wait = exc.retry_after_s
    if exc.window == "day":
        message = f"Daily question limit reached. Try again in about {math.ceil(wait / 3600)} h."
    else:
        message = f"Too many requests. Try again in {wait}s."
    return await _fail(
        request, status.HTTP_429_TOO_MANY_REQUESTS, "rate_limited", message, retry_after_s=wait
    )


async def _budget_exhausted(request: Request, exc: Exception) -> JSONResponse:
    # No Retry-After: the day ends at UTC midnight, and the demo's visitors read "tomorrow" from
    # the message; the UI shows its own friendly state for the code (Tech §19).
    return await _fail(
        request,
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "budget_exhausted",
        "The demo's daily budget has been reached. Try again tomorrow (the day ends at 00:00 UTC).",
    )


async def _provider_unavailable(request: Request, exc: Exception) -> JSONResponse:
    retry_after = exc.retry_after_s if isinstance(exc, ProviderRateLimited) else None
    logger.warning(
        "provider unavailable",
        extra={"request_id": str(request_id_of(request)), "error": type(exc).__name__},
    )
    return await _fail(
        request,
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "provider_unavailable",
        "The answer service is temporarily unavailable. Try again shortly.",
        retry_after_s=retry_after,
    )


async def _deadline_exceeded(request: Request, exc: Exception) -> JSONResponse:
    trace: RequestTrace | None = getattr(request.state, "trace", None)
    logger.warning(
        "request deadline exceeded",
        extra={
            "request_id": str(request_id_of(request)),
            "deadline_s": exc.deadline_s if isinstance(exc, DeadlineExceededError) else None,
            # The stages so far, in ms (0: not reached). The same numbers go to the row.
            "stage_latency_ms": trace.latency_ms() if trace is not None else None,
        },
    )
    # The outcome is ``internal_error``: DB.md §4's CHECK has no value for a deadline and a new one
    # needs a migration. The row's ``error_code`` and ``http_status`` (504) tell it apart.
    return await _fail(
        request,
        status.HTTP_504_GATEWAY_TIMEOUT,
        "deadline_exceeded",
        "The request took too long and was stopped. Try again shortly.",
        outcome="internal_error",
    )


async def _validation_failed(request: Request, exc: Exception) -> JSONResponse:
    logger.warning("model output invalid", extra={"request_id": str(request_id_of(request))})
    return await _fail(
        request,
        status.HTTP_502_BAD_GATEWAY,
        "validation_failed",
        "The model returned an answer that could not be validated.",
    )


async def _internal_error(request: Request, exc: Exception) -> JSONResponse:
    # The traceback goes to the log, keyed by request_id; the body only says "internal error".
    logger.error(
        "unhandled error",
        exc_info=exc,
        extra={"request_id": str(request_id_of(request))},
    )
    return await _fail(
        request, status.HTTP_500_INTERNAL_SERVER_ERROR, "internal_error", "Internal error."
    )


async def _no_active_index(request: Request, exc: Exception) -> JSONResponse:
    logger.error("no active index", extra={"request_id": str(request_id_of(request))})
    return await _fail(
        request,
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "internal_error",
        "The service is not ready: no index is active.",
    )


def install_error_handlers(app: FastAPI) -> None:
    handlers: dict[type[Exception], Any] = {
        RequestValidationError: _bad_request,
        ProxyAuthError: _unauthorized,
        InputRejectedError: _input_rejected,
        RateLimitedError: _rate_limited,
        DeadlineExceededError: _deadline_exceeded,
        BudgetExhaustedError: _budget_exhausted,
        ProviderRateLimited: _provider_unavailable,
        ProviderUnavailable: _provider_unavailable,
        ProviderTimeout: _provider_unavailable,
        EmbedderUnavailableError: _provider_unavailable,
        ProviderBadOutput: _validation_failed,
        NoActiveIndexError: _no_active_index,
        Exception: _internal_error,
    }
    for exc_type, handler in handlers.items():
        app.add_exception_handler(exc_type, handler)
