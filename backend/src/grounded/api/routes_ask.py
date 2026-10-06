"""``POST /v1/ask`` (Tech.md §13). Thin on purpose: validation happens in ``AskRequest``, the work
in ``AskPipeline``, error translation in ``api/errors.py``.

Every request writes one ``request_logs`` row (3.12). ``AskRoute`` wraps the route so the trace
starts before the body is parsed (a malformed body is a ``bad_request`` row too) and logs the
success; the error handlers log the failures, each exactly once (``RequestTrace.written``).

No proxy-secret check yet: the route is open in dev until ticket 5.03 adds it.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

from fastapi import APIRouter, Request, Response
from fastapi.routing import APIRoute

from grounded.api.deps import PipelineDep, RequestIdDep, TraceDep, get_request_logger, request_id_of
from grounded.schemas.api import AskRequest, AskResponse, ErrorResponse


class AskRoute(APIRoute):
    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        handle = super().get_route_handler()

        async def logged(request: Request) -> Response:
            request_logger = get_request_logger(request)
            trace = request_logger.start(request_id_of(request))
            request.state.trace = trace
            response = await handle(request)  # a failure raises and is logged by its handler
            assert trace.status is not None  # the pipeline sets it whenever it returns an answer
            await request_logger.write(
                trace, outcome=trace.status, http_status=response.status_code
            )
            return response

        return logged


router = APIRouter(prefix="/v1", tags=["ask"], route_class=AskRoute)


@router.post(
    "/ask",
    responses={
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def ask(
    body: AskRequest, pipeline: PipelineDep, request_id: RequestIdDep, trace: TraceDep
) -> AskResponse:
    trace.question = body.question
    return await pipeline.ask(body.question, request_id, trace=trace)
