"""``POST /v1/ask`` (Tech.md §13). Thin on purpose: validation happens in ``AskRequest``, the work
in ``AskPipeline``, error translation in ``api/errors.py``.

No proxy-secret check yet: the route is open in dev until ticket 5.03 adds it.
"""

from __future__ import annotations

from fastapi import APIRouter

from grounded.api.deps import PipelineDep, RequestIdDep
from grounded.schemas.api import AskRequest, AskResponse, ErrorResponse

router = APIRouter(prefix="/v1", tags=["ask"])


@router.post(
    "/ask",
    responses={
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def ask(body: AskRequest, pipeline: PipelineDep, request_id: RequestIdDep) -> AskResponse:
    return await pipeline.ask(body.question, request_id)
