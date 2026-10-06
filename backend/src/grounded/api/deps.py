"""FastAPI dependencies that expose app-scoped resources created in the lifespan."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, cast
from uuid import UUID, uuid4

from fastapi import Depends, Request
from psycopg_pool import AsyncConnectionPool

from grounded.settings import Settings

if TYPE_CHECKING:
    from grounded.generation.pipeline import AskPipeline
    from grounded.observability.request_log import RequestLogger, RequestTrace


def get_settings_dep(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


def get_pool(request: Request) -> AsyncConnectionPool:
    return cast(AsyncConnectionPool, request.app.state.db_pool)


def get_pipeline(request: Request) -> AskPipeline:
    return cast("AskPipeline", request.app.state.pipeline)


def get_request_logger(request: Request) -> RequestLogger:
    return cast("RequestLogger", request.app.state.request_logger)


def trace_of(request: Request) -> RequestTrace:
    """The trace ``AskRoute`` started for this request (api/routes_ask.py)."""
    return cast("RequestTrace", request.state.trace)


def request_id_of(request: Request) -> UUID:
    """The id of this request, created on first use and kept on ``request.state``.

    Lazy on purpose: error handlers run outside the dependency system and still need the same id
    the route would have used, so both ask here instead of relying on a middleware having run.
    """
    request_id = getattr(request.state, "request_id", None)
    if request_id is None:
        request_id = uuid4()
        request.state.request_id = request_id
    return cast(UUID, request_id)


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
PoolDep = Annotated[AsyncConnectionPool, Depends(get_pool)]
PipelineDep = Annotated["AskPipeline", Depends(get_pipeline)]
RequestIdDep = Annotated[UUID, Depends(request_id_of)]
TraceDep = Annotated["RequestTrace", Depends(trace_of)]
