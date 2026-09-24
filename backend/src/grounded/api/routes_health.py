"""Liveness and readiness probes (Tech.md §13). Both are public: no proxy secret required."""

from __future__ import annotations

import logging
from typing import Literal

import psycopg
from fastapi import APIRouter, Response, status
from pydantic import BaseModel

from grounded.api.deps import PoolDep, SettingsDep

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

_ACTIVE_INDEX_SQL = """
SELECT git_ref, config_hash
FROM index_versions
WHERE is_active
"""


class HealthResponse(BaseModel):
    status: Literal["ok"]


class ReadinessResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    database: Literal["ok", "error"]
    # "<git_ref>@<config_hash[:8]>" (same format as AskResponse.meta.index_version), or None.
    active_index_version: str | None


@router.get("/healthz")
async def healthz() -> HealthResponse:
    """Process is up. Touches nothing else, so the keepalive ping never depends on the database."""
    return HealthResponse(status="ok")


@router.get(
    "/readyz",
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ReadinessResponse}},
)
async def readyz(pool: PoolDep, settings: SettingsDep, response: Response) -> ReadinessResponse:
    """Database reachable and schema present.

    Querying ``index_versions`` doubles as a "migrations applied" check. Having an active index is
    reported but not yet required: nothing serves answers before Phase 3, when ``/v1/ask`` lands and
    a missing active index becomes a readiness failure.
    """
    try:
        async with pool.connection(timeout=settings.db_pool_timeout_s) as conn:
            cur = await conn.execute(_ACTIVE_INDEX_SQL)
            row = await cur.fetchone()
    except psycopg.Error as exc:  # includes psycopg_pool.PoolTimeout
        logger.warning("readiness check failed", extra={"error": type(exc).__name__})
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadinessResponse(status="unavailable", database="error", active_index_version=None)

    active = None
    if row is not None:
        git_ref, config_hash = row
        active = f"{git_ref}@{config_hash[:8]}"
    return ReadinessResponse(status="ok", database="ok", active_index_version=active)
