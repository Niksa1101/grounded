"""Liveness and readiness probes (Tech.md §13).

``/healthz`` is public. ``/readyz`` needs the proxy secret, like ``/v1`` (PRD D54c): it queries the
database, and a public probe would let anyone keep Neon awake (a ping every 4 minutes is about 180
CU-hours a month, against Neon Free's 100). Without the secret it answers 401 and touches nothing.
"""

from __future__ import annotations

import logging
from typing import Literal

import psycopg
from fastapi import APIRouter, Request, Response, status
from pydantic import BaseModel

from grounded.api.deps import PoolDep, SettingsDep
from grounded.api.security import check_proxy_secret
from grounded.schemas.api import ErrorResponse

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
    responses={
        status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ReadinessResponse},
    },
)
async def readyz(
    request: Request, pool: PoolDep, settings: SettingsDep, response: Response
) -> ReadinessResponse:
    """Database reachable, schema present and an index version active.

    Requires ``X-Proxy-Secret`` (or ``ALLOW_DIRECT_API`` in dev), checked before the pool is used,
    so a rejected call is a 401 in the shared error shape with no database access and no log row.

    Querying ``index_versions`` doubles as a "migrations applied" check. Without an active index
    ``/v1/ask`` cannot retrieve anything, so that is a readiness failure too (503, with
    ``database: ok``).
    """
    check_proxy_secret(request, settings)  # raises ProxyAuthError -> 401 (api/errors.py)
    try:
        async with pool.connection(timeout=settings.db_pool_timeout_s) as conn:
            cur = await conn.execute(_ACTIVE_INDEX_SQL)
            row = await cur.fetchone()
    except psycopg.Error as exc:  # includes psycopg_pool.PoolTimeout
        logger.warning("readiness check failed", extra={"error": type(exc).__name__})
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadinessResponse(status="unavailable", database="error", active_index_version=None)

    if row is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadinessResponse(status="unavailable", database="ok", active_index_version=None)
    git_ref, config_hash = row
    return ReadinessResponse(
        status="ok", database="ok", active_index_version=f"{git_ref}@{config_hash[:8]}"
    )
