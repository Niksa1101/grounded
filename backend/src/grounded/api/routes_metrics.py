"""``GET /v1/metrics/summary`` (Tech.md §13, §14; ticket 8.02): the dashboard's aggregates.

Thin on purpose: the SQL and the arithmetic live in ``observability/metrics_summary.py``, the
contract (fields, denominators, rounding) in ``schemas/metrics.py``.

- **Proxy secret first.** ``require_proxy_secret`` is a router dependency, so it runs before the
  query parameter is validated and before the pool is touched: a stranger gets a 401 in the shared
  error shape, learns no parameter rule and wakes no database (the same reasoning as ``/readyz``).
- **Invalid ``window``** is a 422 ``bad_request`` from the shared handler, which names the parameter
  and the allowed values and never echoes the input.
- **No ``request_logs`` row.** There is no request trace on this route, so neither a success nor a
  failure writes one: reading the dashboard is not a question.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from grounded.api.deps import PoolDep
from grounded.api.security import require_proxy_secret
from grounded.observability.metrics_summary import fetch_metrics_summary
from grounded.schemas.api import ErrorResponse
from grounded.schemas.metrics import MetricsSummary, Window

router = APIRouter(prefix="/v1", tags=["metrics"], dependencies=[Depends(require_proxy_secret)])


@router.get(
    "/metrics/summary",
    responses={
        401: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def metrics_summary(
    pool: PoolDep,
    window: Annotated[Window, Query(description="Rolling period of the aggregates.")] = "7d",
) -> MetricsSummary:
    """Aggregates only: request counts, rates, stage latency, shadow cost, a daily series and the
    latest eval run per (suite, config). Never question text, a hash of it, an IP hash or a
    request id."""
    async with pool.connection() as conn:
        return await fetch_metrics_summary(conn, window)
