"""The per-client rate limiter: Postgres fixed-window counters (Tech.md §12, DB.md §7.4, PRD D31).

One atomic upsert per window, keyed by ``(ip_hash, window_kind, window_start)``. The upsert takes
the row lock, so parallel requests of one client on different Vercel instances get distinct counts
(an in-memory counter would multiply the limit by the unknown number of instances, PRD §12).

**Order.** The minute window is counted first; only a request it admits touches the day window, so
a client that hammers the minute limit does not burn its daily quota on rejected requests. Both
upserts run in one transaction on one pooled connection (one checkout, and the locks are held for
the length of two statements). The order is the same for every request, so two requests cannot
wait on each other's rows in a cycle.

**The clock is the database's** (``now()``), never the instance's, so two instances agree on which
window a request falls in. Tests pass ``clock`` to place a request in a chosen minute and day
without waiting. The window boundaries are UTC regardless of the session time zone.

**Failure policy.** A counter query that fails raises, and the request becomes an ``internal_error``
(Tech §12). The limiter does not fail open: the request path needs the database anyway, and a
limiter that quietly stops limiting when the database misbehaves is a way around the budget.

**No hash key, no limit.** ``ip_hash`` is ``None`` only when ``IP_HASH_SECRET`` is unset, which
``Settings`` refuses in prod (``api/security.py``): a developer's machine or a test. There is no
honest key to count under (one shared bucket would make every local ``curl`` and every test share
a quota), so such a request is admitted. The app logs a warning at startup when this applies
(``main.py``). With the key set, a request without any client IP is not keyless: it is counted
under ``hash_ip("unknown")``, one bucket shared by all such requests (PRD D54e).

**Eval mode never gets a limiter** (``runtime.py``): a 30-question run from one process must not
be answered with our own 429 (Tech §15.6).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

WindowKind = Literal["minute", "day"]

# A pure UTC day is 24 hours (no DST in UTC), so adding the span in Python is exact; the same
# addition done by Postgres on a timestamptz would follow the session's time zone.
_SPANS: dict[WindowKind, timedelta] = {"minute": timedelta(minutes=1), "day": timedelta(days=1)}

# DB.md §7.4. ``%(now)s`` is NULL in production (the DB clock) and a fixed instant in tests. The
# window start is truncated in UTC and converted back, so it does not depend on the session zone.
_HIT = """
WITH clock AS (SELECT coalesce(%(now)s::timestamptz, now()) AS t)
INSERT INTO rate_limit_counters AS rl (ip_hash, window_kind, window_start, hits)
SELECT %(ip_hash)s::text, %(kind)s::text,
       date_trunc(%(kind)s::text, clock.t AT TIME ZONE 'utc') AT TIME ZONE 'utc', 1
FROM clock
ON CONFLICT (ip_hash, window_kind, window_start)
DO UPDATE SET hits = rl.hits + 1
RETURNING rl.hits, rl.window_start, (SELECT t FROM clock) AS now
"""


@dataclass(frozen=True, slots=True)
class RateDecision:
    """Whether a request may go on; if not, which window rejected it and for how long."""

    allowed: bool
    window: WindowKind | None = None  # the window that rejected; None when allowed
    retry_after_s: int = 0  # whole seconds to the end of that window, at least 1 when rejected


ALLOWED = RateDecision(allowed=True)


class RateLimitedError(Exception):
    """Raised by the route when the limiter said no; ``api/errors.py`` turns it into the 429."""

    def __init__(self, decision: RateDecision) -> None:
        super().__init__(f"rate limited by the {decision.window} window")
        self.window = decision.window
        self.retry_after_s = decision.retry_after_s


class RateLimiter:
    def __init__(
        self,
        pool: AsyncConnectionPool,
        *,
        per_minute: int,
        per_day: int,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._pool = pool
        self._limits: dict[WindowKind, int] = {"minute": per_minute, "day": per_day}
        self._clock = clock

    async def check(self, ip_hash: str | None) -> RateDecision:
        """Count this request in the minute window, then (if admitted) in the day window."""
        if ip_hash is None:
            return ALLOWED
        async with self._pool.connection() as conn:
            for kind in ("minute", "day"):
                decision = await self._count(conn, ip_hash, kind)
                if not decision.allowed:
                    return decision
        return ALLOWED

    async def _count(self, conn: AsyncConnection, ip_hash: str, kind: WindowKind) -> RateDecision:
        now = self._clock() if self._clock is not None else None
        cursor = await conn.execute(_HIT, {"now": now, "ip_hash": ip_hash, "kind": kind})
        row = await cursor.fetchone()
        if row is None:  # an upsert with RETURNING always returns its row
            raise RuntimeError("the rate-limit upsert returned no row")
        hits, window_start, db_now = row
        if hits <= self._limits[kind]:
            return ALLOWED
        remaining = (window_start + _SPANS[kind] - db_now).total_seconds()
        return RateDecision(False, kind, max(1, math.ceil(remaining)))
