"""The global daily LLM budget and the daily rerank cap: atomic reserves in ``daily_usage``
(Tech.md §12, §8, DB.md §7.1).

``RerankCap`` (6.03) at the bottom is the same reserve on the ``rerank_calls`` column, with
``RERANK_DAILY_CAP`` as its limit; everything below about the reserve holds for it too, except
that it has no advisory check and no token totals.

**Reserve.** Every LLM call first takes one unit with a single upsert on the UTC day's row. The
``WHERE llm_calls < limit`` on the ``DO UPDATE`` makes the cap atomic: the upsert takes the row
lock, so N parallel requests at ``limit - 1`` get exactly one row back and the rest get none (no
row = exhausted). A validation retry is another call, so it reserves again, and a call that then
fails still used its unit: the provider may have counted it against the quota all the same.

**Advisory check.** One plain ``SELECT`` that reserves nothing, run after the answer-cache lookup
and before the query is embedded (5.01, PRD D31): an exhausted day then refuses a request without
spending a query embedding (a 1K-per-day quota that ingest and the evals share). A race can let a
request through it, which costs one embedding and nothing more: the reserve stays the one thing
that enforces the cap.

**Tokens** are added after each call with a plain ``UPDATE`` of the day's row. Bookkeeping that
fails after a successful call is logged and dropped: the user's answer exists and the row is a
statistic, not a gate. The one case that loses tokens silently is a call that crosses UTC midnight
(the ``UPDATE`` finds no row for the new day).

**The day is the database's** ``now()`` in UTC, like the rate limiter's window, so two instances
agree on it. **No cap, no budget:** ``open_runtime`` builds none when ``DAILY_LLM_BUDGET`` is unset
(dev and test) and none in eval mode (Tech §15.6), so ``daily_usage`` is not touched there.
"""

from __future__ import annotations

import logging

import psycopg
from psycopg_pool import AsyncConnectionPool

from grounded.generation.providers.base import Usage

logger = logging.getLogger(__name__)

# DB.md §7.1, verbatim. No row => the budget is exhausted.
_RESERVE = """
INSERT INTO daily_usage AS du (day, llm_calls)
VALUES ((now() AT TIME ZONE 'utc')::date, 1)
ON CONFLICT (day) DO UPDATE
    SET llm_calls = du.llm_calls + 1, updated_at = now()
    WHERE du.llm_calls < %(limit)s
RETURNING du.llm_calls
"""

_PEEK = "SELECT llm_calls FROM daily_usage WHERE day = (now() AT TIME ZONE 'utc')::date"

# The same statement for the rerank cap (DB.md §7.1). The insert branch takes the first unit of a
# day without looking at the limit, which is why ``RerankCap`` never runs it with a limit below 1.
_RESERVE_RERANK = """
INSERT INTO daily_usage AS du (day, rerank_calls)
VALUES ((now() AT TIME ZONE 'utc')::date, 1)
ON CONFLICT (day) DO UPDATE
    SET rerank_calls = du.rerank_calls + 1, updated_at = now()
    WHERE du.rerank_calls < %(limit)s
RETURNING du.rerank_calls
"""

_ADD_TOKENS = """
UPDATE daily_usage
SET input_tokens = input_tokens + %(input)s,
    output_tokens = output_tokens + %(output)s,
    updated_at = now()
WHERE day = (now() AT TIME ZONE 'utc')::date
"""


class BudgetExhaustedError(Exception):
    """Today's LLM budget is used up; ``api/errors.py`` makes it the 503 ``budget_exhausted``."""


class DailyBudget:
    def __init__(self, pool: AsyncConnectionPool, *, limit: int) -> None:
        self._pool = pool
        self._limit = limit

    async def check_available(self) -> None:
        """Advisory: raise when today's calls have reached the limit. Reserves nothing."""
        async with self._pool.connection() as conn:
            cursor = await conn.execute(_PEEK)
            row = await cursor.fetchone()
        if row is not None and row[0] >= self._limit:
            raise BudgetExhaustedError

    async def reserve(self) -> None:
        """Take one unit for the next LLM call, or raise when none is left."""
        async with self._pool.connection() as conn:
            cursor = await conn.execute(_RESERVE, {"limit": self._limit})
            if await cursor.fetchone() is None:
                raise BudgetExhaustedError

    async def add_tokens(self, usage: Usage) -> None:
        """Add the tokens a call used to today's row. Never raises for a database error."""
        try:
            async with self._pool.connection() as conn:
                await conn.execute(
                    _ADD_TOKENS, {"input": usage.input_tokens, "output": usage.output_tokens}
                )
        except psycopg.Error as exc:
            # Type and SQLSTATE only, like the request log: the message may carry row data.
            logger.error(
                "adding tokens to daily_usage failed",
                extra={"error": type(exc).__name__, "sqlstate": exc.sqlstate},
            )


class RerankCapReachedError(Exception):
    """Today's rerank calls have reached ``RERANK_DAILY_CAP``: the pipeline makes no call and
    uses the RRF order (Tech §8). Never an error of the request."""


class RerankCap:
    """One unit of ``daily_usage.rerank_calls`` per rerank call, reserved before the call.

    Fail-closed like the budget: the caller makes the call only after ``reserve`` returned, and a
    database error raises out of it, so the pipeline degrades instead of spending a call it
    could not count. A reserved unit is never given back: a call that fails, times out or is
    cancelled by the request deadline was attempted, and the Trial quota may have counted it.
    """

    def __init__(self, pool: AsyncConnectionPool, *, limit: int) -> None:
        self._pool = pool
        self._limit = limit

    async def reserve(self) -> None:
        """Take one unit for the next rerank call, or raise when none is left."""
        if self._limit < 1:
            # A cap of 0 is "rerank is off for today". The statement's insert branch would still
            # take the first unit of a day, so the database is not asked at all.
            raise RerankCapReachedError
        async with self._pool.connection() as conn:
            cursor = await conn.execute(_RESERVE_RERANK, {"limit": self._limit})
            if await cursor.fetchone() is None:
                raise RerankCapReachedError
