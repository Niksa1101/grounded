"""Retention and housekeeping: the SQL of DB.md §9, run by ``grounded housekeeping`` (ticket 5.11).

Privacy is enforced here, in tested code, not in workflow YAML (PRD NFR-4, AGENTS.md §6.13):
question text and the IP hash are removed after 29 days, telemetry rows after 180, expired answers
(and answers whose question is older than the text boundary) from the cache, old usage counters and
rate-limit windows that can no longer matter.

**29 days keep a promise of 30.** NFR-4 says user text is kept at most 30 days
(``USER_TEXT_MAX_AGE_DAYS``). The job runs once a day and deletes what is *older than* its
boundary, so a row written just after one run would be cleaned by the run 30 days later and lived up
to 31. Deleting at 29 days (``USER_TEXT_RETENTION_DAYS``) leaves a day of slack for that gap. A cron
that GitHub delays or switches off still lengthens the period; the README says so (D56a).

**Owner role.** The runtime's ``app`` role has no DELETE on ``request_logs``, ``daily_usage`` or
``rate_limit_counters`` and no UPDATE on ``request_logs`` (DB.md §5), by design: a bug or a leaked
``DATABASE_URL`` in the request path cannot erase or rewrite the log. Retention therefore runs with
the owner connection (``DATABASE_URL_DIRECT``), from the daily workflow.

**One transaction.** All rules run in a single transaction, so ``now()`` is the same instant for
every rule and a failure in one leaves nothing half-applied. ``dry_run`` runs the very same
statements and rolls back: the counts are the real ones, with no second copy of the conditions that
could drift from the deleting ones.

**The periods are contract, not tunables.** They come from PRD NFR-4 and DB.md §9 (and the answer
cache's TTL is capped at the same 29 days, ``Settings.answer_cache_ttl_days``), so they are
constants here and not ``Settings`` fields: a setting that could lengthen the privacy retention is
a hole. They are bound as parameters. ``tests/unit/test_housekeeping_docs.py`` keeps the statements
and DB.md §9 identical.

**Counts.** A row older than 180 days that still has its question is first counted by rule 1 (its
text is removed) and then by rule 2 (the row is deleted). A second run counts 0 everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import psycopg

# PRD NFR-4 / DB.md §9. Not configuration: see the module docstring.
USER_TEXT_MAX_AGE_DAYS: Final = 30  # the promise made to users (UI, README)
USER_TEXT_RETENTION_DAYS: Final = USER_TEXT_MAX_AGE_DAYS - 1  # the boundary the daily job applies
TELEMETRY_RETENTION_DAYS: Final = 180
USAGE_RETENTION_DAYS: Final = 90
RATE_LIMIT_RETENTION_DAYS: Final = 2


@dataclass(frozen=True)
class Rule:
    """One statement of DB.md §9. ``name`` is the stable label of its line of output."""

    name: str
    sql: str
    params: dict[str, int]


# Keep each ``sql`` equal to the block of DB.md §9 (a test compares them, whitespace aside).
RULES: Final[tuple[Rule, ...]] = (
    Rule(
        name="request_logs_text_removed",
        sql="""
UPDATE request_logs SET question = NULL, ip_hash = NULL
WHERE created_at < now() - make_interval(days => %(text_days)s)
  AND (question IS NOT NULL OR ip_hash IS NOT NULL)
""",
        params={"text_days": USER_TEXT_RETENTION_DAYS},
    ),
    Rule(
        name="request_logs_deleted",
        sql="""
DELETE FROM request_logs
WHERE created_at < now() - make_interval(days => %(telemetry_days)s)
""",
        params={"telemetry_days": TELEMETRY_RETENTION_DAYS},
    ),
    Rule(
        name="answer_cache_expired_deleted",
        sql="""
DELETE FROM answer_cache
WHERE expires_at < now()
   OR created_at < now() - make_interval(days => %(text_days)s)
""",
        params={"text_days": USER_TEXT_RETENTION_DAYS},
    ),
    Rule(
        name="daily_usage_deleted",
        sql="""
DELETE FROM daily_usage
WHERE day < (now() AT TIME ZONE 'utc')::date - %(usage_days)s
""",
        params={"usage_days": USAGE_RETENTION_DAYS},
    ),
    Rule(
        name="rate_limit_counters_deleted",
        sql="""
DELETE FROM rate_limit_counters
WHERE window_start < now() - make_interval(days => %(window_days)s)
""",
        params={"window_days": RATE_LIMIT_RETENTION_DAYS},
    ),
)


@dataclass(frozen=True)
class RuleResult:
    name: str
    rows: int


def run_housekeeping(conninfo: str, *, dry_run: bool = False) -> list[RuleResult]:
    """Apply every rule in one transaction on ``conninfo`` and return the affected row counts.

    ``dry_run`` rolls the transaction back after counting. Any database error propagates and
    nothing is committed.
    """
    results: list[RuleResult] = []
    with psycopg.connect(conninfo) as conn:
        for rule in RULES:
            cur = conn.execute(rule.sql, rule.params)  # pyright: ignore[reportArgumentType]
            results.append(RuleResult(rule.name, cur.rowcount))
        if dry_run:
            conn.rollback()
        else:
            conn.commit()
    return results


def render_results(results: list[RuleResult], *, dry_run: bool) -> list[str]:
    """The output lines: one per rule in a fixed order and format, then a total line."""
    width = max(len(r.name) for r in results)
    lines = [f"  {r.name.ljust(width)}  {r.rows}" for r in results]
    total = sum(r.rows for r in results)
    if dry_run:
        lines.append(f"Dry run: {total} rows would be affected; nothing was changed.")
    else:
        lines.append(f"Done: {total} rows affected.")
    return lines
