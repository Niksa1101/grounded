# R.27 — Retention boundary: delete user text at 29 days

> **Before you start:** read [AGENTS.md](../../AGENTS.md), the rules in [tickets/README.md](../README.md) (how to work a ticket, ticket types, [standard checks](../README.md#standard-checks)) and the phase notes in [README.md](README.md#second-review-follow-ups-2026-10-12). Work only on this ticket.

> **Owner:** G · **Type:** Build · **Branch:** `phase-5/retention-boundary` · **Blocked by:** R.26 ·
> **Builds on:** `infra/housekeeping.py`, `infra/answer_cache.py`, `settings.py`

**What to build:** review item #1 and decision D56a. The daily job removes question text only once it is *older than*
30 days, so a row written just after one run is cleaned by the run 31 days later: the text lives up to about 31 days,
while the UI and the README promise "at most 30 days" (PRD NFR-4). The job now deletes at **29 days**, so the promise
holds with a daily job. The answer cache gets the same bound.

**Read first:** DB §4 (`answer_cache`), §7.2, §9; Tech §4, §11; PRD NFR-4, D32, D56; `infra/answer_cache.py` (`_PUT`).

**Scope notes**
- **`infra/housekeeping.py`.** Two constants: `USER_TEXT_MAX_AGE_DAYS: Final = 30` (the promise of NFR-4) and
  `USER_TEXT_RETENTION_DAYS = USER_TEXT_MAX_AGE_DAYS - 1` (the boundary the daily job applies). The docstring says why
  they differ: a daily job can be up to one day late, and a late or disabled GitHub cron lengthens the period anyway
  (README Privacy says so). `text_days` is bound to the boundary, not to the promise.
- **The cache rule** becomes `DELETE FROM answer_cache WHERE expires_at < now() OR created_at < now() -
  make_interval(days => %(text_days)s)`. Deleting by `created_at` as well is safe: `_PUT` refreshes `created_at` when it
  replaces an expired row, so a live row's `created_at` is the time its question was stored. It also removes rows that
  were written with the earlier TTL of 30 days.
- **`infra/answer_cache.py`.** `MAX_TTL_DAYS = 29`, with the comment pointing at the housekeeping boundary.
- **`settings.py`.** `answer_cache_ttl_days: int = Field(default=29, gt=0, le=29)`.
- **No migration.** `0001` is not edited (its comment is documentation); the column and its index are unchanged.
- **Stop before the merge.** A merge to `main` deploys the backend. If `ANSWER_CACHE_TTL_DAYS=30` was ever set by hand
  in the Vercel project, the new `le=29` would stop the function from starting. The runbook never sets the variable;
  the Agent asks the Author in the chat to confirm it is absent and merges only after a "yes".

**Acceptance criteria**
- [ ] `tests/integration/test_housekeeping.py` (red first): fixtures at the boundary, 29 days minus one hour stays and
      29 days plus one hour loses its text; a cache row that is 29 days plus one hour old, with `expires_at` in the
      future, is deleted; a younger one stays.
- [ ] `tests/unit/test_settings.py`: `answer_cache_ttl_days=30` is rejected.
- [ ] A unit test pins `Settings` maximum == `MAX_TTL_DAYS` == `USER_TEXT_RETENTION_DAYS`, and
      `USER_TEXT_MAX_AGE_DAYS == 30`.
- [ ] `tests/unit/test_housekeeping_docs.py` keeps DB §9 identical to the module (statement and the `text_days` row).
- [ ] The UI text "at most 30 days" is unchanged, and still true.

**Verify:** standard checks; `uv run grounded housekeeping --dry-run` on the local dev database prints the counts.

**Eval impact:** none: the TTL is not part of the cache key, and no answer changes.

**Docs to update:** DB §4 (the `answer_cache` comment) and §9 (the SQL, the parameter table, the comments); Tech §4
(`ANSWER_CACHE_TTL_DAYS` 29), §11 (cache table and the TTL sentence); `.env.example`; README Privacy (deletion after
29 days, and a late or disabled cron can lengthen it); AGENTS if a command line mentions the period.
