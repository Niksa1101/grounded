# R.20 — Migration `0004`: the `deadline_exceeded` outcome

> **Before you start:** read [AGENTS.md](../../AGENTS.md), the rules in [tickets/README.md](../README.md) (how to work a ticket, ticket types, [standard checks](../README.md#standard-checks)) and the phase notes in [README.md](README.md#review-follow-ups). Work only on this ticket.

> **Owner:** G · **Type:** Build · **Branch:** `phase-5/deadline-outcome` · **Blocked by:** R.19 ·
> **Builds on:** `backend/migrations/0001_init.sql`, `observability/request_log.py`, `api/errors.py`

**What to build:** decision D54b. A request that hits `REQUEST_DEADLINE_S` is logged today with outcome
`internal_error` (`error_code = deadline_exceeded`, `http_status = 504`), because the `request_logs.outcome` CHECK has
no deadline value. Migration `0004` adds `'deadline_exceeded'`. It ships before the first production
`grounded migrate`, so no backfill is needed. Deadline rows and retrieval-timeout rows (R.21) both use it. The
schema change is approved by D54b; show the DDL in the PR.

**Read first:** DB §4 (`request_logs`), §10; Tech §6, §13; PRD D53, D54.

**Scope notes**
- First confirm the constraint name on the local DB:
  `SELECT conname FROM pg_constraint WHERE conrelid = 'request_logs'::regclass AND contype = 'c';`
  (expected: `request_logs_outcome_check`).
- `backend/migrations/0004_deadline_outcome.sql` has two statements: `ALTER TABLE request_logs DROP CONSTRAINT
  request_logs_outcome_check;` and `ALTER TABLE request_logs ADD CONSTRAINT request_logs_outcome_check CHECK (outcome
  IN (…the 9 values of 0001…, 'deadline_exceeded'));`.
- `observability/request_log.py`: add `"deadline_exceeded"` to `Outcome`.
- `api/errors.py`: `_deadline_exceeded` no longer passes `outcome="internal_error"`. Remove the now-unused `outcome`
  parameter of `_fail` (R.21 does not need it). The comment says: every `ErrorCode` except `unauthorized` is an
  outcome, and `unauthorized` never has a trace.

**Acceptance criteria**
- [x] `0004_deadline_outcome.sql` applies on the dev DB (`uv run grounded migrate`) and on a fresh test DB (0001–0004).
- [x] `tests/integration/test_deadline.py` expects outcome `deadline_exceeded`.
- [x] The test that pins the CHECK values adds the new value.
- [x] A row with `deadline_exceeded` inserts; an unknown outcome is still refused.

**Verify:** standard checks; `uv run grounded migrate` on the dev DB.

**Eval impact:** none.

**Docs to update:** DB §4 (CHECK list, the migration note) and §10 (migration list); wherever `docs/DB.md` or
`tickets/phase-8/` defines an error rate or an outcome list next to `internal_error`, add `deadline_exceeded`;
Tech §13 (log convention for 504); PRD §12 (item 2 of "Open after 5.08" resolved).
