# R.23 — Cold-start measurement: distinct questions, cold cache hits, cold retrieval

> **Before you start:** read [AGENTS.md](../../AGENTS.md), the rules in [tickets/README.md](../README.md) (how to work a ticket, ticket types, [standard checks](../README.md#standard-checks)) and the phase notes in [README.md](README.md#review-follow-ups). Work only on this ticket.

> **Owner:** G · **Type:** Build · **Branch:** `phase-5/cold-start-questions` · **Blocked by:** R.22 (same file) ·
> **Builds on:** `infra/cold_start.py`, `cli.py:measure_cold_start`, `tests/unit/test_cold_start.py`

**What to build:** review item #3. The runbook's two runs send the same `DEFAULT_QUESTIONS`, so in run 2
(`--first ask`) every ask is an answer-cache hit; `_pick(..., hit=False)` drops the cold ask and the verdict is
"could not be judged". The tool also does not show the cold retrieval latency, which D54a needs checked against
`RETRIEVAL_TIMEOUT_S`.

**Read first:** Tech §18 (timeout chain, cold start); `docs/runbooks/vercel-deploy.md` step 9; PRD D53, D54.

**Scope notes**
- **Distinct questions per order.** With the default questions, `--first ask` starts at offset
  `len(DEFAULT_QUESTIONS) // 2`, so its asks are not answer-cache hits from a `--first health` run. The pick
  `plan.questions[asked % len(plan.questions)]` becomes `(offset + asked) % len`. A user-supplied `--question` list
  gets offset 0.
- **A cold cache hit is still a cold sample.** In ask order, `judge_margin` and the "ask end to end, cold" row accept
  the cold ask whether or not it hit the cache: end to end minus the server total does not depend on it. The table
  marks it "(cache hit)".
- **Cold retrieval.** Read `meta.latency_ms.retrieval` into a new `Sample.retrieval_ms`. Print the cold ask's value
  with a line comparing it to `RETRIEVAL_TIMEOUT_S` (default 2.0 s). Within 25% of it: recommend raising the Vercel
  value.
- **A cold ask that failed with `deadline_exceeded`** says "the backend's deadline or retrieval timeout fired; look
  up the request_id in request_logs". It is not a margin verdict.
- **Docstrings.** In health order the Neon wake partly runs inside the handler's deadline (the first DB call is
  inside it), so that estimate is conservative.

**Acceptance criteria** (tests in `tests/unit/test_cold_start.py`, against the existing local stub server)
- [ ] The offset: `--first ask` with default questions starts at `len // 2`; a `--question` list starts at 0.
- [ ] A cold cache hit is judged and marked "(cache hit)".
- [ ] The retrieval line, including the "within 25%" recommendation.
- [ ] The `deadline_exceeded` message for a cold ask.

**Verify:** standard checks; `uv run grounded measure-cold-start --frontend-url http://127.0.0.1:3000 --backend-url
http://127.0.0.1:8000 --idle-minutes 0 --first ask --asks 2 --yes` against the local fake stack, with
`PROXY_SHARED_SECRET` exported, shows a cold row and a retrieval line.

**Eval impact:** none.

**Docs to update:** `docs/runbooks/vercel-deploy.md` (step 9 explains the offset; new checklist item V17: cold
`latency_ms.retrieval` against `RETRIEVAL_TIMEOUT_S`, recorded in Tech §18; a failure row for a cold 504
`deadline_exceeded`).
