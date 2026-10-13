# R.31 — Phase 6 review follow-ups

> **Before you start:** read [AGENTS.md](../../AGENTS.md), the rules in [tickets/README.md](../README.md) (how to work a ticket, ticket types, [standard checks](../README.md#standard-checks)) and the phase notes in [README.md](README.md#review-follow-ups-2026-10-13). Work only on this ticket.

> **Owner:** G · **Type:** Build · **Branch:** `phase-6/review-followups` · **Blocked by:** — ·
> **Builds on:** `generation/pipeline.py` (`_rerank`), `docs/runbooks/rerank-eval.md`, `settings.py`
> (`EVAL_RERANK_MIN_INTERVAL_S`)

**What to build:** the findings of the Phase 6 code review (#112–#116), before the Author follows the runbook for
6.04b. The review found no blocker. It found two places where the runbook promises what the tools do not show
(the HTTP status of a rejected rerank; a baseline PR that stays red until the README section is filled), pacing
with no margin, no source for the call tally of the generation run, a missing note on what the live latency
covers, and a stale docstring. Decisions of the Author (2026-10-13): the `EVAL_RERANK_MIN_INTERVAL_S` default
becomes 7.0; only `status_code` of a `rejected` call goes to the log; one ticket, one PR.

**Read first:** Tech §6 (the rerank stage), §8 (eval pacing); the runbook; `tests/unit/test_rerank_readme.py`.

**Scope notes**
- **1. `status_code` in the degradation WARNING.** `_Degraded` carries `status_code` (the HTTP status of a
  `ProviderRequestRejected`, else `None`), and the WARNING `rerank degraded, using the RRF order` logs it. A bare
  number, never the provider's text (AGENTS §6.13). Runbook step 3: 401 is the key; 403 or 404 means
  `RERANK_MODEL=rerank-v4.0-fast`.
- **2. The baseline PR is red until the README is filled.** Runbook step 7 says so and lists what goes into the
  same PR: the empty `rerank-lift` block, the line "Lift: not yet measured", the rerank latency cells of "Cost and
  latency". The README comment and PRD §12 ("Phase 6 closeout", items 5 and 6) say "same PR / same branch". The
  `rerank-lift: NOT FILLED YET` marker is unchanged (the test looks for it).
- **3. The tally of the generation run.** promptfoo prints no Cohere count, so runbook step 6.1 gets a read-only
  count of the rerank cache (`mode=ro`: it never creates the file), run before and after 6.2 and 6.3. The
  difference is the successful live calls; each degraded case is one more call.
- **4. Pacing default 7.0 s.** Under 10 a minute with a margin for jitter (6 s between starts has none); 24 live
  calls take about 3 minutes. The value is in no hash and no baseline. The 429 row of the runbook suggests 9.
  Ticket 6.04 stays as the record of that day.
- **5. Live latency covers the run that wrote the file.** Runbook step 4: copy the file only after a run that
  finished; after a stopped and resumed run, p50 and p95 cover the `N live calls` of the last run and go into the
  README with that `n`.
- **6. Docstring.** `promptfoo_provider.py`: `retrieved_section_ids` is the RRF order, before any rerank; the
  reranked order is the one of `context`.
- **Drift found while planning.** `pipeline.py`: a degraded answer is not cached because a hit would serve it for
  the answer cache's TTL (`ANSWER_CACHE_TTL_DAYS`), not "up to 30 days" (R.27 set 29).
- **7. Parallel pytest runs:** already fixed by R.30 (#121), nothing to do.
- **8. PR size:** no code. #115 and #116 had about 610 lines of non-test code each; from Phase 7 a ticket is split
  once its non-test code passes about 400 lines (AGENTS §11).
- **Out of scope:** no `request_logs` column (a schema change), no `AskResponse`/`Meta` change, no Author-owned
  module, baseline, golden set or prompt, no real API call.

**Acceptance criteria**
- [x] An integration test: a `ProviderRequestRejected(..., status_code=404)` logs `rerank_error` `rejected` and
      `status_code` 404, and neither the question nor the provider's text appears in the log; the existing
      degradation test checks `status_code is None` for `unavailable`.
- [x] A unit test extracts the count command's SQL and path from the runbook, runs the SQL with `mode=ro` on a
      `KVCache` holding 2 entries (2), checks that the file name is `RERANK_CACHE_FILE` and that a missing file is
      not created.
- [x] The default is 7.0 in `Settings`, `.env.example`, Tech §4 and §8, and the runbook; no "6 seconds",
      "2.5 minutes" or `EVAL_RERANK_MIN_INTERVAL_S=8` is left outside ticket 6.04.

**Verify:** standard checks; the count command by hand on a temporary cache and on a folder without one.

**Eval impact:** none: a log field, an eval-only pacing default and docs; no retrieval, prompt, schema or model
change.

**Docs to update:** Tech §4 (the config table), §6 (the WARNING), §8 (pacing); PRD §12 (Phase 6 closeout); the
runbook; the README rerank-lift comment.
