# Phase 6 — Re-ranking with measured lift

**Goal:** optional Cohere re-ranking, behind a flag, that degrades gracefully and whose lift is measured and reported
honestly (PRD §9, Phase 6; D11).

**Exit criteria (checked in 6.05)**
- `hybrid_rerank` rows in both baselines.
- Lift reported honestly (including "no lift" if so).
- Degradation path tested.

**Quota note (verified in 6.01, 2026-10-12, Tech §8, PRD D55):** the Cohere Trial key allows 1,000 API calls a month and
10 Rerank requests a minute, and may not be used in production. One pass of the 30 golden questions is at most 30 rerank
calls (about 25 for the retrieval eval alone), and the Author set 150 real calls for the whole phase. `hybrid_rerank`
therefore runs **by hand only**: it joins neither `ci.yml` nor `eval.yml`, no workflow needs `COHERE_API_KEY`, and its rows
in both baselines carry no thresholds (ungated). The rerank cache is a local convenience for those manual runs, not a CI
requirement. The deployed demo keeps `RERANK_PROVIDER=none`.

**Decision of 6.01 in one line:** `rerank-v4.0-pro` (fallback `rerank-v4.0-fast`), `RERANK_DAILY_CAP=30`,
`RERANK_TIMEOUT_S=3.0`, the `cohere` 7.2.0 SDK (`AsyncClientV2`, `max_retries=0`), $2.50 per 1K searches.

## Tickets (in order)

- [6.01 — Decision: Cohere rerank model, trial limits, cap, price](6.01-decision-cohere.md) (done; PRD D55)
- [6.02 — Cohere rerank adapter and rerank cache](6.02-rerank-adapter.md)
- [6.03 — Rerank in the request path with graceful degradation](6.03-rerank-path.md)
- [6.04 — `hybrid_rerank` in both evals and baseline rows](6.04-eval-rerank.md)
- [6.05 — Lift analysis and Phase 6 closeout](6.05-lift-analysis.md)

**Status (2026-10-14): done.** 6.01 to 6.04a and the tooling of 6.05 were merged on 2026-10-12. The Author ran the real
evals of 6.04b by hand on 2026-10-13 (30 of the 150 Cohere calls), and the baseline PR added the `hybrid_rerank` rows, the
lift analysis in the README and the closeout check. All three exit criteria are met (PRD §9); what stays open is in PRD §12,
"Phase 6 closed".

## Review follow-ups (2026-10-13)

A code review of 6.01–6.05 (#112–#116) found no blocker, but found gaps that hit 6.04b: the runbook asked the
Author to read an HTTP status the log did not carry, and did not say that the baseline PR stays red until the README
section is filled. Also: pacing at exactly the Trial limit, no call tally for the generation run, no note on what
the live latency covers, and a stale docstring. Decisions of the Author (2026-10-13): `EVAL_RERANK_MIN_INTERVAL_S`
defaults to 7.0; only `status_code` of a `rejected` call is logged; one ticket, one PR. The finding on parallel
pytest runs was already fixed by R.30 (#121).

- [R.31 — Phase 6 review follow-ups](R.31-review-followups.md)

Rules for working a ticket: [../README.md](../README.md).
