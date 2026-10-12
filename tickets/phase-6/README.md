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

Rules for working a ticket: [../README.md](../README.md).
