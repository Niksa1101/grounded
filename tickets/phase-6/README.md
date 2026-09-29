# Phase 6 — Re-ranking with measured lift

**Goal:** optional Cohere re-ranking, behind a flag, that degrades gracefully and whose lift is measured and reported
honestly (PRD §9, Phase 6; D11).

**Exit criteria (checked in 6.05)**
- `hybrid_rerank` rows in both baselines.
- Lift reported honestly (including "no lift" if so).
- Degradation path tested.

**Quota note:** the Cohere trial is ~1k calls/month (verify in 6.01). One retrieval eval run is ~25 rerank calls, so CI
must run from the rerank cache. The rerank cache must be warm before `hybrid_rerank` joins any CI job.

## Tickets (in order)

- [6.01 — Decision: Cohere rerank model, trial limits, cap, price](6.01-decision-cohere.md)
- [6.02 — Cohere rerank adapter and rerank cache](6.02-rerank-adapter.md)
- [6.03 — Rerank in the request path with graceful degradation](6.03-rerank-path.md)
- [6.04 — `hybrid_rerank` in both evals and baseline rows](6.04-eval-rerank.md)
- [6.05 — Lift analysis and Phase 6 closeout](6.05-lift-analysis.md)

Rules for working a ticket: [../README.md](../README.md).
