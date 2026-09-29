# Phase 8 — Observability dashboard, cost, calibration

**Goal:** a public, aggregates-only dashboard with latency, cost and quality, plus a calibration check of the
confidence heuristic (PRD §9, Phase 8; FR-27, D26, D38).

**Exit criteria (checked in 8.05)**
- The public dashboard is live.
- p50/p95 per stage and shadow cost per 1k questions are shown.
- The calibration table is in the README.

## Tickets (in order)

- [8.01 — Dashboard views migration](8.01-dashboard-views.md)
- [8.02 — `GET /v1/metrics/summary`](8.02-metrics-api.md)
- [8.03 — Next.js `/metrics` page and `/api/metrics` proxy](8.03-metrics-page.md)
- [8.04 — Confidence calibration report](8.04-calibration.md)
- [8.05 — Cost and latency in README, Phase 8 closeout](8.05-closeout.md)

Rules for working a ticket: [../README.md](../README.md).
