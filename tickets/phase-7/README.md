# Phase 7 — Multi-provider fallback + circuit breaker

**Goal:** when Gemini is rate-limited or down, Groq answers. A circuit breaker stops the service from hammering a
failing provider (PRD §9, Phase 7; D12, D30).

**Exit criteria (checked in 7.04)**
- A simulated 429 on Gemini → Groq answers, and the breaker skips Gemini until reset.
- Fallback rate visible in logs.

**Status (2026-10-15): done.** 7.01 and 7.02 were merged on 2026-10-14 (#124, #125), 7.03 and 7.04a on 2026-10-15 (#126, #127) and 7.04b
(the local demo and the closeout, #128) the same day. Both exit criteria are met by tests with fake providers (the table
is in [PRD §9, Phase 7](../../docs/PRD.md#phase-7--multi-provider-fallback--circuit-breaker)); no real call to Groq or a
real Gemini 429 was made in the phase. What stays open (the Author's review of the 7.02 cases, the Groq-first decision,
real recordings, the Vercel Groq variables, a docstring of `router.py`) is "Phase 7 closed" in
[PRD §12](../../docs/PRD.md#12-assumptions-and-open-items).

## Tickets (in order)

- [7.01 — Decision: breaker state on serverless, quota reset, breaker timings](7.01-decision-breaker.md)
- [7.02 — Provider router — spec](7.02-router-spec.md) (its open criterion, the Author's review of the cases, has a
  [review sheet](7.02-case-review.md))
- [7.03 — Provider router — implementation](7.03-router.md)
- [7.04 — Router in the pipeline, fallback metrics, demo and Phase 7 closeout](7.04-router-wiring.md) (done: #127, #128)

Rules for working a ticket: [../README.md](../README.md).
