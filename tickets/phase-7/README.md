# Phase 7 — Multi-provider fallback + circuit breaker

**Goal:** when Gemini is rate-limited or down, Groq answers. A circuit breaker stops the service from hammering a
failing provider (PRD §9, Phase 7; D12, D30).

**Exit criteria (checked in 7.04)**
- A simulated 429 on Gemini → Groq answers, and the breaker skips Gemini until reset.
- Fallback rate visible in logs.

## Tickets (in order)

- [7.01 — Decision: breaker state on serverless, quota reset, breaker timings](7.01-decision-breaker.md)
- [7.02 — Provider router — spec](7.02-router-spec.md)
- [7.03 — Provider router — implementation](7.03-router.md)
- [7.04 — Router in the pipeline, fallback metrics, demo and Phase 7 closeout](7.04-router-wiring.md)

Rules for working a ticket: [../README.md](../README.md).
