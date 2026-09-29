# Phase 5 — UI, protection, deploy (MVP)

**Goal:** a public, protected, working demo with an honest README. Finishing this phase completes the MVP (PRD §9, Phase 5).

**Exit criteria (MVP definition of done, checked in 5.12)**
- The live URL answers a question after the backend was idle (cold start measured and documented).
- Rate limit and budget verified manually.
- README numbers are copied from committed eval results, never typed by hand.
- All CI workflows green on `main`.

**Production note for the whole phase:** anything touching Neon production, Vercel production, secrets or production
workflows needs the Author's explicit go-ahead **each time** (AGENTS.md §5, §11). The Agent writes workflows and
runbooks, and the Author runs or approves them.

## Tickets (in order)

- [5.01 — Decision: rate limiter on serverless, limit and budget numbers](5.01-decision-rate-limit.md)
- [5.02 — Decision: deployment parameters (region, durations, timeouts)](5.02-decision-deploy.md)
- [5.03 — Proxy secret, trusted client IP and IP hashing](5.03-proxy-auth.md)
- [5.04 — Input limits and the rate limiter](5.04-rate-limit.md)
- [5.05 — Global daily LLM budget](5.05-budget.md)
- [5.06 — Tracer bullet: frontend proxy, generated types, minimal ask page](5.06-frontend-tracer.md)
- [5.07 — Complete ask page](5.07-ask-page.md)
- [5.08 — Backend Vercel configuration](5.08-backend-vercel.md)
- [5.09 — Neon production: migrate, `app` role, `ingest.yml`](5.09-neon-prod.md)
- [5.10 — Vercel projects, env vars, first deploy, cold start](5.10-deploy.md)
- [5.11 — `housekeeping.yml`: retention and daily keep-warm ping](5.11-housekeeping.md)
- [5.12 — README v1 and MVP closeout](5.12-readme-v1.md)

Rules for working a ticket: [../README.md](../README.md).
