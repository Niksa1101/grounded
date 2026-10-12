# Phase 5 — UI, protection, deploy (MVP)

**Goal:** a public, protected, working demo with an honest README. Finishing this phase completes the MVP (PRD §9, Phase 5).

**Exit criteria (MVP definition of done, checked in 5.12)**
- The live URL answers a question after the backend was idle (cold start measured and documented).
- Rate limit and budget verified manually.
- README numbers are copied from committed eval results, never typed by hand.
- All CI workflows green on `main`.

**Status of the exit criteria (5.12, 2026-10-12):** README numbers, pass; CI on `main`, pass for the three workflows that have run; live URL with a measured cold start and the manual rate-limit and budget check, pending the Author's deployment. The MVP is not declared complete. Evidence and the Author's manual procedure: [docs/runbooks/mvp-closeout.md](../../docs/runbooks/mvp-closeout.md).

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

## Review follow-ups

The code review of Phase 5 (PRs #89–#101, 2026-10-12) found no blocking bug in merged code. It found four issues to
fix **before the Author runs the production runbooks**, plus small items: the Neon runbook sets the `app` password
as a client-side hash, which Neon refuses; the public `/readyz` queries Neon on every call, so a 4-minute ping would
use about 180 of the 100 CU-hours a month; `measure-cold-start` sends the same questions in both runs, so the second
run cannot judge its cold ask; and `test_deadline.py` depends on wall-clock speed. Decisions of 2026-10-12 (PRD D54):

- a. A new `Settings` field `RETRIEVAL_TIMEOUT_S` (default 2.0, off in eval mode) bounds the retrieval stage; a
  timeout answers 504 `deadline_exceeded` (R.21).
- b. Migration `0004` adds the outcome `deadline_exceeded`, used by deadline and retrieval-timeout rows (R.20).
- c. `/readyz` requires the proxy secret; `/healthz` stays public; keep-warm and `measure-cold-start` send it (R.22).
- d. The Gemini adapter stays eager (built in the lifespan): it supersedes "providers created lazily" of 5.08.
- e. Requests without a client IP share one rate-limit bucket, `hash_ip("unknown", key)` (R.25).
- f. One PR per ticket on `phase-5/<topic>`; the Agent squash-merges after green CI. Secrets, GitHub settings and
  production stay the Author's.

- R.18 — Review follow-ups: tickets and decisions (D54) (this section)
- [R.19 — Neon runbook: set the `app` password in plain text from a psql variable](R.19-neon-role-password.md)
- [R.20 — Migration `0004`: the `deadline_exceeded` outcome](R.20-deadline-outcome.md)
- [R.21 — `RETRIEVAL_TIMEOUT_S`: the retrieval stage timeout](R.21-retrieval-timeout.md)
- [R.22 — `/readyz` requires the proxy secret](R.22-readyz-secret.md)
- [R.23 — Cold-start measurement: distinct questions, cold cache hits, cold retrieval](R.23-cold-start-questions.md)
- [R.24 — Deterministic deadline tests and env isolation in tests](R.24-deterministic-deadline-tests.md)
- [R.25 — Small hardening and documentation drift](R.25-review-polish.md)

After the series, the Author's steps: create the repository secret `PROXY_SHARED_SECRET` (the Vercel value) before
the first keep-warm run; run the Neon runbook (it now applies `0004` and sets the password the new way), then the
Vercel runbook including V17, then the MVP closeout checks; set `HOUSEKEEPING_ENABLED=true` within 30 days of the
first real question.

Rules for working a ticket: [../README.md](../README.md).
