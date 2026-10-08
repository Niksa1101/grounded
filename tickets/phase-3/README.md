# Phase 3 — `/ask` with structured output

**Goal:** the core product behavior works end to end, without UI: a question goes in, and out comes a validated
`AskResponse` with mapped citations and server-side confidence (PRD §9, Phase 3).

**Exit criteria (checked in 3.14)**
- All golden-set questions return a schema-valid `AskResponse` locally.
- Tests with `FakeLLMProvider` cover: invalid JSON → retry, invalid citation removal, zero valid citations → retry,
  refusal path, cache hit.
- Every request writes a `request_logs` row with stage latencies and shadow cost.

**Phase note:** there is no provider router until Phase 7. In this phase the pipeline calls **one** provider (the first
entry of `GENERATOR_PROVIDERS`), and the "one retry with feedback" of Tech §9.5 lives in the pipeline's generation step.
Ticket 7.04 moves it into the router. Keep that step small and isolated so the move is easy.

## Tickets (in order)

- [3.01 — Decision: Gemini generator model, structured output, thinking budget, prices](3.01-decision-gemini.md)
- [3.02 — `LLMProvider` protocol, `FakeLLMProvider` and `LLMAnswer`](3.02-provider-contract.md)
- [3.03 — `answer_v1` prompt and the versioned prompt loader](3.03-prompts.md)
- [3.04 — Context builder (`c1..cK` source blocks)](3.04-context.md)
- [3.05 — Tracer bullet: `POST /v1/ask` end to end with the fake provider](3.05-ask-tracer.md)
- [3.06 — Citation validation, mapping and marker rewriting](3.06-citations.md)
- [3.07 — Output validation with one retry and error feedback](3.07-validation-retry.md)
- [3.08 — `GeminiProvider` adapter](3.08-gemini-provider.md)
- [3.09 — Confidence heuristic — spec](3.09-confidence-spec.md)
- [3.10 — Confidence heuristic — implementation and wiring](3.10-confidence.md)
- [3.11 — Refusal handling and `no_rag` mode](3.11-refusal-no-rag.md)
- [3.12 — Request logging, stage timing and shadow cost](3.12-request-logs.md)
- [3.13 — Answer cache (Postgres)](3.13-answer-cache.md)
- [3.14 — Phase 3 closeout: the golden set through `/ask`](3.14-closeout.md)

## Review follow-ups

The code review of Phase 3 (PRs #40–#55, 2026-10-07) left 1 🔴, 6 🟡 and several 🟢 items. The 🔴 breaks an
architecture invariant (the query embedder sleeps and retries in the request path), so the follow-ups run as small
PRs **before 4.01**. That is also the cheap moment for the quality-affecting ones: there is no generation baseline
until 4.09. Decisions of 2026-10-07:

- #3: the answer-cache key gets a `generation_config_hash` (provider, temperature, max output tokens, thinking level)
  (PRD D47, R.11).
- #4: a `Settings` validator checks the worst-case bound of confidence invariant 4; the Agent writes it, the Author
  reviews (R.12).
- #5: the retry feedback is compact (no input values, no Pydantic links) and says so plainly when the answer was cut
  off at the token limit; the limit itself does not change (R.13).
- #7: links and URLs in `answer_markdown` are removed and counted in the stdout summary line, no migration (R.14).
- #9 (lowercase normalization of the cache key) and #10 (a client disconnect writes no row) are open items in PRD §12.

The Author runs the real-provider golden set once after R.14 (~35 calls of quota); its summary goes into the R.14 PR
with `n=30`, compared with the 2026-10-07 run, as information and not a baseline.

- [R.10 — Request-path query embedding fails fast](R.10-query-embed-fail-fast.md)
- [R.11 — Answer cache: replace a stale row, key on the generation parameters](R.11-answer-cache-fixes.md)
- [R.12 — Settings guard for confidence invariant 4](R.12-confidence-weight-guard.md)
- [R.13 — Compact retry feedback and the tokens of failed attempts](R.13-retry-feedback.md)
- [R.14 — URLs in the answer, and the question escaped in the prompt](R.14-answer-text-safety.md)
- [R.15 — Small hardening and documentation drift](R.15-review-polish.md)
- [R.16 — Fixes from the review of R.10–R.15](R.16-review-follow-up-fixes.md)
- [R.17 — Reference links, link leftovers and doc wording](R.17-reference-links.md)

Rules for working a ticket: [../README.md](../README.md).
