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

Rules for working a ticket: [../README.md](../README.md).
