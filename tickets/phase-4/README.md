# Phase 4 — Generation eval + CI quality gate

**Goal:** the eval harness that justifies the project. Answer quality is measured with promptfoo and a judge on a
different provider, and a regression blocks the PR (PRD §9, Phase 4).

**Exit criteria (checked in 4.12)**
- Full eval runs locally and in CI. Baseline committed to `eval/baselines/generation.json` (rows `no_rag`, `hybrid`).
- A deliberately degraded prompt is blocked by the gate (demonstrated).
- A quota-limited run ends as `inconclusive` (demonstrated, or simulated with the fake provider).
- The judge agreement number is recorded.

**Quota note for the whole phase:** one full run is ~30 questions × 2 configs of generation calls, plus judge calls
(one per claim for faithfulness, one per answer for correctness). Concurrency is always 1 (`-j 1`). The eval LLM
cache (4.03) makes identical re-runs free. Check the provider's remaining daily quota before a full run, and never loop
on a quota-limited API (AGENTS.md §6.15).

## Tickets (in order)

- [4.01 — Decision: Groq and judge models, promptfoo version and test loading](4.01-decision-groq-promptfoo.md)
- [4.02 — `GroqProvider` adapter](4.02-groq-provider.md)
- [4.03 — Eval mode (`APP_ENV=eval`) and the eval LLM cache](4.03-eval-mode.md)
- [4.04 — Judge rubrics and the judge module](4.04-judge.md)
- [4.05 — Tracer bullet: promptfoo harness with deterministic assertions](4.05-promptfoo-tracer.md)
- [4.06 — Judge-based assertions: faithfulness and correctness](4.06-judge-asserts.md)
- [4.07 — Generation gate and inconclusive rule — spec](4.07-gate-generation-spec.md)
- [4.08 — Generation gate and inconclusive rule — implementation](4.08-gate-generation.md)
- [4.09 — First generation baseline and the eval report](4.09-generation-baseline.md)
- [4.10 — `eval.yml` workflow with the PR comment](4.10-eval-workflow.md)
- [4.11 — Judge–human agreement](4.11-judge-agreement.md)
- [4.12 — Gate demonstrations and Phase 4 closeout](4.12-closeout.md)

Rules for working a ticket: [../README.md](../README.md).
