# Tickets — Phase 2 → Phase 9

Delivery tickets that break [docs/PRD.md §9](../docs/PRD.md) into small, ordered, verifiable units of work.
Each ticket is **one file** and **one PR** on its own branch. Hand an agent a single ticket file, e.g.
"Do `tickets/phase-2/2.01-retrieval-config.md`". Phases 0 and 1 are not covered here: Phase 0 is done and Phase 1
is closed outside these tickets (see [Preconditions](#preconditions)).

**Sources of truth win over the tickets.** The order is [AGENTS.md](../AGENTS.md) → [PRD](../docs/PRD.md) →
[Tech](../docs/Tech.md) → [DB](../docs/DB.md). If a ticket contradicts them, stop, tell the Author and fix the ticket
in the same PR. A ticket never overrides a documented decision.

References like "Tech §9.5" point to sections of those documents. Module paths follow the layout in Tech §3.

## Layout

```
tickets/
├── README.md              ← this file: rules, ticket types, checks, preconditions, index
└── phase-N/
    ├── README.md          ← phase goal, exit criteria, phase-wide notes, ticket list
    └── N.NN-<topic>.md    ← one ticket (the <topic> is also its branch name: phase-N/<topic>)
```

## Phases

- [Phase 2 — Hybrid retrieval](phase-2/README.md)
- [Phase 3 — `/ask` with structured output](phase-3/README.md)
- [Phase 4 — Generation eval + CI quality gate](phase-4/README.md)
- [Phase 5 — UI, protection, deploy (MVP)](phase-5/README.md)
- [Phase 6 — Re-ranking with measured lift](phase-6/README.md)
- [Phase 7 — Multi-provider fallback + circuit breaker](phase-7/README.md)
- [Phase 8 — Observability dashboard, cost, calibration](phase-8/README.md)
- [Phase 9 — Hardening and polish](phase-9/README.md)

---

## How to work a ticket

1. **Pick the frontier.** Tickets run **strictly in order**, one at a time. The frontier is the first ticket in the
   [index](#ticket-index) that isn't `done`. Don't skip ahead, and don't start a ticket owned by the Author.
2. **Read first.** Read AGENTS.md, this README, the phase README, then every section in the ticket's
   *Read first* list, then the code named in
   *Builds on*. Tickets list the facts that are easy to get wrong. Contracts live in the docs, and the docs win.
3. **Branch.** Use the branch named in the ticket (`phase-<N>/<topic>`), cut from an up-to-date `main`.
4. **Build.** Stay inside the ticket's scope. Put any idea outside it in PRD §12 "Assumptions and open items",
   not in the code. **Size:** a PR should be readable in one sitting, at most about 400 added lines of non-test
   code. A larger ticket is split into stacked PRs, each green on its own.
5. **Check.** Run the [standard checks](#standard-checks) and the ticket's *Verify* steps. Report the actual output.
6. **PR.** Use the AGENTS.md §11 template: **What / Why / How tested / Eval impact / Docs updated**. Add the
   `run-eval` label when the ticket says *Eval impact: yes*.
7. **Close.** In the same PR, set the ticket's status in the [index](#ticket-index) to `done` and add the PR
   link. Tick its acceptance criteria in the ticket file. After the merge, the next ticket becomes the frontier.
8. **Explain.** For code the Agent wrote, the PR description or the final message explains the non-obvious parts
   (AGENTS.md §3). The Author must be able to defend every line.

**Status values:** `todo` · `in-progress` · `blocked` (waiting on something outside these tickets, with the reason
next to it) · `done`.

## Ticket types

| Type | Who writes the code | Standard rules (apply to every ticket of this type) |
|---|---|---|
| **Build** | Agent (G) | Normal boilerplate work (AGENTS.md §3 "Agent-owned"). Ships with tests in the same PR. |
| **Spec** | Agent (G) | Prepares an **Author-owned module**. It creates the file with types, signatures, a contract docstring and `raise NotImplementedError`, plus the **spec tests** (failing until the Author implements), and wires the module where it is called if that can land green. It never contains the implementation, and the tests must not give it away either (no oracle that *is* the solution). Expected failures are marked `xfail(strict=True)` with a reason that names the Author ticket, so CI stays green and the marker fails loudly once the code passes. |
| **Author** | Author (A) | The Author writes the implementation. The Agent may explain concepts, give hints of increasing detail when asked (concept → approach → pseudocode), and review thoroughly (correctness, edge cases, performance, readability). The Agent **must not** write the implementation unless the Author says "write it" for this module (then explain it line by line afterwards). The Author removes the `xfail` markers. |
| **Decision** | Author decides, Agent prepares | The Agent researches and writes a short options memo in the PR description: the question, 2–4 options, trade-offs, facts **verified today from official docs with links and the date**, and a recommendation. The Author decides. The PR records the decision in PRD §11 (a new or amended `D` row with the date), updates Tech/DB/`.env.example` where affected, and changes no behavior code. Unverifiable facts are written as "unverified", never guessed. |
| **Baseline** | Agent runs, Author approves | Changes a file under `eval/baselines/`. The PR title is `eval: update baseline (<reason>)`, the description has a before/after table with `n`, and the numbers come only from a committed-tool run (`--write-baseline`), never typed. It needs the Author's explicit approval (AGENTS.md §5, §7). |
| **Closeout** | Agent (G) + Author | The last ticket of a phase. It checks every exit criterion one by one (pass/fail with evidence), updates the README roadmap and the PRD status line, and lists anything deferred. A failed criterion is reported, never hidden. |

## Standard checks

Run from `backend/` unless noted. Integration tests need Docker running (`docker compose -f infra/docker-compose.yml up -d db`).

```bash
uv run ruff check . && uv run ruff format --check . && uv run pyright
```

```bash
uv run pytest
```

Frontend tickets also run (from `frontend/`):

```bash
npm run lint && npm run typecheck && npm run build
```

## Preconditions

Phase 2 starts only after **Phase 1 is closed** (handled outside these tickets):

- The index is built from the pinned tag and active. Counts and token stats are recorded in `index_versions`.
- The `dense` row is committed to `eval/baselines/retrieval.json` through its own baseline PR.
- `golden_set.v1.jsonl` has 30 reviewed items and `grounded golden validate … --against-index` passes.
- README roadmap and the PRD status line say Phase 1 is done.

Phase 1 is closed, but the code review of Phases 0 and 1 left follow-ups (R.00–R.08, B.01 in the index). Ticket 2.01
is `blocked (R.07)`: the chunker fix changes the index, and the baseline is regenerated once, in B.01.

## Cross-cutting rules (short form)

These come from AGENTS.md §6–§9 and apply to every ticket. They are repeated here because they are the ones most
often broken:

- **No real network calls in pytest.** Use `FakeLLMProvider`, fake embedder/reranker and recorded JSON fixtures.
  The socket-blocking fixture enforces this. Integration tests use the local pgvector container, never Neon.
- **Config, not literals.** Every K, threshold, weight, limit, timeout and model ID is a `Settings` field. Add it
  to `.env.example` (the sync test checks it). Only `grounded.settings` reads the environment.
- **Model IDs, limits and prices are verified at implementation time** from provider docs, with a date. Never from
  memory, never a floating alias.
- **SQL:** bound parameters only (vectors included), and a deterministic `ORDER BY` with a tie-breaker on every
  ranked query. Changes go in a new numbered migration, never an edit to an applied one.
- **Async all the way** in the request path. No blocking I/O inside `async def`, no sleeping in the request path.
- **Privacy:** no raw IPs anywhere, no question text in stdout logs. Public surfaces show aggregates only.
- **Stop and ask** before: a new dependency or external service, a schema/migration, a change to
  `LLMAnswer`/`AskResponse`/the HTTP API, a golden set, threshold or baseline edit, a prompt wording change,
  secrets, production, deletions, force-push (AGENTS.md §5).
- **Docs in the same PR** when behavior, schema, config or commands change: PRD/Tech/DB/README/AGENTS (including
  the AGENTS.md §10 command list).
- **Eval discipline:** numbers come from committed eval output, reported with `n`. An `inconclusive` result is not a pass.

---

## Ticket index

Owner: **G** = Agent, **A** = Author, **A+G** = together, **D** = Author decides (Agent prepares).

| # | Ticket | Type | Owner | Status | PR |
|---|---|---|---|---|---|
| **2** | **[Hybrid retrieval](phase-2/README.md)** | | | | |
| **R** | **Review follow-ups** (before 2.01; [details](phase-2/README.md#review-follow-ups)) | | | | |
| [R.00](phase-2/README.md#review-follow-ups) | Tickets in the repo, review process guideline (#15) | Build | G | in-progress | |
| [R.01](phase-2/R.01-settings-defaults.md) | Defaults for the corpus tag and the embedding model | Build | G | todo | |
| [R.02](phase-2/R.02-eval-tooling-fixes.md) | Eval tooling fixes: dirty-tree check, lazy embedder | Build | G | todo | |
| [R.03](phase-2/R.03-baseline-integrity.md) | Baseline integrity: golden-set hash, refuse mixing | Build | G | todo | |
| [R.04](phase-2/R.04-ingest-guards.md) | Ingest guards and documentation drift | Build | G | todo | |
| [R.05](phase-2/R.05-index-integrity-migration.md) | Migration `0002`: index integrity | Build | G | todo | |
| [R.06](phase-2/R.06-ci-hardening.md) | CI hardening: pinned actions, one tokenizer source | Build | G | todo | |
| [R.07](phase-2/R.07-chunker-keep-with-next.md) | Chunker: keep a heading with the block after it | Build | G (delegated) | todo | |
| [R.08](phase-2/R.08-golden-label-check.md) | Manual check of golden labels, explain-back | Author | A (+G worksheet) | todo | |
| [2.01](phase-2/2.01-retrieval-config.md) | `RetrievalConfig` and its hash | Build | G | blocked (R.07) | |
| [B.01](phase-2/B.01-baseline-refresh.md) | Baseline refresh: index v2, golden-set hash, config hash | Baseline | G runs, A approves | todo | |
| [2.02](phase-2/2.02-lexical-spec.md) | Lexical FTS search — spec | Spec | G | todo | |
| [2.03](phase-2/2.03-lexical.md) | Lexical FTS search — implementation | Author | A | todo | |
| [2.04](phase-2/2.04-eval-fts.md) | `fts` mode in the retrieval eval | Build | G | todo | |
| [2.05](phase-2/2.05-hybrid-spec.md) | Hybrid RRF search — spec | Spec | G | todo | |
| [2.06](phase-2/2.06-hybrid-rrf.md) | Hybrid RRF search — implementation | Author | A | todo | |
| [2.07](phase-2/2.07-ablation.md) | `hybrid` mode, ablation run and baseline rows | Build + Baseline | G | todo | |
| [2.08](phase-2/2.08-decision-ci-cache.md) | Decision: seeding the CI corpus and embedding caches | Decision | D | todo | |
| [2.09](phase-2/2.09-gate-spec.md) | Retrieval gate — spec | Spec | G | todo | |
| [2.10](phase-2/2.10-gate.md) | Retrieval gate — implementation | Author | A | todo | |
| [2.11](phase-2/2.11-ci-retrieval-eval.md) | CI `retrieval-eval` job | Build | G | todo | |
| [2.12](phase-2/2.12-closeout.md) | Gate demonstration and Phase 2 closeout | Closeout | A+G | todo | |
| **3** | **[`/ask` with structured output](phase-3/README.md)** | | | | |
| [3.01](phase-3/3.01-decision-gemini.md) | Decision: Gemini generator model, structured output, thinking budget, prices | Decision | D | todo | |
| [3.02](phase-3/3.02-provider-contract.md) | `LLMProvider` protocol, `FakeLLMProvider` and `LLMAnswer` | Build | G | todo | |
| [3.03](phase-3/3.03-prompts.md) | `answer_v1` prompt and the versioned prompt loader | Build | G | todo | |
| [3.04](phase-3/3.04-context.md) | Context builder (`c1..cK` source blocks) | Build | G | todo | |
| [3.05](phase-3/3.05-ask-tracer.md) | Tracer bullet: `POST /v1/ask` end to end with the fake provider | Build | G | todo | |
| [3.06](phase-3/3.06-citations.md) | Citation validation, mapping and marker rewriting | Build | G | todo | |
| [3.07](phase-3/3.07-validation-retry.md) | Output validation with one retry and error feedback | Build | G | todo | |
| [3.08](phase-3/3.08-gemini-provider.md) | `GeminiProvider` adapter | Build | G | todo | |
| [3.09](phase-3/3.09-confidence-spec.md) | Confidence heuristic — spec | Spec | G | todo | |
| [3.10](phase-3/3.10-confidence.md) | Confidence heuristic — implementation and wiring | Author | A (+G wiring) | todo | |
| [3.11](phase-3/3.11-refusal-no-rag.md) | Refusal handling and `no_rag` mode | Build | G | todo | |
| [3.12](phase-3/3.12-request-logs.md) | Request logging, stage timing and shadow cost | Build | G | todo | |
| [3.13](phase-3/3.13-answer-cache.md) | Answer cache (Postgres) | Build | G | todo | |
| [3.14](phase-3/3.14-closeout.md) | Phase 3 closeout: the golden set through `/ask` | Closeout | A+G | todo | |
| **4** | **[Generation eval + CI quality gate](phase-4/README.md)** | | | | |
| [4.01](phase-4/4.01-decision-groq-promptfoo.md) | Decision: Groq and judge models, promptfoo version and test loading | Decision | D | todo | |
| [4.02](phase-4/4.02-groq-provider.md) | `GroqProvider` adapter | Build | G | todo | |
| [4.03](phase-4/4.03-eval-mode.md) | Eval mode (`APP_ENV=eval`) and the eval LLM cache | Build | G | todo | |
| [4.04](phase-4/4.04-judge.md) | Judge rubrics and the judge module | Build | A+G | todo | |
| [4.05](phase-4/4.05-promptfoo-tracer.md) | Tracer bullet: promptfoo harness with deterministic assertions | Build | G | todo | |
| [4.06](phase-4/4.06-judge-asserts.md) | Judge-based assertions: faithfulness and correctness | Build | G | todo | |
| [4.07](phase-4/4.07-gate-generation-spec.md) | Generation gate and inconclusive rule — spec | Spec | G | todo | |
| [4.08](phase-4/4.08-gate-generation.md) | Generation gate and inconclusive rule — implementation | Author | A | todo | |
| [4.09](phase-4/4.09-generation-baseline.md) | First generation baseline and the eval report | Build + Baseline | G | todo | |
| [4.10](phase-4/4.10-eval-workflow.md) | `eval.yml` workflow with the PR comment | Build | G | todo | |
| [4.11](phase-4/4.11-judge-agreement.md) | Judge–human agreement | Build | A+G | todo | |
| [4.12](phase-4/4.12-closeout.md) | Gate demonstrations and Phase 4 closeout | Closeout | A+G | todo | |
| **5** | **[UI, protection, deploy (MVP)](phase-5/README.md)** | | | | |
| [5.01](phase-5/5.01-decision-rate-limit.md) | Decision: rate limiter on serverless, limit and budget numbers | Decision | D | todo | |
| [5.02](phase-5/5.02-decision-deploy.md) | Decision: deployment parameters (region, durations, timeouts) | Decision | D | todo | |
| [5.03](phase-5/5.03-proxy-auth.md) | Proxy secret, trusted client IP and IP hashing | Build | G | todo | |
| [5.04](phase-5/5.04-rate-limit.md) | Input limits and the rate limiter | Build | G | todo | |
| [5.05](phase-5/5.05-budget.md) | Global daily LLM budget | Build | G | todo | |
| [5.06](phase-5/5.06-frontend-tracer.md) | Tracer bullet: frontend proxy, generated types, minimal ask page | Build | G | todo | |
| [5.07](phase-5/5.07-ask-page.md) | Complete ask page | Build | G | todo | |
| [5.08](phase-5/5.08-backend-vercel.md) | Backend Vercel configuration | Build | G | todo | |
| [5.09](phase-5/5.09-neon-prod.md) | Neon production: migrate, `app` role, `ingest.yml` | Build | G (A runs) | todo | |
| [5.10](phase-5/5.10-deploy.md) | Vercel projects, env vars, first deploy, cold start | Build | A (+G checklist) | todo | |
| [5.11](phase-5/5.11-housekeeping.md) | `housekeeping.yml`: retention and daily keep-warm ping | Build | G | todo | |
| [5.12](phase-5/5.12-readme-v1.md) | README v1 and MVP closeout | Closeout | G draft, A final | todo | |
| **6** | **[Re-ranking with measured lift](phase-6/README.md)** | | | | |
| [6.01](phase-6/6.01-decision-cohere.md) | Decision: Cohere rerank model, trial limits, cap, price | Decision | D | todo | |
| [6.02](phase-6/6.02-rerank-adapter.md) | Cohere rerank adapter and rerank cache | Build | G | todo | |
| [6.03](phase-6/6.03-rerank-path.md) | Rerank in the request path with graceful degradation | Build | G | todo | |
| [6.04](phase-6/6.04-eval-rerank.md) | `hybrid_rerank` in both evals and baseline rows | Build + Baseline | G | todo | |
| [6.05](phase-6/6.05-lift-analysis.md) | Lift analysis and Phase 6 closeout | Closeout | A (+G tooling) | todo | |
| **7** | **[Multi-provider fallback + circuit breaker](phase-7/README.md)** | | | | |
| [7.01](phase-7/7.01-decision-breaker.md) | Decision: breaker state on serverless, quota reset, breaker timings | Decision | D | todo | |
| [7.02](phase-7/7.02-router-spec.md) | Provider router — spec | Spec | A+G | todo | |
| [7.03](phase-7/7.03-router.md) | Provider router — implementation | Author | A | todo | |
| [7.04](phase-7/7.04-router-wiring.md) | Router in the pipeline, fallback metrics, demo and Phase 7 closeout | Build + Closeout | G | todo | |
| **8** | **[Observability dashboard, cost, calibration](phase-8/README.md)** | | | | |
| [8.01](phase-8/8.01-dashboard-views.md) | Dashboard views migration | Build | G | todo | |
| [8.02](phase-8/8.02-metrics-api.md) | `GET /v1/metrics/summary` | Build | G | todo | |
| [8.03](phase-8/8.03-metrics-page.md) | Next.js `/metrics` page and `/api/metrics` proxy | Build | G | todo | |
| [8.04](phase-8/8.04-calibration.md) | Confidence calibration report | Author | A (+G data export) | todo | |
| [8.05](phase-8/8.05-closeout.md) | Cost and latency in README, Phase 8 closeout | Closeout | A | todo | |
| **9** | **[Hardening and polish](phase-9/README.md)** | | | | |
| [9.01](phase-9/9.01-failure-modes.md) | Failure-mode analysis from real data | Build | A+G | todo | |
| [9.02](phase-9/9.02-workflow-liveness.md) | Scheduled-workflow liveness check | Build | G | todo | |
| [9.03](phase-9/9.03-index-prune.md) | `grounded index prune` | Build | G | todo | |
| [9.04](phase-9/9.04-readme-final.md) | README final pass, demo GIF, project closeout | Closeout | G draft, A final | todo | |
| [9.05](phase-9/9.05-chunking-comparison.md) | *Optional:* chunking strategy comparison | Build + Baseline | G | todo | |
| [9.06](phase-9/9.06-golden-v2.md) | *Optional:* golden set v2 (grow the set) | Build + Baseline | A+G | todo | |
| [9.07](phase-9/9.07-hnsw-experiment.md) | *Optional:* HNSW index experiment | Build | G | todo | |
| [9.08](phase-9/9.08-pii-masking.md) | *Optional:* PII masking before storage | Build | G | todo | |
| [9.09](phase-9/9.09-tracing.md) | *Optional:* Langfuse tracing | Decision + Build | D, G | todo | |
