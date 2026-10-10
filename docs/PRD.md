# PRD — Grounded

> **Grounded** (working name) — a question-answering service over the FastAPI documentation that returns
> strictly typed, cited answers and proves its quality with a measurable evaluation harness gated in CI.

| | |
|---|---|
| Status | Phases 0–4 done (Phase 4, closed 2026-10-10: first generation baseline committed, the generation gate demonstrated on a degraded prompt, judge–human agreement recorded, ticket 4.12); next: Phase 5 |
| Owner / Author | probniprobic4@gmail.com |
| Last updated | 2026-10-10 |
| Related | [Tech.md](Tech.md) · [DB.md](DB.md) · [../AGENTS.md](../AGENTS.md) · [../README.md](../README.md) |

---

## 1. Summary

Grounded answers developer questions about FastAPI using Retrieval-Augmented Generation (RAG) over a
frozen snapshot of the official FastAPI docs. Every answer is a validated Pydantic object with per-claim
citations that link to the exact docs section, a status (`answered | partial | insufficient_context`) and
per-claim confidence. Quality is measured with a golden set, retrieval metrics (Recall@k, MRR, nDCG) and
generation metrics (faithfulness, correctness, refusal accuracy). Regressions are blocked in CI.

The whole system runs on free tiers (Neon, Vercel, Gemini, Groq, Cohere trial).
Cost is reported as *shadow cost* (what it would cost at paid list prices).

## 2. Problem & motivation

**For users (FastAPI developers):** keyword search on docs sites misses questions phrased differently
from the docs. Generic chatbots answer from stale memory with no sources. Developers need short,
correct answers with code and a link to the exact section they can verify.

**For the Author (primary driver):** learning and portfolio. The project must demonstrate, in a way a
recruiter or interviewer can verify in minutes:

1. RAG done properly (hybrid retrieval, fusion, re-ranking, measured lift).
2. Structured outputs (schema-enforced generation, validation, retries, citation integrity).
3. Evals as system design (golden set, metrics, LLM-as-judge with measured agreement, CI quality gate).
4. Production thinking on a zero budget (fallbacks, rate limits, caching, observability, cost, privacy).

The architecture must not block a later path to real users (auth, paid tiers, more corpora).

## 3. Goals and non-goals

### Goals
- G1. Cited, schema-valid answers to FastAPI questions via a public web demo.
- G2. A reproducible eval table with at least 3 metrics and real numbers, comparing no-RAG vs retrieval variants.
- G3. A CI gate that blocks a PR on a measured quality regression.
- G4. Honest engineering documentation: architecture, cost, latency, failure modes, limitations.
- G5. The Author can explain and defend every core component in an interview.

### Non-goals (for this project)
- Multi-turn conversation (single question → single answer; the UI may *display* history only).
- Streaming responses.
- Multiple corpora or multiple FastAPI versions at once.
- User accounts, login, or billing.
- Non-English questions or answers.
- Agentic tool use, web search, or code execution.
- Using LangChain, LlamaIndex, LiteLLM, or Instructor (the mechanics are the point of the project).

## 4. Users and use cases

| Persona | Need | Primary surface |
|---|---|---|
| FastAPI developer | "How do I do X?" answered correctly with code and a link | `/` ask page |
| Recruiter / interviewer | See in < 5 min that it works, is measured and is engineered | README, live demo, `/metrics`, PR history |
| Author (operator) | Change prompts or retrieval safely; see quality/cost/latency impact | CI eval comments, dashboard, logs |

### Core use cases
- UC1. Ask a how-to question ("How do I add a background task after returning a response?") → answer with code + citations.
- UC2. Ask a factual question ("What status code does `HTTPException` default to?") → short answer + citation.
- UC3. Ask a question spanning multiple sections (dependencies + security) → answer citing several sections.
- UC4. Ask something the docs do not cover ("How do I configure Django middleware?") → explicit `insufficient_context`, no invented answer.
- UC5. Open a citation → land on the exact section of fastapi.tiangolo.com.
- UC6. Author opens a PR changing the prompt → CI posts an eval diff table and blocks the merge if quality regressed.

## 5. Corpus definition

| Property | Value |
|---|---|
| Source | `fastapi/fastapi` GitHub repo, `docs/en/docs/**/*.md` |
| Version | One pinned git tag (latest stable release at Phase 1 start); SHA stored per index version |
| Code examples | `docs_src/` include directives are resolved to real code (one preferred variant, e.g. the newest Python version) |
| Excluded | Translations (`docs/*/` other than `en`), release notes page, sponsor/external-links pages (final list in Tech.md) |
| License | MIT (FastAPI). Attribution in README |
| Expected size | ~150–200 pages, ~2–4k chunks |

Updating the corpus is a deliberate action (new tag → new index version → golden set re-check → new baseline).
It never happens silently.

## 6. Functional requirements

IDs are referenced from Tech.md, tests and PRs.

### Ingestion
- **FR-1** A CLI command builds an index from a given FastAPI git ref: clone/checkout, parse Markdown, resolve code includes, chunk, embed, store.
- **FR-2** Chunking is header-aware (H2/H3 sections) with a max token size and overlap for long sections. Code blocks are never split. Each chunk carries its heading breadcrumb.
- **FR-3** Ingestion is idempotent: unchanged chunks (same content hash) reuse cached embeddings.
- **FR-4** Every build is recorded as an *index version* (git ref/SHA, embedding model + dim, chunking config, counts). Exactly one version is active at a time.

### Retrieval
- **FR-5** Hybrid retrieval: dense (pgvector cosine) top-20 + lexical (Postgres full-text) top-20, fused with Reciprocal Rank Fusion.
- **FR-6** Optional re-ranking (Cohere) behind a feature flag. It degrades gracefully to RRF order on error or quota exhaustion.
- **FR-7** The top 5 chunks go into the prompt. All K values are configuration, not constants in code.

### Answering
- **FR-8** `POST /v1/ask` accepts a question (max 500 chars) and returns a validated `AskResponse`.
- **FR-9** Generation uses the provider's native structured output (JSON schema). The output is always validated with Pydantic. On validation failure: one retry with error feedback, then fallback provider.
- **FR-10** Answers are decomposed into *claims*. Each claim cites chunk labels (`c1..c5`). The server rejects labels not in the retrieved set and maps valid ones to URL + anchor, title, breadcrumb and snippet. The LLM never produces URLs.
- **FR-11** Status is one of `answered | partial | insufficient_context`. For out-of-corpus questions the service refuses rather than answering from general knowledge.
- **FR-12** Per-claim confidence is computed server-side by a documented heuristic from retrieval signals, citation validity and LLM self-report. Its calibration is measured.
- **FR-13** Answers are Markdown with code blocks allowed. Target length ≤ ~250 words.
- **FR-14** Every response includes `meta`: provider, model, fallback used, cache hit, prompt version, index version, stage latencies, tokens and shadow cost.

### Resilience and protection
- **FR-15** Primary LLM is Gemini Flash, fallback is Groq. Fallback triggers on 429, 5xx or timeout, with a circuit breaker honoring `Retry-After`. Fallback is disabled in eval mode.
- **FR-16** Rate limit per client (IP hash): per-minute and per-day. There is also a global daily LLM-call budget; when exceeded, the demo returns a friendly "demo budget reached" state.
- **FR-17** Answer cache keyed by normalized question + prompt version + index version + retrieval config.
- **FR-18** The backend accepts `/v1/*` traffic only from the frontend proxy (shared secret). The real client IP is forwarded by the proxy.

### Evaluation
- **FR-19** Golden set of ~30 questions (≈25 answerable + ≈5 unanswerable). Each item has a reference answer, section-level relevance labels (graded 2/1) and a type.
- **FR-20** Retrieval eval (Python, deterministic, cached): Recall@k, MRR, nDCG@k for configs `dense`, `fts`, `hybrid`, `hybrid_rerank`.
- **FR-21** Generation eval (promptfoo with a Python provider running the real pipeline): faithfulness (per claim), answer correctness, citation precision, refusal accuracy, schema first-try validity, latency and shadow cost. Includes a `no_rag` baseline.
- **FR-22** The LLM judge runs on a different provider than the generator, with a pinned model, temperature 0 and a written rubric. Judge-vs-human agreement is measured on ≥10 verdicts and published.
- **FR-23** CI gate in two tiers. Every PR gets unit/integration tests and the retrieval eval. PRs labeled `run-eval` and pushes to `main` get the full generation eval. Quota/429-dominated runs end as **inconclusive**, not as failures.
- **FR-24** CI posts (and updates) a PR comment with a metrics table vs baseline, Δ and ✅/❌ per threshold, and links the promptfoo report artifact.
- **FR-25** Baselines are committed files. They change only through an explicit PR.

### UI and dashboard
- **FR-26** Ask page: answer with inline citation markers `[1]`, source cards (breadcrumb, snippet, link), status badge, low-confidence hint, a collapsible debug row (latency per stage, tokens, provider, shadow cost), a privacy notice and a cold-start "waking up" state.
- **FR-27** Public `/metrics` dashboard with aggregates only: p50/p95 latency per stage, shadow cost per 1k questions, cache hit rate, fallback rate, validation failure rate, and the latest eval table. It never shows question text.

## 7. Non-functional requirements

| ID | Area | Requirement |
|---|---|---|
| NFR-1 | Latency | Warm p95 end-to-end ≤ 8 s, p50 ≤ 4 s (initial targets; cold starts excluded but measured and documented) |
| NFR-2 | Availability | Best-effort on free tiers. The backend is kept warm by a daily ping. Degraded modes (no rerank, fallback LLM, budget reached) are explicit in the UI |
| NFR-3 | Cost | $0 infrastructure. Shadow cost is computed from a dated pricing file |
| NFR-4 | Privacy | Privacy notice in UI. No raw IPs stored (HMAC hash). Question text retained ≤ 30 days. No question text on public surfaces |
| NFR-5 | Reproducibility | Pinned corpus SHA, pinned models, versioned prompts, versioned golden set, committed baselines |
| NFR-6 | Testability | Tests never call real LLM/embedding/rerank APIs. Integration tests use a real pgvector in Docker |
| NFR-7 | Security | Secrets only in env/CI secrets. Backend protected by proxy secret. No raw HTML rendering of model output |
| NFR-8 | Maintainability | Typed Python (pyright), ruff, small modules, docs updated in the same PR as behavior changes |
| NFR-9 | Accessibility | UI keyboard-navigable, sufficient contrast, citations reachable without hover |

## 8. Success metrics

Initial targets. They are confirmed or adjusted after the first baselines (Phase 1 and Phase 4) and recorded in the
baseline files. Note the granularity: with ~25 answerable questions, one question ≈ 4 percentage points.

| Metric | Definition (details in Tech.md §15) | Initial target |
|---|---|---|
| Recall@5 (hybrid) | share of grade-2 sections found in top-5 | ≥ 0.80 |
| MRR (hybrid) | 1 / rank of first grade-2 section | ≥ 0.60 |
| nDCG@5 | graded gain, section-level | ≥ 0.65 |
| Rerank lift | nDCG@5(hybrid_rerank) − nDCG@5(hybrid) | > 0, reported whatever it is |
| Faithfulness | supported claims / all claims | ≥ 0.90 |
| Answer correctness | judge vs reference answer, 0 / 0.5 / 1 | ≥ 0.75 |
| Refusal accuracy | correct refuse / not-refuse decisions | ≥ 0.90 |
| Schema first-try validity | valid on first attempt | ≥ 0.97 |
| Judge–human agreement | on ≥10 manually reviewed verdicts | reported (≥ 0.8 desired) |
| RAG value | correctness(hybrid) − correctness(no_rag) | > 0, reported |

Portfolio success means the README shows a real eval table, a live URL that works from cold, a demonstrated
blocked PR and a failure-modes section based on observed data.

## 9. Delivery phases

Phases run strictly in order. A phase is done only when its **exit criteria** are met. Each step inside a phase is a
small PR that states its eval impact.

**Ownership legend.** **A** = Author writes the code (Agent gives guidance, interfaces and test cases, and reviews).
**G** = Agent writes the code and explains it (Author reviews). See AGENTS.md §3.

Timeline: Phases 0–5 = MVP (~2 weeks). Phases 6–8 = "wow" features (week 3). Phase 9 = buffer and polish.

---

### Phase 0 — Foundations
**Goal:** a working skeleton where every later piece has a home and CI runs from day one.

| Task | Owner |
|---|---|
| Accounts & keys checklist: GitHub repo, Neon project, Google AI Studio (Gemini) key, Groq key, Cohere trial key, Vercel account | A |
| `git init`, monorepo layout (`backend/`, `frontend/`, `eval/`, `infra/`, `docs/`), `.gitignore`, `.env.example` | G |
| Backend: uv project (Python 3.12), ruff, pyright, pytest, pydantic-settings config, FastAPI app factory with lifespan, `/healthz`, `/readyz` | G |
| `infra/docker-compose.yml` with `pgvector/pgvector:0.8.6-pg17-trixie` (tag pinned) | G |
| SQL migration runner + `0001_init.sql` (full schema from DB.md) | G |
| Typer CLI entry point `grounded` (`migrate` command) | G |
| Frontend: Next.js (App Router, TypeScript strict, Tailwind, shadcn/ui) scaffold | G |
| `ci.yml`: ruff, pyright, pytest with a pgvector service container, frontend lint/typecheck/build | G |

**Exit criteria**
- `docker compose up -d db` → `uv run grounded migrate` → `uv run pytest` passes locally.
- `/readyz` returns OK against the local DB.
- CI is green on the first PR.
- All accounts exist and keys are stored in a local `.env` and GitHub secrets. They are never committed.

---

### Phase 1 — Ingestion, golden set, dense baseline
**Goal:** a real index and a way to measure retrieval before anything is optimized.

| Task | Owner |
|---|---|
| Corpus fetch: clone FastAPI at pinned tag into `.cache/corpus`, record SHA | G |
| Markdown parsing + `docs_src` include resolution (both include syntaxes, preferred variant, strip highlight params) | G |
| **Header-aware chunker** (breadcrumbs, anchors, max tokens, overlap, never split code) | **A** (delegated to G by the Author, 2026-09-26; the keep-with-next fix, review item #5, delegated again on 2026-09-29) |
| Gemini embeddings adapter (batching, `task_type`, 768 dims, L2 normalization, backoff) + SQLite embedding cache | G |
| Ingest pipeline writing `index_versions`, `documents`, `chunks`; activation | G |
| Golden set v1: Agent drafts ~50 candidate questions from random sections → Author selects, rewrites and labels 30 | A (curation) / G (drafts) |
| **Retrieval metrics** (Recall@k, MRR, nDCG@k with section matching rules) | **A** (delegated to G by the Author, 2026-09-26) |
| Retrieval eval runner + `dense` config + results JSON | G |

**Exit criteria**
- Index built from the pinned tag. Chunk count and token stats recorded in `index_versions`.
- `eval/golden/golden_set.v1.jsonl` has 30 reviewed items (≈25 answerable, ≈5 unanswerable) with types balanced.
- Unit tests for chunker and metrics (hand-computed expected values).
- `dense` baseline committed to `eval/baselines/retrieval.json`.

---

### Phase 2 — Hybrid retrieval
**Goal:** measurable lift from lexical + dense fusion. The CI retrieval gate goes live.

| Task | Owner |
|---|---|
| Lexical retrieval query (FTS with OR-semantics tsquery, `ts_rank_cd`) | **A** |
| **Hybrid SQL with RRF** (single query / CTEs, k=60, configurable K) | **A** |
| Retrieval config object + config hash | G |
| Ablation runs: `dense`, `fts`, `hybrid` | G |
| **Retrieval gate logic** (`evaluate_gate`: thresholds, tolerance, setup mismatch) | **A** |
| CI: retrieval eval job (corpus + embedding caches restored, ingest into service DB, gate vs baseline) | G |
| Gate demonstration: a deliberately bad PR (e.g. broken fusion) is blocked. Keep the screenshot | A |

**Exit criteria**
- Ablation rows `dense / fts / hybrid` in `eval/baselines/retrieval.json`.
- Hybrid ≥ dense on Recall@5 and MRR, or the reason is documented.
- The retrieval gate blocks a regression in CI (demonstrated).

---

### Phase 3 — `/ask` with structured output
**Goal:** the core product behavior end-to-end, without UI.

| Task | Owner |
|---|---|
| `LLMProvider` protocol + `GeminiProvider` (native JSON schema, usage, thinking budget config) + `FakeLLMProvider` | G |
| Pydantic schemas: `LLMAnswer`, `AskRequest`, `AskResponse` | G (Author reviews closely) |
| Prompt files (`answer_v1`) + prompt loader with version/hash | G |
| Context builder (`c1..c5` labels, source blocks) | G |
| Validation + one retry with error feedback | G |
| Citation mapping/validation and marker rewriting (`[c3]` → `[n]`) | G |
| **Confidence heuristic** (per claim, components, cap for uncited claims) | **A** |
| Refusal handling (`insufficient_context` rules) | G |
| `no_rag` mode (same schema, no context) for the baseline | G |
| Request logging (stage timers, tokens), shadow cost from `pricing.toml` | G |
| Answer cache (Postgres) | G |
| `POST /v1/ask` route + error model | G |

**Exit criteria**
- All golden-set questions return schema-valid `AskResponse` locally.
- Tests with `FakeLLMProvider` cover: invalid JSON → retry, invalid citation removal, zero valid citations → retry, refusal path, cache hit.
- Every request writes a `request_logs` row with stage latencies and shadow cost.

---

### Phase 4 — Generation eval + CI quality gate
**Goal:** the eval harness that justifies the project.

| Task | Owner |
|---|---|
| `GroqProvider` adapter (also used as judge) | G |
| Judge rubrics: faithfulness per claim, correctness vs reference (prompt files) | A + G together |
| promptfoo config: Python provider calling the pipeline in-process; configs `no_rag`, `hybrid` | G |
| Assertions: faithfulness (Python, per claim), correctness (custom Python grader through the judge module, D50), citation precision, refusal, schema validity | G |
| **Metric aggregation + generation gate logic** (thresholds, tolerance, inconclusive rule; the retrieval part is Phase 2) | **A** (delegated to G by the Author, 2026-10-08: "write it", explained line by line in the 4.08 PR) |
| `grounded eval baseline` (baseline rows written from a results file, never typed) and `grounded eval report` (the README tables from the committed baselines) | G |
| Eval mode: fallback off, temperature 0, LLM response cache (SQLite, keyed by full prompt and the adapter settings), concurrency 1, backoff on a 429 and a bounded retry of a transient 5xx | G |
| `eval.yml`: label `run-eval` + push to `main` (+ `workflow_dispatch`); PR comment with diff table; promptfoo HTML report artifact; the `eval_runs` insert on `main` ships disabled (`EVAL_RECORD_RUNS`) | G |
| Judge–human agreement tooling: a blind sample drawn from a real run with synthetic negative controls, exact agreement and Cohen's kappa | G |
| Judge–human agreement: the Author labels the sheet (20 items); agreement recorded | A |

**Exit criteria**
- Full eval runs locally and in CI. Baseline committed to `eval/baselines/generation.json` (rows `no_rag`, `hybrid`).
- A deliberately degraded prompt is blocked by the gate (demonstrated).
- A quota-limited run ends as `inconclusive` (demonstrated, or simulated with the fake provider).
- Judge agreement number recorded.

**Closed on 2026-10-10** (report with the evidence and the caveats: [4.12](../tickets/phase-4/4.12-closeout.md#closeout-report)).
All four pass. (1) The baseline is `eval/baselines/generation.json` (PR #76) and the full eval passed in CI (run
37906313627), but 8 of the 10 CI eval runs that reached a verdict were `inconclusive` on free-tier quota. (2) PR #86, a
one-line prompt/schema format mismatch, was failed by the gate; the `eval` check is not a required check, and weakening
the instructions instead did not trip the gate (open items below). (3) Two real CI runs ended `inconclusive` with exit
0, and fixture tests pin the rule. (4) The agreement is recorded on 20 items; correctness alone is below 0.8.

---

### Phase 5 — UI, protection, deploy (= MVP complete)
**Goal:** a public, protected, working demo with an honest README.

| Task | Owner |
|---|---|
| Rate limiter (in-memory, per IP hash), daily budget (Postgres atomic counter), input limits | G |
| Proxy secret check + trusted client-IP header | G |
| Next.js ask page: markers, source cards, status badge, low-confidence hint, debug row, privacy notice, cold-start state | G |
| Route Handler proxy `/api/ask` (secret header, client IP, timeout) | G |
| Backend Vercel project config (root `backend/`, FastAPI entrypoint, function region, production-only secrets) | G |
| Neon production: migrate, ingest via `workflow_dispatch` | G (run by A) |
| Vercel projects (frontend root `frontend/`, backend root `backend/`), env vars | A |
| `housekeeping.yml`: daily retention SQL | G |
| README v1: architecture, real eval table, cost/latency method, failure modes v1, live URL | G draft, A final |

**Exit criteria (MVP definition of done)**
- Live URL answers a question after the backend was idle (cold start measured and documented).
- Rate limit and budget verified manually.
- README numbers are copied from committed eval results, never typed by hand.
- All CI workflows green on `main`.

---

### Phase 6 — Re-ranking with measured lift
| Task | Owner |
|---|---|
| Cohere rerank adapter (pinned model, timeout, `top_n`) behind `RERANK_PROVIDER` flag | G |
| Rerank cache (SQLite in dev/CI) + daily rerank cap + graceful degradation | G |
| `hybrid_rerank` config in retrieval + generation evals | G |
| Lift analysis (nDCG/MRR, per-question wins/losses) in README | A |

**Exit:** `hybrid_rerank` rows in both baselines. Lift reported honestly (including "no lift" if so). Degradation path tested.

### Phase 7 — Multi-provider fallback + circuit breaker
| Task | Owner |
|---|---|
| **Provider router with circuit breaker** (closed/open/half-open, `Retry-After`, quota detection, all-open → 503) | **A** |
| Tests with fake providers simulating 429 / 5xx / timeout / invalid JSON | A + G |
| Fallback metrics in logs and response `meta` | G |

**Exit:** simulated 429 on Gemini → Groq answers, breaker skips Gemini until reset. Fallback rate visible in logs.

### Phase 8 — Observability dashboard, cost, calibration
| Task | Owner |
|---|---|
| SQL views for daily stats and percentiles | G |
| `GET /v1/metrics/summary` + Next.js `/metrics` page (aggregates only) | G |
| Latest eval table on dashboard (from `eval_runs`) | G |
| Confidence calibration report (buckets vs judge-supported rate) | A |
| README cost/latency section with real numbers | A |

**Exit:** public dashboard live. p50/p95 per stage and shadow cost per 1k questions shown. Calibration table in README.

### Phase 9 — Hardening and polish (buffer)
- Failure-mode analysis from real logs and eval failures (taxonomy + examples) → README.
- README final pass (problem → architecture → eval → cost → failure modes → limitations), demo GIF.
- Optional: chunking strategy comparison (fixed 500/50 vs header-aware), PII masking, Langfuse tracing, HNSW index experiment.

## 10. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Free-tier quotas (Gemini RPD/RPM, Groq TPM, Cohere ~1k/month) | Demo down, eval inconclusive | Caches everywhere, budget caps, fallback, concurrency 1, `inconclusive` outcome, rerank cap |
| Cold starts (serverless function, Neon scale-to-zero) | Bad first impression | UI "waking up" state, cold start measured in README |
| GitHub disables scheduled workflows after 60 days of repo inactivity | Keepalive silently stops | Documented. Periodic commits or manual re-enable. Check in Phase 9 |
| Vercel function time limit on free plan | Proxy times out on slow answers | Set `maxDuration`, backend deadline < proxy timeout, verify current plan limits |
| LLM judge bias/noise | Misleading metrics | Different provider than generator, pinned model, temp 0, rubric, measured human agreement |
| Small golden set (~30) | Coarse, noisy metrics | Report n and granularity. Tolerances ≥ one question. Grow the set in Phase 9 |
| Gemini free tier may use prompts for model improvement | Privacy | UI notice, minimal logging, README "production: paid tier" note |
| Cohere trial is non-production | Can't serve real users with it | Flagged optional. Documented path: paid key or local cross-encoder |
| Postgres FTS ≠ true BM25 | Weaker lexical ranking | Stated honestly. Ablation shows actual contribution |
| Embedding model change | Full re-index | Model + dim stored per index version. Embedding cache keyed by model |
| Scope creep | MVP slips | Strict phase order and exit criteria. "Wow" features only after Phase 5 |

## 11. Decision log (condensed)

Source: planning Q&A, 2026-09-24. Changing any of these requires an explicit decision by the Author and an update here.

| # | Topic | Decision |
|---|---|---|
| D1 | Goal | Learning + portfolio first; keep a path open to real users |
| D2 | Corpus | FastAPI docs (English), frozen git tag |
| D3 | Language | English everywhere in product and repo |
| D4 | Source | Clone repo, parse `docs/en/*.md`; URLs derived from path + heading anchors |
| D5 | Chunking | Header-aware (H2/H3), ~400–500 token max, overlap, breadcrumbs |
| D6 | Code | Resolve `docs_src` includes; never split code blocks |
| D7 | Embeddings | Gemini embedding, 768 dims, model stored per index version |
| D8 | Database | Neon Postgres + pgvector (Supabase slots unavailable); local pgvector Docker for dev/CI |
| D9 | Lexical | Postgres full-text search (documented as not true BM25) |
| D10 | Fusion | Reciprocal Rank Fusion |
| D11 | Rerank | Cohere behind flag, cached, degrades to RRF |
| D12 | LLM | Gemini Flash primary, Groq fallback, via own adapters over official SDKs |
| D13 | Structured output | Native JSON schema + Pydantic validation + 1 retry, then fallback |
| D14 | Schema | Claims with citation IDs and per-claim confidence |
| D15 | Confidence | Server-side heuristic from retrieval signals (+ self-report), calibration measured |
| D16 | Citations | Chunk labels only; server validates and maps to URLs |
| D17 | Unanswerable | Explicit `insufficient_context` status; refusal accuracy metric |
| D18 | Interaction | Single-turn, no streaming |
| D19 | Protection | Per-IP rate limit, global daily budget, answer cache, input limits |
| D20 | Golden set | LLM-drafted, Author-curated, ~30 items, section-level graded labels |
| D21 | Metrics/tools | Retrieval metrics in Python; generation metrics in promptfoo |
| D22 | Judge | Different provider (Groq), pinned, temp 0, agreement measured |
| D23 | CI gate | Two tiers (always / `run-eval` label + main); quota → inconclusive |
| D24 | Hosting | Backend on Vercel Functions (FastAPI, Fluid compute) as its own Vercel project; frontend on Vercel. Changed 2026-09-24: HF Docker Spaces now require a paid PRO plan |
| D25 | FE↔BE | Next.js Route Handler proxy with shared secret |
| D26 | Observability | Own `request_logs` table + public aggregate dashboard |
| D27 | Cost | Shadow cost from dated pricing file |
| D28 | Top-K | 20 dense + 20 FTS → RRF → rerank → 5 |
| D29 | DB access | psycopg 3 async + plain SQL; plain SQL migrations |
| D30 | Fallback policy | 429/5xx/timeout + circuit breaker; disabled in eval |
| D31 | State | Rate limit in memory; cache + budget in Postgres |
| D32 | Privacy | UI notice; IP HMAC hash; question text ≤ 30 days |
| D33 | Prompts | Files in repo; version logged per request and per eval |
| D34 | Eval report | PR comment with diff table + artifact |
| D35 | Ingest ops | Local/`workflow_dispatch` CLI, idempotent, index versions |
| D36 | Answer format | Markdown + code, ~250 words, max output tokens capped |
| D37 | UI | Inline markers + source panel + status + debug row |
| D38 | Dashboard | Public, aggregates only |
| D39 | Tooling | Monorepo, uv, ruff, pyright, pytest |
| D40 | Tests | Unit + fake LLM + pgvector integration; no real APIs |
| D41 | Order | Eval-first vertical slices (this document's phases) |
| D42 | Collaboration | Author writes core modules; Agent writes boilerplate and reviews |
| D43 | Baselines | no_rag + dense + fts + hybrid + hybrid_rerank |
| D44 | MVP | Phases 0–5 in ~2 weeks |
| D45 | CI cache seeding (2026-10-01) | The CI `.cache/` is seeded by a `workflow_dispatch` "warm-cache" workflow on `main`, which embeds the misses and saves the cache even when the daily embedding quota stops it (run on two consecutive days; the same workflow is the re-seed runbook after a 7-day eviction). Only `main` and that workflow save a cache; PR jobs restore only. A cache miss that hits the quota (or has no `GEMINI_API_KEY`) fails the job as an **infra** error, never as a quality fail and never as a pass. Two caches: corpus by ref, embeddings content-addressed. The only CI secret for this is `GEMINI_API_KEY` |
| D46 | Gemini generator (2026-10-06) | Primary generator is `gemini-3.5-flash-lite` (stable ID, no shutdown date announced), with thinking level `minimal` (its default). Gemini 3.x is controlled by `thinking_level`, so `GEMINI_THINKING_BUDGET` becomes `GEMINI_THINKING_LEVEL` (ticket 3.08). Native structured output is used for the schema parts Gemini documents (`enum`, `minItems`/`maxItems`, `minimum`/`maximum`, `anyOf`); lengths and patterns stay with Pydantic. **Why Flash-Lite and not `gemini-3.6-flash`:** the Author's AI Studio dashboard (project `grounded`, free tier, 2026-10-06) shows 20 requests/day for 3.6 Flash and 500 for 3.5 Flash-Lite. 20 per day cannot run the 30-question golden set or leave a usable `DAILY_LLM_BUDGET`. `gemini-3.8-flash` and `gemini-3.7-flash` were rejected earlier: `minimal` is an error there. Quality of Flash-Lite is unmeasured until the Phase 4 eval; if it is too weak, moving to `gemini-3.6-flash` is a config change plus a new baseline, but needs a bigger quota (paid tier). Memo and sources in the 3.01 PR |
| D47 | Phase 3 review follow-ups (2026-10-07) | From the Phase 3 code review, decided by the Author: (1) the answer-cache key also includes a `generation_config_hash` (provider, temperature, max output tokens, thinking level), so changing a generation parameter is a miss and not 30 days of stale answers; (2) a `Settings` validator checks the worst-case bound of confidence invariant 4 (the self-report share), since the weights are relative and a cap on `w_self` alone guarantees nothing; (3) the retry feedback is compact (no input values, no Pydantic links) and names a cut at the token limit, with the same `max_output_tokens`; (4) links and URLs in `answer_markdown` are removed server-side and counted in the stdout summary line, so AGENTS.md §6.3 is enforced, not only asked for. Tickets R.10–R.15 |
| D48 | Groq model and judge (2026-10-08) | `GROQ_MODEL` (the Phase 7 fallback generator) and `JUDGE_MODEL` are both `openai/gpt-oss-120b`. Checked on 2026-10-08 in Groq's docs: listed under *Production Models* (`console.groq.com/docs/models`), no deprecation entry (`console.groq.com/docs/deprecations` names it only as the replacement for retired models), 131,072 context tokens, 65,536 max completion tokens, `response_format` `json_schema` with `strict: true` (constrained decoding), list price $0.15 / $0.60 per 1M input / output tokens (the model page; `groq.com/pricing` now redirects to the home page). The judge's provider (Groq) differs from the eval generator's (Gemini), as D22 and AGENTS.md §6.5 require. **Why not the others:** `openai/gpt-oss-20b` has the same limits but is the smaller model, and the judge is the part of the eval that needs reasoning quality; `llama-3.3-70b-versatile` is deprecated for free and developer tiers (shutdown 2026-08-16, already past); `qwen/qwen3.8-27b` is a *Preview* model that Groq says "may be discontinued at short notice". **The limits are tight:** the Free Plan tab of Groq's public rate-limit table lists 30 RPM, 1K RPD, 8K TPM and 200K TPD for it (the Developer Plan tab: 1K RPM, 500K RPD, 250K TPM, no TPD). Which plan the Author's account is on is unverified (the console Limits page needs a login). A full eval run makes one correctness call per answer and one faithfulness call per claim, so it is paced by concurrency 1 and `Retry-After`, and a daily stop ends as `inconclusive` (D23). Memo and sources in the 4.01 PR |
| D49 | promptfoo version and test loading (2026-10-08) | promptfoo is pinned at `0.123.1` (a stable release of 2026-09-18; npm `engines.node` is `>=22.22.0`, which the local Node 24.15.0 and CI's Node 24 satisfy) and run as `npx promptfoo@0.123.1`. `0.124.0` (2026-10-06) is newer, but it was two days old at decision time and lists eight breaking changes (SDKs become opt-in installs, a provider is removed); nothing in it is needed. Tests are loaded by promptfoo's own Python test generator, `tests: file://tests_loader.py:generate_tests` (read in the 0.123.1 source: the function is run and must return a list), so the `tests.generated.yaml` pre-step is **not** used. The Python side runs in the backend's uv environment through `PROMPTFOO_PYTHON` (the one variable that reaches the provider, the assertions and the test generator). promptfoo's own result cache is off (`--no-cache`): the cache key of a Python provider does not include the backend's prompt files, so a changed prompt would be served stale results and the gate would not see it. Details in Tech §15.3 |
| D50 | Correctness judge (2026-10-08) | Answer correctness is a custom Python grader (`asserts.py`), not `llm-rubric`: it calls `evals/judge.py`, which goes through the Groq adapter and the eval LLM cache and returns `{pass, score, reason}` with 0 / 0.5 / 1. **Why:** promptfoo's built-in Groq grading provider would bypass our adapter, the typed provider errors that feed the `inconclusive` rule, the cache and the pinned, hashed rubric prompt; `llm-rubric` with our own provider as the grader still sends promptfoo's rubric prompt, not ours. **Cost accepted:** we write and maintain the grader and its rubric (`judge_correctness_v1.md`). Faithfulness was already a custom Python assertion |
| D51 | CI model IDs (2026-10-08) | `eval.yml` sets `GEMINI_MODEL`, `GROQ_MODEL` and `JUDGE_MODEL` as literals in its `env:` block, not as repository variables or secrets. A model ID is not a secret, a change is then a reviewed diff on a PR that has to carry the `run-eval` label anyway, and no hidden variable can change what a baseline means. (The embedding model and the FastAPI ref stay `Settings` defaults, D45.) |
| D52 | Judge verdict labels and failed judge calls (2026-10-10) | The judge returns a label, never a number: faithfulness `SUPPORTED \| NOT_SUPPORTED`, correctness `CORRECT \| PARTIALLY_CORRECT \| INCORRECT`, which the code maps to 1 / 0.5 / 0 (`CORRECTNESS_SCORES`). A claim with no usable cited source is `NOT_SUPPORTED`, decided locally without a call (rubric rule 6, 4.04). A judge call that fails (invalid output after its one retry, or a provider error) is `errored`, not scored: it is left out of the mean and of `n`, and never counted as 0. **Why:** a number written by a model is not calibrated and would move with its wording; a claim with no source cannot be supported, and asking the model would only spend quota on a fixed answer; a call that failed says nothing about the answer, so scoring it 0 would blame the generator for the judge's quota (a provider-side error makes the run `inconclusive`, D23, and a judge bad output is reported in its own column). Built in 4.04 and 4.06, written down here at the Phase 4 closeout. Details: Tech §15.4 |

## 12. Assumptions and open items

- Golden set composition: ~8 factual, ~8 how-to, ~5 code-centric, ~4 multi-section, ~5 unanswerable (adjust while curating).
- Public GitHub repo, MIT license, FastAPI docs attributed.
- Neon project is in AWS US East 2 (Ohio). The backend function region is set closest to it in Phase 5 (Vercel's default is `iad1`, US East).
- Serverless backend (D24): in-process state is per function instance. The in-memory rate limiter (D31) and the circuit breaker (Phase 7) must be revisited: likely a Postgres-backed counter for rate limiting, and an accepted per-instance breaker. Decide in Phase 5 / Phase 7.
- Vercel Hobby is non-commercial only: no ads or paid features on the demo.
- Exact model IDs (Gemini Flash, Groq model, judge model, Cohere rerank model) are pinned in config at implementation time after checking current availability and free-tier limits. They are never assumed from memory. The embedding model is the exception: it was verified on 2026-09-26 (Phase 1) and is a default in `Settings`, since it is tied to the index. The Gemini generator was pinned in 3.01 (D46, 2026-10-06); the Groq model and the judge in 4.01 (D48, 2026-10-08).
- Free-tier limits of `gemini-3.5-flash-lite` (read from the Author's AI Studio rate-limit page, project `grounded`, 2026-10-06; the docs only point to that dashboard): **500 requests per day** (the Limit line sits on the labelled 500 tick). RPM (15) and TPM (250K) are read off the chart as the midpoint of the axis, they are not printed, so treat them as approximate. For comparison `gemini-3.6-flash` showed 5 RPM, 20 RPD and ~250K TPM. The daily quota resets at midnight Pacific time. Ticket 5.01 sets `DAILY_LLM_BUDGET` below 500, and every retry counts as a request.
- The current list price of `gemini-embedding-001`: the pricing page now lists only `gemini-embedding-2` ($0.20 per 1M text tokens). `pricing.toml` uses $0.15 per 1M input tokens from Google's general-availability post (2025-07-14), a dated announcement and not a row of today's page. Re-check it if the page lists the model again.
- `gemini-embedding-001` is still served (shutdown date 2028-05-14, Google's deprecations page, 2026-10-06), but Google's recommended replacement is `gemini-embedding-2`. Moving to it is a new index version (new vectors, new baselines) and is not planned in this project's phases.
- `gemini-3.5-flash-lite` has a single list price ($0.30 / $2.50 per 1M input / output tokens). The 3.6, 3.7 and 3.8 Flash prices double on 2027-01-01 (from $0.75 / $3.75 to $1.50 / $7.50), which matters only if the project moves to one of them.
- Limits of `openai/gpt-oss-120b` on Groq (`console.groq.com/docs/rate-limits`, read 2026-10-08; the page has a Free Plan tab and a Developer Plan tab): Free **30 RPM, 1K RPD, 8K TPM, 200K TPD**; Developer 1K RPM, 500K RPD, 250K TPM, no TPD listed. The page calls the table a summary with possible exceptions and sends the reader to the account's Limits page, which needs a login, so the Author's own numbers were unverified until the first real call in 4.02 (2026-10-08): the response headers `x-ratelimit-limit-requests: 1000` (requests per day) and `x-ratelimit-limit-tokens: 8000` (tokens per minute) are the Free Plan numbers, so the account is on the Free Plan. Cached tokens do not count. The docs do not say whether a request's `max_completion_tokens` is reserved against TPM (still unverified). Reasoning tokens count inside `max_completion_tokens` and inside `usage.completion_tokens` (4.02, Tech §9.1). A 429 carries `retry-after` in seconds (set only on a 429). Ticket 4.09 reads the Limits page before the first full run. A TPD stop ends a run as `inconclusive`, and the eval cache keeps what was already paid for.
- Groq `reasoning_effort` for `openai/gpt-oss-120b` accepts `low`, `medium` and `high` (default `medium`), and Groq's reasoning page says the reasoning chains "are part of the token output". With 8K TPM, `low` saves a lot of tokens per judge call but may change verdict quality. Decided in 4.02: the `Settings` field `GROQ_REASONING_EFFORT`, default `low`. The judge agreement (4.11) was to show whether the choice hurt; 4.11b could not settle it (no run at a higher effort to compare with, see the correctness-rubric item at the end of this section).
- Rate-limit and budget numbers are set below current free-tier limits, verified at Phase 5.
- **Judge budget of a full run (4.06, for 4.09).** The 4.06 run on 3 questions made 10 judge calls: a faithfulness call whose claim cites three chunks took about 1.9K input tokens, a correctness call about 1.3K. Extrapolated to the 30 golden questions (about 25 answered `hybrid` answers with 2-3 claims each, and 60 correctness calls), a full run needs roughly 180K tokens against the free plan's 200K per day. That is an estimate from two answers, not a measurement, and it leaves no room for retries or a second attempt: 4.09 should read the account's Limits page first, expect a TPD stop to be possible, and rely on the eval cache to finish on the next day. (Superseded by the measured run below, which came to about 145K.)
- **Judge budget of the first full run, measured (4.09b, 2026-10-08).** One full run (30 questions, both configs, `-j 1`,
  Groq Free Plan) made 103 live judge calls (the eval cache replayed 10 more from earlier tickets), about 144.6K Groq
  tokens (133,436 input + 11,144 output, 4,943 of the output reasoning tokens), below the 180K estimate above and
  inside the 200K per day. It waited out 68 per-minute 429s (457 s of `Retry-After`) and took 12 min 55 s; the Gemini
  generator made 54 live calls (about 110K tokens) with no 429 and no timeout. These come from the run's log and eval
  cache (gitignored), not from a committed file. The day's headroom was small (earlier tickets had already spent part of
  it), so a second full run with a changed prompt on the same day is not safe; re-running the identical command is free.
- **Generation baseline tooling (4.09a), open items.** (1) A generation baseline row's `git_sha` is the repository's HEAD
  when `grounded eval baseline` runs, not when promptfoo ran (promptfoo's file does not record it), so the baseline PR
  must write it from the checkout that ran the eval, before any other commit. A check of the results file's date
  against HEAD's commit date would catch a mistake; not built. (2) The first `hybrid` thresholds are the table of Tech
  §15.5 as a constant (`evals/generation_baseline.py:INITIAL_THRESHOLDS`), copied once into the baseline file, which the
  Author approves in the baseline PR. A floor above what the first run measures (faithfulness 0.85) makes the new
  baseline fail its own gate: the command says so and the Author decides. *Resolved in #76:* the Author approved the
  numbers and the thresholds as they are, and the baseline passed its own gate. (3) The PRD §8 targets are constants in
  `evals/report.py` too (`met` / `not met` in the README); a change to PRD §8 must change both. The Groq daily budget of
  the first full run is the item above ("Judge budget of a full run").
- **`eval_runs` insert on `main` (4.10c), waiting for the Author's go.** The step and `grounded eval record` are in `eval.yml`
  but disabled: it writes to the production Neon `eval_runs` with the owner role, which AGENTS.md §5 and the ticket
  reserve for the Author. To enable it, set the repository variable `EVAL_RECORD_RUNS` to `true`; the first row appears
  after the next push to `main` whose eval got a verdict. Open points the Author may want to decide first: `status` is
  the run's verdict on every row, `no_rag` stores `none` as `index_config_hash`, and a CI re-run adds rows (DB.md §4).
  **Status at the Phase 4 closeout (2026-10-10):** the variable is not set (`gh variable list` is empty), and the step
  writes to a production `eval_runs` table that exists only once Neon production is migrated (5.09, still `todo`). So
  the Author sets `EVAL_RECORD_RUNS=true` after 5.09, not before; Phase 4 closes with the step shipped and disabled.
- ~~The generation gate says `fail`, not `inconclusive`, when every generator call of a gated config failed (found in
  4.10c).~~ Fixed in #80: a config with no answered case has no index to compare (Tech §15.5, "As built in 4.08",
  item 7), so a total outage is `inconclusive`.
- **CI eval cache scope (4.10b), open item.** The eval LLM cache of a PR run is visible to that PR only (GitHub scopes a
  cache to the ref that saved it; `main` cannot read a PR's), so after a merge that changed a prompt, a model or retrieval,
  the first `main` run pays for the changed calls once more (about 145K Groq tokens for a full judge run, against a
  200K daily quota). `eval.yml` runs on every push to `main`, but a merge that changed nothing the eval sees replays
  from `main`'s cache. Not built: handing the PR's cache file to the `main` run through an artifact, or running the
  `main` eval only when `backend/prompts`, retrieval or the eval config changed. Revisit after the first merges that
  change quality-affecting code.
- **Transient 5xx in the eval path (found in the first `push`-to-`main` run, GitHub Actions run 37817266433).** Gemini
  answered most `generateContent` calls of that run with `503` "high demand", 25 of 30 `hybrid` and 21 of 30 `no_rag`
  cases were lost, and the run ended `inconclusive` (the 20% rule, Tech §15.5). The eval wrapper now waits out a
  `ProviderUnavailable` with a doubling backoff inside the same bounded total as the 429 (Tech §15.6,
  `EVAL_UNAVAILABLE_RETRIES`, `EVAL_UNAVAILABLE_WAIT_S`). Not verified against the real API (no real call was made for
  the fix; the tests use scripted providers), and a spell of 503s longer than the 75 s of waits still costs the case:
  whether 4 retries are enough is for the next real `main` run to show, and the two settings are the knobs.
  **Status at the Phase 4 closeout:** five real runs have used the fix (`main` pushes 37820274623, 37843958678,
  37844464091, 37906313627 and the demo PR run 38044085266). None shows a generator `ProviderUnavailable` in its
  errors-by-kind line (37820274623 has one `ProviderTimeout`), so the fix has not met a 503 spell since; whether it
  would have saved 37817266433 is unknown. Still open until a spell happens again.
- The judge assertions (4.06) pay about a second of imports per judged assertion call, two per row, because `grounded.runtime`, where `open_judge` lives, imports the Gemini SDK. A module that builds the judge without it would cut that; not done, since it is small next to the 8K tokens-per-minute pacing.
- A missing `GROQ_API_KEY` or `JUDGE_MODEL` is found by the first judged assertion of a real run (every judge component of the run is then `errored` with `ProviderConfigError`, and the run stops asking), after generator calls were spent. A check in the test generator, which runs first, would find it before any quota is used; the generator calls are cached, so the cost of the late discovery is one re-run, not lost quota.
- ~~The answer cache (3.13) stores a finished answer, and its key did not include the confidence weights.~~ Closed in
  3.13: the key now includes a hash of the confidence config (Tech §11), so a change of `CONFIDENCE_*` is a cache miss.
  The alternative, computing confidence on read, would have needed the retrieval signals stored next to the answer.
- The question normalization of the cache key and `question_hash` lowercases the text (Tech §11), so `Path` and
  `path`, `Body` and `body` are one key, although in the FastAPI docs a class and a concept can differ. Accepted for
  now (Phase 3 review #9); revisit if Phase 4 or the logs show a wrong cached answer for a case-only difference.
  Phase 4 could not show one: eval mode runs with the answer cache off (Tech §15.6), so the question stays open for the
  logs of Phase 5 and later.
- A client disconnect cancels the request (`CancelledError`, a `BaseException`), and no `request_logs` row is written
  (Phase 3 review #10). Handled with the request deadline and the timeout chain of 5.02.
- Hybrid retrieval was below dense in the first ablation (ticket 2.07, README "Retrieval ablation"). Not tried, because
  tuning against the golden set needs an explicit decision (AGENTS.md §13): a lexical query closer to BM25 (AND-first,
  or weighting rare terms), a smaller weight or a shorter list (`K_FTS`) for the lexical side, `RRF_K`. Any of them
  would be a separate PR that says it was tuned on the golden set. Phase 4 adds a generation-side symptom of the same
  weakness: the baseline's two false refusals (q016 and q026, the only `refusal_correctness` errors of `hybrid`) are
  retrieval failures, the same two questions the README's retrieval reading names (the reading is in the 4.09b PR
  description, #76). A retrieval fix or the Phase 6 rerank should therefore show in the generation table too, not only
  in Recall.
- The answer-cache key is built before generation, with the configured generator's `generator_model` and generation
  parameters. Once the router falls back (Phase 7), an answer from the fallback provider would be stored under the
  primary's key and served as the primary's for up to 30 days. Decide in Phase 7 (`router.py` is Author-owned): do
  not cache a fallback answer, or key the row on the provider that answered (R.16).
- `thinking_level` is read twice: by the Gemini adapter (`runtime.py`) and by `GenerationParams` for the cache key, and
  it is hashed whatever the provider. The plan was to move it to the provider in Phase 4, when Groq arrives: each
  adapter exposes the parameters that change its answer, and the key hashes those (R.16, Author's decision of
  2026-10-07). **Not done in Phase 4:** Groq arrived only as the judge, never as a generator, so nothing forced the move,
  and it is still open. It becomes necessary when `groq` can answer requests (the router, Phase 7), unless the Author
  wants it earlier; this closeout does not decide that.
  `GROQ_REASONING_EFFORT` (4.02) is another such parameter: it is not in the answer-cache key yet, because `groq`
  cannot be the generator before the router, so the move to the provider has to include it. The eval LLM cache (4.03)
  already hashes both, per provider and only the knob that provider uses, through
  `generation/params.py:adapter_params(settings, provider)`: a second reading of `Settings`, the one the move replaces.
- `strip_urls` (Tech §9.6) is a regex, not a Markdown parser: a URL between escaped backticks, or after a "fence" with
  four spaces of indentation, or a bare e-mail address (GFM makes it a `mailto:` link) passes it. Reference
  definitions and `<scheme:…>` autolinks are removed since R.17. The guarantee is planned for the frontend (Phase 5):
  `react-markdown` with `a` and `img` disallowed (`unwrapDisallowed`), and claims and follow-up questions rendered as
  plain text (R.16).
- Confidence invariant 4 and its `Settings` guard (`self_carry_worst_case`) assume rerank off: the weakest support has
  no rerank score. When Phase 6 sets `CONFIDENCE_W_RERANK` above 0, decide what "weakest support" means for a chunk
  that retrieval found weakly but the reranker scores high, and extend the guard and its equality test (R.17).
- Judge–human agreement (4.11) was measured on 20 items, one rater (the Author, who wrote the rubrics), and the real
  baseline run has no `NOT_SUPPORTED` verdict, so the sample has four synthetic controls and cannot show whether the judge
  catches a borderline unsupported *real* claim. The degraded-prompt run of the gate demonstration (4.12) was expected to
  produce real `NOT_SUPPORTED` verdicts; a second sheet drawn from it (the same tool, another seed and results file) would measure that,
  if the agreement number is thin or low. **Status after 4.11b** (the result is in README Evaluation and
  `eval/judge_agreement/README.md`): the number is thin in exactly that respect. Faithfulness agreed on all 10 items, but
  only 6 are real claims, all `SUPPORTED`, and 4 are easy synthetic controls, so the faithfulness baseline of 1.000 is
  not validated for a subtly unsupported claim. Open: draw the second sheet from a run that has real `NOT_SUPPORTED`
  verdicts and label it, and ideally have a second rater who did not write the rubrics. **Status after 4.12:** the demo
  run (#86) is not such a run: only 8 of its 30 `hybrid` cases produced an answer, one of them was scored for
  faithfulness (n = 1, 1.000), so it has no real `NOT_SUPPORTED` verdict. A degraded prompt that produces *answers*
  with unsupported claims is needed, and the six instruction edits of the finding below did not move faithfulness
  either, on this model.
- **Correctness rubric boundary (from the 4.11b disagreements).** Correctness agreed on 7 of 10 grades (kappa below the
  0.8 target), and the three misses (a07, a09, a16) sit on the rubric's own boundaries: *main point* against *an
  important part left out*, and *a wrong detail that does not reverse the main point*. Three items cannot say whether
  the rubric or the judge is the outlier. Open for the Author: whether to write `judge_correctness_v2` with sharper
  boundaries, which means a new judge prompt version, a re-run of the generation eval, a new baseline in its own PR
  (AGENTS.md §7) and a new agreement sheet. Not decided and not part of 4.11. The Groq reasoning-effort question above
  is not settled either: no run at a higher effort exists to compare with.
- **What the 0.8 of §8 means.** §8 says "reported (≥ 0.8 desired)" for judge-human agreement without saying whether it
  is exact agreement or Cohen's kappa, nor over which items. The 4.11b numbers show the choice matters: pooled over all 20
  items both measures meet 0.8, over the 16 real items only exact agreement does, and for correctness alone neither does
  (`eval/judge_agreement/v1.agreement.md`). Open for the Author: state the measure and the scope in §8 (for example kappa,
  per kind, on real items); `grounded eval agreement` prints all of them.
- **What the generation gate can see (found in the 4.12 demonstration).** The ticket suggested degrading `answer_v1` by
  weakening an instruction. Six such edits were tried locally on `gemini-3.5-flash-lite`, each followed by a gate run:
  deleting the "cite every claim" rule, allowing outside knowledge, narrowing the refusal rule, requiring "full explicit"
  answers, and a 15-word and an 8-word answer limit. None moved a gated metric: the output section and the schema keep
  the model citing and refusing, and the length limit only shortened the prose. The outputs were not kept
  (`eval/results/` is gitignored), so this is an account, not an artifact. The edit that did trip the gate is a
  prompt/schema **format mismatch**: PR #86 changed the `citation_ids` example from `["c1", "c3"]` to `["[c1]", "[c3]"]`,
  which the schema (`^c[1-9]$`) rejects; `schema_first_try` fell to 0.233 against the floor of 0.950 (n = 30), 22 of 30
  generator calls were `ProviderBadOutput` after their retry, and correctness and refusal accuracy (both n = 8) fell
  below their thresholds too. **What this implies:** the gate is sensitive to schema and format regressions, and (by how
  `refusal_correctness` is computed) to refusals that retrieval induces, which the baseline's q016 and q026 show but
  this closeout did not demonstrate by a run; it is insensitive, on this model, to instruction-wording regressions that
  the structured output hides. One demonstration is one data point, and another model may behave differently.
  **Consequence for the process:** a PR that changes a prompt needs a human look at the diff as well as the `run-eval`
  gate; AGENTS.md §7's label is necessary, not sufficient. Not built: a set of prompt mutations the gate must fail
  (a canary), which would turn this account into a test at the cost of a full run per mutation.
- **The `eval` check is not a required check, so a red gate does not block the merge button.** On the demo PR (#86),
  GitHub reported `mergeable: MERGEABLE`, `mergeStateStatus: UNSTABLE`; the required checks of `main` are `backend`,
  `frontend` and `retrieval-eval`. `eval.yml` triggers on `labeled` and `synchronize` only, so a PR without the label
  gets no `eval` status at all, and a required check would wait for a status that never comes. Making the gate
  mechanically blocking means a trigger that always reports (for example `opened` and `reopened` too, with the job
  skipped when the label is missing; GitHub reports a job skipped by its own `if` as successful) and adding `eval` to
  the required checks. A labeled PR whose run is `inconclusive` would stay mergeable (the job succeeds), consistent with
  D23 (inconclusive is not a failure) but not a pass either. Not built; the Author decides whether process (the label
  and the review) is enough for a one-person repository.
- **Free-tier headroom of the CI eval (found in the first real runs, 2026-10-08 to 2026-10-10).** Of the 10 `eval.yml`
  runs that reached a gate verdict, 1 passed (37906313627, 2026-10-09), 1 failed on purpose (the demo, 38044085266) and
  8 were `inconclusive`, all on 2026-10-08. The causes read in their logs: the Groq daily quota (the judge:
  `ProviderRateLimited (daily quota)`, 24 to 74 calls per run, in six of the runs) and Gemini 503 spells (generator
  `ProviderUnavailable`, up to 28 calls in one run). The numbers behind it: a full uncached run needs about 145K Groq
  tokens against the Free Plan's 200K a day, so one full run a day fits and two do not, and a local baseline run spends
  the same quota as CI. How the daily limit resets is **not documented** (the rate-limits page says nothing about it,
  read 2026-10-10); that it refills continuously, about 139 tokens a minute (200K over 1,440 minutes), is the Author's
  observation and unverified. Every push to `main` starts a real eval; a merge that changed nothing the eval sees
  replays from `main`'s cache and costs no call, so the cost is the quality-affecting merges and re-runs (see the cache
  scope item above). Not built, options only: run the `main` eval only when `backend/prompts`, retrieval or the eval
  config changed; schedule it; hand a PR's cache to `main`; a paid tier. Until then, expect `inconclusive` on a day that
  already spent the quota, and do not re-run hoping for another result.
- **`warm-cache.yml` still has its own copy of the setup steps** that the composite actions `eval-index` and
  `save-embeddings` hold for `ci.yml` and `eval.yml` (Tech §17, "Shared setup"). They must be kept in step by hand; moving
  `warm-cache.yml` onto the actions is small, but it is the workflow that seeds the embeddings cache, so it was left
  alone in Phase 4.
- **`ubuntu-latest` is about to point at a newer Ubuntu.** All three workflows run on `ubuntu-latest` (`ci.yml`,
  `eval.yml`, `warm-cache.yml`). The runner-images README lists `ubuntu-latest` as Ubuntu 24.04 today with an
  `ubuntu-26.04` label available, and says a `-latest` migration is gradual, over one to two months (read 2026-10-10).
  The Author's task note names 2026-10-19 for the change; that date could not be confirmed from the README. The risk is a
  workflow that fails on the new image (Python and Node toolchains, the `pgvector` service, cache keys that include the
  OS), including the two quota-limited workflows. A separate task already exists; nothing was changed here.
- **The branch `phase-4/gate-demo` is kept** (PR #86 is closed unmerged), so the evidence of the demonstration can be
  re-run or inspected. Deleting a branch needs the Author's go (AGENTS.md §5); the Phase 2 demo branch was deleted by
  the ticket.
- Golden set v2 ideas (from the Phase 0–1 review, item #4): write questions without looking at the documentation
  (so they aren't lexical paraphrases of a section), and report metrics separately for items with `source_section`
  null and not null. Tracked in ticket 9.06.
