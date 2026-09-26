# PRD — Grounded

> **Grounded** (working name) — a question-answering service over the FastAPI documentation that returns
> strictly typed, cited answers and proves its quality with a measurable evaluation harness gated in CI.

| | |
|---|---|
| Status | Phase 0 done; Phase 1 (Ingestion, golden set, dense baseline) next |
| Owner / Author | probniprobic4@gmail.com |
| Last updated | 2026-09-25 |
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
| **Header-aware chunker** (breadcrumbs, anchors, max tokens, overlap, never split code) | **A** (delegated to G by the Author, 2026-09-26) |
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
| Assertions: faithfulness (Python, per claim), correctness (`llm-rubric`), citation precision, refusal, schema validity | G |
| **Metric aggregation + gate logic** (thresholds, tolerance, inconclusive rule) | **A** |
| Eval mode: fallback off, temperature 0, LLM response cache (SQLite, keyed by full prompt), concurrency 1, backoff | G |
| `eval.yml`: label `run-eval` + push to `main`; PR comment with diff table; promptfoo HTML report artifact | G |
| Judge–human agreement: Author labels ≥10 judge verdicts; agreement recorded | A |

**Exit criteria**
- Full eval runs locally and in CI. Baseline committed to `eval/baselines/generation.json` (rows `no_rag`, `hybrid`).
- A deliberately degraded prompt is blocked by the gate (demonstrated).
- A quota-limited run ends as `inconclusive` (demonstrated, or simulated with the fake provider).
- Judge agreement number recorded.

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

## 12. Assumptions and open items

- Golden set composition: ~8 factual, ~8 how-to, ~5 code-centric, ~4 multi-section, ~5 unanswerable (adjust while curating).
- Public GitHub repo, MIT license, FastAPI docs attributed.
- Neon project is in AWS US East 2 (Ohio). The backend function region is set closest to it in Phase 5 (Vercel's default is `iad1`, US East).
- Serverless backend (D24): in-process state is per function instance. The in-memory rate limiter (D31) and the circuit breaker (Phase 7) must be revisited: likely a Postgres-backed counter for rate limiting, and an accepted per-instance breaker. Decide in Phase 5 / Phase 7.
- Vercel Hobby is non-commercial only: no ads or paid features on the demo.
- Exact model IDs (Gemini Flash, Groq model, judge model, Cohere rerank model, embedding model) are pinned in config at implementation time after checking current availability and free-tier limits. They are never assumed from memory.
- Rate-limit and budget numbers are set below current free-tier limits, verified at Phase 5.
