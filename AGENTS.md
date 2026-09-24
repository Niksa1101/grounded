# AGENTS.md — Rules for this repository

These rules apply to every AI coding agent (and every human) working in this repo. They are **mandatory**.
If a rule conflicts with a request, stop and ask the Author. Do not quietly pick one.

**Sources of truth, in order:** this file → [docs/PRD.md](docs/PRD.md) (scope, phases, decisions) →
[docs/Tech.md](docs/Tech.md) (architecture, contracts) → [docs/DB.md](docs/DB.md) (schema, SQL).
Read the relevant sections before changing anything.

---

## 1. Project in one paragraph

Grounded is a RAG service over a frozen snapshot of the FastAPI docs. It uses hybrid retrieval (pgvector + Postgres FTS
+ RRF, optional Cohere rerank) and Pydantic-validated structured answers with per-claim citations and confidence.
An eval harness (Python retrieval metrics + promptfoo generation metrics) gates PRs in CI. The backend is FastAPI on
Hugging Face Spaces, the frontend a thin Next.js app on Vercel, and the database Neon Postgres. Everything runs on free tiers.
**The primary goals are learning and a portfolio**: the Author must be able to explain and defend every core part.

## 2. Language

- **Everything in the repo is English**: code, comments, docstrings, commit messages, PR titles/descriptions, docs, UI text.
- Conversation with the Author happens in **Serbian** unless the Author switches.

## 3. Collaboration model (most important rule)

The Author writes the core of the system. The Agent writes boilerplate and teaches.

### Author-owned modules — the Agent must NOT write the implementation

| Module | Path | Phase |
|---|---|---|
| Header-aware chunker | `backend/src/grounded/ingest/chunker.py` | 1 |
| Retrieval metrics (Recall@k, MRR, nDCG) | `backend/src/grounded/evals/metrics.py` | 1 |
| Lexical FTS query | `backend/src/grounded/retrieval/lexical.py` | 2 |
| Hybrid SQL + RRF | `backend/src/grounded/retrieval/hybrid.py` | 2 |
| Confidence heuristic | `backend/src/grounded/generation/confidence.py` | 3 |
| Eval aggregation + gate logic | `backend/src/grounded/evals/gate.py` | 4 |
| Provider router: fallback + circuit breaker | `backend/src/grounded/generation/router.py` | 7 |

For these modules the Agent **may**:
- explain concepts, trade-offs and pitfalls, and point to the contract in Tech.md/DB.md;
- create the file with types, signatures, docstrings and `raise NotImplementedError`;
- write tests (the spec) **if the Author asks for tests first**;
- give hints in increasing detail when the Author is stuck (concept → approach → pseudocode);
- review the Author's code thoroughly (correctness, edge cases, performance, readability).

The Agent **must not** paste or commit a finished implementation of these modules unless the Author explicitly
says so for that specific module ("write it"). If that happens, explain the code line by line afterwards.

### Agent-owned (boilerplate) — the Agent writes, the Author reviews
Scaffolding, configuration, Docker, CI workflows, migrations, adapters to SDKs, API routes, schemas, caching, logging,
rate limiting, budget, the Next.js UI, deploy scripts, and tests for these. After each such change, give a short
explanation of *what* and *why*, focusing on anything non-obvious. The Author must be able to explain it too.

## 4. Phase discipline

- Work follows the phases in [PRD §9](docs/PRD.md#9-delivery-phases) **in order**. Don't start phase N+1 work until phase N's exit criteria are met, unless the Author says so.
- "Wow" features (rerank, fallback, dashboard) come only after the MVP (Phase 5).
- Each step is a **small PR** on a branch `phase-<N>/<short-topic>` (e.g. `phase-2/hybrid-rrf`).
- At the end of a phase: check every exit criterion explicitly, report which ones pass and which don't, and update the README roadmap status.

## 5. Stop-and-ask triggers

Ask the Author before:
- changing any decision in the [PRD decision log](docs/PRD.md#11-decision-log-condensed), or anything in Non-goals;
- adding a runtime dependency (Python or npm), or any new external service or account;
- changing the DB schema (a new migration), the `LLMAnswer`/`AskResponse` schemas or the HTTP API contract;
- editing the golden set, thresholds, or a baseline file;
- changing a prompt's meaning (wording fixes included: they change the hash and eval results);
- touching secrets, deployment targets, production data, or running anything against Neon production;
- deleting files or data, force-pushing, rewriting history.

## 6. Architecture invariants (never violate)

1. **No LangChain, LlamaIndex, LiteLLM, Instructor, SQLAlchemy/ORMs.** Own adapters over official SDKs; psycopg 3 + plain SQL.
2. **Pydantic is the authority** on every boundary: HTTP I/O, LLM output, golden set rows, settings. LLM output is always validated, even with native structured output.
3. **The LLM never produces URLs.** It cites per-request labels (`c1..cK`). The server validates the labels and maps them to DB-sourced URLs. Invalid labels are removed **and counted**.
4. **Validation failure → exactly one retry on the same provider with error feedback → then fallback.** No infinite loops, no sleeping in the request path.
5. **Fallback is disabled in eval mode.** The judge must be a different provider/model than the generator.
6. **Confidence is computed server-side** (heuristic + components). Never display raw LLM self-confidence as "confidence".
7. **All configuration goes through `grounded.settings.Settings`.** No `os.environ` reads elsewhere. K values, thresholds, weights, limits and model IDs are config, not literals.
8. **Model IDs are pinned** (no "latest" aliases). Model IDs, free-tier limits and prices are **verified at implementation time** from provider docs, never assumed from memory.
9. **Prompts live in `backend/prompts/`** and are versioned by content hash. Every request log and eval run records `prompt_version`, `index_version` and `retrieval_config_hash`.
10. **Index versions:** chunks belong to an index version; exactly one is active. Changing chunking, embedding model/dim or corpus ref means a new index version.
11. **SQL parameters are always bound** (`%s` / named params). Never build SQL with f-strings from input, including vectors.
12. **Async all the way** in the request path. No blocking I/O inside `async def`. Use the SDKs' async clients.
13. **Privacy:** never store or log raw IPs (HMAC hash only). Never log question text to stdout. Question text in the DB is retained ≤ 30 days. Public endpoints return aggregates only.
14. **Frontend renders model Markdown without raw HTML.** Secrets are never exposed to the browser (no `NEXT_PUBLIC_` secrets).
15. **Free-tier discipline:** cache anything that is re-run (embeddings, rerank, eval LLM responses), concurrency 1 in evals, honor `Retry-After`, never loop on paid/quota-limited APIs.

## 7. Evaluation discipline

- **Never fabricate or hand-edit metric numbers** in README, PR descriptions or baselines. Numbers come from committed eval output.
- Baselines (`eval/baselines/*.json`) change only in a dedicated PR titled `eval: update baseline (<reason>)` with a before/after table.
- The golden set is versioned (`golden_set.vN.jsonl`). Don't edit items in place once a baseline references them.
- Any PR that can affect answer quality (prompts, retrieval, chunking, context building, schemas, model IDs) must get the `run-eval` label and report its eval impact in the description.
- `inconclusive` (quota-dominated) runs are not "passes". Re-run before merging quality-affecting changes.
- Report `n` with every metric. Don't claim improvements smaller than one question's worth as real.

## 8. Testing rules

- **pytest never calls real external APIs.** A network-blocking fixture enforces this. Use `FakeLLMProvider`, fake embedder/reranker and recorded JSON fixtures.
- Integration tests use the real pgvector container (local Docker / CI service), never Neon.
- Every bug fix starts with a failing test. Every new module ships with tests in the same PR.
- Tests are deterministic: no reliance on wall-clock timing, random seeds are fixed, no ordering assumptions without `ORDER BY`.

## 9. Code conventions

**Python**
- Python 3.12, fully type-annotated. `pyright` strict on `src/`. `ruff check` and `ruff format` clean.
- Small modules with single responsibility, following the layout in [Tech §3](docs/Tech.md#3-repository-layout). Pure functions where possible (chunker, RRF, metrics, confidence, gate).
- Pydantic v2 models for data crossing boundaries. Frozen dataclasses/models for internal value objects.
- Typed exceptions for provider errors (`ProviderRateLimited`, `ProviderUnavailable`, `ProviderTimeout`, `ProviderBadOutput`). Never swallow exceptions silently.
- Logging via stdlib `logging` (JSON formatter) with `request_id`. No `print` outside the CLI.
- Docstrings explain *why* and contracts, not what the next line does. Match the surrounding comment density.

**SQL**
- Lowercase keywords are fine, but be consistent within a file. CTEs named for what they hold (`dense`, `lexical`, `fused`).
- Every query that returns ranked rows has a deterministic `ORDER BY` including a tie-breaker.
- Migrations: new numbered file only, never edit an applied one (see [DB §10](docs/DB.md#10-migrations)).

**TypeScript / Next.js**
- TypeScript strict. App Router. Server-only code for anything touching secrets (Route Handlers).
- API types generated from the backend OpenAPI schema. Don't hand-maintain divergent types.
- Tailwind + shadcn/ui; accessible components (keyboard reachable, labeled, no hover-only info).

## 10. Commands

Available once Phase 0 lands (keep this list current):

```bash
docker compose -f infra/docker-compose.yml up -d db
```

```bash
cd backend && uv sync
```

```bash
uv run grounded migrate
```

```bash
uv run ruff check . && uv run ruff format --check . && uv run pyright
```

```bash
uv run pytest
```

```bash
uv run grounded ingest --ref <fastapi-tag> --activate
```

```bash
uv run grounded eval retrieval --config hybrid
```

```bash
npx promptfoo@<pinned-version> eval -c eval/promptfoo/promptfooconfig.yaml -j 1
```

```bash
cd frontend && npm run lint && npm run typecheck && npm run build
```

Before saying a task is done, run the relevant checks and report the actual output: lint, types, tests, and evals if quality-affecting.
If something was skipped or failed, say so plainly.

## 11. Git and PR workflow

- Never commit directly to `main`. Branch → PR → CI green → Author approves → squash merge.
- Commits: Conventional Commits (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`, `ci:`, `eval:`), imperative mood, English.
- PR description template: **What / Why / How tested / Eval impact** (numbers or "none: no quality-affecting change") / **Docs updated**.
- Docs are part of the change: if behavior, schema, config or commands change, update PRD/Tech/DB/README/AGENTS **in the same PR**.
- Never commit secrets, `.env`, `.cache/`, `eval/results/`, or large generated files. `.env.example` stays complete and value-free.
- Don't push, merge, deploy, or trigger production workflows (`ingest.yml`, `deploy-backend.yml`) without the Author's explicit go-ahead.

## 12. Definition of done (per PR)

- [ ] Scope matches the current phase; no unrelated changes.
- [ ] Author-owned modules were written by the Author (or explicitly delegated).
- [ ] Lint, types and tests are green locally and in CI.
- [ ] Quality-affecting change → `run-eval` label, eval impact reported with `n`.
- [ ] No real API calls in tests; no secrets; no raw IPs or question text in logs.
- [ ] Config, not literals, for tunables. Pinned model IDs.
- [ ] Docs updated in the same PR.
- [ ] Agent explained the non-obvious parts of any code it wrote.

## 13. What not to do

- Don't "improve" the architecture on your own (new frameworks, a vector DB SaaS, streaming, multi-turn, auth). Propose it and wait.
- Don't paper over failures: no bare `except`, no silent fallbacks without a logged reason, no skipped tests to get green CI.
- Don't tune prompts or thresholds against the golden set to make numbers look better without saying so in the PR.
- Don't widen scope mid-phase. Note ideas under "Assumptions and open items" in PRD §12 instead.
