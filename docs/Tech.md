# Tech — Architecture and technical design

| | |
|---|---|
| Scope | How Grounded is built: components, contracts, algorithms, eval harness, CI/CD, deployment |
| Related | [PRD.md](PRD.md) (what & why, phases) · [DB.md](DB.md) (schema & SQL) · [../AGENTS.md](../AGENTS.md) (rules) |

Sections owned by the **Author** (core modules) are marked **[A]**. For those, this document defines the
*contract* (inputs, outputs, invariants, pitfalls), not the implementation.

---

## 1. System overview

```mermaid
flowchart LR
    subgraph Client
        B[Browser]
    end
    subgraph Vercel
        FE[Next.js UI<br/>/ and /metrics]
        PX[Route Handlers<br/>/api/ask · /api/metrics]
    end
    subgraph HF["Hugging Face Space (Docker)"]
        API[FastAPI /v1/*]
        RET[Retrieval<br/>dense + FTS + RRF]
        GEN[Generation<br/>provider router]
    end
    subgraph Providers
        EMB[Gemini Embeddings]
        RR[Cohere Rerank<br/>optional]
        G1[Gemini Flash<br/>primary]
        G2[Groq<br/>fallback + judge]
    end
    DB[(Neon Postgres<br/>pgvector + FTS)]

    B --> FE --> PX -- "X-Proxy-Secret + X-Client-IP" --> API
    API --> RET --> DB
    API --> EMB
    RET --> RR
    API --> GEN --> G1
    GEN -. "429/5xx/timeout" .-> G2
    API --> DB

    subgraph Offline
        ING[grounded ingest CLI] --> EMB
        ING --> DB
        CI[GitHub Actions<br/>tests · retrieval eval · promptfoo] --> G2
    end
```

Request path summary: **validate → rate-limit → cache lookup → embed query → hybrid retrieve → (rerank) → build
context → budget reserve → generate (structured) → validate/retry/fallback → map citations → confidence → log → respond.**

## 2. Tech stack

| Layer | Choice | Notes |
|---|---|---|
| Language | Python 3.12 | pinned in `pyproject.toml` + Docker base image |
| Packaging | uv (lockfile committed) | `uv sync --frozen` in CI and Docker |
| API | FastAPI + Uvicorn | async end-to-end in the request path |
| Validation | Pydantic v2, pydantic-settings | Pydantic is the authority on all I/O shapes |
| DB driver | psycopg 3 (async) + psycopg_pool + pgvector-python | plain SQL, no ORM |
| CLI | Typer | `grounded` entry point |
| LLM SDKs | `google-genai` (Gemini), `groq`, `cohere` | wrapped by own adapters; async clients |
| HTTP | httpx | health checks, anything without an SDK |
| Tokens | tiktoken (`o200k_base`) | *approximate* token counts for chunk sizing only |
| Markdown | markdown-it-py | heading/code-fence aware parsing |
| Lint/format | ruff | `ruff check` + `ruff format` |
| Types | pyright | strict on `src/`, basic on `tests/` |
| Tests | pytest, pytest-asyncio | real pgvector in Docker for integration |
| Eval | promptfoo (pinned version via `npx promptfoo@<ver>`) | Python provider + Python assertions |
| Frontend | Next.js (App Router), TypeScript strict, Tailwind, shadcn/ui, react-markdown + remark-gfm + syntax highlighting | npm |
| DB | PostgreSQL 17 + pgvector (Neon / Docker) | see DB.md |
| Hosting | HF Spaces (Docker) backend, Vercel frontend | free tiers |
| CI/CD | GitHub Actions | 5 workflows (§17) |

Explicitly **not** used: LangChain, LlamaIndex, LiteLLM, Instructor, SQLAlchemy/ORMs, vector DB SaaS.
The point is to show the mechanics.

## 3. Repository layout

```
.
├── AGENTS.md                      # rules for agents and humans
├── LICENSE                        # MIT
├── .env.example                   # every Settings variable, no values
├── README.md                      # portfolio-facing product spec
├── docs/                          # PRD.md, Tech.md, DB.md, images/
├── backend/
│   ├── pyproject.toml  uv.lock  Dockerfile
│   ├── migrations/                # 0001_init.sql, ...
│   ├── prompts/                   # answer_v1.md, judge_faithfulness_v1.md, judge_correctness_v1.md
│   ├── pricing.toml               # dated list prices for shadow cost
│   ├── src/grounded/
│   │   ├── main.py                # app factory, lifespan (pool, active index, providers)
│   │   ├── settings.py            # pydantic-settings
│   │   ├── cli.py                 # Typer: migrate, ingest, index, eval, ask, golden
│   │   ├── api/                   # routes_ask.py, routes_metrics.py, routes_health.py, deps.py, errors.py, security.py
│   │   ├── schemas/               # llm.py (LLMAnswer), api.py (AskRequest/AskResponse), eval.py (GoldenItem)
│   │   ├── ingest/                # corpus.py, markdown.py, includes.py, chunker.py [A], embed.py, pipeline.py
│   │   ├── retrieval/             # dense.py, lexical.py [A], hybrid.py [A], rerank.py, config.py, types.py
│   │   ├── generation/
│   │   │   ├── providers/         # base.py (Protocol), gemini.py, groq.py, fake.py
│   │   │   ├── router.py          # fallback + circuit breaker [A]
│   │   │   ├── prompts.py         # load + version/hash
│   │   │   ├── context.py         # c1..c5 labeling, source blocks
│   │   │   ├── citations.py       # validation, mapping, marker rewrite
│   │   │   ├── confidence.py      # heuristic [A]
│   │   │   └── pipeline.py        # orchestrates the /ask flow
│   │   ├── evals/                 # metrics.py [A], retrieval_runner.py, gate.py [A], report.py, judge.py
│   │   ├── infra/                 # db.py, migrations.py (runner), kvcache.py (SQLite), answer_cache.py, ratelimit.py, budget.py, timing.py, hashing.py, logging.py
│   │   └── observability/         # request_log.py, cost.py
│   └── tests/                     # unit/, integration/, conftest.py (fixtures), support.py (helpers), fixtures/
├── eval/
│   ├── golden/                    # golden_set.v1.jsonl, README.md (labeling guide)
│   ├── promptfoo/                 # promptfooconfig.yaml, provider.py, asserts.py, tests_loader.py
│   ├── baselines/                 # retrieval.json, generation.json  (committed, changed only by explicit PR)
│   └── results/                   # gitignored run outputs
├── frontend/                      # Next.js app (app/, components/, lib/)
├── infra/
│   ├── docker-compose.yml         # pgvector for dev
│   └── hf-space/README.md         # Space metadata (sdk: docker, app_port: 7860)
└── .github/workflows/             # ci.yml, eval.yml, deploy-backend.yml, ingest.yml, keepalive.yml
```

## 4. Configuration

All configuration is loaded once through `grounded.settings.Settings` (pydantic-settings, env vars, `.env` in dev).
Code never reads `os.environ` directly. `.env.example` lists every variable, with no real values (a unit test keeps it in sync with `Settings`).
`APP_ENV=prod` refuses to start without an explicit `DATABASE_URL`, `PROXY_SHARED_SECRET` and `IP_HASH_SECRET`,
or with `ALLOW_DIRECT_API=true`.

| Variable | Example / default | Purpose |
|---|---|---|
| `APP_ENV` | `dev` \| `test` \| `prod` \| `eval` | behavior switches (see §15.6) |
| `LOG_LEVEL` | `INFO` | stdlib logging level |
| `DATABASE_URL` | pooled Neon URL (app role) | runtime |
| `DATABASE_URL_DIRECT` | direct Neon URL (owner) | migrate/ingest/retention; falls back to `DATABASE_URL` when unset |
| `TEST_DATABASE_URL` | `postgresql://grounded:grounded@localhost:5433/grounded_test` | pytest only; must be a local `*_test` DB |
| `DB_POOL_MIN_SIZE`, `DB_POOL_MAX_SIZE`, `DB_POOL_TIMEOUT_S` | `1`, `5`, `5` | runtime connection pool (DB.md §2) |
| `GEMINI_API_KEY`, `GROQ_API_KEY`, `COHERE_API_KEY` | — | providers |
| `EMBEDDING_MODEL` / `EMBEDDING_DIM` | `gemini-embedding-001` / `768` | must match active index version |
| `GENERATOR_PROVIDERS` | `gemini,groq` | ordered router list (eval: `gemini`) |
| `GEMINI_MODEL`, `GROQ_MODEL`, `JUDGE_MODEL` | pinned IDs, verified at implementation time | no floating aliases |
| `GEMINI_THINKING_BUDGET` | `0` (or minimal) | thinking adds latency and billed output tokens |
| `LLM_TEMPERATURE` | `0` | same in prod and eval |
| `LLM_MAX_OUTPUT_TOKENS` | `800` | answer length cap |
| `RERANK_PROVIDER` | `none` \| `cohere` | feature flag |
| `RERANK_MODEL`, `RERANK_DAILY_CAP` | pinned, `30` | trial quota protection (~1k/month) |
| `K_DENSE`, `K_FTS`, `K_FUSED`, `K_CONTEXT`, `RRF_K` | `20`, `20`, `40`, `5`, `60` | retrieval config |
| `RATE_LIMIT_PER_MIN`, `RATE_LIMIT_PER_DAY` | `5`, `30` | per IP hash |
| `DAILY_LLM_BUDGET` | below provider free RPD | global cap; optional until Phase 5 |
| `ANSWER_CACHE_TTL_DAYS` | `30` | ≤ retention |
| `PROXY_SHARED_SECRET` | random 32+ bytes | FE→BE auth |
| `IP_HASH_SECRET` | random 32+ bytes | HMAC for IPs |
| `ALLOW_DIRECT_API` | `false` (true only in dev) | bypass proxy secret locally |
| `REQUEST_DEADLINE_S` | `25` | must be < proxy timeout |
| `CACHE_DIR` | `.cache` | SQLite caches, cloned corpus |
| `FASTAPI_REF` | pinned tag | corpus version |

Retrieval settings are grouped into a frozen `RetrievalConfig` whose canonical JSON is hashed
(`retrieval_config_hash`). The hash is logged per request, used in cache keys and recorded per eval run.

## 5. Ingestion pipeline (offline)

`uv run grounded ingest --ref <tag> [--database-url …] [--activate]`

```
fetch corpus → discover pages → resolve includes → parse headings → chunk [A] → hash → embed (cached) → store → verify → activate
```

### 5.1 Fetch
- Shallow clone `fastapi/fastapi` at `--ref` into `.cache/corpus/<ref>` (reuse if present). Record the resolved commit SHA.

### 5.2 Discover pages
- Include `docs/en/docs/**/*.md`.
- Exclude (initial list, refined in Phase 1): `release-notes.md`, external-links/sponsor pages, `docs/en/docs/js/**`, `docs/en/docs/css/**`, any non-`en` language dirs.
- URL mapping: `docs/en/docs/<path>.md` → `https://fastapi.tiangolo.com/<path>/`; `<dir>/index.md` → `https://fastapi.tiangolo.com/<dir>/`.

### 5.3 Resolve code includes
FastAPI docs pull code from `docs_src/`. Both syntaxes must be handled; verify against the pinned tag:
- Legacy: a fenced block containing `{!../../docs_src/<path>!}` (optionally with line selections).
- Current: `{* ../../docs_src/<path> hl[…] ln[…] *}`, which the site renders as tabs of variants (e.g. Python version / `Annotated` variants).

Rules:
- Replace the directive with a fenced ```` ```python ```` block containing the file contents.
- If several variants exist, pick **one preferred variant** (newest Python version, `Annotated` style if present). Log which variant was chosen.
- Strip rendering-only parameters (`hl[...]`, `ln[...]`). Respect line-range selection if present.
- A missing file is an ingest **error** (fail loudly), not a silent skip.

### 5.4 Headings and anchors
- Parse with markdown-it-py so headings inside code fences are ignored.
- FastAPI headings often carry explicit anchors: `## Create a task function { #create-a-task-function }`. Use the explicit anchor when present. Otherwise slugify like Python-Markdown's `toc` (lowercase, drop punctuation, spaces → `-`). Strip the `{ #… }` suffix from the visible heading text.
- Breadcrumb = navigation section (from the path, e.g. `Tutorial - User Guide`) + H1 + H2 + H3 of the chunk.

### 5.5 Chunking [A]
Contract for `chunk_document(doc: ParsedDocument, cfg: ChunkingConfig) -> list[Chunk]`:
- Primary boundaries: H2 and H3 sections. Text before the first H2 is the page-intro chunk (anchor `''`).
- A section longer than `max_tokens` (~450) is split recursively (paragraph → sentence), with `overlap_tokens` (~50) of overlap.
- A fenced code block is **atomic**: never split. A single code block larger than `max_tokens` becomes its own chunk (allowed to exceed, flagged in stats).
- Very small sections (< `min_tokens`, ~40) merge with the following sibling section under the same parent. The merged chunk keeps the first section's anchor and the common parent breadcrumb.
- Each chunk has `section_id = "<source_path>#<deepest anchor>"`, `anchor_path` (outermost → deepest), `breadcrumb`, `heading_level`, `ordinal`.
- Pure and deterministic: same input + config → identical output (tests depend on it).
- Token counts use the shared tokenizer helper (approximate; model tokenizers differ).

### 5.6 Hash, embed, cache
- `content_hash = sha256(breadcrumb_text + "\n\n" + content)`. The embedded text is exactly that string (heading context measurably helps retrieval).
- Embedding call: `task_type=RETRIEVAL_DOCUMENT` for chunks and `RETRIEVAL_QUERY` for questions, `output_dimensionality=768`.
- **L2-normalize** vectors ourselves. Truncated (Matryoshka) Gemini embeddings are not unit-length.
- Batch up to the provider's per-request maximum. Exponential backoff with jitter on 429/5xx. Respect RPM.
- Cache: SQLite `.cache/embeddings.sqlite`, key = `(model, dim, task_type, content_hash)`, value = float32 blob. A re-run with no content changes makes **zero** embedding calls.

### 5.7 Store, verify, activate
- Insert `index_versions(status='building')`, then documents and chunks with `COPY` or batched `executemany`, all in one transaction.
- Verify: chunk count > 0, no NULL embeddings, dims match, every golden-set `section_id` label resolves to ≥ 1 chunk (warn if not). Record `token_stats`.
- `--activate` switches the active version atomically (DB.md §7.3). `grounded index prune` retires/deletes old versions.

## 6. Query pipeline (online)

```mermaid
sequenceDiagram
    participant PX as Next.js proxy
    participant API as FastAPI /v1/ask
    participant C as Answer cache (PG)
    participant E as Gemini embed
    participant DB as Postgres
    participant R as Cohere (opt.)
    participant L as LLM router
    PX->>API: POST {question} + secret + client IP
    API->>API: validate, rate-limit (memory)
    API->>C: lookup(cache_key)
    alt hit
        C-->>API: response
    else miss
        API->>E: embed(question, RETRIEVAL_QUERY)
        API->>DB: hybrid SQL (dense ∪ FTS → RRF)
        opt rerank enabled & under cap
            API->>R: rerank(top K_FUSED) → top K_CONTEXT
        end
        API->>DB: reserve daily budget
        API->>L: generate(prompt, LLMAnswer schema)
        L-->>API: LLMAnswer (validated) + usage
        API->>API: citations map, confidence, assemble
        API->>C: store
    end
    API->>DB: insert request_logs
    API-->>PX: AskResponse
```

Stage timeouts (defaults): embed 3 s · retrieval 2 s · rerank 3 s · each LLM attempt 12 s · overall deadline `REQUEST_DEADLINE_S` = 25 s.
Every stage is wrapped in a `timing.stage("name")` context manager that fills the latency fields.

## 7. Retrieval

- **Dense:** exact cosine scan, top `K_DENSE` (DB.md §6.1).
- **Lexical [A]:** OR-semantics tsquery over weighted `tsv`, top `K_FTS` (DB.md §6.2).
- **Hybrid [A]:** RRF, `score = Σ 1/(RRF_K + rank)`, top `K_FUSED` unique chunks, deterministic tie-break (DB.md §6.3).
- Output type: `list[RetrievedChunk]` with `chunk_id, section_id, anchor_path, breadcrumb_text, url, content, token_count, content_hash, dense_rank, fts_rank, rrf_score, rerank_score | None`.
- Retrieval modes (for evals and ablations): `dense`, `fts`, `hybrid`, `hybrid_rerank`. The `no_rag` mode skips retrieval entirely.
- Context selection: the first `K_CONTEXT` chunks after (optional) rerank. If two selected chunks are adjacent parts of one split section, keep both (they are sent in document order within the source block).

## 8. Re-ranking (Phase 6)

- Adapter `CohereReranker.rerank(query, chunks, top_n) -> list[(chunk, relevance_score)]`, pinned model, timeout 3 s.
- Guarded by `RERANK_PROVIDER` and `daily_usage.rerank_calls < RERANK_DAILY_CAP` (atomic reserve, DB.md §7.1).
- Cache: SQLite in dev/CI/eval, key `sha256(model | query | ordered candidate content_hashes)`. In production the answer cache covers repeats.
- **Degradation:** any error, 429 or cap → use RRF order, set `rerank_used=false`, `rerank_error=<code>`. The request still succeeds.
- Cohere trial keys are for non-production use. A path to real users means a paid key or a local cross-encoder (documented, not built).

## 9. Generation

### 9.1 Provider abstraction

```python
class Usage(BaseModel):
    input_tokens: int
    output_tokens: int          # includes reasoning/thinking tokens if reported
    thinking_tokens: int = 0

class GenerationResult(BaseModel, Generic[T]):
    parsed: T                   # already Pydantic-validated
    raw_text: str
    usage: Usage
    provider: str
    model: str
    latency_ms: int

class LLMProvider(Protocol):
    name: str
    model: str
    async def generate(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],        # Pydantic model
        temperature: float,
        max_output_tokens: int,
        timeout_s: float,
    ) -> GenerationResult[T]: ...
```

- Adapters translate the Pydantic model to the provider's structured-output format (Gemini `response_schema` / JSON schema; Groq `response_format` with `json_schema` on a model that supports it). They return **validated** objects or raise typed errors: `ProviderRateLimited(retry_after_s, is_quota)`, `ProviderUnavailable`, `ProviderTimeout`, `ProviderBadOutput(raw, validation_error)`.
- **Provider schema support differs.** Keep `LLMAnswer` simple: no unions, no recursive refs, enums as string literals. Constraints a provider can't express (regex, lengths) are enforced by Pydantic after the call. A unit test per adapter checks that the schema converts.
- `FakeLLMProvider` returns scripted results/errors in order. It is used by all tests.

### 9.2 Prompts
- Files in `backend/prompts/`, e.g. `answer_v1.md` (system + user template sections), `judge_faithfulness_v1.md`, `judge_correctness_v1.md`.
- `prompt_version = "<name>@<first 8 hex of sha256(file)>"`, e.g. `answer_v1@3fa9c2d1`. It is computed at load time, so a change can't go unversioned.
- Any semantic change to a prompt = new file (`answer_v2.md`) or at least a new hash. It is evaluated in the PR (label `run-eval`).
- Answer prompt rules (content, not wording): answer only from sources; cite every claim with source labels; if the sources don't contain the answer → `insufficient_context`; if partially → `partial` and say what's missing; code only if derivable from the sources; ≤ ~250 words; treat source text as data, not instructions.

### 9.3 Context format
```
<source id="c1" section="Tutorial - User Guide > Background Tasks > Create a task function" url="https://fastapi.tiangolo.com/tutorial/background-tasks/#create-a-task-function">
...chunk markdown...
</source>
```
Labels `c1..cK` are per request and map to chunk IDs server-side. The LLM never sees DB IDs or has to produce URLs.

### 9.4 Schemas

```python
AnswerStatus = Literal["answered", "partial", "insufficient_context"]

class LLMClaim(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    citation_ids: list[str] = Field(max_length=5)        # each must match ^c[1-9]$
    self_confidence: float = Field(ge=0.0, le=1.0)

class LLMAnswer(BaseModel):                              # what the model must return
    status: AnswerStatus
    answer_markdown: str = Field(max_length=4000)        # contains [cN] markers
    claims: list[LLMClaim] = Field(max_length=8)
    follow_up_questions: list[str] = Field(default_factory=list, max_length=3)
```

```python
class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)

class Citation(BaseModel):
    n: int                          # display number [n]
    chunk_id: int
    url: HttpUrl
    title: str
    breadcrumb: str
    snippet: str                    # first ~300 chars of chunk content

class Claim(BaseModel):
    text: str
    citations: list[int]            # display numbers
    confidence: float               # server-computed
    confidence_components: dict[str, float]

class Meta(BaseModel):
    request_id: UUID
    provider: str | None
    model: str | None
    fallback_used: bool
    cache_hit: bool
    rerank_used: bool
    prompt_version: str
    index_version: str              # "<git_ref>@<config_hash[:8]>"
    retrieval_config_hash: str
    latency_ms: dict[str, int]      # total, embed, retrieval, rerank, llm
    tokens: dict[str, int]          # input, output
    shadow_cost_usd: float

class AskResponse(BaseModel):
    status: AnswerStatus
    answer_markdown: str            # markers rewritten to [n]
    claims: list[Claim]
    citations: list[Citation]       # ordered by n
    follow_up_questions: list[str]
    min_confidence: float | None
    meta: Meta
```

### 9.5 Validation, retry and fallback flow
1. Call provider → Pydantic validation in the adapter.
2. Post-validation semantic checks (in `citations.py`):
   - status `answered|partial` with **zero valid citations** → treat as bad output.
   - status `insufficient_context` → `claims` must be empty (non-empty claims are dropped and counted).
3. On bad output: **one retry on the same provider**, appending the validation error to the user message ("Your previous output was invalid because …"). Then go to the next provider in the router (that counts as fallback).
4. Rate limit / 5xx / timeout → router behavior (§10).
5. All paths exhausted → `502 validation_failed` or `503 provider_unavailable`, logged with outcome.

### 9.6 Citation mapping
- Valid label = present in this request's label map. Invalid labels are removed from claims and from `answer_markdown`, and counted (`invalid_citation_count`). Invalid labels are a tracked metric; they are *not* silently fixed.
- Display numbering: `[cK]` markers in `answer_markdown` → `[n]` by order of first appearance. Claims' citation lists are mapped to the same `n`.
- Each distinct cited chunk becomes one `Citation` (URL with anchor, H1 title, breadcrumb, snippet).

### 9.7 Refusal
- `insufficient_context` answers are short, have no claims or citations, and may include `follow_up_questions` pointing to what *is* covered.
- There is no retrieval-score threshold short-circuit in MVP. It is a possible later optimization, and eval data will show whether it helps.

### 9.8 Confidence heuristic [A]
Contract for `score_claims(claims, label_map, retrieved) -> list[ClaimScore]`:
- Output per claim: `confidence ∈ [0,1]` plus `components: dict[str, float]` (returned in the API for transparency).
- Available signals: rerank relevance score of cited chunks (when rerank on), RRF score and ranks (dense/FTS agreement), number of valid citations, LLM `self_confidence`, and optionally lexical overlap between claim and cited chunk text.
- Invariants: deterministic; monotonic non-decreasing in retrieval support; **a claim with no valid citation is capped at 0.2**; `self_confidence` alone can't push a claim above 0.6 (weight it low, since self-reports are poorly calibrated).
- Weights live in settings (not magic numbers), documented in the README.
- Calibration (Phase 8): bucket claims (low < 0.5, mid 0.5–0.8, high > 0.8) and compare with the judge's "supported" rate per bucket. A useful heuristic is monotonic across buckets. Report the table even if it isn't.
- `min_confidence` (answer level) = min over claims. The UI flags claims below 0.5.

## 10. Provider router, fallback and circuit breaker [A]

Contract for `ProviderRouter.generate(...) -> GenerationResult` over an ordered provider list (`GENERATOR_PROVIDERS`):

| Event from provider P | Router action |
|---|---|
| Success | return; reset P's failure count |
| `ProviderBadOutput` | 1 retry on P with error feedback (§9.5), then next provider |
| `ProviderRateLimited(retry_after)` (per-minute) | open P's breaker for `retry_after` (default 60 s if absent); go to next provider immediately |
| `ProviderRateLimited(is_quota=True)` (daily quota) | open P's breaker until the provider's quota reset (if not derivable: 15 min, re-probed via half-open) |
| `ProviderUnavailable` / `ProviderTimeout` | count failure; 3 consecutive within 60 s → open 30 s; go to next provider |
| P's breaker open | skip P without calling |
| Breaker time elapsed | half-open: allow exactly one trial request; success → closed, failure → open again |
| All providers open/failed | raise `AllProvidersUnavailable(retry_after=min remaining)` → HTTP 503 with `Retry-After` |

Requirements:
- Breaker state is in memory, per provider, protected for concurrent async access.
- No sleeping inside the request path: fallback is immediate, and the router never waits out a `Retry-After`.
- The router reports `fallback_used`, `provider`, `model` and the list of attempts (for logs and tests).
- **Eval mode:** provider list is only the primary. Fallback disabled, because the fallback provider is also the judge.
- Tests (with fakes): 429 → fallback + breaker open; breaker skip; half-open recovery; timeouts ×3 → open; bad output → retry → fallback; all open → 503.

## 11. Caching

| Cache | Store | Key | Used in | TTL |
|---|---|---|---|---|
| Answer cache | Postgres `answer_cache` | sha256(normalized question \| prompt_version \| index_version_id \| retrieval_config_hash \| generator_model) | prod/dev (off in eval) | 30 days |
| Query embedding | in-memory LRU (prod), SQLite (dev/CI/eval) | (model, dim, RETRIEVAL_QUERY, sha256(question)) | all | process lifetime / persistent |
| Chunk embedding | SQLite `.cache/embeddings.sqlite` | (model, dim, RETRIEVAL_DOCUMENT, content_hash) | ingest | persistent |
| Rerank | SQLite `.cache/rerank.sqlite` | sha256(model \| query \| candidate hashes) | dev/CI/eval | persistent |
| Eval LLM responses | SQLite `.cache/llm_eval.sqlite` | sha256(provider \| model \| temperature \| system \| user \| schema hash) | eval only | persistent |

Question normalization: Unicode NFKC → lowercase → collapse whitespace → strip trailing punctuation.
The eval LLM cache is keyed by the *full prompt*. Any prompt or context change misses the cache, so regressions are still detected, while identical cases are free and deterministic on re-runs.

## 12. Abuse protection and API security

- **Proxy secret:** `/v1/*` requires `X-Proxy-Secret == PROXY_SHARED_SECRET` (constant-time compare). Without it → 401, unless `ALLOW_DIRECT_API=true` (dev only). `/healthz` and `/readyz` are public.
- **Client IP:** trusted from `X-Client-IP` **only** when the proxy secret is valid; otherwise the socket peer. The IP is immediately HMAC-hashed (`IP_HASH_SECRET`). The raw IP is never logged or stored.
- **Rate limit:** in-memory sliding window per IP hash (`RATE_LIMIT_PER_MIN`, `RATE_LIMIT_PER_DAY`). Single instance, so memory is correct. A reset on restart is acceptable. Response: 429 + `Retry-After`, outcome `rate_limited`.
- **Global budget:** atomic reserve in `daily_usage` before each LLM call (DB.md §7.1). Exhausted → HTTP 503 with code `budget_exhausted` (the UI shows a friendly "demo budget reached" state), outcome logged. Cache hits don't consume budget.
- **Input limits:** question 3–500 chars after trimming; reject control characters; body size limit 4 KB.
- **CORS:** not needed (browser talks only to Vercel). Backend CORS is locked to nothing.
- **Output rendering:** the frontend renders Markdown **without** raw HTML (no `rehype-raw`). Links open in a new tab with `rel="noopener noreferrer"`. Citation URLs come from the DB, never from model text.

## 13. HTTP API

Base path `/v1`. JSON only. Errors share one shape:

```json
{ "error": { "code": "rate_limited", "message": "Too many requests. Try again in 20s.", "retry_after_s": 20 }, "request_id": "…" }
```

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/healthz` | public | process up (used by keepalive) |
| GET | `/readyz` | public | DB reachable + active index loaded (Phases 0–2: DB reachable + schema present; reports `active_index_version` but doesn't require it until Phase 3) |
| POST | `/v1/ask` | proxy secret | body `AskRequest` → `AskResponse` |
| GET | `/v1/metrics/summary?window=7d` | proxy secret | aggregates for the dashboard (no question text) |

| Code | HTTP | When |
|---|---|---|
| `bad_request` | 400/422 | invalid input |
| `unauthorized` | 401 | missing/invalid proxy secret |
| `rate_limited` | 429 | per-IP limit |
| `budget_exhausted` | 503 | global daily budget |
| `validation_failed` | 502 | model output invalid after retry + fallback |
| `provider_unavailable` | 503 | all providers down/open |
| `deadline_exceeded` | 504 | overall deadline |
| `internal_error` | 500 | anything else (logged with request_id) |

OpenAPI docs (`/docs`) stay enabled. The API contract is itself part of the portfolio.

## 14. Observability and cost

- **Structured logs:** JSON lines via stdlib `logging` with a JSON formatter. Every line has `request_id`. Question text is never logged to stdout (only `question_hash`).
- **Request log row:** one per `/v1/ask` (all outcomes), fields in DB.md §4 `request_logs`.
- **Stage timing:** `timing.stage()` context manager on the monotonic clock. Totals include cache lookup and logging.
- **Shadow cost:** `backend/pricing.toml`:
  ```toml
  as_of = "YYYY-MM-DD"          # date prices were checked
  source = "<provider pricing page URLs>"
  [models."<gemini model id>"]  input_per_mtok = 0.0  output_per_mtok = 0.0   # fill from pricing page
  [models."<groq model id>"]    input_per_mtok = 0.0  output_per_mtok = 0.0
  [embeddings."<embedding id>"] input_per_mtok = 0.0
  [rerank."<rerank id>"]        per_1k_searches = 0.0
  ```
  `cost = in_tok/1e6·in_price + out_tok/1e6·out_price + embed_tok/1e6·embed_price + rerank_calls/1000·rerank_price`.
  Prices are **never typed from memory**. They are copied from the provider pricing page with the date. Thinking tokens count as output tokens.
- **Dashboard (Phase 8):** `/v1/metrics/summary` reads the DB views (DB.md §8) + latest `eval_runs` per config → Next.js `/metrics` (revalidate 60 s).

## 15. Evaluation architecture

### 15.1 Golden set
`eval/golden/golden_set.v1.jsonl`, one object per line:

```json
{
  "id": "q017",
  "question": "How can I run a function after returning a response?",
  "type": "how_to",
  "answerable": true,
  "reference_answer": "Use BackgroundTasks: declare a parameter of type BackgroundTasks in the path operation and call background_tasks.add_task(func, *args)...",
  "relevant_sections": [
    {"section": "docs/en/docs/tutorial/background-tasks.md#using-backgroundtasks", "grade": 2},
    {"section": "docs/en/docs/tutorial/background-tasks.md#create-a-task-function", "grade": 1}
  ],
  "notes": "multi-step; code expected"
}
```
- `type ∈ {factual, how_to, code, multi_section, unanswerable}`. Unanswerable items have `relevant_sections: []` and a reference answer describing why.
- Grades: **2** = contains the answer, **1** = useful context.
- **Section matching rule:** a retrieved chunk matches label `path#anchor` iff same `source_path` and `anchor ∈ chunk.anchor_path` (so an H2 label also matches its H3 sub-chunks). A label without an anchor matches the whole page.
- Changes to the golden set create a new version file (`v2`) and require new baselines. Items are never edited in place after baselines exist.
- `eval/golden/README.md` holds the labeling guide.
- `grounded golden draft --n 50` asks an LLM to propose candidate questions from random sections. The Author curates.

### 15.2 Retrieval eval (Python) [A: metrics]
`uv run grounded eval retrieval --config hybrid [--config dense ...] --out eval/results/…json`
- Per answerable question: run retrieval mode → ranked chunks → map to section labels (dedupe by first occurrence) → metrics.
- **Recall@k** = |grade-2 labels matched in top-k| / |grade-2 labels|.
- **MRR** = 1 / rank of the first grade-2 match (0 if none within `K_FUSED`).
- **nDCG@k** with gain `2^grade − 1` over deduped section ranking; ideal DCG from labels.
- Report means with `n`, and per-question rows for diffing.
- Deterministic given caches. No LLM calls (only query embeddings, cached).

### 15.3 Generation eval (promptfoo)
- `eval/promptfoo/promptfooconfig.yaml`:
  - `prompts`: passthrough `{{question}}` (the real prompt lives in the backend).
  - `providers`: `file://provider.py` instances labeled by config (`no_rag`, `hybrid`, `hybrid_rerank`), each passing `config: {mode: …}`.
  - `tests`: generated from the golden set by a Python loader (`tests_loader.py`). This uses promptfoo's Python test-generator support, or a pre-step writing `tests.generated.yaml`, whichever the pinned promptfoo version supports.
- `provider.py` (`call_api(prompt, options, context)`): runs the backend pipeline **in-process** with `APP_ENV=eval` and returns `output` = `AskResponse` JSON, `tokenUsage`, and `metadata` (retrieved chunks with text, latencies, cost).
- Assertions (`asserts.py`, returning `{pass, score, reason}`):

| Metric | How | Deterministic |
|---|---|---|
| Schema first-try validity | `meta` validation_retries == 0 | ✅ |
| Citation validity | invalid_citation_count == 0 (pre-filter count) | ✅ |
| Refusal correctness | answerable ↔ status ≠ `insufficient_context` | ✅ |
| Citation precision | share of cited chunks matching any labelled section (grade ≥ 1) | ✅ |
| Faithfulness | judge per claim: SUPPORTED / NOT_SUPPORTED given claim + cited chunk texts; score = supported / claims | judge |
| Answer correctness | `llm-rubric` with judge provider vs `reference_answer`, 0 / 0.5 / 1 | judge |

- `no_rag` config: same schema and prompt family without sources. Faithfulness is N/A there; correctness and refusal are comparable.
- Latency p50/p95 and shadow cost per 1k questions are computed from provider metadata (warm, excluding cold start).

### 15.4 Judge
- Provider: Groq (different from generator), pinned `JUDGE_MODEL`, temperature 0, prompts in `backend/prompts/judge_*_v1.md` with 2–3 worked examples each.
- The judge returns structured output (`{verdict, reason}`) through the same adapter layer.
- **Agreement check:** export ≥ 10 verdicts to `eval/judge_agreement/v1.csv`. The Author labels them by hand. Agreement % (and Cohen's κ if meaningful) goes to the README.

### 15.5 Aggregation and gate [A]
`uv run grounded eval gate --suite {retrieval|generation} --results … --baseline eval/baselines/….json`
- Output: `pass | fail | inconclusive`, a Markdown table (metric, baseline, current, Δ, threshold, ✅/❌) and exit code (0 pass/inconclusive, 1 fail).
- **Inconclusive:** if > 20% of cases errored for provider reasons (429/quota/timeout) → `inconclusive`. Metrics are shown with `n` but not gated. The PR comment says so explicitly.
- Otherwise metrics are computed over non-errored cases, and `n` is reported.
- Initial thresholds (tuned after first baselines; stored in the baseline file, not in code):

| Suite | Metric | Rule |
|---|---|---|
| retrieval | Recall@5, MRR (hybrid) | ≥ baseline − 0.04 (≈ one question) |
| retrieval | nDCG@5 | ≥ baseline − 0.04 |
| generation | faithfulness | ≥ max(0.85, baseline − 0.05) |
| generation | answer correctness | ≥ baseline − 0.08 |
| generation | refusal accuracy | ≥ baseline − 0.07 (≈ two questions of 30) |
| generation | schema first-try validity | ≥ 0.95 |

- promptfoo's own exit code is ignored. The gate script is the single source of truth.

### 15.6 Eval mode (`APP_ENV=eval`)
Fallback off · temperature 0 · answer cache off · rate limit and budget off · LLM/rerank/embedding SQLite caches on · concurrency 1 (`-j 1`) · backoff honoring `Retry-After` (bounded total wait) · `request_logs.source='eval'` (CI DB only).

### 15.7 Baselines and history
- `eval/baselines/retrieval.json`, `eval/baselines/generation.json`: per config → metrics, `n`, thresholds, golden set version, prompt version, model IDs, index config hash, git SHA, date.
- Updated only by an explicit PR titled `eval: update baseline (<reason>)`, with the before/after table in the description.
- Runs on `main` also insert into production `eval_runs` (owner connection via CI secret) for the dashboard.

## 16. Testing strategy

| Level | What | Tools |
|---|---|---|
| Unit | chunker, include resolution, anchors/slugify, RRF math, metrics, citation mapping, confidence invariants, cache keys, normalization, cost calc, rate limiter, breaker state machine, gate logic | pytest (pure functions, table-driven) |
| Adapter | schema conversion per provider; response parsing from recorded **fixtures** (JSON files), error mapping (429 → `ProviderRateLimited`) | pytest + fixtures, no network |
| Service | `/v1/ask` via `httpx.AsyncClient(app)` with `FakeLLMProvider`, fake embedder, real DB | pytest-asyncio |
| Integration | migrations apply cleanly; dense/lexical/hybrid SQL on a small fixture index (~30 hand-made chunks with known vectors); budget atomicity under concurrency | pgvector service container |
| Eval | retrieval eval + promptfoo (separate from pytest) | see §15 |

Rules: **no real network calls in pytest.** A socket-blocking fixture fails any test that tries. Fixtures are small and committed. Every bug fix comes with a failing test first.

## 17. CI/CD workflows

| Workflow | Trigger | Jobs |
|---|---|---|
| `ci.yml` | PR, push `main` | **backend**: `uv sync --frozen`, ruff check/format --check, pyright, pytest (pgvector service). **frontend**: `npm ci`, lint, typecheck, build. **retrieval-eval**: restore `.cache` (corpus + embeddings) → migrate → ingest pinned ref into service DB → retrieval eval (all configs) → gate vs `eval/baselines/retrieval.json` → job summary |
| `eval.yml` | PR `labeled`/`synchronize` with label `run-eval`; push `main` | same DB setup → `npx promptfoo@<ver> eval -j 1` → gate → PR comment (create/update by marker `<!-- grounded-eval -->`) → upload HTML/JSON report artifact → on `main`: insert `eval_runs` |
| `deploy-backend.yml` | push `main` touching `backend/**` (after `ci.yml` success) | build context = `backend/` + `infra/hf-space/README.md` → push to HF Space repo via `huggingface_hub` with `HF_TOKEN` |
| `ingest.yml` | `workflow_dispatch(ref, activate)` | ingest into Neon with `DATABASE_URL_DIRECT`; prints index stats |
| `keepalive.yml` | daily cron + `workflow_dispatch` | GET backend `/healthz` (keeps Space < 48 h idle) + retention SQL (DB.md §9) |

Notes:
- Cron can't express "every 26 h", so it runs daily (still under the 48 h Space sleep window).
- GitHub disables scheduled workflows after 60 days without repo activity. Documented in README limitations.
- Secrets: `GEMINI_API_KEY`, `GROQ_API_KEY`, `COHERE_API_KEY`, `DATABASE_URL_DIRECT`, `HF_TOKEN`, `BACKEND_URL`. PR workflows only run for same-repo branches, so secrets are available.
- `.cache` key: `hash(FASTAPI_REF, chunking config, EMBEDDING_MODEL, EMBEDDING_DIM)` with a restore-key fallback. GitHub evicts caches unused for 7 days, which means one full re-embed (~3k texts), acceptable.

## 18. Deployment

### Backend — Hugging Face Space (Docker)
- Free CPU Space: sleeps after ~48 h without traffic (verify current policy). The daily keepalive prevents that.
- `Dockerfile`: `python:3.12-slim`, install uv, `uv sync --frozen --no-dev`, copy `src/`, `migrations/`, `prompts/`, `pricing.toml`. Run as **uid 1000** (Spaces requirement). `EXPOSE 7860`. `CMD uvicorn grounded.main:app --host 0.0.0.0 --port 7860 --proxy-headers`.
- Space metadata README (`infra/hf-space/README.md`): `sdk: docker`, `app_port: 7860`. It is pushed to the Space repo root; the project README stays in GitHub.
- Secrets in Space settings: DB URL (app role, pooled), provider keys, `PROXY_SHARED_SECRET`, `IP_HASH_SECRET`.
- Startup: open pool, load the active index version, warm providers lazily. `/readyz` reflects readiness.
- The disk is ephemeral. Nothing persistent lives there.

### Frontend — Vercel
- Project root directory `frontend/`. Env: `BACKEND_URL`, `PROXY_SHARED_SECRET` (server-only, never `NEXT_PUBLIC_`).
- Route Handlers `app/api/ask/route.ts`, `app/api/metrics/route.ts`: forward to backend with the secret and `X-Client-IP` (from the platform's forwarded-for header), and a timeout slightly above `REQUEST_DEADLINE_S`.
- **Function duration limit:** set `export const maxDuration` in the ask route and verify the current Hobby-plan maximum. It must exceed the backend deadline plus a possible Space wake-up, or the UI must show a retry state.

### Database — Neon
- Project `grounded`, Postgres 17, region closest to the Space. Enable `vector`. Create role `app` (DB.md §5).
- Migrations and ingest via `workflow_dispatch` or locally with the direct URL.

## 19. Frontend

- Pages: `/` (ask), `/metrics` (dashboard, Phase 8), `/about` optional (links to README sections).
- Ask page states: idle · loading (after 5 s: "Waking up the free-tier backend, this can take up to a minute") · answered/partial · insufficient_context · rate_limited (with countdown) · budget_exhausted · error (with request_id).
- Answer rendering: react-markdown + remark-gfm + code highlighting. `[n]` markers become buttons that focus/open the matching source card. Source cards show breadcrumb, snippet and an external link. Claims with confidence < 0.5 get a subtle indicator with a tooltip that says it is heuristic.
- Debug row (collapsed): stage latencies, tokens, provider/model, fallback/cache/rerank flags, shadow cost, prompt/index version.
- Privacy notice under the input: free AI APIs are used; do not enter personal data.
- Types: `frontend/lib/types.ts` mirrors `AskResponse`. It is kept in sync by generating from the backend OpenAPI schema (`openapi-typescript`) as an npm script.
- Accessibility: all interactive citation elements are keyboard reachable. Nothing is hover-only.

## 20. Known failure modes and limitations (seed list)

This list is expanded with observed examples in Phase 9. The README shows the final version.

| Failure mode | Where | Mitigation / how we see it |
|---|---|---|
| Lexical query too strict/loose | retrieval | OR-tsquery; ablation rows show contribution |
| Relevant info split across sections not retrieved together | retrieval | K_CONTEXT=5, multi_section golden items measure it |
| Model cites a chunk that doesn't support the claim | generation | faithfulness per claim; confidence components |
| Model answers from prior knowledge despite weak context | generation | refusal metric, `no_rag` comparison, prompt rules |
| Invalid/missing citations | generation | server validation, retry, tracked rate |
| Provider schema drift / unsupported schema features | adapters | simple schema, adapter tests, validation retry |
| Quota exhaustion / 429 storms | providers | breaker, fallback, budget, cache, inconclusive evals |
| Cold start latency | infra | keepalive, UI state, documented numbers |
| Stale corpus vs live docs | data | pinned tag shown in UI/meta; deliberate re-index |
| Judge disagreement with humans | eval | agreement measured and published |
| Small golden set noise | eval | n reported, tolerances ≥ 1 question |
| FTS is not true BM25 | retrieval | stated; ablation shows the effect |
