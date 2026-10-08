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
    subgraph BE["Vercel Function (FastAPI)"]
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
| Eval | promptfoo `0.123.1` (`npx promptfoo@0.123.1`, Node `>=22.22.0`; D49) | Python provider + Python assertions + Python test generator |
| Frontend | Next.js (App Router), TypeScript strict, Tailwind, shadcn/ui, react-markdown + remark-gfm + syntax highlighting | npm |
| DB | PostgreSQL 17 + pgvector (Neon / Docker) | see DB.md |
| Hosting | Vercel: frontend (Next.js) and backend (FastAPI as a Vercel Function) | free tiers |
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
│   ├── pyproject.toml  uv.lock
│   ├── migrations/                # 0001_init.sql, ...
│   ├── prompts/                   # answer_v1.md, judge_faithfulness_v1.md, judge_correctness_v1.md
│   ├── pricing.toml               # dated list prices for shadow cost
│   ├── src/grounded/
│   │   ├── main.py                # app factory, lifespan
│   │   ├── runtime.py             # open_runtime: pool, query embedder, provider and AskPipeline, shared by the API and the CLI; open_judge: the judge of an eval run (§15.4)
│   │   ├── settings.py            # pydantic-settings
│   │   ├── cli.py                 # Typer: migrate, ingest, index, eval, ask, golden
│   │   ├── api/                   # routes_ask.py, routes_metrics.py, routes_health.py, deps.py, errors.py, security.py
│   │   ├── schemas/               # llm.py (LLMAnswer), api.py (AskRequest/AskResponse), eval.py (GoldenItem, retrieval eval results/baseline), generation_eval.py (the normalized promptfoo results and the generation baseline row, §15.3, §15.7), judge.py (judge verdicts and judgments, §15.4)
│   │   ├── ingest/                # types.py, corpus.py, markdown.py, includes.py, tokens.py, chunker.py [A], embed.py, pipeline.py
│   │   ├── retrieval/             # index.py (active version), dense.py, lexical.py [A], hybrid.py [A], rerank.py, config.py, types.py
│   │   ├── generation/
│   │   │   ├── providers/         # base.py (Protocol), gemini.py, groq.py, fake.py, eval_wrappers.py (eval LLM cache + 429/5xx backoff, §15.6)
│   │   │   ├── router.py          # fallback + circuit breaker [A]
│   │   │   ├── prompts.py         # load + version/hash
│   │   │   ├── context.py         # c1..c5 labeling, source blocks
│   │   │   ├── citations.py       # validation, mapping, marker rewrite
│   │   │   ├── confidence.py      # heuristic [A]
│   │   │   ├── params.py          # GenerationParams: the settings that change an answer (call + cache key)
│   │   │   └── pipeline.py        # orchestrates the /ask flow
│   │   ├── evals/                 # metrics.py [A] (Recall@k, MRR, nDCG@k), golden.py, retrieval_runner.py, ask_batch.py, gate.py [A], generation_results.py (promptfoo's JSON → `GenerationRun`, §15.3), generation_baseline.py (`grounded eval baseline`, §15.7), report.py (`grounded eval report`, §15.7), eval_record.py (`grounded eval record`, §15.7), judge.py, promptfoo_provider.py / promptfoo_asserts.py / promptfoo_tests.py (the code behind eval/promptfoo/, §15.3)
│   │   ├── infra/                 # db.py, migrations.py (runner), kvcache.py (SQLite), provider_errors.py, gemini_errors.py, answer_cache.py, ratelimit.py, budget.py, timing.py, hashing.py, logging.py, event_loop.py
│   │   └── observability/         # request_log.py, cost.py
│   └── tests/                     # unit/, integration/, conftest.py (fixtures), support.py (helpers), provider_rigs.py (adapter rigs), fixtures/
├── eval/
│   ├── golden/                    # golden_set.v1.jsonl, README.md (labeling guide)
│   ├── promptfoo/                 # promptfooconfig.yaml and the shims provider.py, asserts.py, tests_loader.py (§15.3)
│   ├── baselines/                 # retrieval.json, generation.json  (committed, changed only by explicit PR)
│   └── results/                   # gitignored run outputs
├── frontend/                      # Next.js app (app/, components/, lib/)
├── infra/
│   └── docker-compose.yml         # pgvector for dev
└── .github/workflows/             # ci.yml, eval.yml, ingest.yml, housekeeping.yml
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
| `EMBEDDING_MODEL` / `EMBEDDING_DIM` | default `gemini-embedding-001` / `768` | pinned defaults (model verified 2026-09-26); must match the active index version, a different value is a new index version; blank is rejected |
| `EMBEDDING_BATCH_SIZE`, `EMBEDDING_RPM`, `EMBEDDING_TPM` | `100` (max 100, ≤ RPM), `100`, `30000` | batching and pacing, defaults = free tier; RPM counts texts (§5.6) |
| `EMBEDDING_MAX_INPUT_TOKENS` | `2048` | longer texts fail before any call; must be ≤ `EMBEDDING_TPM` |
| `EMBEDDING_MAX_RETRIES`, `EMBEDDING_TIMEOUT_S` | `5`, `30.0` | per batch, on 429 / 5xx / timeout |
| `EMBEDDING_MAX_RETRY_WAIT_S` | `60` | a server-given `Retry-After` above this stops the run instead of waiting (§5.6) |
| `GENERATOR_PROVIDERS` | `gemini,groq` | ordered router list. `APP_ENV=eval` takes exactly one entry (`gemini`) and refuses `groq`, the judge's provider, at startup (§15.6). Until the router (Phase 7) the first entry is the only provider used. The extra value `fake` selects the canned stub provider for local runs (`StubLLMProvider`, and `StubJudgeProvider` for the judge, §15.3); it is refused with `APP_ENV=prod` |
| `GEMINI_MODEL`, `GROQ_MODEL`, `JUDGE_MODEL` | pinned IDs, verified at implementation time | no floating aliases. `GEMINI_MODEL` is `gemini-3.5-flash-lite` (D46, verified 2026-10-06). `GROQ_MODEL` and `JUDGE_MODEL` are both `openai/gpt-oss-120b` (D48, verified 2026-10-08); they have no default in `Settings`, like `GEMINI_MODEL` (CI sets them as literals, D51). Required, with `GEMINI_API_KEY`, when `gemini` is the first provider; a blank value is an error at startup |
| `GEMINI_THINKING_LEVEL` | `minimal` (D46) | `minimal` \| `low` \| `medium` \| `high`; thinking adds latency and billed output tokens. Gemini 3.x models are controlled by a level, not a token budget (renamed from `GEMINI_THINKING_BUDGET` in 3.08). Which levels a model accepts differs (`gemini-3.7-flash` and `gemini-3.8-flash` reject `minimal`), so the API validates the pair |
| `GROQ_REASONING_EFFORT` | `low` | `low` \| `medium` \| `high` (what `openai/gpt-oss-120b` takes; `none` and `default` are for Qwen only). Reasoning tokens are billed as output, count inside `LLM_MAX_OUTPUT_TOKENS` and use the free plan's 8K tokens per minute, so the default is the smallest; the judge agreement (4.11) shows whether it costs verdict quality (§9.1, PRD §12) |
| `LLM_TEMPERATURE` | `0` | same in prod and eval; `APP_ENV=eval` refuses any other value at startup |
| `EVAL_MAX_TOTAL_WAIT_S` | `120.0` | eval only: the most seconds one LLM call (generator or judge) may spend waiting out per-minute 429s (`Retry-After`) and transient 5xx together; a longer wait is not taken and the call fails as provider-errored (§15.6). The request path never waits |
| `EVAL_UNAVAILABLE_RETRIES`, `EVAL_UNAVAILABLE_WAIT_S` | `4`, `5.0` | eval only: how often a transient 5xx (`ProviderUnavailable`, e.g. Gemini's 503 "high demand") is retried, and the seconds before the first retry, doubling after each (5, 10, 20, 40 = 75 s), inside `EVAL_MAX_TOTAL_WAIT_S` (§15.6). `0` retries turns it off; at most `10`; the wait is not negative |
| `JUDGE_MAX_OUTPUT_TOKENS` | `800` | eval only: the output cap of one judge call (§15.4); Groq's reasoning tokens count inside it. The judge's own setting rather than `LLM_MAX_OUTPUT_TOKENS`: the cap is part of the eval cache key (§11), so sharing it would drop the cached verdicts, which cost scarce Groq quota, whenever the generator's cap moved |
| `LLM_MAX_OUTPUT_TOKENS` | `800` | answer length cap |
| `LLM_TIMEOUT_S` | `12.0` | each generation attempt (§6) |
| `EVAL_LLM_TIMEOUT_S` | `40.0` | eval only: the timeout of one generator or judge call when `APP_ENV=eval`, in place of `LLM_TIMEOUT_S` (`Settings.call_timeout_s` chooses, for the pipeline and the judge). The request path's 12 s is a user-facing deadline that an eval does not have: in 4.05 two of six real Gemini calls hit it (typical call: 2-3 s), and a timeout counts as a provider-side error toward `inconclusive` (§15.5). The timeout is not part of the eval cache key (§11), so changing it invalidates nothing |
| `RERANK_PROVIDER` | `none` \| `cohere` | feature flag |
| `RERANK_MODEL`, `RERANK_DAILY_CAP` | pinned, `30` | trial quota protection (~1k/month) |
| `CHUNK_MAX_TOKENS`, `CHUNK_OVERLAP_TOKENS`, `CHUNK_MIN_TOKENS` | `450`, `50`, `40` | chunking config (§5.5); min < max, overlap < max |
| `TOKENIZER_ENCODING` | `o200k_base` | tiktoken encoding for chunk sizing (approximate counts) |
| `K_DENSE`, `K_FTS`, `K_FUSED`, `K_CONTEXT`, `RRF_K` | `20`, `20`, `40`, `5`, `60` | retrieval config; `K_CONTEXT` ≤ 9 (the labels `c1..c9`) |
| `ACTIVE_INDEX_TTL_S` | `300.0` | how long the request path caches the active index version (DB.md §7.3); not part of `RetrievalConfig` |
| `CONFIDENCE_W_RETRIEVAL`, `CONFIDENCE_W_AGREEMENT`, `CONFIDENCE_W_CITATIONS`, `CONFIDENCE_W_SELF`, `CONFIDENCE_W_RERANK` | `0.40`, `0.25`, `0.20`, `0.15`, `0.0` | relative weights of the confidence signals (§9.8), each in `[0, 1]`, not all zero; `W_SELF` at most `0.6`, and `Settings` refuses weights that let the self-report carry a weakly supported claim above `0.6` (§9.8); `W_RERANK` stays `0` until Phase 6 measures the rerank lift. Proposed in 3.09, kept in 3.10 |
| `CONFIDENCE_UNCITED_CAP` | `0.2` | most a claim without a valid citation can score; at most `0.2` (§9.8 invariant) |
| `QUERY_EMBEDDING_CACHE_SIZE` | `256` | prod only: size of the in-memory LRU of question vectors (§11); dev/CI/eval use SQLite |
| `QUERY_EMBEDDING_TIMEOUT_S` | `3.0` | the embed stage of `/v1/ask` (§6): one attempt, no retry, no pacing; the `EMBEDDING_*` retry settings apply to ingest and the evals only |
| `RATE_LIMIT_PER_MIN`, `RATE_LIMIT_PER_DAY` | `5`, `30` | per IP hash |
| `DAILY_LLM_BUDGET` | below provider free RPD | global cap; optional until Phase 5 |
| `ANSWER_CACHE_TTL_DAYS` | `30` | ≤ retention |
| `PROXY_SHARED_SECRET` | random 32+ bytes | FE→BE auth |
| `IP_HASH_SECRET` | random 32+ bytes | HMAC for IPs |
| `ALLOW_DIRECT_API` | `false` (true only in dev) | bypass proxy secret locally |
| `REQUEST_DEADLINE_S` | `25` | must be < proxy timeout |
| `CACHE_DIR` | `.cache` | SQLite caches, cloned corpus; a relative path is taken from the repo root (like `.env`), so `backend/` commands and CI share one `.cache/` |
| `FASTAPI_REF` | default `0.141.1` | pinned corpus tag (commit `95f8322e`); blank is rejected |

Retrieval settings are grouped into a frozen `RetrievalConfig` whose canonical JSON is hashed
(`retrieval_config_hash`). The hash is logged per request, used in cache keys and recorded per eval run.
`retrieval/config.py:RetrievalConfig` (a frozen Pydantic model, built by `RetrievalConfig.from_settings(settings, mode)`) has the fields `mode` (`dense` \| `fts` \| `hybrid`; Phase 6 adds `hybrid_rerank`), `k_dense`, `k_fts`, `k_fused`, `k_context`, `rrf_k`, `rerank_provider` and `rerank_model` (the two rerank fields come from `RERANK_PROVIDER` / `RERANK_MODEL`; a model named while the provider is `none` is dropped, so "no rerank" has one hash). `canonical_json()` is key-sorted and whitespace-free, and `config_hash` is its sha256 hex. Every field is hashed, whether or not the mode uses it, so changing `K_FTS` also changes the `dense` hash.

## 5. Ingestion pipeline (offline)

`uv run grounded ingest [--ref <tag>] [--database-url …] [--activate] [--dry-run]` (`--ref` defaults to `FASTAPI_REF`, the database to `DATABASE_URL_DIRECT`)

```
fetch corpus → discover pages → resolve includes → parse headings → chunk [A] → config hash → (reuse) → embed (cached) → store + verify → activate
```

- `--dry-run` parses and chunks, prints the page/chunk/token stats and the number of texts that would be sent for embedding (cache misses). No API call, no database.
- `grounded index list` shows every index version (id, ref, SHA, config hash, status, active, counts, token p50/p95/max).
- Code: `ingest/pipeline.py` (`prepare_corpus`, `index_spec`, `ingest`, `activate_index_version`), wired by `cli.py`.

### 5.1 Fetch
- `--ref` must be a **tag** (branches move). `git ls-remote` resolves it first; a missing tag fails before anything is cloned.
- Shallow clone `fastapi/fastapi` at the tag into `.cache/corpus/<ref>` with `core.autocrlf=false`, so files are byte-identical to the tag on every OS (a CRLF checkout would change every content hash). Record the resolved commit SHA.
- Reuse an existing checkout only if its tag ref resolves to `HEAD` and `git status --porcelain` is empty. Anything else is an error that asks the operator to delete the directory; it is never silently re-cloned.

### 5.2 Discover pages
- Include `docs/en/docs/**/*.md` (other languages are translations of the same pages), sorted by path.
- Exclude (patterns relative to `docs/en/docs`, reviewed against tag `0.141.1`):

| Pattern | Why |
|---|---|
| `release-notes.md` | changelog (~700 KB of version bullets), not documentation |
| `external-links.md`, `fastapi-people.md` | lists generated by Jinja from data files that aren't rendered here |
| `newsletter.md` | an embedded signup form, no text |
| `translation-banner.md`, `_llm-test.md` | snippets/fixtures for the translation tooling |
| `reference/*` | API reference rendered by mkdocstrings from docstrings; the Markdown only holds `:::` directives |
| `js/*`, `css/*`, `img/*` | site assets |

- URL mapping: `docs/en/docs/<path>.md` → `https://fastapi.tiangolo.com/<path>/`; `<dir>/index.md` → `https://fastapi.tiangolo.com/<dir>/`; the root `index.md` → `https://fastapi.tiangolo.com/`.
- Navigation names come from the H1 of each enclosing directory's `index.md` (`tutorial/` → "Tutorial - User Guide", `tutorial/security/` → "Security"), so `mkdocs.yml` isn't parsed. A directory's own `index.md` is the section page, so its nav path stops at the parent. A directory without an `index.md` title falls back to its name (logged).

### 5.3 Resolve code includes
FastAPI docs pull code from `docs_src/` (and once from `fastapi/` itself). Paths are relative to `docs/en/` (where mkdocs runs), e.g. `../../docs_src/x.py`. Both syntaxes exist at tag `0.141.1` and are expanded line by line before parsing, like the mkdocs preprocessors do:
- Current (`markdown-include-variants`, 440 uses): a whole line `{* <path> hl[…] ln[…] title["…"] *}`.
- Legacy (`mdx_include`, 6 uses): `{!<path>!}` or `{!> <path>!}` on its own line inside a fence the page already has; the line is replaced by the file's lines (keeping its indentation).

Rules:
- A `{* *}` directive becomes a fenced ```` ```python ```` block with the file contents.
- **Variant:** keep the file the directive references. The plugin shows that file first and puts the other variants (e.g. non-`Annotated`) in a collapsed "Other versions" panel, which is skipped. Preferring `Annotated` regardless would be wrong: at `0.141.1` three directives reference the non-`Annotated` file on purpose because the surrounding text explains that form. At this tag only `_py310` variants exist.
- `ln[a:b,c:d]` keeps those 1-based inclusive ranges and adds the site's `# Code above omitted 👆` / `# Code here omitted 👈` / `# Code below omitted 👇` comments, so the excerpt reads as partial. `ln[0:0]` means the whole file.
- `hl[…]` (and a legacy fence's `hl_lines="…"`) only highlight and are dropped. `title["app/main.py"]` stays on the fence as `title="app/main.py"`, because it names the file.
- If the code contains a run of backticks, the fence is made longer than that run.
- A missing file, a path outside the repository, an out-of-range `ln` or an unknown option is an ingest **error** (fail loudly), not a silent skip. So is a page line that *looks* like a directive (starts with `{* ` or `{!`, indented or not) but matches neither form: an indented `{* *}`, a trailing space, an unclosed `{! `. Left alone it would be indexed as literal text. Only the page source is checked, not the included code.
- **Noise code is dropped after parsing** (Author decision, 2026-09-26). A code block identical to an earlier one on the same page is dropped, keeping the first. The docs re-include one file per section with other lines highlighted, so repeats are common: 138 blocks on 55 pages at `0.141.1`. A block holding embedded binary data (a base64 run of ≥ 400 characters) is dropped too: it has nothing to retrieve and risks the embedding input limit. At `0.141.1` that is one block, an image in `advanced/stream-data.md`.

### 5.4 Headings and anchors
- Parse with markdown-it-py so headings inside code fences are ignored.
- FastAPI headings often carry explicit anchors: `## Create a task function { #create-a-task-function }`. Use the explicit anchor when present. Otherwise slugify like Python-Markdown's `toc` (lowercase, drop punctuation, spaces → `-`). Strip the `{ #… }` suffix from the visible heading text.
- Breadcrumb = navigation section (from the path, e.g. `Tutorial - User Guide`) + H1 + H2 + H3 of the chunk.
- Output of `ingest/markdown.py`: `ParsedDocument(source_path, url, title, nav_path, blocks, content_hash)`, where `blocks` is a flat, document-ordered sequence of `HeadingBlock(level, text, anchor, markdown)`, `TextBlock(markdown)` and `CodeBlock(markdown, lang)`. Block Markdown is sliced verbatim from the source lines (`token.map`), not re-rendered. The parser builds no sections or chunks; that is the chunker's job.
- `TextBlock(markdown, kind, items)` = one top-level paragraph, list, table, blockquote or HTML block. Admonition/tab markers (`/// tip` … `///`, `//// tab | …`) aren't CommonMark containers; a container that holds only text is merged into one `TextBlock` (so a tip isn't cut from its marker), while one holding code or a heading stays flat so the `CodeBlock` stays atomic.
- A container marker that is never closed leaves its blocks flat and logs a `WARNING` naming the page and the opening marker line(s). The blocks are unchanged by the warning, so `PARSER_VERSION` stays.
- `kind ∈ {paragraph, list, table, blockquote, html, container}` tells the chunker how a block may be split without re-parsing Markdown. For lists, `items` holds each top-level item verbatim with its marker (nested lists stay inside their item); it is empty for every other kind.
- Dropped before/while parsing: YAML front matter, Jinja `{% raw %}` markers, HTML comments, HTML blocks containing Jinja (e.g. the home page's sponsor grids, generated from data files), thematic breaks.
- `content_hash` = sha256 of the text **after front matter, includes and Jinja raw markers**. HTML comments, Jinja HTML and thematic breaks are dropped later, while parsing, so they are still in the hashed text. The code stays this way on purpose: changing what is hashed would change every stored hash.

### 5.5 Chunking [A]
Contract for `chunk_document(doc: ParsedDocument, cfg: ChunkingConfig, count_tokens: TokenCounter) -> list[Chunk]`.
`ChunkingConfig` (frozen: `max_tokens`, `overlap_tokens`, `min_tokens`, `tokenizer`, `strategy="headers"`) and `Chunk`
live in `ingest/types.py`. `ChunkingConfig.canonical_json()` is stored as `index_versions.chunking_config` and feeds
`config_hash`. The full rule set is the docstring of `ingest/chunker.py`; the spec tests are `tests/unit/test_chunker.py`. Summary:
- Primary boundaries: H2 and H3 sections. Text between the H1 and the first H2/H3 is the page-intro chunk (anchor `''`). H4–H6 are ordinary content. A chunk's content starts with its section's heading line.
- A parent directly followed by its child (intro → H2/H3, H2 → H3) isn't a chunk when it is empty (always) or its own content is < `min_tokens` (if the child stays ≤ `max_tokens`): its heading line and text are prepended to the child, which keeps its metadata. This chains (small intro → small H2 → H3). A section left with nothing but heading lines (no child took it) yields no chunk.
- A section longer than `max_tokens` (~450) is split by greedy packing of blocks; a paragraph that doesn't fit breaks into sentences, a list into its top-level items. Overlap (≤ `overlap_tokens`, ~50) is whole trailing sentences of a paragraph that ends the previous part; nothing else is copied.
- **A heading never ends a part** (keep-with-next, `CHUNKER_VERSION` 2). A heading is an H4–H6 line, or the H2/H3 line of a thin parent that the rule above carried into a section. If the block after a heading doesn't fit into the current part, the heading (and any headings right before it) moves to the next part together with that block, and no overlap is put before it. The block is then broken up as usual, or, if it is atomic, stays whole with the heading even over `max_tokens`. Before this rule `deployment/docker.md#dockerfile` ended with `#### Docker Cache`.
- Code blocks, tables, list items, containers, HTML and blockquotes are **atomic**: never split. An atomic block larger than `max_tokens` becomes its own chunk (allowed to exceed, flagged in stats).
- Very small sections (< `min_tokens`, ~40) merge with an adjacent sibling under the same parent (next first, else previous) if the result stays ≤ `max_tokens`. The merged chunk keeps the first section's anchor and the common parent breadcrumb. The intro never merges.
- Each chunk has `section_id = "<source_path>#<deepest anchor>"`, `anchor_path` (outermost → deepest), `breadcrumb`, `heading_level`, `ordinal`.
- Sentence ends: `.`/`!`/`?` (plus closing quotes, brackets, emphasis) before whitespace, except inside inline code and after `e.g.`, `i.e.`, `vs.`, `cf.`. Versions and URLs (`3.10`, `a.b.com`) don't split, since no whitespace follows the dot.
- Pure and deterministic: same input + config + counter → identical output (tests depend on it).
- On tag `0.141.1` with the defaults (450/50/40, `o200k_base`): 125 pages → 1,045 chunks, ~196K tokens, p50 152 / p95 423 / max 1,356 tokens. 13 chunks exceed 450, and each is a single atomic block (code, table, HTML). Keep-with-next (index version 2) leaves the counts and stats unchanged and changes the text of 2 chunks (the two parts around the moved heading in `deployment/docker.md#dockerfile`); no chunk ends with a heading any more (`grounded index list` shows the numbers for the active version).
- `count_tokens` is injected: ingest uses tiktoken (`ingest/tokens.py`, `TOKENIZER_ENCODING`), the tests a "one word = one token" counter. Counts are approximate (model tokenizers differ). tiktoken downloads its encoding on first use; pytest blocks the network, so CI warms `TIKTOKEN_CACHE_DIR` (cached by `actions/cache`) in a step before the tests. A step reads the encoding from `Settings` (`get_settings().tokenizer_encoding`) and the cache key is `tiktoken-<encoding>-v1`, so `Settings` stays the only source.

### 5.6 Hash, embed, cache
- `content_hash = sha256(breadcrumb_text + "\n\n" + content)`. The embedded text is exactly that string (heading context measurably helps retrieval).
- Embedding call: `task_type=RETRIEVAL_DOCUMENT` for chunks and `RETRIEVAL_QUERY` for questions, `output_dimensionality=768`.
- **L2-normalize** vectors ourselves. Truncated (Matryoshka) Gemini embeddings are not unit-length (the recorded fixture has norms ≈ 0.58).
- Free-tier limits for `gemini-embedding-001` (AI Studio, checked 2026-09-26): 100 RPM, 30K TPM, 1K RPD (reset at midnight Pacific); 2,048 input tokens per text. The API rejects more than 100 texts per batch call. This cap comes from the API's error message, not the docs. The limits count **texts, not HTTP calls**. The docs don't say this, but AI Studio's usage page showed the one 3-text fixture batch as 1 API request and 3 "Embedding Requests" for the model (checked 2026-09-26). So RPM caps texts per minute, and RPD caps a full ingest (~1–1.5K texts) at 1K texts per day: the first ingest takes two days, and the cache resumes it on day two.
- **No truncation.** The Gemini API has no `auto_truncate` (the SDK allows it only on Vertex), so a text over `EMBEDDING_MAX_INPUT_TOKENS` fails **before any call** (ingest names the chunk's `section_id`). Counts are tiktoken estimates, and Gemini's tokenizer differs. The largest block on the pinned tag is ~1.4K tokens, which leaves ~30% margin.
- Batches hold ≤ `EMBEDDING_BATCH_SIZE` texts and ≤ `EMBEDDING_TPM` estimated tokens. Each call is paced against a rolling 60 s window of texts (each one a request) and estimated tokens, and rejected calls count toward that window too. 429/5xx/timeouts are retried up to `EMBEDDING_MAX_RETRIES` times. The wait is the server's delay (`RetryInfo.retryDelay`, else `Retry-After`) or exponential backoff with equal jitter. **A server delay is never shortened**: retrying before the deadline it gave would break Retry-After (AGENTS.md §6.15), and rejected calls still count toward the quota. If it is longer than `EMBEDDING_MAX_RETRY_WAIT_S` (60 s, the length of the RPM/TPM window), `ProviderRateLimited` is raised at once with that in the message. When no retry is left (always so on the request path, `QUERY_EMBEDDING_TIMEOUT_S`) the cap plays no part: the server's own error is raised, with its delay. A **daily-quota** 429 (`QuotaFailure` with a `…PerDay…` quota ID) is raised at once, since waiting hours inside a CLI run helps nobody. Other 4xx raise `ProviderRequestRejected` without a retry.
- Cache: SQLite `.cache/embeddings.sqlite` (`infra/kvcache.py`), key = `model|dim|task_type|sha256(text)` (for chunks, that sha256 is `content_hash`), value = little-endian float32 blob. A re-run with no content changes makes **zero** embedding calls. Misses are embedded in slices, and each slice is stored before the next is sent, so a run stopped by the daily quota resumes where it stopped. Fresh vectors also go through the float32 round trip, so a first run and a cached re-run return bit-identical vectors.
- **Lazy embedder.** The CLI builds the provider client through `LazyEmbedder(model, dim, factory)` only when a text has to be sent. `ingest` of an already built version, and an eval whose query vectors are all cached, need no `GEMINI_API_KEY`. With a cold cache and no key, the command fails with `EmbedderUnavailableError` and says how many texts are not cached (for ingest, before anything is written to the database).

### 5.7 Store, verify, activate
- **Index identity.** `index_versions.chunking_config` = the `ChunkingConfig` fields plus `excluded_pages` (the sorted §5.2 patterns) and `parser_version` (`ingest/markdown.py:PARSER_VERSION`, bumped by hand whenever the same page parses to different blocks). `config_hash = sha256(git_sha | embedding_model | embedding_dim | canonical JSON of chunking_config)`. So a new tag, model, dimension, chunk setting, exclusion or parser change is always a new index version. The chunker has its own version, `ingest/chunker.py:CHUNKER_VERSION` (stored as `chunker_version` in `chunking_config`), bumped by hand whenever the same page chunks to different text, exactly like `PARSER_VERSION` for parsing. It is 2 since keep-with-next; a change to the chunker's output without a bump would be served the old index.
- **Reuse.** If a `ready` version with the same `config_hash` exists, nothing is built: the command reports it (and `--activate` activates it). No duplicate rows. This is also a database guarantee since migration `0002`: a unique index on `config_hash WHERE status = 'ready'`. If two ingests of one config race, the second fails when it marks its version ready with "another ingest already built this config; re-run to reuse it" and leaves no `failed` row (a lost race isn't a failure worth recording).
- **Before any embedding call:** the `chunks.embedding` column dimension must equal `EMBEDDING_DIM`, and every text must fit `EMBEDDING_MAX_INPUT_TOKENS` (the error lists the offending `section_id`s).
- **Embed before touching the database.** Vectors go through the SQLite cache in slices of `EMBEDDING_BATCH_SIZE`, each stored before the next is sent. A daily-quota 429 stops the run with the number of texts still missing and a non-zero exit; the database is untouched, and the same command the next day sends only the rest. Other provider errors (after the embedder's retries) also leave the database untouched.
- **One transaction:** insert `index_versions(status='building')`, documents (`executemany … RETURNING`), chunks (batched `executemany`, vectors bound as pgvector `Vector`), verify, then `status='ready'` with `ready_at`, `document_count`, `chunk_count` and `token_stats` (`min`, nearest-rank `p50`/`p95`, `max`, `total`, `over_max` = chunks past `max_tokens`).
- Verify reads back what was written: document and chunk counts equal what was prepared, chunk count > 0, no NULL embeddings, every vector has `EMBEDDING_DIM` dimensions. A chunk can only reference a document of its own version (composite FK from `0002`). Golden-set labels are checked separately by `golden validate --against-index` (the golden set is versioned on its own).
- **Failure while storing or verifying** rolls the transaction back and leaves one `index_versions` row with `status='failed'` and the error in `notes` (no documents or chunks).
- `--activate` switches the active version atomically (DB.md §7.3). `grounded index prune` (later) retires/deletes old versions.

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
The query embedding is one attempt within `QUERY_EMBEDDING_TIMEOUT_S` (`GeminiEmbedder.for_request_path`): no pacing window and
no retry, because both sleep (§10). A 429, 5xx or timeout fails the request at once as a 503 `provider_unavailable`
(with `Retry-After` for a 429). The golden batch (§15.8) waits out a per-minute 429 in the CLI layer instead.
Every stage is wrapped in `StageTimer.stage("name")` (`infra/timing.py`, §14) that fills the latency fields.

## 7. Retrieval

- **Dense:** exact cosine scan, top `K_DENSE` (DB.md §6.1): `retrieval/dense.py:dense_search(conn, query_vector, index_version_id=, k=)`, async, the vector bound as a pgvector `Vector`, `ORDER BY distance, id`.
- **Active version:** `retrieval/index.py:active_index_version(conn)` → `IndexVersion` (id, ref, SHA, embedding model/dim, config hash); none active → `NoActiveIndexError`.
- **Lexical [A]:** OR-semantics tsquery over weighted `tsv`, top `K_FTS` (DB.md §6.2).
- **Hybrid [A]:** `retrieval/hybrid.py:hybrid_search(conn, question, query_vector, *, index_version_id, cfg)`. One statement; RRF, `score = Σ 1/(RRF_K + rank)` over a dense list cut to `K_DENSE` and a lexical list cut to `K_FTS`, top `K_FUSED` unique chunks, deterministic tie-break (DB.md §6.3).
- Output type: `list[RetrievedChunk]` (frozen dataclass, `retrieval/types.py`) with `chunk_id, section_id, anchor_path, breadcrumb_text, url, content, token_count, content_hash` and the signals `dense_rank, dense_distance, fts_rank, fts_score, rrf_score, rerank_score` (each `None` unless the mode produced it).
- Retrieval modes (for evals and ablations): `dense`, `fts`, `hybrid`, `hybrid_rerank`. The `no_rag` mode skips retrieval entirely: no embedding, no index lookup, no database access, the sibling prompt `answer_no_rag_v1` (§9.2) and no zero-citations check (§9.5). It exists for the Phase 4 baseline and is reachable only from the eval harness and the CLI (`grounded ask --mode no_rag`), never from the HTTP API (`AskPipeline.ask(..., mode=AskMode.NO_RAG)`; the route does not pass a mode, and an unknown `mode` field in the request body is ignored). Its `Meta` has `index_version = "none"`, `retrieval_config_hash = "no_rag"` and `embed`/`retrieval` latencies of 0.
- Context selection: the first `K_CONTEXT` chunks after (optional) rerank. If two selected chunks are adjacent parts of one split section, keep both. `generation/context.py` pulls the worse-ranked parts right behind the best-ranked one and sends them in document order (`chunk_id` ascending, which ingest guarantees per document; `RetrievedChunk` has no `ordinal`). Labels are assigned after that, so they ascend in the prompt.

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
    cache_hit: bool = False     # True only for a reply the eval LLM cache served (§15.6)

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

- Adapters translate the Pydantic model to the provider's structured-output format (Gemini `response_schema` / JSON schema; Groq `response_format` with `json_schema` on a model that supports it). They return **validated** objects or raise typed errors: `ProviderRateLimited(retry_after_s, is_quota)`, `ProviderUnavailable`, `ProviderTimeout`, `ProviderBadOutput(raw, validation_error, input_tokens, output_tokens)` (the usage of the billed but unusable reply), `ProviderRequestRejected(status_code)` (a 4xx other than 429, e.g. a bad request or key: retrying the same call won't help). They live in `infra/provider_errors.py`, which the embedding adapter uses too. The Gemini-specific mapping (429 with `RetryInfo`/`QuotaFailure` → `ProviderRateLimited(retry_after_s, is_quota)`, 5xx → `ProviderUnavailable`, other 4xx → `ProviderRequestRejected`) is one function, `infra/gemini_errors.py:map_api_error`, used by both the embedder and the generator.
- **Provider schema support differs.** Keep `LLMAnswer` simple: no unions, no recursive refs, enums as string literals. Constraints a provider can't express (regex, lengths) are enforced by Pydantic after the call. A unit test per adapter checks that the schema converts.
- `FakeLLMProvider` returns scripted results/errors in order. It is used by all tests.
- **`GeminiProvider`** (`generation/providers/gemini.py`, async `google-genai` client): one `generate_content` call with `response_mime_type="application/json"` and `response_json_schema` = the model's JSON Schema reduced to the keywords Gemini documents (`to_gemini_schema`: `$defs`/`$ref`, `type`, `enum`, `items`, `minItems`/`maxItems`, `minimum`/`maximum`, `anyOf`, `properties`, `required`, …; `minLength`/`maxLength`/`pattern`/`default` are dropped and enforced by Pydantic after the call). `thinking_config.thinking_level` comes from `GEMINI_THINKING_LEVEL`; the per-attempt timeout is an `asyncio.timeout` around the call. The reply text is always re-validated (`ProviderBadOutput` with `raw`, the reply's usage and the retry feedback `compact_validation_error(exc)`; a no-candidate or blocked reply is bad output too, and a `MAX_TOKENS` cut shows up as invalid JSON with the finish reason in the message and `TRUNCATED_FEEDBACK` (in `infra/provider_errors.py`, shared with the Groq adapter) as the feedback, "the answer was cut off at the output token limit … write a shorter answer", because the JSON error alone would not tell the model why). Usage: `input_tokens = prompt_token_count`, `thinking_tokens = thoughts_token_count`, and `output_tokens = candidates_token_count + thoughts_token_count`, because Gemini keeps thoughts out of `candidates_token_count` but bills them as output (`max_output_tokens` limits both together). Errors go through `infra/gemini_errors.py`. `build_provider` picks it when `GENERATOR_PROVIDERS[0]` is `gemini`.
- **`GroqProvider`** (`generation/providers/groq.py`, built in 4.02; model `openai/gpt-oss-120b`, D48; async `groq` `1.7.0`, PyPI 2026-08-26). One `chat.completions.create` call: the system and user messages, `response_format={"type": "json_schema", "json_schema": {"name": <model name>, "strict": True, "schema": to_groq_schema(model)}}`, `max_completion_tokens` = `LLM_MAX_OUTPUT_TOKENS`, `reasoning_effort` = `GROQ_REASONING_EFFORT` and `include_reasoning=False`. The reply text is always re-validated with Pydantic, like Gemini's (`ProviderBadOutput` with `raw`, the reply's usage and the compact retry feedback). Read from Groq's docs and the SDK source (4.01) and checked on 2026-10-08 against five real calls (4.02; the recordings are in `tests/fixtures/groq/`); what stayed unverified is marked *unverified*.
  - **Client.** `AsyncGroq(max_retries=0)`, built by `GroqProvider.create`; the constructor refuses a client that has retries. The SDK default is two retries with a sleep on connection errors, 408, 409, 429 and 5xx (it honors `retry-after` up to 60 s), which would put sleeping in the request path (§9.5) and hide a 429 from the router. The per-attempt timeout is an `asyncio.timeout` around the call, as in the Gemini adapter. `runtime.py:build_groq_provider(settings, model=…)` builds it from `GROQ_API_KEY` and the model the caller names (the judge in 4.04, the router in Phase 7); `GENERATOR_PROVIDERS` cannot select `groq` as the generator before the router exists.
  - **Strict mode.** `to_groq_schema` lists every property as `required` and closes every object with `additionalProperties: false` (`LLMAnswer.follow_up_questions` has a default, so Pydantic does not list it), and keeps only `type`, `enum`, `items`, `anyOf`, `properties` and `$defs`/`$ref`. `minLength`, `maxLength`, `minItems`, `maxItems`, `minimum`, `maximum` and `pattern` are accepted in the schema but **not enforced while decoding**: Groq validates the finished JSON itself and answers a violation with `400 json_validate_failed` (an `error` object with `message`, `type`, `code` and `failed_generation`) and no usage (recorded with `maxLength`). Sending them would turn a reply that our Pydantic check reports with its usage into a usage-less error, so they are dropped, like `title` and `default`, and Pydantic enforces them after the call. The judge verdict shape `{verdict: enum, reason: str}` was accepted. `name` is `a-zA-Z0-9_-`, 64 characters at most. Structured Outputs cannot be combined with streaming or tool use (neither is used). A system message works, and temperature `0` is accepted although Groq recommends 0.5–0.7 for reasoning models; the judge agreement (4.11) shows whether it matters.
  - **Reasoning.** `reasoning_effort` takes `low | medium | high` for this model (`GROQ_REASONING_EFFORT`, default `low`: the reasoning is billed as output, counts inside `max_completion_tokens` and uses the free plan's 8K tokens per minute, PRD §12); `reasoning_format` is not supported; `include_reasoning=False` keeps the reasoning out of the reply (the message came back with `content` only). Reasoning tokens count **inside** `max_completion_tokens` and **inside** `usage.completion_tokens`: a cap of 20 tokens, which the reasoning alone uses up, left the content empty, and the recorded reply reports `completion_tokens=323` with `completion_tokens_details.reasoning_tokens=27` for an answer of roughly 280 tokens. So `Usage.output_tokens` is `completion_tokens` as it comes and `thinking_tokens` is `reasoning_tokens` (nothing is added, unlike Gemini).
  - **Errors.** `retry-after` (seconds, set on a 429 only) is read from the `groq.RateLimitError` response. 429 → `ProviderRateLimited(retry_after_s, is_quota)`; 5xx and connection errors → `ProviderUnavailable`; `APITimeoutError` and our timeout → `ProviderTimeout`; any other 4xx → `ProviderRequestRejected(status_code)`. Groq documents neither the body of a 429 nor how a daily limit differs from a per-minute one, and a 429 cannot be provoked, so `is_quota` is a guess from two signals: the message names a per-day limit (`(TPD)`, `(RPD)`, "per day") or the wait is longer than 60 s, the window of the per-minute limits. *Unverified until a real daily 429 is recorded*; the 429 and 5xx fixtures are hand-made. **`400 json_validate_failed` is bad output, not a rejected request.** It is how a schema violation and a cut-off reply both arrive (a cap that the reasoning and the answer use up gives an empty `failed_generation`; recorded with a cap of 20). The adapter makes it `ProviderBadOutput(raw=failed_generation)` with 0 tokens, because the error carries no usage (whether such a call is billed is *unverified*). The feedback is our compact validation error, or `TRUNCATED_FEEDBACK` when the text is not JSON at all, which strict decoding only produces when the limit cut it off; a `200` with `finish_reason=length` gets `TRUNCATED_FEEDBACK` too (not seen). `TRUNCATED_FEEDBACK` lives in `infra/provider_errors.py`, shared by both generator adapters.
  - **Rate-limit headers.** On every response, `x-ratelimit-*-requests` are requests per **day** and `x-ratelimit-*-tokens` tokens per **minute** (Groq's rate-limits page). The first real call returned limits of 1000 and 8000, the Free Plan numbers of PRD §12. The adapter does not read them.
  - **Usage and price.** `prompt_tokens`, `completion_tokens`, `completion_tokens_details.reasoning_tokens`. The list price is in `pricing.toml` (§14).

### 9.2 Prompts
- Files in `backend/prompts/`, e.g. `answer_v1.md` (system + user template sections), `judge_faithfulness_v1.md`, `judge_correctness_v1.md`.
- `prompt_version = "<name>@<first 8 hex of sha256(file)>"`, e.g. `answer_v1@3fa9c2d1`. It is computed at load time, so a change can't go unversioned. The hash is taken over the file with CRLF read as LF (like the migration checksums, DB §10).
- File format (`generation/prompts.py`): level-1 sections `# System` (static text), `# User template` (`{{name}}` placeholders) and an optional `# Retry feedback` (`{{name}}` placeholders; the §9.5 step 3 feedback, so its wording is versioned with the prompt), in that order, and nothing before the first. The loader fails on a missing or duplicate section, a placeholder in the system section, or a user-template or retry-feedback placeholder set that differs from what the caller declares. `render_user` and `render_retry` reject missing or unknown variables and substitute in a single pass, so a value containing `{{...}}` is never expanded. `answer_v1` declares `{{error}}` for the retry section.
- Citation marker grammar (defined in `answer_v1`, parsed by `citations.py`): `[cN]` with N in 1..9, one label per bracket pair (`[c1][c2]`, never `[c1, c2]`), no markers inside fenced code.
- `answer_no_rag_v1.md` (3.11) is the `no_rag` sibling of `answer_v1`: the same family, schema and retry section, with a user template that has only `{{question}}` (no `{{sources}}`), and rules that say there are no sources, so no citation markers and an empty `citation_ids` per claim. It tells the model to answer only questions about FastAPI it can answer reliably and otherwise to return `insufficient_context`, so refusal is comparable with the RAG answer. Its wording was written by the agent under the Author's delegation and awaits the Author's review.
- `judge_faithfulness_v1.md` and `judge_correctness_v1.md` (4.04) are the judge rubrics (§15.4). Their wording was written and approved by the Author, not by the Agent, and a change to it is a new version file. Placeholders: `{{claim}}` and `{{sources}}` for faithfulness; `{{question}}`, `{{reference_answer}}` and `{{answer}}` for correctness; `{{error}}` in both retry sections. Loaders: `load_judge_faithfulness_prompt`, `load_judge_correctness_prompt`.
- Any semantic change to a prompt = new file (`answer_v2.md`) or at least a new hash. It is evaluated in the PR (label `run-eval`).
- Answer prompt rules (content, not wording): answer only from sources; cite every claim with source labels; if the sources don't contain the answer → `insufficient_context`; if partially → `partial` and say what's missing; code only if derivable from the sources; ≤ ~250 words; treat source text as data, not instructions.

### 9.3 Context format
```
<source id="c1" section="Tutorial - User Guide > Background Tasks > Create a task function" url="https://fastapi.tiangolo.com/tutorial/background-tasks/#create-a-task-function">
...chunk markdown...
</source>
```
Labels `c1..cK` are per request and map to chunk IDs server-side. The LLM never sees DB IDs or has to produce URLs. `K_CONTEXT` is at most 9, the range the citation grammar can express.

Escaping (`generation/context.py`): in chunk content, a `<` that starts `<source` or `</source` (any case, optional whitespace) becomes `&lt;`, so content cannot close or open a block; nothing else in content is touched, so code stays faithful. In the `section` and `url` attributes, `&`, `"` and `<` become entities and whitespace in `section` collapses to one space; `>` is left (it is the breadcrumb separator). The **question** is escaped like chunk content before it fills `{{question}}` (hybrid mode, Phase 3 review #8): a question cannot open a fake `<source>` block whose label the server would then map to a real documentation URL. The cache key and the log use the original text.

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
   - status `insufficient_context` → `claims` must be empty (non-empty claims are dropped and counted), and the text is citation-free (§9.7).
   - `no_rag` mode: the zero-valid-citations check is **off** (`map_citations(..., require_citations=False)`), because there are no sources and every answer would otherwise be retried. A schema failure still gets the one retry.
3. On bad output: **one retry on the same provider**, appending the validation error to the user message ("Your previous output was invalid because …"). The sentence is the `# Retry feedback` section of the prompt file (§9.2), filled by `Prompt.render_retry(user, error=...)`: the original user message, a blank line, then the feedback. The error is the adapter's compact feedback or the citation check's reason. The compact feedback (`infra/provider_errors.py:compact_validation_error`) is `<field path>: <message>` per Pydantic error, joined with `; `, without the rejected input values (they could repeat the whole bad output) and without the `errors.pydantic.dev` links (the prompt forbids URLs); a reply cut off at the token limit gets the truncation sentence instead, with the same `max_output_tokens` (PRD D47). Bad output that the same request would hit again is not retried: a prompt the content filter blocked (no candidate) or an answer it stopped (finish reason `SAFETY`, `BLOCKLIST`, `PROHIBITED_CONTENT` or `SPII`; not `RECITATION`, where a resampled answer may pass). The adapter marks it `ProviderBadOutput(retryable=False)`. Then go to the next provider in the router (that counts as fallback). Until the router exists (Phase 7) the pipeline's `_generate_validated` does this and a second failure is the `502`; it counts `validation_retries` for the request log.
4. Rate limit / 5xx / timeout → router behavior (§10).
5. All paths exhausted → `502 validation_failed` or `503 provider_unavailable`, logged with outcome.

### 9.6 Citation mapping
- Valid label = present in this request's label map. Invalid labels are removed from claims and from `answer_markdown`, and counted (`invalid_citation_count`). Invalid labels are a tracked metric; they are *not* silently fixed.
- Display numbering: `[cK]` markers in `answer_markdown` → `[n]` by order of first appearance. Claims' citation lists are mapped to the same `n`.
- Each distinct cited chunk becomes one `Citation` (URL with anchor, H1 title, breadcrumb, snippet). A label cited only by a claim (not in the text) is numbered after the labels that appear in the text, in claim order.
- A marker is any `[c<digits>]` outside fenced code (so `[c0]` and `[c12]` are invalid labels, not prose). A bracket pair holding several labels (`[c1, c2]`) is removed and counted once. Markers inside fenced code blocks are left untouched and not counted (a fence is a line of 3+ backticks or tildes, closed by the same character at least as long; an unclosed fence runs to the end).
- `invalid_citation_count` counts removed references: invalid markers in the text plus invalid labels in claims' `citation_ids` (the same bad label in both places counts twice). Labels of claims dropped under `insufficient_context` are not counted again; those claims are counted in `dropped_claim_count`.
- **URLs from the model are removed** (AGENTS.md §6.3, PRD D47): `citations.py:strip_urls` runs on `answer_markdown` before the markers are rewritten. A Markdown link or image keeps only its text (a marker text such as `[c1](…)` stays `[c1]`, so it still cites; a grouped one such as `[c1, c2](…)` stays `[c1, c2]` and is then removed and counted as any group), an autolink or a bare `http(s)://` / `www.` URL (matched case-insensitively, as a renderer does) is removed with the space before it, and a loopback URL (`localhost`, `127.0.0.1`, `0.0.0.0`, `::1`) is kept but wrapped in inline code, because the FastAPI docs tell readers to open them. A link reference definition (`[label]: destination "title"`, also inside `>` or a list item, also with the destination on the next line) is removed: otherwise a rewritten marker `[1]` could be a link to the host it defines. `<scheme:…>` and `<user@host>` autolinks are removed. The link text may hold one level of nested brackets and the destination balanced parentheses or `<…>`; a link or image inside a link's text is a link of its own, and a URL in a link's text is part of that link (counted once). A `](…)` left after that (a code span in the link text, text over two lines, deeper nesting) loses its destination, up to the matching `)` or the end of the line. Fenced code and inline code spans are untouched. Every removal is counted in `removed_url_count` (the stdout summary line, §14; no column). It is not bad output and causes no retry.
- `snippet` = the first 300 characters of the chunk content, stripped and cut back to a word boundary (an unbroken run longer than 300 is hard-cut); no ellipsis.

### 9.7 Refusal
- `insufficient_context` answers are short, have no claims or citations, and may include `follow_up_questions` pointing to what *is* covered. The server enforces it: claims are dropped and counted (`dropped_claim_count`), and **every marker is removed from the text**, valid or not, so the response has an empty `citations` list. Only markers with an invalid label are counted in `invalid_citation_count` (a valid label in a refusal is not counted: it is out of place, not invented). A refusal is never retried for missing citations, and `min_confidence` is `null` because there are no claims. Markers inside fenced code stay untouched, as everywhere (§9.6).
- There is no retrieval-score threshold short-circuit in MVP. It is a possible later optimization, and eval data will show whether it helps.

### 9.8 Confidence heuristic [A]
Contract for `score_claims(claims, label_map, retrieved, *, config) -> list[ClaimScore]` (`generation/confidence.py`; the full contract is the module docstring, the spec tests are `tests/unit/test_confidence.py`):
- Inputs: the model's `LLMClaim`s (labels and `self_confidence`), this request's label map (`BuiltContext.labels`), the fused list as `hybrid_search` returned it (every labelled chunk is in it) and a frozen `ConfidenceConfig` built from `Settings` (`ConfidenceConfig.from_settings`), so the function stays pure. A *valid* citation is a label of the claim that is in the label map; repeated labels count once; unknown labels are ignored here (they are counted in §9.6).
- Output: one `ClaimScore(confidence, components)` per claim, same order; `confidence ∈ [0,1]`. `components` (returned in the API for transparency) always has the keys `retrieval`, `agreement`, `citations`, `self_confidence`, `rerank` (`COMPONENT_KEYS`), each a finite float in `[0,1]` (`rerank` is `0.0` when rerank is off).
- How they combine (3.10, written by the agent under the Author's delegation, to be reviewed): the confidence is the weighted mean `sum(w_k * c_k) / sum(w_k)` of the five components (`retrieval` and `agreement` are the best over the claim's valid citations, `citations = n / (n + 1)`, `self_confidence` as given, `rerank` the best rerank score or `0.0`), then `min(.., CONFIDENCE_UNCITED_CAP)` when the claim has no valid citation. Per cited chunk, `retrieval` is the mean of three fixed slots (the dense list's verdict, the lexical list's verdict, `rrf_score / best rrf_score`) and `agreement` is `min(dense verdict, lexical verdict)`; a list's verdict is `0` when it did not find the chunk, else the mean of its rank (`(depth - rank + 1) / depth`, depth = the deepest rank in the retrieved list) and its score (`1 - distance / 2`, or `fts_score / best fts_score`). The one free constant is the prior doubt `1` in `n / (n + 1)`. The proof that each invariant holds for every input is in the `score_claims` docstring; the pipeline scores `answer.claims` against `BuiltContext.labels` and the fused list, and `min_confidence` is the minimum over the claims.
- Available signals: rerank relevance score of cited chunks (when rerank on), RRF score and ranks (dense/FTS agreement), number of valid citations, LLM `self_confidence`, and optionally lexical overlap between claim and cited chunk text.
- Invariants: deterministic and pure; monotonic non-decreasing in retrieval support (a better rank, both lists instead of one, a higher RRF score or lexical score, a smaller dense distance, a higher rerank score, and one more distinct valid citation **even to a weaker chunk**, so a plain average over the cited chunks is not allowed); **a claim with no valid citation is capped at 0.2** (`CONFIDENCE_UNCITED_CAP`, never above 0.2); `self_confidence` alone can't push a claim above 0.6 (weight it low, since self-reports are poorly calibrated). The weights are relative, so a cap on `CONFIDENCE_W_SELF` alone (0.6, kept as a first fence) guarantees nothing; `Settings` refuses any config whose worst case `(w_self + w_retrieval·2/3 + w_citations·1/2) / Σw` (one citation to a chunk only one list found, no rerank, `self_confidence = 1`) is above 0.6, and all-zero weights (PRD D47, Phase 3 review #4); `rerank_score is None` is a normal input.
- Weights live in settings (not magic numbers), documented in the README.
- Calibration (Phase 8): bucket claims (low < 0.5, mid 0.5–0.8, high > 0.8) and compare with the judge's "supported" rate per bucket. A useful heuristic is monotonic across buckets. Report the table even if it isn't.
- `min_confidence` (answer level) = min over claims (`None` without claims). The UI flags claims below 0.5.

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
- **Eval mode:** provider list is only the primary, enforced at startup by `Settings` (§15.6). Fallback disabled, because the fallback provider is also the judge.
- Tests (with fakes): 429 → fallback + breaker open; breaker skip; half-open recovery; timeouts ×3 → open; bad output → retry → fallback; all open → 503.

## 11. Caching

| Cache | Store | Key | Used in | TTL |
|---|---|---|---|---|
| Answer cache | Postgres `answer_cache` | sha256(normalized question \| prompt_version \| index_version_id \| retrieval_config_hash \| generator_model \| confidence_config_hash \| generation_config_hash) | prod/dev (off in eval and in `no_rag`) | `ANSWER_CACHE_TTL_DAYS` (≤ 30 days) |
| Query embedding | in-memory LRU (prod), SQLite (dev/CI/eval), in front of the request-path embedder (§6: one attempt, no sleep) | (model, dim, RETRIEVAL_QUERY, sha256(question)) | all | process lifetime / persistent |
| Chunk embedding | SQLite `.cache/embeddings.sqlite` | (model, dim, RETRIEVAL_DOCUMENT, content_hash) | ingest | persistent |
| Rerank | SQLite `.cache/rerank.sqlite` | sha256(model \| query \| candidate hashes) | dev/CI/eval | persistent |
| Eval LLM responses | SQLite `.cache/llm_eval.sqlite` | sha256 of the JSON array [provider, model, temperature, max output tokens, system, user, schema hash, adapter settings] (details below) | eval only (generator and judge) | persistent |

Question normalization: Unicode NFKC → lowercase → collapse whitespace → strip trailing sentence punctuation (`. , ; : ! ? … 。`, not every symbol: `C#` stays distinct from `C`). One function, `infra/hashing.py:normalize_question`, shared by the answer cache key and `request_logs.question_hash`.

Answer cache details (`infra/answer_cache.py`, 3.13):
- The stored `response` is the `AskResponse` **without `meta`**. On a hit the pipeline rebuilds `meta`: a new `request_id`, `cache_hit=true`, the real latency of the hit, zero tokens and zero shadow cost.
- `confidence_config_hash` is a hash of the `CONFIDENCE_*` weights and the uncited cap (`ConfidenceConfig`). The stored answer carries server-computed confidence, so changing a weight must miss instead of serving stale numbers for up to 30 days. It is the sixth part of the key and has no column of its own (the key is the primary key). The parts are hashed as a JSON array, so a `|` in a question cannot shift a field.
- `generation_config_hash` (the seventh part, PRD D47) is `GenerationParams.config_hash` (`generation/params.py`): the provider name, `LLM_TEMPERATURE`, `LLM_MAX_OUTPUT_TOKENS` and `GEMINI_THINKING_LEVEL`. Changing any of them is a miss, not 30 days of answers made with the old values. The pipeline passes `GenerationParams.temperature` and `max_output_tokens` to every call, so for those the key and the call cannot disagree; `GEMINI_THINKING_LEVEL` is read a second time by the Gemini adapter when `runtime.py` builds it (same `Settings`, so they agree within a process; it moves to the provider in Phase 4, PRD §12). No column either.
- A row that no longer fits `AskResponse` (written before a schema change that left the key alone) is treated as a miss **and deleted** (`AnswerCache.discard`), so the fresh answer of that request replaces it; the write path only overwrites expired rows (DB.md §7.2).
- The lookup comes right after the active index is known and before the embedding, so a hit costs no embedding, retrieval or LLM call, and the budget reservation (5.05) must come after it: a hit never consumes budget. The lookup is skipped when `APP_ENV=eval` and in `no_rag` mode, which neither read nor write the cache.
- Only a built, valid response is stored (`answered`, `partial`, `insufficient_context`); every failure raises first, so errors are never cached. A database error on the cache is a logged miss or a skipped write, never a failed request.
- `expires_at = created_at + ANSWER_CACHE_TTL_DAYS`, and `Settings` and the store both cap it at 30 days (question retention).
The eval LLM cache is keyed by the *full prompt*. Any prompt or context change misses the cache, so regressions are still detected, while identical cases are free and deterministic on re-runs.

Eval LLM cache details (`generation/providers/eval_wrappers.py`, 4.03):
- **The key** is the sha256 of the parts as a JSON array (a `|` in a prompt cannot shift a field): the provider name, the model, the call's `temperature` and `max_output_tokens`, the `system` and `user` text, the schema hash (sha256 of the sorted JSON Schema of the requested Pydantic model, so the generator's `LLMAnswer` and a judge verdict never collide) and the adapter settings (`generation/params.py:adapter_params`: `{"thinking_level": …}` for Gemini, `{"reasoning_effort": …}` for Groq, `{}` for the fake and stub providers). The ticket's list was provider, model, temperature, system, user and schema; **the output cap and the adapter settings are added** because they change the reply (a cap cuts it off, and Groq's reasoning shares it; thinking level and reasoning effort change what the model writes) and a stale hit would then pass for the current configuration, the same reasoning as D47 for the answer cache. Only the settings the provider uses are in its key, so a moved `GEMINI_THINKING_LEVEL` does not throw away the Groq judge's verdicts (scarce daily quota). The timeout is not in it: it decides whether a reply arrives, not what it says. The adapter settings are read from `Settings` (the same one `runtime.py` builds the adapter from), the same "read twice" as `thinking_level` in `GenerationParams` (PRD §12).
- **The value** is the raw reply text and its `Usage` (JSON), nothing else; provider and model come from the key.
- **A hit** is validated again with Pydantic exactly like a live reply (§9.1), and returns the stored `Usage` with `GenerationResult.cache_hit=True`. The pipeline counts hits on `RequestTrace.llm_cache_hits`. `latency_ms` of a hit is the time of the lookup, not a provider latency; latency statistics must skip results with the flag (§15.3).
- **Only a successful reply is stored.** A provider error or a bad output is never cached (a re-run pays it again). An entry that no longer validates (a corrupt file, a validator changed under an unchanged schema) is deleted and counted as a miss, so the live reply replaces it: one extra live call, no loop. The pipeline's own retry (§9.5) sends a different `user` text (the feedback is appended), hence another key, so a cached first attempt that fails the citation check is replayed once and its retry is the retry's own entry, never the same bad one.
- The file is a `KVCache` (`infra/kvcache.py`); the lookup and the write run in `asyncio.to_thread`.

## 12. Abuse protection and API security

- **Proxy secret:** `/v1/*` requires `X-Proxy-Secret == PROXY_SHARED_SECRET` (constant-time compare). Without it → 401, unless `ALLOW_DIRECT_API=true` (dev only). `/healthz` and `/readyz` are public.
- **Client IP:** trusted from `X-Client-IP` **only** when the proxy secret is valid; otherwise the socket peer. The IP is immediately HMAC-hashed (`IP_HASH_SECRET`). The raw IP is never logged or stored.
- **Rate limit:** in-memory sliding window per IP hash (`RATE_LIMIT_PER_MIN`, `RATE_LIMIT_PER_DAY`). Single instance, so memory is correct. A reset on restart is acceptable. Response: 429 + `Retry-After`, outcome `rate_limited`.
- **Global budget:** atomic reserve in `daily_usage` before each LLM call (DB.md §7.1). Exhausted → HTTP 503 with code `budget_exhausted` (the UI shows a friendly "demo budget reached" state), outcome logged. Cache hits don't consume budget.
- **Input limits:** question 3–500 chars after trimming; reject control characters; body size limit 4 KB.
- **CORS:** not needed (browser talks only to Vercel). Backend CORS is locked to nothing.
- **Output rendering:** the frontend renders Markdown **without** raw HTML (no `rehype-raw`). Citation URLs come from the DB, never from model text. The server removes the links and URLs the model writes in `answer_markdown` (§9.6), but that is a regex, not a Markdown parser: it is the first layer and a metric, not the guarantee (what it cannot see: PRD §12). The guarantee is planned for the frontend (Phase 5, PRD §12): model Markdown is rendered with `a` and `img` disallowed (their text kept), and claims and follow-up questions as plain text. The only links a reader can click are then the citation source cards, which open in a new tab with `rel="noopener noreferrer"`.

## 13. HTTP API

Base path `/v1`. JSON only. Errors share one shape:

```json
{ "error": { "code": "rate_limited", "message": "Too many requests. Try again in 20s.", "retry_after_s": 20 }, "request_id": "…" }
```

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/healthz` | public | process up (liveness) |
| GET | `/readyz` | public | DB reachable, schema present and an index version active; no active index → 503 with `database: ok` and `active_index_version: null` (required since Phase 3; Phases 0–2 only reported it) |
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

`request_id` is created per request (also for validation errors and unhandled exceptions) and is the same id as `meta.request_id`. An invalid body is `422` with code `bad_request`, and the message names the field and the rule but never echoes the input. A missing active index on `/v1/ask` is `500 internal_error`.

OpenAPI docs (`/docs`) stay enabled. The API contract is itself part of the portfolio.

## 14. Observability and cost

- **Structured logs:** JSON lines via stdlib `logging` with a JSON formatter. Every line has `request_id`. Question text is never logged to stdout (only `question_hash`). Each `/v1/ask` ends with one `request completed` line (outcome, status, total latency, `validation_retries`, `invalid_citation_count`, `dropped_claim_count`, `removed_url_count`, `question_hash`); `dropped_claim_count` and `removed_url_count` have no column.
- **Request log row (3.12):** one per `/v1/ask` (all outcomes written today: `answered`, `partial`, `insufficient_context`, `bad_request`, `validation_failed`, `provider_unavailable`, `internal_error`), fields in DB.md §4 `request_logs`. `observability/request_log.py:RequestTrace` is the per-request scratchpad: the route starts it before the body is parsed (`AskRoute`), the pipeline fills it, and the route (success) or the error handler (`api/errors.py:_fail`, the one place that knows status and code) writes the row exactly once. `id` = `request_id`; `source='api'`, `ip_hash` NULL until 5.03. A `bad_request` row has `question` NULL (only its hash, or the hash of the empty text when the body had no string `question`) and no stage, token or version columns. A failed write is logged at ERROR with `request_id`, the exception type and the SQLSTATE, **never the message or traceback** (Postgres puts the failing row, i.e. the question, in the error DETAIL), and never breaks the response; the insert is bounded only by the pool timeout (`DB_POOL_TIMEOUT_S`), so a database outage delays the response by up to that.
- **Stage timing:** `infra/timing.py:StageTimer` on the monotonic clock (injected in tests). `stage("embed" | "retrieval" | "rerank" | "llm")` accumulates time, also when the block raises; a stage that never ran reports `None` (NULL in the row), not 0. `total_ms` counts from the timer's creation, which is the start of the route, so `latency_total_ms` includes validation, any cache lookup (3.13) and response assembly, up to the moment the row is written. Two writes are outside it: the answer-cache write (`put`, after the response is built) and the log insert itself. A client that disconnects cancels the request (`CancelledError`, a `BaseException`) and no row is written; that is handled with the request deadline of 5.02 (PRD §12). For an answered request the response and the row carry the same value; a stage that did not run is absent (0 for embed/retrieval/llm) in `meta.latency_ms`.
- **Shadow cost:** `backend/pricing.toml`:
  ```toml
  as_of = "YYYY-MM-DD"          # date prices were checked
  source = "<provider pricing page URLs>"
  [models."<gemini model id>"]  input_per_mtok = 0.0  output_per_mtok = 0.0   # fill from pricing page
  [models."<groq model id>"]    input_per_mtok = 0.0  output_per_mtok = 0.0
  [embeddings."<embedding id>"] input_per_mtok = 0.0
  [rerank."<rerank id>"]        per_1k_searches = 0.0
  ```
  `cost = in_tok/1e6·in_price + out_tok/1e6·out_price + embed_tok/1e6·embed_price + rerank_calls/1000·rerank_price`, in `Decimal`, rounded half up to the 8 places of `shadow_cost_usd` (`observability/cost.py`, validated by Pydantic). A model that is missing from the file **or marked `unverified`** is a `PricingError` (`Pricing.require_generator` / `require_embedding`, called at pipeline construction, so a startup error; decided in 3.12 because `meta.shadow_cost_usd` is a plain number). The embedding model checked is `EMBEDDING_MODEL`. The `fake` provider is exempt (no billable call, true cost 0). A request that fails halfway is charged for the part that ran (the embedding, and every generation attempt that reported usage, a schema-invalid one included: the adapter puts the billed usage on `ProviderBadOutput`). The query embedder does not report tokens, so embedding tokens are estimated as the question's length in characters, an upper bound of a very small number (≤ 500 chars ≈ $0.000075). It is charged also when the vector came from the query-embedding cache (LRU or SQLite), since the pipeline cannot tell a cached vector from a fresh one; the error is at most that same amount.
  Prices are **never typed from memory**. They are copied from the provider pricing page with the date. Thinking tokens count as output tokens (for Groq, `completion_tokens` already includes the reasoning tokens, so nothing is added; verified in 4.02, §9.1).
  Two rules from the 3.01 decision. (1) A price that comes from somewhere other than today's pricing page (e.g. a launch announcement), or was read on another day than the file's `as_of`, carries its own `price_source` and `price_as_of` on the row. The Groq row is one (4.01): Groq's model page, read 2026-10-08, because `groq.com/pricing` now redirects to the home page. (2) A price that can't be verified at all is left out and marked `status = "unverified"`; the cost code reports it as unknown, never as `0`.
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
  "source_section": "docs/en/docs/tutorial/background-tasks.md#using-backgroundtasks",
  "notes": "multi-step; code expected"
}
```
- Validated by `schemas/eval.py:GoldenItem` (Pydantic, `extra="forbid"`): `id` = `q` + 3 digits, question 3–500 chars (the `AskRequest` limits), labels `docs/en/docs/<page>.md[#<anchor>]`, grade 1 or 2, no label twice. `source_section` is provenance only (the sampled section the question was drafted from, or `null`).
- `type ∈ {factual, how_to, code, multi_section, unanswerable}`. `answerable` is false exactly for `unanswerable` items, which have `relevant_sections: []` and a reference answer describing why. An answerable item needs ≥ 1 grade-2 label, a `multi_section` item ≥ 2.
- Grades: **2** = contains the answer, **1** = useful context.
- **Section matching rule:** a retrieved chunk matches label `path#anchor` iff same `source_path` and `anchor ∈ chunk.anchor_path` (so an H2 label also matches its H3 sub-chunks). A label without an anchor matches the whole page.
- **No nested labels:** within one item, no label may be an ancestor of another (a page and a section on it, an H2 and one of its H3s). One chunk could then match two labels at the same rank and push nDCG above 1. The schema rejects a page + section pair; `golden validate` reports any two labels that match the same chunk; the metrics raise if a chunk matches two labels.
- A label on a small section that the chunker merged into a *previous* sibling never matches: the merged chunk carries the first section's `anchor_path` (§5.5). `golden validate` reports such labels; label the first section or the parent instead.
- Changes to the golden set create a new version file (`v2`) and require new baselines. Items are never edited in place after baselines exist.
- `eval/golden/README.md` holds the labeling guide and the provenance of each version.
- Drafting (decision 2026-09-25): the Agent drafts ~50 candidates in a session from seeded random sections, with no LLM API call, into `eval/golden/candidates.vN.jsonl` (committed as provenance); the Author selects and edits ~30 into `golden_set.vN.jsonl`. Helpers in `evals/golden.py`, all offline over the chunked pinned corpus:
  - `grounded golden sample --n 50 --seed <S>`: distinct H2/H3 `section_id`s (page intros excluded) drawn with a seeded RNG from the sorted list.
  - `grounded golden sections <page>`: a page's sections as the chunker produced them (the labels that can match), with sizes.
  - `grounded golden validate [file] [--against-index]`: schema, unique IDs, type mix vs the PRD §12 target (reported, not enforced), then label resolution against the corpus chunks with `metrics.section_matches`: every label must match a chunk and no two labels of an item may match the same chunk. `--against-index` repeats it on the active index. Any problem → exit 1.

### 15.2 Retrieval eval (Python) [A: metrics]
`uv run grounded eval retrieval [--config dense|fts ...] [--golden <golden_set.vN.jsonl>] [--out <file>] [--write-baseline]` (code: `evals/retrieval_runner.py`; modes: `dense`, `fts`, `hybrid`; `--config` is repeated, one per mode; `hybrid_rerank` arrives in Phase 6)
- Runs against the **active** index of `DATABASE_URL`. The embedding model/dim in settings must equal the index's, or the run fails (vectors of another model are not comparable). Questions are embedded in one `RETRIEVAL_QUERY` batch through the SQLite embedding cache, so a re-run makes no API calls.
- The golden-set file must be named `golden_set.v<N>.jsonl`; the version is recorded with the results.
- Per answerable question: run retrieval mode → ranked chunks → rank of each label = 1-based position of the first **chunk** that matches it (each label counted once; every chunk takes a position, including further parts of an already ranked section, since those also fill the `K_CONTEXT` slots) → metrics. Contract and spec: `evals/metrics.py`, `tests/unit/test_metrics.py`.
- **Recall@k** = |grade-2 labels ranked ≤ k| / |grade-2 labels|. Grade-1 labels don't count.
- **MRR** = 1 / rank of the best-ranked grade-2 label over the whole retrieved list (`K_DENSE`, `K_FTS` or `K_FUSED`, depending on mode); 0 if none.
- **nDCG@k** with gain `2^grade − 1` and discount `log2(rank + 1)` over labels ranked ≤ k; ideal DCG = all labels (both grades) sorted by grade, over the first `min(k, |labels|)` positions.
- `k` beyond the retrieved list: missing positions are not relevant. A question without a grade-2 label, a grade other than 1 or 2, or `k < 1` is an error (unanswerable items are skipped, not scored 0).
- Report means with `n`, and per-question rows for diffing. Metrics are reported at k = 5 and 10 (`recall@5`, `recall@10`, `mrr`, `ndcg@5`, `ndcg@10`); those cutoffs are part of the metric definitions, not a retrieval setting.
- **Results file** (default `eval/results/<UTC timestamp>-retrieval.json`, gitignored), validated by `schemas/eval.py:RetrievalRun`: `info` (date, repo git SHA + dirty flag (tracked files changed; untracked files don't count), golden-set version and sha256 (of the file's bytes with CRLF read as LF), index version id and config hash, FastAPI ref/SHA, embedding model/dim) and per config `retrieval_config` (the `RetrievalConfig` fields, §4), `retrieval_config_hash` (must equal the hash of that config and its `mode` must equal the config's name, or the file is rejected on read), `k`, `n`, `skipped_unanswerable`, mean `metrics` and per-question rows (`metrics`, `ranks` per label with `null` = not retrieved, `retrieved` section IDs in rank order).
- `--write-baseline` copies this run's configs (with each one's `retrieval_config_hash`, an optional field only so a row written before it existed can be read) into `eval/baselines/retrieval.json` (other configs' rows are kept) and warns if tracked files had uncommitted changes. **It refuses to mix setups:** before writing, every kept row is compared with the run on `golden_set_version`, `golden_set_sha256` and `index_config_hash` (the hash already covers the FastAPI SHA, the embedding model and dimension and the chunking config). A difference, or a legacy row without `golden_set_sha256`, raises `BaselineMismatchError`: the CLI prints "Baseline not updated: …" with the joint command (`--config dense --config fts …`) and exits 1. The results file is still written, and the baseline file is left byte-identical. Baseline numbers are never typed by hand.
- Deterministic given caches. No LLM calls (only query embeddings, cached).

### 15.3 Generation eval (promptfoo)
- **Version and command (D49, read from the npm registry, the GitHub release and the 0.123.1 source on 2026-10-08).** promptfoo is pinned at `0.123.1` (a stable release of 2026-09-18, `engines.node >=22.22.0`; the local Node 24.15.0 and the Node 24 of `actions/setup-node` both satisfy it). From the repository root, after `uv sync` in `backend/`. 4.05 ran the first two forms end to end on 2026-10-08; the third is the form `eval.yml` (4.10) takes:

  ```bash
  # Git Bash on Windows, Linux, macOS
  export PROMPTFOO_PYTHON="$(uv run --project backend python -c 'import sys; print(sys.executable)')"
  PROMPTFOO_DISABLE_TELEMETRY=1 PROMPTFOO_DISABLE_UPDATE=1 \
    npx promptfoo@0.123.1 eval -c eval/promptfoo/promptfooconfig.yaml -j 1 --no-cache -o eval/results/<name>.json
  ```

  ```powershell
  # PowerShell
  $env:PROMPTFOO_PYTHON = (uv run --project backend python -c "import sys; print(sys.executable)")
  $env:PROMPTFOO_DISABLE_TELEMETRY = "1"; $env:PROMPTFOO_DISABLE_UPDATE = "1"
  npx promptfoo@0.123.1 eval -c eval/promptfoo/promptfooconfig.yaml -j 1 --no-cache -o eval/results/<name>.json
  ```

  ```yaml
  # CI (Linux). `--yes`: npx asks before installing a package.
  env:
    PROMPTFOO_PYTHON: ${{ github.workspace }}/backend/.venv/bin/python
    PROMPTFOO_DISABLE_TELEMETRY: "1"
    PROMPTFOO_DISABLE_UPDATE: "1"
  run: npx --yes promptfoo@0.123.1 eval -c eval/promptfoo/promptfooconfig.yaml -j 1 --no-cache -o eval/results/generation.json
  ```

  `uv run --project backend python -c …` prints the interpreter of the backend venv, so no path is typed by hand. Two more variables are optional and described under "As built in 4.05": `EVAL_QUESTION_IDS` (run only some golden questions) and `GENERATOR_PROVIDERS=fake` (the canned stub instead of the model). What the pinned version does, and why each part is there:
  - **Interpreter.** promptfoo picks the provider's interpreter as `config.pythonExecutable`, then `PROMPTFOO_PYTHON`, then a detected `python`. The test generator and the file-based Python assertions are started with no per-call option, so only `PROMPTFOO_PYTHON` (or a `python` first on `PATH`) reaches them: set it once for the whole process. The value is the venv created by `uv sync --frozen` in `backend/`: `backend/.venv/bin/python` (Linux, macOS) or `backend\.venv\Scripts\python.exe` (Windows), as an absolute path. `grounded` is installed in that venv, so the shims in `eval/promptfoo/` import it. `uv run … npx` is not an alternative on Windows: it fails with "program not found" (checked 2026-10-08). Run end to end in 4.05: the test generator, the provider workers and every assertion process used that interpreter.
  - **`-j 1`** limits concurrent calls and also sets the Python provider's worker count to 1 (order: provider `config.workers`, `PROMPTFOO_PYTHON_WORKERS`, `-j`, then 1). The default concurrency is 4. Eval mode concurrency is 1 (§15.6).
  - **`--no-cache`** is required. The cache key of a Python provider is the provider script's hash, the prompt, the provider options and the test `vars`; the backend's prompt files, models and retrieval settings are not in it, so a cached run would answer "no change" after a degraded prompt, and the gate would never see it. Our own eval cache (§15.6) is the only cache.
  - **`PROMPTFOO_DISABLE_TELEMETRY=1`** and **`PROMPTFOO_DISABLE_UPDATE=1`** turn off the usage telemetry and the npm update check (promptfoo's telemetry page). The eval needs neither.
  - **`-o <path> [<path> …]`** writes results; the format comes from the extension (`csv`, `html`, `json`, `jsonl`, `txt`, `xml`, `yaml`, `yml`, `.junit.xml`). The JSON file feeds the gate, the HTML file is the CI artifact.
  - **Exit codes.** 0 when the pass rate reaches `PROMPTFOO_PASS_RATE_THRESHOLD` (default 100), `PROMPTFOO_FAILED_TEST_EXIT_CODE` (default 100) when any test failed, 1 for a configuration or run error, 130 for an interrupt. They are ignored (§15.5); a missing results file is the gate's exit 2.
  - **`npx --yes` in CI.** npx asks before installing a package; its docs say `--yes` suppresses the prompt and say nothing about non-interactive runs, so the workflow passes it.
- **Where the code is.** `eval/promptfoo/provider.py`, `asserts.py` and `tests_loader.py` are thin shims: promptfoo loads them by path, and they only re-export `call_api`, the four assertions and `generate_tests` from `backend/src/grounded/evals/promptfoo_provider.py`, `promptfoo_asserts.py` and `promptfoo_tests.py`. The logic lives in the package so that ruff, pyright strict and pytest cover it (CI runs them from `backend/` and would not see `eval/`); `tests/unit/test_promptfoo_shims.py` checks that everything the config names exists in the shims. `infra/event_loop.py` is the one place that picks the event loop psycopg async can run on (the CLI's `asyncio.run` and the provider's loop thread both use it).
- `eval/promptfoo/promptfooconfig.yaml`:
  - `prompts`: passthrough `{{question}}` (the real prompt lives in the backend).
  - `providers`: two `file://provider.py` entries, `no_rag` then `hybrid` (`hybrid_rerank` joins in Phase 6), each with `config: {mode: …}`. promptfoo starts one persistent Python worker per entry, and a results row names its config in `provider.label`. The mode is read from `config.mode`; the options promptfoo passes do not include the label.
  - `tests`: `file://tests_loader.py:generate_tests` with `config: {golden_set: golden_set.v1.jsonl}` (D49). promptfoo itself starts the function in a Python process of its own; it returns a list of test cases (`description`, `vars`, `assert`, `metadata`), and the YAML `config:` is passed as its one argument. The pre-step that writes `tests.generated.yaml` is **not** needed with 0.123.1 (confirmed: an error in the function stops the run with its message).
  - `commandLineOptions: {maxConcurrency: 1, cache: false}`: the two flags again, so a forgotten flag cannot turn a run concurrent or cached (checked: both are honored, promptfoo prints "Cache is disabled." and "concurrency: 1").
  - `defaultTest.assert`: the four deterministic assertions, then the two judge ones (`faithfulness`, `correctness`, 4.06), each a `type: python` with `value: file://asserts.py:<function>` and a `metric:` equal to the function name.
- `provider.py` (`call_api(prompt, options, context)`): runs the backend pipeline **in-process** and returns a dict with an own `output` (here the `AskResponse` as a JSON object) or an own `error`, plus `tokenUsage` (`total`, `prompt`, `completion`, `numRequests`), `cost`, `latencyMs` and `metadata`. The keys are camelCase as written: the provider result is not case-mapped. promptfoo runs a Python provider in persistent worker processes (module state survives between calls: verified, one process id and a call counter across a run) with a default call timeout of 5 minutes (`config.timeout` in milliseconds, or `REQUEST_TIMEOUT_MS`). It runs an `async def call_api` with a plain `asyncio.run` per call, which on Windows is the Proactor loop where psycopg async cannot run. So `call_api` is synchronous and hands the work to one event loop that lives in a thread of the worker, where the runtime (connection pool, query embedder, provider behind the eval LLM cache) is opened once, lazily, on the first call and closed at exit.
- Assertions (`asserts.py`, returning `{pass, score, reason}`). Each is `type: python`, `value: file://asserts.py:<function>`; promptfoo calls it as `fn(output, context)` (verified in 0.123.1). `output` is the provider's `output`, passed as the same object (a dict stays a dict). `context` has `vars`, `test` (the test case, so `context["test"]["metadata"]` is the test's metadata), `providerResponse` (with the provider's `metadata`, `tokenUsage`, `cost`, `latencyMs`) and `metadata` (the provider's `metadata` again). A function may return a bool, a number, or a dict with `pass`, `score`, `reason` and optional `named_scores`, `component_results` and any other keys, which promptfoo keeps in the component result (snake_case keys are also mapped to camelCase, the original stays). **Every file-based assertion call starts a new Python process**, not the provider's worker, so `promptfoo_asserts.py` imports only the standard library at the top (and `metrics.section_matches`, itself standard-library only, inside the one function that needs it); a test pins that importing it loads no pydantic, psycopg, settings or pipeline.

| Metric | How | Deterministic |
|---|---|---|
| Schema first-try validity | `meta` validation_retries == 0 | ✅ |
| Citation validity | invalid_citation_count == 0 (pre-filter count) | ✅ |
| Refusal correctness | answerable ↔ status ≠ `insufficient_context` | ✅ |
| Citation precision | share of cited chunks matching any labelled section (grade ≥ 1) | ✅ |
| Faithfulness | judge per claim: SUPPORTED / NOT_SUPPORTED given claim + cited chunk texts; score = supported / claims | judge |
| Answer correctness | custom Python grader (`asserts.py` → `evals/judge.py` → Groq adapter → eval cache, D50) vs `reference_answer`, 0 / 0.5 / 1 | judge |

- `no_rag` config: same schema and prompt family without sources (`answer_no_rag_v1`, §9.2; `AskMode.NO_RAG`, §7). Its claims have no citations, so the confidence cap applies to all of them. Faithfulness is N/A there; correctness and refusal are comparable.
- Latency p50/p95 and shadow cost per 1k questions are computed from provider metadata (warm, excluding cold start). A case with an eval LLM cache hit (`RequestTrace.llm_cache_hits > 0`, §15.6) is left out of the latency statistics, because a replayed reply has no provider latency, and stays in the cost: a hit returns the original call's usage. `metadata.cold_start` marks the first call of each worker process, which pays for the connection pool and the embedder.

**As built in 4.05** (a recorded promptfoo output with every shape below is `backend/tests/fixtures/promptfoo/results_sample.json`; `tests/unit/test_promptfoo_results_sample.py` pins it):
- **Environment of the provider process.** The shim sets `APP_ENV=eval` (forced) and `GENERATOR_PROVIDERS=gemini` unless the shell already set it, before `grounded.settings` is read: eval mode refuses the default two-provider list (§15.6), and the shell and `.env` are what the process sees. It is the one place outside `Settings` that touches the environment, and it only writes. `default_runner` refuses to run if the settings are not `eval`. **The fake switch** is `GENERATOR_PROVIDERS=fake`, the same canned stub as `grounded ask --fake` (§15.6, accepted in eval mode, refused in prod): the harness then runs end to end without a generator key. Hybrid still embeds the question: it reads `.cache/embeddings.sqlite`, and for a question missing there it calls the embedding API, unless the key is blank (`GEMINI_API_KEY=` in Git Bash; in PowerShell an empty assignment removes the variable, so a key in `.env` stays in force). A full fake run of the 30 questions and both configs took 40 s.
- **`EVAL_QUESTION_IDS`** (golden ids separated by commas or spaces, e.g. `q003,q045`) limits a run to those questions, in file order; an id that is not in the file is an error. It is read by `tests_loader.py` only, in the generator's own process; the backend package never sees it, and it is not a `Settings` field. Unset, the whole golden set runs.
- **Test cases.** `vars` holds only `question` (promptfoo prints every var as a column and sends them to the prompt). The golden payload is in the test's `metadata.golden`: `id`, `type`, `answerable`, `relevant` (label → grade), `reference_answer`, `golden_set_version`, `golden_set_sha256`. promptfoo gives the assertions the test's metadata (`context["test"]["metadata"]["golden"]`) and copies it into `results[].metadata.golden`, so no assertion process reloads the golden file. The file is only read. **promptfoo redacts** any string of 64 or more token characters in `results[].testCase` (so `testCase.metadata.golden.golden_set_sha256` is `"[REDACTED]"`) and not in `results[].metadata`: read the digest from the row-level `metadata.golden`.
- **Provider result, success.** `output` is the `AskResponse`; `tokenUsage.prompt` / `completion` are the trace's input / output tokens (`completion` already holds the thinking tokens), `numRequests` is `1 + validation_retries`; `cost` is the shadow cost (§14); `latencyMs` is the pipeline's own total. `metadata`: `mode`, `cold_start`, `status`, `provider`, `model`, `prompt_version`, `index_version`, `retrieval_config_hash`, `validation_retries`, `invalid_citation_count` (the count of the attempt that produced the returned answer, before the invalid labels were removed; an attempt that was retried shows in `validation_retries`), `dropped_claim_count`, `removed_url_count`, `llm_cache_hits`, `latency_ms` (total, embed, retrieval, llm), `shadow_cost_usd`, `tokens`, `claims` (text, display numbers `citations`, `chunk_ids`, server-side `confidence` and its components), `context` (what the model could cite: `label`, `chunk_id`, `section_id`, `anchor_path`, `url`, `breadcrumb` and the chunk `content`) and `retrieved_section_ids` (the whole retrieval in rank order). `context` and `retrieved_section_ids` are empty in `no_rag`. They come from two new fields of `RequestTrace` (`context_chunks`, `retrieved_section_ids`), filled by the pipeline; no column and no response field changed.
- **Provider result, failure: the tagged error.** A `ProviderError` (any subclass), `EmbedderUnavailableError`, `NoActiveIndexError`, `IndexMismatchError` or `psycopg.Error` becomes `error` = `"[<ErrorClass> quota=<true|false>] <short message>"` (the class name, `BackoffExhaustedError` included; the message is whitespace-collapsed and cut at 200 characters), with `metadata`: `error_kind` (the class name), `error_bases` (the names of its base classes below `Exception`, e.g. `["ProviderRateLimited", "ProviderError"]` for `BackoffExhaustedError`, so a reader needs no list of subclasses), `is_quota`, the common fields above (`mode`, `cold_start`, `validation_retries`, the citation counts, `llm_cache_hits`; `invalid_citation_count` is `null` when no attempt completed), `retry_after_s` for a rate limit, `waited_s` for `BackoffExhaustedError`, `validation_error` for `ProviderBadOutput`, and `rateLimitKind: "quota"` (below). `tokenUsage` and `cost` are what the request had spent before it failed (absent / the embedding only when nothing was). promptfoo keeps `error` and `metadata` on the row (`results[].error`, `results[].response.metadata`, merged into `results[].metadata`), marks it `failureReason: 2` and runs **no assertion** on it (`gradingResult` is `null`). So the gate counts a case's errors from `error_kind` and `is_quota` (which tags are "provider reasons" is 4.08's rule, §15.5), and a `ProviderBadOutput` row has no schema or citation scores: the gate counts it as a quality miss itself (§15.5), the parser only records it. Anything else (a bug, a missing key or model) is raised and shows up as a promptfoo error with the traceback.
- **`rateLimitKind: "quota"` is promptfoo's "do not retry" switch.** promptfoo's scheduler calls a provider again (up to 3 more times, with growing delays) when a result's `error` contains `429` or `rate limit`, unless `metadata.rateLimitKind` is `"quota"` (read in the source, and seen in the first scripted run, which waited on a "rate limited" message). Our messages can contain those words, and a repeat call would spend the quota-limited API again on top of the eval backoff (§15.6), so every tagged error and every skipped call carries it. It is a switch for promptfoo, not our tag.
- **A stopped run is not retried.** After a daily quota (`is_quota`) or a rejected request (`ProviderRequestRejected`), the provider answers every later call of that worker with the same `error_kind` and `is_quota`, `"… skipped: not asked, the run stopped on an earlier <ErrorClass>"` and `metadata.skipped: true`, without calling anything. A per-minute limit that the backoff gave up on, a timeout and a 5xx do not stop the run. The `no_rag` and `hybrid` workers are separate processes, so each finds the quota once.
- **Deterministic assertions** (`evals/promptfoo_asserts.py`; `pass` marks a perfect score, the metric is `score`, and the gate aggregates scores, not promptfoo's pass rate):
  - `schema_first_try`: `validation_retries == 0` → 1, else 0. Both configs.
  - `citation_validity`: pre-filter `invalid_citation_count == 0` → 1, else 0. Both modes' refusals count; N/A in `no_rag`.
  - `refusal_correctness`: answerable and `status != insufficient_context`, or unanswerable and `status == insufficient_context` → 1, else 0 (`partial` counts as an answer). Both configs.
  - `citation_precision`: distinct cited chunks (the response's `citations`) whose section matches a labelled section of grade ≥ 1 by `metrics.section_matches`, over all distinct cited chunks; the chunk's section comes from `metadata.context` by `chunk_id`. N/A in `no_rag`, for an unanswerable item (no labelled section; the retrieval metrics skip those too) and for an answer that cites nothing (a refusal).
  - **The not-applicable convention** (4.06's faithfulness uses the same, and adds the errored state: see "As built in 4.06"): a metric that does not apply returns `pass: true, score: 1.0`, a `reason` that starts with `"N/A: "` and **`not_applicable: true`**; every applicable result has `not_applicable: false`. An aggregation must skip every component result with `not_applicable` true; its `score` is a placeholder. promptfoo keeps the extra key in `gradingResult.componentResults[]`, next to `assertion.metric`, which names the metric. promptfoo's own per-metric averages (`namedScores`, the pass rate in its summary) include the placeholders and are not results.
  - Missing or ill-formed input (a `validation_retries` that is not an integer, a cited chunk that is not in `context`, an output that is not JSON) fails the assertion with a `malformed input: …` reason instead of raising.
- **Reading the results file.** `-o <name>.json` is `{evalId, results: {version, timestamp, prompts, results: […], stats}, config, metadata, vars, runtimeOptions}`; `config.tests` echoes every test case. One row per question and provider: `provider.label`, `testIdx`, `success`, `failureReason` (0 pass, 1 a failed assertion, 2 error), `error`, `response` (`output`, `metadata`, `tokenUsage`, `cost`, `latencyMs`), `gradingResult.componentResults[]` and `metadata` (the test's `golden` merged with the provider's metadata). Read the file as UTF-8 (`Path.read_bytes()` into `model_validate_json` is safe; the default encoding of `open()` on Windows is not).
- **The 4.05 run on real models** (3 questions × 2 configs, information only, n = 3 means nothing statistically) is in the 4.05 PR description, not here: numbers in this document come from committed eval output.

**As built in 4.06** (a recorded promptfoo output with the judge shapes is `backend/tests/fixtures/promptfoo/results_sample_judge.json`, scripted models; `tests/unit/test_promptfoo_judge_results_sample.py` pins it, and its README says how it was made):
- **The two assertions.** `faithfulness` and `correctness` are `type: python` assertions like the other four (`asserts.py` shim → `evals/promptfoo_asserts.py`, `metric` = function name). The function decides N/A and reads its input with the standard library only, and imports `evals/promptfoo_judge.py` (the judge, the Groq adapter and the eval cache) just for a case that needs a judge call: about a second of imports, most of it the Gemini SDK that `grounded.runtime` pulls in, paid by every judged assertion call, two per row. An N/A or malformed case pays nothing (a test pins it). The asserts shim now sets the same eval environment as the provider shim (`APP_ENV=eval`, `GENERATOR_PROVIDERS` defaulting to `gemini`), because the judge reads `Settings` in the assertion's own process.
- **Faithfulness** is one `Judge.judge_claims` call set per answer (§15.4): the claims come from `metadata.claims`, the sources of a claim from its `chunk_ids` resolved through `metadata.context` (label and chunk text). Score = supported claims / all claims, `pass` only at 1.0. A claim with no valid source is `NOT_SUPPORTED` without a call and counts in the denominator. **N/A** (same convention as above, plus `errored: false`): `no_rag`, a refusal (`status == insufficient_context`) and an answer with no claim. A per-question mean excludes N/A cases, and `n_faithfulness` is the number of cases that were scored (so it is never 0 or 1 for a case without sources).
- **Correctness** is one `Judge.judge_correctness` call: the question (`vars.question`), the golden `reference_answer` and `output.answer_markdown`, citation markers included (rubric rule 2 ignores them). 1.0 / 0.5 / 0.0 from `CORRECT` / `PARTIALLY_CORRECT` / `INCORRECT`, `pass` only at 1.0. It is never N/A: it is the metric that compares `no_rag` with `hybrid`, and a refusal of an unanswerable question is graded against its reference answer like any other answer.
- **The three states of a judge component.** Every judge component has `not_applicable` and `errored` (booleans, never both true), in `gradingResult.componentResults[]` next to `assertion.metric`. promptfoo keeps every extra key of the returned dict, nested lists and objects included (checked in the recorded run).

  | State | `not_applicable` | `errored` | `score` | `judge` |
  |---|---|---|---|---|
  | scored | false | false | the metric | the record below, `judge.error` null |
  | N/A | true | false | 1.0 (a placeholder, `pass` true) | absent; `reason` starts with `N/A: ` |
  | errored | false | true | 0.0 (a placeholder, `pass` false) | the record, `judge.error` set; `reason` is `judge error (<kind>): <detail>` |

  **A parser must leave out N/A and errored components** (`n` counts only the scored ones) and must not read promptfoo's own `namedScores` or pass rate (they include the placeholders). The `judge` object: `metric`, `prompt_version` (`name@hash` of the rubric), `judge_provider`, `judge_model` (all three `null` when no call was made), `usage` (`input_tokens`, `output_tokens`, `thinking_tokens`; for faithfulness the sum over the claims), `error` and, for faithfulness, `n_claims`, `n_supported`, `n_errored` and `claims[]`: per claim `claim_index`, `claim`, `cited_labels`, **`confidence`** (the server-side value, so 4.11 and 8.04 can join verdict and confidence without the provider's metadata), `verdict`, `reason`, `decided_locally`, `cache_hit`, `attempts`, `usage` and its own `error`. For correctness: `verdict`, `reason`, `attempts`, `cache_hit`. The order of `claims[]` is the order of `metadata.claims`.
- **What errored means, and what counts toward `inconclusive`** (4.08 fixes the rule; the tags are what it reads). A case is errored when the judge could not grade it, and it is never a score: for faithfulness when any of its claims is (a partly judged answer has no faithfulness; the verdicts that were obtained stay in `judge.claims`), and `judge.error` is then the first provider-side error if there is one, else the first. `judge.error.kind` is the exception's class name and:

  | `kind` | `provider_side` | `is_quota` | Meaning |
  |---|---|---|---|
  | `ProviderRateLimited` (a daily quota), `BackoffExhaustedError`, `ProviderUnavailable`, `ProviderTimeout` | true | true for the quota only | the provider failed: counts toward `inconclusive` |
  | `ProviderBadOutput` | false | false | invalid output after the one retry: unscored, not counted |
  | `ProviderRequestRejected`, `ProviderConfigError` | false | false | the judge cannot work (a refused key, a missing `GROQ_API_KEY` or `JUDGE_MODEL`, a judge on the generator's provider): "the gate could not run" |
  | `MalformedInput` | false | false | the row did not have what the assertion reads (a bug in the harness, the reason says what): unscored |

- **A judge error fails the assertion, not the row, and is never retried.** promptfoo marks the row `failureReason: 1` (a failed assertion) and copies the failing reason into the row's `error`, which for a judge error is `judge error (...)` and not a provider tag: use `failureReason == 2` for provider-errored rows and the component's `errored` for judge-errored ones. promptfoo's retry on `429` / `rate limit` applies to a provider call's `error` only: a judge reason that contains both words left the provider called once per row (checked). A Python assertion that raises becomes a failing result with the message (`Python code execution failed: ...`), score 0 and none of our keys, so the judge assertions catch what they can name and return an errored component; only a bug or `APP_ENV` not being `eval` is raised.
- **Judge sessions run one at a time, and a stop is shared.** promptfoo runs up to three assertions of a row at once (`PROMPTFOO_ASSERTIONS_MAX_CONCURRENCY`, default 3; `-j 1` does not limit it) and every one is a separate process, while the free Groq plan allows 8K tokens per minute. So a judge session holds an exclusive SQLite lock (`<CACHE_DIR>/judge_run.sqlite`, released with the process) from before its first call to after its last, and a daily quota, a refused key or a missing judge configuration is written to the same file under the run's id (`metadata.run_id`, one random id per test-generator call): every later judge assertion of that run answers `skipped: not asked, the run stopped on an earlier <kind>` with the same tag and makes no call, the way the provider treats a stopped run (§15.3 above). A per-minute 429 that the backoff gave up on is not a stop. Inside one answer `judge_claims` already stops asking after the first provider-side failure (§15.4).
- **The fake switch covers the judge.** With `GENERATOR_PROVIDERS=fake` the judge is `StubJudgeProvider` (every claim `SUPPORTED`, every answer `PARTIALLY_CORRECT`, a reason that says it is a stand-in, `judge_provider` `fake-judge`), built by `runtime.build_judge_provider`, refused with `APP_ENV=prod`, and never put behind the eval cache. The whole harness then runs with no network: 4 test cases (2 questions, both configs) took 10 s.
- **Timeout.** Judge calls take `EVAL_LLM_TIMEOUT_S` like the generator (§15.6).

**As built in 4.07, part 1: the normalized results and the parser** (`schemas/generation_eval.py`, `evals/generation_results.py:parse_results`; `tests/unit/test_generation_results.py` runs it on both recorded samples above and on malformed input). The gate reads promptfoo's `-o <file>.json` through this parser, which records and decides nothing about gating:
- **`GenerationRun`** (validated on read, `extra="forbid"`): `info` and `configs` by provider label. `info` has `date` (`results.timestamp`), `promptfoo_version`, `golden_set_version` and `golden_set_sha256` (from the row-level `metadata.golden`; a redacted digest is a parse error), `judge_provider`, `judge_model` and `judge_prompt_versions` (metric → `name@hash`; empty or `None` when no judge call was made), and `git_sha` / `git_dirty`, which promptfoo's file does not have: the reader supplies them, else they are `None`. A config has the identity of its answers (`provider`, `model`, `prompt_version`, `index_version`, `retrieval_config_hash`; each is the one value all its rows report, rows that disagree are a parse error) and its `cases`, sorted by question id, at most one per question. The results carry the index only as `index_version` (`<fastapi_ref>@<config_hash[:8]>`, as in `AskResponse.meta`), not as the full `index_config_hash`, so that label is what the gate compares.
- **A case** (`GenerationCase`) is either a failed generator call (`failureReason` 2): `error` set with `stage="generator"` and no metric; or a graded row (0 or 1), whose `metrics` hold a `MetricResult` per assertion, keyed by `assertion.metric`. It also has `type`, `cold_start`, `llm_cache_hits`, `latency_ms` (the pipeline's `latency_ms.total`, graded cases only), `input_tokens` / `output_tokens` and `cost_usd` (the shadow cost; a failed call keeps what it had spent).
- **A metric is `scored`, `na` or `errored`.** Scored has `value` (0..1; faithfulness also `claims` and `supported`). The placeholder score of an N/A or errored component is dropped: `value` is `None` there. Errored covers a judge component with `errored` (its `judge.error`), a deterministic assertion that reported `malformed input` (kind `MalformedInput`, stage `assertion`) and an assertion that raised, which has no `not_applicable` key (kind `AssertionFailed`).
- **`ErrorInfo`**: `stage` (`generator`, `judge`, `assertion`), `kind` (the exception's class name), `bases`, `is_quota`, `provider_side` (the judge's own flag; `None` for a generator error, which the gate classifies by `kind`), `skipped` (never asked, after a stop: the row's `metadata.skipped`, or a judge detail starting with "skipped:") and `detail` (whitespace-collapsed, cut at 500 characters). A generator error without the metadata keys falls back to the `[Kind quota=…]` tag of its text, and to kind `UnknownError` without a tag.
- **Not done by the parser:** a generator `ProviderBadOutput` is not turned into `schema_first_try = 0`, `namedScores` and the pass rate are not read, and promptfoo's exit code is not looked at.
- **A parse error** (`PromptfooResultsError`, a `ValueError`; the CLI exits 2): not JSON or not promptfoo's shape, no rows, a row without a provider label, a `failureReason` other than 0, 1 or 2, a graded row without assertion results, a metric twice in a row, a question twice in a config, rows from different golden sets, rows of one config that disagree on an identity field, a score outside 0..1.

### 15.4 Judge
- Provider: Groq (different from generator), pinned `JUDGE_MODEL` = `openai/gpt-oss-120b` (D48, verified 2026-10-08), temperature 0, prompts in `backend/prompts/judge_*_v1.md` with 2–3 worked examples each.
- The judge returns structured output (`{verdict, reason}`) through the same adapter layer (the Groq strict-mode notes are in §9.1).
- **Both judge metrics run through our code, not promptfoo's graders (D50).** Faithfulness and correctness are Python assertions that call `evals/judge.py`, which uses the Groq adapter and the eval LLM cache (§15.6). That keeps the pinned rubric prompt and its `prompt_version`, the typed provider errors that feed the `inconclusive` rule (§15.5), and cached re-runs. promptfoo's `llm-rubric` would send its own rubric prompt, and its built-in Groq grading provider would bypass the adapter and the cache.
- **Judge quota (Groq, `openai/gpt-oss-120b`, read 2026-10-08).** The Free Plan limits in Groq's public table are 30 RPM, 1K RPD, **8K TPM** and **200K TPD** (PRD §12); the account is on the Free Plan (the first real call in 4.02 returned `x-ratelimit-limit-requests: 1000` and `x-ratelimit-limit-tokens: 8000`, 2026-10-08). Consequences: judge calls are sequential (`-j 1`), a 429 is waited out for its `retry-after` within the bounded total wait (§15.6) and then ends the run as provider-errored, and a daily stop leaves the rest of the run unscored (`inconclusive`, §15.5), with everything already judged kept in the cache for the next day. Keep each judge request (rubric, claim, cited chunks, and the reasoning it asks for) well under the 8K per-minute budget: a request bigger than that cannot fit in any minute (how Groq reports it is unverified).
- **As built in 4.04** (`evals/judge.py`, verdict and judgment models in `schemas/judge.py`, the wiring in `runtime.py:open_judge`):
  - **Labels and scores.** The rubrics are the Author's (`backend/prompts/judge_faithfulness_v1.md`, `judge_correctness_v1.md`). Faithfulness returns `SUPPORTED | NOT_SUPPORTED`; correctness returns `CORRECT | PARTIALLY_CORRECT | INCORRECT`, which the code maps to 1.0 / 0.5 / 0.0 (`CORRECTNESS_SCORES`): the model never writes a number. Both return `{verdict, reason}` (`FaithfulnessVerdict`, `CorrectnessVerdict`: `extra="forbid"`, `reason` 1-600 characters, enforced by Pydantic after the call because Groq's strict mode does not enforce lengths, §9.1).
  - **Calls.** `Judge.judge_claim(index, claim, sources)`: one call per claim, with the claim and the text of the sources that claim cites, as `<source id="c1">…</source>` blocks (the answer prompt's blocks without `section` and `url`, with the same `escape_content`, so chunk or claim text cannot close a block or open a fake one; `{{…}}` in a value stays text, §9.2). `cited_sources(citation_ids, chunks)` maps a claim's labels to chunk text in citation order, skipping unknown and repeated labels. `judge_claims` runs a whole answer's claims in order, one at a time. `Judge.judge_correctness(question, reference_answer, answer)`: one call per answer; the three values are inserted as they are, because the correctness template has no `<source>` framing. Temperature is 0 by design (FR-22, `JUDGE_TEMPERATURE`, not a setting); `max_output_tokens` is `JUDGE_MAX_OUTPUT_TOKENS` and the timeout `Settings.call_timeout_s` (`EVAL_LLM_TIMEOUT_S` in eval mode, §4).
  - **A claim with no usable source is decided locally.** Rubric rule 6 gives `NOT_SUPPORTED` for "no source", so the judge returns that with a fixed reason and makes no call (the model could only spend quota on a fixed answer, or judge from its own knowledge). The judgment says `decided_locally=True`, `attempts=0` and zero usage, so a report can tell it from a model verdict. "Usable" means a source with any text; a claim whose labels are all unknown has none.
  - **Invalid output: one retry, then errored, never scored.** A `ProviderBadOutput` (invalid JSON, an unknown label, a reason over the bound, an extra key) gets exactly one more call on the same provider with the prompt's `# Retry feedback` section filled with the adapter's compact error (the same rule as `AskPipeline`, §9.5, and the same `MAX_VALIDATION_RETRIES`); a second failure, or one the adapter marks not retryable, makes the judgment **errored**: `verdict`, `reason` and the score are `None` and `error` carries the exception's class name as `kind`. The provider-side failures (`ProviderRateLimited` including `is_quota`, `BackoffExhaustedError`, `ProviderUnavailable`, `ProviderTimeout`) are errored judgments too, with `error.provider_side=True`; the judge does not retry them (waiting is the backoff's job, §15.6). Which kinds count toward `inconclusive` is 4.08's rule; the tags are there to read. `ProviderRequestRejected` (a bad key) is not caught: no judgment can be made and the run must stop. `judge_claims` stops asking after the first provider-side failure of an answer, so a daily quota is not asked again for each remaining claim: those claims come back errored with the same `kind` and `is_quota` (`attempts=0`, detail "not asked"); a claim that has no source is still decided locally, and a claim that fails on bad output does not stop the others.
  - **What a judgment records.** `prompt_version` (`name@hash` of the rubric file), `judge_provider`, `judge_model`, `usage` (the sum over the calls of that judgment, a failed first attempt included, `thinking_tokens` inside `output_tokens` on Groq), `cache_hit` (true only when no live call was made: a hit replays the original usage, §11), `attempts` (0, 1 or 2), and per claim `claim_index`, `claim`, `cited_labels`. Judgments are frozen Pydantic models, so 4.06 persists them with `model_dump`.
  - **Startup checks.** `open_judge(settings, eval_llm=None, provider=None)` builds the Groq adapter with `JUDGE_MODEL` (`build_judge_provider`: a blank value is a `ProviderConfigError` naming the variable; `GROQ_API_KEY` is checked by `build_groq_provider`), refuses a judge on the generator's provider (`check_judge_provider`, by provider name against `GENERATOR_PROVIDERS[0]`; `Settings` already refuses a `groq` generator in eval mode, this holds in every environment and for an injected provider), wraps it with `EvalLLM.wrap` (the cache and the backoff; pass the generator's `Runtime.eval_llm` to share one file and one set of counters) and closes the adapter on exit. It makes no call.
  - **Smoke run.** On 2026-10-08 the committed rubrics were run through this path against the real `openai/gpt-oss-120b` (`GROQ_REASONING_EFFORT=low`, `JUDGE_MAX_OUTPUT_TOKENS=800`), four calls: one supported and one contradicted claim over the real `q003` source section, and two candidate answers to `q003` (a complete one and a vague one). All four verdicts were the expected ones, and a second run over the same cache replayed them with no live call. The per-call token counts are in the 4.04 PR; four calls are a smoke test of the plumbing and the rubric text, not a measure of the judge (that is 4.11).
- **In the promptfoo run (4.06).** The judge is opened once per assertion process (`promptfoo_judge`, through `open_judge`, with `Settings.call_timeout_s`), not once per run: promptfoo starts a process for every assertion call, so judge sessions are serialized with a file lock, and a daily quota stops the judging of the run for all of them (§15.3, "As built in 4.06"). Observed on Groq in the 4.06 PR run (10 calls; the output is not committed, so read these as sizes, not results): a faithfulness call whose claim cites three chunks took about 1.9K input tokens, a correctness call about 1.3K, and the reply 100-150 output tokens, a third of them reasoning. The 8K tokens-per-minute limit therefore allows about four calls a minute, and the backoff waited out five short `Retry-After` windows (1-9 s) in a 56 s run of six cases.
- **Agreement check:** export ≥ 10 verdicts to `eval/judge_agreement/v1.csv`. The Author labels them by hand. Agreement % (and Cohen's κ if meaningful) goes to the README.

### 15.5 Aggregation and gate [A]
`uv run grounded eval gate --suite {retrieval|generation} --results … --baseline eval/baselines/….json`
- Output: `pass | fail | inconclusive`, a Markdown table (metric, baseline, current, Δ, threshold, ✅/❌) and exit code (0 pass/inconclusive, 1 fail). The report goes to stdout (CI appends it to the job summary). Exit code 2 means the gate **could not run** (a missing or invalid results/baseline file, a baseline row without a required field, or results that show a rejected key or a missing judge configuration), so CI can tell a broken input from a quality fail (§17).
- **Retrieval gate** (`evals/gate.py:evaluate_gate`, pure: `RetrievalRun` + the baseline rows → `GateReport`; code split: the function is [A], the Markdown and the exit code are boilerplate). The spec is in `tests/unit/test_gate.py`, the contract in the function's docstring:
  - A config is **gated** when its baseline row has `thresholds` (metric → `{"tolerance": t}`, the rule `current >= baseline - t`); a config without them is reported only (`·` in the table) and never changes the status. Today the thresholds are meant for `hybrid` (Recall@5, MRR, nDCG@5, tolerance 0.04). They live in the baseline file only, added in their own baseline PR.
  - It **fails closed, with a reason and never an exception**: a gated config or thresholded metric missing from the results; a run whose golden-set version, golden-set sha256 or index config hash differs from the gated row's (decided 2026-10-01: *fail*, not an error, because the PR is blocked either way and the table with the reason shows in the job summary; numbers from another setup are not compared); a baseline in which no config is gated (decided 2026-10-01: a gate that checks nothing must not look green).
  - Retrieval never calls a provider, so it is never `inconclusive`.
- **Generation gate** (`evals/gate.py:evaluate_generation_gate`, pure: `GenerationRun` + the baseline rows → `GenerationGateReport`, which is `GateReport` plus a `ConfigSummary` per config; the function is [A]; the Author delegated it, "write it", and 4.08 implemented it and explained it line by line in its PR). 4.07 delivered the contract (the function's docstring has every point below in full), the report types, the Markdown (`render_generation_markdown`), the CLI wrapper and the spec tests (`tests/unit/test_gate_generation.py`, strict `xfail` until 4.08, which removed the markers: all of them pass). A gated config is one whose baseline row has `thresholds`, as in the retrieval gate; the rest is reported only. Decided by the Author on 2026-10-08:
  - **Faithfulness is macro:** the mean of the per-question scores, each question weighing the same (`MetricResult` keeps `claims` / `supported` for faithfulness, which is what lets a test tell macro from micro).
  - **N/A is excluded:** a not-applicable metric (faithfulness for `no_rag`, a refusal or an answer with no claim; citation validity and precision for `no_rag`; citation precision for an unanswerable item or an answer with no citation) is out of the mean and out of its `n`, never 0 and never 1. `n` is per metric, so `n_faithfulness` differs from the overall `n`. A metric with `n = 0` has no value.
  - **Provider reasons:** a case counts toward `inconclusive` when its generator call, or one of its metrics, failed with `ProviderRateLimited` (a daily quota or a per-minute limit), `BackoffExhaustedError`, `ProviderUnavailable` (5xx) or `ProviderTimeout`; a case skipped after a daily quota carries the quota's kind and counts; a case counts once. More than 20% → `inconclusive`, exactly 20% is not. A generator `ProviderBadOutput` is a quality miss: it stays in `n` as `schema_first_try = 0` and its other metrics are unscored. A judge `ProviderBadOutput` leaves that judge metric unscored and does not count. `MalformedInput` and `AssertionFailed` (a harness bug) are unscored and reported. `ProviderRequestRejected` and `ProviderConfigError` anywhere mean the gate could not run: `GateCannotRunError`, exit 2.
  - **Inconclusive gates nothing:** its rows keep `current` and `n` with `passed = None`. All cases errored is inconclusive too (no division by zero).
  - Proposed in 4.07, not in the Author's list (the Author may veto any): the 20% is taken over the config's own cases, because the PR comment needs a verdict per config and `no_rag` and `hybrid` fail differently; `EmbedderUnavailableError` counts as a provider reason and any other unnamed kind (`NoActiveIndexError`, `IndexMismatchError`, a database error, `UnknownError`) is `GateCannotRunError`; latency p50/p95 are nearest-rank percentiles; a reported-only config never changes the status, inconclusive or not.
  - **Thresholds** (table below, stored as `GenerationThreshold` in the baseline row): a gated metric passes iff `current >= max(floor, baseline - tolerance)` over the parts that are set. A value exactly at the threshold passes. Metric names are the assertion metrics: `refusal accuracy` is `refusal_correctness`.
  - **Fail closed, with a reason and never an exception** (except `GateCannotRunError`): a gated config missing from the results; a golden-set version, golden-set sha256 or `index_version` that differs from the gated row's (numbers on other data are not compared; a changed prompt, model or retrieval config is compared, because catching it is the point; a config with no answered case has no index to compare, item 7 below); a thresholded metric with `n = 0` in a config that is not inconclusive; a baseline in which no config is gated.
  - **Status:** `fail` if there is a reason or a gated row failed, else `inconclusive` if a gated config is, else `pass`.
  - **Reported, never gated:** latency p50/p95 over the answered cases that are warm (no `cold_start`, no `llm_cache_hits`; §15.3) and cost per 1k questions (the mean `cost_usd` of the answered cases, cache hits included). The PR comment table has the metric rows, then a per-config line with `cases`, `n`, `n_faithfulness`, the error counts, latency and cost.
  - **As built in 4.08** (where the code says more than the lines above): (1) the rows of an inconclusive config have `threshold` `None` as well as `passed` `None`, so the table says "not gated" and a reader never sees a threshold that was not applied; (2) `schema_first_try` also gets a row when only generator bad outputs scored it, even if no case and no baseline row has the metric; (3) the 20% test compares exact fractions (`Fraction(provider_errors, cases) > Fraction(str(MAX_PROVIDER_ERROR_RATE))`): the limit is the decimal `0.20`, exactly one fifth, not the binary float nearest to it, so "exactly 20%" is an exact tie that no rounding decides; the nearest-rank index is computed in integers too, and means use `math.fsum`, so the order of the cases cannot move the last digit; (4) a value exactly on a threshold passes through the same `1e-9` absorption as the retrieval gate (`0.8 - 0.08` is `0.7200000000000001`), pinned by a sweep over every baseline from 0.50 to 1.00 in `test_gate_generation.py`; (5) `GateCannotRunError` is raised before anything else is computed, from every config of the run (reported-only ones too), with one reason per error kind in the form `<kind>: <n> case(s) (<config> <count>, …): <hint>`, sorted by kind; every kind that is not provider, bad output or harness is "cannot run", so an unclassified error stops the gate and is never counted as a quality result; (6) `ConfigSummary` is built for every config of the run, also one whose rows are withheld for an index or golden-set mismatch; (7) *added after 4.10c* (a fix to 4.08, found by the `eval_runs` work): a config with no answered case has no `index_version`, because the run reads it from the answers' metadata and a failed call carries none. That is "unknown", not "different", so the comparability check skips the index for such a config (the golden set, which is the run's own identity, is still compared) and the 20% rule judges it: a total provider outage (every generator call a 429, 5xx or timeout) is `inconclusive`, exit 0, not a `fail`. Every case a generator `ProviderBadOutput` is not provider-side and stays a quality result (`schema_first_try` 0 over `n`, the other gated metrics without a scored case, `fail`); a config that answered but names no index still differs.
- **Inconclusive:** if > 20% of a config's cases errored for provider reasons (429/quota, 5xx, timeout; the generator or the judge: exact definition in the generation gate below) → `inconclusive`. Metrics are shown with `n` but not gated. The PR comment says so explicitly.
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

- For generation, `grounded eval baseline` copies this table once into a new `hybrid` row (§15.7). The baseline file is the source of truth from then on; no code reads the table at run time.
- promptfoo's own exit code is ignored. The gate script is the single source of truth.

### 15.6 Eval mode (`APP_ENV=eval`)
Fallback off · temperature 0 · answer cache off · rate limit and budget off · LLM/rerank/embedding SQLite caches on · concurrency 1 (`-j 1`) · backoff on a 429 (honoring `Retry-After`) and on a transient 5xx (bounded total wait) · `request_logs.source='eval'` (CI DB only).

As built in 4.03 (the checks are the first two items; everything else is wiring in `runtime.py:open_runtime`):
- **One provider, enforced at startup.** `Settings` raises (a `ValidationError`, so the process does not start) when `GENERATOR_PROVIDERS` does not have exactly one entry, when that entry is `groq` (the judge's provider, AGENTS.md §6.5), or when `LLM_TEMPERATURE` is not `0`. Nothing is corrected silently, so `APP_ENV=eval` needs `GENERATOR_PROVIDERS=gemini` set explicitly (the default list is the router's `gemini,groq`). `fake` (the canned stub, for smoke runs) is accepted.
- **Answer cache off** (`AskPipeline` builds none). The caches that stay on are SQLite files (§11): the query embeddings (`.cache/embeddings.sqlite`, `build_query_embedder`) and the eval LLM cache below.
- **Rate limit and budget off.** They do not exist yet; 5.04 and 5.05 must bypass their check when `settings.app_env == "eval"` and their tests assert it (both tickets say so).
- **`request_logs.source='eval'`**: `open_runtime` builds the `RequestLogger` with it, so an eval request is told apart from `api`/`web` traffic in the table (DB.md §4; the CI database only).
- **The eval LLM cache and the backoff** (`generation/providers/eval_wrappers.py`). `open_runtime` wraps a provider it builds from the settings (not an injected one, and not the `fake` stub) as `CachingProvider(BackoffProvider(adapter))` through one `EvalLLM`; `EvalLLM.wrap(provider)` does the same for the judge (4.04), so the generator and the judge share the cache file and the counters. The cache is §11. The backoff sits behind it, so a hit never waits or calls:
  - A 429 with `is_quota` (a daily quota) is raised **at once**, as the original `ProviderRateLimited`. A per-minute 429 waits the advertised `Retry-After` (60 s when absent) and asks again. The waits of one call add up to at most `EVAL_MAX_TOTAL_WAIT_S` (the same bound as the 5xx waits below) and there are at most 3 of them; a wait that would pass the bound, or a fourth, raises `BackoffExhaustedError` (a `ProviderRateLimited` subclass with `is_quota=False` and `waited_s`) without sleeping. A timeout, a bad output and a rejected request pass through, and a failed call is not repeated. The sleep and the clock are injected, so no test sleeps.
  - **A transient 5xx is waited out too** (added after the first `push`-to-`main` run in CI, GitHub Actions run 37817266433: Gemini answered most `generateContent` calls with `503 UNAVAILABLE` "high demand", 25 of 30 `hybrid` and 21 of 30 `no_rag` cases were lost to it, and the run ended `inconclusive` for no quality reason; the same afternoon a local run saw no 503). A `ProviderUnavailable` (a 5xx or a transport failure) is retried up to `EVAL_UNAVAILABLE_RETRIES` times (4), waiting `EVAL_UNAVAILABLE_WAIT_S` (5 s) before the first retry and twice as long before each next one: 5, 10, 20, 40 s, 75 s in all. The wait is ours, not the server's: a `ProviderUnavailable` carries no `Retry-After`. The 5xx waits and the 429 waits of one call draw on the **same** `EVAL_MAX_TOTAL_WAIT_S` (75 s of 5xx waits leave 45 s for a 429 in the same call), and the two sequences count separately (a 429 does not advance the doubling); a retry whose wait would pass what is left is not taken. When the retries or the bound run out, the **original `ProviderUnavailable` is raised**, not a new type: the tag stays `ProviderUnavailable`, which the gate (§15.5), the judge (§15.4) and `ask --golden` already read as a provider-side 5xx (`BackoffExhaustedError` is a `ProviderRateLimited`, and `ask --golden` stops the run on it). `EVAL_UNAVAILABLE_RETRIES=0` turns it off. The retry sits in the same wrapper as the 429 backoff, below the cache: a reply that came after a wait is cached like any success, a failure never is, and `EvalStats.unavailable_waits` counts the waits. A timeout is not retried (a slow reply is not a capacity blip, and each retry would cost another `EVAL_LLM_TIMEOUT_S`), and neither are a bad output, a rejected request or a daily quota. The request path is untouched: it does not sleep (§9.5, §10).
  - This waiting is allowed because it is the eval path; the adapters never sleep and the request path never waits (§9.5). `grounded ask --golden` does not wait a second time for a `BackoffExhaustedError` (`evals/ask_batch.py`): it stops the run, like a quota.
  - **Provider-errored cases.** The error types already say it: `ProviderRateLimited` (a quota, or `BackoffExhaustedError` after the bounded wait), `ProviderUnavailable` (after its retries, above) and `ProviderTimeout` are provider-side failures, which is what the `inconclusive` rule counts (§15.5); `ProviderBadOutput` is a reply the model got wrong, and `ProviderRequestRejected` means the run cannot work (a bad key). The promptfoo provider (4.05, §15.3) tags a case with the exception's class name (and `is_quota`); 4.08 fixes which tags count.
- **Cache hits are marked, not hidden.** A replayed reply has `GenerationResult.cache_hit=True`, the original usage (tokens and shadow cost add up on a cached run as on a live one) and the lookup time as `latency_ms`. The pipeline puts the number of hits of a request on `RequestTrace.llm_cache_hits`; a case with a hit is left out of the latency percentiles (§15.3).
- **Counts at the end of a run.** `EvalLLM.stats` (`EvalStats`: `hits`, `misses`, `invalid_entries`, `rate_limit_waits`, `unavailable_waits`, `waited_s` (the total over both kinds of wait); `snapshot()` copies it, `render()` is the line to print) is on `Runtime.eval_llm`. `grounded ask --golden` prints it after the summary (`Eval LLM cache: 3 hits, 27 misses (30 calls); …`) when the run is in eval mode; the judge module uses the same object. The promptfoo provider does not print it: each case records its own `llm_cache_hits` in the result metadata (§15.3).
- **Call timeout.** In eval mode the pipeline and the judge pass `EVAL_LLM_TIMEOUT_S` (default 40 s, §4) to the provider instead of `LLM_TIMEOUT_S` (12 s), through `Settings.call_timeout_s`. Added in 4.06 after 4.05's real run, where 2 of 6 Gemini calls timed out at 12 s; a timeout is a provider-side error that counts toward `inconclusive`, so a limit tighter than the slow tail of the provider would make runs inconclusive for no quality reason. The timeout is not in the cache key, so no cached reply is lost.
- **Concurrency 1** stays a property of the callers: promptfoo `-j 1` (§15.3) and the sequential `ask --golden` (§15.8).

### 15.7 Baselines and history
- `eval/baselines/retrieval.json`, `eval/baselines/generation.json`: per config → metrics, `n`, thresholds, golden set version, prompt version, model IDs, index config hash, git SHA, date.
- `retrieval.json` today (`schemas/eval.py:RetrievalBaselineEntry`): per config `metrics`, `thresholds`, `n`, `k`, `golden_set_version`, `index_config_hash`, `retrieval_config_hash`, `fastapi_ref`/`fastapi_sha`, `embedding_model`/`embedding_dim`, `git_sha`, `git_dirty`, `golden_set_sha256`, `date`. `golden_set_sha256` and `git_dirty` are **required** (since 2.09): a row without them fails to load, and the gate exits 2. `retrieval_config_hash` is still optional (no gate rule uses it). `thresholds` (optional, default none) is the one hand-written part of a row: it maps a metric name (a key of `metrics`) to `{"tolerance": t}` with `t >= 0`. `--write-baseline` keeps a refreshed row's thresholds, and refuses a run without git state (`git_dirty` unknown). Adding thresholds to the committed file is a baseline PR.
- `generation.json` today: committed by 4.09b, with the rows `no_rag` and `hybrid`, written by `grounded eval baseline` (below) from the results file of the first full run (2026-10-08, one run of the 30 questions, `-j 1`). The numbers are in the file and in the README report block, not repeated here; only `hybrid` has thresholds (the §15.5 table, approved by the Author in the baseline PR). A row is `schemas/generation_eval.py:GenerationBaselineEntry` (read with `evals/generation_results.py:read_generation_baseline`): per config `metrics` (per-question means) and `n` (per metric, the questions it was scored on: a metric with no scored question has no entry in either), `thresholds`, `cases` (questions asked of the config), `golden_set_version`, `golden_set_sha256`, `provider`, `model`, `prompt_version`, `index_version`, `retrieval_config_hash` (optional), `judge_provider`, `judge_model`, `judge_prompt_versions`, `promptfoo_version`, `git_sha`, `git_dirty`, `date`, and the reported-only `latency_p50_ms`, `latency_p95_ms` (with `n_latency`) and `cost_per_1k_usd` (with `n_cost`), which are optional (a row without them prints `—`). Everything but `retrieval_config_hash`, `git_sha` (nullable) and the reported-only fields is required; metric names are the assertion metrics (`faithfulness`, `correctness`, `refusal_correctness`, `schema_first_try`, and the reported ones `citation_validity`, `citation_precision`). `thresholds` is the hand-written part: a metric maps to `{"tolerance": t}` (`current >= baseline - t`), `{"floor": f}` (`current >= f`, whatever the baseline) or both (`current >= max(f, baseline - t)`), and every key must be a metric of the row. A config with thresholds is gated. The gate compares `golden_set_version`, `golden_set_sha256` and `index_version` with the run's; prompt, model and retrieval-config identity is recorded for the reader (a prompt change is what the gate is there to judge).
- **Writing the generation baseline (4.09a, `evals/generation_baseline.py`).** `uv run grounded eval baseline --suite generation --results <promptfoo-output>.json [--baseline <file>]` (the retrieval rows are written by `eval retrieval --write-baseline`, §15.2, because that eval runs in-process; this one runs in promptfoo first, so writing is a second step on its results file). It reads the file with `parse_results` and the repo's `git_sha` / `git_dirty` (`repo_state()`), takes each metric's value, `n`, latency and cost from `evaluate_generation_gate(run, {})` (so a baseline row and a later gate row of the same results are the same numbers), and merges one `GenerationBaselineEntry` per config into the file, keeping the other configs' rows. Nothing is typed by hand (AGENTS.md §7). Choices of 4.09a, decided by the Agent for the Author to veto:
  - **Refused (exit 1, the file byte-identical, every problem listed):** a config that is `inconclusive`; any case that errored for a provider reason (a quota, a 5xx, a timeout, of the generator or the judge, a case skipped after a stop included); the judge could not run (`GateCannotRunError`, or no judge call at all); an assertion that reported a harness bug (`MalformedInput`, `AssertionFailed`); an unknown git state (`git_sha` / `git_dirty` missing, as for retrieval). A baseline comes from a fully scored run; a re-run is cheap because the eval cache (§15.6) replays what was already paid for. A generator `ProviderBadOutput` is a quality miss (a scored 0 for `schema_first_try`) and a judge `ProviderBadOutput` leaves one metric unscored: neither blocks a write, and both show in `n`. The command warns when tracked files are dirty (`git_dirty` is recorded).
  - **One setup per file.** Before writing, every kept row (a config the run does not have) is compared with the run on `golden_set_version`, `golden_set_sha256`, `index_version` (only between rows that use an index: `no_rag` has `none`), the generator (`provider` / `model`), `judge_provider`, `judge_model` and the judge prompt version of each metric both sides have. The retrieval rule (`BaselineMismatchError`) is widened because the RAG value subtracts the correctness of two rows, which must have been generated and judged alike. A difference refuses the write; the fix is one run of every config from one results file.
  - **Thresholds are policy, not measurements.** A `hybrid` row written for the first time gets `generation_baseline.INITIAL_THRESHOLDS`, the table of §15.5 as a constant (faithfulness floor 0.85 and tolerance 0.05, correctness tolerance 0.08, `refusal_correctness` tolerance 0.07, `schema_first_try` floor 0.95), which the Author approves in the baseline PR; `no_rag` and any other new config get none (reported only). A row that already exists keeps the thresholds it has, even none: the file is the source of truth afterwards. A threshold on a metric the run could not score refuses the write. After writing, the command prints the gate's verdict on the new rows against the same run (the 4.09 check "the gate on the new baseline against itself") and warns when it is not `pass`: a floor above the measured value (faithfulness 0.85 on a run that scored 0.80) fails its own gate, and that is the Author's call, not a reason to refuse.
  - **`git_sha` is HEAD when the baseline is written**, not when the eval ran (promptfoo's file does not record it): write it from the checkout that ran the eval, before committing anything else. `latency_*` is empty when every case was an eval-cache hit (a replayed reply has no provider latency, §15.3).
- **The report (4.09a, `evals/report.py`).** `uv run grounded eval report [--retrieval <file>] [--generation <file>]` prints the README's tables as Markdown from the committed baselines only: the retrieval ablation (dense / fts / hybrid with `n` and `k`) and the generation table (one column per config: every metric with its `n`, latency p50 / p95 with `n`, shadow cost per 1k with `n`), each with a provenance line (golden set version and sha256, index, models and prompt versions, git SHA and date; more than one value when the rows disagree). Hybrid is compared with the PRD §8 targets, `met` / `not met` on the unrounded value, and the **RAG value** row is `correctness(hybrid) - correctness(no_rag)` with both `n`, target `> 0`. The targets are constants in the module (`RETRIEVAL_TARGETS`, `GENERATION_TARGETS`, `RAG_VALUE_TARGET`), copied from PRD §8: change them together. A baseline file that does not exist yet is said so (`Generation baseline: not committed yet`), not an error. The output is deterministic (fixed order, 3 decimals, no clock). The README block between `<!-- eval-report:start -->` and `<!-- eval-report:end -->` is this output, pasted; `tests/unit/test_eval_report.py` fails when it differs from the output for the committed files, so a baseline PR re-pastes it.
- Updated only by an explicit PR titled `eval: update baseline (<reason>)`, with the before/after table in the description.
- Runs on `main` also insert into production `eval_runs` (owner connection via CI secret) for the dashboard: `uv run grounded eval record --suite generation --results <promptfoo-output>.json --branch <ref> --report-url <url> [--write]` (4.10c, `evals/eval_record.py`). It reads the results and the baseline as the gate does, takes the verdict and every number from `evaluate_generation_gate` (so a row and the PR comment of the same results agree), reads the full index hash from `DATABASE_URL` (the database the eval ran against), and builds one row per config (DB.md §4 says what each column holds). **It is a dry run unless `--write`** is given, and `--write` inserts into the database of `DATABASE_URL_DIRECT` (required, no fallback to `DATABASE_URL`), all rows in one transaction. It refuses (exit 1, nothing written) a checkout that is not git or has modified tracked files, results made on another index than the active one, unreadable files, and a missing owner connection; a run the gate cannot judge is exit 2. The step in `eval.yml` ships **disabled** (§17).

### 15.8 Golden set through `/ask` (Phase 3 closeout, `ask_batch.py`)
`uv run grounded ask --golden <golden_set.vN.jsonl> [--out <file>] [--limit N] [--fake] [--mode hybrid|no_rag]` asks every golden question through the same `AskPipeline` as `POST /v1/ask`. It is CLI tooling for the closeout and for manual smoke runs, **not an eval**: no metric is computed, the summary is informational and never a baseline.
- **Sequential, concurrency 1.** The pipeline never sleeps (§9.5, §10); the waiting lives in the CLI layer (`evals/ask_batch.py`). A per-minute 429 is waited out for exactly the advertised `Retry-After` (60 s if absent) and the same question is asked again, at most `--max-rate-limit-retries` times (default 3) per question. A daily quota, a `Retry-After` over `--max-wait-s` (default 120), a rejected request (bad key), a missing or mismatched index, a database error, or `--max-consecutive-failures` (default 3) provider-side failures in a row **stop the run**; what was answered so far is written and the reason is printed. A question whose answer fails validation twice is a recorded failure, not a stop.
- **Output.** The summary on stdout (status counts, schema-valid count over the file's questions, validation retries, invalid citations removed, dropped claims, URLs removed, cache hits, rate-limit waits, shadow cost of the answered questions (every attempt of each, a failed retry included), failures with reasons) and a JSON file, default `eval/results/<UTC timestamp>-ask.json` (gitignored), with the golden-set version and hash, the mode, `fake_provider`, the summary and one result per question, including its full `AskResponse`. "Schema-valid" means the response the pipeline built round-trips through `AskResponse`. That is not an independent check: the response is assembled from validated parts, so the evidence that matters is that the model output passed `LLMAnswer` validation in the adapter (with at most one retry) and the pipeline returned without an error. Read "30/30 schema-valid" as "30/30 answered without a validation failure". Only question ids are printed or written, never the text (AGENTS.md §6.13). The golden file is read-only, and `--out` refuses to be it.
- It does not write `request_logs` rows: it reads the pipeline's `RequestTrace` for the retry and citation counts instead. The answer cache stays on in `APP_ENV=dev`, so a re-run reports cache hits; use `APP_ENV=eval` (with `GENERATOR_PROVIDERS=gemini`, §15.6) to measure generation instead of the cache: it adds the eval LLM cache, so a re-run of an unchanged setup makes no provider call, and the run ends with the cache's hit and miss counts.

## 16. Testing strategy

| Level | What | Tools |
|---|---|---|
| Unit | chunker, include resolution, anchors/slugify, RRF math, metrics, citation mapping, confidence invariants, cache keys, normalization, cost calc, rate limiter, breaker state machine, gate logic | pytest (pure functions, table-driven) |
| Adapter | schema conversion per provider; response parsing from recorded **fixtures** (JSON files), error mapping (429 → `ProviderRateLimited`). One **shared contract test** (`tests/unit/test_provider_contract.py`) runs the same scenarios against every generator adapter, so they stay behaviorally identical: success, usage, bad output with its feedback, 429 per minute and daily, 5xx, other 4xx, timeouts, no retry or sleep inside the adapter, and (through the real pipeline in `no_rag` mode) the one retry with the feedback in the second request. Rigs in `tests/provider_rigs.py`: a fake SDK client for Gemini, the real `groq` client over an `httpx.MockTransport` for Groq. A fixture that cannot be recorded (a 429, a 5xx) is hand-made and marked `_synthetic` or listed as such in its README | pytest + fixtures, no network |
| Service | `/v1/ask` via `httpx.AsyncClient(app)` with `FakeLLMProvider`, fake embedder, real DB | pytest-asyncio |
| Integration | migrations apply cleanly; dense/lexical/hybrid SQL on a small fixture index (~30 hand-made chunks with known vectors); budget atomicity under concurrency | pgvector service container |
| Eval | retrieval eval + promptfoo (separate from pytest) | see §15 |

Rules: **no real network calls in pytest.** A socket-blocking fixture fails any test that tries. Fixtures are small and committed. Every bug fix comes with a failing test first.

## 17. CI/CD workflows

| Workflow | Trigger | Jobs |
|---|---|---|
| `warm-cache.yml` | `workflow_dispatch` (on `main`) | restore `.cache` → migrate → ingest pinned ref into a service DB → embed the golden questions (`eval retrieval --config dense`, scores unused) → save `.cache` (`if: always()`). Seeds and re-seeds the CI caches (below) |
| `ci.yml` | PR, push `main` | **backend**: `uv sync --frozen`, ruff check/format --check, pyright, pytest (pgvector service). **frontend**: `npm ci`, lint, typecheck, build. **retrieval-eval**: the shared setup (composite action `eval-index`: restore `.cache` (corpus + embeddings), migrate, ingest pinned ref into service DB) → retrieval eval (all configs) → gate vs `eval/baselines/retrieval.json` → Markdown report to the job summary, results JSON uploaded as the `retrieval-eval-results` artifact |
| `eval.yml` | PR `labeled` (only the `run-eval` label) or `synchronize` on a PR that carries `run-eval`; push `main`; `workflow_dispatch` (to seed or re-seed the main caches by hand, like `warm-cache.yml`) | one job `eval`: the shared setup (`eval-index`, above) → Node 24 (`actions/setup-node`) → restore the eval LLM cache → `npx --yes promptfoo@0.123.1 eval … -j 1 --no-cache` with `PROMPTFOO_PYTHON` set to the backend venv (§15.3), exit code ignored → `grounded eval gate --suite generation` → save the eval LLM cache → upload the `generation-eval-report` artifact (promptfoo HTML + JSON, gate report, log) → report to the job summary and, on a PR, to the one comment marked `<!-- grounded-eval -->` → verdict (gate exit 1 and 2 fail the job; on `main`, `grounded eval record` inserts `eval_runs`, shipped disabled, 4.10c). Details below |
| `ingest.yml` | `workflow_dispatch(ref, activate)` | ingest into Neon with `DATABASE_URL_DIRECT`; prints index stats |
| `housekeeping.yml` | daily cron + `workflow_dispatch` | retention SQL (DB.md §9) with `DATABASE_URL_DIRECT` |

Notes:
- Every third-party action in a workflow is pinned to a **commit SHA** with the version in a comment (`uses: actions/checkout@<sha> # v7.0.1`), read with `gh api repos/<owner>/<repo>/git/ref/tags/<tag>` (dereferencing annotated tags). A tag can move; a SHA can't. Bump them deliberately, not by a floating tag.
- GitHub disables scheduled workflows after 60 days without repo activity. Documented in README limitations.
- Secrets: `GEMINI_API_KEY`, `GROQ_API_KEY`, `COHERE_API_KEY`, `DATABASE_URL_DIRECT`. `eval.yml` uses `GEMINI_API_KEY` (the generator and the query embeddings) and `GROQ_API_KEY` (the judge); `DATABASE_URL_DIRECT` is referenced only by the disabled `eval_runs` step, only on a push to `main`, never in a PR run (4.10c). Backend deploys go through the Vercel Git integration, not a workflow. PR workflows only run for same-repo branches, so secrets are available.
- **Model IDs in `eval.yml` are literals (PRD D51).** The `env:` block of the workflow names `GEMINI_MODEL`, `GROQ_MODEL` and `JUDGE_MODEL` with their pinned values (§4), not repository variables or secrets. A model change is then a reviewed diff on a PR that carries `run-eval` anyway, and nothing outside the repository can change what a baseline means. `Settings` has no default for these three, so CI must set them.
- **Shared setup (4.10a).** `retrieval-eval` (ci.yml) and `eval` (eval.yml, 4.10b) share two composite actions in `.github/actions/`, so the jobs cannot drift apart. **`eval-index`**: `setup-uv`, `uv sync --frozen`, read the cache keys from `Settings` (`.github/scripts/cache-keys.sh`), restore the corpus, embeddings and tiktoken caches, migrate, ingest through `embedding-step.sh`, and save the corpus clone when the caller passes `save-corpus: 'true'` (a push to `main`). **`save-embeddings`**: the last step of the job, `if: always() && github.event_name == 'push'`. It reads the cache keys from `Settings` again instead of taking them from `eval-index`, so it works after an `eval-index` that failed half-way (a quota stop on the ingest is exactly when it matters) without relying on the outputs of a failed step. What stays in each job because a composite action cannot hold it: the checkout (the actions are read from it), the `services: db` pgvector container (job-level: keep the image in step with the `backend` job and `infra/docker-compose.yml`) and the env `APP_ENV: test`, `DATABASE_URL`, `TIKTOKEN_CACHE_DIR`. The tiktoken cache is restored under the same absolute path as in the `backend` job, because the path is part of a cache's version. `warm-cache.yml` still has its own copy of these steps: keep it in step with the action.
- **CI caches (PRD D45, checked 2026-10-01).** Two `actions/cache` entries, not one:
  - corpus clone `.cache/corpus`: key `corpus-<FASTAPI_REF>`;
  - `.cache/embeddings.sqlite`: key `embeddings-<EMBEDDING_MODEL>-<EMBEDDING_DIM>-<run_id>-<run_attempt>`, `restore-keys: embeddings-<EMBEDDING_MODEL>-<EMBEDDING_DIM>-`. The chunking config is **not** in the key: vectors are keyed by `sha256(text)` (§5.6), so a chunker change misses only the chunks whose text changed, and the file only grows. A key is immutable and `save` never overwrites one, hence the unique `run_id`-`run_attempt` suffix. `run_attempt` matters because a re-run keeps its `run_id`: with the run id alone, a re-run would match its first attempt's key exactly, restore that (possibly partial) cache instead of the newest one, and fail to save. Of several prefix matches, GitHub restores the most recently created.
  - Scope: a run restores caches of its own branch and of the default branch. A cache saved on a PR branch is invisible to `main` and other PRs, so **PR jobs restore only**. Only `main` pushes and the warm-cache workflow save.
  - `actions/cache` saves in a post step that runs only if the job succeeded. The jobs that save use `actions/cache/restore` and `actions/cache/save` with `if: always()`, so vectors paid for before a quota stop are kept.
  - GitHub evicts caches not used for 7 days (10 GB per repository, least recently used first). Every restore resets the clock, and `main` pushes restore it.
- **`eval.yml` (4.10b).**
  - **Triggers and concurrency.** A PR runs only for the `run-eval` label: `labeled` when the label just added is `run-eval` (another label must not start a paid run), `synchronize` when the PR carries it. `push` to `main` and `workflow_dispatch` always run. `permissions: contents: read, pull-requests: write` (the comment). Concurrency is set on the **job**, per PR (or ref), with `cancel-in-progress` for PR runs only: at the workflow level, a run whose job is skipped (some other label was added) would still cancel an eval in progress. Runs on `main` always finish. `timeout-minutes: 45` (a full run took 13 minutes locally at the 8K tokens-per-minute pacing of the free Groq plan); the promptfoo step has its own 38, so the steps that report and save still have time.
  - **Environment.** The setup steps run as in `retrieval-eval` (`APP_ENV: test`). The promptfoo step sets `APP_ENV=eval`, `GENERATOR_PROVIDERS=gemini`, the two keys from the secrets and `PROMPTFOO_DISABLE_TELEMETRY` / `PROMPTFOO_DISABLE_UPDATE`; the pinned `GEMINI_MODEL`, `GROQ_MODEL`, `JUDGE_MODEL` are literals in the job's `env:` (D51). The command adds `--no-table --no-progress-bar` to the one in §15.3 (checked against 0.123.1: they only shorten the log) and writes the JSON and the HTML in one go: `-o eval/results/generation.json eval/results/generation.html`.
  - **The gate decides the job, promptfoo's exit code is ignored** (§15.5; promptfoo exits 100 whenever one assertion failed). The promptfoo step records its code and keeps the log; the gate step records its code; the last step (`Verdict`) turns them into the job result **after** the cache, the artifact and the comment are out:

    | Situation | `status` | Job | Report |
    |---|---|---|---|
    | gate exit 0, headline `pass` | `pass` | success | ✅ |
    | gate exit 0, headline `inconclusive` (more than 20% provider errors, §15.5) | `inconclusive` | success, with a warning annotation | ⚠️ "inconclusive, not a pass", in the comment, the summary and the annotation |
    | gate exit 1 | `fail` | fails, error annotation "Generation gate failed" | ❌ |
    | gate exit 2, also when promptfoo crashed and left no results file | `cannot-run` | fails with exit 2, error annotation "could not run" | ❌ with the gate's message and the end of the promptfoo log |
    | the setup failed (for example exit 3, the embeddings quota: named in the job summary by `embedding-step.sh`) or the promptfoo step timed out | `not-run` | fails | ❌ "the eval did not run", never a pass |
    | cancelled (a newer push, a manual cancel) | `cancelled` | cancelled | ⚪ |
    | gate exit 0 with a report whose headline is not a known verdict | `unknown` | fails | ❓ |

    Exit 0 does not tell `pass` from `inconclusive`, so `.github/scripts/eval-report.sh` reads the first line of the gate's Markdown, `### Generation gate: <label>`; `tests/unit/test_gate_generation.py` pins that line.
  - **The PR comment.** `eval-report.sh` writes `eval/results/comment.md` (first line the marker `<!-- grounded-eval -->`, then the run's verdict, the gate's own Markdown with the metric table, `n`, the per-config summary and the inconclusive notice, an "Errors by kind" line (`.github/scripts/eval-error-kinds.py` counts the tagged errors of the results file: a quota, a 5xx or a timeout, of the generator and of the judge, which the gate's counts do not name), and a footer with the commit, the models, links to the run and the artifact, and the eval cache line) and appends the same text to the job summary. `.github/scripts/eval-comment.sh` lists the PR's comments, picks the first one written by `github-actions[bot]` that contains the marker, and edits it with `gh api -X PATCH`; if there is none it posts one. So a later push updates the comment and never adds a second. A failure to post is a warning (the verdict belongs to the gate, and the summary has the same text). The `report` and `comment` steps run with `always()`: a failed setup, a crashed or timed-out promptfoo and a cancel each get their own text.
  - **Artifact.** `generation-eval-report` (14 days): `generation.json` (the gate's input), `generation.html`, `gate-report.md`, `gate-errors.txt` and `promptfoo.log`. Missing files (a crash) are ignored.
- **The `eval_runs` insert (4.10c), disabled.** `eval.yml` has one step, "Record the run in eval_runs", that runs `grounded eval record ... --write` with `DATABASE_URL_DIRECT` (the production Neon owner URL) in its own `env:`. Guards: it runs only `if: github.event_name == 'push' && github.ref == 'refs/heads/main' && vars.EVAL_RECORD_RUNS == 'true'` and only for a run the gate gave a verdict (`pass`, `fail`, `inconclusive`; not "could not run", not "did not run"). The repository variable is **not set**, so the step is skipped (an unset variable is the empty string); the Author enables the write by setting `EVAL_RECORD_RUNS` to `true` in Settings, Secrets and variables, Actions, Variables (AGENTS.md §5, ticket 4.10). It is the only place `DATABASE_URL_DIRECT` appears in `eval.yml`, it never runs in a PR run, and `tests/unit/test_ci_workflows.py` pins all of it (the secret in no other step or workflow, the three conditions, `--write`, the order). Without `--write` the command is a dry run, so a mistaken local call writes nothing either. The step comes after "Report" and before "Verdict", so a failing insert is visible and the PR comment and artifact are out already.
- **The eval LLM cache in CI (4.10b).** One `actions/cache` entry for the one file `.cache/llm_eval.sqlite` (generator and judge replies, §15.6). `.cache/judge_run.sqlite`, the per-run judge-quota state (§15.3), sits next to it and is **not** cached: the cache `path` names only the one file, so a stopped run never carries over to the next. The calls in the file are content-addressed (provider, model, temperature, token limit, prompts, schema), so a restored entry is an identical call, never a stale answer, and the file only grows.
  - **Key.** `llm-eval-<scope>-<run_id>-<run_attempt>`, with `restore-keys` `llm-eval-<scope>-` and then `llm-eval-main-`, in that order. `<scope>` is `pr-<number>` in a PR run and the branch name otherwise (`main` for pushes and dispatches from `main`). A key is immutable and `save` never overwrites one, hence the unique `run_id`-`run_attempt` suffix, as for the embeddings.
  - **Why the scope is in the name.** A PR run reads the caches of its own merge ref, of its base branch and of the default branch, and what it saves is visible only to later runs of that PR. "Newest of all matches" would let a newer cache of `main` win over the PR's own earlier one, and the PR would pay again for its own changed calls. The ordered keys prefer the PR's own cache (the second push replays everything the first one paid for), and start the PR's first run from `main`'s cache (an unchanged setup, for example a CI-only PR that carries the label, costs no call).
  - **PR runs save too** (unlike the embeddings, D45): the point of this cache is the second push. It is saved with `if: always()`, so the calls paid for before a daily quota stop, a failure or a cancel are kept, right after promptfoo and the gate and before the report. It is saved only when the entry count changed: `.github/scripts/eval-cache-entries.sh` counts the rows before and after the eval, so a run that only replayed the cache does not add an identical copy to the 10 GB. The script first folds the SQLite write-ahead log into the main file (`wal_checkpoint(TRUNCATE)`): the caches run in WAL mode, and a process killed by a cancel or the timeout can leave committed rows only in the `-wal` file, which `actions/cache` would not save.
  - **Limit, accepted.** A PR's cache is invisible to `main`. After a merge that changed a prompt, a model or retrieval (the PR's eval paid for those calls), the first `main` run pays for them again; later `main` runs replay from `main`'s cache. A merge that changed nothing the eval sees (docs, CI) replays everything. Recorded in PRD §12.
- **Seeding and re-seeding.** A full ingest is ~1,045 chunk texts plus 25 answerable golden questions, which is over one day of the free embedding quota (1K texts per day, §5.6). `warm-cache.yml` (`workflow_dispatch`, `main`, `GEMINI_API_KEY`) runs the same ingest into a service pgvector DB and saves the cache even when the daily quota stops it. Run it on two consecutive days; day two sends only the ~70 texts still missing. The same workflow is the runbook after an eviction or a corpus/model change. The cache is derived data and is never committed (AGENTS.md §11) or uploaded by hand.
- **Infra failure is not a quality failure.** In `retrieval-eval`, a cold-cache miss that hits the daily quota, or a miss with no `GEMINI_API_KEY` (`EmbedderUnavailableError`), fails the ingest/eval step with a distinct exit code and a message that says how many texts are missing. The gate step runs only after those steps succeed, so only the gate step can mean "quality fail". The job summary names which of the two happened. It is never a skip and never a pass (AGENTS.md §7).
  - **Exit codes.** `grounded ingest` and `grounded eval retrieval` exit **3** (`EXIT_EMBEDDINGS_UNAVAILABLE` in `cli.py`) when the vectors could not be had: no key on a cold cache (a blank `GEMINI_API_KEY` counts as unset, which is what an unset GitHub secret looks like) or the daily quota. Any other failure is 1. `grounded eval gate` exits 1 for a quality fail and 2 when it could not run (§15.5). Exit 2 is also click's usage error, so 3 never collides with either.
  - `.github/scripts/embedding-step.sh "<title>" <command>` wraps the embedding steps (the ingest in `eval-index`, `grounded eval retrieval`): it passes stderr through, writes the exit code to the step output `exit_code` and, on failure, adds an "infrastructure failure" (3) or "step failed" (other) section with the message to the job summary. Its headings name the step through the title, not the eval, because the generation workflow uses it too. The corpus clone is saved only when `exit_code` is 0 or 3, so an interrupted clone is never saved under its immutable key.
  - The gate step writes the report to the job summary, then an error annotation for exit 1 (quality fail) or a "could not run" section for exit 2.
- **First runs of `eval.yml`.** The PR that adds the workflow runs it itself (a `pull_request` run uses the PR's own workflow file): add the `run-eval` label to it. On `main` it first runs on the merge push, and a manual run (`workflow_dispatch`, from `main`) seeds the main eval cache without a PR. The free Groq plan allows 200K tokens a day and a full judge run is about 145K (PRD §12), so a run on a day when part of that is spent ends `inconclusive` by design. Do not re-run hoping for another result: the cache keeps what was paid for, and the quota resets daily.
- **First run.** `workflow_dispatch` can only start a workflow file that is on the default branch, so `warm-cache.yml` has to be merged before it can be run. Until it has run on two consecutive days, `retrieval-eval` fails with exit 3 on every PR. Only after that is it made a required status check.
- CI configuration: the only secret for this is `GEMINI_API_KEY`. `DATABASE_URL` points at the service pgvector container (a plain env value, like the `backend` job). `EMBEDDING_MODEL` and `FASTAPI_REF` are **not** CI variables: they have defaults in `Settings`, which stays the single source.

## 18. Deployment

### Backend — Vercel Functions (FastAPI)
- A separate Vercel project with Root Directory `backend/`. Vercel runs the FastAPI app as one Vercel Function on Fluid compute and supports lifespan events (shutdown cleanup ≤ 500 ms).
- Hobby limits (checked 2026-09-24, re-verify in Phase 5): max duration 300 s, 2 GB / 1 vCPU, Python bundle ≤ 500 MB; monthly allotment 1M invocations, 4 h Active CPU (time spent waiting on I/O, e.g. LLM calls, doesn't count), 360 GB-h provisioned memory; runtime logs kept 1 h; non-commercial use only.
- Entrypoint: Vercel looks for a module-level FastAPI `app`. A thin entry module (`app = create_app()`) is referenced via `[tool.vercel] entrypoint` in `pyproject.toml`; `grounded.main` itself stays free of import-time side effects.
- Deploys via the Vercel Git integration: production from `main` only. Preview deployments get no production secrets.
- Production env: `APP_ENV=prod`, `DATABASE_URL` (app role, **pooled**), provider keys, `PROXY_SHARED_SECRET`, `IP_HASH_SECRET`.
- Serverless consequences: no long-lived process. The DB pool is per instance (Neon's pooler absorbs connections). In-process state (rate limiter, circuit breaker) is per instance: see PRD §12. Durable telemetry lives in `request_logs`, not in platform logs. The filesystem is ephemeral.
- Function region: set closest to Neon (AWS US East 2) in Phase 5.
- Startup: open pool, load the active index version, warm providers lazily. `/readyz` reflects readiness.
- Outside Vercel (local dev) the API runs with `grounded serve` (app factory, JSON logs, uvicorn access log off because it prints raw IPs).

### Frontend — Vercel
- Project root directory `frontend/`. Env: `BACKEND_URL`, `PROXY_SHARED_SECRET` (server-only, never `NEXT_PUBLIC_`).
- Route Handlers `app/api/ask/route.ts`, `app/api/metrics/route.ts`: forward to backend with the secret and `X-Client-IP` (from the platform's forwarded-for header), and a timeout slightly above `REQUEST_DEADLINE_S`.
- **Function duration limit:** set `export const maxDuration` in the ask route and verify the current Hobby-plan maximum. It must exceed the backend deadline plus a possible backend cold start, or the UI must show a retry state.

### Database — Neon
- Project `grounded`, Postgres 17, AWS US East 2 (Ohio), default branch `production`. Enable `vector`. Create role `app` (DB.md §5).
- Migrations and ingest via `workflow_dispatch` or locally with the direct URL.

## 19. Frontend

- Pages: `/` (ask), `/metrics` (dashboard, Phase 8), `/about` optional (links to README sections).
- Ask page states: idle · loading (after 5 s: "Waking up the free-tier backend…") · answered/partial · insufficient_context · rate_limited (with countdown) · budget_exhausted · error (with request_id).
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
| Cold start latency | infra | UI state, documented numbers |
| Stale corpus vs live docs | data | pinned tag shown in UI/meta; deliberate re-index |
| Judge disagreement with humans | eval | agreement measured and published |
| Small golden set noise | eval | n reported, tolerances ≥ 1 question |
| FTS is not true BM25 | retrieval | stated; ablation shows the effect |
