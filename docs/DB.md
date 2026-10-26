# DB — Data model and database design

| | |
|---|---|
| Engine | PostgreSQL 17 + pgvector (Neon in production, `pgvector/pgvector:0.8.6-pg17-trixie` Docker locally and in CI) |
| Access | psycopg 3 (async), plain SQL, no ORM |
| Migrations | Forward-only numbered `.sql` files in `backend/migrations/` |
| Related | [PRD.md](PRD.md) · [Tech.md](Tech.md) · [../AGENTS.md](../AGENTS.md) |

---

## 1. Principles

1. **One database, two roles.** Index data (read-mostly) and operational data (logs, cache, budget, eval runs) live in one Postgres database. Least-privilege roles separate what the runtime can do from what ingest/migrations can do.
2. **Plain SQL is the interface.** Queries live in `.sql`-like string constants next to the code that uses them. Hybrid retrieval is a single readable SQL statement.
3. **Everything is versioned.** Chunks belong to an `index_version`. Requests and eval runs record which index version, prompt version and retrieval config produced them.
4. **Privacy by design.** No raw IP addresses. Question text has a retention limit. Public surfaces read only aggregates.
5. **Fit the free tier.** Neon free storage is small (0.5 GB per project at the time of writing; verify). The design stays well under 100 MB.

## 2. Environments and connections

| Environment | Database | Connection |
|---|---|---|
| Local dev | Docker `pgvector/pgvector:0.8.6-pg17-trixie`, db `grounded`, port `5433` | `DATABASE_URL=postgresql://grounded:grounded@localhost:5433/grounded` |
| Tests | Same container, db `grounded_test` (created/dropped by the pytest session) | `TEST_DATABASE_URL` |
| CI | GitHub Actions service container `pgvector/pgvector:0.8.6-pg17-trixie` | set in workflow |
| Production | Neon project `grounded` (AWS US East 2), branch `production` | runtime: **pooled** URL (`-pooler` host) as `app` role; migrations/ingest: **direct** URL as owner role |

Neon notes:
- Neon scales compute to zero after inactivity; the first query after idle pays a wake-up (~0.5–1 s). This is measured, not hidden.
- Runtime uses the pooled endpoint (PgBouncer, transaction mode). Neon's PgBouncer supports protocol-level prepared statements (PgBouncer 1.22+, `max_prepared_statements=1000`; Neon's connection-pooling docs, read 2026-10-11), which is what psycopg 3 uses, so the default is kept. If prepared-statement errors appear on the real pooler (5.10 checks), configure psycopg with `prepare_threshold=None`.
- Neon branches can be used for a throwaway staging copy (copy-on-write). Optional.

Pool settings (runtime): `psycopg_pool.AsyncConnectionPool(min_size=1, max_size=5, timeout=5)`. pgvector types are registered per connection via `configure=register_vector_async`.

## 3. Entity overview

```mermaid
erDiagram
    index_versions ||--o{ documents : contains
    index_versions ||--o{ chunks : contains
    documents ||--o{ chunks : "split into"
    index_versions ||--o{ answer_cache : "scopes"
    index_versions ||--o{ request_logs : "served by"
    index_versions ||--o{ eval_runs : "evaluated"

    index_versions {
        int id PK
        text git_ref
        char git_sha
        text embedding_model
        int embedding_dim
        jsonb chunking_config
        text status
        bool is_active
    }
    documents {
        bigint id PK
        int index_version_id FK
        text source_path
        text url
        text title
    }
    chunks {
        bigint id PK
        int index_version_id FK
        bigint document_id FK
        text section_id
        text[] anchor_path
        text[] breadcrumb
        text content
        vector embedding
        tsvector tsv
    }
    answer_cache {
        text cache_key PK
        jsonb response
        timestamptz expires_at
    }
    daily_usage {
        date day PK
        int llm_calls
        int rerank_calls
    }
    request_logs {
        uuid id PK
        text outcome
        int latency_total_ms
        numeric shadow_cost_usd
    }
    eval_runs {
        uuid id PK
        text suite
        text config_name
        jsonb metrics
        text status
    }
```

Tables by lifecycle:

| Group | Tables | Written by | Read by |
|---|---|---|---|
| Index | `index_versions`, `documents`, `chunks` | ingest CLI (owner role) | runtime, eval |
| Runtime state | `answer_cache`, `daily_usage`, `rate_limit_counters` (`0003`) | runtime (`app` role) | runtime |
| Telemetry | `request_logs` | runtime | dashboard views, retention job |
| Quality | `eval_runs` | CI on `main` only | dashboard |
| Meta | `schema_migrations` | migration runner | migration runner |

Not in the database, on purpose:
- **Embedding cache, rerank cache, eval LLM-response cache.** These are SQLite files under `.cache/` used by ingest/eval locally and, for the embedding cache, restored in CI via `actions/cache`. This keeps CI independent of production and avoids re-embedding the corpus on every run. The rerank cache is a local convenience for manual `hybrid_rerank` runs: no CI job restores or writes it (Tech §8, PRD D55), and it is never created in prod.

## 4. Schema (migrations `0001_init.sql` to `0005_dashboard_views.sql`)

```sql
-- 0001_init.sql
CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- Meta
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     text PRIMARY KEY,              -- e.g. '0001_init'
    checksum    text NOT NULL,                 -- sha256 of file contents
    applied_at  timestamptz NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- Index data
-- ---------------------------------------------------------------------------
CREATE TABLE index_versions (
    id               integer GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    corpus           text        NOT NULL DEFAULT 'fastapi-docs',
    git_ref          text        NOT NULL,                 -- tag, e.g. '0.1xx.y'
    git_sha          char(40)    NOT NULL,
    embedding_model  text        NOT NULL,                 -- e.g. 'gemini-embedding-001'
    embedding_dim    integer     NOT NULL CHECK (embedding_dim > 0),
    chunking_config  jsonb       NOT NULL,                 -- {"strategy":"headers","max_tokens":450,...,"excluded_pages":[...],"parser_version":1,"chunker_version":2}
    config_hash      text        NOT NULL,                 -- sha256(git_sha|model|dim|chunking_config)
    status           text        NOT NULL DEFAULT 'building'
                     CHECK (status IN ('building', 'ready', 'failed', 'retired')),
    is_active        boolean     NOT NULL DEFAULT false,
    document_count   integer,
    chunk_count      integer,
    token_stats      jsonb,                                -- {"min":..,"p50":..,"p95":..,"max":..}
    notes            text,
    created_at       timestamptz NOT NULL DEFAULT now(),
    ready_at         timestamptz,
    CONSTRAINT active_must_be_ready CHECK (NOT is_active OR status = 'ready')
);
-- At most one active index version.
CREATE UNIQUE INDEX index_versions_one_active ON index_versions (is_active) WHERE is_active;

CREATE TABLE documents (
    id                bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    index_version_id  integer NOT NULL REFERENCES index_versions(id) ON DELETE CASCADE,
    source_path       text    NOT NULL,     -- 'docs/en/docs/tutorial/background-tasks.md'
    url               text    NOT NULL,     -- 'https://fastapi.tiangolo.com/tutorial/background-tasks/'
    title             text    NOT NULL,     -- H1
    content_hash      text    NOT NULL,     -- sha256 of the resolved page markdown
    UNIQUE (index_version_id, source_path)
);

CREATE TABLE chunks (
    id                bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    index_version_id  integer  NOT NULL REFERENCES index_versions(id) ON DELETE CASCADE,
    document_id       bigint   NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal           integer  NOT NULL,              -- position within the document
    section_id        text     NOT NULL,              -- '<source_path>#<deepest anchor>' ('' anchor = page intro)
    anchor_path       text[]   NOT NULL,              -- ancestor anchors, outermost first: {'oauth2-with-password','hash-and-verify'}
    breadcrumb        text[]   NOT NULL,              -- {'Tutorial - User Guide','Security','OAuth2 ...','Hash and verify'}
    breadcrumb_text   text     NOT NULL,              -- breadcrumb joined with ' > ' (denormalized for FTS/prompt)
    heading_level     smallint,                       -- 1..6 of the deepest heading
    url               text     NOT NULL,              -- document url + '#' + deepest anchor
    content           text     NOT NULL,              -- markdown with resolved code, as shown to the LLM
    token_count       integer  NOT NULL CHECK (token_count > 0),
    content_hash      text     NOT NULL,              -- sha256(breadcrumb_text || '\n\n' || content) = embedding cache key
    embedding         vector(768) NOT NULL,           -- L2-normalized
    tsv               tsvector GENERATED ALWAYS AS (
                          setweight(to_tsvector('english', breadcrumb_text), 'A') ||
                          setweight(to_tsvector('english', content), 'B')
                      ) STORED,
    UNIQUE (index_version_id, document_id, ordinal)
);
CREATE INDEX chunks_index_version_idx ON chunks (index_version_id);
CREATE INDEX chunks_section_idx       ON chunks (index_version_id, section_id);
CREATE INDEX chunks_tsv_idx           ON chunks USING gin (tsv);
-- No ANN (HNSW) index initially: see §6.

-- ---------------------------------------------------------------------------
-- Runtime state
-- ---------------------------------------------------------------------------
CREATE TABLE answer_cache (
    cache_key             text PRIMARY KEY,           -- sha256(normalized_question|prompt_version|index_version_id|retrieval_config_hash|generator_model)
    normalized_question   text        NOT NULL,       -- user text: subject to retention
    response              jsonb       NOT NULL,       -- AskResponse without per-request meta
    prompt_version        text        NOT NULL,
    index_version_id      integer     NOT NULL REFERENCES index_versions(id) ON DELETE CASCADE,
    retrieval_config_hash text        NOT NULL,
    generator_model       text        NOT NULL,
    hit_count             integer     NOT NULL DEFAULT 0,
    created_at            timestamptz NOT NULL DEFAULT now(),
    last_hit_at           timestamptz,
    expires_at            timestamptz NOT NULL        -- <= created_at + 29 days (0001's comment says 30; see §9)
);
CREATE INDEX answer_cache_expires_idx ON answer_cache (expires_at);

CREATE TABLE daily_usage (
    day            date        PRIMARY KEY,           -- UTC day
    llm_calls      integer     NOT NULL DEFAULT 0,
    rerank_calls   integer     NOT NULL DEFAULT 0,
    input_tokens   bigint      NOT NULL DEFAULT 0,
    output_tokens  bigint      NOT NULL DEFAULT 0,
    updated_at     timestamptz NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- Telemetry
-- ---------------------------------------------------------------------------
CREATE TABLE request_logs (
    id                     uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at             timestamptz NOT NULL DEFAULT now(),
    source                 text NOT NULL CHECK (source IN ('web', 'api', 'eval')),
    question               text,                      -- NULLed after 29 days (§9)
    question_hash          text NOT NULL,             -- sha256(normalized question); kept for dedup stats
    ip_hash                text,                      -- HMAC-SHA256(ip, IP_HASH_SECRET); NULLed after 29 days (§9)
    http_status            smallint NOT NULL,
    outcome                text NOT NULL CHECK (outcome IN (
                               'answered', 'partial', 'insufficient_context',
                               'rate_limited', 'budget_exhausted', 'bad_request',
                               'validation_failed', 'provider_unavailable', 'internal_error')),
                                                      -- 0004 adds 'deadline_exceeded' (below)
    cache_hit              boolean  NOT NULL DEFAULT false,
    provider               text,                      -- 'gemini' | 'groq' | 'fake': the provider that answered; on a failed request the last one that reported back, else the primary (Tech §10, "Wired in 7.04"); NULL when the request was rejected before the pipeline (bad_request, rate_limited)
    model                  text,                      -- the model of that provider
    fallback_used          boolean  NOT NULL DEFAULT false,  -- a provider other than the first entry of GENERATOR_PROVIDERS was called (7.04): true for an answer that fell back and for a request that tried the fallback and failed; counted in v_request_stats_daily (§8)
    validation_retries     smallint NOT NULL DEFAULT 0,      -- retry attempts across providers (the router's one retry per provider after bad output), not retries of one provider
    rerank_used            boolean  NOT NULL DEFAULT false,  -- the context was built from the reranked order (6.03)
    rerank_error           text,                      -- why the RRF order stood although rerank is on: 'cap_reached' | 'rate_limited' | 'timeout' | 'unavailable' | 'rejected' | 'bad_output' (Tech §8); NULL otherwise
    prompt_version         text,
    index_version_id       integer REFERENCES index_versions(id) ON DELETE SET NULL,
    retrieval_config_hash  text,
    latency_total_ms       integer NOT NULL,
    latency_embed_ms       integer,
    latency_retrieval_ms   integer,
    latency_rerank_ms      integer,
    latency_llm_ms         integer,
    input_tokens           integer,
    output_tokens          integer,                   -- includes "thinking" tokens if the model reports them
    shadow_cost_usd        numeric(12, 8),
    citation_count         smallint,
    invalid_citation_count smallint,
    min_claim_confidence   real,
    error_code             text                       -- the API ErrorCode; no CHECK. A deadline or retrieval timeout (504) is outcome deadline_exceeded + error_code deadline_exceeded
);
CREATE INDEX request_logs_created_idx ON request_logs (created_at DESC);
CREATE INDEX request_logs_source_created_idx ON request_logs (source, created_at DESC);

-- ---------------------------------------------------------------------------
-- Quality
-- ---------------------------------------------------------------------------
CREATE TABLE eval_runs (
    id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at          timestamptz NOT NULL DEFAULT now(),
    suite               text NOT NULL CHECK (suite IN ('retrieval', 'generation')),
    config_name         text NOT NULL,     -- 'no_rag' | 'dense' | 'fts' | 'hybrid' | 'hybrid_rerank'
    git_sha             char(40) NOT NULL,
    branch              text NOT NULL,
    golden_set_version  text NOT NULL,     -- 'v1'
    prompt_version      text,
    index_config_hash   text NOT NULL,     -- index_versions.config_hash (CI builds its own DB, so no FK)
    generator_model     text,
    judge_model         text,
    status              text NOT NULL CHECK (status IN ('pass', 'fail', 'inconclusive')),
    metrics             jsonb NOT NULL,    -- {"recall@5":0.84,"mrr":0.71,...,"n":25}
    case_count          integer NOT NULL,
    errored_case_count  integer NOT NULL DEFAULT 0,
    report_url          text               -- link to CI artifact / run
);
CREATE INDEX eval_runs_latest_idx ON eval_runs (suite, config_name, created_at DESC);
```

**`0002_index_integrity.sql`** (Phase 2 review follow-up, approved 2026-09-29) adds two guarantees on top of the schema above:

```sql
-- At most one ready version per config (two concurrent ingests of the same config).
CREATE UNIQUE INDEX index_versions_one_ready_per_config
    ON index_versions (config_hash) WHERE status = 'ready';
-- A chunk's document belongs to the chunk's index version.
ALTER TABLE documents ADD CONSTRAINT documents_id_version_key UNIQUE (id, index_version_id);
ALTER TABLE chunks ADD CONSTRAINT chunks_document_same_version_fkey
    FOREIGN KEY (document_id, index_version_id)
    REFERENCES documents (id, index_version_id) ON DELETE CASCADE;
ALTER TABLE chunks DROP CONSTRAINT chunks_document_id_fkey;  -- superseded by the composite FK
```

The old single-column `chunks.document_id` FK is dropped because the composite one implies it. No data is lost. The `chunks` and `documents` DDL above is the `0001` shape; after `0002` a chunk is tied to a document *of its own version*, and a `retired` version no longer blocks rebuilding its config (only `ready` rows are unique).

**`0003_rate_limit.sql`** (applied; decided in 5.01, D31, written in 5.04) adds the per-IP rate-limit counters. The rate limiter is in Postgres because Vercel's instance count under a burst is not documented (PRD §12), so a per-instance counter would multiply the limit by an unknown number. The migration holds exactly this table; the `app` grant is not in it (the role is created by the one-time runbook of 5.09, which applies the grants of §5):

```sql
-- 0003_rate_limit.sql
CREATE TABLE rate_limit_counters (
    ip_hash      text        NOT NULL,   -- HMAC-SHA256(ip, IP_HASH_SECRET); never a raw IP
    window_kind  text        NOT NULL CHECK (window_kind IN ('minute', 'day')),
    window_start timestamptz NOT NULL,   -- start of the UTC minute / UTC day, from the DB clock
    hits         integer     NOT NULL,
    PRIMARY KEY (ip_hash, window_kind, window_start)
);
```

**`0004_deadline_outcome.sql`** (applied; decided in PRD D54b, written in R.20) adds `'deadline_exceeded'` to the `request_logs.outcome` CHECK. A request stopped by `REQUEST_DEADLINE_S` or by the retrieval stage timeout (`RETRIEVAL_TIMEOUT_S`, Tech §6) gets that outcome, with `error_code = 'deadline_exceeded'` and `http_status = 504`; before `0004` such a row reused `internal_error`, which mixed timeouts with bugs. It shipped before the first production `grounded migrate`, so no row needed a backfill. The CHECK is replaced, not altered (Postgres cannot change a CHECK in place); `request_logs_outcome_check` is the name Postgres generated for the inline CHECK of `0001`:

```sql
-- 0004_deadline_outcome.sql
ALTER TABLE request_logs DROP CONSTRAINT request_logs_outcome_check;
ALTER TABLE request_logs ADD CONSTRAINT request_logs_outcome_check CHECK (outcome IN (
    'answered', 'partial', 'insufficient_context',
    'rate_limited', 'budget_exhausted', 'bad_request',
    'validation_failed', 'provider_unavailable', 'internal_error',
    'deadline_exceeded'));
```

`tests/integration/test_schema.py` checks that the CHECK names exactly the values of `Outcome` (`observability/request_log.py`).

**`0005_dashboard_views.sql`** (decided by the Author on 2026-10-26, written in 8.01) adds no table and changes no column: it creates the three read-only views of the dashboard over `request_logs` and grants `app` SELECT on them when the role exists. The views and the reasons behind them are in §8, the grant in §5.

No secondary index: the table holds the windows of two days at most, so the retention delete (§9) scans a few thousand rows, and an index would only add write cost to a table written on every request. The pseudonymous `ip_hash` is kept 2 days here, well inside the 30-day rule of `request_logs`.

Design notes:
- **`chunking_config` keys.** The comment in the applied `0001_init.sql` (`"overlap":50`) is stale and can't be edited (an applied migration never changes). The stored keys are those of `ChunkingConfig` (`strategy`, `max_tokens`, `overlap_tokens`, `min_tokens`, `tokenizer`) plus `excluded_pages`, `parser_version` and (since Phase 2's keep-with-next fix) `chunker_version`. This document wins.
- **Why `vector(768)` is fixed.** pgvector column types carry the dimension. A different dimension means a new migration and a full re-index. This is intentional: it makes a model change a visible decision.
- **Why `breadcrumb_text` is denormalized.** Generated columns need immutable expressions. Joining an array inside the generated `tsv` expression is avoided, and the same string is reused in the prompt and in the embedded text.
- **Why FTS weights.** Headings (weight A) matter more than body text (B). `ts_rank_cd` respects weights.
- **Why `eval_runs.index_config_hash` has no FK.** CI builds its own index in a throwaway database. The config hash identifies the same index across databases.
- **What an `eval_runs` row holds (generation suite, written by `grounded eval record`, 4.10c).** One row per config of a run (`hybrid`, `no_rag`, ...), written by CI on a push to `main` with the owner connection (§5), aggregates only: never a question, an answer or a question id.
  - `git_sha` and `branch`: the commit of the checkout that ran the eval and its branch; `golden_set_version`, `prompt_version` (`name@hash` of the answer prompt), `generator_model` and `judge_model` come from the run; `report_url` is the link to the CI run.
  - `status` is the verdict of the **whole run** (the gate's: `fail` beats `inconclusive` beats `pass`, Tech.md §15.5), the same on every row of the run. A reported-only config such as `no_rag` has no verdict of its own.
  - `metrics` is flat: `"<metric>"` is the value and `"n_<metric>"` the questions it was scored on (a metric with no scored question has neither key), plus `"n"` (cases that did not error for a provider reason), `"latency_p50_ms"`, `"latency_p95_ms"`, `"n_latency"` and `"cost_per_1k_usd"` when they exist. Metric names are the assertion metrics (`faithfulness`, `correctness`, `refusal_correctness`, `schema_first_try`, `citation_validity`, `citation_precision`). The numbers are the gate's own.
  - `case_count` is the questions asked of the config; `errored_case_count` the cases that errored for a provider reason (a quota, a 5xx, a timeout: what the 20% `inconclusive` rule counts).
  - `index_config_hash` is the full `index_versions.config_hash` of the index the run used, read from the CI database; **`none`** for a config that retrieves nothing (`no_rag`), the word the baseline rows use as `index_version`.
  - No natural key: a re-run of a CI job is a new run and a new row (the dashboard reads the latest row per `(suite, config_name)`). The retrieval suite is not recorded by CI.
- **`request_logs.id` is the request's `request_id`** (the one in `meta.request_id` and in every log line), so a row and a log line correlate directly. A `bad_request` row has `question` NULL: the rejected text may be huge or not text at all; its `question_hash` is kept.
- **Why `request_logs.source` has no `ci`.** Evals run the pipeline in-process with `source='eval'` and write logs only to the CI database, never to production.

## 5. Roles and privileges

```sql
-- Run once per environment by the owner (Neon: the project owner role), after the migrations.
-- The role is created without a password; the password is set afterwards in psql from a variable,
-- `ALTER ROLE app PASSWORD :'app_password'` (generated, kept in a secret manager, never in a migration,
-- a file or a shell history). Plain text, not `\password app`: Neon does not accept a hashed password.
-- Created in SQL, not in the Neon console, CLI or API: those roles join `neon_superuser`, SQL roles do not.
CREATE ROLE app LOGIN;

GRANT USAGE ON SCHEMA public TO app;
GRANT SELECT ON index_versions, documents, chunks, eval_runs TO app;
GRANT SELECT, INSERT, UPDATE, DELETE ON answer_cache TO app;
GRANT SELECT, INSERT, UPDATE ON daily_usage TO app;
GRANT INSERT, SELECT ON request_logs TO app;
-- For the rate-limit table of migration 0003 (5.04); `SELECT` because `ON CONFLICT … DO UPDATE … RETURNING` reads the row.
-- No DELETE: the retention job runs as the owner (§9):
GRANT SELECT, INSERT, UPDATE ON rate_limit_counters TO app;
-- The dashboard views of migration 0005 (§8). 0005 grants this itself where `app` exists already;
-- on a fresh environment the role is created after the migrations, so this line grants it:
GRANT SELECT ON v_request_stats_daily, v_stage_latency_7d, v_request_metrics TO app;
```

| Role | Used by | Can |
|---|---|---|
| owner (Neon default) | migrations, ingest CLI, retention job, CI on `main` writing `eval_runs` | everything |
| `app` | FastAPI runtime | read index, write logs/cache/budget, read the dashboard views |

The runtime can never modify the index or delete telemetry.

The production procedure is `docs/runbooks/neon-production.md` (5.09). Its SQL block is a copy of the block above, and
`tests/unit/test_neon_runbook.py` fails when the two differ, so they change together.

**Why the view grant is in two places (8.01).** A `GRANT … TO app` fails where the role does not exist, and it does not
exist locally, in CI or on a fresh Neon branch before the runbook. So `0005` grants inside a guarded block
(`DO $$ … IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'app') THEN GRANT … END IF; … $$`): where `app` exists
(production, which got the role before `0005`), the migration grants; where it does not, the block does nothing and
the role block above grants the same SELECT when the role is created. Both orders end with the same privileges, and
`GRANT` can be repeated. `tests/integration/test_dashboard_views.py` applies `0005` without the role and with it.

## 6. Retrieval queries

The query specs below are **contracts**. The Author implements the lexical and hybrid queries in Phase 2
(`backend/src/grounded/retrieval/`). Agents should not hand over finished SQL for these (see AGENTS.md §3).

### 6.1 Dense (Phase 1, Agent)
- Input: query embedding (768, normalized), `index_version_id`, `k_dense` (20).
- Order by cosine distance `embedding <=> $query_vec` ascending, filtered by `index_version_id`.
- Output rows: `chunk_id, rank, distance`.
- **No ANN index at first.** With ~2–4k rows, an exact scan takes single-digit milliseconds and gives 100% recall. An HNSW index combined with a `WHERE index_version_id = …` filter post-filters its candidates and can silently return fewer than `k` rows. Scale path, if ever needed: HNSW (`vector_cosine_ops`) plus `SET hnsw.iterative_scan = relaxed_order` (pgvector ≥ 0.8), or a partial HNSW index per active index version. This trade-off is worth explaining in the README.

### 6.2 Lexical (Phase 2, Author)
- Input: raw question text, `index_version_id`, `k_fts` (20).
- **Pitfall to handle:** `plainto_tsquery` / `websearch_to_tsquery` use AND semantics. A natural-language question with 8 content words almost never matches a single chunk. Build an **OR** query from the question's lexemes instead. Approach: stem with `to_tsvector('english', q)`, extract lexemes with `tsvector_to_array`, quote each one, join with `|`, and build the query with `to_tsquery('simple', …)`. The `simple` config avoids stemming the already-stemmed lexemes twice.
- Empty query (only stopwords) → return no rows, not an error.
- Rank with `ts_rank_cd(tsv, query)` descending.
- Output rows: `chunk_id, rank, score`. `score` is the `ts_rank_cd` value and is kept as `RetrievedChunk.fts_score` (decided 2026-09-29), next to `fts_rank`. Ties in `score` break by chunk id ascending.

### 6.3 Hybrid with RRF (Phase 2, Author)
- One SQL statement with CTEs: `dense` (ranked), `lexical` (ranked), `fused` = full outer join on `chunk_id`.
- RRF score = Σ over lists of `1 / (k_rrf + rank)`, with `k_rrf = 60`. A missing rank contributes 0.
- Return the top `k_fused` (up to 40 unique chunks) with `rrf_score, dense_rank, dense_distance, fts_rank, fts_score` (kept for confidence and debugging; a signal is `NULL` for a chunk that is not in that list), joined with chunk fields needed downstream (`section_id, anchor_path, breadcrumb_text, url, content, token_count, content_hash`).
- Deterministic tie-break: `rrf_score DESC, chunk_id ASC`, applied before the cut to `k_fused`. Eval reproducibility depends on it.
- Each list is cut to its own K (`k_dense`, `k_fts`) *before* fusion. A question with no lexemes leaves the `lexical` list empty, so the result is the dense ranking (RRF over one list), not an error. `k_dense`, `k_fts`, `k_fused` and `k_rrf` come from `RetrievalConfig` (`rrf_k`), never literals.
- Code: `retrieval/hybrid.py:hybrid_search(conn, question, query_vector, *, index_version_id, cfg)`. The contract is its docstring, the spec is `tests/integration/test_hybrid.py`.
- Parameters are bound (`%s`), never string-formatted, including the vector.

## 7. Runtime state queries (Agent)

### 7.1 Daily budget: atomic reserve
```sql
-- Returns a row only if the call is within budget. No row => budget exhausted.
INSERT INTO daily_usage AS du (day, llm_calls)
VALUES ((now() AT TIME ZONE 'utc')::date, 1)
ON CONFLICT (day) DO UPDATE
    SET llm_calls = du.llm_calls + 1, updated_at = now()
    WHERE du.llm_calls < %(limit)s
RETURNING du.llm_calls;
```
Token totals are added after the call with a plain `UPDATE ... WHERE day = (now() AT TIME ZONE 'utc')::date` (`infra/budget.py`, 5.05; the `app` grant on `daily_usage` is exactly what the reserve, the advisory read and this update need, and `tests/integration/test_budget.py` runs them under a role with only that grant). `rerank_calls` uses the same pattern with its own cap (6.03, `infra/budget.py:RerankCap`, `RERANK_DAILY_CAP`):

```sql
INSERT INTO daily_usage AS du (day, rerank_calls)
VALUES ((now() AT TIME ZONE 'utc')::date, 1)
ON CONFLICT (day) DO UPDATE
    SET rerank_calls = du.rerank_calls + 1, updated_at = now()
    WHERE du.rerank_calls < %(limit)s
RETURNING du.rerank_calls;
```
No row back => the cap is reached. The insert branch takes the first unit of a day without looking at the limit, so a cap of 0 is decided in code and the statement is not run. There is no advisory check and no token total for rerank: the reserve is made right before each call, and a unit is never given back (a failed or cut-off call was attempted). The row of a day is shared with the LLM counters, which the two reserves leave alone (`tests/integration/test_rerank_cap.py` also runs it under a role with only the `app` grant).

**Advisory check (5.01, D31).** After the answer-cache lookup and before the query is embedded, a plain read decides whether to refuse early; it reserves nothing, so it cannot leak a unit:
```sql
SELECT llm_calls FROM daily_usage WHERE day = (now() AT TIME ZONE 'utc')::date;  -- no row => 0 calls today
```
`llm_calls >= DAILY_LLM_BUDGET` → `budget_exhausted` without an embedding. A request that passes can still lose the race at the reserve above, which stays the only thing that enforces the cap.

### 7.4 Rate limit: fixed-window counter (`0003`, 5.04; `infra/ratelimit.py`)
```sql
-- One statement per window; the returned count is this request's place in the window.
-- %(now)s is NULL in production (the database clock) and a fixed instant in tests.
WITH clock AS (SELECT coalesce(%(now)s::timestamptz, now()) AS t)
INSERT INTO rate_limit_counters AS rl (ip_hash, window_kind, window_start, hits)
SELECT %(ip_hash)s::text, %(kind)s::text,
       date_trunc(%(kind)s::text, clock.t AT TIME ZONE 'utc') AT TIME ZONE 'utc', 1
FROM clock
ON CONFLICT (ip_hash, window_kind, window_start)
DO UPDATE SET hits = rl.hits + 1
RETURNING rl.hits, rl.window_start, (SELECT t FROM clock) AS now;
```
- Windows are the UTC minute (`kind = 'minute'`, limit `RATE_LIMIT_PER_MIN`) and the UTC day (`kind = 'day'`, `RATE_LIMIT_PER_DAY`). `window_start` is computed in the statement from one clock, the database's `now()` (or an injected `now` in tests), never from the instance's, so two instances agree on the window, and it is truncated in UTC whatever the session time zone is.
- Order: the minute counter first. Only a request it admits touches the day counter, so a client that hammers the minute limit does not burn its daily quota with rejected requests. `hits > limit` → 429; `Retry-After` is the seconds to the end of that window (the `RETURNING` row gives the window start and the clock, so the remainder is computed without a second query). Both upserts of one request run in one transaction on one pooled connection, minute then day.
- Fixed windows allow up to twice the limit across a boundary (10 requests in two seconds at 5 per minute). The day counter and the global budget bound the harm; a sliding log would need a row per request.
- Atomic: the upsert takes the row lock, so parallel requests of one IP hash on different instances get distinct counts. Not run in eval mode, and not run for a request with no `ip_hash` (dev and test only, Tech §12).

### 7.2 Answer cache
- Read: `SELECT response FROM answer_cache WHERE cache_key = %s AND expires_at > now()`. On hit: `UPDATE … SET hit_count = hit_count + 1, last_hit_at = now()`. The runtime does both as one statement, `UPDATE … WHERE cache_key = %s AND expires_at > now() RETURNING response`: one round trip, and the row cannot expire between the read and the update.
- Write: `INSERT … ON CONFLICT (cache_key) DO NOTHING`, except that an **expired** row is replaced (`DO UPDATE … WHERE answer_cache.expires_at <= now()`, resetting `hit_count`, `created_at`, `last_hit_at` and `expires_at`). Expired rows stay until the retention job deletes them, and a plain `DO NOTHING` would leave their question uncacheable until then. A live row is never overwritten.
- The `cache_key` column comment above lists five parts; the key also includes a sixth, the hash of the confidence config, and a seventh, the hash of the generation parameters (Tech.md §11, PRD D47). They change only the hash input, not the column, so no migration is needed.
- Discard: a row whose `response` no longer validates as `AskResponse` is deleted on the lookup that finds it (`DELETE FROM answer_cache WHERE cache_key = %s`), so the same request's fresh answer can be stored; the write above never overwrites a live row.
- Only schema-valid responses with outcome `answered | partial | insufficient_context` are cached. Errors are never cached.
- The cache is disabled in eval mode.

### 7.3 Index activation (ingest, owner role)
Ingest marks a version `ready` (with `ready_at`) in the same transaction that stores and verifies it
(Tech.md §5.7). The unique index `index_versions_one_ready_per_config` (`0002`) means two concurrent ingests of one config can't both finish: the second one fails at this step, and ingest reports it as "another ingest already built this config; re-run to reuse it", without a `failed` row. Activation then only flips `is_active`, in one transaction, and only for a `ready` version:
```sql
UPDATE index_versions SET is_active = false WHERE is_active AND id <> %s;
UPDATE index_versions SET is_active = true WHERE id = %s AND status = 'ready';  -- 0 rows => error, rolled back
```
The runtime caches the active `index_version_id` and refreshes it every 5 minutes (or on restart).

## 8. Dashboard views (Phase 8, Agent)

Migration `0005_dashboard_views.sql` (8.01). The statements below are the ones in the file, without its comments
(`tests/unit/test_dashboard_views_docs.py` keeps the two identical):

```sql
CREATE VIEW v_request_stats_daily AS
SELECT
    (created_at AT TIME ZONE 'utc')::date                                  AS day,
    count(*)                                                               AS requests,
    count(*) FILTER (WHERE cache_hit)                                      AS cache_hits,
    count(*) FILTER (WHERE fallback_used)                                  AS fallbacks,
    count(*) FILTER (WHERE validation_retries > 0)                         AS validation_retries,
    count(*) FILTER (WHERE outcome = 'validation_failed')                  AS validation_failures,
    count(*) FILTER (WHERE outcome = 'insufficient_context')               AS refusals,
    count(*) FILTER (WHERE outcome IN ('rate_limited','budget_exhausted')) AS throttled,
    percentile_cont(0.5)  WITHIN GROUP (ORDER BY latency_total_ms)         AS p50_ms,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_total_ms)         AS p95_ms,
    sum(shadow_cost_usd)                                                   AS shadow_cost_usd,
    sum(input_tokens)                                                      AS input_tokens,
    sum(output_tokens)                                                     AS output_tokens
FROM request_logs
WHERE source IN ('web', 'api')
GROUP BY 1;

CREATE VIEW v_stage_latency_7d AS
SELECT
    percentile_cont(0.5)  WITHIN GROUP (ORDER BY latency_embed_ms)     AS embed_p50,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_embed_ms)     AS embed_p95,
    percentile_cont(0.5)  WITHIN GROUP (ORDER BY latency_retrieval_ms) AS retrieval_p50,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_retrieval_ms) AS retrieval_p95,
    percentile_cont(0.5)  WITHIN GROUP (ORDER BY latency_rerank_ms)    AS rerank_p50,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_rerank_ms)    AS rerank_p95,
    percentile_cont(0.5)  WITHIN GROUP (ORDER BY latency_llm_ms)       AS llm_p50,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_llm_ms)       AS llm_p95
FROM request_logs
WHERE source IN ('web', 'api') AND NOT cache_hit
  AND created_at > now() - interval '7 days';

CREATE VIEW v_request_metrics AS
SELECT
    created_at,
    outcome,
    http_status,
    cache_hit,
    fallback_used,
    validation_retries,
    provider,
    model,
    latency_total_ms,
    latency_embed_ms,
    latency_retrieval_ms,
    latency_rerank_ms,
    latency_llm_ms,
    input_tokens,
    output_tokens,
    shadow_cost_usd
FROM request_logs
WHERE source IN ('web', 'api');

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'app') THEN
        GRANT SELECT ON v_request_stats_daily, v_stage_latency_7d, v_request_metrics TO app;
    END IF;
END
$$;
```

- **No private column, no eval row.** No view selects `question`, `question_hash`, `ip_hash` or `id` (the request id),
  and every view keeps `source IN ('web', 'api')`: CI's `eval` rows are not users. The integration tests
  (`tests/integration/test_dashboard_views.py`) assert the exact column set of each view. A view reads `request_logs`
  with its owner's rights, so SELECT on the views is all a reader of the dashboard needs, and it reaches only the
  columns the view selects.
- **`v_request_stats_daily`**, one row per **UTC day**: `(created_at AT TIME ZONE 'utc')::date` does not depend on the
  session `TimeZone` (`date_trunc('day', created_at)` would cut the day at the session's midnight), and a `date` cannot
  be shifted into a client's zone. `validation_retries` counts the requests with at least one retry (`> 0`), not the
  retries; `validation_failures` counts the requests whose outcome is `validation_failed` (no provider's output passed
  validation). Both rates are shown, over `requests`.
- **`v_stage_latency_7d`**, one row: stage percentiles over the last 7 days, **without cache hits** (a cache hit skips
  the stages). A stage that did not run is `NULL` and `percentile_cont` ignores it, so `rerank_*` are over the
  reranked requests only; with no row at all every column is `NULL`.
- **`v_request_metrics`**, one row per user request, public columns only. The daily view cannot give a p95 over 7 or
  30 days (percentiles do not add up across days), so `GET /v1/metrics/summary` (8.02) aggregates this view over its
  window with a bound parameter, e.g. `WHERE created_at > now() - make_interval(days => %s)`, and computes stage
  percentiles with `FILTER (WHERE NOT cache_hit)`. The window filter can use `request_logs_source_created_idx`.
- **Why `percentile_cont`.** It interpolates between the two nearest values (the p95 of 100, 200, …, 700 is 670), so a
  small sample gives a number that moves smoothly as rows arrive; `percentile_disc` returns one of the observed values
  and jumps between them. Latencies are integers; the result is `double precision`.
- The views have no `ORDER BY`: a reader orders (`ORDER BY day`).

## 9. Retention and housekeeping

Run daily by `housekeeping.yml` (owner connection) through `grounded housekeeping`
(`backend/src/grounded/infra/housekeeping.py`, ticket 5.11; implemented and integration-tested with rows of
every age). The statements below are the ones in the module, with the periods bound as parameters
(`tests/unit/test_housekeeping_docs.py` keeps the two identical):

```sql
-- Privacy: strip user text and pseudonymous IDs after 29 days (the promise is 30, NFR-4).
UPDATE request_logs SET question = NULL, ip_hash = NULL
WHERE created_at < now() - make_interval(days => %(text_days)s)
  AND (question IS NOT NULL OR ip_hash IS NOT NULL);

-- Telemetry rows (no user text left) kept 180 days.
DELETE FROM request_logs
WHERE created_at < now() - make_interval(days => %(telemetry_days)s);

-- Expired cache, and cache rows whose question is older than the text boundary.
DELETE FROM answer_cache
WHERE expires_at < now()
   OR created_at < now() - make_interval(days => %(text_days)s);

-- Old usage counters.
DELETE FROM daily_usage
WHERE day < (now() AT TIME ZONE 'utc')::date - %(usage_days)s;

-- Rate-limit windows (0003): the day window is the longest one, so two days is enough.
DELETE FROM rate_limit_counters
WHERE window_start < now() - make_interval(days => %(window_days)s);
```

| Parameter | Value | Source |
|---|---|---|
| `text_days` | 29 | PRD NFR-4, D32, D56a |
| `telemetry_days` | 180 | this section |
| `usage_days` | 90 | this section |
| `window_days` | 2 | 5.01 (the day window is the longest) |

The periods are module constants, not `Settings` fields: they are the privacy contract, and a setting
could only lengthen them (`ANSWER_CACHE_TTL_DAYS` is capped at the same 29 days).

**29 days keep a promise of 30.** NFR-4 and the UI say user text is kept *at most 30 days*. The job runs once a day
and deletes what is older than its boundary, so with a boundary of 30 a row written just after one run was cleaned by
the run 30 days later and lived up to 31 (the review of 2026-10-12). With 29 the promise holds with a daily job; a
run that GitHub delays or an unswitched workflow still lengthens the period (the README says so). The cache rule
also deletes by `created_at`: `_PUT` refreshes it when it replaces an expired row, so a live row's `created_at` is
the time its question was stored, and rows written under the earlier 30-day TTL are cleaned too.

- **One transaction.** The five statements run in one transaction (one `now()` for all of them, all or nothing).
  The command prints one line per rule with the affected rows, in the order above (`request_logs_text_removed`,
  `request_logs_deleted`, `answer_cache_expired_deleted`, `daily_usage_deleted`, `rate_limit_counters_deleted`).
  `--dry-run` runs the same statements and rolls back, so the counts are the real ones.
- **Counts.** A row older than 180 days that still has its text is counted by the first statement (stripped) and by
  the second (deleted). A second run counts 0 everywhere. `day = today - 90` is kept (the rule is `<`).
- **Why the owner.** `app` has no DELETE on the log and counter tables and no UPDATE on `request_logs` (§5).

Index versions: keep the active one plus at most 2 previous ones. Older versions are set to `retired`
and then deleted by the ingest CLI (`grounded index prune`). Chunks and documents cascade.

## 10. Migrations

- Files: `backend/migrations/NNNN_short_name.sql` (4-digit, sequential). Forward-only; no down migrations.
- Applied so far: `0001_init` (the schema), `0002_index_integrity` (index integrity, R.05), `0003_rate_limit` (rate-limit counters, 5.04), `0004_deadline_outcome` (the `deadline_exceeded` outcome, R.20), all in §4, and `0005_dashboard_views` (the dashboard views and their guarded grant, 8.01; §8, §5).
- Runner (`grounded migrate`) applies pending files in order, each in its own transaction, and records `version` + `checksum` in `schema_migrations`.
- A pending file numbered below the newest applied one aborts the run (e.g. a late-merged branch): renumber it instead.
- **Never edit an applied migration.** If the checksum differs from the recorded one, the runner aborts. Fixes go into a new migration.
- Checksums are sha256 over the file with CRLF normalized to LF (`.gitattributes` also forces LF), so a Windows checkout doesn't look like an edit.
- The runner (`grounded.infra.migrations`) uses **sync** psycopg: it is offline tooling, not the request path. It takes a Postgres advisory lock so two concurrent runs can't interleave. Files run with no bind parameters, so psycopg uses the simple query protocol and a file may hold several statements.
- Destructive changes (dropping columns/tables) need an explicit Author decision and a note in the PR.
- Production migrations run from CI (`workflow_dispatch`): `.github/workflows/migrate.yml` runs `grounded migrate` and nothing else, from `main` only, with `DATABASE_URL_DIRECT` behind the same guard as `ingest.yml` (Tech §17; `ingest.yml` also migrates before it ingests). Or locally with the owner's direct URL. Never from the runtime process.

## 11. Capacity estimate

| Item | Estimate |
|---|---|
| Chunk row (768 × 4 B vector + ~2 KB text + ~1.5 KB tsvector + overhead) | ~7 KB |
| One index version (~3k chunks) + GIN index | ~25 MB |
| 3 retained index versions | ~75 MB |
| `request_logs` row | ~0.5 KB → 100k rows ≈ 50 MB |
| Total expected | < 150 MB (well under the 0.5 GB free limit) |

## 12. Local tooling

```bash
docker compose -f infra/docker-compose.yml up -d db
```

```bash
uv run grounded migrate
```

```bash
docker compose -f infra/docker-compose.yml exec db psql -U grounded -d grounded
```

Useful checks:
```sql
SELECT id, git_ref, status, is_active, chunk_count FROM index_versions ORDER BY id DESC;
SELECT section_id, token_count FROM chunks WHERE index_version_id = 1 ORDER BY token_count DESC LIMIT 10;
SELECT extversion FROM pg_extension WHERE extname = 'vector';
```
