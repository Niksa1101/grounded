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
| Production | Neon project `grounded`, branch `main` | runtime: **pooled** URL (`-pooler` host) as `app` role; migrations/ingest: **direct** URL as owner role |

Neon notes:
- Neon scales compute to zero after inactivity; the first query after idle pays a wake-up (~0.5–1 s). This is measured, not hidden.
- Runtime uses the pooled endpoint (PgBouncer, transaction mode). If prepared-statement errors appear, configure psycopg with `prepare_threshold=None`.
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
| Runtime state | `answer_cache`, `daily_usage` | runtime (`app` role) | runtime |
| Telemetry | `request_logs` | runtime | dashboard views, retention job |
| Quality | `eval_runs` | CI on `main` only | dashboard |
| Meta | `schema_migrations` | migration runner | migration runner |

Not in the database, on purpose:
- **Embedding cache, rerank cache, eval LLM-response cache.** These are SQLite files under `.cache/` used by ingest/eval locally and restored in CI via `actions/cache`. This keeps CI independent of production and avoids re-embedding the corpus on every run.
- **Rate-limit counters.** In memory, since there is a single backend instance (see Tech.md §12).

## 4. Schema (migration `0001_init.sql`)

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
    chunking_config  jsonb       NOT NULL,                 -- {"strategy":"headers","max_tokens":450,"overlap":50,...}
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
    expires_at            timestamptz NOT NULL        -- <= created_at + 30 days
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
    question               text,                      -- NULLed after 30 days
    question_hash          text NOT NULL,             -- sha256(normalized question); kept for dedup stats
    ip_hash                text,                      -- HMAC-SHA256(ip, IP_HASH_SECRET); NULLed after 30 days
    http_status            smallint NOT NULL,
    outcome                text NOT NULL CHECK (outcome IN (
                               'answered', 'partial', 'insufficient_context',
                               'rate_limited', 'budget_exhausted', 'bad_request',
                               'validation_failed', 'provider_unavailable', 'internal_error')),
    cache_hit              boolean  NOT NULL DEFAULT false,
    provider               text,                      -- 'gemini' | 'groq' | 'fake'
    model                  text,
    fallback_used          boolean  NOT NULL DEFAULT false,
    validation_retries     smallint NOT NULL DEFAULT 0,
    rerank_used            boolean  NOT NULL DEFAULT false,
    rerank_error           text,
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
    error_code             text
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

Design notes:
- **Why `vector(768)` is fixed.** pgvector column types carry the dimension. A different dimension means a new migration and a full re-index. This is intentional: it makes a model change a visible decision.
- **Why `breadcrumb_text` is denormalized.** Generated columns need immutable expressions. Joining an array inside the generated `tsv` expression is avoided, and the same string is reused in the prompt and in the embedded text.
- **Why FTS weights.** Headings (weight A) matter more than body text (B). `ts_rank_cd` respects weights.
- **Why `eval_runs.index_config_hash` has no FK.** CI builds its own index in a throwaway database. The config hash identifies the same index across databases.
- **Why `request_logs.source` has no `ci`.** Evals run the pipeline in-process with `source='eval'` and write logs only to the CI database, never to production.

## 5. Roles and privileges

```sql
-- Run once per environment by the owner (Neon: the project owner role).
CREATE ROLE app LOGIN PASSWORD '<from secret manager>';

GRANT USAGE ON SCHEMA public TO app;
GRANT SELECT ON index_versions, documents, chunks, eval_runs TO app;
GRANT SELECT, INSERT, UPDATE, DELETE ON answer_cache TO app;
GRANT SELECT, INSERT, UPDATE ON daily_usage TO app;
GRANT INSERT, SELECT ON request_logs TO app;
-- Added by the dashboard-views migration (Phase 8):
-- GRANT SELECT ON v_request_stats_daily, v_stage_latency_7d TO app;
```

| Role | Used by | Can |
|---|---|---|
| owner (Neon default) | migrations, ingest CLI, retention job, CI on `main` writing `eval_runs` | everything |
| `app` | FastAPI runtime | read index, write logs/cache/budget |

The runtime can never modify the index or delete telemetry.

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
- Output rows: `chunk_id, rank, score`.

### 6.3 Hybrid with RRF (Phase 2, Author)
- One SQL statement with CTEs: `dense` (ranked), `lexical` (ranked), `fused` = full outer join on `chunk_id`.
- RRF score = Σ over lists of `1 / (k_rrf + rank)`, with `k_rrf = 60`. A missing rank contributes 0.
- Return the top `k_fused` (up to 40 unique chunks) with `rrf_score, dense_rank, fts_rank` (kept for confidence and debugging), joined with chunk fields needed downstream (`section_id, anchor_path, breadcrumb_text, url, content, token_count, content_hash`).
- Deterministic tie-break: `rrf_score DESC, chunk_id ASC`. Eval reproducibility depends on it.
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
Token totals are added after the call with a plain `UPDATE`. `rerank_calls` uses the same pattern with its own cap.

### 7.2 Answer cache
- Read: `SELECT response FROM answer_cache WHERE cache_key = %s AND expires_at > now()`. On hit: `UPDATE … SET hit_count = hit_count + 1, last_hit_at = now()`.
- Write: `INSERT … ON CONFLICT (cache_key) DO NOTHING`.
- Only schema-valid responses with outcome `answered | partial | insufficient_context` are cached. Errors are never cached.
- The cache is disabled in eval mode.

### 7.3 Index activation (ingest, owner role)
Performed in one transaction:
```sql
UPDATE index_versions SET is_active = false WHERE is_active;
UPDATE index_versions SET is_active = true, status = 'ready', ready_at = now() WHERE id = %s;
```
The runtime caches the active `index_version_id` and refreshes it every 5 minutes (or on restart).

## 8. Dashboard views (Phase 8, Agent)

```sql
CREATE VIEW v_request_stats_daily AS
SELECT
    date_trunc('day', created_at)                                          AS day,
    count(*)                                                               AS requests,
    count(*) FILTER (WHERE cache_hit)                                      AS cache_hits,
    count(*) FILTER (WHERE fallback_used)                                  AS fallbacks,
    count(*) FILTER (WHERE validation_retries > 0)                         AS validation_retries,
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
```
The views ship in a later migration (`000N_dashboard_views.sql`). They never expose `question` or `ip_hash`.
Percentiles over cached requests are excluded from stage latency, since a cache hit skips the stages.

## 9. Retention and housekeeping

Run daily by `keepalive.yml` (owner connection):

```sql
-- Privacy: strip user text and pseudonymous IDs after 30 days.
UPDATE request_logs SET question = NULL, ip_hash = NULL
WHERE created_at < now() - interval '30 days' AND (question IS NOT NULL OR ip_hash IS NOT NULL);

-- Telemetry rows (no user text left) kept 180 days.
DELETE FROM request_logs WHERE created_at < now() - interval '180 days';

-- Expired cache.
DELETE FROM answer_cache WHERE expires_at < now();

-- Old usage counters.
DELETE FROM daily_usage WHERE day < (now() AT TIME ZONE 'utc')::date - 90;
```
Index versions: keep the active one plus at most 2 previous ones. Older versions are set to `retired`
and then deleted by the ingest CLI (`grounded index prune`). Chunks and documents cascade.

## 10. Migrations

- Files: `backend/migrations/NNNN_short_name.sql` (4-digit, sequential). Forward-only; no down migrations.
- Runner (`grounded migrate`) applies pending files in order, each in its own transaction, and records `version` + `checksum` in `schema_migrations`.
- A pending file numbered below the newest applied one aborts the run (e.g. a late-merged branch): renumber it instead.
- **Never edit an applied migration.** If the checksum differs from the recorded one, the runner aborts. Fixes go into a new migration.
- Checksums are sha256 over the file with CRLF normalized to LF (`.gitattributes` also forces LF), so a Windows checkout doesn't look like an edit.
- The runner (`grounded.infra.migrations`) uses **sync** psycopg: it is offline tooling, not the request path. It takes a Postgres advisory lock so two concurrent runs can't interleave. Files run with no bind parameters, so psycopg uses the simple query protocol and a file may hold several statements.
- Destructive changes (dropping columns/tables) need an explicit Author decision and a note in the PR.
- Production migrations run from CI (`workflow_dispatch`) or locally with the owner's direct URL, never from the runtime process.

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
