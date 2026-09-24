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
