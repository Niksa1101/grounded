-- 0005_dashboard_views.sql
-- The dashboard's read side (Phase 8, ticket 8.01; DB.md §8): three views over request_logs that
-- expose aggregates or public, non-identifying columns only. None of them selects question,
-- question_hash, ip_hash or id, and every one leaves out source = 'eval' (CI runs, not users).
-- Days are UTC days whatever the session TimeZone is.

-- One row per UTC day.
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

-- One row: stage latency over the last 7 days. A cache hit skips the stages, so it is left out.
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

-- One row per user request, public columns only: GET /v1/metrics/summary (8.02) aggregates it over
-- a window it binds as a parameter (1d, 7d, 30d).
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

-- The runtime role reads the views. `app` exists only where the one-time role runbook ran (Neon
-- production), and a GRANT to a missing role is an error, so the grant is guarded: on a database
-- without `app` (local, CI, a fresh Neon branch before the runbook) this block does nothing, and the
-- runbook's role block (DB.md §5) grants the same SELECT when it creates the role.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'app') THEN
        GRANT SELECT ON v_request_stats_daily, v_stage_latency_7d, v_request_metrics TO app;
    END IF;
END
$$;
