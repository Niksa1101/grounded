-- 0004_deadline_outcome.sql
-- A request stopped by the request deadline or the retrieval stage timeout gets its own outcome
-- (PRD D54b, DB.md §4). Until now such a row reused 'internal_error', which mixed timeouts with bugs
-- in every error-rate view. Shipped before the first production migrate, so nothing needs a backfill.
-- The constraint name is the one Postgres generated for the inline CHECK of 0001.
ALTER TABLE request_logs DROP CONSTRAINT request_logs_outcome_check;

ALTER TABLE request_logs ADD CONSTRAINT request_logs_outcome_check CHECK (outcome IN (
    'answered', 'partial', 'insufficient_context',
    'rate_limited', 'budget_exhausted', 'bad_request',
    'validation_failed', 'provider_unavailable', 'internal_error',
    'deadline_exceeded'));
