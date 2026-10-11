-- 0003_rate_limit.sql
-- Per-IP rate-limit counters: fixed windows in Postgres, not in memory (PRD D31, DB.md §4 and §7.4).
-- The `app` grant (SELECT, INSERT, UPDATE; no DELETE) is not here: the role is created by the
-- one-time runbook of ticket 5.09, which applies the grants of DB.md §5.
CREATE TABLE rate_limit_counters (
    ip_hash      text        NOT NULL,   -- HMAC-SHA256(ip, IP_HASH_SECRET); never a raw IP
    window_kind  text        NOT NULL CHECK (window_kind IN ('minute', 'day')),
    window_start timestamptz NOT NULL,   -- start of the UTC minute / UTC day, from the DB clock
    hits         integer     NOT NULL,
    PRIMARY KEY (ip_hash, window_kind, window_start)
);
