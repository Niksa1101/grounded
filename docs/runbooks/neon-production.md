# Runbook: Neon production (migrate, `app` role, index)

Ticket 5.09. **The Author runs every step below.** An agent writes this file and the workflow
[`ingest.yml`](../../.github/workflows/ingest.yml) but never runs them against Neon production
(AGENTS.md §5, §11). Commands are copy-pasteable, one per block, and meant for Git Bash, Linux or
macOS. Never paste a password or a connection string into a file, a commit, a PR or a chat.

What you end up with: the schema (migrations `0001`–`0003`) in the Neon `production` branch, an
active index version, a least-privilege `app` role whose **pooled** URL is Vercel's `DATABASE_URL`,
and proof that `app` can do what the runtime needs and nothing more.

Facts checked on 2026-10-11 against Neon's docs (`neon.com/docs/manage/roles`,
`neon.com/docs/connect/connection-pooling`): the pooled hostname is the direct one with `-pooler`
added to the endpoint id; migrations should use the direct connection, because the pooler is
PgBouncer in transaction mode, which does not support session-level advisory locks (the migration
runner takes one, DB.md §10); roles created in the Console, CLI or API join `neon_superuser`, roles
created with SQL do not and get only what you grant. So `app` is created **with SQL**.

## Order of steps

| # | Step | Where |
|---|---|---|
| 1 | Secrets and the approval gate | GitHub |
| 2 | Check the embeddings cache | local `gh` |
| 3 | Run `ingest.yml` without activation (migrates and builds the index) | GitHub |
| 4 | Check the index version | local |
| 5 | Run `ingest.yml` with activation | GitHub |
| 6 | Create the `app` role and its grants | `psql` as owner |
| 7 | Least-privilege checks | `psql` as `app` |
| 8 | Set Vercel's `DATABASE_URL` | Vercel |
| 9 | Optional: record eval runs | GitHub |
| 10 | Housekeeping: first manual run, then enable the daily schedule | GitHub |

The role comes after the first workflow run because the grants name tables that the migrations
create.

## Prerequisites

- A Neon project `grounded` (Postgres 17, AWS US East 2), default branch `production`, with a
  database and the default owner role. Neon's console shows two connection strings per role: the
  **direct** one (`ep-…`) and the **pooled** one (`ep-…-pooler`). Keep both in your secret manager.
- `psql` (any recent version), `gh` (logged in to this repository), `vercel` (the CLI, step 8; the
  dashboard works too) and `uv` (run `uv sync` once in `backend/`).
- The extension `vector` is created by migration `0001` (`CREATE EXTENSION IF NOT EXISTS vector`),
  which the owner role may run on Neon. Unverified until the first run: if step 3 stops there with a
  permission error, run that one statement in the Neon SQL editor and re-run.

## 1. Secrets and the approval gate

The workflow reads two repository secrets and refuses to start without the first one:

| Secret | Value |
|---|---|
| `DATABASE_URL_DIRECT` | the owner role's **direct** URL (no `-pooler` in the host), with `?sslmode=require` as Neon shows it |
| `GEMINI_API_KEY` | already set for CI; used only for chunk texts the embeddings cache lacks |

Set the first one (the command prompts for the value, so it does not enter your shell history):

```bash
gh secret set DATABASE_URL_DIRECT
```

Check that both exist (names only):

```bash
gh secret list
```

Recommended: add a second confirmation. In the repository, Settings, Environments, create the
environment `neon-production` and add yourself under Required reviewers. The workflow's job declares
this environment, so every run then waits for your approval. Without it the environment is created
empty on the first run and changes nothing.

The same secret is the one `eval.yml`'s disabled `eval_runs` step would use (Tech §17); do not create
a second one.

## 2. Check the embeddings cache

A full ingest is about 1,045 chunk texts, more than one day of the free embedding quota (1K texts a
day), so the workflow depends on the cache that `warm-cache.yml` and the eval jobs keep on `main`.

```bash
gh cache list --key embeddings-
```

One or more `embeddings-gemini-embedding-001-768-…` rows mean it is warm. An empty list means it was
evicted (7 days unused): run `warm-cache.yml` on two consecutive days first (Tech §17), then come
back. If you skip this, the workflow stops at the ingest with exit code 3 and writes nothing to
Neon, and it saves the vectors it did embed; the next day's run sends only the rest.

## 3. Run the workflow without activation

The first run migrates the schema and builds the index version, but does not make it active, so
nothing serves it yet. Leave `ref` empty to use the pinned `FASTAPI_REF`.

```bash
gh workflow run ingest.yml --ref main -f activate=false
```

Approve the run if you set up the gate, then follow it:

```bash
gh run watch
```

In the run: the preflight step says how many texts the cache lacks (expect `0 texts to embed`), the
migrate step lists `0001_init`, `0002_index_integrity` and `0003_rate_limit` as applied, the ingest
step ends with `Stored index version N (ready).` and `Active: no`, and the job summary shows the
index versions table. The host of the connection string is masked in the log.

## 4. Check the index version

From a local clone, with the **direct** URL in your shell for this one session (typed, not saved):

```bash
read -rs DATABASE_URL_DIRECT && export DATABASE_URL_DIRECT
```

```bash
cd backend
```

```bash
uv run grounded index list
```

Expect one row: status `ready`, not active, the pinned ref and sha, `docs` and `chunks` counts
(about 1,045 chunks for the current tag), and token stats. Its `config` column (first 8 characters
of the config hash) equals the one `grounded index list` shows on a local database built from the
same commit, because the hash depends only on the corpus sha, the embedding model and dimension and
the chunking config. Then check the golden set against it (read-only):

```bash
uv run grounded golden validate ../eval/golden/golden_set.v1.jsonl --against-index
```

Stop here if anything differs, and see "When something fails" below.

## 5. Run the workflow with activation

The built version already exists, so this run embeds nothing: it reuses the version and flips the
active flag in one transaction (DB.md §7.3).

```bash
gh workflow run ingest.yml --ref main -f activate=true
```

```bash
gh run watch
```

The summary now shows `Active: yes` and a `*` in the active column. Check it from your shell:

```bash
uv run grounded index list
```

## 6. Create the `app` role and its grants

Do this after step 3, when the tables exist. Open a session as the owner on the **direct** URL:

```bash
psql "$DATABASE_URL_DIRECT"
```

Paste this block. It is the block of DB.md §5 (a test fails if the two differ), applied as the
owner. If `CREATE ROLE` says the role exists, a previous attempt got that far: carry on with the
`GRANT` lines, which can be repeated.

```sql
CREATE ROLE app LOGIN;

GRANT USAGE ON SCHEMA public TO app;
GRANT SELECT ON index_versions, documents, chunks, eval_runs TO app;
GRANT SELECT, INSERT, UPDATE, DELETE ON answer_cache TO app;
GRANT SELECT, INSERT, UPDATE ON daily_usage TO app;
GRANT INSERT, SELECT ON request_logs TO app;
GRANT SELECT, INSERT, UPDATE ON rate_limit_counters TO app;
```

Still in `psql`, set the password. Generate a long random one in your secret manager first and paste
it at the prompt. `\password` hashes it on your side, so the clear text is not sent to the server
and not stored in `psql`'s history:

```text
\password app
```

Leave `psql` with `\q`. Then build the runtime URL: take the owner's **pooled** string from the
Console and replace the user with `app` and the password with the one you just set. It has this shape, and goes into your secret manager as "grounded app pooled URL":

```text
postgresql://app:<password>@ep-<id>-pooler.us-east-2.aws.neon.tech/<database>?sslmode=require
```

A later grant (for example the Phase 8 views) is a new statement added to DB.md §5 **and** to this
file in the same PR, then applied here as the owner.

## 7. Least-privilege checks (the Author's manual check)

First, the catalog view, as the **owner** (open `psql "$DATABASE_URL_DIRECT"` again). It lists what
`app` may do on every table of the `public` schema:

```sql
SELECT c.relname AS table_name,
       string_agg(p.priv, ', ' ORDER BY p.ord) AS app_can
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace AND n.nspname = 'public'
CROSS JOIN (VALUES ('SELECT', 1), ('INSERT', 2), ('UPDATE', 3), ('DELETE', 4), ('TRUNCATE', 5))
    AS p(priv, ord)
WHERE c.relkind = 'r' AND has_table_privilege('app', c.oid, p.priv)
GROUP BY c.relname
ORDER BY c.relname;
```

The result must be exactly these rows (a test derives this table from DB.md §5). A table that is
missing from the result, such as `schema_migrations`, gives `app` nothing:

| table_name | app_can |
|---|---|
| answer_cache | SELECT, INSERT, UPDATE, DELETE |
| chunks | SELECT |
| daily_usage | SELECT, INSERT, UPDATE |
| documents | SELECT |
| eval_runs | SELECT |
| index_versions | SELECT |
| rate_limit_counters | SELECT, INSERT, UPDATE |
| request_logs | SELECT, INSERT |

Second, behave like the runtime. Connect as `app` through the **pooled** URL, which also proves the
URL and the pooler work:

```bash
read -rs APP_DATABASE_URL && export APP_DATABASE_URL
```

```bash
psql "$APP_DATABASE_URL"
```

**What `app` must be able to do.** One transaction, rolled back at the end, so production data is
untouched. It needs an active index (step 5). The two upserts run twice to exercise their
`ON CONFLICT … DO UPDATE` branch, which reads the row: the last of each must return `hits` /
`llm_calls` = 2.

```sql
BEGIN;
SELECT count(*) AS active_chunks FROM chunks
 WHERE index_version_id = (SELECT id FROM index_versions WHERE is_active);
INSERT INTO request_logs (source, question_hash, http_status, outcome, latency_total_ms)
 VALUES ('api', 'runbook-check', 200, 'answered', 1);
INSERT INTO answer_cache (cache_key, normalized_question, response, prompt_version,
                          index_version_id, retrieval_config_hash, generator_model, expires_at)
 SELECT 'runbook-check', 'runbook-check', '{}', 'runbook-check', id, 'runbook-check',
        'runbook-check', now() + interval '1 minute'
 FROM index_versions WHERE is_active;
UPDATE answer_cache SET hit_count = hit_count + 1 WHERE cache_key = 'runbook-check' RETURNING hit_count;
DELETE FROM answer_cache WHERE cache_key = 'runbook-check';
INSERT INTO daily_usage AS du (day, llm_calls) VALUES (date '2000-01-01', 1)
 ON CONFLICT (day) DO UPDATE SET llm_calls = du.llm_calls + 1, updated_at = now()
 WHERE du.llm_calls < 5 RETURNING du.llm_calls;
INSERT INTO daily_usage AS du (day, llm_calls) VALUES (date '2000-01-01', 1)
 ON CONFLICT (day) DO UPDATE SET llm_calls = du.llm_calls + 1, updated_at = now()
 WHERE du.llm_calls < 5 RETURNING du.llm_calls;
INSERT INTO rate_limit_counters AS rl (ip_hash, window_kind, window_start, hits)
 VALUES ('runbook-check', 'minute', date_trunc('minute', now()), 1)
 ON CONFLICT (ip_hash, window_kind, window_start) DO UPDATE SET hits = rl.hits + 1
 RETURNING rl.hits;
INSERT INTO rate_limit_counters AS rl (ip_hash, window_kind, window_start, hits)
 VALUES ('runbook-check', 'minute', date_trunc('minute', now()), 1)
 ON CONFLICT (ip_hash, window_kind, window_start) DO UPDATE SET hits = rl.hits + 1
 RETURNING rl.hits;
ROLLBACK;
```

**What `app` must not be able to do.** Each line is its own transaction, because a failed statement
aborts the one it is in. Every one must fail with `ERROR:  permission denied for table …` (the last
one with `permission denied for schema public`). If one of them succeeds or fails with a different
error (for example a NOT NULL violation, which means the permission check passed), `app` has too
much: go to "When something fails". The index, the telemetry and the counters are protected:

```sql
BEGIN; INSERT INTO chunks DEFAULT VALUES; ROLLBACK;
BEGIN; UPDATE chunks SET embedding = embedding WHERE false; ROLLBACK;
BEGIN; DELETE FROM chunks WHERE false; ROLLBACK;
BEGIN; TRUNCATE chunks; ROLLBACK;
BEGIN; DELETE FROM documents WHERE false; ROLLBACK;
BEGIN; UPDATE index_versions SET is_active = false WHERE false; ROLLBACK;
BEGIN; INSERT INTO eval_runs DEFAULT VALUES; ROLLBACK;
BEGIN; UPDATE request_logs SET question = NULL WHERE false; ROLLBACK;
BEGIN; DELETE FROM request_logs WHERE false; ROLLBACK;
BEGIN; DELETE FROM daily_usage WHERE false; ROLLBACK;
BEGIN; DELETE FROM rate_limit_counters WHERE false; ROLLBACK;
BEGIN; SELECT 1 FROM schema_migrations; ROLLBACK;
BEGIN; CREATE TABLE runbook_probe (x int); ROLLBACK;
```

Leave `psql` with `\q`, and unset the variables of this shell (`unset DATABASE_URL_DIRECT
APP_DATABASE_URL`). Paste the output (no URLs) into the PR of this ticket or a follow-up, as the
evidence the acceptance criteria ask for.

## 8. Set Vercel's `DATABASE_URL`

In the **backend** Vercel project, Production environment only, set `DATABASE_URL` to the pooled
`app` URL of step 6, marked as sensitive. The command prompts for the value:

```bash
vercel env add DATABASE_URL production
```

Preview deployments must not get it (Tech §18: a preview uses placeholders). The rest of the
production variables (`APP_ENV=prod`, `PROXY_SHARED_SECRET`, `IP_HASH_SECRET`, `DAILY_LLM_BUDGET`,
`GEMINI_API_KEY`, `GEMINI_MODEL`, …) and the first deploy are ticket 5.10. The runtime never gets
`DATABASE_URL_DIRECT`: migrations and ingest run only from this workflow or from your shell
(DB.md §10).

## 9. Optional, last: record eval runs

Only after the steps above are verified, you may let CI record generation-eval runs in the
production `eval_runs` table (it uses the owner URL, `DATABASE_URL_DIRECT`, on a push to `main`
only; Tech §17, ticket 4.10). It is off until you set the repository variable:

```bash
gh variable set EVAL_RECORD_RUNS --body true
```

Skip it if you are not ready; nothing else depends on it.

## 10. Housekeeping: retention and keep-warm (ticket 5.11)

[`housekeeping.yml`](../../.github/workflows/housekeeping.yml) enforces the privacy retention in
production (question text and IP hashes removed after 30 days, logs deleted after 180, expired cache
rows, old usage and rate-limit windows; DB.md §9) and pings the backend's `/readyz`. It is a write to
production with the owner URL, so it **ships with its daily schedule off**: until you set
`HOUSEKEEPING_ENABLED`, the scheduled run skips both jobs, and only a manual run from `main` does
anything. Do this after step 5 (the schema must exist) and after the backend is deployed (the Vercel
runbook), because the keep-warm job needs its URL.

Set the repository variable with the backend's **production origin**: the same value as the
frontend's `BACKEND_URL` (no path, `https` only). It is a variable, not a secret; the URL is public.
Unset, the keep-warm job prints a notice and does nothing.

```bash
gh variable set BACKEND_URL --body "https://<your-backend-production-domain>"
```

First, a dry run: the same statements, rolled back. The job summary lists one line per rule with its
row count. On a young database most are `0`; what you check is that the run is green and the
numbers are plausible.

```bash
gh workflow run housekeeping.yml --ref main -f dry_run=true
```

```bash
gh run watch
```

Then the real run (both jobs: `retention` and `keep-warm`; the second needs `BACKEND_URL` and a
backend that answers `/readyz` 200 within about three minutes):

```bash
gh workflow run housekeeping.yml --ref main -f dry_run=false
```

Only after one green real run, switch the daily schedule on. Setting the variable to anything but
`true`, or deleting it, switches it off again:

```bash
gh variable set HOUSEKEEPING_ENABLED --body true
```

```bash
gh variable delete HOUSEKEEPING_ENABLED
```

**The 60-day caveat.** GitHub disables scheduled workflows after 60 days without repository
activity. A disabled workflow stops stripping question text, which breaks the 30-day promise of the
privacy notice. If the repository goes quiet, re-enable the workflow in the Actions tab (or
run the first command of this step again with `dry_run=false`) and check the date of the last
scheduled run now and then. The README lists this limitation.

## When something fails

| Symptom | What it means | What to do |
|---|---|---|
| The guard step fails | `DATABASE_URL_DIRECT` is missing, is not a Postgres URL, is the `-pooler` host or is local | fix the secret (step 1) |
| Ingest exits 3 | the cache lacks texts and the key is missing or the daily quota (midnight Pacific) is spent. **Neon was not touched by the ingest** | run `warm-cache.yml` or wait for the reset and re-run; the vectors embedded so far were saved |
| Ingest exits 1 with `storing the index failed` | the store and verify transaction rolled back; one `failed` row (with the reason in `notes`, no documents or chunks) remains | read the error, fix the cause, re-run: it builds a new version. Failed rows are harmless and not active; `grounded index prune` (9.03) removes them later |
| Ingest says `another ingest already built this config` | two ingests raced | re-run: it reuses the built version |
| Migrate aborts | each file runs in its own transaction, so earlier files stay applied and the failing one is rolled back. A changed checksum means an applied file was edited | never edit an applied migration (DB.md §10): fix forward with a new numbered file in a PR |
| A bad version was activated | the runtime refreshes the active version every 5 minutes (`ACTIVE_INDEX_TTL_S`) | in `psql` as the owner, put the previous version back (DB.md §7.3), replacing `<old>` and `<bad>`: see below |
| A probe in step 7 that must fail succeeds | `app` has a grant it should not | revoke it (`REVOKE … ON <table> FROM app`), fix DB.md §5 and this file together if the doc was wrong |
| The `app` password or an owner URL leaked | | owner: reset the password in the Neon console and update the secret; `app`: `\password app` as the owner, update Vercel and redeploy |
| The `housekeeping` guard step fails | same as the ingest guard: `DATABASE_URL_DIRECT` is missing, pooled or local | fix the secret (step 1) |
| `retention` fails with a connection error | the direct URL is wrong or Neon is unreachable (the job stops after 10 minutes at the latest) | fix the secret or wait, then re-run; nothing was changed (one transaction) |
| `keep-warm` fails with `HTTP 000` after four tries | no answer: the backend URL is wrong, or the deployment is down | open the URL in a browser; fix `BACKEND_URL` or the deployment |
| `keep-warm` fails with `HTTP 503` | `/readyz` says the database or the active index is not ready | Neon runbook steps 4 and 5; Vercel runbook V10 |
| The scheduled run shows as skipped | `HOUSEKEEPING_ENABLED` is not `true` (the default) | step 10: set it after the first green manual run |
| You want to start over with the role | | `DROP OWNED BY app; DROP ROLE app;` as the owner, then step 6 |

Putting the previous index version back (only after the check that `<old>` is `ready`):

```sql
UPDATE index_versions SET is_active = false WHERE id = <bad>;
UPDATE index_versions SET is_active = true WHERE id = <old> AND status = 'ready';
```

## Checklist

Tick each box as you do it; the acceptance criteria of ticket 5.09 are the last two.

- [ ] `DATABASE_URL_DIRECT` secret set (direct URL, owner role); `GEMINI_API_KEY` present
- [ ] Environment `neon-production` with a required reviewer (recommended)
- [ ] Embeddings cache is warm (`gh cache list --key embeddings-`)
- [ ] `ingest.yml` run with `activate=false`: three migrations applied, a `ready` version, `docs`/`chunks` plausible
- [ ] `grounded golden validate … --against-index` passes against Neon
- [ ] `ingest.yml` run with `activate=true`: one active version
- [ ] `app` role created with SQL, grants applied, password set with `\password` and kept in the secret manager
- [ ] Catalog table of step 7 matches exactly
- [ ] The "must be able to" transaction ran as `app` over the pooled URL, upserts returned 2
- [ ] Every "must not" line failed with `permission denied`
- [ ] Vercel production `DATABASE_URL` is the pooled `app` URL; no Preview or Development value
- [ ] Optional: `EVAL_RECORD_RUNS` set, deliberately
- [ ] `BACKEND_URL` variable set; `housekeeping.yml` run with `dry_run=true`, then `dry_run=false`: both jobs green (ticket 5.11 acceptance)
- [ ] `HOUSEKEEPING_ENABLED=true` set after that run, deliberately; a calendar reminder to look at the scheduled runs within 60 days
