# Runbook: the fallback demo (ticket 7.04b)

`grounded demo-fallback` shows Phase 7 working, **with fakes only**: a 429 on Gemini is answered by Groq, the next
request skips Gemini while its circuit breaker is open, and after the open time one trial brings Gemini back. No real
API is called, no key is needed, nothing sleeps, and the same code runs that serves `POST /v1/ask`.

An agent wrote this file and the command and ran them once on a local database on 2026-10-15 (the output below is
that run). The command is a **dev tool**: it refuses `APP_ENV=prod` and `APP_ENV=eval`.

**How the steps are marked.** Every command is for **PowerShell**, in the folder the step names (the repository root
or `backend/`), and the Git Bash form follows it where it differs. Nothing here needs a value from you: no `YOU TYPE`
step, no placeholder.

## What it does

Three requests go through the real ASGI app (`create_app`, in process) over your **local** database. The two
providers are `FakeLLMProvider` objects named `gemini` and `groq`, taking their model names from `GEMINI_MODEL` and
`GROQ_MODEL` (so `pricing.toml` prices them and the shadow cost is a real calculation). The breaker's clock is moved
by hand (`RouterClock`).

| # | Simulated time | The fake Gemini | What the service does |
|---|---|---|---|
| 1 | 0 s | answers its first call with a 429, `Retry-After` 30 s | opens Gemini's breaker for 30 s; Groq answers; `fallback_used` is true |
| 2 | +5 s | not called | Gemini is skipped (breaker open, 25 s left); Groq answers; `fallback_used` is true |
| 3 | +30 s more (35 s after the 429) | one trial call, which answers | the breaker closes; Gemini answers; `fallback_used` is false |

It prints one line per request, the `request_logs` row that request left, and the service's own log lines (the JSON the
service writes: `breaker opened`, `provider attempt failed`, `provider skipped`, and the `request completed` summary
with `provider` and `fallback_used`). It ends with `PASS` (exit 0) or `FAIL` (exit 1). It never prints a question.

## What it needs

- The local database up and migrated, with an **active index** (Phase 1: `grounded ingest --activate`). Check it with
  `uv run grounded index list`. The demo embeds the question with a fake embedder under your `EMBEDDING_MODEL`, so it
  makes no embedding call either; the full-text half of retrieval still finds chunks, and the dense half ranks by a
  stand-in vector. The demo shows routing, not answer quality.
- `GEMINI_MODEL` and `GROQ_MODEL` set (your `.env` has them). Keys are not read.

## Steps

**1. Start the database** (repository root; skip it if the container runs):

```powershell
docker compose -f infra/docker-compose.yml up -d db
```

**2. Run the demo** (`backend/`):

```powershell
cd backend
uv run grounded demo-fallback
```

Git Bash is the same two commands. To see that no key is involved, run it with every key blanked (the demo ignores
them either way):

```powershell
$env:GEMINI_API_KEY = ''; $env:GROQ_API_KEY = ''; $env:COHERE_API_KEY = ''
uv run grounded demo-fallback
Remove-Item Env:GEMINI_API_KEY, Env:GROQ_API_KEY, Env:COHERE_API_KEY
```

```bash
GEMINI_API_KEY= GROQ_API_KEY= COHERE_API_KEY= uv run grounded demo-fallback
```

## Expected output

The Git Bash command above on 2026-10-15 (timestamps, request ids, latencies and hashes differ on every run; everything
else is stable, and the exit code is 0):

```text
The fake gemini answers its first call with a 429 (Retry-After 30 s).
Time is a clock the demo moves by hand: nothing sleeps.
request 1: HTTP 200  provider=groq  model=openai/gpt-oss-120b  fallback_used=true  cache_hit=false  calls so far: gemini=1 groq=1
   request_logs row: outcome=answered  provider=groq  fallback_used=true  validation_retries=0  tokens=1500/300
   log {"ts": "2026-10-15T12:12:20.234+00:00", "level": "WARNING", "logger": "grounded.generation.router", "msg": "breaker opened", "request_id": "436b314a-a75b-4f03-952a-3e0afcc525af", "provider": "gemini", "reason": "rate_limit", "open_s": 30.0, "derived": "retry_after"}
   log {"ts": "2026-10-15T12:12:20.234+00:00", "level": "INFO", "logger": "grounded.generation.router", "msg": "provider attempt failed", "request_id": "436b314a-a75b-4f03-952a-3e0afcc525af", "provider": "gemini", "outcome": "rate_limited", "is_retry": false, "latency_ms": 0}
   log {"ts": "2026-10-15T12:12:20.235+00:00", "level": "INFO", "logger": "grounded.observability.request_log", "msg": "request completed", "request_id": "436b314a-a75b-4f03-952a-3e0afcc525af", "outcome": "answered", "http_status": 200, "error_code": null, "latency_total_ms": 79, "validation_retries": 0, "provider": "groq", "fallback_used": true, "invalid_citation_count": 0, "dropped_claim_count": 0, "removed_url_count": 0, "rerank_used": false, "rerank_error": null, "question_hash": "05dec5c038e995593ff900df06a6b5dce11de2c3f2b274877c372e2c47e9e716"}
-- simulated clock: +5 s --
request 2: HTTP 200  provider=groq  model=openai/gpt-oss-120b  fallback_used=true  cache_hit=false  calls so far: gemini=1 groq=2
   request_logs row: outcome=answered  provider=groq  fallback_used=true  validation_retries=0  tokens=1500/300
   log {"ts": "2026-10-15T12:12:20.250+00:00", "level": "INFO", "logger": "grounded.generation.router", "msg": "provider skipped", "request_id": "6f9506f0-85f4-4074-aaa3-c029c42dd1df", "provider": "gemini", "reason": "breaker_open", "remaining_s": 25.0}
   log {"ts": "2026-10-15T12:12:20.254+00:00", "level": "INFO", "logger": "grounded.observability.request_log", "msg": "request completed", "request_id": "6f9506f0-85f4-4074-aaa3-c029c42dd1df", "outcome": "answered", "http_status": 200, "error_code": null, "latency_total_ms": 11, "validation_retries": 0, "provider": "groq", "fallback_used": true, "invalid_citation_count": 0, "dropped_claim_count": 0, "removed_url_count": 0, "rerank_used": false, "rerank_error": null, "question_hash": "a7c5c266db0649034ec0abd975ce719c2d46d1692383a0c5999fe87ac8bbda1a"}
-- simulated clock: +30 s more, 35 s after the 429 --
request 3: HTTP 200  provider=gemini  model=gemini-3.5-flash-lite  fallback_used=false  cache_hit=false  calls so far: gemini=2 groq=2
   request_logs row: outcome=answered  provider=gemini  fallback_used=false  validation_retries=0  tokens=1000/300
   log {"ts": "2026-10-15T12:12:20.272+00:00", "level": "INFO", "logger": "grounded.observability.request_log", "msg": "request completed", "request_id": "8cefcd0a-9cea-4e25-892b-9780880cc783", "outcome": "answered", "http_status": 200, "error_code": null, "latency_total_ms": 10, "validation_retries": 0, "provider": "gemini", "fallback_used": false, "invalid_citation_count": 0, "dropped_claim_count": 0, "removed_url_count": 0, "rerank_used": false, "rerank_error": null, "question_hash": "7330439a09450ce8feaef48075e200ad2fb77128733db95c1c2c6393dfe3574c"}
PASS: the 429 on gemini was answered by groq; the next request skipped gemini without a call; after the open time one trial went to gemini, which answered again; request_logs.fallback_used is true, true, false.
```

How to read it:

- **Request 1.** `calls so far: gemini=1 groq=1`: Gemini was called once (the 429) and Groq answered. The log has the
  breaker opening for 30 s (`derived: retry_after`: the 429's own wait) and the failed attempt. The summary line has
  `provider: groq` and `fallback_used: true`: that is the fallback rate in the logs, and the `request_logs` row has the
  same.
- **Request 2.** `gemini=1` did not move: Gemini was **not called**. The log line `provider skipped` says why
  (`breaker_open`, 25 s left). Groq answered again (a fallback answer is not cached, so the same question would pay a
  Groq call again).
- **Request 3.** `gemini=2`: one trial, which answered, and `fallback_used=false`. Gemini's answer is the only one of
  the three that is cached.

## What it leaves behind

Three `request_logs` rows (their questions begin with `Demo <8 hex characters>:`, a different nonce on every run, so
a re-run is never answered from the cache) and one `answer_cache` row (request 3). The demo copy of the settings has
rerank off, no budget and no client IP hash, so it touches neither `daily_usage` nor the rate-limit counters. Look at
the rows (Git Bash or PowerShell, the container of the compose file):

```powershell
docker exec grounded-db-1 psql -U grounded -d grounded -c "select created_at, outcome, provider, fallback_used from request_logs where question like 'Demo %' order by created_at desc limit 6"
```

Delete them if you like (your database, your call):

```powershell
docker exec grounded-db-1 psql -U grounded -d grounded -c "delete from request_logs where question like 'Demo %'"
```

## If it fails

| Output | Cause | Fix |
|---|---|---|
| `demo-fallback is for dev and test: refused with APP_ENV=prod` (or `eval`) | the environment says so | use a dev shell; this command must not run against production |
| `set GEMINI_MODEL and GROQ_MODEL ...` | a model name is missing | set the pinned IDs from `.env.example`; no key is needed |
| `The demo's model names need a verified price` | a model ID is not in `backend/pricing.toml` | use the pinned IDs of PRD D46 and D48 |
| `Database error: ...` | the local database is not running | step 1 |
| `FAIL: a request did not answer (statuses [500, ...])` | there is no active index | `uv run grounded index list`; Phase 1's `ingest --activate` (that one uses the real embedding API, once) |
| `FAIL: ... fallback_used true, true, false` mismatch | the wiring is broken | this is the test failing for real: `uv run pytest tests/integration/test_fallback.py` |

## What it does not show

- **Nothing here is measured against live Groq or Gemini.** The 429 is scripted. The real Gemini 429 body and the
  Groq daily-429 `retry-after` have never been recorded (PRD §12, D57), and the Groq free plan (8K tokens a minute)
  is the real limit of the fallback: about 2 to 3 questions a minute and about 65 to 80 a day at the median size.
- **The breaker is per function instance and in memory** (D57a): a cold start forgets it, and instances do not share
  it. The demo is one process.
- The deployed demo has **no fallback until `GROQ_API_KEY` and `GROQ_MODEL` are set in Vercel** (the Author's step;
  without them the backend logs `fallback provider skipped, it cannot be built` and runs on Gemini alone).

The service tests behind the same behavior, with a scripted fake Gemini and Groq and the checks of the Phase 7 exit
criteria: `backend/tests/integration/test_fallback.py` (PRD §9, Phase 7).
