# Runbook: MVP closeout (ticket 5.12)

Ticket 5.12 checks the four MVP exit criteria of Phase 5 ([PRD §9](../PRD.md#9-delivery-phases),
[tickets/phase-5/README.md](../../tickets/phase-5/README.md)). **The MVP is not declared complete.**
Two criteria need a live deployment, which only the Author can create (AGENTS.md §5, §11): until the
Author has run the [Neon runbook](neon-production.md) and the [Vercel runbook](vercel-deploy.md) and
the manual checks below, they are `PENDING-AUTHOR`. An agent wrote this file and never touched Neon,
Vercel, a secret, a workflow dispatch or a live URL.

Status as of **2026-10-12** (UTC). When the Author has done the work, fill the
[Results](#results) section, then do the [closing edits](#closing-edits).

## The four exit criteria

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | The live URL answers a question after the backend was idle (cold start measured and documented) | **PENDING-AUTHOR** | There is no live URL: no Vercel project exists, so nothing was deployed or measured. Ready: the deploy procedure ([Vercel runbook](vercel-deploy.md) steps 3 to 6), `grounded measure-cold-start` (step 9; tested by `backend/tests/unit/test_cold_start.py`) and the empty "Results" section the numbers go into. Tickets 5.08 to 5.10 are `blocked` on exactly this |
| 2 | Rate limit and budget verified manually | **PENDING-AUTHOR** | Needs the live site. What exists without it: the limiter and the budget are tested against a real Postgres (`backend/tests/integration/test_ratelimit.py`, `test_budget.py`, `test_ask_protection.py`: exact hit counts, a new window, independent IP hashes, atomicity under parallel requests, the eval-mode switch). That is not the criterion, which says manually. The step-by-step procedure for the Author is [below](#verify-the-rate-limit-and-the-budget-the-authors-manual-check) |
| 3 | README numbers are copied from committed eval results, never typed by hand | **PASS** | The two generated blocks of the README are compared with their commands by tests that fail on any drift: `backend/tests/unit/test_eval_report.py::test_the_readme_block_is_the_report_of_the_committed_baselines` (`grounded eval report`, reads only `eval/baselines/*.json`) and `backend/tests/unit/test_agreement_record.py` (`grounded eval agreement`). Every number in the README prose was traced by hand on 2026-10-12: [the audit](#readme-number-audit). Numbers that need the live site are `_TBD_`, not estimated |
| 4 | All CI workflows green on `main` | **PASS for the three that have run; two have never run** | [Table below](#workflows-on-main-2026-10-12) |

Why criterion 3 is a PASS and not "pending": the criterion is about the origin of numbers, and the
README contains none that came from anywhere else. What it cannot contain yet (cold start, live
latency) is marked `_TBD_` with the step that fills it, never estimated.

### Workflows on `main` (2026-10-12)

Read with `gh run list --workflow <file> --limit 3` and, for the eval verdicts, the gate line
(`### Generation gate: <verdict>`) of each run's log.

| Workflow | Last completed run on `main` | Notes |
|---|---|---|
| `ci.yml` | success (run 38220208054, commit `71d4e62` = #100, the newest on `main`, 2026-10-12) | Required checks `backend`, `frontend`, `retrieval-eval`. The run before it (38218904147, `b9ffef8`) was green too |
| `eval.yml` | success (run 38220207903, commit `71d4e62`, 2026-10-12); gate verdict `pass` | Of the 21 runs on `main` that finished green by 2026-10-12, 15 gated and passed (every one since 2026-10-09) and 6 were `inconclusive` (all on 2026-10-08: the Groq daily quota and Gemini 503 spells). No run on `main` failed; one was cancelled by a newer push. `inconclusive` is a green job and not a pass |
| `warm-cache.yml` | success (run 37018354661, `workflow_dispatch`, 2026-10-02) | Manual workflow; it ran once |
| `ingest.yml` | never ran | Manual, writes to Neon production: the Author runs it (Neon runbook steps 3 and 5) |
| `housekeeping.yml` | never ran | Manual dispatch from `main` works; the daily schedule ships off until the Author sets `HOUSEKEEPING_ENABLED` (Neon runbook step 10) |

The first green run of `ingest.yml` and `housekeeping.yml` is part of the Author's deployment work, so
"all workflows green" is observed for three of five and unobservable for the two that touch production.

## Verify the rate limit and the budget (the Author's manual check)

Every step here is the Author's. Nothing below is run by an agent. The procedure goes through the
frontend (`$FRONTEND_URL/api/ask`), the way a visitor does, so it exercises the proxy, the client-IP
header and the backend together. Use the production domains of the [Vercel runbook](vercel-deploy.md).

Two facts decide the design of the checks:

- **The limiter runs before the answer cache, and a cache hit costs no LLM call.** So the limit can be
  proven with a question that is already cached (the smoke-test question of Vercel runbook step 6)
  and no quota is spent. This is the order of Tech §1: validate, rate-limit, cache lookup, budget.
- **The budget check is `llm_calls >= DAILY_LLM_BUDGET`** on today's row of `daily_usage`, run after
  the cache lookup and before the query is embedded (Tech §12, DB §7.1). So the 503 does not need 300
  real calls: lower the limit to what today's counter already is (at least 1; `Settings` rejects 0), ask
  one question that was never asked, and the request is refused without an embedding or an LLM call.

Preparation (once):

```bash
FRONTEND_URL="https://<the frontend production domain>"
```

```bash
Q='{"question":"How do I declare a query parameter?"}'
```

That question is cached for 30 days after the smoke test, so every request below that uses it is a
cache hit. If it is not cached yet (the first `cache_hit` below is `false`), that one request costs one
or two LLM calls, which is fine. For the SQL you need the owner URL, exported for one session as in
the [Neon runbook](neon-production.md) step 4 (`psql "$DATABASE_URL_DIRECT"`). Never paste it anywhere.

Keep the three parts on **different clients or days**, because part 1 and 2 use up the allowance of
your IP (5 a minute, 20 a UTC day; D31) and everything you asked earlier the same UTC day (the smoke
tests, the cold-start asks) counts too. Do not run this close to 00:00 UTC.

### 1. The per-minute limit: 5 a minute, the 6th is a 429

Windows are fixed UTC minutes (Tech §12), so a burst that straddles a minute boundary is not limited.
Start right after a boundary:

```bash
while [ "$(date -u +%-S)" -ne 1 ]; do sleep 0.5; done
for i in 1 2 3 4 5 6; do
  curl -s -D - -o /dev/null -X POST "$FRONTEND_URL/api/ask" -H 'content-type: application/json' -d "$Q" \
    | tr -d '\r' | awk -v i="$i" 'NR==1 {printf "request %d: %s\n", i, $2} tolower($1)=="retry-after:" {print "  Retry-After: " $2}'
done
```

Expected: requests 1 to 5 print `200`, request 6 prints `429` and a `Retry-After` between 1 and 60. If
you do not see exactly that, the burst probably crossed a minute boundary: repeat once in the next
minute (every admitted request counts towards the day's 20). Then read the 429 body:

```bash
curl -s -X POST "$FRONTEND_URL/api/ask" -H 'content-type: application/json' -d "$Q"
```

Expected while the minute is still the same: `{"error":{"code":"rate_limited","message":"Too many requests. Try again in <N>s.","retry_after_s":<N>},"request_id":"…"}`.
In the browser the same six quick asks show the rate-limited state with a countdown (screenshots
`docs/screenshots/5.07-rate-limited-countdown.jpg`). Wait for the countdown, ask once more: `200`.

### 2. The per-day limit: 20 a UTC day, the 21st is a 429

First see where your IP's day counter stands (hashes only, no address). Run as the owner:

```sql
SELECT left(ip_hash, 8) AS bucket, hits, window_start
FROM rate_limit_counters
WHERE window_kind = 'day'
ORDER BY window_start DESC, hits DESC
LIMIT 5;
```

Your bucket is the one whose `hits` grew with your requests. Then send one request every 13 seconds
(never more than 5 in a UTC minute, so the minute limit cannot be the one that stops you) until the
first 429. It stops by itself after 25 tries:

```bash
n=0
while [ "$n" -lt 25 ]; do
  n=$((n+1))
  code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$FRONTEND_URL/api/ask" -H 'content-type: application/json' -d "$Q")
  echo "request $n: $code"
  [ "$code" = 429 ] && break
  sleep 13
done
```

Expected: `200` until your bucket's counter reaches 20 admitted requests, then `429`. Read it:

```bash
curl -s -i -X POST "$FRONTEND_URL/api/ask" -H 'content-type: application/json' -d "$Q"
```

Expected: status `429`, a `Retry-After` header of up to 86400 (the seconds to 00:00 UTC), and the body
`{"error":{"code":"rate_limited","message":"Daily question limit reached. Try again in about <N> h.","retry_after_s":<S>},…}`
with `<S>` equal to the header. The page shows "try again tomorrow" and no ticking counter (a wait over an
hour is not counted down, Tech §19). Rerun the SQL: your bucket's `hits` is 21 or more (the counter also
counts the rejected tries; 20 were admitted). A second client (a phone on mobile data) asking the same
question at this moment gets `200`: the buckets are per IP hash.

Then check what the backend stored, as the owner:

```sql
SELECT outcome, http_status, cache_hit, count(*) AS requests
FROM request_logs
WHERE created_at > now() - interval '1 hour'
GROUP BY 1, 2, 3
ORDER BY 4 DESC;
```

Expected: many `answered | 200 | true` rows and at least two `rate_limited | 429 | false` rows (one per
window), and no LLM spent by the 429s.

### 3. The daily budget: `503 budget_exhausted` without 300 real calls

Use a client whose day allowance is not spent (the phone on mobile data, or tomorrow). Every
step is yours: it touches the production environment variables.

1. Read today's counter (UTC day). Call the number `N` (no row for today means `0`):

   ```sql
   SELECT day, llm_calls FROM daily_usage ORDER BY day DESC LIMIT 3;
   ```

2. Choose the temporary limit `L = max(N, 1)`. With `N >= 1` the day is already at the limit, so the
   next question that is not cached is refused at the advisory check, before any embedding or LLM
   call. With `N = 0`, `L = 1` lets one question through (one or two calls) and refuses the next.
   Do **not** edit `daily_usage` by hand, and do not raise the limit above 300 (the Gemini quota of
   500 a day is shared with the CI evals, PRD §12).
3. Set it for Production only, in the **backend** project directory:

   ```bash
   cd backend && vercel env rm DAILY_LLM_BUDGET production --yes
   ```

   ```bash
   vercel env add DAILY_LLM_BUDGET production --value <L> --no-sensitive --yes
   ```

4. Environment changes apply only to a new deployment (Vercel runbook 3.4): Dashboard, Deployments, the
   latest production deployment, Redeploy; wait until it is Ready. **While the limit is lowered, a
   real visitor's uncached question is refused too** (cached answers are still served), so do this at
   a quiet time and finish step 8 promptly.
5. Ask a question that was never asked (any new text, 3 to 500 characters) from the phone, in the UI
   or with `curl`:

   ```bash
   curl -s -i -X POST "$FRONTEND_URL/api/ask" -H 'content-type: application/json' -d '{"question":"How do I add a tag to a path operation? budget check"}'
   ```

   Expected: `503`, `{"error":{"code":"budget_exhausted","message":"The demo's daily budget has been reached. Try again tomorrow (the day ends at 00:00 UTC).",…},"request_id":"…"}`, no `Retry-After`
   header. The page shows the friendly budget notice (`docs/screenshots/5.07-budget-exhausted.jpg`).
6. A cached question is still served on an exhausted day (a hit never consumes budget):

   ```bash
   curl -s -X POST "$FRONTEND_URL/api/ask" -H 'content-type: application/json' -d "$Q" | grep -o '"cache_hit":[a-z]*'
   ```

   Expected: `"cache_hit":true`.
7. Check that nothing was spent. As the owner, the counter is unchanged when `N >= 1` (and at most `L`
   when `N = 0`), and the refused request has no embedding time and no tokens:

   ```sql
   SELECT created_at, outcome, http_status, cache_hit, latency_embed_ms, latency_llm_ms, input_tokens
   FROM request_logs
   WHERE created_at > now() - interval '30 minutes'
   ORDER BY created_at DESC
   LIMIT 10;
   ```

   Expected: a `budget_exhausted | 503` row with `latency_embed_ms`, `latency_llm_ms` and `input_tokens`
   all NULL (the check comes before the embed stage), and the cached request as `answered | 200 | true`.
8. **Restore.** In the backend project directory:

   ```bash
   vercel env rm DAILY_LLM_BUDGET production --yes
   ```

   ```bash
   vercel env add DAILY_LLM_BUDGET production --value 300 --no-sensitive --yes
   ```

   Redeploy production, wait for Ready, then confirm: `curl -s "$BACKEND_URL/healthz"` answers
   `{"status":"ok"}`, the variable reads `300` in the dashboard (Settings, Environment Variables;
   it is stored as Config), and one new question from the phone is answered (`200`, `answered`).

If a result differs from the expectation, it is a finding: write it in "Results" as it is and open the
follow-up. Do not adjust the expectation.

## README number audit

Done on 2026-10-12 by reading every digit of `README.md` outside the two generated blocks. All of them
trace to committed output, a dated measurement, a decision record or a pinned configuration value.

| README text | Source |
|---|---|
| 30 questions, 25 answerable, 5 unanswerable; one question = 0.04 | `eval/golden/golden_set.v1.jsonl`; `n` in `eval/baselines/*.json` (25 retrieval, 30 generation) |
| Hybrid vs dense deltas (-0.02, -0.06, -0.14) | the retrieval table above them (0.74 vs 0.76, 0.657 vs 0.714, 0.78 vs 0.92) |
| "4 better, 9 worse", the q-numbers | the per-question results file of the 2.07 run on 2026-10-01 (`eval/results/`, gitignored, not committed): a dated measurement, labelled as such in the README |
| Gate demo, retrieval: 0.74 to 0.58, 0.66 to 0.44, 0.66 to 0.47, n = 25 | CI run 37453667731, closed PR #37 (public) |
| Gate demo, generation: run 38044085266, PR #86 | public CI run and PR comment |
| Judge agreement figures, 20 items, 53 claims, kappa, the three disagreements | the committed `eval/judge_agreement/v1.agreement.md` (a test keeps the README block identical to the command's output) |
| `main` runs: 21 green, 15 pass, 6 inconclusive, dates | `gh run list` and the gate line of each log, read 2026-10-12 (table above) |
| Rate limit 5 a minute, 20 a day, budget 300 of 500, region `cle1`, timeout chain | PRD D31 and D53, Tech §12 and §18; values of the runbooks |
| 30 days, 180 days | NFR-4, DB §9, `grounded housekeeping` |
| Weights 0.40, 0.25, 0.20, 0.15, 0.0, `w_self` at most 0.6, uncited cap 0.2 | `Settings` defaults (`CONFIDENCE_*`), Tech §9.8 |
| Pins: `0.141.1`, commit `95f8322`, promptfoo `0.123.1` | `FASTAPI_REF`, the pinned clone's `git log`, D49 |
| Failure-mode table: 25 of 30 and 21 of 30 cases lost, run 37817266433; 24 to 74 judge calls, six runs; about 145K of 200K tokens, 68 per-minute 429s; six prompt edits | PRD §12 (the Transient 5xx item, the free-tier headroom item, the 4.09b judge budget item, the gate-visibility item), each with its date and run id |
| 20 requests a day (`gemini-3.6-flash`) against 500 (`gemini-3.5-flash-lite`); 265 of 500 and 1.05K of 1K on 2026-10-11 | PRD D46 and §12 (budget and shared quota), read from the Author's AI Studio page |
| Cost and latency: the 25 s deadline, the 35 s proxy timeout, the 10 s between them; NFR-1 p50 4 s and p95 8 s; Neon scale to zero after 5 minutes | PRD D53, §7 and §12 |
| 2 days (rate-limit counters), 30 and 180 days (retention) | DB §4 and §9 |
| Operations table: dates 2026-10-02 (`warm-cache.yml`), 2026-10-12 (`gh variable list` empty, read that day) | `gh run list --workflow warm-cache.yml`, `gh variable list` |

The cost and latency numbers of the README are the generated eval block (shadow cost: token counts
times the paid list prices of `backend/pricing.toml`, dated; latency: the eval's warm cases) and nothing
else. No cold-start, live latency or live cost figure exists yet.

## Results

_Not done yet. The Author pastes here, each with its date and the method:_

- _Criterion 1: the reports of Vercel runbook step 9 (both orders) are in that runbook's "Results"; link them._
- _Criterion 2: the outcome of parts 1 to 3 above (what was seen: status codes, `Retry-After`, the SQL results, the date, the client used)._
- _Criterion 4: `gh run list` after the first green runs of `ingest.yml` and `housekeeping.yml`._

## Closing edits

When all four criteria pass, in one PR:

1. Fill the `_TBD_` markers of the README ([where each comes from](#where-the-readme-gets-the-remaining-tbd-cells)).
2. Set the Phase 5 row of the README roadmap to done, the PRD status line to "Phase 5 done, MVP complete"
   (with the date), and the status of tickets 5.08 to 5.12 in `tickets/README.md` to `done` with the
   criteria ticked in each ticket file.
3. Add a dated line to PRD §12 for each finding of the manual checks.

### Where the README gets the remaining TBD cells

| README place | Filled from |
|---|---|
| Live demo URL (header table, "Try it") | the frontend production domain, [Vercel runbook](vercel-deploy.md) step 6 |
| Cold-start and warm end-to-end rows (Cost and latency) | the reports under "Results" of the Vercel runbook, copied as printed, with the date and idle period |
| "Vercel's runtime logs show no client IP" (Privacy) | checklist item V1 of the Vercel runbook |
| "Rate limit and budget verified" (MVP status) | the Results above |
| Per-stage p50/p95 in production, cache hit rate, fallback rate | Phase 8 (dashboard), not Phase 5 |

## Deferred, and open decisions

Nothing below was done in 5.12; each item has its source.

1. **Retrieval stage timeout** (`RETRIEVAL_TIMEOUT_S` proposal) and the deadline row reusing
   `internal_error` + `deadline_exceeded` and the eager Gemini adapter: decided by the Author on
   2026-10-12 (PRD D54); the deadline outcome shipped in R.20 (migration `0004`), the timeout in R.21
   (`RETRIEVAL_TIMEOUT_S`).
2. **Production steps, all the Author's:** the Neon runbook (steps 1 to 10), the Vercel runbook (all),
   the first dispatch of `housekeeping.yml`, the variables `HOUSEKEEPING_ENABLED` and `EVAL_RECORD_RUNS`
   (both unset, so retention and the `eval_runs` write are off).
3. **The ask page**: no component-test setup (needs a dev dependency approval), claim confidence is
   beside the prose and not on it, light theme only (PRD §12, 5.07).
4. **Third-party notice.** The corpus is MIT (the pinned clone's `LICENSE` reads "Copyright (c) 2018
   Sebastián Ramírez"). The README attributes it; MIT also asks that the notice travel with substantial
   portions, and the demo shows snippets. A notice in the UI or a `NOTICE` file is not built; the
   Author decides.
5. **Eval quality items:** hybrid below dense (a tuned lexical side or `RRF_K` would be a separate PR
   that says it was tuned on the golden set), `judge_correctness_v2`, a second agreement sheet and a
   second rater, the meaning of the 0.8 in PRD §8 (PRD §12).
6. **CI items:** `warm-cache.yml` has its own copy of the setup steps; the `run-eval` label is the weak
   point of the required `eval` check; `ubuntu-latest` becomes Ubuntu 26 from 2026-10-19 (PRD §12).
7. **Telemetry gap:** a client that disconnects leaves no `request_logs` row (PRD §12).
8. **Phases 6 to 9:** rerank, fallback and breaker, the metrics dashboard, calibration, the final
   failure-mode analysis, a liveness check for the scheduled workflow (ticket 9.02).
