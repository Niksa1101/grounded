# Runbook: Vercel projects, env vars, first deploy, cold start

Ticket 5.10. **The Author runs every step below.** An agent writes this file and the measurement
command `grounded measure-cold-start` but never creates a Vercel project, sets a variable, deploys or
measures anything live (AGENTS.md §5, §11). Commands are copy-pasteable, one per block, for Git Bash,
Linux or macOS. Never paste a secret or a connection string into a file, a commit, a PR or a chat.

What you end up with: two Vercel projects built from `main` (the backend FastAPI function in
`backend/`, the Next.js frontend in `frontend/`), both in region `cle1` next to Neon (D53), the
production variables set for **Production only**, previews that carry no production secret, a live
URL that answers a question, the open questions that earlier tickets deferred to "the first
deployment" answered, and the cold start measured with a method and a date.

Do the [Neon runbook](neon-production.md) first, through its step 8 (the schema, an active index, the
`app` role and the pooled `app` URL). Everything about Neon stays there; this file does not repeat it.

Facts below come from Vercel's documentation, read on **2026-10-12** (each page's own date in
brackets). What the documentation does not say is marked **unverified** and is a line of the
[checklist](#8-verify-on-the-first-deployment) with the way to find out.

## Order of steps

| # | Step | Where |
|---|---|---|
| 1 | Preflight | local |
| 2 | Generate the two secrets | local shell |
| 3 | Backend project: create, variables, deploy | Vercel |
| 4 | Backend smoke test | `curl` |
| 5 | Frontend project: create, variables, deploy | Vercel |
| 6 | End-to-end smoke test | browser, `curl` |
| 7 | Previews carry no production secret | Vercel, a throwaway branch |
| 8 | Verify on the first deployment | Vercel logs, `curl`, `psql` |
| 9 | Measure the cold start | local |
| 10 | Record the results and close the ticket | repo |

## The two projects at a glance

| | Backend | Frontend |
|---|---|---|
| Suggested name | `grounded-api` | `grounded` |
| Root Directory | `backend` | `frontend` |
| Framework preset | FastAPI (detected from the dependency) | Next.js (detected) |
| Entry | `[tool.vercel] entrypoint = "app:app"` is `backend/app.py` | the App Router, `app/api/ask/route.ts` |
| Region | `cle1`, from `backend/vercel.json` | `cle1`, from `frontend/vercel.json` |
| `maxDuration` | 30 s, `backend/vercel.json` | 40 s, in `route.ts` |
| Production branch | `main` | `main` |
| Production variables | 9 (below) | 2 (below) |
| Preview variables | 7 placeholders (below) | none |

The project name decides the production domain (`<name>.vercel.app`; Vercel adds a suffix if it is
taken). Whatever the domain turns out to be, it is the value of `BACKEND_URL` and of the URLs in the
commands below. Use the **production domain**, never the unique URL of a deployment: Vercel's
default **Standard Protection** puts every domain except the production domain behind Vercel
Authentication, so a protected URL answers the frontend's proxy with an HTML login page, not JSON
(`deployment-protection` [2026-09-15]).

## 1. Preflight

On `main`, up to date, with the PR of this ticket merged (it adds `frontend/vercel.json`):

```bash
git switch main && git pull --ff-only
```

Check the Neon runbook's checklist is ticked, in particular an active index version. The pooled `app`
URL is in your secret manager. Also have the Gemini key (the one CI uses is fine: the quota is shared,
PRD §12) and the Vercel CLI (the dashboard works too, but the steps below use the CLI for secrets so
that no value is ever on screen):

```bash
npm i -g vercel
```

```bash
vercel login
```

The repository is public, so a Hobby team can deploy it. Hobby is for non-commercial use (Tech §18).

## 2. Generate the two secrets

They live in your shell for this session only. `PROXY_SHARED_SECRET` goes into **both** projects (the
frontend sends it, the backend compares it); `IP_HASH_SECRET` only into the backend.

```bash
PROXY_SHARED_SECRET="$(openssl rand -hex 32)"
```

```bash
IP_HASH_SECRET="$(openssl rand -hex 32)"
```

A Secret variable is write-only after saving (`environment-variables/sensitive-environment-variables`
[2026-08-28]), so keep both in your password manager if you want to be able to re-enter them. If you
lose them, generate new ones and update the project(s) (new deployment needed). Rotating
`IP_HASH_SECRET` only resets everyone's rate-limit bucket; `PROXY_SHARED_SECRET` must change in both
projects before either is redeployed, or the frontend gets 401s in between.

## 3. Backend project

### 3.1 Create the project

Dashboard, Add New, Project, import the GitHub repository, then set:

- **Root Directory:** `backend`. Framework preset FastAPI should be detected; leave build and output
  settings at their defaults. (`git` [2026-09-18]: the import screen lets you pick the root directory.)
- **Production Branch:** `main` (the default when the repository has a `main`).
- Do **not** add variables on this screen; step 3.2 does it with the CLI. The first deployment will
  therefore start without configuration and fail at startup (a missing `GEMINI_API_KEY`). That is
  expected and harmless: `/v1/*` refuses everything without the proxy secret anyway.

Then link the directory to the project (it creates `backend/.vercel/`, which `.gitignore` excludes):

```bash
cd backend && vercel link
```

### 3.2 Production variables

Production only: the CLI takes the environment as an argument, so none of these reaches Preview or
Development. Variables that are not secrets are stored as Config (readable later), the rest as
Secret (`cli/env` [2026-08-20]: `--no-sensitive` is Config, `--sensitive` is Secret).

| Name | Value | Type |
|---|---|---|
| `APP_ENV` | `prod` | Config |
| `DATABASE_URL` | the pooled `app` URL (Neon runbook step 8) | Secret |
| `GEMINI_API_KEY` | the Gemini key | Secret |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | Config |
| `PROXY_SHARED_SECRET` | `$PROXY_SHARED_SECRET` of step 2 | Secret |
| `IP_HASH_SECRET` | `$IP_HASH_SECRET` of step 2 | Secret |
| `DAILY_LLM_BUDGET` | `300` | Config |
| `RATE_LIMIT_PER_MIN` | `5` | Config |
| `RATE_LIMIT_PER_DAY` | `20` | Config |

`Settings._check_prod` refuses to start without `DATABASE_URL`, `PROXY_SHARED_SECRET`,
`IP_HASH_SECRET` and `DAILY_LLM_BUDGET`, and the lifespan builds the Gemini adapter, which needs the
key and the pinned model (Tech §18; a test keeps that list and this table in step). Do **not** set
`DATABASE_URL_DIRECT` (the owner URL never reaches the runtime), `ALLOW_DIRECT_API` (it must be
false), `GROQ_API_KEY`, `COHERE_API_KEY`, `CACHE_DIR` or anything with a `NEXT_PUBLIC_` prefix.

The plain values:

```bash
vercel env add APP_ENV production --value prod --no-sensitive --yes
```

```bash
vercel env add GEMINI_MODEL production --value gemini-3.5-flash-lite --no-sensitive --yes
```

```bash
vercel env add DAILY_LLM_BUDGET production --value 300 --no-sensitive --yes
```

```bash
vercel env add RATE_LIMIT_PER_MIN production --value 5 --no-sensitive --yes
```

```bash
vercel env add RATE_LIMIT_PER_DAY production --value 20 --no-sensitive --yes
```

The secrets. The two generated ones come from the shell variable, so they are never typed:

```bash
printf %s "$PROXY_SHARED_SECRET" | vercel env add PROXY_SHARED_SECRET production --sensitive
```

```bash
printf %s "$IP_HASH_SECRET" | vercel env add IP_HASH_SECRET production --sensitive
```

If the CLI prompts for a value instead of reading standard input (the combination of `--sensitive`
with a pipe is not spelled out in the docs: unverified), paste the value at the prompt. The other two
are typed at the prompt, which does not echo them:

```bash
vercel env add DATABASE_URL production --sensitive
```

```bash
vercel env add GEMINI_API_KEY production --sensitive
```

If `DATABASE_URL` was already set in step 8 of the Neon runbook, `vercel env ls production` shows it
and you skip the first of these two.

### 3.3 Preview variables (placeholders, not secrets)

A preview must still start (Tech §18), and `APP_ENV=prod` refuses to start without the four required
values, so previews get throwaway ones. They are stored as **Config** on purpose: you can read them
back in the dashboard, which is how you verify in step 7 that no real value is there. `db.invalid`
is a reserved name that never resolves, so a preview can never reach a database.

```bash
vercel env add APP_ENV preview --value prod --no-sensitive --yes
```

```bash
vercel env add DATABASE_URL preview --value "postgresql://preview:preview@db.invalid:5432/preview" --no-sensitive --yes
```

```bash
vercel env add PROXY_SHARED_SECRET preview --value preview-placeholder-not-a-secret --no-sensitive --yes
```

```bash
vercel env add IP_HASH_SECRET preview --value preview-placeholder-not-a-secret --no-sensitive --yes
```

```bash
vercel env add DAILY_LLM_BUDGET preview --value 1 --no-sensitive --yes
```

```bash
vercel env add GEMINI_API_KEY preview --value preview-placeholder --no-sensitive --yes
```

```bash
vercel env add GEMINI_MODEL preview --value gemini-3.5-flash-lite --no-sensitive --yes
```

If the CLI asks for a Git branch, leave it empty: the variable then applies to every non-production
branch (`environment-variables` [2026-09-17]).

### 3.4 Deploy

Environment changes apply only to **new** deployments (`environment-variables` [2026-09-17]). Make
one from the dashboard: Deployments, Create Deployment, branch `main` (`git` [2026-09-18], "Creating
a deployment from a Git reference"). Watch the build log: that log is the evidence for several lines
of the checklist (V3, V4, V14). Then check the project settings once:

- Settings, Functions: the region reads `cle1` (it comes from `vercel.json`, which overrides the
  dashboard value: `fluid-compute` [2026-08-24], "Order of settings precedence") and Fluid compute
  is on (it is the default for projects created after 2025-04-23).
- Settings, Deployment Protection: Standard Protection with Vercel Authentication. The production
  domain stays public, deployment URLs and previews require a Vercel login.
- Settings, Environments, Production: Branch Tracking is `main`.

## 4. Backend smoke test

Put the production domain in a shell variable (not a secret):

```bash
BACKEND_URL="https://<the backend production domain>"
```

```bash
curl -s "$BACKEND_URL/healthz"
```

Expect `{"status":"ok"}`.

`/readyz` queries the database, so it needs the proxy secret, like `/v1` (PRD D54c; without it the
answer is `401` and nothing is queried). Read the production `PROXY_SHARED_SECRET` from your password
manager into the shell without echoing it (paste it at the empty prompt, then press Enter). Steps 8
and 9 use the same variable:

```bash
read -rs PROXY_SHARED_SECRET && export PROXY_SHARED_SECRET
```

```bash
curl -s -i -H "X-Proxy-Secret: $PROXY_SHARED_SECRET" "$BACKEND_URL/readyz"
```

Expect `200` and `{"status":"ok","database":"ok","active_index_version":"<ref>@<hash8>"}`, the same
`<ref>@<hash8>` as `uv run grounded index list` shows for the active version. A `503` is explained
in "When something fails".

```bash
curl -s -i -X POST "$BACKEND_URL/v1/ask" -H 'content-type: application/json' -d '{"question":"x"}'
```

Expect `401` with the `unauthorized` error: the proxy secret is enforced and nothing was written to
`request_logs` (Tech §12).

## 5. Frontend project

Create it like the backend, with **Root Directory** `frontend` (preset Next.js), Production Branch
`main`, no variables on the import screen. The first deployment will answer `proxy_misconfigured`
until step 5.1 and a new deployment; that is expected. Then:

```bash
cd ../frontend && vercel link
```

### 5.1 Production variables

Server-only, never `NEXT_PUBLIC_` (AGENTS.md §6.14). The names are the ones `frontend/.env.example`
and `app/api/ask/route.ts` use.

| Name | Value | Type |
|---|---|---|
| `BACKEND_URL` | the backend **production domain**, `https://…`, no trailing path | Config |
| `PROXY_SHARED_SECRET` | the same value as the backend's | Secret |

```bash
vercel env add BACKEND_URL production --value "$BACKEND_URL" --no-sensitive --yes
```

```bash
printf %s "$PROXY_SHARED_SECRET" | vercel env add PROXY_SHARED_SECRET production --sensitive
```

No Preview variables for the frontend: a preview of it answers `500 proxy_misconfigured` to every
question, which is the proof that it holds neither the backend URL nor the secret (step 7). It has
no `vercel.json` setting but `regions: ["cle1"]` (D53, `project-configuration/vercel-json`
[2026-08-14]: it overrides the dashboard region; Hobby takes a single region).

### 5.2 Deploy

Dashboard, Deployments, Create Deployment, branch `main`. Settings as in 3.4 (region `cle1`, Standard
Protection, Branch Tracking `main`). The ask route's `maxDuration` of 40 is in the code.

## 6. End-to-end smoke test

```bash
FRONTEND_URL="https://<the frontend production domain>"
```

Open `$FRONTEND_URL` in a browser, ask "How do I declare a query parameter?" and expect an answer with
citations, a confidence badge and a debug row with provider and model. The same through the API
(this is one real question, it uses quota):

```bash
curl -s -i -X POST "$FRONTEND_URL/api/ask" -H 'content-type: application/json' -d '{"question":"How do I declare a query parameter?"}'
```

Expect `200`, JSON with `status` `answered` and `meta.provider` `gemini`, and an `x-vercel-id` header
that starts with `cle1` or contains `cle1::` (V12). Open the **runtime logs** of both projects now:
Vercel keeps them for one hour (Tech §18), and V1, V2 and V9 need them.

## 7. Previews carry no production secret

The acceptance criterion needs a real preview of each project. Push a throwaway branch (an empty
commit is enough; both projects build it as a Preview deployment):

```bash
git switch -c chore/preview-check && git commit --allow-empty -m "chore: preview check" && git push -u origin HEAD
```

Then, in each project directory (`backend/`, `frontend/`):

```bash
vercel env ls preview
```

The backend lists exactly the seven names of 3.3. The frontend lists none. Then:

```bash
vercel env ls production
```

The backend lists the nine names of 3.2 and the frontend the two of 5.1, each for Production only
(the environments column). Then, in the dashboard (Settings, Environment Variables, filter
Preview): every row of the backend is type **Config** and its value is a placeholder you can read, in
particular `DATABASE_URL` is the `db.invalid` URL. No **Secret** row targets Preview or Development.
Optional hardening: the team policy "Separate Production Secret Values" (Team Settings, Security;
owner role) makes Vercel refuse a Secret whose value is shared between Production and any other
environment (`sensitive-environment-variables` [2026-08-28]).

Last, the behaviour, which is the stronger proof. Open the two preview URLs in a browser where you are
logged in to Vercel (a preview is protected):

- Backend preview `/healthz` answers `{"status":"ok"}` and `/readyz` answers `503` with
  `"database":"error"` and `"active_index_version":null`. It starts (V9) and it cannot reach the
  production database. Had it the production `DATABASE_URL`, `/readyz` would be `200` with an index.
- Frontend preview: ask a question on the page. Expect the error state (HTTP `500`,
  `proxy_misconfigured`).

Delete the branch afterwards:

```bash
git switch main && git push origin --delete chore/preview-check && git branch -D chore/preview-check
```

## 8. Verify on the first deployment

Earlier tickets deferred these to the first deployment (Tech §18, PRD §12, tickets 5.03, 5.06, 5.08,
5.10). Each has an expected result and a place where the answer is recorded. Answer them while the
deployment is fresh (runtime logs last one hour), then write the answers into the places named, in
the PR that closes this ticket. If a result differs from the expectation, it is a finding: record it
as it is and open the follow-up; do not adjust the expectation.

| # | Question | How | Expected | Record in |
|---|---|---|---|---|
| V1 | Does a Vercel runtime log show a client IP? (5.03) | Dashboard, each project, Logs, open the entry of one `/api/ask` and one `/v1/ask` request. Also read the application's own JSON log lines of that request | Method, path, status, host, request id, user agent, region; **no client address**. No question text and no address in the application lines either | Tech §12 ("a documentation reading, not an observation" becomes an observation with the date); ticket 5.10 criterion 3 |
| V2 | Is the first entry of `X-Forwarded-For` the visitor, one address per visitor? (5.06) | Ask one question from your home network and one from a phone on mobile data, then run the SQL below as the owner | Two different `ip_hash` prefixes, both `source = web`. No `proxy sent no usable client ip` warning in the backend logs | Tech §12 and §18; `frontend/lib/ask-proxy.ts` comment if the header turns out different |
| V3 | What does the build install, with which tool? (5.08) | Backend build log, "Installing dependencies" | uv or pip, runtime dependencies from `pyproject.toml`. Whether the project package or the `dev` group is installed is whatever the log shows; both work because `app.py` puts `src` first on `sys.path` | Tech §18, "What the docs do not say" |
| V4 | The real bundle size on Linux (5.08) | Build log or the deployment's function details (where Vercel shows it is **unverified**) | Well under 500 MB (about 70 MB measured locally with Windows wheels) | Tech §18, "The bundle" |
| V5 | Are `prompts/` and `pricing.toml` found in the bundle? (5.08) | The JSON of the step 6 answer | `meta.prompt_version` is set and `meta.shadow_cost_usd` is above 0 | Tech §18, "Path assumptions" |
| V6 | Is `CACHE_DIR` a problem? (5.08) | Production uses the in-memory query-embedding LRU and opens no file cache (`build_query_embedder`); read the backend log of a cold start | No filesystem error. `/tmp` writability is **unverified** in the docs and moot for this app | Tech §18, preview bullet |
| V7 | Does `prepare_threshold` work against Neon's pooler? (5.08) | The loop below, then read the backend logs | `30 200`; no `prepared statement … does not exist` or `already exists` in the logs, including during step 9's asks | Tech §18, "The pooled connection". A failure means `prepare_threshold=None` in a separate PR (DB.md §2) |
| V8 | Does the platform count initialization inside `maxDuration`? (5.10) | Not observable directly: the docs say the duration "includes time spent processing the request and sending the response" and nothing about initialization (`functions/limitations` [2026-08-24]). Judge it from step 9 | No `FUNCTION_INVOCATION_TIMEOUT` on a cold path, and a cold start under 5 s, so the question is moot (30 s minus a start under 5 s still exceeds the 25 s deadline) | Tech §18, timeout chain "Unverified" line |
| V9 | Does a preview start? Does `/readyz` report `database: error` there? (5.08 criterion 1) | Step 7 | `/healthz` 200, `/readyz` 503 `database: error` | ticket 5.08 first criterion; Tech §18 preview bullet |
| V10 | `/readyz` and the first request when Neon is asleep; the eager Gemini adapter (5.08, PRD §12) | Step 9, `--first health`: the cold `/healthz` and `/readyz` rows | Cold `/healthz` is 200 (the pool opens in the background, so the function starts with Neon down). Cold `/readyz` is 200 if the wake takes less than `DB_POOL_TIMEOUT_S` (5 s), otherwise 503 `database: error` and 200 on the next call. If the wake is slower than 5 s the cold ask can fail with `internal_error`: raise `DB_POOL_TIMEOUT_S` in Vercel's variables (a setting, not code), and say so | Tech §18 "Startup"; PRD §12 open item (3) |
| V11 | The deadline and the proxy timeout hold on the platform (5.08, D53) | Step 9 | No platform 504 where our own `deadline_exceeded` should have answered | Tech §18 timeout chain |
| V12 | Are both functions in `cle1`? (D53) | `x-vercel-id` of every response (the report prints the regions it saw); Settings, Functions | `cle1` for both. The latency gain over `iad1` stays unmeasured unless you choose to compare | Tech §18 region bullet; PRD §12 |
| V13 | Is Fluid compute on for both projects? | Settings, Functions | On (the default for projects created after 2025-04-23) | Tech §18 |
| V14 | Is `functions."app.py"` matched (so `maxDuration` 30 applies)? (5.08) | The build log: a `functions` pattern that matches nothing is reported there (**unverified** wording for framework presets) | A clean build | Tech §18 deploy settings |
| V15 | Production only from `main`; other branches are Preview | Settings, Environments, Production; the Deployments list during step 7 | Branch Tracking `main`; the throwaway branch built as Preview | Tech §18 |
| V16 | Optional: latency in `cle1` against `iad1` (PRD §12) | A second project or a temporary region change, then step 9 again | Only if you want the comparison; it is not required | PRD §12 |
| V17 | Cold retrieval against `RETRIEVAL_TIMEOUT_S` (R.21, PRD D54a) | Step 9, `--first ask`: the report's "Retrieval" line (the cold ask's `meta.latency_ms.retrieval`) | Under 75% of the timeout (1.5 s of the default 2 s); within 25% of it the report recommends raising `RETRIEVAL_TIMEOUT_S` in the backend's variables | Tech §18 timeout chain |

V2 as the owner, on the direct URL (the Neon runbook step 4 shows how to export it for one session).
It prints no address, only the first 8 characters of each hash:

```sql
SELECT left(ip_hash, 8) AS bucket, source, count(*) AS requests
FROM request_logs
WHERE created_at > now() - interval '1 hour'
GROUP BY 1, 2
ORDER BY 3 DESC;
```

V7, a free request loop (no LLM, no quota). The same statement is executed far more than five times
per pooled connection, which is when psycopg starts preparing it:

```bash
for i in $(seq 1 30); do curl -s -o /dev/null -w "%{http_code}\n" -H "X-Proxy-Secret: $PROXY_SHARED_SECRET" "$BACKEND_URL/readyz"; done | sort | uniq -c
```

## 9. Measure the cold start

The question: do the function start and the Neon wake, which both happen **before** the handler's
25 s deadline starts, fit in the 10 s between that deadline and the proxy's 35 s timeout (Tech §18,
D53)? `grounded measure-cold-start` times it. It calls the backend's public `/healthz`, the
backend's `/readyz` with `X-Proxy-Secret` (that one request only; D54c) and the frontend's public
`/api/ask`, what a visitor can reach. The command reads the secret from `PROXY_SHARED_SECRET`
through `Settings` and never prints it. Export the **production** value first (step 4 did, in the
same shell): a variable in the environment wins over a development `.env`, whose value would only
earn a 401. Without it, `--backend-url` is refused before anything is sent.

```bash
read -rs PROXY_SHARED_SECRET && export PROXY_SHARED_SECRET
```

Read before running:

- **The asks spend quota.** Each is one query embedding and one or two LLM calls (a share of the 300
  budget and of the Gemini quota that the CI evals use too), counts toward your IP's limits (5 a
  minute, 20 a day) and is stored in the production `request_logs` with its question for 30 days. The
  default is 4 asks per run (1 cold, 3 warm), the maximum 10, and the command asks you to confirm
  unless `--yes`. A `429` or `503` stops the asks; nothing is retried.
- **The idle period decides what "cold" means.** Neon scales to zero after 5 minutes without queries
  (PRD §12). How long a Vercel instance lives idle is not documented. Use at least 30 minutes, state
  it with `--idle-minutes`, and send nothing to either site meanwhile (not a browser tab, not a
  monitor). Pushes to `main` with `EVAL_RECORD_RUNS` set touch Neon too (Tech §17). Run this before
  ticket 5.11's keep-warm ping exists, or outside its hour.
- **Two orders, two idle cycles, because the first request warms what the next one measures.**

Run 1, `--first health`: the cold `/healthz` is the function start (it touches no database), the cold
`/readyz` is the database wake. The first ask follows probes, so it is labelled "after probes" and is
not the visitor's cold figure. `--wait` makes the command sleep the idle period itself:

```bash
cd "$(git rev-parse --show-toplevel)/backend" && uv run grounded measure-cold-start --frontend-url "$FRONTEND_URL" --backend-url "$BACKEND_URL" --idle-minutes 30 --wait --first health --asks 4 --warm-probes 10 | tee "$HOME/cold-start-health.md"
```

Run 2, `--first ask`: the first request after the idle period is the question itself, which is what a
visitor sees end to end (the frontend function's own start is in it too). With the built-in questions
this run starts halfway through the list, so its asks are not answer-cache hits of run 1 (which
starts at the top), as long as each run asks at most 5 (the list has 10). A cold ask that does hit
the cache is still judged: end to end minus the server total does not depend on it, and the table
marks the row "(cache hit)":

```bash
cd "$(git rev-parse --show-toplevel)/backend" && uv run grounded measure-cold-start --frontend-url "$FRONTEND_URL" --backend-url "$BACKEND_URL" --idle-minutes 30 --wait --first ask --asks 4 --warm-probes 3 | tee "$HOME/cold-start-ask.md"
```

Both print a Markdown report: a table with the method, the date (UTC), the idle period, `n`, min,
median and max per measurement; the margin verdict (enough, **NOT enough**, or "could not be
judged"); the warm median next to the NFR-1 target (p50 of 4 s; the `n` is too small for a p95); the
regions seen in `x-vercel-id`; and every raw sample. The two files in your home directory are outside the repository on purpose: paste their content into
step 10.

How the margin is judged, so you can defend it: in order `health`, the estimate is (cold `/healthz`
minus the warm `/healthz` median) plus (cold `/readyz` minus the warm `/readyz` median); the pool opens
its first connection in the background at start, so the wake may already run during the cold
`/healthz` and the sum is an estimate, not a decomposition. In order `ask`, it is the cold ask's end to
end time minus the server-reported `meta.latency_ms.total`, which also contains the frontend
function's start (outside the 35 s timeout) and so over-states the backend part. If the verdict is
**NOT enough**, nothing is hidden: the UI's retry state covers it (Tech §19) and the numbers go into
the README as they are. Each request uses a new connection, so every figure includes DNS, TCP and TLS.
In order `health` the estimate is also conservative: on a real ask the Neon wake happens at the first
database call, which is already inside the handler's deadline, so only part of it competes for the
margin.

The report's "Retrieval" line is V17: the first ask's `meta.latency_ms.retrieval` against
`RETRIEVAL_TIMEOUT_S` (2 s by default; the command reads the value from `Settings`, so export the
Vercel value if you changed it there). In order `ask` that is the cold retrieval; in order `health`
the database is already awake, so it is a lower bound. Within 25% of the timeout the report says to
raise `RETRIEVAL_TIMEOUT_S` in the backend's Vercel variables (a setting, not code). A cold ask that
fails with `504 deadline_exceeded` is not a margin verdict: the backend's deadline or retrieval timeout
fired, and the report prints the `request_id` to look up in `request_logs`.

To rehearse the command without touching production, run `grounded serve` and `npm run dev` locally
with `GENERATOR_PROVIDERS=fake` and point it at `http://127.0.0.1:8000` and `http://127.0.0.1:3000`
(plain `http` is accepted for localhost only).

## 10. Record the results and close the ticket

Paste both reports under "Results" below, with the answers to V1–V17. In the same PR: replace the
"unverified" sentences of Tech §18 and §12 with what was observed (date included), strike the
resolved items of PRD §12, tick the criteria in `tickets/phase-5/5.10-deploy.md` and the first
criterion of `tickets/phase-5/5.08-backend-vercel.md`, and set both index rows to `done`. The numbers
reach the README in ticket 5.12, copied from here, never typed.

## Results

_Not measured yet. The Author pastes the reports of step 9 here, each with its date, and the answers to
the checklist (V1–V17)._

## When something fails

| Symptom | What it means | What to do |
|---|---|---|
| Backend build: no Python entrypoint | The Root Directory is not `backend` | Settings, General, Root Directory |
| Backend `/healthz` is 500 | The lifespan failed: a missing `GEMINI_API_KEY` or `GEMINI_MODEL` (`ProviderConfigError`), or `Settings` refused the production values (the message names the missing variables) | Read the runtime log, fix the variable, create a new deployment (changes apply only to new ones) |
| `/readyz` 503, `database: error` | The pooled URL, user or password is wrong, `app` lacks a grant, or Neon was slower than `DB_POOL_TIMEOUT_S` | Neon runbook step 7; retry once; V10 |
| `/readyz` 503, `database: ok`, no index | No active index version | Neon runbook step 5 |
| `/readyz` 401 | The request had no `X-Proxy-Secret`, or a value that differs from the backend's `PROXY_SHARED_SECRET` (D54c) | Step 4: export the production value and send the header; for `housekeeping.yml`, the repository secret `PROXY_SHARED_SECRET` (Neon runbook step 10) |
| Frontend `500 proxy_misconfigured` | `BACKEND_URL` or `PROXY_SHARED_SECRET` is unset in the Production environment of the frontend, or the deployment predates them | Step 5.1, then a new deployment |
| Frontend `502 upstream_unavailable` | `BACKEND_URL` points at a protected deployment URL (an HTML login page), or at nothing | Use the production domain |
| Frontend `504 upstream_timeout` | The backend did not answer within 35 s | Step 9 tells whether it is the cold start |
| The cold ask of step 9 is `504 deadline_exceeded` | Our own backend 504: the 25 s deadline or the 2 s retrieval timeout fired (D53, D54a) | Look up the printed `request_id` in `request_logs` (Neon runbook step 4 shows how to connect as the owner): `latency_retrieval_ms` near 2000 and `latency_llm_ms` NULL is the retrieval timeout (raise `RETRIEVAL_TIMEOUT_S`, V17); otherwise the stage timings show where the 25 s went |
| Every question is `401 unauthorized` through the frontend | The two projects hold different `PROXY_SHARED_SECRET` values | Set the same value in both, redeploy both |
| Everything is `429` at once | Every visitor shares one bucket: the proxy's address was used | V2: the forwarded-for header is not what `clientIp` expects |
| A secret leaked | | Generate a new one, update the project(s), redeploy; for the Neon URL see the Neon runbook |
| A bad production deployment | | Deployments, an earlier good deployment, promote it (the dashboard's labels are **unverified**); then fix forward by a PR |

## Checklist

Tick each box as you do it; the acceptance criteria of ticket 5.10 are the last three.

- [ ] Neon runbook done through step 8
- [ ] `PROXY_SHARED_SECRET` and `IP_HASH_SECRET` generated, kept in the password manager
- [ ] Backend project: root `backend`, production variables (9) set for Production only
- [ ] Backend preview placeholders (7) set as Config
- [ ] Backend deployed from `main`: `/healthz` 200, `/readyz` with the secret 200 with the active index (without it 401), `/v1/ask` without a secret 401
- [ ] Frontend project: root `frontend`, `BACKEND_URL` (production domain) and `PROXY_SHARED_SECRET` set for Production only
- [ ] Frontend deployed from `main`; the page answers a question
- [ ] Both functions in `cle1`, Fluid compute on, Branch Tracking `main`
- [ ] Previews verified (step 7); throwaway branch deleted
- [ ] V1–V17 answered and recorded
- [ ] Cold-start reports (both orders) pasted under "Results"
- [ ] The live URL answers a question; cold start and warm latency are recorded with method and date
- [ ] Preview deployments have no production secrets (verified)
- [ ] One request's runtime log shows no client IP (V1)
