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

Then link the directory to the project:

```bash
cd backend && vercel link
```

When it asks which project, pick the existing one (marked `linked by git`), not "Create a new
project". What Vercel CLI 63 does here (2026-10-12): it links the whole repository, in
`.vercel/repo.json` at the repository root (one entry per project, with its directory), not in
`backend/.vercel/`; it writes `<directory>/.env.local` with a `VERCEL_OIDC_TOKEN`; and it appends
`.vercel` and `.env*` to `.gitignore`. The existing rules already ignore both files. Undo the
`.gitignore` edit, because `.env*` would also match the committed `.env.example`:

```bash
git checkout -- .gitignore
```

The project name decides the domain only when it is free. On the first run both names were taken:
the backend project `grounded` got `grounded-snowy.vercel.app`, and the frontend project was created
as `grounded-11zh`. Use the domain that Settings, Domains shows, wherever this runbook says
`BACKEND_URL` or `FRONTEND_URL`.

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

On Windows, `vercel` is a `.cmd` shim, so `cmd.exe` parses its arguments: an `&` in a value (Neon's
URL ends in `&channel_binding=require`) splits the command, which is one more reason to type every
secret at the prompt. PowerShell has no `printf`: put the generated value in the clipboard
(`$env:PROXY_SHARED_SECRET | Set-Clipboard`), run the command without the pipe and paste at the
prompt; the same for `IP_HASH_SECRET`:

```powershell
vercel env add PROXY_SHARED_SECRET production --sensitive
```

The two secrets of step 2 in PowerShell (the `openssl rand -hex 32` equivalent), each to the session
and the clipboard; paste it into the password manager before generating the next:

```powershell
$b = New-Object byte[] 32; [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($b); $env:PROXY_SHARED_SECRET = ([BitConverter]::ToString($b) -replace '-','').ToLower(); $env:PROXY_SHARED_SECRET | Set-Clipboard
```

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
a deployment from a Git reference"). Where the dashboard shows no Create Deployment (2026-10-12),
open the latest production deployment and use the menu next to Visit, Redeploy: it builds the same
commit with the current variables. Untick "Use existing Build Cache" for a complete build log. Watch the build log: that log is the evidence for several lines
of the checklist (V3, V4, V14). Then check the project settings once:

- Settings, Functions: the region reads `cle1` (it comes from `vercel.json`, which overrides the
  dashboard value: `fluid-compute` [2026-08-24], "Order of settings precedence") and Fluid compute
  is on (it is the default for projects created after 2025-04-23). The project settings may still
  show the default `iad1` (`vercel project inspect` did on 2026-10-12): what counts is the
  deployment, whose function is listed as `λ fastapi (… MB) [cle1]` (`vercel inspect <url>`), and
  the `x-vercel-id` header of a response, `…::cle1::…`.
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
  Since R.22 `/readyz` needs `X-Proxy-Secret`, which an address bar cannot send: open the preview's
  `/docs`, then run this in the browser console (the value is the preview placeholder of 3.3, not a
  secret). Not from the `/healthz` page: Firefox shows JSON in a viewer whose content security policy
  blocks `fetch`.

  ```js
  fetch('/readyz',{headers:{'X-Proxy-Secret':'preview-placeholder-not-a-secret'}}).then(r=>r.text().then(t=>console.log(r.status,t)))
  ```

- Frontend preview: ask a question on the page. Expect the error state (HTTP `500`,
  `proxy_misconfigured`).

A preview can end in Error before its build starts, with the reason "Git information retrieval
failed for this deployment" (`vercel inspect <url>`). On 2026-10-12 that was a Vercel incident
(vercel-status.com, "Increased build and deployment failures"); another empty commit on the branch
built both previews.

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

The question: does the function start, which happens **before** the handler's 25 s deadline starts,
fit in the 10 s between that deadline and the proxy's 35 s timeout (Tech §18, D53)? The Neon wake
runs at the first database call, inside the deadline, so counting it here (order `health`) only makes
the estimate stricter. `grounded measure-cold-start` times it. It calls the backend's public `/healthz`, the
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

First deployment, 2026-10-12, run by the Author with the agent reading logs and reports (`vercel logs`,
`vercel inspect`, `gh run view`; nothing written to production by the agent). The two projects:

| | Backend | Frontend |
|---|---|---|
| Project | `grounded` | `grounded-11zh` |
| Production domain | `https://grounded-snowy.vercel.app` | `https://grounded-11zh.vercel.app` |
| Function | `λ fastapi (44.89MB) [cle1]` | Next.js, `cle1` |

### Checklist

| # | Answer (2026-10-12) |
|---|---|
| V1 | **No client address.** The runtime logs of both projects (`vercel logs --json`, 14 backend and 9 frontend entries) carry no IP field and no IPv4-like string; the application lines carry `question_hash`, never the question text |
| V2 | **One address per visitor.** A question from the home network and one from a phone on mobile data: two different `ip_hash` prefixes (`34b5579f`, `891e9f37`), both `source = web`. No `proxy sent no usable client ip` warning in the backend logs |
| V3 | **uv 0.10.11**, Python 3.12 from `backend/.python-version`, "Installing required dependencies from uv.lock" (build log of `grounded-kvub7msew`) |
| V4 | **44.89 MB** (the function in `vercel inspect`), against the 500 MB limit |
| V5 | **Found.** The smoke-test answer had `meta.prompt_version` `answer_v1@08cc49e5` and `meta.shadow_cost_usd` 0.0046 |
| V6 | **No problem.** No filesystem error in the production logs |
| V7 | **Works.** 30 consecutive `/readyz` with the secret: `200` 30 times; no `prepared statement` error in the logs, including the measurement asks |
| V8 | **Moot.** The cold function start was about 2.6 s (step 9), far under 5 s; no `FUNCTION_INVOCATION_TIMEOUT` |
| V9 | **Starts, and cannot reach the database.** Backend preview `/healthz` `{"status":"ok"}`; `/readyz` with the placeholder secret `503 {"status":"unavailable","database":"error","active_index_version":null}` after about 5.2 s (`DB_POOL_TIMEOUT_S`) |
| V10 | **Cold `/readyz` 200** at the first call after 30 minutes idle; the Neon wake was about 0.7 s, under `DB_POOL_TIMEOUT_S` |
| V11 | **Held.** Every measured request answered `200`; no platform 504 |
| V12 | **`cle1` for both**: every `x-vercel-id` was `fra1::cle1::…` (edge in Frankfurt, function in Cleveland). The project settings show the default `iad1`; `vercel.json` overrides it per deployment |
| V13 | **Not checked.** The Functions page with the Fluid compute switch was not found in the dashboard; it is the default for projects created after 2025-04-23 (unverified for these two) |
| V14 | **Clean build**, no warning about a `functions` pattern |
| V15 | **Yes.** Branch Tracking `main`; the throwaway branch `chore/preview-check` built as Preview in both projects |
| V16 | Not done (optional) |
| V17 | **22%.** The cold ask's retrieval took 0.44 s against `RETRIEVAL_TIMEOUT_S` = 2 s (after probes in the other order: 0.38 s, 19%). No change to the timeout |

Step 7 in numbers: the backend has 7 Preview variables, all Config placeholders, and 9 Production
variables (4 Secret); the frontend has none in Preview and 2 in Production. The frontend preview answers
`500 proxy_misconfigured`. One frontend preview first failed with "Git information retrieval failed"
during a Vercel incident and built on a second empty commit. `PROXY_SHARED_SECRET` and `IP_HASH_SECRET`
were added with `--value` from a PowerShell session variable (so they were in a process's argument list
on the Author's machine for a moment); step 3.2 now says to use the prompt on Windows too.

The smoke test (step 6), one real question through the frontend: `200`, `answered`, 2 claims, 3
citations, `gemini-3.5-flash-lite`, index `0.141.1@4949e8a3`, 6.07 s total (embed 0.25, retrieval 0.46,
LLM 4.61), with `validation_retries` 1 (the first model output failed validation and the one retry
passed, AGENTS.md §6.4).

### Step 9: the two reports, as printed

Run 1, `--first health` (the tool waited the 30 minutes itself):

### Cold start and warm latency

- Date (UTC): 2026-10-12 13:51; idle before the first request: 30 min (waited by the tool)
- Order: `--first health`; frontend `https://grounded-11zh.vercel.app`, backend `https://grounded-snowy.vercel.app`

| Measurement | Method | Date (UTC) | Idle | n | min | median | max |
|---|---|---|---|---|---|---|---|
| function start, cold | `GET {backend}/healthz` | 2026-10-12 13:51 | 30 min | 1 | 2.91 s | 2.91 s | 2.91 s |
| database wake, cold | `GET {backend}/readyz` | 2026-10-12 13:51 | 30 min | 1 | 0.96 s | 0.96 s | 0.96 s |
| ask end to end, after probes | `POST {frontend}/api/ask` | 2026-10-12 13:51 | 30 min | 1 | 4.05 s | 4.05 s | 4.05 s |
| `/healthz`, warm | `GET {backend}/healthz` | 2026-10-12 13:51 | 30 min | 10 | 0.26 s | 0.28 s | 0.45 s |
| `/readyz`, warm | `GET {backend}/readyz` | 2026-10-12 13:51 | 30 min | 10 | 0.27 s | 0.29 s | 0.31 s |
| ask end to end, warm | `POST {frontend}/api/ask` | 2026-10-12 13:51 | 30 min | 3 | 2.28 s | 2.47 s | 5.09 s |

#### Margin

The 10 s margin is enough: estimated cold overhead 3.30 s; function start about 2.62 s (cold `/healthz` minus the warm median), database wake about 0.67 s (cold `/readyz` minus the warm median).

#### Retrieval

Retrieval of the first ask (after probes): 0.38 s against `RETRIEVAL_TIMEOUT_S` = 2 s (19% of it).

Warm ask median 2.47 s (n = 3) against the NFR-1 target of p50 <= 4 s. This n says nothing about p95.

Regions in `x-vercel-id` (expected `cle1`, D53): `fra1::cle1`

#### Raw samples

| # | Request | Phase | Result | Time | Server total | Cache hit |
|---|---|---|---|---|---|---|
| 1 | healthz | cold | HTTP 200 | 2.91 s |  |  |
| 2 | readyz | cold | HTTP 200 | 0.96 s |  |  |
| 3 | healthz | warm | HTTP 200 | 0.28 s |  |  |
| 4 | readyz | warm | HTTP 200 | 0.29 s |  |  |
| 5 | healthz | warm | HTTP 200 | 0.30 s |  |  |
| 6 | readyz | warm | HTTP 200 | 0.29 s |  |  |
| 7 | healthz | warm | HTTP 200 | 0.28 s |  |  |
| 8 | readyz | warm | HTTP 200 | 0.31 s |  |  |
| 9 | healthz | warm | HTTP 200 | 0.26 s |  |  |
| 10 | readyz | warm | HTTP 200 | 0.29 s |  |  |
| 11 | healthz | warm | HTTP 200 | 0.27 s |  |  |
| 12 | readyz | warm | HTTP 200 | 0.27 s |  |  |
| 13 | healthz | warm | HTTP 200 | 0.28 s |  |  |
| 14 | readyz | warm | HTTP 200 | 0.29 s |  |  |
| 15 | healthz | warm | HTTP 200 | 0.35 s |  |  |
| 16 | readyz | warm | HTTP 200 | 0.29 s |  |  |
| 17 | healthz | warm | HTTP 200 | 0.45 s |  |  |
| 18 | readyz | warm | HTTP 200 | 0.28 s |  |  |
| 19 | healthz | warm | HTTP 200 | 0.28 s |  |  |
| 20 | readyz | warm | HTTP 200 | 0.28 s |  |  |
| 21 | healthz | warm | HTTP 200 | 0.28 s |  |  |
| 22 | readyz | warm | HTTP 200 | 0.28 s |  |  |
| 23 | ask | after probes | HTTP 200 | 4.05 s | 2.89 s | no |
| 24 | ask | warm | HTTP 200 | 2.47 s | 2.12 s | no |
| 25 | ask | warm | HTTP 200 | 5.09 s | 4.69 s | no |
| 26 | ask | warm | HTTP 200 | 2.28 s | 1.90 s | no |

Run 2, `--first ask`, right after run 1 (another 30 minutes waited). Its asks were different questions
from run 1's (the offset of R.23), and the cold ask was not a cache hit:

### Cold start and warm latency

- Date (UTC): 2026-10-12 14:23; idle before the first request: 30 min (waited by the tool)
- Order: `--first ask`; frontend `https://grounded-11zh.vercel.app`, backend `https://grounded-snowy.vercel.app`

| Measurement | Method | Date (UTC) | Idle | n | min | median | max |
|---|---|---|---|---|---|---|---|
| ask end to end, cold | `POST {frontend}/api/ask` | 2026-10-12 14:23 | 30 min | 1 | 7.04 s | 7.04 s | 7.04 s |
| `/healthz`, warm | `GET {backend}/healthz` | 2026-10-12 14:23 | 30 min | 3 | 0.27 s | 0.28 s | 0.35 s |
| `/readyz`, warm | `GET {backend}/readyz` | 2026-10-12 14:23 | 30 min | 3 | 0.27 s | 0.28 s | 0.44 s |
| ask end to end, warm | `POST {frontend}/api/ask` | 2026-10-12 14:23 | 30 min | 3 | 2.56 s | 2.63 s | 2.76 s |

#### Margin

The 10 s margin is enough: estimated cold overhead 3.75 s; cold ask 7.04 s end to end minus the server-reported total 3.29 s.

#### Retrieval

Retrieval of the cold ask: 0.44 s against `RETRIEVAL_TIMEOUT_S` = 2 s (22% of it).

Warm ask median 2.63 s (n = 3) against the NFR-1 target of p50 <= 4 s. This n says nothing about p95.

Regions in `x-vercel-id` (expected `cle1`, D53): `fra1::cle1`

#### Raw samples

| # | Request | Phase | Result | Time | Server total | Cache hit |
|---|---|---|---|---|---|---|
| 1 | ask | cold | HTTP 200 | 7.04 s | 3.29 s | no |
| 2 | healthz | warm | HTTP 200 | 0.35 s |  |  |
| 3 | readyz | warm | HTTP 200 | 0.27 s |  |  |
| 4 | healthz | warm | HTTP 200 | 0.28 s |  |  |
| 5 | readyz | warm | HTTP 200 | 0.28 s |  |  |
| 6 | healthz | warm | HTTP 200 | 0.27 s |  |  |
| 7 | readyz | warm | HTTP 200 | 0.44 s |  |  |
| 8 | ask | warm | HTTP 200 | 2.56 s | 2.15 s | no |
| 9 | ask | warm | HTTP 200 | 2.76 s | 2.40 s | no |
| 10 | ask | warm | HTTP 200 | 2.63 s | 2.28 s | no |

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
| A deployment stays in Initializing for minutes | Usually on Vercel's side: check vercel-status.com. The previous deployment keeps serving with its own variables | Wait; if it does not start, cancel it and redeploy later |
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
