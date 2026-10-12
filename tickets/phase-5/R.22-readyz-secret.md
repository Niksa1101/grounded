# R.22 — `/readyz` requires the proxy secret

> **Before you start:** read [AGENTS.md](../../AGENTS.md), the rules in [tickets/README.md](../README.md) (how to work a ticket, ticket types, [standard checks](../README.md#standard-checks)) and the phase notes in [README.md](README.md#review-follow-ups). Work only on this ticket.

> **Owner:** G · **Type:** Build · **Branch:** `phase-5/readyz-secret` · **Blocked by:** R.21 ·
> **Builds on:** `api/security.py`, `api/routes_health.py`, `.github/scripts/keep-warm.sh`,
> `.github/workflows/housekeeping.yml`, `infra/cold_start.py`, `cli.py`

**What to build:** review item #2 and decision D54c. The public `/readyz` queries Neon on every call; a ping every
4 minutes keeps Neon Free awake at about 180 CU-h a month, against a 100 CU-h allowance. `/readyz` now requires the
same `X-Proxy-Secret` as `/v1`: without it the route answers 401 and queries nothing. `/healthz` stays public. This
is an HTTP contract change (Tech §13), approved by D54c.

**Read first:** Tech §12, §13, §17 (keep-warm), §18 (startup, smoke test); `docs/runbooks/vercel-deploy.md` steps 4
and 9; `docs/runbooks/neon-production.md` step 10; AGENTS.md §6.7.

**Scope notes**
- **Backend.** `api/security.py` exposes `_is_proxy` as a public `check_proxy_secret(request, settings) -> bool`
  (same behavior); `authenticate` calls it. `routes_health.py:readyz` calls it first: a missing or wrong secret raises
  `ProxyAuthError`, so the existing `_unauthorized` handler answers 401 in the shared shape, with no DB access and no
  row. With `ALLOW_DIRECT_API=true` (dev and tests; `app_client` sets it by default) a request without a secret
  still passes. The route's `responses` gets `401: {"model": ErrorResponse}`. `/healthz` is unchanged.
- **Frontend.** `npm run generate:api`; commit `frontend/lib/openapi.json` and `lib/api-types.ts`.
- **keep-warm.** `.github/scripts/keep-warm.sh` sends the header from stdin, so the value never appears in argv:
  `printf 'X-Proxy-Secret: %s\n' "$PROXY_SHARED_SECRET" | curl … -H @- …`. `BACKEND_URL` unset: notice, exit 0
  (unchanged). `BACKEND_URL` set but `PROXY_SHARED_SECRET` empty: `::error`, exit 1. A 401 gets its own message
  ("the secret differs from the backend's"). In `housekeeping.yml` only the keep-warm step's `env:` gets
  `PROXY_SHARED_SECRET: ${{ secrets.PROXY_SHARED_SECRET }}`.
- **measure-cold-start.** The secret comes from `get_settings().proxy_shared_secret` (no `os.environ`). It is sent
  **only** on `GET {backend}/readyz` (`_check_url` already enforces https except for localhost), never printed and
  never in an error message. `--backend-url` given without a secret: `MeasureError` naming `PROXY_SHARED_SECRET`.
  The module docstring no longer says "this module never has a secret".

**Acceptance criteria**
- [ ] `test_health.py`, `test_readyz.py`: 401 without a secret when direct API is off, with no pool use; 200 with the
      right secret; 401 with a wrong secret.
- [ ] `test_housekeeping_workflow.py`: the local HTTP server received the header; the missing-secret error; the 401
      message. `test_ci_workflows.py` allows `PROXY_SHARED_SECRET` for that one step.
- [ ] `test_cold_start.py`: the header goes to `/readyz` only (not `/healthz`, not the frontend); the missing-secret
      error.
- [ ] `npm run generate:api` leaves no diff after the commit.

**Verify:** standard checks plus the frontend checks; `curl -i localhost:8000/readyz` with `ALLOW_DIRECT_API=false`
gives 401 without the header and 200 with it; `bash .github/scripts/keep-warm.sh` against the local server exits 0
with the secret and 1 without it.

**Eval impact:** none.

**Docs to update:** Tech §12, §13, §17, §18; `docs/runbooks/vercel-deploy.md` (step 4 and the V7 loop send
`-H "X-Proxy-Secret: $PROXY_SHARED_SECRET"`, step 9 exports the production `PROXY_SHARED_SECRET`, a `/readyz 401`
failure row); `docs/runbooks/neon-production.md` step 10 (the Author creates the repository secret
`PROXY_SHARED_SECRET` before the keep-warm run); README operations; AGENTS §10.
