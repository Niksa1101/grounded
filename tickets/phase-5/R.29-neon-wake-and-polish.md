# R.29 — Neon wake rationale, optional firewall step and small items

> **Before you start:** read [AGENTS.md](../../AGENTS.md), the rules in [tickets/README.md](../README.md) (how to work a ticket, ticket types, [standard checks](../README.md#standard-checks)) and the phase notes in [README.md](README.md#second-review-follow-ups-2026-10-12). Work only on this ticket.

> **Owner:** G · **Type:** Build · **Branch:** `phase-5/neon-wake-and-polish` · **Blocked by:** R.28 ·
> **Builds on:** `api/routes_health.py`, `api/security.py`, `frontend/lib/ask-proxy.ts`, `cli.py` (`measure-cold-start`),
> `infra/cold_start.py`, `docs/runbooks/vercel-deploy.md`

**What to build:** review items #3 and #5 and decision D56b. `/api/ask` wakes Neon on every request, on purpose (the rate
limiter and the request log are database writes, and a 422 or 429 writes a row too). D54c removed only the *extra* path
(`/readyz`), and the code comments say it closed the risk of anyone keeping Neon awake. They overclaim. The risk stays;
the Author accepts it, and the docs now say so, with an optional Vercel Firewall step.

**Read first:** Tech §12, §13, §17; PRD D54c, D56, §12; `docs/runbooks/vercel-deploy.md` (steps 4 and 9).

**Scope notes**
- **Rationale in code.** In `api/routes_health.py` and `api/security.py`, "an anonymous caller must not be able to keep
  Neon awake" becomes: `/readyz` must not add a free path that wakes Neon, while `/api/ask` touches the database by
  design (PRD §12). Tech §12 and §13 say the same.
- **PRD §12, two open items.** (1) The residual risk that anyone sending `/api/ask` requests keeps Neon awake, with the
  advice to watch compute hours in the Neon console. (2) IPv6 `/64`: the deployment has no AAAA record on 2026-10-12;
  if that changes, `_canonical_ip` should group addresses by `/64`.
- **Runbook, optional Author step.** `docs/runbooks/vercel-deploy.md` gets "Firewall rate-limit rule for `/api/ask`
  (optional)". Before writing it, read Vercel's documentation (link and date in the step): does the Hobby plan offer
  rate limiting, and with which limits? If it does not, the step says so and the PRD item stays open. The Agent
  configures nothing on Vercel.
- **Proxy log.** `frontend/lib/ask-proxy.ts` (the fetch `catch`): when `request.signal.aborted`, log
  `console.info("ask proxy: client went away")` instead of `console.error("backend unreachable")`. The response is the
  same as before (nobody reads it).
- **`measure-cold-start`.** A new option `--retrieval-timeout-s`; its default is the value of `Settings`. The
  `retrieval_line` of the report says where the number came from (the option or the local `Settings`), because the
  deployed value may differ from the local one.

**Acceptance criteria**
- [x] `lib/ask-proxy.test.ts`: with an aborted request signal `console.error` is not called and `console.info` is.
- [x] `tests/unit/test_cold_start.py`, `test_cli.py`: the option overrides the default; the report line names its source.
- [x] The runbook step exists and cites the Vercel documentation with a date, or says the plan has no such rule.
- [x] PRD §12 lists the two open items; the code comments no longer claim that Neon cannot be kept awake.

**Verify:** standard checks plus the frontend checks (`npm run lint && npm run typecheck && npm test && npm run build`).

**Eval impact:** none.

**Docs to update:** Tech §12, §13; PRD §12; `docs/runbooks/vercel-deploy.md` (the new step, and step 9 for the new
option); AGENTS §10 (the `measure-cold-start` line).
