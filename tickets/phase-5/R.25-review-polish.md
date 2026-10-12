# R.25 — Small hardening and documentation drift

> **Before you start:** read [AGENTS.md](../../AGENTS.md), the rules in [tickets/README.md](../README.md) (how to work a ticket, ticket types, [standard checks](../README.md#standard-checks)) and the phase notes in [README.md](README.md#review-follow-ups). Work only on this ticket.

> **Owner:** G · **Type:** Build · **Branch:** `phase-5/review-polish` · **Blocked by:** R.24 ·
> **Builds on:** `api/security.py`, `infra/ratelimit.py`, `frontend/components/state-notices.tsx`,
> `frontend/lib/ask-proxy.ts`, `cli.py` (housekeeping)

**What to build:** the small items of the Phase 5 review and decision D54e.

**Read first:** Tech §12, §18, §19; PRD D53, D54; `docs/runbooks/mvp-closeout.md` (part 2).

**Scope notes**
- **Keyless bucket (D54e).** In `api/security.py:authenticate`, when `ip is None` and the HMAC key is set, hash the
  sentinel `"unknown"`: requests without a client IP share one strict bucket, never unlimited. Without
  `IP_HASH_SECRET` (dev and test only) nothing changes. Update the docstrings in `infra/ratelimit.py` ("No client key,
  no limit") and `security.py`.
- **UTC wording.** In `frontend/components/state-notices.tsx`, the "tomorrow" mode of `RateLimitedNotice` says
  "Today's question limit (per UTC day) is used up. It resets at 00:00 UTC.", matching `BudgetNotice`.
- **Proxy body size.** In `frontend/lib/ask-proxy.ts:forwardAsk`, a `content-length` header over `MAX_BODY_BYTES` is
  rejected with 422 before `request.text()`. The post-read check stays for bodies without a length; fix the comment.
- **Housekeeping error.** At `cli.py` (the housekeeping failure print), print the exception type and `sqlstate` only,
  like `infra/budget.py`.
- **Tech §18 and D53 rationale.** The Neon wake (first DB call) and the active-index load run **inside** the 25 s
  deadline, not before it; only the function start sits in the 10 s margin. The failure write-up after a deadline is
  not bounded by our code: the platform's `maxDuration` 30 s covers it, and the proxy maps the platform's non-JSON
  504 to `upstream_timeout`. This corrects the rationale only; the D53 values do not change.

**Acceptance criteria**
- [x] `test_proxy_auth.py`: `test_without_a_peer_or_a_client_ip_there_is_no_hash` becomes "…share the unknown
      bucket" (`== expected_hash("unknown")`); without the key the hash stays `None`.
- [x] Integration test: two keyless requests count in one bucket.
- [x] A `renderToStaticMarkup` test of the "tomorrow" notice (like `answer-view.test.tsx`).
- [x] `lib/ask-proxy.test.ts`: an oversized `content-length` gives 422 and the body is not read.
- [x] The failing-rule housekeeping test checks that only the type and `sqlstate` are printed.

**Verify:** standard checks plus the frontend checks (`npm run lint && npm run typecheck && npm test && npm run build`).

**Eval impact:** none.

**Docs to update:** Tech §12 (keyless bucket), §18, §19; PRD D53 rationale; `docs/runbooks/mvp-closeout.md` (part 2:
"The page shows …").
