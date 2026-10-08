# R.16 — Fixes from the review of R.10–R.15

> **Before you start:** read [AGENTS.md](../../AGENTS.md), the rules in [tickets/README.md](../README.md) (how to work a ticket, ticket types, [standard checks](../README.md#standard-checks)) and the phase notes in [README.md](README.md). Work only on this ticket.

> **Owner:** G · **Type:** Build · **Branch:** `phase-3/review-follow-up-fixes` · **Blocked by:** R.15 ·
> **Builds on:** `generation/citations.py`, `ingest/embed.py`, `generation/providers/gemini.py`,
> `generation/pipeline.py`, `settings.py`

**What to build:** the findings of the Author's review of R.10–R.15 (2026-10-07). Decisions: (a) the Agent writes an
equality test for the invariant-4 guard in `test_confidence.py`; (b) the retry feedback does **not** get the rejected
value back; (c) `thinking_level` moves to the provider in Phase 4, not here.

**Scope notes**
- **R.14** `strip_urls`: a link around a grouped marker (`[c1, c2](…)`) keeps its brackets, so the rewrite removes and
  counts it like any group, instead of leaving `c1, c2` as prose; the URL regex is case-insensitive (`HTTPS://`,
  `WWW.`). The bypasses a regex cannot close (escaped backticks, an indented "fence", other autolink schemes) go to the
  frontend, Phase 5 (PRD §12).
- **R.10** `_embed_batch`: with no retry left the server's own error is raised, so the request path never names
  `EMBEDDING_MAX_RETRY_WAIT_S`.
- **R.13** `ProviderBadOutput.retryable`: a prompt the content filter blocked (no candidate) or an answer it stopped
  (`SAFETY`, `BLOCKLIST`, `PROHIBITED_CONTENT`, `SPII`) is not retried; it is still a 502. A wording fix in
  `ask_batch.py`.
- **R.12** `Settings.self_carry_worst_case()` holds the formula; `test_confidence.py` pins it to `score_claims` with
  equality for five weight configs; the test in `test_settings.py` no longer copies the formula.
  `SELF_CARRY_CEILING` no longer splits `_REPO_ROOT` from `_ENV_FILE`.
- PRD §12: fallback answers and the cache key (Phase 7), `thinking_level` on the provider (Phase 4), the frontend
  layer for URLs (Phase 5).

**Acceptance criteria**
- [x] Each fix starts with a test that failed before it.
- [x] The equality test fails when the `Settings` formula drifts (checked by hand with two mutations).
- [x] Docs updated as listed.

**Verify:** standard checks.

**Eval impact:** none measured. The URL changes touch only answers that contain URLs (0 in the fake and the real
golden runs of R.14, n=30 each); the retry change touches only blocked replies, which give no answer either way.

**Docs to update:** Tech §5.6, §9.5, §9.6, §12 (output rendering); PRD §12.
