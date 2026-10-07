# R.15 — Small hardening and documentation drift

> **Before you start:** read [AGENTS.md](../../AGENTS.md), the rules in [tickets/README.md](../README.md) (how to work a ticket, ticket types, [standard checks](../README.md#standard-checks)) and the phase notes in [README.md](README.md). Work only on this ticket.

> **Owner:** G · **Type:** Build · **Branch:** `phase-3/review-polish` · **Blocked by:** R.14 ·
> **Builds on:** `api/routes_ask.py`, `generation/pipeline.py`, `infra/answer_cache.py`, `api/errors.py`

**What to build:** review items #10–#13 (🟢).

**Read first:** Tech §13, §14, §15.8.

**Scope notes**
- **#11** `assert` in the request path disappears under `python -O`. Replace with explicit checks:
  `routes_ask.py` (`trace.status is None` raises `RuntimeError`, which the existing `_internal_error` handler logs and
  turns into a row), `pipeline._cached_response` (`if self._cache is None: return None`), `AnswerCache.get` (a stored
  value that is not an object is logged, discarded and treated as a miss). The asserts in `observability/cost.py`
  stay: Pydantic already guarantees them.
- **#13** a test pins `ProviderRequestRejected` (bad key, 400) on `/v1/ask`: 500 `internal_error`, an
  `internal_error` row, and a message with no provider details.
- **#12** Tech §15.8 says what "schema-valid" in the batch summary proves: the pipeline validated the model output
  (`LLMAnswer`) and the response round-trips through `AskResponse`. It is not an independent check.
- **#10** Tech §14: the latency does not include the cache write or the log insert; the embedding tokens are an upper
  bound and are charged on an embedding-cache hit too. A client disconnect (`CancelledError`) writes no row; that
  goes with the deadline work of 5.02 (PRD §12).

**Acceptance criteria**
- [x] No `assert` left in `api/`, `generation/pipeline.py` or `infra/answer_cache.py`; each replacement has a test.
- [x] The `ProviderRequestRejected` test above.
- [x] Docs updated as listed.

**Verify:** standard checks.

**Eval impact:** none.

**Docs to update:** Tech §14, §15.8.
