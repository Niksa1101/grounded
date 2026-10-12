# R.21 — `RETRIEVAL_TIMEOUT_S`: the retrieval stage timeout

> **Before you start:** read [AGENTS.md](../../AGENTS.md), the rules in [tickets/README.md](../README.md) (how to work a ticket, ticket types, [standard checks](../README.md#standard-checks)) and the phase notes in [README.md](README.md#review-follow-ups). Work only on this ticket.

> **Owner:** G · **Type:** Build · **Branch:** `phase-5/retrieval-timeout` · **Blocked by:** R.20 ·
> **Builds on:** `settings.py`, `generation/pipeline.py`, `api/deadline.py`, `api/errors.py`

**What to build:** decision D54a. The 2 s retrieval stage timeout of Tech §6 and D53 is not enforced: a slow query is
bounded only by the 25 s request deadline. A new `Settings` field bounds the search block of the pipeline and a
timeout answers 504 `deadline_exceeded` with outcome `deadline_exceeded` (R.20). Off in eval mode, like the deadline.

**Read first:** Tech §4, §6, §18 (timeout chain); PRD D53, D54; `api/deadline.py` (the own-expiry pattern).

**Scope notes**
- **Settings.** `retrieval_timeout_s: float = Field(default=2.0, gt=0)`, next to `query_embedding_timeout_s`. Add it
  to `.env.example` (the sync test checks it), the Tech §4 table and `tests/unit/test_settings.py`.
- **Pipeline.** `RetrievalTimeoutError(Exception)` lives next to `IndexMismatchError`: the pipeline must not import
  from `api/`. Bound the **second** `with timer.stage("retrieval"):` body (pool checkout, `hybrid_search`,
  `build_context`, `chunk_titles`), not the index lookup at the top. Use the own-expiry pattern inline:

  ```python
  scope = asyncio.timeout(s)
  try:
      async with scope: ...
  except TimeoutError as exc:
      if scope.expired(): raise RetrievalTimeoutError(s) from exc
      raise
  ```

  Do **not** extract a shared helper with `api/deadline.py`: R.24's test seam patches the `asyncio` name inside
  `grounded.api.deadline` only. With `settings.app_env == "eval"` there is no timeout.
- Do not touch `retrieval/hybrid.py` (Author-owned).
- **Errors.** New handler `_retrieval_timeout`: log `"retrieval stage timeout"` with `request_id`, the timeout and
  `trace.latency_ms()`, then `_fail(504, "deadline_exceeded", …)`. Register it in `install_error_handlers`.

**Acceptance criteria**
- [ ] With `hybrid_search` patched to await `asyncio.Event().wait()`, `retrieval_timeout_s=0.05` and
      `request_deadline_s=30`: 504 `deadline_exceeded`; the row has outcome `deadline_exceeded`,
      `latency_retrieval_ms` set and `latency_llm_ms` NULL; the LLM was never called; `daily_usage` has no row.
- [ ] Eval mode (`EVAL_SETTINGS`): a search that sleeps 0.1 s with `retrieval_timeout_s=0.02` gives 200.
- [ ] A request within the timeout is unaffected.
- [ ] `grounded serve` with `GENERATOR_PROVIDERS=fake` answers normally.

**Verify:** standard checks.

**Eval impact:** none (no timeout in eval mode). Say so in the PR.

**Docs to update:** Tech §4, §6 (the retrieval timeout is enforced), §18 (timeout chain table, the "not enforced yet"
text); PRD D53 ("not yet" points at D54); `.env.example`.
