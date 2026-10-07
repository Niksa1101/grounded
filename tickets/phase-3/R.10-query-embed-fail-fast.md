# R.10 — Request-path query embedding fails fast

> **Before you start:** read [AGENTS.md](../../AGENTS.md), the rules in [tickets/README.md](../README.md) (how to work a ticket, ticket types, [standard checks](../README.md#standard-checks)) and the phase notes in [README.md](README.md). Work only on this ticket.

> **Owner:** G · **Type:** Build · **Branch:** `phase-3/query-embed-fail-fast` · **Blocked by:** R.09 ·
> **Builds on:** `ingest/embed.py`, `retrieval/query_embedding.py`, `settings.py`, `api/errors.py`

**What to build:** review item #1 (🔴). The query embedder of `/v1/ask` is the ingest `GeminiEmbedder`, which paces
calls against a 60 s RPM/TPM window, retries up to `EMBEDDING_MAX_RETRIES` (5) times with backoff or the server's
delay (up to 60 s), and gives each attempt `EMBEDDING_TIMEOUT_S` (30 s). One request can wait minutes. That breaks
Tech §10 ("No sleeping inside the request path") and the embed stage timeout of Tech §6 (3 s).

**Read first:** Tech §5.6, §6, §10, §11, §15.8; AGENTS.md §6.4, §6.12, §6.15.

**Scope notes**
- `GeminiEmbedder` gets a keyword `paced: bool = True`. With `paced=False`, `_embed_batch` skips `_RateWindow`
  (no wait before the call; the window is not recorded either).
- New classmethod `GeminiEmbedder.for_request_path(settings, count_tokens)`: `max_retries=0`, `paced=False`,
  `timeout_s=settings.query_embedding_timeout_s`. Everything else as `from_settings`. Ingest and the retrieval eval
  (`cli.py`, their own `LazyEmbedder`) keep `from_settings` and do not change.
- `retrieval/query_embedding.py`: the factory of `build_query_embedder` uses `for_request_path`.
- New setting `QUERY_EMBEDDING_TIMEOUT_S` (`query_embedding_timeout_s: float = Field(default=3.0, gt=0)`), the
  Tech §6 embed stage timeout.
- Behavior: a 429, 5xx or timeout becomes `ProviderRateLimited` / `ProviderUnavailable` / `ProviderTimeout` at once.
  `api/errors.py` already maps them to 503 `provider_unavailable` (with `Retry-After` for a 429). The golden batch
  (`grounded ask --golden`) still waits, in the CLI layer (`evals/ask_batch.py`), as Tech §15.8 intends.
- Out of scope: the retrieval stage timeout and the overall `REQUEST_DEADLINE_S` (the timeout chain of 5.02).

**Acceptance criteria**
- [ ] `test_embed.py`: with `max_retries=0`, a 429 with `retryDelay` raises `ProviderRateLimited` without any `sleep`
      call (an injected `sleep` that fails the test); a full RPM window with `paced=False` sends at once.
- [ ] `test_embed.py`: `for_request_path` builds an embedder with 0 retries, no pacing and the new timeout.
- [ ] `test_query_embedding.py`: `build_query_embedder` uses the request-path settings (prod and non-prod).
- [ ] `tests/integration/test_ask.py`: an embedder that raises `ProviderRateLimited(retry_after_s=50)` gives 503 with
      `Retry-After: 50` and a `provider_unavailable` row, without waiting.
- [ ] Ingest tests unchanged and green (pacing and retries still on there).

**Verify:** standard checks.

**Eval impact:** none (the retrieval eval does not use this embedder; the golden batch waits in the CLI as before).

**Docs to update:** Tech §4 (new setting), §6, §11 (query embedding row), `.env.example`.
