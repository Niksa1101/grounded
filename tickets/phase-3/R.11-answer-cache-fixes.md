# R.11 — Answer cache: replace a stale row, key on the generation parameters

> **Before you start:** read [AGENTS.md](../../AGENTS.md), the rules in [tickets/README.md](../README.md) (how to work a ticket, ticket types, [standard checks](../README.md#standard-checks)) and the phase notes in [README.md](README.md). Work only on this ticket.

> **Owner:** G · **Type:** Build · **Branch:** `phase-3/answer-cache-fixes` · **Blocked by:** R.10 ·
> **Builds on:** `infra/answer_cache.py`, `generation/pipeline.py`, `settings.py`

**What to build:** review items #2 and #3.

- **#2:** a cached row that no longer fits `AskResponse` is treated as a miss, but `_HIT` has already counted a hit,
  and `put` cannot replace the row (`ON CONFLICT … WHERE expires_at <= now()` only overwrites expired rows). After a
  schema change that leaves the key alone, the question pays a full LLM call on every request for up to 30 days.
- **#3:** the key misses the generation parameters. Changing `GEMINI_THINKING_LEVEL`, `LLM_TEMPERATURE` or
  `LLM_MAX_OUTPUT_TOKENS` serves up to 30 days of answers made with the old values. Decision of 2026-10-07 (D47): the
  key gets a `generation_config_hash`.

**Read first:** Tech §11, DB.md §4 (`answer_cache`), §7.2; PRD D47.

**Scope notes**
- #2: `AnswerCache.discard(key)` runs `DELETE FROM answer_cache WHERE cache_key = %s`. A database error is logged with
  the type and SQLSTATE only (same as `_log_failure`) and never raised. `AskPipeline._cached_response` calls it on a
  `ValidationError` before returning `None`, so the `put` at the end of the request stores a fresh row.
- #3: new module `generation/params.py` with a frozen `GenerationParams(provider, temperature, max_output_tokens,
  thinking_level)`, `GenerationParams.from_settings(settings, provider_name)` and a `config_hash` property (canonical
  JSON, sha256, the same pattern as `confidence_config_hash` and `RetrievalConfig.config_hash`). The pipeline builds it
  once in `__init__` and uses it in `_generate` too, so the key and the call read the same values.
- `CacheKey` gets `generation_hash`, part of the `digest` JSON array. No migration: like `confidence_hash`, it is not
  a column.
- Every existing row becomes a miss once (the key changes). There is no production data yet.

**Acceptance criteria**
- [x] `test_a_row_that_no_longer_fits_the_schema_is_a_miss` is extended: the stale row is gone after the miss, the
      third request is a hit, and the new row's `hit_count` is 1.
- [x] New `test_changed_generation_params_are_a_miss` (temperature, max output tokens, thinking level, each alone).
- [x] Unit tests: `GenerationParams.config_hash` is stable across instances and changes with every field;
      `discard` swallows and logs a database error.
- [x] `test_a_live_row_is_not_overwritten_by_a_concurrent_writer` still passes.

**Verify:** standard checks.

**Eval impact:** none (the cache is off in eval mode).

**Docs to update:** Tech §11 (key), DB.md §4 (`answer_cache.cache_key` comment), the `answer_cache.py` docstring.
