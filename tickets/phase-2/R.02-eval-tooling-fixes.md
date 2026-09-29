# R.02 — Eval tooling fixes: dirty-tree check and lazy embedder

> **Before you start:** read [AGENTS.md](../../AGENTS.md), the rules in [tickets/README.md](../README.md) (how to work a ticket, ticket types, [standard checks](../README.md#standard-checks)) and the phase notes in [README.md](README.md). Work only on this ticket.

> **Owner:** G · **Type:** Build · **Branch:** `phase-2/eval-tooling-fixes` · **Blocked by:** R.01 ·
> **Builds on:** `evals/retrieval_runner.py` (`repo_state`), `ingest/embed.py`, `cli.py`

**What to build:** two review fixes to the tools that produce baselines (items #11 and #12).

**Read first:** Tech §5.6 (embedder, cache), Tech §15.2.

**Scope notes**
- **#11:** `repo_state()` runs `git status --porcelain --untracked-files=no`. Untracked files (`tickets/`, scratch
  scripts) don't make a run "dirty", a modified tracked file does.
- **#12:** `LazyEmbedder(model, dim, factory)` in `ingest/embed.py` implements the `Embedder` protocol and builds the
  real `GeminiEmbedder` on the first `embed()` call. If the factory can't (no key), it raises the typed
  `EmbedderUnavailableError`. `api_calls` is `0` until the inner embedder exists. `cli.py` (`ingest`,
  `eval retrieval`) uses it and turns the error into a clear message ("GEMINI_API_KEY is needed: N texts are not
  cached"). The pipeline is unchanged: `ingest()` already checks reuse and `model`/`dim` before any embed call, so an
  ingest of an already built version and an eval with a warm cache need no key.

**Acceptance criteria**
- [ ] A temporary git repo test: an untracked file leaves the tree clean, a modified tracked file makes it dirty.
- [ ] `LazyEmbedder` builds nothing until called, and passes the error through.
- [ ] Integration: `ingest` of an existing version without a key says "already built"; a repeated eval with every
      vector cached passes without a key; a cold cache without a key ends in a clean error.

**Verify:** standard checks.

**Eval impact:** none.

**Docs to update:** Tech §5.6 (lazy embedder), Tech §15.2 if it mentions the dirty flag.
