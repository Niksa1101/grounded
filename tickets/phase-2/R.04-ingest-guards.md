# R.04 — Ingest guards and documentation drift

> **Before you start:** read [AGENTS.md](../../AGENTS.md), the rules in [tickets/README.md](../README.md) (how to work a ticket, ticket types, [standard checks](../README.md#standard-checks)) and the phase notes in [README.md](README.md). Work only on this ticket.

> **Owner:** G · **Type:** Build · **Branch:** `phase-2/ingest-guards` · **Blocked by:** R.03 ·
> **Builds on:** `ingest/includes.py`, `ingest/markdown.py`, `ingest/embed.py`, `settings.py`

**What to build:** four review items (#6, #7, #8, #13) that make ingest fail loudly instead of silently producing
different chunks.

**Read first:** Tech §5.3, §5.4, §5.6, AGENTS.md §6.15.

**Scope notes**
- **#7** `includes.py`: in the `else` branch, a line that looks like a directive (`^\s*\{[*!]`) but doesn't match the
  real regex (indented `{* *}`, trailing space, a bad `{! !}`) raises `IncludeError` with `file:line`. Only the page
  source is checked, not included code.
- **#6** `markdown.py`: `_merge_prose_containers` receives `source_path`. A non-empty stack at the end of a page logs
  a `WARNING` naming the page and the opening marker(s). Blocks don't change, so `PARSER_VERSION` stays.
- **#8** `embed.py`, a **deliberate deviation from the review's proposal** (`min(server_delay, cap)`): retrying before
  the server's deadline violates AGENTS.md §6.15 ("honor Retry-After"), and rejected calls still count against the
  quota, so they would only burn attempts. Instead, when the server asks for more than
  `EMBEDDING_MAX_RETRY_WAIT_S` (new setting, default `60`, the length of the RPM/TPM window), raise
  `ProviderRateLimited` at once with a clear message. Otherwise wait exactly what the server said. The setting goes
  in `Settings`, `.env.example`, Tech §4 and §5.6.
- **#13** docs only: Tech §5.4 states that `content_hash` is the sha256 of the text after front matter, includes and
  Jinja raw markers (HTML comments, Jinja HTML and `hr` are dropped later, in parsing, after the hash; the code stays
  as is because changing it would change stored hashes). DB.md §4 notes that the `"overlap":50` comment in the applied
  `0001` is stale and DB.md wins.

**Acceptance criteria**
- [ ] `test_includes.py`: three shapes of a fake directive fail with `file:line`.
- [ ] `test_markdown.py`: an unclosed marker logs a warning (`caplog`), blocks are unchanged.
- [ ] `test_embed.py`: over the limit raises without any `sleep`; under it waits the server's delay.
- [ ] `grounded ingest --dry-run` on 0.141.1: 125 pages, 1,045 chunks, **"0 texts to embed"**, no include errors, no
      marker warnings (proves the chunks are byte-identical).

**Verify:** standard checks, then the dry run above.

**Eval impact:** none (proven by the dry run).

**Docs to update:** Tech §4, §5.4, §5.6, DB.md §4, `.env.example`.
