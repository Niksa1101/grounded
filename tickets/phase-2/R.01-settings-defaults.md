# R.01 — Defaults for the corpus tag and the embedding model

> **Before you start:** read [AGENTS.md](../../AGENTS.md), the rules in [tickets/README.md](../README.md) (how to work a ticket, ticket types, [standard checks](../README.md#standard-checks)) and the phase notes in [README.md](README.md). Work only on this ticket.

> **Owner:** G · **Type:** Build · **Branch:** `phase-2/settings-defaults` · **Blocked by:** R.00 ·
> **Builds on:** `settings.py`, `cli.py` (`ingest`, `eval retrieval`, `_prepare_corpus`), `ingest/embed.py`

**What to build:** `FASTAPI_REF` and `EMBEDDING_MODEL` get defaults in `Settings` (review item #3, decided
2026-09-29). Both are already pinned in Tech §4, and the embedding model was verified on 2026-09-26. With defaults, a
fresh checkout can run `grounded ingest --dry-run` without a `.env`, and the CI job (2.08, 2.11) needs no variable for
either.

**Read first:** Tech §4, Tech §5.6, AGENTS.md §6.8 (pinned model IDs), PRD §12 (the sentence about model IDs).

**Scope notes**
- `embedding_model: str = Field(default="gemini-embedding-001", min_length=1)` and
  `fastapi_ref: str = Field(default="0.141.1", min_length=1)`. An empty value is rejected, not treated as "unset".
- The module docstring says: generator, judge and rerank models have no default until they are verified; the
  embedding model was verified on 2026-09-26 and is tied to the index (a different value is a new index version).
- Remove the checks that can no longer fire: `if not settings.embedding_model` in `ingest` and `eval retrieval`,
  `if not ref` in `_prepare_corpus`, and the model half of `GeminiEmbedder.from_settings`' check (it validates only
  the key).
- `.env.example`: `# EMBEDDING_MODEL=gemini-embedding-001` and `# FASTAPI_REF=0.141.1` (a comment shows the default,
  like the other rows).

**Acceptance criteria**
- [ ] Defaults match Tech §4, and an empty `EMBEDDING_MODEL` / `FASTAPI_REF` fails validation.
- [ ] `test_ingest_needs_a_ref` and `test_ingest_needs_an_embedding_model` are replaced by the two tests above.
- [ ] `grounded ingest --dry-run` works without `--ref`.
- [ ] `grounded index list` shows the same config hash for the active version (values equal the Author's `.env`).

**Verify:** standard checks, then `uv run grounded ingest --dry-run` and `uv run grounded index list`.

**Eval impact:** none.

**Docs to update:** Tech §4 (default column), AGENTS.md §10 (`ingest` without a required `--ref`), PRD §12 (the pinned
model-ID sentence gets the embedding-model exception), `.env.example`.
