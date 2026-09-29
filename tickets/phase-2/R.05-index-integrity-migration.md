# R.05 — Migration `0002`: index integrity

> **Before you start:** read [AGENTS.md](../../AGENTS.md), the rules in [tickets/README.md](../README.md) (how to work a ticket, ticket types, [standard checks](../README.md#standard-checks)) and the phase notes in [README.md](README.md). Work only on this ticket.

> **Owner:** G · **Type:** Build · **Branch:** `phase-2/index-integrity-migration` · **Blocked by:** R.04 ·
> **Builds on:** `migrations/0001_init.sql`, `ingest/pipeline.py` (`_MARK_READY`, `_store`)

**What to build:** two database guarantees from the review (#9, #10), in a new migration (approved 2026-09-29):

```sql
-- At most one ready version per config (two concurrent ingests of the same config).
CREATE UNIQUE INDEX index_versions_one_ready_per_config
    ON index_versions (config_hash) WHERE status = 'ready';
-- A chunk's document belongs to the chunk's index version.
ALTER TABLE documents ADD CONSTRAINT documents_id_version_key UNIQUE (id, index_version_id);
ALTER TABLE chunks ADD CONSTRAINT chunks_document_same_version_fkey
    FOREIGN KEY (document_id, index_version_id)
    REFERENCES documents (id, index_version_id) ON DELETE CASCADE;
ALTER TABLE chunks DROP CONSTRAINT chunks_document_id_fkey;  -- superseded by the composite FK
```

**Read first:** DB §4, §7.3, §10 (migrations), Tech §5.7.

**Scope notes**
- The old FK becomes redundant and is dropped. No data is lost, but the PR says so (DB §10).
- `pipeline.py`: a `UniqueViolation` on that index at `_MARK_READY` becomes a clear `IngestError` ("another ingest
  built this config; re-run to reuse it"), without a `failed` row (it would only be noise).
- Applied migrations are never edited. Neon is not touched (Phase 5).

**Acceptance criteria**
- [ ] A second `ready` row with the same hash is rejected.
- [ ] A chunk pointing at a document of another version is rejected.
- [ ] Storing the same spec twice gives the clear error.
- [ ] The migration runs over a database that already holds v1 data.
- [ ] Locally: `grounded migrate`, then `grounded index list` shows v1 unchanged.

**Verify:** standard checks, `uv run grounded migrate`, `uv run grounded index list`.

**Eval impact:** none.

**Docs to update:** DB.md §4 (a block for `0002`) and §7.3, Tech §5.7 (reuse is now also a DB guarantee).
