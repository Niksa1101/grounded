-- 0002_index_integrity.sql
-- Two guarantees the ingest code relied on but the schema did not enforce (review items #9, #10).

-- At most one ready version per config: two concurrent ingests of the same config would otherwise
-- both pass the "is there a ready version?" check and both store one. The loser fails when it marks
-- its version ready (ingest reports that as a clear error, not a failed row).
CREATE UNIQUE INDEX index_versions_one_ready_per_config
    ON index_versions (config_hash) WHERE status = 'ready';

-- A chunk's document belongs to the chunk's index version. Before, chunks.index_version_id and
-- chunks.document_id were checked separately, so a chunk could point at a document of another
-- version. The composite key makes the pair one reference.
ALTER TABLE documents ADD CONSTRAINT documents_id_version_key UNIQUE (id, index_version_id);
ALTER TABLE chunks ADD CONSTRAINT chunks_document_same_version_fkey
    FOREIGN KEY (document_id, index_version_id)
    REFERENCES documents (id, index_version_id) ON DELETE CASCADE;
-- The single-column FK is now implied by the composite one; keeping both would only add a check.
ALTER TABLE chunks DROP CONSTRAINT chunks_document_id_fkey;
