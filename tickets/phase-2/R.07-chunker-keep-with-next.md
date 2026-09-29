# R.07 — Chunker: keep a heading with the block after it

> **Before you start:** read [AGENTS.md](../../AGENTS.md), the rules in [tickets/README.md](../README.md) (how to work a ticket, ticket types, [standard checks](../README.md#standard-checks)) and the phase notes in [README.md](README.md). Work only on this ticket.

> **Owner:** G (**delegated by the Author for this change, 2026-09-29**, AGENTS.md §3) · **Type:** Build ·
> **Branch:** `phase-2/chunker-keep-with-next` · **Blocked by:** R.06 · **Builds on:** `ingest/chunker.py`,
> `ingest/pipeline.py` (`index_chunking_config`)

**What to build:** review item #5. A chunk must never end with a heading (today `deployment/docker.md#dockerfile`
does). This is an edit of an Author-owned module, written by the Agent only because the Author delegated it here. The
Agent explains the change line by line afterwards.

**Read first:** Tech §5.5, §5.7, the module docstring of `chunker.py`, PRD §9 (Phase 1).

**Rule (extends rule 4 of the chunker docstring):** a heading never ends a part. A heading is an H4+ line, or an
H2/H3 line of a thin parent that rule 2 carried over. If the block after a heading doesn't fit in the current part, the
heading moves to the next part together with that block. No overlap is added before a heading.

**Scope notes**
- `_pieces`: every `HeadingBlock` becomes a `"heading"` piece, not a `"block"`.
- `_Packer._place`: after the "fits in current" check and the "current holds only headings" branch, if the current part
  ends in headings, cut those headings into a new part (`_start(tail)`) and place `piece` again. The existing rule
  "a heading is never alone" then applies: the block splits, or stays whole above max if atomic.
- Overlap branch: if `piece` is a heading, the new part starts without overlap, otherwise moving a heading would leave
  a part made only of copied sentences.
- Optional simplification: the `head` loop in `pack()` may become redundant, if the tests confirm it.
- `CHUNKER_VERSION = 2` in `chunker.py` and `"chunker_version"` in `index_chunking_config`. This closes the note in
  Tech §5.7 that the chunker has no version. The config hash changes, so this builds a new index version.
- The module docstring records the delegation ("keep-with-next delegated by the Author, 2026-09-29").

**Acceptance criteria**
- [x] Tests (word counter, 1 word = 1 token): an H4 followed by a block that doesn't fit moves to the next part; a chain
      intro → empty H2 → H3 whose first block doesn't fit moves together; no overlap before a heading; a moved heading
      plus an oversized atomic block stay together above max.
- [x] All existing tests pass (an expectation of overlap before an H4, if any, is changed with an explanation).
- [x] `test_ingest_pipeline.py`: `chunker_version` is in the config and changes the hash.
- [ ] Real corpus, locally, **with the Author's confirmation** (it spends their embedding key for a few calls):
      (1) an out-of-repo script finds 0 chunks whose last non-empty line is a heading (before: ≥ 1);
      (2) `ingest --dry-run` reports N texts to embed (N = changed chunks, expected small; the cache is content
      addressed); (3) `ingest --activate` builds v2, `golden validate --against-index` passes, and
      `eval retrieval --config dense` runs **without** `--write-baseline`.

**Verify:** standard checks plus the corpus steps above.

**Eval impact:** **yes**, a new index. Label `run-eval`. The description has dense v1 vs v2 (Recall@5/@10, MRR,
nDCG@5/@10, n=25) and the number of changed chunks. A difference smaller than one question is noise.

**Docs to update:** Tech §5.5 (the rule, the new stats copied from `grounded index list`), Tech §5.7
(`chunker_version`), DB.md (`chunking_config` example), PRD §9 (delegation note).
