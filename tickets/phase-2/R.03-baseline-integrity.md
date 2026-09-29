# R.03 — Baseline integrity: golden-set hash and the mixing guard

> **Before you start:** read [AGENTS.md](../../AGENTS.md), the rules in [tickets/README.md](../README.md) (how to work a ticket, ticket types, [standard checks](../README.md#standard-checks)) and the phase notes in [README.md](README.md). Work only on this ticket.

> **Owner:** G · **Type:** Build · **Branch:** `phase-2/baseline-integrity` · **Blocked by:** R.02 ·
> **Builds on:** `schemas/eval.py`, `evals/retrieval_runner.py` (`update_baseline`), `cli.py`

**What to build:** `--write-baseline` can no longer produce a baseline file whose rows come from different setups
(review items #1 and #2, decided 2026-09-29: **refuse mixing**).

**Read first:** Tech §15.2, Tech §15.7, AGENTS.md §7.

**Scope notes**
- `RetrievalRunInfo.golden_set_sha256: str` is required in new runs (old local results files are gitignored and are
  discarded). `RetrievalBaselineEntry` gains `golden_set_sha256` and `git_dirty`, both `| None = None`: optional only
  so the current `dense` row can still be read. `None` means "written before this field".
- `golden_set_digest(path)`: sha256 of the file bytes with CRLF→LF (same normalization as the migration checksum).
- `update_baseline` compares the identity of every **kept** row (a config not in this run) with the run:
  `golden_set_version`, `golden_set_sha256`, `index_config_hash`. The index hash already covers the FastAPI SHA, the
  embedding model and dimension and the chunking config. A difference raises `BaselineMismatchError` listing the
  fields and the command for a joint run (`--config dense --config fts …`). The file stays untouched. A legacy row
  with `None` counts as a mismatch.
- The CLI still writes the results file, then reports "Baseline not updated: …" and exits 1. The dirty-tree warning
  stays.
- **This PR does not touch `eval/baselines/retrieval.json`** (tests use `tmp_path`). B.01 regenerates it.

**Acceptance criteria**
- [ ] Each identity field is refused on its own, and the file is byte-identical afterwards.
- [ ] A legacy `None` row is refused; a joint run of every config passes.
- [ ] The digest ignores CRLF/LF and changes with content; the new fields are copied into the baseline.
- [ ] CLI: exit 1 with the message, results file written.
- [ ] `eval retrieval --config dense` gives the same metrics as the committed `dense` row.

**Verify:** standard checks, then `uv run grounded eval retrieval --config dense` (no `--write-baseline`).

**Eval impact:** none (bookkeeping).

**Docs to update:** Tech §15.2 (results file fields, the refusal rule), Tech §15.7 (baseline fields),
`eval/README.md` if it describes `--write-baseline`.
