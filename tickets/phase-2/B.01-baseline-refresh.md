# B.01 — Baseline refresh: index v2, golden-set hash, retrieval config hash

> **Before you start:** read [AGENTS.md](../../AGENTS.md), the rules in [tickets/README.md](../README.md) (how to work a ticket, ticket types, [standard checks](../README.md#standard-checks)) and the phase notes in [README.md](README.md). Work only on this ticket.

> **Owner:** G runs, **A approves** · **Type:** Baseline · **Branch:** `phase-2/baseline-refresh` ·
> **Blocked by:** 2.01, R.07, R.08 · **Builds on:** R.03 (baseline schema), R.07 (index v2), 2.01 (config hash)

**What to build:** the one regeneration of the `dense` baseline row that closes review items #1, #2 and #5. It waits
until every change that affects the baseline is merged, so the file is regenerated once.

**Scope notes**
- PR title: `eval: update baseline (index v2, golden set hash, retrieval config hash)`.
- On a clean tree (untracked files no longer count after R.02):
  `uv run grounded eval retrieval --config dense --golden ../eval/golden/golden_set.vN.jsonl --write-baseline`.
- The `dense` row gets `golden_set_sha256`, `git_dirty=false` and `retrieval_config_hash`.
- The README table (row `dense`, around line 63) is rewritten from the baseline file, never typed.
- Embedding calls: 0 (everything is cached after R.07).

**Acceptance criteria**
- [x] `git diff eval/baselines/` contains only the expected fields and the metric changes from index v2.
- [x] The description has a before/after table with `n`.
- [ ] The Author has approved explicitly before the merge.

**Verify:** standard checks, and the command above.

**Eval impact:** **yes** (a new index). Label `run-eval`.

**Docs to update:** README roadmap table.
