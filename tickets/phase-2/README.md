# Phase 2 — Hybrid retrieval

**Goal:** a measurable lift from lexical + dense fusion, and a live CI retrieval gate (PRD §9, Phase 2).

**Exit criteria (checked in 2.12)**
- Ablation rows `dense / fts / hybrid` in `eval/baselines/retrieval.json`.
- Hybrid ≥ dense on Recall@5 and MRR, or the reason is documented.
- The retrieval gate blocks a regression in CI (demonstrated, screenshot kept).

**Phase note (decided 2026-09-28):** the Author-owned `evals/gate.py` is split across two phases. Its **retrieval**
part is written in Phase 2 (2.09–2.10), and the **generation** part plus the inconclusive rule in Phase 4 (4.07–4.08).
Ticket 2.09 updates the AGENTS.md §3 table to match.

## Review follow-ups

The code review of Phases 0 and 1 (PRs #1–#11) approved both phases and left 4 🟡 and 11 🟢 items. Most of them touch
the integrity of the baseline, which must be clean before the retrieval gate works (2.09–2.11). They are worked as small
PRs **before 2.01**, and the baseline is regenerated **once**, at the end (B.01). Decisions of 2026-09-29:

- #3: `FASTAPI_REF` and `EMBEDDING_MODEL` get defaults in `Settings` (R.01).
- #1/#2: `--write-baseline` **refuses** to mix rows from different setups. The new baseline fields are optional only for
  reading the old row (R.03). One Baseline PR at the end (B.01).
- #5: the chunker fix is written by the Agent, an explicit delegation for this change (AGENTS.md §3) (R.07).
- #9/#10: both go into migration `0002` now (R.05).
- #15: PRs are readable in one sitting (AGENTS.md §11, tickets README).

R.08 (the Author's manual label check) runs in parallel with R.01–R.06 and must be done before B.01.

- [R.01 — Defaults for the corpus tag and the embedding model](R.01-settings-defaults.md)
- [R.02 — Eval tooling fixes: dirty-tree check, lazy embedder](R.02-eval-tooling-fixes.md)
- [R.03 — Baseline integrity: golden-set hash, refuse mixing](R.03-baseline-integrity.md)
- [R.04 — Ingest guards and documentation drift](R.04-ingest-guards.md)
- [R.05 — Migration `0002`: index integrity](R.05-index-integrity-migration.md)
- [R.06 — CI hardening: pinned actions, one tokenizer source](R.06-ci-hardening.md)
- [R.07 — Chunker: keep a heading with the block after it](R.07-chunker-keep-with-next.md)
- [R.08 — Manual check of golden labels, explain-back](R.08-golden-label-check.md)
- [B.01 — Baseline refresh: index v2, golden-set hash, config hash](B.01-baseline-refresh.md) (after 2.01)

## Tickets (in order)

- [2.01 — `RetrievalConfig` and its hash](2.01-retrieval-config.md)
- [2.02 — Lexical FTS search — spec](2.02-lexical-spec.md)
- [2.03 — Lexical FTS search — implementation](2.03-lexical.md)
- [2.04 — `fts` mode in the retrieval eval](2.04-eval-fts.md)
- [2.05 — Hybrid RRF search — spec](2.05-hybrid-spec.md)
- [2.06 — Hybrid RRF search — implementation](2.06-hybrid-rrf.md)
- [2.07 — `hybrid` mode, ablation run and baseline rows](2.07-ablation.md)
- [2.08 — Decision: seeding the CI corpus and embedding caches](2.08-decision-ci-cache.md)
- [2.09 — Retrieval gate — spec](2.09-gate-spec.md)
- [2.10 — Retrieval gate — implementation](2.10-gate.md)
- [2.11 — CI `retrieval-eval` job](2.11-ci-retrieval-eval.md)
- [2.12 — Gate demonstration and Phase 2 closeout](2.12-closeout.md)

Rules for working a ticket: [../README.md](../README.md).
