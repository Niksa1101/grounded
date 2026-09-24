# Evaluation

The eval harness that gates quality-affecting PRs (see [docs/Tech.md](../docs/Tech.md) and AGENTS.md §7).

| Path | Contents | Arrives in |
|---|---|---|
| `golden/` | versioned golden set (`golden_set.vN.jsonl`) and the labeling guide | Phase 1 |
| `baselines/` | committed baseline metrics; changed only by an `eval: update baseline (<reason>)` PR | Phase 1 (retrieval), Phase 4 (generation) |
| `promptfoo/` | generation eval config, provider and asserts | Phase 4 |
| `results/` | run outputs, **gitignored** | — |

Metric numbers in the README, PRs and baselines come only from committed eval output. Never edit them by hand.
