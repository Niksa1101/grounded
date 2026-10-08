### Generation gate: ✅ pass

| config | metric | baseline | current | Δ | threshold | n | |
|---|---|---:|---:|---:|---:|---:|:-:|
| hybrid | citation_precision | — | 0.500 | — | not gated | 3 | · |
| hybrid | citation_validity | — | 0.750 | — | not gated | 4 | · |
| hybrid | correctness | 0.875 | 0.875 | +0.000 | ≥ 0.795 | 4 | ✅ |
| hybrid | faithfulness | 0.500 | 0.500 | +0.000 | ≥ 0.450 | 2 | ✅ |
| hybrid | refusal_correctness | 1.000 | 1.000 | +0.000 | ≥ 0.930 | 4 | ✅ |
| hybrid | schema_first_try | 1.000 | 1.000 | +0.000 | ≥ 0.950 | 4 | ✅ |
| no_rag | citation_precision | — | — | — | not gated | 0 | · |
| no_rag | citation_validity | — | — | — | not gated | 0 | · |
| no_rag | correctness | 0.125 | 0.125 | +0.000 | not gated | 4 | · |
| no_rag | faithfulness | — | — | — | not gated | 0 | · |
| no_rag | refusal_correctness | — | 0.750 | — | not gated | 4 | · |
| no_rag | schema_first_try | — | 1.000 | — | not gated | 4 | · |

· = not gated: reported only (no thresholds, or the config is inconclusive).

| config | cases | n | n faithfulness | provider errors | generator bad output | judge bad output | malformed | latency p50 / p95 (ms) | cost / 1k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hybrid | 4 | 4 | 2 | 0 | 0 | 1 | 0 | 15 / 18 (n=3) | $1.1702 |
| no_rag | 4 | 4 | 0 | 0 | 0 | 0 | 0 | 0 / 0 (n=3) | $1.1550 |
