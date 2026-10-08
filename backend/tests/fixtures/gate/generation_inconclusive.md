### Generation gate: ⚠️ inconclusive

> ⚠️ **hybrid: inconclusive.** 3 of 7 cases (42.9%) errored for provider reasons (limit 20%: a quota, a 5xx or a timeout). Its metrics are shown with n but not gated, and an inconclusive run is not a pass: re-run it.

> ⚠️ **no_rag: inconclusive.** 2 of 7 cases (28.6%) errored for provider reasons (limit 20%: a quota, a 5xx or a timeout). Its metrics are shown with n but not gated, and an inconclusive run is not a pass: re-run it.

| config | metric | baseline | current | Δ | threshold | n | |
|---|---|---:|---:|---:|---:|---:|:-:|
| hybrid | citation_precision | — | 0.375 | — | not gated | 4 | · |
| hybrid | citation_validity | — | 0.857 | — | not gated | 7 | · |
| hybrid | correctness | 0.875 | 0.875 | +0.000 | not gated | 4 | · |
| hybrid | faithfulness | 0.500 | 0.667 | +0.167 | not gated | 3 | · |
| hybrid | refusal_correctness | 1.000 | 0.714 | -0.286 | not gated | 7 | · |
| hybrid | schema_first_try | 1.000 | 1.000 | +0.000 | not gated | 7 | · |
| no_rag | citation_precision | — | — | — | not gated | 0 | · |
| no_rag | citation_validity | — | — | — | not gated | 0 | · |
| no_rag | correctness | 0.125 | 0.100 | -0.025 | not gated | 5 | · |
| no_rag | faithfulness | — | — | — | not gated | 0 | · |
| no_rag | refusal_correctness | — | 0.571 | — | not gated | 7 | · |
| no_rag | schema_first_try | — | 1.000 | — | not gated | 7 | · |

· = not gated: reported only (no thresholds, or the config is inconclusive).

| config | cases | n | n faithfulness | provider errors | generator bad output | judge bad output | malformed | latency p50 / p95 (ms) | cost / 1k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hybrid | 7 | 4 | 3 | 3 | 0 | 1 | 0 | 14 / 18 (n=6) | $1.1686 |
| no_rag | 7 | 5 | 0 | 2 | 0 | 0 | 0 | 0 / 0 (n=6) | $1.1550 |
