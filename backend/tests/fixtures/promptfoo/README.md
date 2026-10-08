# promptfoo results sample

`results_sample.json` is the JSON file `promptfoo eval -o <file>.json` wrote (promptfoo 0.123.1, 2026-10-08) for
the 4.05 harness, so the parser of the generation results (4.07) is built on the real shape and not on a guess.

**What ran.** The real promptfoo, the real `eval/promptfoo/` config, tests loader and assertions, and the real
`AskPipeline` over the real index and the real provider code (`evals/promptfoo_provider.py`). **Only the model was
scripted**, by question: `q003` and `q007` answered (the second with an invented citation label, so citation validity
fails), `q045` refused, `q008` returned invalid output twice (`ProviderBadOutput`), `q015` raised
`BackoffExhaustedError`, `q047` raised a daily-quota `ProviderRateLimited`, and `q049` came after that and was
skipped. Both configs ran (`no_rag` and `hybrid`), so 14 rows. The numbers in it (tokens, cost, latencies) are
scripted values, not measurements. It is a shape sample, not an eval result.

**What was trimmed to keep the file small** (nothing else was changed): every `context[].content` is cut to 60
characters, `context` to its first 2 chunks, `retrieved_section_ids` to its first 3, each citation `snippet` to 60
characters; `metadata.exportedAt` is fixed; the JSON is written compactly.

**What to read from it** (the contract is Tech §15.3, `tests/unit/test_promptfoo_results_sample.py` pins it):

- `results.results[]` is one row per question and provider. `provider.label` is `no_rag` or `hybrid`.
  `metadata.golden` (the test's golden payload, merged with the provider's metadata) holds the item and the golden-set
  version and sha256. `testCase.metadata.golden.golden_set_sha256` is `"[REDACTED]"`: promptfoo redacts any string of
  64 or more token characters in the echo of the test case, and does not in `results[].metadata`. Read the row-level copy.
- A row with `failureReason: 2` is an errored case: `error` is `"[<ErrorClass> quota=<true|false>] <message>"`,
  `gradingResult` is `null`, no assertion ran, and `response.metadata` has `error_kind`, `error_bases`, `is_quota`
  and what the request had spent. `failureReason: 1` is a failed assertion, `0` a pass.
- A scored row has `gradingResult.componentResults[]`, one per assertion, with `assertion.metric` as the name. A
  component with `not_applicable: true` has a placeholder `score` of 1 and must be left out of every mean.
