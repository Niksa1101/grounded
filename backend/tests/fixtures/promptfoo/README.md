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

## `results_sample_judge.json` (4.06)

The same kind of recording for the full config, with the two judge metrics (`faithfulness`, `correctness`), so the
parser of 4.07 sees the real shape of a judge component: scored, not applicable, and errored by the judge.
promptfoo 0.123.1, 2026-10-08, `-j 1 --no-cache`, 7 golden questions (`q003 q007 q008 q015 q045 q047 q049`) and both
configs, 14 rows, all graded (a judge error does not error the row).

**What ran.** The real promptfoo, the real config, test loader and assertions, the real `AskPipeline` over the real
index, and the real judge module. **Only the two models were scripted** (nothing touched the network): a generator that
answers each question with fixed claims (so `q007` cites an invented label and `q045` refuses), and a judge that returns
fixed verdicts keyed by the claim or the question, and fails on cue: invalid output twice (`q008`, hybrid claim 1), a
timeout (`q015`, correctness), and a daily quota (`q047`, hybrid correctness). The tokens, costs, latencies and the
judge's `scripted-judge` model are scripted values, and the wording of the answers and reasons is made up. It is a shape
sample, not an eval result. The rows after the quota (`q049`, and the other assertion of `q047` hybrid) show the stop:
every later judge assertion of the run says "skipped" and made no call.

**How it was recorded** (the throwaway shims are not committed): copies of `eval/promptfoo/promptfooconfig.yaml` and
`tests_loader.py` next to a `provider.py` and an `asserts.py` that set `APP_ENV=eval` and `GENERATOR_PROVIDERS=fake`
and replace `grounded.runtime.StubLLMProvider` / `StubJudgeProvider` with the scripted classes (a subclass of the stub
is never put behind the eval LLM cache, so the real cache file stays clean), with `CACHE_DIR` pointing at an empty
directory for the assertion processes.

**What was trimmed** (nothing else was changed): the same as for `results_sample.json` (`context[].content` to 60
characters, `context` to its first 2 chunks, `retrieved_section_ids` to 3, each citation `snippet` to 60 characters;
`metadata.exportedAt` fixed, `config.outputPath` neutral, compact JSON).

**What to read from it** (the contract is Tech §15.3; `tests/unit/test_promptfoo_judge_results_sample.py` pins it):

- Every judge component has `not_applicable` and `errored` (booleans, never both true). A component with neither is
  scored: `score` is the metric and `judge` has the record. With `not_applicable` it has no `judge`. With `errored` its
  `score` 0 is a placeholder, and `judge.error` has `kind`, `is_quota`, `provider_side` and `detail`.
- A row whose judge component is errored has `failureReason: 1` (a failed assertion), never 2, and its `error` text is
  the failing reason, not a provider tag.
- `judge.claims[]` (faithfulness) holds each claim's verdict, reason, server-side `confidence`, cited labels,
  `decided_locally`, `cache_hit`, `attempts`, `usage` and `error`, in the order of `metadata.claims`.
