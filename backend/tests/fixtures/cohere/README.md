# Cohere fixtures

`CohereReranker` (`retrieval/rerank.py`) is tested against these files, replayed through the real
`cohere` SDK client over an `httpx.MockTransport` (`tests/rerank_rig.py`), so the SDK's request
building, response parsing and error classes run for real, a retry would show up as a second HTTP
request, and no test touches the network.

**None of them is recorded.** No real Cohere call was made when they were written (6.01 and 6.02:
there was no `COHERE_API_KEY` in `.env`, and the Phase 6 budget is 150 real calls). Every file
follows the documented shape of `POST /v2/rerank` (`docs.cohere.com/reference/rerank`, read
2026-10-12) and the SDK's own response and error classes, and says so in its `_note`. The scores and
the error wording are made up. A real recording replaces a file only after a real call, and the
`_synthetic` and `_note` fields go with it.

| File | What it stands for |
|---|---|
| `rerank_success.json` | a 200 for five documents, no `top_n`, so all five results, best first |
| `rerank_429_retry_after.json` | a 429 with a `retry-after` header (hypothetical: Cohere does not document one) |
| `rerank_429_no_retry_after.json` | the 429 the API reference documents: no header |
| `rerank_500.json` | a 5xx |
| `rerank_401.json` | a rejected key |
| `rerank_404.json` | a model the key cannot use |

The malformed 200s (wrong count, repeated or out-of-range `index`, a score outside `[0, 1]`, a
missing field) are not files: the tests derive them from `rerank_success.json` in code, one change
each, so the difference from the valid shape is visible next to the assertion.
