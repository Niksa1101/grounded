# Groq fixtures

`GroqProvider` is tested against these files, replayed through the real `groq` SDK client over an
`httpx.MockTransport` (`tests/provider_rigs.py`), so no test touches the network.

| File | Origin |
|---|---|
| `generate_success.json` | **Recorded** from a real `openai/gpt-oss-120b` call, 2026-10-08 (ids replaced, timings removed) |
| `generate_invalid_output.json` | **Recorded**: `max_completion_tokens=20`, which the reasoning alone uses up, gives `400 json_validate_failed` with an empty `failed_generation` (how a cut-off reply arrives) |
| `generate_400_schema_violation.json` | **Recorded**: Groq's own check of the finished JSON against a `maxLength` rejects it with `400 json_validate_failed` and the text in `failed_generation` |
| `generate_429_per_minute.json` | **Hand-made** (`"_synthetic": true`) until replaced by a recording: `record_generate_fixture.py 429_per_minute --burst N` provokes one on purpose (see below) |
| `generate_429_daily_quota.json` | **Hand-made** until replaced by a recording: `record_generate_fixture.py 429_daily_quota`, run right after a real daily 429 happened (a daily quota is never burned on purpose) |
| `generate_500.json` | **Hand-made**: a 5xx can't be provoked on demand |

The hand-made files take their shape from the SDK's error classes and the recorded 4xx bodies. Groq
documents neither the body of a 429 nor how a daily limit differs from a per-minute one, so the
wording in them is modeled, not observed (see the `groq.py` docstring). Replace them with recordings
when a real 429 or 5xx is seen.

`record_generate_fixture.py` records the five recordable kinds (one real call each, through
`GroqProvider`; read its docstring). Never run by pytest. The two 429 modes make one minimal call and
write the 429 only if they get one (otherwise nothing is written and they say so); the file they write
has no `_synthetic` marker or note. Before writing, `org_...` ids, request ids and key shapes are
replaced by fixed placeholders (`tests/fixtures/recording_tools.py`, unit-tested without network):
a real daily 429 names the organization, so read the file once before you commit it.

How to record them (by hand, from `backend/`, with `GROQ_API_KEY` and `GROQ_MODEL` in the root `.env`):

- **Per minute:** `uv run python tests/fixtures/groq/record_generate_fixture.py 429_per_minute --burst 12`
  sends 12 minimal requests at once (default 1, hard cap 40), keeps the first 429 and exits non-zero
  without writing if none came. It prints how many requests it is about to make, and that they use
  quota (the free plan's, shared with the CI judge).
- **Daily:** `uv run python tests/fixtures/groq/record_generate_fixture.py 429_daily_quota`, only after a
  real daily 429 (for example when `eval.yml` stopped on `ProviderRateLimited (daily quota)`). It is one
  request; `--burst` is refused with this mode.

A new recording can change what `test_provider_contract.py` and `test_groq_provider.py` expect (the
`retry-after`, the limit wording): update the expected values from the file, and keep the file as it
came.
