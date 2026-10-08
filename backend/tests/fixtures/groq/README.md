# Groq fixtures

`GroqProvider` is tested against these files, replayed through the real `groq` SDK client over an
`httpx.MockTransport` (`tests/provider_rigs.py`), so no test touches the network.

| File | Origin |
|---|---|
| `generate_success.json` | **Recorded** from a real `openai/gpt-oss-120b` call, 2026-10-08 (ids replaced, timings removed) |
| `generate_invalid_output.json` | **Recorded**: `max_completion_tokens=20`, which the reasoning alone uses up, gives `400 json_validate_failed` with an empty `failed_generation` (how a cut-off reply arrives) |
| `generate_400_schema_violation.json` | **Recorded**: Groq's own check of the finished JSON against a `maxLength` rejects it with `400 json_validate_failed` and the text in `failed_generation` |
| `generate_429_per_minute.json` | **Hand-made** (`"_synthetic": true`): a 429 can't be provoked on demand |
| `generate_429_daily_quota.json` | **Hand-made**: same |
| `generate_500.json` | **Hand-made**: a 5xx can't be provoked on demand |

The hand-made files take their shape from the SDK's error classes and the recorded 4xx bodies. Groq
documents neither the body of a 429 nor how a daily limit differs from a per-minute one, so the
wording in them is modeled, not observed (see the `groq.py` docstring). Replace them with recordings
when a real 429 or 5xx is seen.

`record_generate_fixture.py` records the three recorded kinds again (one real call each, through
`GroqProvider`; read its docstring). Never run by pytest.
