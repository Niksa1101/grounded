# R.13 — Compact retry feedback and the tokens of failed attempts

> **Before you start:** read [AGENTS.md](../../AGENTS.md), the rules in [tickets/README.md](../README.md) (how to work a ticket, ticket types, [standard checks](../README.md#standard-checks)) and the phase notes in [README.md](README.md). Work only on this ticket.

> **Owner:** G · **Type:** Build · **Branch:** `phase-3/retry-feedback` · **Blocked by:** R.12 ·
> **Builds on:** `infra/provider_errors.py`, `generation/providers/gemini.py`, `generation/providers/fake.py`,
> `generation/pipeline.py`

**What to build:** review items #5 and #6.

- **#5:** the retry quotes `str(ValidationError)` back to the model: it carries `input_value` (up to the whole bad
  output) and `https://errors.pydantic.dev/...` links, which contradict the prompt's "do not write URLs". On a
  `MAX_TOKENS` cut the finish reason is only in the exception message, not in the feedback, so the model is not told
  why and the retry, with the same limit, likely fails again. Decision of 2026-10-07: compact feedback, plus an explicit
  sentence on truncation; same `max_output_tokens`.
- **#6:** the adapter raises `ProviderBadOutput` without usage, so a request whose output fails the schema twice logs
  0 tokens and 0 cost, though both calls were billed. The budget (5.05) will rely on these numbers.

**Read first:** Tech §9.1, §9.5, §14; `prompts/answer_v1.md` (`# Retry feedback`).

**Scope notes**
- `infra/provider_errors.py`: `compact_validation_error(exc: ValidationError) -> str` built on
  `exc.errors(include_url=False, include_input=False)`, as `"<loc>: <msg>; …"`. `ProviderBadOutput` gets
  `input_tokens: int = 0` and `output_tokens: int = 0` (plain ints, so `infra` does not import `generation`).
- `providers/gemini.py`: usage is read before validation and passed to `ProviderBadOutput`, also for the no-candidate
  case. `validation_error` is the compact text; when `finish_reason == "MAX_TOKENS"` it is "the answer was cut off at
  the output token limit before the JSON was complete; write a shorter answer with fewer, shorter claims".
- `providers/fake.py`: the same helper, so tests see the same shape.
- `pipeline._generate_validated`: on `ProviderBadOutput`, add the exception's tokens to the trace before deciding on
  the retry. The citation-check error the pipeline raises itself carries 0 (that attempt's usage was already added),
  so nothing is counted twice.
- `prompts/answer_v1.md` does **not** change (the prompt hash stays); only the text filled into `{{error}}` does.

**Acceptance criteria**
- [x] `test_gemini_provider.py`: the feedback has no `input_value` and no `errors.pydantic.dev`; a `MAX_TOKENS` reply
      gets the truncation sentence; `ProviderBadOutput` carries the usage of the recorded `generate_invalid_output.json`.
- [x] `test_request_log.py::test_output_that_fails_validation_is_logged_with_what_it_spent` now expects the tokens of
      both attempts and a shadow cost above 0.
- [x] `test_ask.py`: the retry's user message ends with the compact feedback.

**Verify:** standard checks. The real-provider golden run is done once after R.14 (see the phase README).

**Eval impact:** yes (what the model sees on a retry) → `run-eval` label. There is no generation gate yet; the
2026-10-07 golden run had 0 retries, so a change is not expected. Report it after R.14.

**Docs to update:** Tech §9.1, §9.5, §14 (failed attempts are charged), the `pipeline.py` and `request_log.py`
docstrings.
