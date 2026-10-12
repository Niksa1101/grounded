# R.24 — Deterministic deadline tests and env isolation in tests

> **Before you start:** read [AGENTS.md](../../AGENTS.md), the rules in [tickets/README.md](../README.md) (how to work a ticket, ticket types, [standard checks](../README.md#standard-checks)) and the phase notes in [README.md](README.md#review-follow-ups). Work only on this ticket.

> **Owner:** G · **Type:** Build (tests only) · **Branch:** `phase-5/deterministic-deadline-tests` ·
> **Blocked by:** R.23 · **Builds on:** `tests/integration/test_deadline.py`, `tests/conftest.py`, `api/deadline.py`

**What to build:** review item #4. `tests/integration/test_deadline.py` depends on wall-clock speed: a 0.2 s deadline
must outlast every stage before the LLM call. That breaks AGENTS.md §8 (no reliance on wall-clock timing) and it
flaked once. The tests also inherit the developer's environment (a `.env`-style `DATABASE_URL`, `FORCE_COLOR`), which
needed per-test workarounds in #97.

**Read first:** AGENTS.md §8; `api/deadline.py`; the #97 workarounds (e.g. the `delenv` loop in
`tests/unit/test_vercel_bundle.py`).

**Scope notes**
- **`test_deadline.py`.** The deadline expires exactly when the hanging stage starts:
  - a fixture monkeypatches the `asyncio` name **inside `grounded.api.deadline` only** with a small object whose
    `timeout(delay)` creates a real `asyncio.timeout(delay)`, records the scope and returns it;
  - `DEADLINE_S = 30.0`;
  - `HangingProvider.generate` and `HangingEmbedder.embed` call an `on_start` callback that does
    `scopes[-1].reschedule(asyncio.get_running_loop().time())`, then await the never-set event;
  - `latency_total_ms >= round(DEADLINE_S*1000) - 5` becomes `is not None`;
  - every other assertion stays: 504, `request_id`, the row, the stages, `llm.cancelled`, the budget unit.
- **`tests/conftest.py`.** A new autouse fixture deletes every env var named after a `Settings` field
  (`{n.upper() for n in Settings.model_fields}`) except `TEST_DATABASE_URL`, deletes `FORCE_COLOR` and sets
  `NO_COLOR=1`. Tests that set env vars themselves (`test_settings.py`) still work, because the fixture runs before
  the test body. Remove the per-test workarounds added for #97 once the fixture covers them.

**Acceptance criteria**
- [ ] `test_deadline.py` has no timing-dependent assertion and no deadline below 30 s.
- [ ] The autouse fixture exists and the #97 workarounds are gone.
- [ ] The CI-like environment passes: `DATABASE_URL=postgresql://grounded:grounded@localhost:5433/grounded
      TEST_DATABASE_URL=postgresql://grounded:grounded@localhost:5433/grounded_test uv run pytest`.
- [ ] `test_deadline.py` passes 20 times in a loop while 12 CPU-busy processes run: 20/20.

**Verify:** standard checks plus the two runs above.

**Eval impact:** none.

**Docs to update:** none expected (tests only); Tech §16 if it describes the deadline tests.
