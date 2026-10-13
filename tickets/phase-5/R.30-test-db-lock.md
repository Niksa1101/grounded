# R.30 — A test database that survives parallel runs

> **Before you start:** read [AGENTS.md](../../AGENTS.md), the rules in [tickets/README.md](../README.md) (how to work a ticket, ticket types, [standard checks](../README.md#standard-checks)) and the phase notes in [README.md](README.md#second-review-follow-ups-2026-10-12). Work only on this ticket.

> **Owner:** G · **Type:** Build · **Branch:** `phase-5/test-db-lock` · **Blocked by:** R.29 ·
> **Builds on:** `backend/tests/conftest.py` (`test_database_url`)

**What to build:** review item #4. Two pytest sessions on the same machine drop each other's `grounded_test`
(`DROP DATABASE … WITH (FORCE)` at session start): a second run killed the first one's connections and produced 302
errors. Two sessions must not share the database at the same time, and must not corrupt each other.

**Read first:** AGENTS §8 (testing rules); Tech §16 (testing strategy); `tests/conftest.py`.

**Scope notes**
- **A session advisory lock.** In the `test_database_url` fixture, after the `admin` connection and before DROP/CREATE,
  take a session advisory lock on `hashtext(<dbname>)`, held on the `admin` connection for the whole session. The lock
  is released when that connection closes, also if the process dies, so nothing stays behind.
- **Waiting, not failing.** The lock is taken with `pg_try_advisory_lock`. If another session holds it, print one line
  to `sys.__stderr__` ("another pytest session is using <db>, waiting…") and retry once a second, up to 15 minutes.
  After that, `pytest.fail` with a message: wait for the other run, or point `TEST_DATABASE_URL` at another local
  `*_test` database.
- **Real parallel work** stays possible by giving each session its own database through `TEST_DATABASE_URL`.
- **No change to what the tests see:** the fixture still yields a fresh, migrated database.

**Acceptance criteria**
- [x] An integration test opens a second admin connection and checks that `pg_try_advisory_lock(hashtext(dbname))`
      returns false while the session runs (the lock is held), and that the key is the one the fixture uses.
- [x] The wait loop is covered without sleeping on a wall clock: its clock and sleep are injected (a small helper in
      `tests/support` or the conftest), and a unit test drives it (lock free at once; free after N tries; never free
      until the deadline, which fails).
- [x] Two parallel `uv run pytest tests/integration/test_schema.py` in two terminals: the second waits and both pass.

**Verify:** standard checks; the two-terminal run above.

**Eval impact:** none.

**Docs to update:** AGENTS §10 (parallel runs wait for each other; `TEST_DATABASE_URL` for real parallel work; a longer
tool timeout); Tech §16 (testing strategy).
