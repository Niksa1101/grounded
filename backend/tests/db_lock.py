"""A session lock on the test database: two pytest runs on one machine wait for each other (R.30).

The ``test_database_url`` fixture drops and recreates ``grounded_test`` at the start of a session
(``DROP DATABASE ... WITH (FORCE)``), which kills the connections of any other session using it: a
second run used to give the first one hundreds of errors. A Postgres *session* advisory lock named
after the database fixes that without a lock file: the admin connection holds it for the whole
session, the server releases it when that connection closes, and also when the process is killed,
so nothing stays behind.

Waiting is the default because the failure is silent and confusing; a developer who wants two
sessions at once points ``TEST_DATABASE_URL`` at another local ``*_test`` database (the lock key is
the database name, so different names never wait for each other).

The loop takes its clock and its sleep as arguments, so its tests move a fake clock by hand
instead of waiting (AGENTS.md §8).
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from typing import Any, Final, LiteralString, Protocol

LOCK_TIMEOUT_S: Final = 15 * 60
LOCK_POLL_S: Final = 1.0

# hashtext is the stock way to turn a name into the bigint key an advisory lock takes. The admin
# connection is to the maintenance database, so every session that targets the same test database
# name asks for the same key in the same place.
LOCK_SQL: Final = "SELECT pg_try_advisory_lock(hashtext(%s))"


class Output(Protocol):
    """Where the wait message goes: a ``StringIO`` in a test, the real stderr in the fixture."""

    def write(self, text: str, /) -> object: ...

    def flush(self) -> object: ...


class _Cursor(Protocol):
    def fetchone(self) -> tuple[Any, ...] | None: ...


class AdminConnection(Protocol):
    """The part of a psycopg connection the lock needs (and a fake can offer).

    ``LiteralString`` because that is what psycopg accepts: a query built at run time is how SQL
    injection starts, and the type checker enforces the rule (AGENTS.md §6.11).
    """

    def execute(self, query: LiteralString, params: tuple[Any, ...] = ..., /) -> _Cursor: ...


def wait_for_lock(
    try_lock: Callable[[], bool],
    *,
    announce: Callable[[], None],
    sleep: Callable[[float], None],
    now: Callable[[], float],
    timeout_s: float,
    poll_s: float,
) -> bool:
    """Call ``try_lock`` until it says yes (True) or ``timeout_s`` has passed (False).

    ``announce`` is called once, the first time the lock is found held: one line of output, not one
    per poll.
    """
    deadline = now() + timeout_s
    announced = False
    while not try_lock():
        if now() >= deadline:
            return False
        if not announced:
            announce()
            announced = True
        sleep(poll_s)
    return True


def lock_test_database(
    admin: AdminConnection,
    dbname: str,
    *,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
    out: Output | None = None,
) -> bool:
    """Take the session lock for ``dbname`` on ``admin``, waiting for another session to finish.

    True once the lock is held (by this connection, until it closes); False after
    ``LOCK_TIMEOUT_S`` of waiting. The wait message goes to ``out`` (default ``sys.__stderr__``).
    pytest captures file descriptor 2 while a fixture runs, so the fixture passes an ``out`` that
    suspends capture for the write (``conftest.py``): a developer who sees nothing thinks the run
    hung.
    """
    stream = out if out is not None else sys.__stderr__

    def try_lock() -> bool:
        row = admin.execute(LOCK_SQL, (dbname,)).fetchone()
        return bool(row and row[0])

    def announce() -> None:
        if stream is not None:
            stream.write(f"another pytest session is using {dbname}, waiting...\n")
            stream.flush()

    return wait_for_lock(
        try_lock,
        announce=announce,
        sleep=sleep,
        now=now,
        timeout_s=LOCK_TIMEOUT_S,
        poll_s=LOCK_POLL_S,
    )
