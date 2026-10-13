"""The session lock on the test database, against a real server (R.30).

Two pytest sessions on one machine used to drop each other's ``grounded_test``. The
``test_database_url`` fixture now holds an advisory lock named after the database for the whole
session; these tests look at it from a second connection, the way a second session would.
"""

from __future__ import annotations

import io

import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from tests.db_lock import LOCK_SQL, lock_test_database
from tests.support import FakeClock

pytestmark = pytest.mark.integration


def second_session(test_database_url: str) -> psycopg.Connection[tuple[object, ...]]:
    """What a second pytest session would open: an admin connection to the maintenance database."""
    return psycopg.connect(make_conninfo(test_database_url, dbname="postgres"), autocommit=True)


def test_this_session_holds_the_lock_on_its_test_database(test_database_url: str) -> None:
    dbname = str(conninfo_to_dict(test_database_url)["dbname"])

    with second_session(test_database_url) as other:
        taken = other.execute(LOCK_SQL, (dbname,)).fetchone()

    assert taken == (False,)  # the fixture's connection owns it


def test_a_session_with_another_test_database_is_not_blocked(test_database_url: str) -> None:
    with second_session(test_database_url) as other:
        taken = other.execute(LOCK_SQL, ("grounded_r30_other_test",)).fetchone()

    assert taken == (True,)  # TEST_DATABASE_URL on another *_test database runs in parallel


def test_a_waiting_session_gets_the_lock_when_the_holder_closes(test_database_url: str) -> None:
    name = "grounded_r30_contention_test"
    holder = second_session(test_database_url)
    assert holder.execute(LOCK_SQL, (name,)).fetchone() == (True,)
    clock = FakeClock()
    out = io.StringIO()

    def holder_leaves(seconds: float) -> None:
        clock.advance(seconds)
        holder.close()  # a closed connection releases its session locks, a killed process too

    with second_session(test_database_url) as waiting:
        taken = lock_test_database(waiting, name, sleep=holder_leaves, now=clock, out=out)

    assert taken is True
    assert out.getvalue() == f"another pytest session is using {name}, waiting...\n"
