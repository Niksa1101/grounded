"""The demo writes ``request_logs`` and ``answer_cache`` rows, so it refuses a database that is not
local (7.04b, AGENTS.md §5): ``check_local_database`` and its place in ``demo_settings``.

A Neon URL in a ``dev`` shell is the case it exists for. The guard parses the URL with libpq's own
rules, allows only ``localhost``, ``127.0.0.1`` and ``::1``, treats "cannot tell" as no, and its
message names the host and never the credentials, the URL or libpq's parse error. No database and no
network: nothing here connects.
"""

from __future__ import annotations

import pytest

from grounded.fallback_demo import DemoRefusedError, check_local_database, demo_settings
from tests.support import make_settings

PASSWORD = "hunter2pw"  # in every URL below, so a message that echoed one would show it
NEON = "ep-cool-darkness-123456.us-east-2.aws.neon.tech"


@pytest.mark.parametrize(
    "conninfo",
    [
        f"postgresql://grounded:{PASSWORD}@localhost:5433/grounded",
        f"postgresql://grounded:{PASSWORD}@127.0.0.1:5433/grounded",
        f"postgresql://grounded:{PASSWORD}@[::1]:5433/grounded",
        f"postgres://grounded:{PASSWORD}@LOCALHOST/grounded",  # case does not matter
        f"postgresql://grounded:{PASSWORD}@localhost/grounded?sslmode=disable",
        f"host=localhost port=5433 dbname=grounded user=grounded password={PASSWORD}",
        f"host=127.0.0.1 port=5433 dbname=grounded user=grounded password={PASSWORD}",
        f"host=::1 port=5433 dbname=grounded user=grounded password={PASSWORD}",
        f"host=localhost hostaddr=127.0.0.1 dbname=grounded password={PASSWORD}",
        f"host=localhost,127.0.0.1 dbname=grounded password={PASSWORD}",
    ],
)
def test_a_local_host_passes_the_guard(conninfo: str) -> None:
    check_local_database(conninfo)  # does not raise


@pytest.mark.parametrize(
    ("conninfo", "named"),
    [
        (f"postgresql://app:{PASSWORD}@{NEON}/neondb?sslmode=require", NEON),
        (f"postgresql://app:{PASSWORD}@{NEON}:5432/neondb", NEON),
        (f"host={NEON} dbname=neondb user=app password={PASSWORD} sslmode=require", NEON),
        (f"postgresql://app:{PASSWORD}@10.0.0.7:5432/grounded", "10.0.0.7"),
        (f"postgresql://app:{PASSWORD}@db.internal/grounded", "db.internal"),
        (f"postgresql://app:{PASSWORD}@localhost.example.com/grounded", "localhost.example.com"),
        # One remote entry in a failover list is enough to refuse.
        (f"host=localhost,{NEON} dbname=grounded password={PASSWORD}", NEON),
        (f"postgresql://app:{PASSWORD}@localhost:5433,{NEON}:5432/grounded", NEON),
        # libpq connects to hostaddr when it is given, whatever host says.
        (f"host=localhost hostaddr=203.0.113.9 dbname=grounded password={PASSWORD}", "203.0.113.9"),
        # A Unix socket is not on the list either.
        (f"host=/var/run/postgresql dbname=grounded password={PASSWORD}", "/var/run/postgresql"),
    ],
)
def test_a_host_that_is_not_local_is_refused_by_name(conninfo: str, named: str) -> None:
    with pytest.raises(DemoRefusedError) as caught:
        check_local_database(conninfo)

    message = str(caught.value)
    assert named in message  # the problem and the host...
    assert "local database" in message
    assert "localhost, 127.0.0.1, ::1" in message  # ...and what is allowed
    assert PASSWORD not in message  # never the credentials
    assert conninfo not in message  # nor the URL


@pytest.mark.parametrize(
    "conninfo",
    [
        f"dbname=grounded user=grounded password={PASSWORD}",  # no host: libpq would ask PGHOST
        f"postgresql:///grounded?user=grounded&password={PASSWORD}",  # nor does this one
        f"service=neon password={PASSWORD}",  # the host would come from a service file
        "",
    ],
)
def test_no_host_means_the_environment_decides_and_is_refused(conninfo: str) -> None:
    with pytest.raises(DemoRefusedError, match="names no host") as caught:
        check_local_database(conninfo)
    assert PASSWORD not in str(caught.value)


@pytest.mark.parametrize(
    "conninfo",
    [
        f"postgresql://app:{PASSWORD}@[::1/grounded",  # an unclosed IPv6 bracket
        f"not a url {PASSWORD}",  # libpq's own parse error would quote this
        f"host=localhost {PASSWORD}",
    ],
)
def test_an_unparsable_url_is_refused_and_libpqs_message_is_not_passed_on(conninfo: str) -> None:
    with pytest.raises(DemoRefusedError, match="cannot be parsed") as caught:
        check_local_database(conninfo)

    assert PASSWORD not in str(caught.value)
    assert caught.value.__cause__ is None  # not chained to the parse error either
    assert caught.value.__suppress_context__


def test_the_guard_is_part_of_the_settings_the_demo_runs_on() -> None:
    models = {"gemini_model": "gemini-3.5-flash-lite", "groq_model": "openai/gpt-oss-120b"}
    local = make_settings(database_url="postgresql://g:g@localhost:5433/grounded", **models)
    remote = make_settings(database_url=f"postgresql://app:{PASSWORD}@{NEON}/neondb", **models)

    assert demo_settings(local).allow_direct_api  # passes, and is the copy the demo runs on
    with pytest.raises(DemoRefusedError, match="local database") as caught:
        demo_settings(remote)
    assert NEON in str(caught.value)
    assert PASSWORD not in str(caught.value)


def test_a_remote_database_is_refused_even_with_a_dev_app_env_and_no_models_set() -> None:
    # The database is checked before the model names, so a dev shell pointed at Neon is told about
    # the database first, not asked for variables it would then use against it.
    settings = make_settings(
        app_env="dev", database_url=f"postgresql://app:{PASSWORD}@{NEON}/neondb"
    )
    with pytest.raises(DemoRefusedError, match="local database"):
        demo_settings(settings)
