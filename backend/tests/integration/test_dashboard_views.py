"""The dashboard views of migration 0005 (ticket 8.01, DB.md §8), on the local pgvector DB only.

What is pinned: the exact column set of each view (none of them may expose ``question``, its hash,
``ip_hash`` or the row id), the numbers on known rows (counts, ``percentile_cont`` percentiles,
sums), that ``source = 'eval'`` rows and, for stage latency, cache hits and rows older than 7 days
are left out, that a day is a UTC day whatever the session ``TimeZone``, and the guarded grant:
without an ``app`` role the migration applies and grants nothing, with one (created here, dropped
afterwards) ``app`` can read the views and still cannot reach a private column through them.

The content tests run in one transaction that starts by deleting every ``request_logs`` row and is
rolled back at the end, so rows other tests committed cannot change an aggregate. The grant tests
use databases of their own and the cluster-wide role ``app``, under an advisory lock on the
maintenance database so that two pytest sessions never race on the role. A role ``app`` these tests
did not create is never dropped: the test fails and says how to remove it.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg.rows import dict_row

from grounded.infra.migrations import DEFAULT_MIGRATIONS_DIR, migrate

pytestmark = pytest.mark.integration

VIEWS = ["v_request_stats_daily", "v_stage_latency_7d", "v_request_metrics"]
PRIVATE_COLUMNS = ["question", "question_hash", "ip_hash", "id"]

EXPECTED_COLUMNS = {
    "v_request_stats_daily": [
        "day",
        "requests",
        "cache_hits",
        "fallbacks",
        "validation_retries",
        "validation_failures",
        "refusals",
        "throttled",
        "p50_ms",
        "p95_ms",
        "shadow_cost_usd",
        "input_tokens",
        "output_tokens",
    ],
    "v_stage_latency_7d": [
        "embed_p50",
        "embed_p95",
        "retrieval_p50",
        "retrieval_p95",
        "rerank_p50",
        "rerank_p95",
        "llm_p50",
        "llm_p95",
    ],
    "v_request_metrics": [
        "created_at",
        "outcome",
        "http_status",
        "cache_hit",
        "fallback_used",
        "validation_retries",
        "provider",
        "model",
        "latency_total_ms",
        "latency_embed_ms",
        "latency_retrieval_ms",
        "latency_rerank_ms",
        "latency_llm_ms",
        "input_tokens",
        "output_tokens",
        "shadow_cost_usd",
    ],
}

_INSERT = """
INSERT INTO request_logs (
    created_at, source, question, question_hash, ip_hash, http_status, outcome, cache_hit,
    fallback_used, validation_retries, provider, model, latency_total_ms, latency_embed_ms,
    latency_retrieval_ms, latency_rerank_ms, latency_llm_ms, input_tokens, output_tokens,
    shadow_cost_usd)
VALUES (
    %(created_at)s, %(source)s, 'secret question', 'secret-hash', 'secret-ip', %(http_status)s,
    %(outcome)s, %(cache_hit)s, %(fallback_used)s, %(validation_retries)s, %(provider)s, %(model)s,
    %(latency_total_ms)s, %(latency_embed_ms)s, %(latency_retrieval_ms)s, %(latency_rerank_ms)s,
    %(latency_llm_ms)s, %(input_tokens)s, %(output_tokens)s, %(shadow_cost_usd)s)
"""

NOON = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)


def insert(conn: psycopg.Connection[Any], **overrides: Any) -> None:
    row: dict[str, Any] = {
        "created_at": NOON,
        "source": "api",
        "http_status": 200,
        "outcome": "answered",
        "cache_hit": False,
        "fallback_used": False,
        "validation_retries": 0,
        "provider": "gemini",
        "model": "gemini-test",
        "latency_total_ms": 100,
        "latency_embed_ms": None,
        "latency_retrieval_ms": None,
        "latency_rerank_ms": None,
        "latency_llm_ms": None,
        "input_tokens": None,
        "output_tokens": None,
        "shadow_cost_usd": None,
    }
    unknown = set(overrides) - set(row)
    assert not unknown, unknown
    row.update(overrides)
    conn.execute(_INSERT, row)


@pytest.fixture
def conn(test_database_url: str) -> Iterator[psycopg.Connection[Any]]:
    """One transaction, rolled back: it starts with an empty ``request_logs``."""
    with psycopg.connect(test_database_url) as connection:
        connection.execute("DELETE FROM request_logs")
        yield connection
        connection.rollback()


def rows(conn: psycopg.Connection[Any], query: str, params: Any = None) -> list[dict[str, Any]]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, params)  # pyright: ignore[reportArgumentType]
        return cur.fetchall()


# --- shape ----------------------------------------------------------------------------------------


def test_the_public_schema_has_exactly_these_views_with_exactly_these_columns(
    conn: psycopg.Connection[Any],
) -> None:
    views = rows(
        conn, "SELECT table_name FROM information_schema.views WHERE table_schema = 'public'"
    )
    columns = rows(
        conn,
        "SELECT table_name, column_name, data_type FROM information_schema.columns"
        " WHERE table_schema = 'public' AND table_name = ANY(%s)"
        " ORDER BY table_name, ordinal_position",
        (VIEWS,),
    )

    assert sorted(v["table_name"] for v in views) == sorted(VIEWS)
    found: dict[str, list[str]] = {}
    for column in columns:
        found.setdefault(column["table_name"], []).append(column["column_name"])
    assert found == EXPECTED_COLUMNS
    for names in found.values():
        assert not set(names) & set(PRIVATE_COLUMNS)
    # A UTC calendar date, not a timestamp a client could shift into its own time zone.
    (day,) = [c for c in columns if c["column_name"] == "day"]
    assert day["data_type"] == "date"


# --- v_request_stats_daily ------------------------------------------------------------------------


def test_daily_stats_count_and_percentiles_on_known_rows(conn: psycopg.Connection[Any]) -> None:
    insert(
        conn,
        latency_total_ms=100,
        input_tokens=10,
        output_tokens=5,
        shadow_cost_usd=Decimal("0.001"),
    )
    insert(conn, latency_total_ms=200, source="web", cache_hit=True, provider=None, model=None)
    insert(
        conn,
        latency_total_ms=300,
        fallback_used=True,
        validation_retries=1,
        provider="groq",
        input_tokens=20,
        output_tokens=10,
        shadow_cost_usd=Decimal("0.002"),
    )
    insert(
        conn,
        latency_total_ms=400,
        source="web",
        outcome="validation_failed",
        http_status=502,
        fallback_used=True,
        validation_retries=2,
    )
    insert(conn, latency_total_ms=500, outcome="insufficient_context")
    insert(conn, latency_total_ms=600, outcome="rate_limited", http_status=429, provider=None)
    insert(conn, latency_total_ms=700, outcome="budget_exhausted", http_status=429, provider=None)
    # CI's eval rows never count, however extreme.
    insert(
        conn,
        source="eval",
        latency_total_ms=100_000,
        outcome="validation_failed",
        cache_hit=True,
        fallback_used=True,
        validation_retries=3,
        input_tokens=1_000,
        shadow_cost_usd=Decimal(9),
    )
    insert(conn, created_at=NOON + timedelta(days=1), latency_total_ms=50)

    days = rows(conn, "SELECT * FROM v_request_stats_daily ORDER BY day")

    assert [d["day"] for d in days] == [date(2026, 10, 20), date(2026, 10, 21)]
    first = days[0]
    assert {k: v for k, v in first.items() if k not in {"p50_ms", "p95_ms"}} == {
        "day": date(2026, 10, 20),
        "requests": 7,
        "cache_hits": 1,
        "fallbacks": 2,
        "validation_retries": 2,  # requests with at least one retry, not the sum (3)
        "validation_failures": 1,
        "refusals": 1,
        "throttled": 2,
        "shadow_cost_usd": Decimal("0.003"),
        "input_tokens": 30,
        "output_tokens": 15,
    }
    # percentile_cont interpolates: 100..700 has its median at 400 and its p95 at
    # 600 + 0.7 * (700 - 600), position 0.95 * (7 - 1) = 5.7.
    assert first["p50_ms"] == pytest.approx(400)
    assert first["p95_ms"] == pytest.approx(670)
    assert days[1]["requests"] == 1
    assert days[1]["p50_ms"] == pytest.approx(50)


@pytest.mark.parametrize(
    ("time_zone", "local_days"),
    [("Asia/Tokyo", 2), ("America/Los_Angeles", 2), ("UTC", 1)],
)
def test_a_day_is_the_utc_day_whatever_the_session_time_zone(
    conn: psycopg.Connection[Any], time_zone: str, local_days: int
) -> None:
    conn.execute("SELECT set_config('TimeZone', %s, true)", (time_zone,))
    insert(conn, created_at=datetime(2026, 10, 20, 0, 30, tzinfo=UTC))
    insert(conn, created_at=datetime(2026, 10, 20, 23, 30, tzinfo=UTC))

    # The session's own calendar splits the two rows in Tokyo (+9) and Los Angeles (-7)...
    local = rows(conn, "SELECT count(DISTINCT created_at::date) AS n FROM request_logs")
    assert local == [{"n": local_days}]
    # ...the view does not: both are on 2026-10-20 in UTC.
    assert rows(conn, "SELECT day, requests FROM v_request_stats_daily") == [
        {"day": date(2026, 10, 20), "requests": 2}
    ]


# --- v_stage_latency_7d ---------------------------------------------------------------------------


def test_stage_latency_leaves_out_cache_hits_eval_rows_and_rows_older_than_7_days(
    conn: psycopg.Connection[Any],
) -> None:
    recent = datetime.now(UTC) - timedelta(hours=1)
    for embed, retrieval, rerank, llm in [
        (10, 1, None, 100),
        (20, 2, 50, 200),
        (30, 3, None, 300),
        (40, 4, 70, 400),
    ]:
        insert(
            conn,
            created_at=recent,
            latency_embed_ms=embed,
            latency_retrieval_ms=retrieval,
            latency_rerank_ms=rerank,
            latency_llm_ms=llm,
        )
    # A real cache hit has no stage timings at all; these carry huge ones so that letting any of
    # them in would move every percentile.
    huge = {
        "latency_embed_ms": 9_999,
        "latency_retrieval_ms": 9_999,
        "latency_rerank_ms": 9_999,
        "latency_llm_ms": 9_999,
    }
    insert(conn, created_at=recent, source="web", cache_hit=True, **huge)
    insert(conn, created_at=recent, source="eval", **huge)
    insert(conn, created_at=datetime.now(UTC) - timedelta(days=8), **huge)

    assert rows(conn, "SELECT * FROM v_stage_latency_7d") == [
        {
            "embed_p50": pytest.approx(25),
            "embed_p95": pytest.approx(38.5),  # 30 + 0.85 * 10, position 0.95 * 3 = 2.85
            "retrieval_p50": pytest.approx(2.5),
            "retrieval_p95": pytest.approx(3.85),
            "rerank_p50": pytest.approx(60),  # NULLs (rerank off) are ignored: 50 and 70
            "rerank_p95": pytest.approx(69),
            "llm_p50": pytest.approx(250),
            "llm_p95": pytest.approx(385),
        }
    ]


def test_stage_latency_is_one_row_of_nulls_when_there_is_nothing_to_measure(
    conn: psycopg.Connection[Any],
) -> None:
    insert(conn, cache_hit=True, created_at=datetime.now(UTC), latency_llm_ms=5)

    assert rows(conn, "SELECT llm_p50, llm_p95 FROM v_stage_latency_7d") == [
        {"llm_p50": None, "llm_p95": None}
    ]


# --- v_request_metrics ----------------------------------------------------------------------------


def test_request_metrics_has_one_row_per_user_request_and_no_eval_rows(
    conn: psycopg.Connection[Any],
) -> None:
    insert(
        conn,
        latency_total_ms=1_200,
        latency_embed_ms=30,
        latency_retrieval_ms=40,
        latency_llm_ms=1_000,
        input_tokens=900,
        output_tokens=120,
        shadow_cost_usd=Decimal("0.00012345"),
    )
    insert(conn, source="web", created_at=NOON + timedelta(minutes=1), cache_hit=True)
    insert(conn, source="eval", created_at=NOON + timedelta(minutes=2))

    found = rows(conn, "SELECT * FROM v_request_metrics ORDER BY created_at")

    assert found == [
        {
            "created_at": NOON,
            "outcome": "answered",
            "http_status": 200,
            "cache_hit": False,
            "fallback_used": False,
            "validation_retries": 0,
            "provider": "gemini",
            "model": "gemini-test",
            "latency_total_ms": 1_200,
            "latency_embed_ms": 30,
            "latency_retrieval_ms": 40,
            "latency_rerank_ms": None,
            "latency_llm_ms": 1_000,
            "input_tokens": 900,
            "output_tokens": 120,
            "shadow_cost_usd": Decimal("0.00012345"),
        },
        {
            "created_at": NOON + timedelta(minutes=1),
            "outcome": "answered",
            "http_status": 200,
            "cache_hit": True,
            "fallback_used": False,
            "validation_retries": 0,
            "provider": "gemini",
            "model": "gemini-test",
            "latency_total_ms": 100,
            "latency_embed_ms": None,
            "latency_retrieval_ms": None,
            "latency_rerank_ms": None,
            "latency_llm_ms": None,
            "input_tokens": None,
            "output_tokens": None,
            "shadow_cost_usd": None,
        },
    ]


def test_a_window_aggregate_over_request_metrics_takes_the_window_as_a_bound_parameter(
    conn: psycopg.Connection[Any],
) -> None:
    # The shape of the 8.02 query: a window in days, bound; stage percentiles without cache hits.
    now = datetime.now(UTC)
    for hours, llm in [(1, 100), (2, 200), (3, 300)]:
        insert(conn, created_at=now - timedelta(hours=hours), latency_llm_ms=llm)
    insert(conn, created_at=now - timedelta(hours=1), cache_hit=True, latency_llm_ms=9_999)
    insert(conn, created_at=now - timedelta(days=2), latency_llm_ms=9_999)

    query = """
        SELECT count(*) AS requests,
               count(*) FILTER (WHERE cache_hit) AS cache_hits,
               percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_llm_ms)
                   FILTER (WHERE NOT cache_hit) AS llm_p50
        FROM v_request_metrics
        WHERE created_at > now() - make_interval(days => %s)
    """

    assert rows(conn, query, (1,)) == [
        {"requests": 4, "cache_hits": 1, "llm_p50": pytest.approx(200)}
    ]
    assert rows(conn, query, (7,))[0]["requests"] == 5


# --- the guarded grant ----------------------------------------------------------------------------

# The role `app` is cluster-wide, so two pytest sessions with different TEST_DATABASE_URLs (which do
# not wait for each other, tests/db_lock.py) would race on it. Both grant tests take this session
# advisory lock on the maintenance database first; the server releases it when the connection
# closes, also when the process is killed.
_ROLE_LOCK_KEY = "grounded-tests-cluster-role-app"
_ROLE_LOCK_TIMEOUT = "10min"
# Marks a role these tests created, so a leftover of an interrupted run can be told from a role a
# developer made (which is never dropped here).
_ROLE_MARKER = "created by tests/integration/test_dashboard_views.py"


def _admin_url(test_database_url: str) -> str:
    return make_conninfo(test_database_url, dbname="postgres")


def _scratch_name(test_database_url: str, tag: str) -> str:
    dbname = str(conninfo_to_dict(test_database_url)["dbname"])
    return f"{dbname.removesuffix('_test')}_views_{tag}_test"


def _drop_database(admin: psycopg.Connection[Any], name: str) -> None:
    admin.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name)))


def _drop_marked_app_role(admin: psycopg.Connection[Any], test_database_url: str) -> None:
    """Drop this file's ``app``, first its scratch database, then its privileges anywhere else.

    A run interrupted while the role existed can leave grants behind in other local databases too:
    the next session's ``grounded_test`` is migrated with the role present, so ``0005`` grants it
    there. ``DROP OWNED BY`` works per database, so it runs in each one ``pg_shdepend`` names.
    """
    _drop_database(admin, _scratch_name(test_database_url, "app"))
    databases = admin.execute(
        "SELECT DISTINCT d.datname FROM pg_shdepend s JOIN pg_database d ON d.oid = s.dbid"
        " WHERE s.refclassid = 'pg_authid'::regclass"
        " AND s.refobjid = (SELECT oid FROM pg_roles WHERE rolname = 'app')"
        " ORDER BY d.datname"
    ).fetchall()
    for (database,) in databases:
        url = make_conninfo(test_database_url, dbname=database)
        with psycopg.connect(url, autocommit=True) as conn:
            conn.execute("DROP OWNED BY app")
    admin.execute("DROP ROLE app")


@pytest.fixture
def no_app_role(test_database_url: str) -> Iterator[psycopg.Connection[Any]]:
    """Hold the cluster lock for the test and make sure no role ``app`` exists.

    A role ``app`` carrying this file's marker is the leftover of an interrupted run (nobody else
    can be using it while the lock is held): it is dropped with its privileges and its scratch
    database. Any other ``app`` fails the test with the commands to remove it, and is left alone.
    """
    with psycopg.connect(_admin_url(test_database_url), autocommit=True) as admin:
        admin.execute(sql.SQL("SET lock_timeout = {}").format(sql.Literal(_ROLE_LOCK_TIMEOUT)))
        admin.execute("SELECT pg_advisory_lock(hashtext(%s))", (_ROLE_LOCK_KEY,))
        row = admin.execute(
            "SELECT shobj_description(oid, 'pg_authid') FROM pg_roles WHERE rolname = 'app'"
        ).fetchone()
        if row is not None and row[0] == _ROLE_MARKER:
            _drop_marked_app_role(admin, test_database_url)
        elif row is not None:
            scratch = _scratch_name(test_database_url, "app")
            pytest.fail(
                "a role named app already exists in the local cluster, and these tests did not "
                "create it. If it is a leftover, remove it as a superuser and run again: "
                f"DROP DATABASE IF EXISTS {scratch}; DROP OWNED BY app; DROP ROLE app; "
                "(DROP OWNED BY app in every database where it holds privileges, DB.md §5)."
            )
        yield admin


@contextmanager
def scratch_database(test_database_url: str, tag: str) -> Iterator[str]:
    """An empty database of its own, dropped afterwards (a role cannot be dropped while it holds
    privileges in a database, so this closes before the role does)."""
    name = _scratch_name(test_database_url, tag)
    admin_url = _admin_url(test_database_url)
    with psycopg.connect(admin_url, autocommit=True) as admin:
        _drop_database(admin, name)
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    try:
        yield make_conninfo(test_database_url, dbname=name)
    finally:
        with psycopg.connect(admin_url, autocommit=True) as admin:
            _drop_database(admin, name)


@contextmanager
def app_role(admin: psycopg.Connection[Any], test_database_url: str) -> Iterator[None]:
    """Create ``app`` (marked as this file's), and drop it with its scratch database afterwards."""
    admin.execute("CREATE ROLE app")
    admin.execute(sql.SQL("COMMENT ON ROLE app IS {}").format(sql.Literal(_ROLE_MARKER)))
    try:
        yield
    finally:
        _drop_marked_app_role(admin, test_database_url)


def test_without_an_app_role_the_migration_applies_on_a_fresh_database_and_grants_nothing(
    test_database_url: str, no_app_role: psycopg.Connection[Any]
) -> None:
    with scratch_database(test_database_url, "noapp") as url:
        assert migrate(url).applied[-1] == "0005_dashboard_views"

        with psycopg.connect(url) as conn:
            acl = rows(
                conn,
                "SELECT relname, relacl FROM pg_class WHERE relname = ANY(%s) ORDER BY relname",
                (VIEWS,),
            )
    # NULL: the owner's default privileges only, nobody was granted anything.
    assert acl == [{"relname": view, "relacl": None} for view in sorted(VIEWS)]


def test_with_an_app_role_the_migration_from_0004_grants_select_on_the_views_only(
    test_database_url: str, no_app_role: psycopg.Connection[Any], tmp_path: Path
) -> None:
    up_to_0004 = tmp_path / "migrations"
    up_to_0004.mkdir()
    for path in sorted(DEFAULT_MIGRATIONS_DIR.glob("*.sql")):
        if path.name < "0005":
            shutil.copy(path, up_to_0004)

    with (
        app_role(no_app_role, test_database_url),
        scratch_database(test_database_url, "app") as url,
    ):
        assert migrate(url, up_to_0004).applied[-1] == "0004_deadline_outcome"
        with psycopg.connect(url) as conn:
            insert(conn)  # a database at the previous version that already holds rows
            insert(conn, source="eval")
            conn.commit()

        assert migrate(url).applied == ["0005_dashboard_views"]

        with psycopg.connect(url) as conn:
            for view in VIEWS:
                granted = conn.execute(
                    "SELECT has_table_privilege('app', %s, 'SELECT')", (view,)
                ).fetchone()
                assert granted == (True,), view
            # Only SELECT, and nothing on the table under the views (in this database).
            assert conn.execute(
                "SELECT has_table_privilege('app', 'v_request_metrics', 'INSERT'),"
                " has_table_privilege('app', 'request_logs', 'SELECT')"
            ).fetchone() == (False, False)

            conn.execute("GRANT USAGE ON SCHEMA public TO app")  # DB.md §5 gives it too
            conn.execute("SET ROLE app")
            # A view reads its table with its owner's rights, so app sees the rows through it...
            assert rows(conn, "SELECT count(*) AS n FROM v_request_metrics") == [{"n": 1}]
            assert rows(conn, "SELECT requests FROM v_request_stats_daily") == [{"requests": 1}]
            assert len(rows(conn, "SELECT * FROM v_stage_latency_7d")) == 1
            # ...and only the columns the view selects: no private column is reachable.
            for view in VIEWS:
                for column in PRIVATE_COLUMNS:
                    with pytest.raises(psycopg.errors.UndefinedColumn), conn.transaction():
                        conn.execute(
                            sql.SQL("SELECT {} FROM {}").format(
                                sql.Identifier(column), sql.Identifier(view)
                            )
                        )
            with pytest.raises(psycopg.errors.InsufficientPrivilege), conn.transaction():
                conn.execute("SELECT question FROM request_logs")
            conn.rollback()
