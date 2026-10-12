"""Retention against the real schema (ticket 5.11, DB.md §9), on the local pgvector database only.

Every table gets rows of each age on both sides of each rule's boundary, so the tests prove which
rows a rule touches and that it leaves the others alone. Ages are written as hours around the
boundary (an hour either side), never as the exact boundary: the rows are inserted a few
milliseconds before ``now()`` of the run, and the exact instant would be a coin toss. The one
exact boundary is the usage rule's, which works in whole UTC days.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import psycopg
import pytest
from typer.testing import CliRunner

import grounded.cli
from grounded.cli import app
from grounded.infra import housekeeping
from grounded.infra.housekeeping import RULES, Rule, RuleResult, run_housekeeping
from tests.support import make_settings

pytestmark = pytest.mark.integration

runner = CliRunner()
DAY = 24

TABLES = "request_logs, answer_cache, daily_usage, rate_limit_counters, index_versions"


@pytest.fixture
def db(test_database_url: str) -> Iterator[str]:
    def wipe() -> None:
        with psycopg.connect(test_database_url, autocommit=True) as conn:
            conn.execute(f"TRUNCATE {TABLES} RESTART IDENTITY CASCADE")  # pyright: ignore[reportArgumentType]

    wipe()
    yield test_database_url
    wipe()


def counts(results: list[RuleResult]) -> dict[str, int]:
    return {r.name: r.rows for r in results}


# (label, age in hours, question, ip_hash). The label travels in question_hash.
REQUEST_LOGS: list[tuple[str, int, str | None, str | None]] = [
    ("fresh", 1, "q", "ip"),
    ("29d_text", 30 * DAY - 1, "q", "ip"),  # one hour inside the 30 days: untouched
    ("31d_text", 30 * DAY + 1, "q", "ip"),  # one hour outside: text removed, row kept
    ("31d_question_only", 30 * DAY + 1, "q", None),
    ("31d_ip_only", 30 * DAY + 1, None, "ip"),
    ("31d_clean", 30 * DAY + 1, None, None),  # nothing to remove: not counted
    ("179d_text", 180 * DAY - 1, "q", "ip"),  # inside the 180 days: text removed, row kept
    ("179d_clean", 180 * DAY - 1, None, None),
    ("181d_clean", 180 * DAY + 1, None, None),  # outside: deleted
    ("181d_text", 180 * DAY + 1, "q", "ip"),  # first stripped, then deleted: both counts include it
]
# After the first run: the rows that remain, with whether question / ip_hash are still set.
LOGS_AFTER = {
    "fresh": (True, True),
    "29d_text": (True, True),
    "31d_text": (False, False),
    "31d_question_only": (False, False),
    "31d_ip_only": (False, False),
    "31d_clean": (False, False),
    "179d_text": (False, False),
    "179d_clean": (False, False),
}

# (cache_key, hours until expiry; negative = expired)
CACHE = [("live_1h", 1.0), ("live_1min", 0.02), ("expired_1min", -0.02), ("expired_1h", -1.0)]
CACHE_AFTER = {"live_1h", "live_1min"}

# (label, days before today in UTC). The rule is `day < today - 90`, so day -90 itself is kept.
USAGE = [("today", 0), ("d89", 89), ("d90", 90), ("d91", 91), ("d400", 400)]
USAGE_AFTER = {0, 89, 90}

# (ip_hash, window_kind, age in hours of window_start)
WINDOWS = [
    ("a", "minute", 1),
    ("a", "day", 47),  # the day window is the longest: still needed within two days
    ("b", "day", 49),
    ("c", "minute", 49),
    ("d", "day", 24 * 30),
]
WINDOWS_AFTER = {("a", "minute"), ("a", "day")}

EXPECTED_FIRST_RUN = {
    # 31d_text, 31d_question_only, 31d_ip_only, 179d_text, 181d_text; clean rows are not counted.
    "request_logs_text_removed": 5,
    "request_logs_deleted": 2,  # 181d_clean, 181d_text
    "answer_cache_expired_deleted": 2,
    "daily_usage_deleted": 2,  # d91, d400
    "rate_limit_counters_deleted": 3,  # b/day 49h, c/minute 49h, d/day 30 days
}
NOTHING = dict.fromkeys(EXPECTED_FIRST_RUN, 0)


def seed(db: str) -> None:
    with psycopg.connect(db) as conn:
        conn.execute(
            "INSERT INTO index_versions (git_ref, git_sha, embedding_model, embedding_dim, "
            "chunking_config, config_hash) VALUES ('0.0.1', repeat('a', 40), 'm', 8, '{}', 'h')"
        )
        for label, hours, question, ip_hash in REQUEST_LOGS:
            conn.execute(
                "INSERT INTO request_logs (created_at, source, question, question_hash, ip_hash, "
                "http_status, outcome, latency_total_ms) VALUES "
                "(now() - make_interval(hours => %s), 'web', %s, %s, %s, 200, 'answered', 10)",
                (hours, question, label, ip_hash),
            )
        for key, hours in CACHE:
            conn.execute(
                "INSERT INTO answer_cache (cache_key, normalized_question, response, "
                "prompt_version, index_version_id, retrieval_config_hash, generator_model, "
                "expires_at) SELECT %s, 'q', '{}', 'p', id, 'r', 'g', "
                "now() + make_interval(secs => %s) FROM index_versions",
                (key, hours * 3600),
            )
        for _, days in USAGE:
            conn.execute(
                "INSERT INTO daily_usage (day) VALUES ((now() AT TIME ZONE 'utc')::date - %s)",
                (days,),
            )
        for ip_hash, kind, hours in WINDOWS:
            conn.execute(
                "INSERT INTO rate_limit_counters (ip_hash, window_kind, window_start, hits) "
                "VALUES (%s, %s, now() - make_interval(hours => %s), 1)",
                (ip_hash, kind, hours),
            )


def snapshot(db: str) -> dict[str, Any]:
    with psycopg.connect(db) as conn:
        logs = conn.execute(
            "SELECT question_hash, question IS NOT NULL, ip_hash IS NOT NULL FROM request_logs"
        ).fetchall()
        cache = conn.execute("SELECT cache_key FROM answer_cache").fetchall()
        usage = conn.execute("SELECT (now() AT TIME ZONE 'utc')::date - day FROM daily_usage")
        windows = conn.execute("SELECT ip_hash, window_kind FROM rate_limit_counters")
        return {
            "logs": {label: (q, ip) for label, q, ip in logs},
            "cache": {key for (key,) in cache},
            "usage": {days for (days,) in usage.fetchall()},
            "windows": set(windows.fetchall()),
        }


SEEDED = {
    "logs": {label: (q is not None, ip is not None) for label, _, q, ip in REQUEST_LOGS},
    "cache": {key for key, _ in CACHE},
    "usage": {days for _, days in USAGE},
    "windows": {(ip, kind) for ip, kind, _ in WINDOWS},
}
AFTER = {
    "logs": LOGS_AFTER,
    "cache": CACHE_AFTER,
    "usage": USAGE_AFTER,
    "windows": WINDOWS_AFTER,
}


def test_the_seed_holds_every_age_before_the_run(db: str) -> None:
    seed(db)

    assert snapshot(db) == SEEDED


def test_each_rule_touches_exactly_the_right_rows(db: str) -> None:
    seed(db)

    results = run_housekeeping(db)

    assert counts(results) == EXPECTED_FIRST_RUN
    assert snapshot(db) == AFTER  # text and hashes gone at 30 days, rows at 180, the rest intact


def test_the_results_follow_the_order_of_the_rules(db: str) -> None:
    results = run_housekeeping(db)

    assert [r.name for r in results] == [rule.name for rule in RULES]
    assert counts(results) == NOTHING  # empty tables: every rule counts 0


def test_a_second_run_affects_nothing(db: str) -> None:
    seed(db)
    run_housekeeping(db)
    first = snapshot(db)

    again = run_housekeeping(db)

    assert counts(again) == NOTHING
    assert snapshot(db) == first


def test_a_dry_run_reports_the_real_counts_and_changes_nothing(db: str) -> None:
    seed(db)

    dry = run_housekeeping(db, dry_run=True)

    assert counts(dry) == EXPECTED_FIRST_RUN
    assert snapshot(db) == SEEDED
    assert counts(run_housekeeping(db)) == EXPECTED_FIRST_RUN  # and the real run agrees


def test_a_failing_rule_rolls_every_rule_back(db: str, monkeypatch: pytest.MonkeyPatch) -> None:
    seed(db)
    broken = Rule("broken", "DELETE FROM no_such_table", {})
    monkeypatch.setattr(housekeeping, "RULES", (*RULES, broken))

    with pytest.raises(psycopg.errors.UndefinedTable):
        run_housekeeping(db)

    assert snapshot(db) == SEEDED  # the five good rules before it were not committed


def test_telemetry_that_is_not_user_text_survives_the_text_removal(db: str) -> None:
    seed(db)
    run_housekeeping(db)

    with psycopg.connect(db) as conn:
        row = conn.execute(
            "SELECT http_status, outcome, source, latency_total_ms FROM request_logs "
            "WHERE question_hash = '31d_text'"
        ).fetchone()

    assert row == (200, "answered", "web", 10)  # only question and ip_hash are removed


# --- the command ---------------------------------------------------------------------------------


@pytest.fixture
def cli(db: str, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = make_settings(database_url=db, database_url_direct=db)
    monkeypatch.setattr(grounded.cli, "get_settings", lambda: settings)


def lines(output: str) -> list[str]:
    return output.strip().splitlines()


@pytest.mark.usefixtures("cli")
def test_the_command_prints_one_stable_line_per_rule_and_a_total(db: str) -> None:
    seed(db)

    result = runner.invoke(app, ["housekeeping"])

    assert result.exit_code == 0, result.output
    assert lines(result.output)[1:] == [
        "  request_logs_text_removed     5",
        "  request_logs_deleted          2",
        "  answer_cache_expired_deleted  2",
        "  daily_usage_deleted           2",
        "  rate_limit_counters_deleted   3",
        "Done: 14 rows affected.",
    ]
    assert snapshot(db) == AFTER


@pytest.mark.usefixtures("cli")
def test_the_command_names_the_target_without_credentials(db: str) -> None:
    result = runner.invoke(app, ["housekeeping"])

    first = lines(result.output)[0]
    assert first.startswith("Housekeeping on ")
    assert "@" not in first
    assert "password" not in result.output.lower()


@pytest.mark.usefixtures("cli")
def test_the_command_dry_run_prints_counts_and_changes_nothing(db: str) -> None:
    seed(db)

    result = runner.invoke(app, ["housekeeping", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert lines(result.output)[0].endswith("(dry run)")
    assert "  request_logs_text_removed     5" in result.output
    assert lines(result.output)[-1] == "Dry run: 14 rows would be affected; nothing was changed."
    assert snapshot(db) == SEEDED


@pytest.mark.usefixtures("cli")
def test_the_command_run_twice_reports_zero_the_second_time(db: str) -> None:
    seed(db)
    runner.invoke(app, ["housekeeping"])

    result = runner.invoke(app, ["housekeeping"])

    assert result.exit_code == 0
    assert lines(result.output)[-1] == "Done: 0 rows affected."
