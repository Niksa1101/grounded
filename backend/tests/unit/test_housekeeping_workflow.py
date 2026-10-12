"""``housekeeping.yml`` and its keep-warm script (ticket 5.11).

Retention is a write to production Neon with the owner connection, so the workflow ships with its
schedule off: the daily run does nothing until the Author sets the repository variable
``HOUSEKEEPING_ENABLED`` to ``true`` (after one green manual run), the same pattern as
``EVAL_RECORD_RUNS`` in ``eval.yml``. What these tests pin is that gate, the two steps that see the
secret (the guard, kept identical to ``ingest.yml``'s, and the command), that no input reaches a
script inline, and the behavior of the keep-warm script against a local HTTP server. The helpers
that read the YAML as text come from ``test_ci_workflows``; the list of steps that may see the
secret is checked there with the other workflows'.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from tests.unit.test_ci_workflows import (
    ROOT,
    SECRET,
    code,
    run_blocks,
    step_named,
    steps,
)

HOUSEKEEPING = "housekeeping.yml"
KEEP_WARM = ROOT / ".github" / "scripts" / "keep-warm.sh"


def job_conditions(name: str) -> dict[str, str]:
    """Each job's own ``if:`` (a folded scalar at the indentation of a job's keys), by job id."""
    jobs = code(name).split("\njobs:\n", 1)[1]
    found: dict[str, str] = {}
    for block in re.split(r"(?m)^(?=  [\w-]+:$)", jobs):
        job = re.match(r"  ([\w-]+):$", block, re.MULTILINE)
        match = re.search(r"(?m)^    if: >-\n((?:      .*\n)+)", block)
        if job and match:
            found[job.group(1)] = " ".join(match.group(1).split())
    return found


def test_it_runs_daily_and_by_hand_and_never_from_a_pr_or_a_push() -> None:
    triggers = code(HOUSEKEEPING).split("\njobs:\n", 1)[0]

    assert re.findall(r'(?m)^    - cron: "(.+)"', triggers) == ["23 3 * * *"]  # once a day
    assert re.search(r"(?m)^  workflow_dispatch:$", triggers)
    for other in ("pull_request", "push", "workflow_run", "workflow_call"):
        assert not re.search(rf"(?m)^  {other}:", triggers)


def test_the_schedule_touches_production_only_when_the_variable_is_true() -> None:
    conditions = job_conditions(HOUSEKEEPING)

    # Both jobs reach production (the database, the live backend) and have the very same gate.
    assert set(conditions) == {"retention", "keep-warm"}
    assert len(set(conditions.values())) == 1
    clauses = [c.strip() for c in conditions["retention"].split("||")]
    assert clauses == [
        # By hand: from main only, always (the Author's first run, with or without the variable).
        "(github.event_name == 'workflow_dispatch' && github.ref == 'refs/heads/main')",
        # By the clock: only once HOUSEKEEPING_ENABLED is 'true'. Unset it is '', so it is skipped.
        "(github.event_name == 'schedule' && vars.HOUSEKEEPING_ENABLED == 'true')",
    ]


def test_it_has_read_only_permissions_and_no_environment_to_wait_for() -> None:
    text = code(HOUSEKEEPING)

    assert re.search(r"(?m)^permissions:\n  contents: read$", text)
    assert "write" not in text.split("\njobs:\n", 1)[0]
    # ingest.yml's environment can hold a required reviewer; a daily unattended run must not wait.
    assert not re.search(r"(?m)^\s+environment:", text)


def test_retention_never_overlaps_itself() -> None:
    retention = code(HOUSEKEEPING).split("\n  keep-warm:\n", 1)[0]

    assert (
        "    concurrency:\n      group: housekeeping-neon-production\n"
        "      cancel-in-progress: false"
    ) in retention


def test_the_guard_is_the_one_of_ingest_yml() -> None:
    title = "Guard the production connection"

    assert step_named(HOUSEKEEPING, title) == step_named("ingest.yml", title)


def test_the_database_secret_is_the_only_secret_and_keep_warm_sees_none() -> None:
    text = code(HOUSEKEEPING)

    assert set(re.findall(r"secrets\.([A-Z_]+)", text)) == {SECRET}
    assert text.count(f"secrets.{SECRET}") == 2  # the guard and the command, nothing else
    keep_warm = text.split("\n  keep-warm:\n", 1)[1]
    assert SECRET not in keep_warm  # a public GET needs no credential
    assert "secrets." not in keep_warm


def test_inputs_reach_scripts_only_through_env() -> None:
    for script in run_blocks(HOUSEKEEPING):
        assert "${{" not in script, script
    retention = step_named(HOUSEKEEPING, "Retention")
    assert "DRY_RUN: ${{ inputs.dry_run }}" in retention
    assert 'if [ "$DRY_RUN" = "true" ]; then' in retention  # the flag is conditional, never fixed
    assert re.findall(r"--dry-run", retention) == ["--dry-run"]
    assert "grounded housekeeping" in retention
    keep_warm = step_named(HOUSEKEEPING, "Wake the backend and the database")
    assert "BACKEND_URL: ${{ vars.BACKEND_URL }}" in keep_warm  # a variable: the URL is public
    assert "bash .github/scripts/keep-warm.sh" in keep_warm


def test_the_connection_is_checked_before_anything_is_installed_or_run() -> None:
    names = re.findall(r"(?m)^      - name: (.*)$", "".join(steps(HOUSEKEEPING)))

    assert names.index("Guard the production connection") < names.index("Install dependencies")
    assert names.index("Install dependencies") < names.index("Retention")


def test_it_never_prints_a_secret_and_pins_every_action() -> None:
    for script in run_blocks(HOUSEKEEPING):
        for line in script.splitlines():
            if re.search(r"\b(echo|printf|tee)\b", line):
                assert not re.search(r"\$\{?(DATABASE_URL_DIRECT|url)\b", line), line
    uses = re.findall(r"(?m)^\s+(?:- )?uses: (\S+)", code(HOUSEKEEPING))
    assert uses
    for action in uses:
        assert re.fullmatch(r".+@[0-9a-f]{40}", action), action


# --- the keep-warm script, against a local server (no real network: AGENTS.md §8) ----------------


def usable_bash() -> str | None:
    bash = shutil.which("bash")
    if bash is None or "system32" in bash.lower():  # Windows' WSL launcher cannot reach localhost
        return None
    probe = subprocess.run([bash, "-c", "command -v curl"], capture_output=True, check=False)
    return bash if probe.returncode == 0 else None


needs_bash = pytest.mark.skipif(usable_bash() is None, reason="needs bash and curl")


class ReadyzServer:
    """Answers GET requests with the given status codes in turn (the last one repeats)."""

    def __init__(self, codes: list[int]) -> None:
        outer = self
        self.requests: list[tuple[str, str]] = []
        self.codes = codes

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                outer.requests.append((self.path, self.headers.get("User-Agent", "")))
                index = min(len(outer.requests), len(outer.codes)) - 1
                self.send_response(outer.codes[index])
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, format: str, *args: object) -> None:
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def readyz(request: pytest.FixtureRequest) -> Iterator[ReadyzServer]:
    server = ReadyzServer(list(request.param))
    yield server
    server.close()


FAST = {"KEEP_WARM_PAUSE": "0", "KEEP_WARM_TIMEOUT": "5"}


def keep_warm(url: str | None, **env: str) -> subprocess.CompletedProcess[str]:
    bash = usable_bash()
    assert bash is not None
    variables = {k: v for k, v in os.environ.items() if not k.startswith(("BACKEND_", "KEEP_WARM"))}
    variables.update(env)
    if url is not None:
        variables["BACKEND_URL"] = url
    return subprocess.run(
        [bash, str(KEEP_WARM)],
        env=variables,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


@needs_bash
@pytest.mark.parametrize("readyz", [[200]], indirect=True)
def test_keep_warm_asks_readyz_once_when_it_answers(readyz: ReadyzServer) -> None:
    done = keep_warm(readyz.url + "/", **FAST)  # a trailing slash must not double up

    assert done.returncode == 0, done.stdout + done.stderr
    assert readyz.requests == [("/readyz", "grounded-keep-warm")]
    assert "attempt 1/4: HTTP 200" in done.stdout


@needs_bash
@pytest.mark.parametrize("readyz", [[503, 503, 200]], indirect=True)
def test_keep_warm_retries_a_cold_start_and_stops_at_the_first_200(readyz: ReadyzServer) -> None:
    done = keep_warm(readyz.url, **FAST)

    assert done.returncode == 0, done.stdout + done.stderr
    assert len(readyz.requests) == 3
    assert "attempt 2/4: HTTP 503" in done.stdout


@needs_bash
@pytest.mark.parametrize("readyz", [[503]], indirect=True)
def test_keep_warm_gives_up_after_the_configured_tries_and_fails_loudly(
    readyz: ReadyzServer,
) -> None:
    done = keep_warm(readyz.url, KEEP_WARM_ATTEMPTS="3", **FAST)

    assert done.returncode == 1
    assert len(readyz.requests) == 3  # bounded: never an open loop
    assert "::error title=Keep-warm failed" in done.stdout


@needs_bash
def test_keep_warm_fails_when_nothing_answers() -> None:
    done = keep_warm("http://127.0.0.1:1", KEEP_WARM_ATTEMPTS="2", **FAST)

    assert done.returncode == 1
    assert "attempt 2/2: HTTP 000" in done.stdout


@needs_bash
@pytest.mark.parametrize("url", [None, ""])
def test_keep_warm_skips_with_a_notice_when_the_variable_is_unset(url: str | None) -> None:
    done = keep_warm(url, **FAST)  # an unset repository variable is the empty string

    assert done.returncode == 0
    assert "::notice title=Keep-warm skipped" in done.stdout
    assert "attempt" not in done.stdout


@needs_bash
@pytest.mark.parametrize("url", ["http://example.com", "example.com", "ftp://example.com"])
def test_keep_warm_refuses_a_url_that_is_not_https(url: str) -> None:
    done = keep_warm(url, **FAST)

    assert done.returncode == 1
    assert "not an https URL" in done.stdout


def test_the_script_file_is_where_the_workflow_expects_it() -> None:
    assert KEEP_WARM.is_file()
    assert Path(".github/scripts/keep-warm.sh").as_posix() in code(HOUSEKEEPING)
