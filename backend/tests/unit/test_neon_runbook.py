"""The Neon production runbook (ticket 5.09) cannot drift from DB.md §5 or from ``ingest.yml``.

The runbook is prose the Author follows by hand, and the ``app`` role it creates is the only thing
between the public runtime and the index. The grants live in DB.md §5, the integration tests of the
rate limiter and the budget apply lines of that section to a throwaway role, and the runbook
applies them to Neon. These tests make the three say the same thing: the SQL block of the runbook
is the block of DB.md §5, the table of expected privileges is derived from it, and the secrets the
runbook asks for are the ones the workflow reads. No database is needed.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DB_MD = (ROOT / "docs" / "DB.md").read_text(encoding="utf-8")
RUNBOOK = (ROOT / "docs" / "runbooks" / "neon-production.md").read_text(encoding="utf-8")
INGEST = (ROOT / ".github" / "workflows" / "ingest.yml").read_text(encoding="utf-8")

PRIVILEGE_ORDER = ["SELECT", "INSERT", "UPDATE", "DELETE", "TRUNCATE"]


def section(text: str, heading: str) -> str:
    """The text from ``heading`` to the next heading of the same level."""
    level = heading.split(" ", 1)[0]
    start = text.index(heading)
    rest = text[start + len(heading) :]
    end = re.search(rf"(?m)^{level} (?!#)", rest)
    return rest[: end.start()] if end else rest


def first_block(text: str, language: str) -> str:
    match = re.search(rf"```{language}\n(.*?)```", text, re.DOTALL)
    assert match is not None, f"no {language} block"
    return match.group(1)


def statements(sql: str) -> list[str]:
    """The SQL statements of a block, comments dropped and whitespace normalised."""
    code = "\n".join(line for line in sql.splitlines() if not line.lstrip().startswith("--"))
    return [" ".join(part.split()) for part in code.split(";") if part.strip()]


DB_BLOCK = first_block(section(DB_MD, "## 5. Roles and privileges"), "sql")
RUNBOOK_BLOCK = first_block(section(RUNBOOK, "## 6. Create the `app` role and its grants"), "sql")


def test_the_runbook_applies_exactly_the_statements_of_db_md_section_5() -> None:
    assert statements(RUNBOOK_BLOCK) == statements(DB_BLOCK)
    assert statements(DB_BLOCK)[0] == "CREATE ROLE app LOGIN"  # the role statement is in the block


def test_no_password_is_written_anywhere_in_the_role_sql() -> None:
    for text in (DB_BLOCK, RUNBOOK_BLOCK):
        assert not re.search(r"(?i)\bpassword\s+'", text)
    # The password is set interactively, so that it reaches neither a file nor the shell history.
    assert "\\password app" in RUNBOOK


def expected_privileges(sql: str) -> dict[str, list[str]]:
    """Table privileges of ``app`` that the GRANT statements give (the schema USAGE is not one)."""
    granted: dict[str, set[str]] = {}
    for statement in statements(sql):
        match = re.fullmatch(r"GRANT (.+?) ON (.+) TO app", statement)
        assert match is not None or statement.startswith("CREATE ROLE"), statement
        if match is None or match.group(2).startswith("SCHEMA"):
            continue
        privileges = {p.strip() for p in match.group(1).split(",")}
        for table in (t.strip() for t in match.group(2).split(",")):
            granted.setdefault(table, set()).update(privileges)
    return {
        table: [p for p in PRIVILEGE_ORDER if p in privileges]
        for table, privileges in granted.items()
    }


def test_the_expected_privilege_table_is_what_db_md_section_5_grants() -> None:
    step = section(RUNBOOK, "## 7. Least-privilege checks (the Author's manual check)")
    rows = re.findall(r"(?m)^\| (\w+) \| ([A-Z, ]+) \|$", step)
    table = {name: privileges.split(", ") for name, privileges in rows if name != "table_name"}

    assert table == expected_privileges(DB_BLOCK)
    assert "UPDATE" not in table["request_logs"]  # telemetry is append-only for the runtime
    assert "DELETE" not in table["rate_limit_counters"]
    assert "DELETE" not in table["daily_usage"]
    assert table["chunks"] == ["SELECT"]


def test_every_bash_block_is_one_command() -> None:
    blocks = re.findall(r"```bash\n(.*?)```", RUNBOOK, re.DOTALL)

    assert blocks
    assert all(len(block.strip().splitlines()) == 1 for block in blocks)


def test_the_runbook_names_the_secrets_the_workflow_reads_and_the_workflow_file() -> None:
    secrets = set(re.findall(r"secrets\.([A-Z_]+)", INGEST))

    assert secrets == {"DATABASE_URL_DIRECT", "GEMINI_API_KEY"}
    for name in secrets:
        assert f"`{name}`" in RUNBOOK
    assert "ingest.yml" in RUNBOOK
    assert "EVAL_RECORD_RUNS" in RUNBOOK  # the optional last step, which the Author sets
