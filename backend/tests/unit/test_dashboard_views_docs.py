"""DB.md §8 shows the dashboard views as migration 0005 creates them (ticket 8.01).

The doc is where a reader learns what the dashboard counts, so its SQL must be the migration's,
statement for statement (comments and whitespace aside). A change to the views is a new migration
and a new §8 block in the same PR. No database is needed.
"""

from __future__ import annotations

from grounded.infra.migrations import DEFAULT_MIGRATIONS_DIR
from tests.unit.test_neon_runbook import DB_MD, first_block, section, statements

MIGRATION = (DEFAULT_MIGRATIONS_DIR / "0005_dashboard_views.sql").read_text(encoding="utf-8")
DB_SECTION_8 = section(DB_MD, "## 8. Dashboard views (Phase 8, Agent)")


def test_db_md_section_8_holds_exactly_the_statements_of_migration_0005() -> None:
    documented = statements(first_block(DB_SECTION_8, "sql"))

    assert documented == statements(MIGRATION)
    assert [s.split(" AS ", 1)[0] for s in documented if s.startswith("CREATE VIEW")] == [
        "CREATE VIEW v_request_stats_daily",
        "CREATE VIEW v_stage_latency_7d",
        "CREATE VIEW v_request_metrics",
    ]


def test_the_view_grant_of_db_md_section_5_names_the_views_the_migration_grants() -> None:
    grant = "GRANT SELECT ON v_request_stats_daily, v_stage_latency_7d, v_request_metrics TO app"

    assert grant in statements(first_block(section(DB_MD, "## 5. Roles and privileges"), "sql"))
    assert grant in " ".join(statements(MIGRATION))
    assert "0005_dashboard_views.sql" in DB_MD
