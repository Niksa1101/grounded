"""Migration discovery and planning rules. No database needed; integration/ covers the real run."""

from __future__ import annotations

from pathlib import Path

import pytest

from grounded.infra.migrations import (
    DEFAULT_MIGRATIONS_DIR,
    ChecksumMismatchError,
    InvalidMigrationFileError,
    Migration,
    OutOfOrderMigrationError,
    UnknownAppliedMigrationError,
    checksum,
    discover_migrations,
    plan,
)


def _write(directory: Path, name: str, body: str = "SELECT 1;\n") -> None:
    (directory / name).write_text(body, encoding="utf-8", newline="")


def test_discovers_in_numeric_order(tmp_path: Path) -> None:
    _write(tmp_path, "0002_second.sql")
    _write(tmp_path, "0001_first.sql")
    _write(tmp_path, "0010_tenth.sql")
    assert [m.version for m in discover_migrations(tmp_path)] == [
        "0001_first",
        "0002_second",
        "0010_tenth",
    ]


@pytest.mark.parametrize("name", ["1_init.sql", "0001-init.sql", "0001_Init.sql", "init.sql"])
def test_rejects_bad_filenames(tmp_path: Path, name: str) -> None:
    _write(tmp_path, name)
    with pytest.raises(InvalidMigrationFileError, match="bad migration filename"):
        discover_migrations(tmp_path)


def test_rejects_duplicate_numbers(tmp_path: Path) -> None:
    _write(tmp_path, "0001_a.sql")
    _write(tmp_path, "0001_b.sql")
    with pytest.raises(InvalidMigrationFileError, match="duplicate migration number 0001"):
        discover_migrations(tmp_path)


def test_missing_directory_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(InvalidMigrationFileError, match="not found"):
        discover_migrations(tmp_path / "nope")


def test_checksum_ignores_line_endings_but_not_content() -> None:
    assert checksum("SELECT 1;\r\nSELECT 2;\r\n") == checksum("SELECT 1;\nSELECT 2;\n")
    assert checksum("SELECT 1;\n") != checksum("SELECT 2;\n")


class TestPlan:
    @pytest.fixture
    def migrations(self, tmp_path: Path) -> list[Migration]:
        _write(tmp_path, "0001_first.sql", "SELECT 1;\n")
        _write(tmp_path, "0002_second.sql", "SELECT 2;\n")
        return discover_migrations(tmp_path)

    def test_fresh_database_gets_everything(self, migrations: list[Migration]) -> None:
        assert [m.version for m in plan(migrations, {})] == ["0001_first", "0002_second"]

    def test_only_pending_are_returned(self, migrations: list[Migration]) -> None:
        applied = {"0001_first": migrations[0].checksum}
        assert [m.version for m in plan(migrations, applied)] == ["0002_second"]

    def test_edited_applied_migration_aborts(self, migrations: list[Migration]) -> None:
        applied = {"0001_first": "0" * 64}
        with pytest.raises(ChecksumMismatchError, match="0001_first"):
            plan(migrations, applied)

    def test_pending_migration_older_than_applied_aborts(self, tmp_path: Path) -> None:
        # e.g. a branch adding 0002 merged after 0003 was already applied somewhere
        for name in ("0001_first.sql", "0002_late.sql", "0003_third.sql"):
            _write(tmp_path, name)
        migrations = discover_migrations(tmp_path)
        applied = {m.version: m.checksum for m in migrations if m.version != "0002_late"}
        with pytest.raises(OutOfOrderMigrationError, match="0002_late"):
            plan(migrations, applied)

    def test_applied_migration_missing_on_disk_aborts(self, migrations: list[Migration]) -> None:
        applied = {m.version: m.checksum for m in migrations} | {"0003_gone": "x"}
        with pytest.raises(UnknownAppliedMigrationError, match="0003_gone"):
            plan(migrations, applied)


def test_repo_migrations_are_valid() -> None:
    versions = [m.version for m in discover_migrations(DEFAULT_MIGRATIONS_DIR)]
    assert versions[0] == "0001_init"
