"""Forward-only SQL migration runner (DB.md §10).

Rules it enforces:
- Files are ``NNNN_short_name.sql``, applied in numeric order, each in its own transaction together
  with its ``schema_migrations`` row, so a failed file leaves no trace.
- An applied file must never change: every recorded checksum is verified before anything runs,
  and a mismatch aborts the whole run.
- A recorded version with no file on disk also aborts (an applied migration was deleted or renamed).
- A session-level advisory lock serializes concurrent runners (e.g. two CI jobs on one database).

This is offline tooling, not the request path, so it uses the synchronous psycopg API.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import psycopg

# Default location: backend/migrations (this file is backend/src/grounded/infra/migrations.py).
DEFAULT_MIGRATIONS_DIR = Path(__file__).resolve().parents[3] / "migrations"

_FILENAME_RE = re.compile(r"^(?P<number>\d{4})_[a-z0-9_]+\.sql$")

# Arbitrary but fixed key for pg_advisory_lock; only has to be unique within this database.
_ADVISORY_LOCK_KEY = 7_401_202_600

# Must stay identical to the definition in 0001_init.sql, which uses IF NOT EXISTS so that this
# bootstrap and the migration can both run.
_BOOTSTRAP_SQL = b"""
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     text PRIMARY KEY,
    checksum    text NOT NULL,
    applied_at  timestamptz NOT NULL DEFAULT now()
)
"""


class MigrationError(Exception):
    """Base class for runner failures."""


class InvalidMigrationFileError(MigrationError):
    pass


class ChecksumMismatchError(MigrationError):
    pass


class UnknownAppliedMigrationError(MigrationError):
    pass


class MigrationFailedError(MigrationError):
    """A migration's SQL failed and was rolled back; earlier ones in the same run stay applied."""


@dataclass(frozen=True, slots=True)
class Migration:
    version: str  # file stem, e.g. "0001_init"
    path: Path
    sql: str
    checksum: str


@dataclass(slots=True)
class MigrationReport:
    applied: list[str] = field(default_factory=list[str])
    already_applied: list[str] = field(default_factory=list[str])


def checksum(sql: str) -> str:
    """sha256 of the file text with line endings normalized.

    Normalizing means a Windows checkout with CRLF (despite .gitattributes) does not look like an
    edited migration, while any real content change still does.
    """
    normalized = sql.replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def discover_migrations(directory: Path) -> list[Migration]:
    if not directory.is_dir():
        raise InvalidMigrationFileError(f"migrations directory not found: {directory}")

    migrations: list[Migration] = []
    seen_numbers: dict[str, str] = {}
    for path in sorted(directory.glob("*.sql")):
        match = _FILENAME_RE.match(path.name)
        if match is None:
            raise InvalidMigrationFileError(
                f"bad migration filename {path.name!r}: expected NNNN_short_name.sql"
            )
        number = match["number"]
        if number in seen_numbers:
            raise InvalidMigrationFileError(
                f"duplicate migration number {number}: {seen_numbers[number]} and {path.name}"
            )
        seen_numbers[number] = path.name
        sql = path.read_text(encoding="utf-8")
        migrations.append(Migration(path.stem, path, sql, checksum(sql)))
    return migrations


def plan(migrations: list[Migration], applied: dict[str, str]) -> list[Migration]:
    """Return pending migrations, or raise if the recorded history disagrees with the files.

    ``applied`` maps version -> recorded checksum. Pure function, so the safety rules are
    unit-testable without a database.
    """
    by_version = {m.version: m for m in migrations}
    unknown = sorted(set(applied) - set(by_version))
    if unknown:
        raise UnknownAppliedMigrationError(
            f"applied migrations missing on disk: {', '.join(unknown)}"
        )
    mismatched = [
        m.version for m in migrations if m.version in applied and applied[m.version] != m.checksum
    ]
    if mismatched:
        raise ChecksumMismatchError(
            f"applied migrations were edited: {', '.join(mismatched)}. "
            "Never edit an applied migration; add a new one instead."
        )
    return [m for m in migrations if m.version not in applied]


def migrate(conninfo: str, directory: Path = DEFAULT_MIGRATIONS_DIR) -> MigrationReport:
    migrations = discover_migrations(directory)
    report = MigrationReport()

    # autocommit: the advisory lock is session-level (released when the connection closes, even on
    # error) and each conn.transaction() below is a real BEGIN/COMMIT.
    with psycopg.connect(conninfo, autocommit=True) as conn:
        conn.execute("SELECT pg_advisory_lock(%s)", (_ADVISORY_LOCK_KEY,))
        conn.execute(_BOOTSTRAP_SQL)
        rows = conn.execute("SELECT version, checksum FROM schema_migrations").fetchall()
        applied: dict[str, str] = {version: recorded for version, recorded in rows}

        pending = plan(migrations, applied)
        report.already_applied = [m.version for m in migrations if m.version in applied]

        for migration in pending:
            try:
                with conn.transaction():
                    # Passed as bytes with no parameters: psycopg then uses the simple query
                    # protocol, which allows several statements per call and no placeholder parsing.
                    conn.execute(migration.sql.encode("utf-8"))
                    conn.execute(
                        "INSERT INTO schema_migrations (version, checksum) VALUES (%s, %s)",
                        (migration.version, migration.checksum),
                    )
            except psycopg.Error as exc:
                raise MigrationFailedError(
                    f"{migration.version} failed and was rolled back "
                    f"(applied earlier in this run: {report.applied or 'none'}): {exc}"
                ) from exc
            report.applied.append(migration.version)
    return report
