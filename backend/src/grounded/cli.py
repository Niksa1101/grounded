"""``grounded`` command-line entry point (Typer).

Later phases add the ingest, index, eval, ask and golden commands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from psycopg.conninfo import conninfo_to_dict

from grounded.infra.migrations import DEFAULT_MIGRATIONS_DIR, MigrationError
from grounded.infra.migrations import migrate as run_migrations
from grounded.settings import get_settings

app = typer.Typer(no_args_is_help=True, help="Grounded command-line tools.")


@app.callback()
def main() -> None:
    """Grounded command-line tools."""
    # An explicit callback keeps `migrate` a subcommand even while it is the only command.


def _describe_target(conninfo: str) -> str:
    """host:port/dbname without credentials, safe to print."""
    params = conninfo_to_dict(conninfo)
    return f"{params.get('host', '?')}:{params.get('port', '5432')}/{params.get('dbname', '?')}"


@app.command()
def migrate(
    database_url: Annotated[
        str | None,
        typer.Option(
            help="Target database. Default: DATABASE_URL_DIRECT, else DATABASE_URL.",
            show_default=False,
        ),
    ] = None,
    migrations_dir: Annotated[
        Path, typer.Option(help="Directory with NNNN_name.sql files.")
    ] = DEFAULT_MIGRATIONS_DIR,
) -> None:
    """Apply pending SQL migrations in order, each in its own transaction."""
    conninfo = database_url or get_settings().migration_database_url.get_secret_value()
    typer.echo(f"Migrating {_describe_target(conninfo)}")
    try:
        report = run_migrations(conninfo, migrations_dir)
    except MigrationError as exc:
        typer.echo(f"Migration aborted: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    for version in report.applied:
        typer.echo(f"  applied  {version}")
    typer.echo(
        f"Done: {len(report.applied)} applied, {len(report.already_applied)} already up to date."
    )
