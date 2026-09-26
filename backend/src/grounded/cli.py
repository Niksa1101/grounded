"""``grounded`` command-line entry point (Typer).

Later phases add the eval, ask and golden commands.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import psycopg
import typer
import uvicorn
from psycopg.conninfo import conninfo_to_dict

from grounded.infra.kvcache import KVCache
from grounded.infra.logging import configure_logging
from grounded.infra.migrations import DEFAULT_MIGRATIONS_DIR, MigrationError
from grounded.infra.migrations import migrate as run_migrations
from grounded.infra.provider_errors import ProviderError
from grounded.ingest.corpus import CorpusError, fetch_corpus
from grounded.ingest.embed import GeminiEmbedder
from grounded.ingest.includes import IncludeError
from grounded.ingest.markdown import ParseError
from grounded.ingest.pipeline import (
    EmbeddingQuotaExhaustedError,
    IngestError,
    count_uncached,
    index_spec,
    list_index_versions,
    prepare_corpus,
    token_stats,
)
from grounded.ingest.pipeline import ingest as run_ingest
from grounded.ingest.tokens import make_token_counter
from grounded.ingest.types import ChunkingConfig
from grounded.settings import get_settings

app = typer.Typer(no_args_is_help=True, help="Grounded command-line tools.")
index_app = typer.Typer(no_args_is_help=True, help="Inspect index versions.")
app.add_typer(index_app, name="index")

_EMBEDDINGS_CACHE = "embeddings.sqlite"


@app.callback()
def main() -> None:
    """Grounded command-line tools."""
    # An explicit callback keeps commands as subcommands (`grounded migrate`), whatever their count.


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Interface to bind (0.0.0.0 in a container).")] = (
        "127.0.0.1"
    ),
    port: Annotated[int, typer.Option(help="Port to listen on.")] = 8000,
    reload: Annotated[bool, typer.Option(help="Restart on code changes (dev only).")] = False,
) -> None:
    """Run the API with uvicorn: app factory, JSON logs, no access log."""
    configure_logging(get_settings().log_level)
    uvicorn.run(
        "grounded.main:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        # uvicorn's access log prints raw client IPs (AGENTS.md §6.13). Request logging with an
        # HMAC'd IP comes with request_logs.
        access_log=False,
        # Keep our JSON root logger instead of uvicorn's own dictConfig.
        log_config=None,
        # psycopg async can't run on the Proactor loop, which uvicorn picks on Windows unless it
        # runs the server in a subprocess (--reload/--workers). Elsewhere uvicorn's choice is fine.
        loop="asyncio:SelectorEventLoop" if sys.platform == "win32" else "auto",
    )


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


def _fail(message: str) -> typer.Exit:
    """Print ``message`` to stderr and return the exit to raise (``raise _fail(...)``)."""
    typer.echo(message, err=True)
    return typer.Exit(code=1)


@app.command()
def ingest(
    ref: Annotated[
        str | None,
        typer.Option(help="FastAPI tag to index. Default: FASTAPI_REF.", show_default=False),
    ] = None,
    database_url: Annotated[
        str | None,
        typer.Option(
            help="Target database. Default: DATABASE_URL_DIRECT, else DATABASE_URL.",
            show_default=False,
        ),
    ] = None,
    activate: Annotated[
        bool, typer.Option(help="Make the built (or already built) version the active one.")
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(help="Parse and chunk, count texts that need embedding. No API, no database."),
    ] = False,
) -> None:
    """Build an index version from a pinned FastAPI tag: parse, chunk, embed, store, verify."""
    settings = get_settings()
    configure_logging(settings.log_level)
    ref = ref or settings.fastapi_ref
    if not ref:
        raise _fail("No tag given: pass --ref or set FASTAPI_REF.")
    if not settings.embedding_model:
        raise _fail("EMBEDDING_MODEL must be set to the pinned embedding model ID.")

    cfg = ChunkingConfig.from_settings(settings)
    count_tokens = make_token_counter(cfg.tokenizer)
    try:
        checkout = fetch_corpus(ref, settings.cache_dir)
        corpus = prepare_corpus(checkout, cfg, count_tokens)
    except (CorpusError, IncludeError, ParseError, IngestError) as exc:
        raise _fail(f"Ingest aborted: {exc}") from exc
    spec = index_spec(
        checkout,
        cfg,
        embedding_model=settings.embedding_model,
        embedding_dim=settings.embedding_dim,
    )
    chunks = corpus.chunks
    stats = token_stats([chunk.token_count for chunk in chunks], cfg.max_tokens)
    typer.echo(
        f"Corpus {checkout.ref} @ {checkout.sha[:8]}: {len(corpus.documents)} pages, "
        f"{len(chunks)} chunks, {_describe_stats(stats, cfg.max_tokens)}"
    )
    typer.echo(f"Index {spec.label}: {spec.embedding_model}, {spec.embedding_dim} dims")

    with KVCache(settings.cache_dir / _EMBEDDINGS_CACHE) as cache:
        if dry_run:
            uncached = count_uncached(cache, spec, chunks)
            typer.echo(f"Dry run: {uncached} texts to embed, the rest are cached.")
            return
        conninfo = database_url or settings.migration_database_url.get_secret_value()
        typer.echo(f"Target {_describe_target(conninfo)}")
        try:
            embedder = GeminiEmbedder.from_settings(settings, count_tokens)
        except ValueError as exc:
            raise _fail(str(exc)) from exc
        try:
            report = run_ingest(
                conninfo,
                corpus,
                spec,
                embedder=embedder,
                cache=cache,
                # One provider batch per slice: a call's vectors are cached before the next call.
                write_every=settings.embedding_batch_size,
                count_tokens=count_tokens,
                max_input_tokens=settings.embedding_max_input_tokens,
                activate=activate,
            )
        except EmbeddingQuotaExhaustedError as exc:
            raise _fail(
                f"Stopped: {exc}. Nothing was written to the database. Run the same command "
                "again after the quota resets (midnight Pacific); cached texts are not re-sent."
            ) from exc
        except ProviderError as exc:
            raise _fail(
                f"Embedding failed ({type(exc).__name__}): {exc}. Nothing was written to the "
                "database; texts embedded so far are cached."
            ) from exc
        except IngestError as exc:
            raise _fail(f"Ingest failed: {exc}") from exc
        except psycopg.Error as exc:
            raise _fail(f"Database error: {exc}. Embedded texts are cached.") from exc

    if report.created:
        typer.echo(
            f"Embedded {report.embedded} texts in {embedder.api_calls} API calls, "
            f"{report.cache_hits} from cache."
        )
        typer.echo(f"Stored index version {report.index_version_id} (ready).")
    else:
        typer.echo(f"Index version {report.index_version_id} with this config is already built.")
    typer.echo("Active: yes." if report.activated else "Active: no (use --activate).")


@index_app.command("list")
def index_list(
    database_url: Annotated[
        str | None,
        typer.Option(
            help="Database to read. Default: DATABASE_URL_DIRECT, else DATABASE_URL.",
            show_default=False,
        ),
    ] = None,
) -> None:
    """Show index versions, newest first."""
    conninfo = database_url or get_settings().migration_database_url.get_secret_value()
    try:
        rows = list_index_versions(conninfo)
    except (psycopg.Error, IngestError) as exc:
        raise _fail(f"Cannot read index versions: {exc}") from exc
    if not rows:
        typer.echo("No index versions.")
        return
    typer.echo(
        f"{'id':>4}  {'ref':<10} {'sha':<8}  {'config':<8}  {'status':<8} {'active':<6} "
        f"{'docs':>5} {'chunks':>6}  tokens p50/p95/max"
    )
    for row in rows:
        stats = row.token_stats or {}
        tokens = "/".join(str(stats.get(key, "-")) for key in ("p50", "p95", "max"))
        typer.echo(
            f"{row.id:>4}  {row.git_ref:<10} {row.git_sha[:8]:<8}  {row.config_hash[:8]:<8}  "
            f"{row.status:<8} {'*' if row.is_active else '':<6} "
            f"{row.document_count or 0:>5} {row.chunk_count or 0:>6}  {tokens}"
        )


def _describe_stats(stats: dict[str, int], max_tokens: int) -> str:
    return (
        f"~{stats['total']} tokens (p50 {stats['p50']}, p95 {stats['p95']}, max {stats['max']}; "
        f"{stats['over_max']} over {max_tokens})"
    )
