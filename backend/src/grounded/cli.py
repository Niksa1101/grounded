"""``grounded`` command-line entry point (Typer).

Later phases add the ask command and more eval suites.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Coroutine
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import psycopg
import typer
import uvicorn
from psycopg.conninfo import conninfo_to_dict

from grounded.evals.golden import (
    GOLDEN_DIR,
    TARGET_TYPE_COUNTS,
    GoldenSetError,
    active_index_chunks,
    load_golden_set,
    normalize_page,
    page_sections,
    resolve_labels,
    sample_sections,
    type_counts,
)
from grounded.evals.retrieval_runner import (
    RETRIEVAL_BASELINE,
    BaselineMismatchError,
    RetrievalEvalError,
    golden_set_digest,
    golden_set_version,
    metric_names,
    repo_state,
    results_path,
    run_retrieval_eval,
    update_baseline,
    write_run,
)
from grounded.infra.kvcache import KVCache
from grounded.infra.logging import configure_logging
from grounded.infra.migrations import DEFAULT_MIGRATIONS_DIR, MigrationError
from grounded.infra.migrations import migrate as run_migrations
from grounded.infra.provider_errors import ProviderError
from grounded.ingest.corpus import CorpusError, fetch_corpus
from grounded.ingest.embed import (
    CachedEmbedder,
    EmbedderUnavailableError,
    GeminiEmbedder,
    LazyEmbedder,
    count_uncached_texts,
)
from grounded.ingest.includes import IncludeError
from grounded.ingest.markdown import ParseError
from grounded.ingest.pipeline import (
    EmbeddingQuotaExhaustedError,
    IngestError,
    PreparedCorpus,
    count_uncached,
    index_spec,
    list_index_versions,
    prepare_corpus,
    token_stats,
)
from grounded.ingest.pipeline import ingest as run_ingest
from grounded.ingest.tokens import TokenCounter, make_token_counter
from grounded.ingest.types import ChunkingConfig
from grounded.retrieval.config import RetrievalConfig, RetrievalMode
from grounded.retrieval.index import NoActiveIndexError
from grounded.settings import Settings, get_settings

app = typer.Typer(no_args_is_help=True, help="Grounded command-line tools.")
index_app = typer.Typer(no_args_is_help=True, help="Inspect index versions.")
app.add_typer(index_app, name="index")
golden_app = typer.Typer(no_args_is_help=True, help="Draft, label and validate the golden set.")
app.add_typer(golden_app, name="golden")
eval_app = typer.Typer(no_args_is_help=True, help="Run evals against the active index.")
app.add_typer(eval_app, name="eval")

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
    corpus, count_tokens = _prepare_corpus(settings, ref)
    checkout, cfg = corpus.checkout, corpus.chunking
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
        # The key is only needed if a text must be sent: an already built version or a fully
        # cached corpus goes through without one.
        embedder = LazyEmbedder(
            settings.embedding_model,
            settings.embedding_dim,
            lambda: GeminiEmbedder.from_settings(settings, count_tokens),
        )
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
        except EmbedderUnavailableError as exc:
            raise _fail(
                f"{exc}: {count_uncached(cache, spec, chunks)} texts are not cached. "
                "Nothing was written to the database."
            ) from exc
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


def _prepare_corpus(settings: Settings, ref: str | None) -> tuple[PreparedCorpus, TokenCounter]:
    """Fetch (or reuse) the pinned checkout, then parse and chunk it with the configured chunker."""
    ref = ref or settings.fastapi_ref
    cfg = ChunkingConfig.from_settings(settings)
    count_tokens = make_token_counter(cfg.tokenizer)
    try:
        checkout = fetch_corpus(ref, settings.cache_dir)
        return prepare_corpus(checkout, cfg, count_tokens), count_tokens
    except (CorpusError, IncludeError, ParseError, IngestError) as exc:
        raise _fail(f"Corpus preparation failed: {exc}") from exc


def _describe_stats(stats: dict[str, int], max_tokens: int) -> str:
    return (
        f"~{stats['total']} tokens (p50 {stats['p50']}, p95 {stats['p95']}, max {stats['max']}; "
        f"{stats['over_max']} over {max_tokens})"
    )


_REF_OPTION = typer.Option(help="FastAPI tag. Default: FASTAPI_REF.", show_default=False)


@golden_app.command("sample")
def golden_sample(
    n: Annotated[int, typer.Option(help="How many sections to draw.")] = 50,
    seed: Annotated[int, typer.Option(help="RNG seed; record it with the candidates.")] = 20260926,
    ref: Annotated[str | None, _REF_OPTION] = None,
) -> None:
    """Draw random H2/H3 sections of the chunked corpus to draft candidate questions from."""
    settings = get_settings()
    configure_logging(settings.log_level)
    corpus, _ = _prepare_corpus(settings, ref)
    chunks = corpus.chunks
    try:
        sample = sample_sections(chunks, n, seed)
    except ValueError as exc:
        raise _fail(str(exc)) from exc
    breadcrumbs = {chunk.section_id: chunk.breadcrumb_text for chunk in chunks}
    for section_id in sample:
        typer.echo(f"{section_id}  {breadcrumbs[section_id]}")


@golden_app.command("sections")
def golden_sections(
    page: Annotated[str, typer.Argument(help="docs/en/docs/<page>.md or <page>.md")],
    ref: Annotated[str | None, _REF_OPTION] = None,
) -> None:
    """List a page's labelable sections as the chunker produced them."""
    settings = get_settings()
    configure_logging(settings.log_level)
    corpus, _ = _prepare_corpus(settings, ref)
    sections = page_sections(corpus.chunks, normalize_page(page))
    if not sections:
        raise _fail(f"No chunks for {normalize_page(page)}: not a page of this corpus.")
    for info in sections:
        indent = "  " * max(0, info.heading_level - 1)
        parts = f", {info.chunk_count} chunks" if info.chunk_count > 1 else ""
        typer.echo(f"{indent}{info.section_id}  ({info.token_count} tokens{parts})")
        typer.echo(f"{indent}    {info.breadcrumb_text}")


@golden_app.command("validate")
def golden_validate(
    path: Annotated[Path, typer.Argument(help="Golden-set JSONL file.")] = GOLDEN_DIR
    / "golden_set.v1.jsonl",
    against_index: Annotated[
        bool, typer.Option(help="Also resolve labels against the active index in the database.")
    ] = False,
    database_url: Annotated[
        str | None,
        typer.Option(
            help="Database for --against-index. Default: DATABASE_URL_DIRECT, else DATABASE_URL.",
            show_default=False,
        ),
    ] = None,
    ref: Annotated[str | None, _REF_OPTION] = None,
) -> None:
    """Check the schema, unique IDs and type balance, and resolve every label to chunks."""
    settings = get_settings()
    configure_logging(settings.log_level)
    try:
        items = load_golden_set(path)
    except (GoldenSetError, OSError) as exc:
        raise _fail(f"Invalid golden set:\n{exc}") from exc

    counts = type_counts(items)
    answerable = sum(1 for item in items if item.answerable)
    typer.echo(f"{path.name}: {len(items)} items, {answerable} answerable")
    for kind, target in TARGET_TYPE_COUNTS.items():
        typer.echo(f"  {kind:<14} {counts[kind]:>3}  (target ~{target})")

    corpus, _ = _prepare_corpus(settings, ref)
    problems = [f"corpus: {p}" for p in resolve_labels(items, corpus.chunks)]
    if against_index:
        conninfo = database_url or settings.migration_database_url.get_secret_value()
        try:
            version_id, indexed = active_index_chunks(conninfo)
        except (GoldenSetError, psycopg.Error) as exc:
            raise _fail(f"Cannot read the active index: {exc}") from exc
        problems += [f"index {version_id}: {p}" for p in resolve_labels(items, indexed)]
    if problems:
        raise _fail("Label problems:\n" + "\n".join(f"  {p}" for p in problems))
    typer.echo("All labels resolve" + (" in the corpus and the index." if against_index else "."))


def _run_async[T](coro: Coroutine[Any, Any, T]) -> T:
    # psycopg async can't run on the Proactor loop that asyncio.run picks on Windows.
    loop_factory = asyncio.SelectorEventLoop if sys.platform == "win32" else None
    return asyncio.run(coro, loop_factory=loop_factory)


@eval_app.command("retrieval")
def eval_retrieval(
    config: Annotated[
        list[str] | None,
        typer.Option(
            "--config",
            help="Retrieval mode to score (repeatable): dense, fts or hybrid.",
            show_default=False,
        ),
    ] = None,
    golden: Annotated[
        Path, typer.Option(help="Golden-set file, named golden_set.v<N>.jsonl.")
    ] = GOLDEN_DIR / "golden_set.v1.jsonl",
    out: Annotated[
        Path | None,
        typer.Option(
            help="Results file. Default: eval/results/<UTC timestamp>-retrieval.json.",
            show_default=False,
        ),
    ] = None,
    write_baseline: Annotated[
        bool,
        typer.Option(help=f"Also copy this run's rows into {RETRIEVAL_BASELINE.name}."),
    ] = False,
    database_url: Annotated[
        str | None,
        typer.Option(
            help="Database with the active index. Default: DATABASE_URL.", show_default=False
        ),
    ] = None,
) -> None:
    """Score retrieval on the golden set's answerable questions: Recall@k, MRR, nDCG@k."""
    settings = get_settings()
    configure_logging(settings.log_level)
    modes = _retrieval_modes(config or ["dense"])
    try:
        version = golden_set_version(golden)
        digest = golden_set_digest(golden)
        items = load_golden_set(golden)
    except (RetrievalEvalError, GoldenSetError, OSError) as exc:
        raise _fail(f"Invalid golden set: {exc}") from exc

    now = datetime.now(UTC)
    conninfo = database_url or settings.database_url.get_secret_value()
    typer.echo(f"Target {_describe_target(conninfo)}; golden set {version} ({len(items)} items)")
    count_tokens = make_token_counter(settings.tokenizer_encoding)
    with KVCache(settings.cache_dir / _EMBEDDINGS_CACHE) as cache:
        gemini = LazyEmbedder(
            settings.embedding_model,
            settings.embedding_dim,
            lambda: GeminiEmbedder.from_settings(settings, count_tokens),
        )
        # Query vectors are cached like chunk vectors (Tech.md §11), so a re-run is free, and
        # with every vector cached no API key is needed.
        embedder = CachedEmbedder(gemini, cache, write_every=settings.embedding_batch_size)
        try:
            run = _run_async(
                run_retrieval_eval(
                    conninfo,
                    items,
                    configs=[RetrievalConfig.from_settings(settings, mode) for mode in modes],
                    embedder=embedder,
                    golden_set_version=version,
                    golden_set_sha256=digest,
                    now=now,
                    repo=repo_state(),
                )
            )
        except (RetrievalEvalError, NoActiveIndexError) as exc:
            raise _fail(f"Eval failed: {exc}") from exc
        except EmbedderUnavailableError as exc:
            uncached = count_uncached_texts(
                cache,
                embedder.model,
                embedder.dim,
                "RETRIEVAL_QUERY",
                [item.question for item in items if item.answerable],
            )
            raise _fail(f"{exc}: {uncached} questions are not cached.") from exc
        except ProviderError as exc:
            raise _fail(f"Query embedding failed ({type(exc).__name__}): {exc}") from exc
        except psycopg.Error as exc:
            raise _fail(f"Database error: {exc}") from exc

    typer.echo(
        f"Index {run.info.fastapi_ref}@{run.info.index_config_hash[:8]}; "
        f"{embedder.misses} questions embedded, {embedder.hits} from cache"
    )
    names = metric_names()
    typer.echo(f"{'config':<8} {'n':>3} {'k':>3}  " + "  ".join(f"{m:>9}" for m in names))
    for result in run.configs.values():
        values = "  ".join(f"{result.metrics[m]:>9.3f}" for m in names)
        typer.echo(f"{result.config:<8} {result.n:>3} {result.k:>3}  {values}")
    skipped = next(iter(run.configs.values())).skipped_unanswerable
    typer.echo(f"Skipped {skipped} unanswerable questions (not scored).")

    path = out or results_path(now)
    write_run(path, run)
    typer.echo(f"Results: {path}")
    if write_baseline:
        if run.info.git_dirty:
            typer.echo(
                "Warning: uncommitted changes; the baseline's git_sha isn't the code that ran."
            )
        try:
            update_baseline(RETRIEVAL_BASELINE, run)
        except BaselineMismatchError as exc:
            raise _fail(f"Baseline not updated: {exc}") from exc
        typer.echo(f"Baseline updated: {RETRIEVAL_BASELINE} ({', '.join(run.configs)})")


_RETRIEVAL_MODES: tuple[RetrievalMode, ...] = ("dense", "fts", "hybrid")


def _retrieval_modes(names: list[str]) -> list[RetrievalMode]:
    modes: list[RetrievalMode] = []
    for name in names:
        for mode in _RETRIEVAL_MODES:
            if mode == name:
                modes.append(mode)
                break
        else:
            available = ", ".join(_RETRIEVAL_MODES)
            raise _fail(f"Unknown retrieval config {name!r}; available: {available}.")
    return modes
