"""``grounded`` command-line entry point (Typer).

Later phases add more eval suites.
"""

from __future__ import annotations

import asyncio
import io
import json
import sys
from collections.abc import Coroutine
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

import psycopg
import typer
import uvicorn
from psycopg.conninfo import conninfo_to_dict
from pydantic import ValidationError

from grounded.evals.agreement import (
    DEFAULT_KEY,
    DEFAULT_SHEET,
    LabelError,
    SheetError,
    parse_sheet,
    rate_items,
    render_report,
)
from grounded.evals.ask_batch import (
    DEFAULT_POLICY,
    QuestionResult,
    RetryPolicy,
    ask_results_path,
    is_complete,
    render_summary,
    run_batch,
    summarize,
    write_ask_batch,
)
from grounded.evals.eval_record import (
    EvalRecordError,
    EvalRunRow,
    build_rows,
    insert_rows,
    read_active_index,
)
from grounded.evals.gate import (
    GateCannotRunError,
    evaluate_gate,
    evaluate_generation_gate,
    exit_code,
    render_generation_markdown,
    render_markdown,
)
from grounded.evals.generation_baseline import (
    GenerationBaselineError,
    check_against_itself,
    update_generation_baseline,
)
from grounded.evals.generation_results import (
    GENERATION_BASELINE,
    read_generation_baseline,
    read_generation_results,
)
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
from grounded.evals.report import build_report
from grounded.evals.retrieval_runner import (
    RETRIEVAL_BASELINE,
    BaselineMismatchError,
    RetrievalEvalError,
    golden_set_digest,
    golden_set_version,
    metric_names,
    read_baseline,
    read_run,
    repo_state,
    results_path,
    run_retrieval_eval,
    update_baseline,
    write_run,
)
from grounded.generation.pipeline import AskMode
from grounded.generation.providers.eval_wrappers import EvalStats
from grounded.generation.providers.fake import StubLLMProvider
from grounded.infra.event_loop import loop_factory
from grounded.infra.kvcache import KVCache
from grounded.infra.logging import configure_logging
from grounded.infra.migrations import DEFAULT_MIGRATIONS_DIR, MigrationError
from grounded.infra.migrations import migrate as run_migrations
from grounded.infra.provider_errors import ProviderError, ProviderRateLimited
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
from grounded.observability.request_log import RequestTrace
from grounded.retrieval.config import RetrievalConfig, RetrievalMode
from grounded.retrieval.index import NoActiveIndexError
from grounded.runtime import ProviderConfigError, open_runtime
from grounded.schemas.api import AskRequest, AskResponse
from grounded.schemas.eval import GoldenItem
from grounded.schemas.judge_agreement import AgreementKey
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


@app.command()
def ask(
    question: Annotated[
        str | None,
        typer.Argument(help="The question, 3 to 500 characters.", show_default=False),
    ] = None,
    fake: Annotated[
        bool,
        typer.Option(help="Answer with the stub provider instead of a model (no generator key)."),
    ] = False,
    mode: Annotated[
        AskMode,
        typer.Option(
            help="hybrid: retrieve, then answer from the sources (the API's behavior). "
            "no_rag: the model alone, no retrieval and no sources (the eval baseline; CLI only)."
        ),
    ] = AskMode.HYBRID,
    golden: Annotated[
        Path | None,
        typer.Option(
            help="Batch: ask every question of this golden-set file, one at a time, instead of "
            "QUESTION. Prints a summary and writes the answers to --out.",
            show_default=False,
        ),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option(
            help="Batch results file. Default: eval/results/<UTC timestamp>-ask.json (gitignored).",
            show_default=False,
        ),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option(min=1, help="Batch: only the first N questions (a cheap trial run)."),
    ] = None,
    max_rate_limit_retries: Annotated[
        int, typer.Option(min=0, help="Batch: waits on a 429 per question before stopping.")
    ] = DEFAULT_POLICY.max_rate_limit_retries,
    max_wait_s: Annotated[
        float,
        typer.Option(help="Batch: a Retry-After longer than this stops the run (seconds)."),
    ] = DEFAULT_POLICY.max_wait_s,
    default_wait_s: Annotated[
        float, typer.Option(help="Batch: wait when a 429 gives no Retry-After (seconds).")
    ] = DEFAULT_POLICY.default_wait_s,
    max_consecutive_failures: Annotated[
        int,
        typer.Option(min=1, help="Batch: provider-side failures in a row that stop the run."),
    ] = DEFAULT_POLICY.max_consecutive_failures,
) -> None:
    """Ask one question through the same pipeline as POST /v1/ask and print the AskResponse JSON.

    In hybrid mode the question is still embedded for real, from the cache or with GEMINI_API_KEY.
    The no_rag mode embeds nothing and does not touch the database.

    With --golden the whole golden set goes through the pipeline instead (Phase 3 closeout):
    sequentially, honoring Retry-After, stopping on a daily quota. It prints question ids, never
    question text.
    """
    settings = get_settings()
    configure_logging(settings.log_level)
    if golden is not None:
        if question is not None:
            raise _fail("Give either QUESTION or --golden, not both.")
        policy = RetryPolicy(
            max_rate_limit_retries=max_rate_limit_retries,
            default_wait_s=default_wait_s,
            max_wait_s=max_wait_s,
            max_consecutive_failures=max_consecutive_failures,
        )
        _ask_golden(settings, golden, out, limit=limit, fake=fake, mode=mode, policy=policy)
        return
    if question is None:
        raise _fail("Give a QUESTION, or --golden <file> to ask the whole golden set.")
    try:
        request = AskRequest(question=question)
    except ValidationError as exc:
        raise _fail(f"Invalid question: {exc.errors()[0]['msg']}") from exc

    async def run() -> str:
        async with open_runtime(settings, provider=StubLLMProvider() if fake else None) as runtime:
            response = await runtime.pipeline.ask(request.question, uuid4(), mode=mode)
        return json.dumps(response.model_dump(mode="json"), indent=2)  # ASCII-escaped: any console

    try:
        typer.echo(_run_async(run()))
    except ProviderConfigError as exc:
        raise _fail(str(exc)) from exc
    except NoActiveIndexError as exc:
        raise _fail(f"Cannot answer: {exc}") from exc
    except EmbedderUnavailableError as exc:
        raise _fail(f"{exc}: this question is not in the embedding cache.") from exc
    except ProviderError as exc:
        raise _fail(f"Provider failed ({type(exc).__name__}): {exc}") from exc
    except psycopg.Error as exc:
        raise _fail(f"Database error: {exc}") from exc


def _ask_golden(
    settings: Settings,
    golden: Path,
    out: Path | None,
    *,
    limit: int | None,
    fake: bool,
    mode: AskMode,
    policy: RetryPolicy,
) -> None:
    """``grounded ask --golden``: see ``evals/ask_batch.py``. The golden file is only read."""
    try:
        version = golden_set_version(golden)
        digest = golden_set_digest(golden)
        items = load_golden_set(golden)
    except (RetrievalEvalError, GoldenSetError, OSError) as exc:
        raise _fail(f"Invalid golden set: {exc}") from exc
    items = items[:limit]
    now = datetime.now(UTC)
    path = out or ask_results_path(now)
    if path.resolve() == golden.resolve():
        raise _fail("--out must not be the golden-set file.")

    results: list[QuestionResult] = []
    eval_stats: list[EvalStats] = []  # one entry when APP_ENV=eval wrapped the provider (4.03)

    def record(result: QuestionResult) -> None:
        results.append(result)
        typer.echo(f"{result.id} {result.status if result.outcome == 'ok' else 'FAILED'}")

    async def run() -> str | None:
        async with open_runtime(settings, provider=StubLLMProvider() if fake else None) as runtime:

            async def ask_one(item: GoldenItem, trace: RequestTrace) -> AskResponse:
                return await runtime.pipeline.ask(
                    item.question, trace.request_id, mode=mode, trace=trace
                )

            try:
                return await run_batch(items, ask_one, policy=policy, on_result=record)
            finally:  # also when the run is interrupted: the quota it spent is still worth knowing
                if runtime.eval_llm is not None:
                    eval_stats.append(runtime.eval_llm.stats.snapshot())

    typer.echo(f"Golden set {version}: {len(items)} questions, {mode.value} mode")
    aborted: str | None = None
    try:
        aborted = _run_async(run())
    except ProviderConfigError as exc:
        raise _fail(str(exc)) from exc
    finally:
        # Also after Ctrl-C or an unexpected error: what was answered is not thrown away.
        if results:
            summary = summarize(results, total=len(items), aborted=aborted)
            write_ask_batch(
                path,
                summary,
                results,
                started_at=now,
                golden_set_version=version,
                golden_set_sha256=digest,
                mode=mode.value,
                fake=fake,
            )
            typer.echo(f"Results: {path}")
    summary = summarize(results, total=len(items), aborted=aborted)
    typer.echo(render_summary(summary))
    for stats in eval_stats:
        typer.echo(stats.render())
    if not is_complete(summary):
        raise typer.Exit(code=1)


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


# CI tells "the vectors could not be had" from every other failure by this code (Tech.md §17):
# a cold cache with no GEMINI_API_KEY, or the daily embedding quota. It is the provider's side, not
# a result about quality. 1 stays the plain failure and 2 is click's usage error (and the gate's
# "could not run"), so neither can be confused with it.
EXIT_EMBEDDINGS_UNAVAILABLE = 3


def _embeddings_unavailable(message: str) -> typer.Exit:
    typer.echo(message, err=True)
    return typer.Exit(code=EXIT_EMBEDDINGS_UNAVAILABLE)


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
            raise _embeddings_unavailable(
                f"{exc}: {count_uncached(cache, spec, chunks)} texts are not cached. "
                "Nothing was written to the database."
            ) from exc
        except EmbeddingQuotaExhaustedError as exc:
            raise _embeddings_unavailable(
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
    return asyncio.run(coro, loop_factory=loop_factory())


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
            raise _embeddings_unavailable(f"{exc}: {uncached} questions are not cached.") from exc
        except ProviderError as exc:
            message = f"Query embedding failed ({type(exc).__name__}): {exc}"
            if isinstance(exc, ProviderRateLimited) and exc.is_quota:
                raise _embeddings_unavailable(message) from exc
            raise _fail(message) from exc
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
        except (BaselineMismatchError, RetrievalEvalError) as exc:
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


class GateSuite(StrEnum):
    retrieval = "retrieval"
    generation = "generation"


def _gate_error(message: str) -> typer.Exit:
    """Exit 2: the gate could not run. Exit 1 is reserved for a real fail, so CI can tell a
    quality regression from a broken input (Tech.md §17)."""
    typer.echo(message, err=True)
    return typer.Exit(code=2)


@eval_app.command("gate")
def eval_gate(
    suite: Annotated[GateSuite, typer.Option(help="Which eval suite to gate.")],
    results: Annotated[
        Path,
        typer.Option(
            help="Results file of the run to check: the retrieval run, or promptfoo's JSON output "
            "(`promptfoo eval -o <file>.json`) for generation."
        ),
    ],
    baseline: Annotated[
        Path | None,
        typer.Option(
            help="Baseline file. Default: eval/baselines/<suite>.json.", show_default=False
        ),
    ] = None,
) -> None:
    """Compare a run with the committed baseline. Exit 0 on pass or inconclusive, 1 on fail and
    2 if the gate could not run. The Markdown report goes to stdout."""
    if suite is GateSuite.generation:
        markdown, code = _generation_gate(results, baseline or GENERATION_BASELINE)
    else:
        markdown, code = _retrieval_gate(results, baseline or RETRIEVAL_BASELINE)
    _echo_markdown(markdown)
    if code:
        raise typer.Exit(code=code)


def _echo_markdown(markdown: str) -> None:
    """Print a Markdown report as UTF-8 with LF line endings. It has ✅ ❌ Δ ≥ —: redirected, a
    Windows stdout is cp1252 and would crash on them (and would write CRLF into a pasted file)."""
    stdout: object = sys.stdout
    if isinstance(stdout, io.TextIOWrapper):
        stdout.reconfigure(encoding="utf-8", newline="\n")
    typer.echo(markdown, nl=False)


def _retrieval_gate(results: Path, baseline_path: Path) -> tuple[str, int]:
    try:
        run = read_run(results)
    except (OSError, ValueError) as exc:
        raise _gate_error(f"Cannot read the results file {results}: {exc}") from exc
    try:
        rows = read_baseline(baseline_path)
    except OSError as exc:
        raise _gate_error(f"Cannot read the baseline {baseline_path}: {exc}") from exc
    except ValueError as exc:
        raise _gate_error(
            f"Invalid baseline {baseline_path}: {exc}\n"
            "A row written before the golden-set hash and git state became required has to be "
            "regenerated in a baseline PR."
        ) from exc
    report = evaluate_gate(run, rows)
    return render_markdown(report), exit_code(report)


def _generation_gate(results: Path, baseline_path: Path) -> tuple[str, int]:
    """promptfoo's own exit code is ignored (Tech.md §15.5): the gate reads its output file."""
    try:
        run = read_generation_results(results)
    except (OSError, ValueError) as exc:
        raise _gate_error(f"Cannot read the results file {results}: {exc}") from exc
    try:
        rows = read_generation_baseline(baseline_path)
    except OSError as exc:
        raise _gate_error(f"Cannot read the baseline {baseline_path}: {exc}") from exc
    except ValueError as exc:
        raise _gate_error(
            f"Invalid baseline {baseline_path}: {exc}\n"
            "A row missing a required field has to be regenerated in a baseline PR."
        ) from exc
    try:
        report = evaluate_generation_gate(run, rows)
    except GateCannotRunError as exc:
        raise _gate_error(
            "The gate could not run:\n" + "\n".join(f"- {r}" for r in exc.reasons)
        ) from exc
    return render_generation_markdown(report), exit_code(report)


class BaselineSuite(StrEnum):
    generation = "generation"  # retrieval rows are written by `eval retrieval --write-baseline`


@eval_app.command("baseline")
def eval_baseline(
    suite: Annotated[
        BaselineSuite,
        typer.Option(help="Which baseline to write. Retrieval: `eval retrieval --write-baseline`."),
    ],
    results: Annotated[
        Path,
        typer.Option(help="promptfoo's JSON output of the run (`promptfoo eval -o <file>.json`)."),
    ],
    baseline: Annotated[
        Path | None,
        typer.Option(
            help="Baseline file. Default: eval/baselines/generation.json.", show_default=False
        ),
    ] = None,
) -> None:
    """Copy the rows of a generation run into the baseline file (other configs' rows are kept).

    Refuses, with the file untouched and exit 1, a run that is inconclusive or has a provider or
    judge error, rows from another setup than the kept ones, and an unknown git state (Tech.md
    §15.7). A new hybrid row gets the initial thresholds of Tech.md §15.5: they need the Author's
    approval in the baseline PR. Numbers are never typed by hand (AGENTS.md §7).
    """
    del suite  # generation is the only suite; the option keeps `eval baseline` unambiguous
    path = baseline or GENERATION_BASELINE
    git_sha, git_dirty = repo_state()
    try:
        run = read_generation_results(results, git_sha=git_sha, git_dirty=git_dirty)
    except (OSError, ValueError) as exc:
        raise _fail(f"Cannot read the results file {results}: {exc}") from exc
    if git_dirty:
        typer.echo("Warning: uncommitted changes; the baseline's git_sha isn't the code that ran.")
    try:
        update = update_generation_baseline(path, run)
    except GenerationBaselineError as exc:
        raise _fail(f"Baseline not updated: {exc}\nThe baseline file was not changed.") from exc
    except OSError as exc:
        raise _fail(f"Cannot read or write the baseline {path}: {exc}") from exc

    typer.echo(f"Baseline updated: {path} ({', '.join(update.written)})")
    for name in update.written:
        row = update.rows[name]
        scores = ", ".join(f"{m} {v:.3f} (n={row.n[m]})" for m, v in sorted(row.metrics.items()))
        typer.echo(f"  {name}: {row.cases} cases; {scores}")
    for name in update.seeded:
        typer.echo(
            f"  {name}: thresholds are the initial table of Tech.md §15.5, written for the first "
            "time: the Author approves them in the baseline PR."
        )
    check = check_against_itself(run, update)
    if check is not None:
        typer.echo("")
        _echo_markdown(render_generation_markdown(check))
        if check.status != "pass":
            typer.echo(
                "Warning: the new baseline does not pass its own gate on the run it was made "
                "from. The file is written; decide the thresholds before the baseline PR."
            )


class RecordSuite(StrEnum):
    generation = "generation"  # the only suite CI records on main (the retrieval eval is not)


@eval_app.command("record")
def eval_record(
    suite: Annotated[RecordSuite, typer.Option(help="Which suite's run to record.")],
    results: Annotated[
        Path,
        typer.Option(help="promptfoo's JSON output of the run (`promptfoo eval -o <file>.json`)."),
    ],
    branch: Annotated[str, typer.Option(help="Branch the run was made on (CI: the ref name).")],
    report_url: Annotated[
        str, typer.Option(help="Link to the CI run (or its artifact), stored as report_url.")
    ],
    baseline: Annotated[
        Path | None,
        typer.Option(
            help="Baseline file. Default: eval/baselines/generation.json.", show_default=False
        ),
    ] = None,
    write: Annotated[
        bool,
        typer.Option(
            "--write",
            help="Insert the rows into eval_runs of DATABASE_URL_DIRECT. Without it: a dry run "
            "that prints the rows and writes nothing.",
        ),
    ] = False,
) -> None:
    """Record a generation run in `eval_runs`, one row per config (DB.md §4, aggregates only).

    CI runs this on a push to main only, behind the repository variable EVAL_RECORD_RUNS, with the
    owner connection DATABASE_URL_DIRECT; the index hash is read from DATABASE_URL, the database
    the eval ran against. Nothing is written without --write. Exit 1: not recorded; exit 2: the
    gate could not run on these results (Tech.md §15.7).
    """
    del suite  # generation is the only suite; the option keeps `eval record` unambiguous
    settings = get_settings()
    git_sha, git_dirty = repo_state()
    if git_sha is None or git_dirty is None:
        raise _fail("Not recorded: this is not a git checkout, so the row has no commit to name.")
    if git_dirty:
        raise _fail("Not recorded: tracked files are modified, so HEAD is not the code that ran.")
    try:
        run = read_generation_results(results, git_sha=git_sha, git_dirty=git_dirty)
    except (OSError, ValueError) as exc:
        raise _fail(f"Cannot read the results file {results}: {exc}") from exc
    try:
        rows = read_generation_baseline(baseline or GENERATION_BASELINE)
    except (OSError, ValueError) as exc:
        raise _fail(f"Cannot read the baseline: {exc}") from exc

    source = settings.database_url.get_secret_value()
    try:
        active = read_active_index(source)
        records = build_rows(
            run, rows, git_sha=git_sha, branch=branch, report_url=report_url, active_index=active
        )
    except GateCannotRunError as exc:
        raise _gate_error(
            "The gate could not run:\n" + "\n".join(f"- {r}" for r in exc.reasons)
        ) from exc
    except (EvalRecordError, psycopg.Error) as exc:
        raise _fail(f"Not recorded: {exc}") from exc

    typer.echo(
        f"Run of {git_sha[:7]} on {branch}; index {active.label} (from {_describe_target(source)})"
    )
    for record in records:
        typer.echo(_describe_record(record))
    if not write:
        typer.echo("Dry run: nothing was written. Pass --write to insert these rows.")
        return
    if settings.database_url_direct is None:
        raise _fail(
            "Not recorded: DATABASE_URL_DIRECT is not set (eval_runs needs the owner role)."
        )
    target = settings.database_url_direct.get_secret_value()
    try:
        ids = insert_rows(target, records)
    except (EvalRecordError, psycopg.Error) as exc:
        raise _fail(f"Not recorded, nothing was inserted: {exc}") from exc
    typer.echo(f"Inserted {len(ids)} rows into eval_runs at {_describe_target(target)}.")


def _describe_record(record: EvalRunRow) -> str:
    """One line per row: what is about to be (or was) written, without any text of the eval."""
    return (
        f"  {record.config_name}: {record.status}, {record.case_count} cases "
        f"({record.errored_case_count} provider errors), index {record.index_config_hash[:8]}, "
        f"{len(record.metrics)} metric entries"
    )


@eval_app.command("report")
def eval_report(
    retrieval: Annotated[
        Path,
        typer.Option(
            help="Retrieval baseline. Default: eval/baselines/retrieval.json.", show_default=False
        ),
    ] = RETRIEVAL_BASELINE,
    generation: Annotated[
        Path,
        typer.Option(
            help="Generation baseline. Default: eval/baselines/generation.json.",
            show_default=False,
        ),
    ] = GENERATION_BASELINE,
) -> None:
    """Print the eval tables of the README (retrieval ablation, generation) as Markdown, from the
    committed baseline files only. A baseline that does not exist yet is said so, not an error."""
    try:
        report = build_report(retrieval, generation)
    except (OSError, ValueError) as exc:
        raise _fail(f"Cannot read the baselines: {exc}") from exc
    _echo_markdown(report)


@eval_app.command("agreement")
def eval_agreement(
    labels: Annotated[
        Path,
        typer.Option(
            help="The labeling sheet with the human_label column filled in.", show_default=False
        ),
    ] = DEFAULT_SHEET,
    key: Annotated[
        Path,
        typer.Option(
            help="The key of the sample (written by `eval export-verdicts`).", show_default=False
        ),
    ] = DEFAULT_KEY,
) -> None:
    """Judge-human agreement (Tech.md §15.4): exact agreement, Cohen's kappa and the confusion
    matrices, overall, per kind, for the real items and for the controls, as Markdown with `n`.

    Refuses (exit 1) a sheet with a missing, invalid or unknown label and lists every item at
    fault. The output states whether the PRD §8 target (0.8) is met and decides nothing else.
    """
    try:
        sheet = parse_sheet(labels.read_bytes())
    except (OSError, SheetError) as exc:
        raise _fail(f"Cannot read the labels {labels}: {exc}") from exc
    try:
        sample = AgreementKey.model_validate_json(key.read_bytes())
    except (OSError, ValidationError) as exc:
        raise _fail(f"Cannot read the key {key}: {exc}") from exc
    try:
        ratings = rate_items(sheet, sample)
    except LabelError as exc:
        problems = "\n".join(f"- {p}" for p in exc.problems)
        raise _fail(f"Not computed, the labels do not match the key:\n{problems}") from exc
    _echo_markdown(render_report(ratings, sample))
