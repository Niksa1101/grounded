"""The Vercel deployment assumptions of 5.08 (Tech §18), checked without Vercel.

Vercel itself is not available to the tests (no CLI, no preview deploy), so what can break a bundle
is pinned here instead: the entry module and the two config files agree with each other, the
`excludeFiles` pattern keeps everything the request path reads, the app starts without the
installed package's metadata, and nothing the request path imports needs the repo layout (the
golden set, the migrations directory).
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import subprocess
import sys
import tomllib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from grounded import main as main_module
from grounded import runtime as runtime_module
from grounded.generation.providers.fake import StubLLMProvider
from grounded.main import create_app
from grounded.settings import Settings, _repo_root  # pyright: ignore[reportPrivateUsage]
from tests.support import make_settings

BACKEND = Path(__file__).resolve().parents[2]
ROOT = BACKEND.parent


def pyproject() -> dict[str, Any]:
    return tomllib.loads((BACKEND / "pyproject.toml").read_text(encoding="utf-8"))


def vercel_json() -> dict[str, Any]:
    return json.loads((BACKEND / "vercel.json").read_text(encoding="utf-8"))


def entry_file() -> Path:
    """The file `[tool.vercel] entrypoint` names: `a.b:app` is `a/b.py` (Vercel's Python docs)."""
    module, _, variable = pyproject()["tool"]["vercel"]["entrypoint"].partition(":")
    assert variable == "app"
    return BACKEND / (module.replace(".", "/") + ".py")


def expand_braces(pattern: str) -> Iterator[str]:
    """`{a,b/**}` -> `a`, `b/**` (one level, which is all vercel.json uses)."""
    match = re.fullmatch(r"\{(.*)\}", pattern)
    yield from (match.group(1).split(",") if match else [pattern])


def excluded(relative: str, pattern: str) -> bool:
    # fnmatch's `*` also crosses `/`, which over-matches `*`: the safe side for "is it kept?".
    return any(fnmatch.fnmatchcase(relative, alt) for alt in expand_braces(pattern))


# --- the config files agree ----------------------------------------------------------------------


def test_the_entrypoint_names_a_file_that_exports_app_and_vercel_json_keys_on_it() -> None:
    entry = entry_file()
    assert entry.is_file()
    assert re.search(r"^app = create_app\(\)$", entry.read_text(encoding="utf-8"), re.MULTILINE)

    config = vercel_json()
    assert list(config["functions"]) == [entry.relative_to(BACKEND).as_posix()]
    assert config["regions"] == ["cle1"]  # D53: the same AWS region as Neon


def test_max_duration_leaves_room_between_the_deadline_and_the_proxy_timeout() -> None:
    # The chain of Tech §18: deadline 25 < maxDuration 30 < proxy fetch timeout 35.
    max_duration = vercel_json()["functions"]["app.py"]["maxDuration"]
    deadline = Settings.model_fields["request_deadline_s"].default
    assert deadline < max_duration == 30
    proxy_timeout_s = 35  # frontend/app/api/ask/route.ts, kept in step by hand (Tech §18)
    assert max_duration < proxy_timeout_s


def test_exclude_files_trims_the_tests_but_keeps_what_the_request_path_reads() -> None:
    pattern = vercel_json()["functions"]["app.py"]["excludeFiles"]
    for gone in (
        "tests/unit/test_vercel_bundle.py",
        "src/grounded/__pycache__/main.cpython-312.pyc",
    ):
        assert excluded(gone, pattern), gone
    kept = [
        "app.py",
        "pyproject.toml",
        "pricing.toml",
        "src/grounded/main.py",
        "src/grounded/api/routes_ask.py",
        *(
            p.relative_to(BACKEND).as_posix()
            for p in (BACKEND / "prompts").rglob("*")
            if p.is_file()
        ),
    ]
    assert any(k.startswith("prompts/") for k in kept)
    assert not [k for k in kept if excluded(k, pattern)]


def test_the_paths_the_request_path_reads_are_inside_the_backend_directory() -> None:
    # prompts/ and pricing.toml are found relative to their module (parents[3] of
    # src/grounded/<pkg>/<module>.py), so they only resolve in a bundle that keeps this layout.
    from grounded.generation.prompts import DEFAULT_PROMPTS_DIR
    from grounded.observability.cost import DEFAULT_PRICING_PATH

    assert DEFAULT_PROMPTS_DIR == BACKEND / "prompts"
    assert DEFAULT_PROMPTS_DIR.is_dir()
    assert DEFAULT_PRICING_PATH == BACKEND / "pricing.toml"
    assert DEFAULT_PRICING_PATH.is_file()


# --- the app starts in a bundle ------------------------------------------------------------------

_ENTRY_PROBE = """
import asyncio, importlib.metadata as metadata, importlib.util, json, sys

def _missing(name):
    raise metadata.PackageNotFoundError(name)

metadata.version = _missing  # a bundle without the installed package's metadata

spec = importlib.util.spec_from_file_location("vercel_entry", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

import grounded, httpx

async def healthz():
    transport = httpx.ASGITransport(app=module.app)  # no lifespan: the pool is not opened
    async with httpx.AsyncClient(transport=transport, base_url="http://bundle") as client:
        return (await client.get("/healthz")).json()

print(json.dumps({
    "version": module.app.version,
    "grounded_file": grounded.__file__,
    "src_first": sys.path[0],
    "health": asyncio.run(healthz()),
    "layout_modules": sorted(
        name for name in sys.modules
        if name in ("grounded.evals.golden", "grounded.infra.migrations", "grounded.cli")
    ),
}))
"""


def test_the_entry_module_serves_without_package_metadata_or_the_repo_layout(
    tmp_path: Path,
) -> None:
    done = subprocess.run(
        [sys.executable, "-I", "-c", _ENTRY_PROBE, str(entry_file())],
        capture_output=True,
        text=True,
        cwd=tmp_path,  # not the repo: nothing may depend on the working directory
        env={**os.environ, "APP_ENV": "test"},
        timeout=60,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    report = json.loads(done.stdout.splitlines()[-1])
    assert report["version"] == "0+unknown"  # importlib.metadata.version failed, soft
    assert report["health"] == {"status": "ok"}
    # The package in use is this bundle's source tree: it is what finds prompts/ and pricing.toml.
    assert Path(report["grounded_file"]).resolve().is_relative_to(BACKEND / "src")
    assert Path(report["src_first"]).resolve() == BACKEND / "src"
    # GOLDEN_DIR (evals/golden.py) and DEFAULT_MIGRATIONS_DIR (infra/migrations.py) are repo-layout
    # paths of the CLI and the evals; the app never imports the modules that hold them.
    assert report["layout_modules"] == []


def test_the_app_builds_when_the_package_metadata_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(name: str) -> str:
        raise main_module.PackageNotFoundError(name)

    monkeypatch.setattr(main_module, "version", missing)
    assert create_app(make_settings()).version == "0+unknown"


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_the_repo_root_of_a_shallow_bundle_is_a_directory_not_an_index_error(depth: int) -> None:
    # `parents[3]` raises for /a/grounded/settings.py; the root is only used for .env and a
    # relative CACHE_DIR, so a neighbouring directory is a fine stand-in.
    path = Path("/".join(["/x"] * (depth - 1) + ["grounded", "settings.py"])).resolve()
    assert _repo_root(path) in path.parents


# --- shutdown and the production env -------------------------------------------------------------


async def test_the_pool_is_closed_with_a_timeout_inside_the_platforms_500_ms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from psycopg_pool import AsyncConnectionPool

    seen: list[float] = []
    real_close = AsyncConnectionPool.close

    async def recording_close(self: AsyncConnectionPool, *args: float) -> None:
        seen.extend(args)
        await real_close(self, *args)

    monkeypatch.setattr(AsyncConnectionPool, "close", recording_close)
    settings = make_settings(
        database_url="postgresql://grounded:grounded@127.0.0.1:1/grounded", db_pool_min_size=0
    )
    async with runtime_module.open_runtime(settings, provider=StubLLMProvider()):
        pass
    assert seen == [runtime_module.POOL_CLOSE_TIMEOUT_S]
    assert runtime_module.POOL_CLOSE_TIMEOUT_S < 0.5


def test_the_production_env_of_tech_18_covers_what_prod_startup_demands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = (ROOT / "docs" / "Tech.md").read_text(encoding="utf-8")
    [line] = [ln for ln in text.splitlines() if ln.startswith("- Production env:")]
    documented = set(re.findall(r"`([A-Z][A-Z0-9_]+)(?:=[^`]*)?`", line))
    # Settings._check_prod refuses to start without these ...
    required = {"DATABASE_URL", "PROXY_SHARED_SECRET", "IP_HASH_SECRET", "DAILY_LLM_BUDGET"}
    # ... and open_runtime builds the Gemini adapter (the first of GENERATOR_PROVIDERS), which
    # needs a key and a pinned model.
    required |= {"APP_ENV", "GEMINI_API_KEY", "GEMINI_MODEL"}
    assert required <= documented, sorted(required - documented)

    # The CI job sets DATABASE_URL in the environment, which would fill a field the test removes.
    for var in ("DATABASE_URL", "PROXY_SHARED_SECRET", "IP_HASH_SECRET", "DAILY_LLM_BUDGET"):
        monkeypatch.delenv(var, raising=False)
    env = {
        "app_env": "prod",
        "database_url": "postgresql://u:p@db.example/app",
        "proxy_shared_secret": "s" * 32,
        "ip_hash_secret": "h" * 32,
        "daily_llm_budget": 300,
    }
    Settings(_env_file=None, **env)  # pyright: ignore[reportCallIssue]
    for name in ("database_url", "proxy_shared_secret", "ip_hash_secret", "daily_llm_budget"):
        with pytest.raises(ValueError, match=name.upper()):
            Settings(_env_file=None, **{k: v for k, v in env.items() if k != name})  # pyright: ignore[reportCallIssue]
