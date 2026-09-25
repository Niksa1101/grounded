"""Corpus fetch (against local git repos, no network) and page discovery (against corpus_mini)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from grounded.ingest import corpus
from grounded.ingest.corpus import (
    CorpusError,
    discover_pages,
    fetch_corpus,
    is_excluded,
    page_url,
    read_title,
)

CORPUS_MINI = Path(__file__).resolve().parents[1] / "fixtures" / "corpus_mini"


# --- URL mapping and exclusions ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("docs_relative_path", "url"),
    [
        ("index.md", "https://fastapi.tiangolo.com/"),
        ("async.md", "https://fastapi.tiangolo.com/async/"),
        ("tutorial/index.md", "https://fastapi.tiangolo.com/tutorial/"),
        ("tutorial/first-steps.md", "https://fastapi.tiangolo.com/tutorial/first-steps/"),
        ("tutorial/security/index.md", "https://fastapi.tiangolo.com/tutorial/security/"),
        (
            "tutorial/security/first-steps.md",
            "https://fastapi.tiangolo.com/tutorial/security/first-steps/",
        ),
        # Only a whole "index" segment is special.
        ("how-to/reindex.md", "https://fastapi.tiangolo.com/how-to/reindex/"),
    ],
)
def test_page_url(docs_relative_path: str, url: str) -> None:
    assert page_url(docs_relative_path) == url


@pytest.mark.parametrize(
    ("docs_relative_path", "excluded"),
    [
        ("release-notes.md", True),
        ("external-links.md", True),
        ("fastapi-people.md", True),
        ("_llm-test.md", True),
        ("reference/fastapi.md", True),
        ("reference/openapi/docs.md", True),
        ("js/notes.md", True),
        ("index.md", False),
        ("tutorial/index.md", False),
        # Patterns are anchored at the docs root, not matched anywhere in the path.
        ("tutorial/release-notes.md", False),
        ("advanced/reference-guide.md", False),
    ],
)
def test_exclusions(docs_relative_path: str, excluded: bool) -> None:
    assert is_excluded(docs_relative_path) is excluded


# --- Discovery --------------------------------------------------------------------------------


def test_discover_pages_selects_english_docs_in_path_order() -> None:
    pages = discover_pages(CORPUS_MINI)
    assert [p.source_path for p in pages] == [
        "docs/en/docs/advanced/settings.md",
        "docs/en/docs/index.md",
        "docs/en/docs/tutorial/background-tasks.md",
        "docs/en/docs/tutorial/index.md",
        "docs/en/docs/tutorial/security/first-steps.md",
        "docs/en/docs/tutorial/security/index.md",
    ]


def test_discover_pages_maps_urls_and_nav_names() -> None:
    by_path = {p.source_path.removeprefix("docs/en/docs/"): p for p in discover_pages(CORPUS_MINI)}

    assert by_path["index.md"].url == "https://fastapi.tiangolo.com/"
    assert by_path["index.md"].nav_path == ()
    # A section's own index page is the section, so its nav stops at the parent.
    assert by_path["tutorial/index.md"].nav_path == ()
    assert by_path["tutorial/background-tasks.md"].nav_path == ("Tutorial - User Guide",)
    assert by_path["tutorial/security/index.md"].nav_path == ("Tutorial - User Guide",)
    assert by_path["tutorial/security/first-steps.md"].nav_path == (
        "Tutorial - User Guide",
        "Security",
    )
    assert by_path["tutorial/security/first-steps.md"].url == (
        "https://fastapi.tiangolo.com/tutorial/security/first-steps/"
    )


def test_section_without_index_falls_back_to_directory_name(
    caplog: pytest.LogCaptureFixture,
) -> None:
    by_path = {p.source_path: p for p in discover_pages(CORPUS_MINI)}
    assert by_path["docs/en/docs/advanced/settings.md"].nav_path == ("Advanced",)
    assert "section has no index.md title" in caplog.text


def test_discover_pages_is_deterministic() -> None:
    assert discover_pages(CORPUS_MINI) == discover_pages(CORPUS_MINI)


def test_discover_pages_requires_docs_dir(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match="docs/en/docs"):
        discover_pages(tmp_path)


def test_read_title_skips_front_matter_and_code_fences() -> None:
    docs = CORPUS_MINI / "docs" / "en" / "docs"
    assert read_title(docs / "index.md") == "FastAPI"
    assert read_title(docs / "tutorial" / "security" / "index.md") == "Security"


def test_read_title_none_without_h1(tmp_path: Path) -> None:
    page = tmp_path / "page.md"
    page.write_text("Just text.\n\n## Only an H2\n", encoding="utf-8")
    assert read_title(page) is None


# --- Fetch (local repositories only) ----------------------------------------------------------


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


@pytest.fixture
def origin(tmp_path: Path) -> tuple[str, str]:
    """A local repo with tag ``1.0.0`` (annotated) and branch ``main``; returns (url, tag sha)."""
    repo = tmp_path / "origin"
    (repo / "docs" / "en" / "docs").mkdir(parents=True)
    (repo / "docs" / "en" / "docs" / "index.md").write_text("# Home\n", encoding="utf-8")
    _git(tmp_path, "init", "--quiet", "--initial-branch=main", str(repo))
    _git(repo, "add", ".")
    _git(repo, "commit", "--quiet", "-m", "first")
    _git(repo, "tag", "-a", "1.0.0", "-m", "release")
    sha = _git(repo, "rev-parse", "HEAD")
    # A later commit on main: the tag, not the branch tip, must be checked out.
    (repo / "docs" / "en" / "docs" / "later.md").write_text("# Later\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "--quiet", "-m", "second")
    # --depth needs a URL, not a plain path, for git to honor it on a local clone.
    return repo.as_uri(), sha


def test_fetch_clones_tag_and_records_sha(tmp_path: Path, origin: tuple[str, str]) -> None:
    url, sha = origin
    checkout = fetch_corpus("1.0.0", tmp_path / "cache", repo_url=url)

    assert checkout.ref == "1.0.0"
    assert checkout.sha == sha
    assert checkout.path == tmp_path / "cache" / "corpus" / "1.0.0"
    assert (checkout.path / "docs" / "en" / "docs" / "index.md").is_file()
    assert not (checkout.path / "docs" / "en" / "docs" / "later.md").exists()


def test_fetch_reuses_clean_checkout_without_cloning(
    tmp_path: Path, origin: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    url, sha = origin
    fetch_corpus("1.0.0", tmp_path / "cache", repo_url=url)

    calls: list[tuple[str, ...]] = []
    real_git = corpus._git  # pyright: ignore[reportPrivateUsage]

    def spy(*args: str, cwd: Path | None = None) -> str:
        calls.append(args)
        return real_git(*args, cwd=cwd)

    monkeypatch.setattr(corpus, "_git", spy)
    again = fetch_corpus("1.0.0", tmp_path / "cache", repo_url=url)

    assert again.sha == sha
    assert not any(args[0] in {"clone", "ls-remote"} for args in calls)


def test_fetch_keeps_line_endings_as_committed(tmp_path: Path, origin: tuple[str, str]) -> None:
    # A global core.autocrlf=true would otherwise turn LF into CRLF on Windows checkouts.
    url, _ = origin
    checkout = fetch_corpus("1.0.0", tmp_path / "cache", repo_url=url)
    assert _git(checkout.path, "config", "core.autocrlf") == "false"
    assert b"\r\n" not in (checkout.path / "docs" / "en" / "docs" / "index.md").read_bytes()


def test_fetch_refuses_a_branch(tmp_path: Path, origin: tuple[str, str]) -> None:
    url, _ = origin
    with pytest.raises(CorpusError, match="not a tag"):
        fetch_corpus("main", tmp_path / "cache", repo_url=url)
    assert not (tmp_path / "cache" / "corpus" / "main").exists()


def test_fetch_refuses_unknown_tag(tmp_path: Path, origin: tuple[str, str]) -> None:
    url, _ = origin
    with pytest.raises(CorpusError, match="not a tag"):
        fetch_corpus("9.9.9", tmp_path / "cache", repo_url=url)


@pytest.mark.parametrize("ref", ["", "../escape", "-uoption", "a/b", "1.0 0"])
def test_fetch_rejects_unsafe_ref_names(tmp_path: Path, ref: str) -> None:
    with pytest.raises(CorpusError, match="invalid tag name"):
        fetch_corpus(ref, tmp_path / "cache", repo_url="file:///nowhere")


def test_fetch_refuses_dirty_checkout(tmp_path: Path, origin: tuple[str, str]) -> None:
    url, _ = origin
    checkout = fetch_corpus("1.0.0", tmp_path / "cache", repo_url=url)
    (checkout.path / "docs" / "en" / "docs" / "stray.md").write_text("# Stray\n", encoding="utf-8")

    with pytest.raises(CorpusError, match="local changes"):
        fetch_corpus("1.0.0", tmp_path / "cache", repo_url=url)


def test_fetch_refuses_directory_that_is_not_the_tag(tmp_path: Path) -> None:
    (tmp_path / "cache" / "corpus" / "1.0.0").mkdir(parents=True)
    with pytest.raises(CorpusError, match=r"not a checkout of tag 1\.0\.0"):
        fetch_corpus("1.0.0", tmp_path / "cache", repo_url="file:///nowhere")
