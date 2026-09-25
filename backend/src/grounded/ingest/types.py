"""Value objects passed between ingestion stages.

Frozen and slotted: every stage is a pure-ish function of the previous stage's output, and ingest
determinism (same tag + config → same index) depends on nobody mutating them along the way.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CorpusCheckout:
    """A local checkout of the docs repository at one pinned tag."""

    path: Path
    ref: str  # the tag, e.g. "0.141.1"
    sha: str  # resolved commit SHA (40 hex), recorded per index version


@dataclass(frozen=True, slots=True)
class PageRef:
    """One docs page selected for indexing, before it is read or parsed."""

    source_path: str  # repo-relative POSIX path: "docs/en/docs/tutorial/background-tasks.md"
    url: str  # canonical page URL: "https://fastapi.tiangolo.com/tutorial/background-tasks/"
    # Section names of the enclosing directories, outermost first: ("Tutorial - User Guide",).
    # Empty for top-level pages. The first breadcrumb levels of every chunk on the page.
    nav_path: tuple[str, ...]
