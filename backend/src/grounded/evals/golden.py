"""Golden-set helpers (Tech.md §15.1): load and check the JSONL, sample sections to draft questions
from, list a page's sections for labeling, and resolve labels against chunks.

Labels resolve with the metrics' own rule (``metrics.section_matches``), against the chunks of the
pinned corpus (offline, no index needed) or of the active index. The same checks therefore catch
what would silently break a metric: a label no chunk matches (a typo, or a small section the
chunker merged into its previous sibling), and two labels matching one chunk (nested labels,
which ``label_ranks`` rejects).
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import psycopg
from pydantic import ValidationError

from grounded.evals.metrics import ChunkRef, section_matches
from grounded.ingest.types import Chunk
from grounded.schemas.eval import GoldenItem, GoldenType

# eval/golden at the repo root (this file is backend/src/grounded/evals/golden.py).
GOLDEN_DIR: Final = Path(__file__).resolve().parents[4] / "eval" / "golden"
DOCS_PREFIX: Final = "docs/en/docs/"

# Composition the curated set aims for (PRD §12). Reported, not enforced: curation may shift it.
TARGET_TYPE_COUNTS: Final[dict[GoldenType, int]] = {
    "factual": 8,
    "how_to": 8,
    "code": 5,
    "multi_section": 4,
    "unanswerable": 5,
}


class GoldenSetError(Exception):
    """The golden-set file can't be used; the message lists every problem found."""


def load_golden_set(path: Path) -> list[GoldenItem]:
    """Parse and validate every line (blank lines are skipped). All problems are collected and
    reported together as ``<file>:<line>: <problem>``, so one run shows everything to fix."""
    items: list[GoldenItem] = []
    problems: list[str] = []
    first_line: dict[str, int] = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = GoldenItem.model_validate_json(line)
        except ValidationError as exc:
            for error in exc.errors():
                where = ".".join(str(part) for part in error["loc"]) or "item"
                problems.append(f"{path.name}:{lineno}: {where}: {error['msg']}")
            continue
        if item.id in first_line:
            first = first_line[item.id]
            problems.append(f"{path.name}:{lineno}: duplicate id {item.id} (first on line {first})")
            continue
        first_line[item.id] = lineno
        items.append(item)
    if problems:
        raise GoldenSetError("\n".join(problems))
    if not items:
        raise GoldenSetError(f"{path.name}: no items")
    return items


def type_counts(items: Iterable[GoldenItem]) -> dict[GoldenType, int]:
    counts = Counter(item.type for item in items)
    return {kind: counts.get(kind, 0) for kind in TARGET_TYPE_COUNTS}


def normalize_page(page: str) -> str:
    """Accept ``tutorial/x.md`` as well as ``docs/en/docs/tutorial/x.md``."""
    page = page.replace("\\", "/")
    return page if page.startswith(DOCS_PREFIX) else DOCS_PREFIX + page.lstrip("/")


# --- Sampling and section listing --------------------------------------------------------------


def sample_sections(chunks: Sequence[Chunk], n: int, seed: int) -> list[str]:
    """``n`` distinct H2/H3 ``section_id``s, drawn with a seeded RNG from the sorted list, so the
    same corpus, chunking and seed always give the same sample. Page intros are left out: they
    are mostly navigation text, and questions about them are rarely what users ask."""
    sections = sorted({chunk.section_id for chunk in chunks if chunk.anchor_path})
    if n > len(sections):
        raise ValueError(f"asked for {n} sections, the corpus has {len(sections)}")
    return random.Random(seed).sample(sections, n)


@dataclass(frozen=True, slots=True)
class SectionInfo:
    section_id: str
    heading_level: int
    breadcrumb_text: str
    chunk_count: int
    token_count: int


def page_sections(chunks: Sequence[Chunk], page: str) -> list[SectionInfo]:
    """The sections of one page as the chunker produced them, in document order.

    These are the labels that can match: a section merged into its previous sibling doesn't
    appear under its own anchor, and labeling it would match nothing.
    """
    grouped: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        if chunk.section_id.partition("#")[0] == page:
            grouped[chunk.section_id].append(chunk)  # dicts keep first-seen (document) order
    return [
        SectionInfo(
            section_id=section_id,
            heading_level=parts[0].heading_level,
            breadcrumb_text=parts[0].breadcrumb_text,
            chunk_count=len(parts),
            token_count=sum(part.token_count for part in parts),
        )
        for section_id, parts in grouped.items()
    ]


# --- Label resolution --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IndexedChunk:
    """A chunk as stored in the index: all label resolution needs."""

    section_id: str
    anchor_path: tuple[str, ...]


def resolve_labels(items: Sequence[GoldenItem], chunks: Sequence[ChunkRef]) -> list[str]:
    """Problems with the items' labels against ``chunks`` (empty list = all labels resolve).

    - a label that matches no chunk (with a hint whether the page itself exists);
    - two labels of one item that match the same chunk (nested labels, Tech.md §15.1).
    """
    by_page: dict[str, list[ChunkRef]] = defaultdict(list)
    for chunk in chunks:
        by_page[chunk.section_id.partition("#")[0]].append(chunk)

    problems: list[str] = []
    for item in items:
        labels = [s.section for s in item.relevant_sections]
        for label in labels:
            page = label.partition("#")[0]
            if not any(section_matches(label, page, c.anchor_path) for c in by_page[page]):
                hint = (
                    "the anchor is not a section of its own (typo, or merged into the previous "
                    "sibling: label that one or the parent)"
                    if by_page[page]
                    else "no such page in the corpus"
                )
                problems.append(f"{item.id}: {label} matches no chunk: {hint}")
        reported: set[tuple[str, ...]] = set()
        for page in dict.fromkeys(label.partition("#")[0] for label in labels):
            for chunk in by_page[page]:
                both = tuple(x for x in labels if section_matches(x, page, chunk.anchor_path))
                if len(both) > 1 and both not in reported:
                    reported.add(both)
                    problems.append(
                        f"{item.id}: {' and '.join(both)} all match chunk {chunk.section_id} "
                        "(nested labels: keep one)"
                    )
    return problems


_ACTIVE_INDEX_CHUNKS = """
SELECT v.id, c.section_id, c.anchor_path
FROM chunks c
JOIN index_versions v ON v.id = c.index_version_id
WHERE v.is_active
ORDER BY c.id
"""


def active_index_chunks(conninfo: str) -> tuple[int, list[IndexedChunk]]:
    """The active index version's id and the location of each of its chunks."""
    with psycopg.connect(conninfo) as conn:
        rows: list[tuple[Any, ...]] = conn.execute(_ACTIVE_INDEX_CHUNKS).fetchall()
    if not rows:
        raise GoldenSetError(
            "no active index version with chunks: run `grounded ingest --activate`"
        )
    return int(rows[0][0]), [IndexedChunk(str(r[1]), tuple(r[2])) for r in rows]
