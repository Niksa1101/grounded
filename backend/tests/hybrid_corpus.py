"""A hand-made index with vectors *and* words for the hybrid retrieval spec (tickets 2.05, 2.06).

Every chunk sits at a known angle from the query vector (``unit_vector(0)``) in one plane, so the
dense order is the order of ``step`` ascending. Some chunks also repeat a rare word: ``quokka`` or
``wombat``. A single-term question ranks chunks by how often they repeat the word, so the lexical
order is known too, and the two lists overlap only partly (that is what the full outer join is for):

    key          step  dense rank   word          lexical rank ("quokka")
    q_top           0           1   quokka x5     1
    f1              1           2   -             -
    q_third         2           3   quokka x3     3
    f2              3           4   -             -
    f3              4           5   -             -
    f4              5           6   -             -
    q_fourth        6           7   quokka x2     4
    f5              7           8   -             -
    f6              8           9   -             -
    q_fifth        12          10   quokka x1     5
    w_top          30          11   wombat x2     - (rank 1 for "wombat")
    w_second       31          12   wombat x1     - (rank 2 for "wombat")
    q_second       40          13   quokka x4     2   (far from the query, second best lexically)

Insert order is ids ascending, and it is deliberate. Where two chunks tie on the fused score, one
of them comes from the lexical list only and the other from the dense list only, and the id decides:
the lower id is the lexical-only chunk in some ties and the dense-only chunk in others, so a
tie-break that prefers one list over the other would show.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import psycopg

from tests.support import DocumentRow, insert_chunk, insert_document, unit_vector

EMBEDDING_DIM = len(unit_vector())
PAGE = ("docs/en/docs/hybrid.md", "https://fastapi.tiangolo.com/hybrid/", "Hybrid")
RADIANS_PER_STEP = 0.05  # 40 steps stay below pi, where cosine distance is still monotonic


class Spec(NamedTuple):
    key: str
    step: int  # angle from the query vector, in units of RADIANS_PER_STEP
    word: str | None  # the rare word the content repeats
    count: int  # how many times

    @property
    def content(self) -> str:
        if self.word is None:
            return "Plain filler paragraph about nothing in particular."
        return " ".join([self.word] * self.count) + " sleeps."


CORPUS: tuple[Spec, ...] = (
    Spec("w_top", 30, "wombat", 2),
    Spec("q_top", 0, "quokka", 5),
    Spec("q_second", 40, "quokka", 4),
    Spec("f1", 1, None, 0),
    Spec("q_third", 2, "quokka", 3),
    Spec("f2", 3, None, 0),
    Spec("q_fourth", 6, "quokka", 2),
    Spec("q_fifth", 12, "quokka", 1),
    Spec("f3", 4, None, 0),
    Spec("f4", 5, None, 0),
    Spec("f5", 7, None, 0),
    Spec("f6", 8, None, 0),
    Spec("w_second", 31, "wombat", 1),
)


def at_angle(step: float) -> list[float]:
    """A unit vector ``step`` units away from ``unit_vector(0)``: cosine distance ``1 - cos``."""
    angle = step * RADIANS_PER_STEP
    return [math.cos(angle), math.sin(angle)] + [0.0] * (EMBEDDING_DIM - 2)


def insert_page(
    conn: psycopg.Connection[Any], version: int, specs: tuple[Spec, ...]
) -> dict[str, int]:
    """Insert ``specs`` as the chunks of one page of ``version``, in order; return key -> id."""
    source_path, url, title = PAGE
    document: DocumentRow = insert_document(
        conn, version, source_path=source_path, url=url, title=title
    )
    ids: dict[str, int] = {}
    for ordinal, spec in enumerate(specs):
        ids[spec.key] = insert_chunk(
            conn,
            version,
            document,
            ordinal=ordinal,
            breadcrumb=(title, f"Part {ordinal}"),
            anchors=(f"part-{ordinal}",),
            content=spec.content,
            embedding=at_angle(spec.step),
        )
    return ids
