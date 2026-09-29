"""A hand-made text corpus for the lexical (2.02) and hybrid (2.05) retrieval specs.

Controlled words: every group of chunks shares one rare word (quokka, wombat, ocelot, capybara, ...)
that appears nowhere else, so a test knows exactly which chunks a question can match. The chunks
are ordinary rows of the real schema; the generated ``tsv`` column is what production searches.

Insert order is ids ascending, and it is deliberate: where a test needs "the lower id must win a
tie" or "rank must not follow insertion order", the chunks are inserted so that a missing
tie-breaker or a missing weight would show.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import psycopg

from tests.support import DocumentRow, insert_chunk, insert_document

ZOO = ("docs/en/docs/tutorial/zoo.md", "https://fastapi.tiangolo.com/tutorial/zoo/", "Zoo")
NOTES = ("docs/en/docs/notes.md", "https://fastapi.tiangolo.com/notes/", "Notes")
PENGUINS = ("docs/en/docs/penguins.md", "https://fastapi.tiangolo.com/penguins/", "Penguins")

CAPYBARA_COUNT = 12
CAPYBARA_KEYS = tuple(f"capybara_{i:02d}" for i in range(CAPYBARA_COUNT))
TIE_KEYS = ("tie_a", "tie_b", "tie_c")


class Spec(NamedTuple):
    key: str  # how a test refers to the chunk
    page: tuple[str, str, str]  # (source_path, url, title)
    breadcrumb: tuple[str, ...]  # starts with the page title
    anchors: tuple[str, ...]  # the anchor_path; () for a page intro
    content: str


def _spec(key: str, page: tuple[str, str, str], heading: str, content: str) -> Spec:
    anchor = heading.lower().replace(" ", "-")
    return Spec(key, page, (page[2], heading), (anchor,), content)


CORPUS: tuple[Spec, ...] = (
    # OR semantics: a long question that names all three animals. Each chunk has a different subset.
    _spec("or_one", ZOO, "Marsupials", "The quokka is a small marsupial that smiles at visitors."),
    _spec("or_two", ZOO, "Whales", "The narwhal has a long tusk. A quokka would never meet one."),
    _spec(
        "or_three",
        ZOO,
        "Scales",
        "The pangolin wears scales. Neither the quokka nor the narwhal has any.",
    ),
    # Weights: the same word (wombat) in the body of the first chunk and in the breadcrumb of the
    # second. The body chunk goes in first, so it has the lower id.
    _spec("weight_body", NOTES, "Reference", "The wombat digs deep burrows."),
    _spec("weight_title", NOTES, "Wombat", "The animal digs deep burrows."),
    # Ties: three identical chunks, then one that says "ocelot" three times and must rank first
    # even though it has the highest id.
    *(_spec(key, NOTES, "Naps", "The ocelot naps in the shade.") for key in TIE_KEYS),
    _spec("tie_strong", NOTES, "Naps", "The ocelot naps. The ocelot hunts. The ocelot climbs."),
    # k: many identical matches.
    *(_spec(key, NOTES, "River", "The capybara swims in the river.") for key in CAPYBARA_KEYS),
    # Stemming: the index has "runs", questions say "running".
    _spec("stem_runs", NOTES, "Servers", "The scheduler runs nightly reports."),
    # A token whose lexeme contains a colon (Postgres keeps host:port as one lexeme).
    _spec("host_port", NOTES, "Proxy", "The proxy listens on example.com:8080 for requests."),
    # Non-ASCII.
    _spec("accent", NOTES, "Menu", "The café serves espresso."),
    # A page intro: empty anchor_path, section_id ending in "#", the page's own url.
    Spec(
        "intro",
        PENGUINS,
        ("Penguins",),
        (),
        "Penguins are flightless birds. This page describes the emperor colony.",
    ),
)


def build_text_index(
    conn: psycopg.Connection[Any],
    version: int,
    *,
    embeddings: Mapping[str, Sequence[float]] | None = None,
) -> dict[str, int]:
    """Insert ``CORPUS`` into ``version`` in order; return ``{spec key: chunk id}``.

    ``embeddings`` overrides the default unit vector per key (hybrid tests place chunks in vector
    space; the lexical tests don't care).
    """
    documents: dict[str, DocumentRow] = {}
    ordinals: dict[str, int] = {}
    ids: dict[str, int] = {}
    for spec in CORPUS:
        source_path, url, title = spec.page
        if source_path not in documents:
            documents[source_path] = insert_document(
                conn, version, source_path=source_path, url=url, title=title
            )
        ordinal = ordinals.get(source_path, 0)
        ordinals[source_path] = ordinal + 1
        ids[spec.key] = insert_chunk(
            conn,
            version,
            documents[source_path],
            ordinal=ordinal,
            breadcrumb=spec.breadcrumb,
            anchors=spec.anchors,
            content=spec.content,
            embedding=(embeddings or {}).get(spec.key),
        )
    return ids
