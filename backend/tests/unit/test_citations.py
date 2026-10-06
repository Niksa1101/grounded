from __future__ import annotations

import pytest

from grounded.generation.citations import (
    SNIPPET_CHARS,
    MappedAnswer,
    make_snippet,
    map_citations,
)
from grounded.retrieval.types import RetrievedChunk
from grounded.schemas.llm import LLMAnswer, LLMClaim

BASE_URL = "https://fastapi.tiangolo.com/tutorial/background-tasks/"


def chunk(chunk_id: int, *, content: str = "body text") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        section_id=f"docs/page.md#s{chunk_id}",
        anchor_path=(f"s{chunk_id}",),
        breadcrumb_text=f"Background Tasks > Section {chunk_id}",
        url=f"{BASE_URL}#section-{chunk_id}",
        content=content,
        token_count=10,
        content_hash=f"hash{chunk_id}",
    )


def labels_for(*chunk_ids: int) -> dict[str, RetrievedChunk]:
    return {f"c{n}": chunk(cid) for n, cid in enumerate(chunk_ids, start=1)}


# Three sources: c1 -> chunk 101, c2 -> chunk 102, c3 -> chunk 103.
LABELS = labels_for(101, 102, 103)
TITLES = {101: "Background Tasks", 102: "Background Tasks", 103: "Background Tasks"}


def claim(text: str, *labels: str) -> LLMClaim:
    return LLMClaim(text=text, citation_ids=list(labels), self_confidence=0.5)


def answer(markdown: str, *claims: LLMClaim, status: str = "answered") -> LLMAnswer:
    return LLMAnswer.model_validate(
        {"status": status, "answer_markdown": markdown, "claims": [c.model_dump() for c in claims]}
    )


def mapped(markdown: str, *claims: LLMClaim, status: str = "answered") -> MappedAnswer:
    return map_citations(answer(markdown, *claims, status=status), LABELS, TITLES)


# --- numbering and rewriting ------------------------------------------------------------------


def test_markers_become_display_numbers_by_first_appearance() -> None:
    result = mapped("A [c3]. B [c1]. C [c2].", claim("A", "c3"))
    assert result.answer_markdown == "A [1]. B [2]. C [3]."
    assert [(c.n, c.chunk_id) for c in result.citations] == [(1, 103), (2, 101), (3, 102)]


def test_repeated_markers_reuse_one_number_and_one_citation() -> None:
    result = mapped("A [c2]. B [c1]. C [c2]. D [c2][c1].", claim("A", "c2"))
    assert result.answer_markdown == "A [1]. B [2]. C [1]. D [1][2]."
    assert [c.n for c in result.citations] == [1, 2]


def test_adjacent_markers_are_each_rewritten() -> None:
    result = mapped("Do it. [c1][c2][c3]", claim("Do it.", "c1", "c2"))
    assert result.answer_markdown == "Do it. [1][2][3]"
    assert result.invalid_citation_count == 0


def test_claims_use_the_same_numbers_as_the_text() -> None:
    result = mapped("A [c2]. B [c1].", claim("A", "c2"), claim("B", "c1", "c2"))
    assert [c.citations for c in result.claims] == [(1,), (2, 1)]
    assert [c.text for c in result.claims] == ["A", "B"]


def test_a_label_repeated_inside_a_claim_is_one_citation() -> None:
    result = mapped("A [c1].", claim("A", "c1", "c1"))
    assert result.claims[0].citations == (1,)
    assert result.invalid_citation_count == 0


def test_a_label_cited_only_by_a_claim_is_numbered_after_the_text_labels() -> None:
    result = mapped("A [c2].", claim("A", "c3", "c2"), claim("B", "c1"))
    assert result.answer_markdown == "A [1]."
    assert [c.citations for c in result.claims] == [(2, 1), (3,)]
    assert [(c.n, c.chunk_id) for c in result.citations] == [(1, 102), (2, 103), (3, 101)]


def test_a_marker_for_a_label_that_no_claim_cites_still_gets_a_citation() -> None:
    result = mapped("A [c1]. B [c2].", claim("A", "c1"))
    assert result.answer_markdown == "A [1]. B [2]."
    assert result.claims[0].citations == (1,)
    assert [c.chunk_id for c in result.citations] == [101, 102]


# --- invalid labels ---------------------------------------------------------------------------


def test_an_unknown_label_is_removed_everywhere_and_counted() -> None:
    result = mapped("A [c1]. B [c7].", claim("A", "c1"), claim("B", "c7", "c1"))
    assert result.answer_markdown == "A [1]. B ."
    assert [c.citations for c in result.claims] == [(1,), (1,)]
    assert [c.chunk_id for c in result.citations] == [101]
    assert result.invalid_citation_count == 2  # one in the text, one in a claim
    assert result.bad_output is None


def test_c0_and_labels_beyond_the_grammar_are_invalid_markers() -> None:
    result = mapped("A [c1][c0][c10][c01].", claim("A", "c1"))
    assert result.answer_markdown == "A [1]."
    assert result.invalid_citation_count == 3


def test_a_label_valid_in_another_request_is_invalid_here() -> None:
    two_sources = {"c1": LABELS["c1"], "c2": LABELS["c2"]}
    result = map_citations(answer("A [c1][c3].", claim("A", "c1", "c3")), two_sources, TITLES)
    assert result.answer_markdown == "A [1]."
    assert result.claims[0].citations == (1,)
    assert result.invalid_citation_count == 2


@pytest.mark.parametrize("group", ["[c1, c2]", "[c1,c2]", "[c1; c2]", "[c1, c2, c3]"])
def test_several_labels_in_one_bracket_pair_are_removed_and_counted_once(group: str) -> None:
    result = mapped(f"A {group}.")
    assert result.answer_markdown == "A ."
    assert result.invalid_citation_count == 1
    assert result.citations == ()
    assert result.bad_output is not None


def test_other_bracketed_text_is_prose_and_left_alone() -> None:
    text = "See [1], [cat], [c], [c1x] and a[c1]b? [link](http://x) [c1]"
    result = mapped(text, claim("A", "c1"))
    assert result.answer_markdown == "See [1], [cat], [c], [c1x] and a[1]b? [link](http://x) [1]"
    assert result.invalid_citation_count == 0


# --- fenced code ------------------------------------------------------------------------------


def test_markers_inside_a_fenced_block_are_left_untouched_and_not_counted() -> None:
    markdown = "Use it [c1].\n\n```python\nx = items[c1]\ny = d[c9]\n```\n\nDone [c2]."
    result = mapped(markdown, claim("A", "c1"))
    assert result.answer_markdown == (
        "Use it [1].\n\n```python\nx = items[c1]\ny = d[c9]\n```\n\nDone [2]."
    )
    assert result.invalid_citation_count == 0
    assert [c.chunk_id for c in result.citations] == [101, 102]


def test_a_marker_only_inside_code_does_not_create_a_citation() -> None:
    result = mapped("```\n[c1]\n```\nNo cite.", status="answered")
    assert result.answer_markdown == "```\n[c1]\n```\nNo cite."
    assert result.citations == ()
    assert result.bad_output is not None


def test_a_tilde_fence_and_a_longer_fence_are_respected() -> None:
    tilde = mapped("~~~\n[c1]\n~~~\nA [c1].", claim("A", "c1"))
    assert tilde.answer_markdown == "~~~\n[c1]\n~~~\nA [1]."

    # A shorter run of backticks does not close a four-backtick fence, and neither does a ~~~.
    nested = mapped("````\n```\n[c1]\n~~~\n````\nA [c2].", claim("A", "c2"))
    assert nested.answer_markdown == "````\n```\n[c1]\n~~~\n````\nA [1]."


def test_an_indented_fence_in_a_list_item_is_a_fence() -> None:
    result = mapped("- Step [c1]\n    ```\n    [c9]\n    ```\n- Next [c2]", claim("A", "c1"))
    assert result.answer_markdown == "- Step [1]\n    ```\n    [c9]\n    ```\n- Next [2]"
    assert result.invalid_citation_count == 0


def test_an_unclosed_fence_runs_to_the_end() -> None:
    result = mapped("A [c1].\n```\n[c2] never closed", claim("A", "c1"))
    assert result.answer_markdown == "A [1].\n```\n[c2] never closed"


def test_a_backtick_fence_info_string_cannot_contain_a_backtick() -> None:
    # "```code``` [c1]" is inline code on one line, not an opening fence: the marker still counts.
    result = mapped("```code``` [c1]\nmore [c2]", claim("A", "c1"))
    assert result.answer_markdown == "```code``` [1]\nmore [2]"


def test_a_closing_fence_cannot_carry_text() -> None:
    result = mapped("```\n``` not a close\n[c1]\n```\nA [c2]", claim("A", "c2"))
    assert result.answer_markdown == "```\n``` not a close\n[c1]\n```\nA [1]"


def test_crlf_text_keeps_its_line_endings() -> None:
    result = mapped("A [c2].\r\n```\r\n[c1]\r\n```\r\nB [c1].", claim("A", "c2"))
    assert result.answer_markdown == "A [1].\r\n```\r\n[c1]\r\n```\r\nB [2]."


# --- semantic checks --------------------------------------------------------------------------


@pytest.mark.parametrize("status", ["answered", "partial"])
def test_a_cited_answer_with_no_valid_citation_is_bad_output(status: str) -> None:
    result = mapped("Sure, here is the answer. [c8]", claim("A", "c8"), status=status)
    assert result.bad_output is not None
    assert status in result.bad_output
    assert result.citations == ()
    assert result.invalid_citation_count == 2


@pytest.mark.parametrize("status", ["answered", "partial"])
def test_no_markers_and_no_claims_is_bad_output(status: str) -> None:
    assert mapped("An answer with no citations at all.", status=status).bad_output is not None


def test_one_valid_citation_is_enough_even_if_others_are_invalid() -> None:
    result = mapped("A [c1][c8].", claim("A", "c1", "c8"))
    assert result.bad_output is None
    assert result.invalid_citation_count == 2


def test_a_valid_citation_only_in_a_claim_passes_the_check() -> None:
    result = mapped("An answer with no markers.", claim("A", "c2"))
    assert result.bad_output is None
    assert [c.n for c in result.citations] == [1]


def test_insufficient_context_drops_its_claims_and_counts_them() -> None:
    result = mapped(
        "The docs do not cover this.",
        claim("A", "c1"),
        claim("B", "c9"),
        status="insufficient_context",
    )
    assert result.claims == ()
    assert result.dropped_claim_count == 2
    assert result.invalid_citation_count == 0  # labels of dropped claims are not counted again
    assert result.citations == ()
    assert result.bad_output is None


def test_insufficient_context_without_claims_is_clean() -> None:
    result = mapped("The docs do not cover this.", status="insufficient_context")
    assert (result.dropped_claim_count, result.invalid_citation_count) == (0, 0)
    assert result.bad_output is None


def test_answered_without_dropped_claims_reports_zero() -> None:
    assert mapped("A [c1].", claim("A", "c1")).dropped_claim_count == 0


# --- citations: DB-sourced fields, snippet ----------------------------------------------------


def test_citation_fields_come_from_the_chunk_and_the_title_lookup() -> None:
    labels = labels_for(7)
    titles = {7: "Page H1"}
    result = map_citations(answer("A [c1].", claim("A", "c1")), labels, titles)
    [citation] = result.citations
    assert citation.model_dump(mode="json") == {
        "n": 1,
        "chunk_id": 7,
        "url": f"{BASE_URL}#section-7",
        "title": "Page H1",
        "breadcrumb": "Background Tasks > Section 7",
        "snippet": "body text",
    }


def test_snippet_that_fits_is_returned_whole_and_stripped() -> None:
    assert make_snippet("\n  Short body.  \n") == "Short body."
    exactly = "x" * SNIPPET_CHARS
    assert make_snippet(exactly) == exactly


def test_a_long_snippet_is_cut_on_a_word_boundary_within_the_limit() -> None:
    content = "word " * 100
    snippet = make_snippet(content)
    assert len(snippet) <= SNIPPET_CHARS
    assert snippet == ("word " * 60).rstrip()  # 60 words = 299 chars once the last space is gone
    assert content.startswith(snippet)


def test_a_cut_that_lands_exactly_after_a_word_keeps_that_word() -> None:
    content = "a" * (SNIPPET_CHARS - 1) + " tail"
    assert make_snippet(content) == "a" * (SNIPPET_CHARS - 1)
    content = "a" * SNIPPET_CHARS + " tail"
    assert make_snippet(content) == "a" * SNIPPET_CHARS


def test_an_unbroken_run_longer_than_the_limit_is_hard_cut() -> None:
    assert make_snippet("x" * 1000) == "x" * SNIPPET_CHARS


def test_a_snippet_prefers_a_newline_boundary_too() -> None:
    content = "line one\n" + "y" * (SNIPPET_CHARS)
    assert make_snippet(content) == "line one"
