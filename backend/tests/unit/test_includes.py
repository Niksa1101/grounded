"""Include directive resolution against corpus_mini and small throwaway repos (Tech.md §5.3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from grounded.ingest.includes import IncludeError, resolve_includes

CORPUS_MINI = Path(__file__).resolve().parents[1] / "fixtures" / "corpus_mini"
TASKS = CORPUS_MINI / "docs_src" / "background_tasks" / "tutorial001_py310.py"


def _resolve(text: str, root: Path = CORPUS_MINI) -> str:
    return resolve_includes(text, root, source_path="docs/en/docs/page.md")


def _file_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n").removesuffix("\n").split("\n")


def test_variant_include_becomes_a_python_fence_with_the_whole_file() -> None:
    out = _resolve("Before\n\n{* ../../docs_src/background_tasks/tutorial001_py310.py *}\n\nAfter")
    assert out.split("\n") == [
        "Before",
        "",
        "```python",
        *_file_lines(TASKS),
        "```",
        "",
        "After",
    ]


def test_highlight_option_is_dropped() -> None:
    out = _resolve("{* ../../docs_src/background_tasks/tutorial001_py310.py hl[1,13:14] *}")
    assert out.split("\n")[0] == "```python"
    assert "hl" not in out.split("\n")[0]


def test_line_ranges_keep_selected_lines_with_omission_markers() -> None:
    # tutorial001_py310.py has 15 lines; keep 6-7 and 12-13 (1-based, inclusive).
    lines = _file_lines(TASKS)
    out = _resolve("{* ../../docs_src/background_tasks/tutorial001_py310.py ln[12:13,6:7] hl[6] *}")
    assert out.split("\n") == [
        "```python",
        "# Code above omitted 👆",
        "",
        lines[5],
        lines[6],
        "",
        "# Code here omitted 👈",
        "",
        lines[11],
        lines[12],
        "",
        "# Code below omitted 👇",
        "```",
    ]


def test_range_from_line_one_to_the_end_has_no_markers() -> None:
    lines = _file_lines(TASKS)
    out = _resolve(
        f"{{* ../../docs_src/background_tasks/tutorial001_py310.py ln[1:{len(lines)}] *}}"
    )
    assert out.split("\n") == ["```python", *lines, "```"]


def test_title_option_stays_on_the_fence() -> None:
    out = _resolve(
        '{* ../../docs_src/background_tasks/tutorial001_py310.py title["app/main.py"] *}'
    )
    assert out.split("\n")[0] == '```python title="app/main.py"'


def test_referenced_variant_is_used_even_when_an_annotated_sibling_exists() -> None:
    # The page text sometimes discusses the non-Annotated form on purpose; the site shows the
    # referenced file first, so that is the one we keep.
    out = _resolve("{* ../../docs_src/dependencies/tutorial001_py310.py *}")
    assert "Depends(common_parameters)" in out
    assert "Annotated" not in out


def test_legacy_include_inside_an_existing_fence() -> None:
    html = CORPUS_MINI / "docs_src" / "templates" / "item.html"
    out = _resolve('```jinja hl_lines="3"\n{!../../docs_src/templates/item.html!}\n```')
    assert out.split("\n") == ["```jinja", *_file_lines(html), "```"]


def test_legacy_include_with_marker_and_indentation() -> None:
    out = _resolve("* item\n\n    ```html\n    {!> ../../docs_src/templates/item.html!}\n    ```")
    assert "    <html>" in out.split("\n")
    assert "    </html>" in out.split("\n")


def test_fence_grows_when_the_code_contains_backticks(tmp_path: Path) -> None:
    (tmp_path / "docs" / "en").mkdir(parents=True)
    (tmp_path / "docs_src").mkdir()
    (tmp_path / "docs_src" / "doc.py").write_text('"""Use ```x``` here."""\n', encoding="utf-8")
    out = resolve_includes("{* ../../docs_src/doc.py *}", tmp_path, source_path="p.md")
    assert out.split("\n") == ["````python", '"""Use ```x``` here."""', "````"]


def test_text_without_directives_is_unchanged() -> None:
    text = 'Some text\n\nnew_dict = {**old_dict, "new key": "new value"}\n\n```python\nx = 1\n```'
    assert _resolve(text) == text


def test_missing_file_fails_loudly_with_location() -> None:
    with pytest.raises(IncludeError, match=r"page\.md:3: included file not found"):
        _resolve("a\n\n{* ../../docs_src/nope/tutorial001_py310.py *}")


def test_include_outside_the_repository_is_refused() -> None:
    with pytest.raises(IncludeError, match="outside the repository"):
        _resolve("{* ../../../../../secrets.py *}")


@pytest.mark.parametrize("ranges", ["0:3", "10:20", "5:4", "a:b"])
def test_bad_line_ranges_fail(ranges: str) -> None:
    with pytest.raises(IncludeError):
        _resolve(f"{{* ../../docs_src/background_tasks/tutorial001_py310.py ln[{ranges}] *}}")


def test_unknown_option_fails() -> None:
    with pytest.raises(IncludeError, match="unknown include option"):
        _resolve("{* ../../docs_src/background_tasks/tutorial001_py310.py zz[1] *}")
