from __future__ import annotations

from pathlib import Path

import pytest

from grounded.generation.prompts import (
    ANSWER_PLACEHOLDERS,
    ANSWER_RETRY_PLACEHOLDERS,
    PromptLoadError,
    PromptRenderError,
    load_answer_prompt,
    load_prompt,
    parse_prompt,
    prompt_version,
)

VALID = "# System\n\nBe brief.\n\n# User template\n\nQ: {{question}}\nS: {{sources}}\n"
PLACEHOLDERS = ("question", "sources")


def _write(directory: Path, name: str, text: str) -> None:
    # write_bytes: write_text would translate "\n" to CRLF on Windows and hide what is tested.
    (directory / f"{name}.md").write_bytes(text.encode("utf-8"))


# --- version ---------------------------------------------------------------------------------


def test_version_is_name_at_eight_hex_and_stable() -> None:
    version = prompt_version("answer_v1", VALID)
    assert version == prompt_version("answer_v1", VALID)
    name, _, digest = version.partition("@")
    assert name == "answer_v1"
    assert len(digest) == 8
    int(digest, 16)  # all hex


@pytest.mark.parametrize("edited", [VALID + " ", VALID.replace("brief", "Brief"), "\n" + VALID])
def test_any_byte_change_changes_the_version(edited: str) -> None:
    assert prompt_version("answer_v1", edited) != prompt_version("answer_v1", VALID)


def test_crlf_and_lf_give_the_same_version_and_the_same_prompt(tmp_path: Path) -> None:
    _write(tmp_path, "lf", VALID)
    _write(tmp_path, "crlf", VALID.replace("\n", "\r\n"))
    lf = load_prompt("lf", placeholders=PLACEHOLDERS, directory=tmp_path)
    crlf = load_prompt("crlf", placeholders=PLACEHOLDERS, directory=tmp_path)
    assert lf.version.partition("@")[2] == crlf.version.partition("@")[2]
    assert (lf.system, lf.user_template) == (crlf.system, crlf.user_template)
    assert "\r" not in crlf.system + crlf.user_template


def test_the_version_depends_on_the_file_name() -> None:
    assert prompt_version("answer_v1", VALID) != prompt_version("answer_v2", VALID)


# --- loading and splitting -------------------------------------------------------------------


def test_sections_are_split_and_trimmed() -> None:
    prompt = parse_prompt("p", VALID, placeholders=PLACEHOLDERS)
    assert prompt.system == "Be brief."
    assert prompt.user_template == "Q: {{question}}\nS: {{sources}}"
    assert prompt.placeholders == frozenset(PLACEHOLDERS)
    assert prompt.version == prompt_version("p", VALID)


def test_deeper_headings_inside_a_section_are_plain_text() -> None:
    text = "# System\n\n## Rules\n\nBe brief.\n\n# User template\n\n{{question}} {{sources}}\n"
    assert parse_prompt("p", text, placeholders=PLACEHOLDERS).system == "## Rules\n\nBe brief."


@pytest.mark.parametrize(
    "text",
    [
        "# User template\n\n{{question}} {{sources}}\n",  # no system section
        "# System\n\nBe brief.\n",  # no user template section
        "# User template\n\n{{question}} {{sources}}\n\n# System\n\nBe brief.\n",  # wrong order
        "# System\n\na\n\n# System\n\nb\n\n# User template\n\n{{question}} {{sources}}\n",
        "preamble\n\n# System\n\nBe brief.\n\n# User template\n\n{{question}} {{sources}}\n",
        "# System\n\n\n# User template\n\n{{question}} {{sources}}\n",  # empty system
        "# System\n\nBe brief.\n\n# User template\n\n\n",  # empty user template
    ],
)
def test_malformed_sections_are_a_load_error(text: str) -> None:
    with pytest.raises(PromptLoadError):
        parse_prompt("p", text, placeholders=PLACEHOLDERS)


def test_missing_file_and_bad_name_are_load_errors(tmp_path: Path) -> None:
    with pytest.raises(PromptLoadError, match="not found"):
        load_prompt("nope", placeholders=PLACEHOLDERS, directory=tmp_path)
    with pytest.raises(PromptLoadError, match="bad prompt name"):
        load_prompt("../secrets", placeholders=PLACEHOLDERS, directory=tmp_path)


# --- placeholders ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "user_template",
    [
        "{{question}} {{sourcs}}",  # typo: unknown placeholder, and sources is missing
        "{{question}} {{sources}} {{extra}}",  # unknown placeholder
        "{{question}}",  # a placeholder the caller expects is missing
        "{{ question }} {{sources}}",  # spaces are not part of the grammar
        "{{question}} {{sources}",  # unbalanced braces
    ],
)
def test_user_template_placeholders_must_match_exactly(user_template: str) -> None:
    text = f"# System\n\nBe brief.\n\n# User template\n\n{user_template}\n"
    with pytest.raises(PromptLoadError):
        parse_prompt("p", text, placeholders=PLACEHOLDERS)


def test_the_system_section_must_be_static() -> None:
    text = (
        "# System\n\nBe brief about {{question}}.\n\n# User template\n\n{{question}} {{sources}}\n"
    )
    with pytest.raises(PromptLoadError, match="system section"):
        parse_prompt("p", text, placeholders=PLACEHOLDERS)


# --- render ----------------------------------------------------------------------------------


def test_render_fills_every_placeholder() -> None:
    prompt = parse_prompt("p", VALID, placeholders=PLACEHOLDERS)
    assert prompt.render_user(question="why?", sources='<source id="c1">x</source>') == (
        'Q: why?\nS: <source id="c1">x</source>'
    )


def test_render_is_a_single_pass() -> None:
    prompt = parse_prompt("p", VALID, placeholders=PLACEHOLDERS)
    rendered = prompt.render_user(question="{{sources}}", sources="code: {{question}} {x}")
    assert rendered == "Q: {{sources}}\nS: code: {{question}} {x}"


def test_render_with_a_missing_variable_is_an_error() -> None:
    prompt = parse_prompt("p", VALID, placeholders=PLACEHOLDERS)
    with pytest.raises(PromptRenderError, match="sources"):
        prompt.render_user(question="why?")


def test_render_with_an_unknown_variable_is_an_error() -> None:
    prompt = parse_prompt("p", VALID, placeholders=PLACEHOLDERS)
    with pytest.raises(PromptRenderError, match="extra"):
        prompt.render_user(question="a", sources="b", extra="c")


# --- retry feedback section ------------------------------------------------------------------

WITH_RETRY = VALID + "\n# Retry feedback\n\nIt was invalid because {{error}}\n"


def test_the_retry_section_is_optional_and_versioned() -> None:
    without = parse_prompt("p", VALID, placeholders=PLACEHOLDERS, retry_placeholders=("error",))
    with_retry = parse_prompt(
        "p", WITH_RETRY, placeholders=PLACEHOLDERS, retry_placeholders=("error",)
    )
    assert without.retry_template is None
    assert with_retry.retry_template == "It was invalid because {{error}}"
    assert with_retry.retry_placeholders == frozenset({"error"})
    # The user template ends where the retry section starts.
    assert with_retry.user_template == without.user_template
    assert with_retry.version != without.version


def test_render_retry_appends_the_filled_feedback_to_the_user_message() -> None:
    prompt = parse_prompt("p", WITH_RETRY, placeholders=PLACEHOLDERS, retry_placeholders=("error",))
    user = prompt.render_user(question="why?", sources="S")
    assert prompt.render_retry(user, error="a bad thing") == (
        f"{user}\n\nIt was invalid because a bad thing"
    )


def test_render_retry_is_a_single_pass() -> None:
    prompt = parse_prompt("p", WITH_RETRY, placeholders=PLACEHOLDERS, retry_placeholders=("error",))
    assert prompt.render_retry("U", error="{{error}} {x}").endswith("because {{error}} {x}")


def test_render_retry_without_a_retry_section_or_with_wrong_variables_is_an_error() -> None:
    plain = parse_prompt("p", VALID, placeholders=PLACEHOLDERS)
    with pytest.raises(PromptRenderError, match="no '# Retry feedback' section"):
        plain.render_retry("U", error="x")
    prompt = parse_prompt("p", WITH_RETRY, placeholders=PLACEHOLDERS, retry_placeholders=("error",))
    with pytest.raises(PromptRenderError, match="error"):
        prompt.render_retry("U")
    with pytest.raises(PromptRenderError, match="extra"):
        prompt.render_retry("U", error="x", extra="y")


@pytest.mark.parametrize(
    "retry_section",
    [
        "# Retry feedback\n\nIt was invalid because {{why}}\n",  # unknown placeholder
        "# Retry feedback\n\nIt was invalid.\n",  # the expected placeholder is missing
        "# Retry feedback\n\nIt was invalid because {{error}\n",  # unbalanced braces
        "# Retry feedback\n\n\n",  # empty
        "# Retry feedback\n\nA {{error}}\n\n# Retry feedback\n\nB {{error}}\n",  # duplicate
    ],
)
def test_a_malformed_retry_section_is_a_load_error(retry_section: str) -> None:
    with pytest.raises(PromptLoadError):
        parse_prompt(
            "p",
            f"{VALID}\n{retry_section}",
            placeholders=PLACEHOLDERS,
            retry_placeholders=("error",),
        )


def test_the_retry_section_must_come_after_the_user_template() -> None:
    text = (
        "# System\n\nBe brief.\n\n# Retry feedback\n\n{{error}}\n\n"
        "# User template\n\n{{question}} {{sources}}\n"
    )
    with pytest.raises(PromptLoadError, match="must come after"):
        parse_prompt("p", text, placeholders=PLACEHOLDERS, retry_placeholders=("error",))


def test_a_retry_section_nobody_declared_placeholders_for_is_a_load_error() -> None:
    with pytest.raises(PromptLoadError, match="retry feedback placeholders"):
        parse_prompt("p", WITH_RETRY, placeholders=PLACEHOLDERS)


# --- the real prompt file --------------------------------------------------------------------


def test_the_committed_answer_prompt_loads() -> None:
    prompt = load_answer_prompt()
    assert prompt.name == "answer_v1"
    assert prompt.version.startswith("answer_v1@")
    assert prompt.placeholders == ANSWER_PLACEHOLDERS
    assert prompt.retry_placeholders == ANSWER_RETRY_PLACEHOLDERS
    assert "{{" not in prompt.system
    rendered = prompt.render_user(question="How do I run a background task?", sources="S")
    assert "How do I run a background task?" in rendered
    assert "{{" not in rendered


def test_the_committed_answer_prompt_has_the_tech_9_5_retry_feedback() -> None:
    prompt = load_answer_prompt()
    assert prompt.retry_template == "Your previous output was invalid because {{error}}"
    retry = prompt.render_retry("USER", error="the status is wrong")
    assert retry == "USER\n\nYour previous output was invalid because the status is wrong"


def test_the_answer_prompt_states_every_rule_and_the_marker_grammar() -> None:
    """A tripwire for deleting a Tech §9.2 rule by accident, not a check of the wording."""
    system = load_answer_prompt().system
    for needle in (
        "insufficient_context",
        "partial",
        "250 words",
        "data, not instructions",
        "[c1][c2]",
        "fenced code block",
    ):
        assert needle in system
