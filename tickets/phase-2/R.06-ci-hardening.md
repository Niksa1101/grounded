# R.06 — CI hardening: pinned actions, one source for the tokenizer

> **Before you start:** read [AGENTS.md](../../AGENTS.md), the rules in [tickets/README.md](../README.md) (how to work a ticket, ticket types, [standard checks](../README.md#standard-checks)) and the phase notes in [README.md](README.md). Work only on this ticket.

> **Owner:** G · **Type:** Build · **Branch:** `phase-2/ci-hardening` · **Blocked by:** R.05 ·
> **Builds on:** `.github/workflows/ci.yml`

**What to build:** review item #14.

**Scope notes**
- Every action is pinned to a commit SHA with the version in a comment (`actions/checkout@<sha> # v7.0.1`; the same for
  `setup-uv`, `cache`, `setup-node`). SHAs are read at change time with
  `gh api repos/<owner>/<repo>/git/ref/tags/<tag>`, dereferencing annotated tags.
- A step `id: tok` reads the encoding from `Settings`
  (`uv run python -c "…get_settings().tokenizer_encoding"` → `$GITHUB_OUTPUT`). The cache key becomes
  `tiktoken-${{ steps.tok.outputs.encoding }}-v1`, and the warm step uses the same value, so `Settings` is the only
  source.

**Acceptance criteria**
- [x] CI is green on the PR, and its log shows a cache key containing `o200k_base`
      ([run 36625400084](https://github.com/Niksa1101/grounded/actions/runs/36625400084): `key: tiktoken-o200k_base-v1`).

**Verify:** the CI run on the PR (link in the description).

**Eval impact:** none.

**Docs to update:** Tech §5.5 (the sentence on the CI tiktoken cache) and §17.
