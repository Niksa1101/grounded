# Golden set

The questions every retrieval and generation eval runs on (Tech.md §15.1).

| File | What it is |
|---|---|
| `golden_set.v1.jsonl` | the curated set: 30 items (25 answerable, 5 unanswerable) |
| `candidates.v1.jsonl` | the drafts the set was curated from, kept as provenance |

Once a baseline references a version, its items are never edited in place: changes go into a new file
(`golden_set.v2.jsonl`) and need new baselines.

## Row format

One JSON object per line, validated by `grounded.schemas.eval.GoldenItem`:

```json
{"id": "q017", "question": "How can I run a function after returning a response?", "type": "how_to",
 "answerable": true, "reference_answer": "Use BackgroundTasks: ...",
 "relevant_sections": [{"section": "docs/en/docs/tutorial/background-tasks.md#using-backgroundtasks", "grade": 2}],
 "source_section": "docs/en/docs/tutorial/background-tasks.md#using-backgroundtasks", "notes": ""}
```

- `type`: `factual`, `how_to`, `code`, `multi_section` or `unanswerable`. Target mix for v1 (PRD §12):
  ~8 factual, ~8 how_to, ~5 code, ~4 multi_section, ~5 unanswerable.
- `answerable` is `false` exactly for `unanswerable` items, which have no `relevant_sections`. Their
  `reference_answer` says why the docs don't answer them.
- `source_section`: the sampled section a question was drafted from (provenance only), or `null`.

## Labeling guide

- **Grade 2**: the section contains (part of) the answer. **Grade 1**: useful context that doesn't answer
  by itself. Recall and MRR count only grade 2; nDCG counts both.
- Every answerable item has at least one grade-2 label; a `multi_section` item has at least two, because
  its answer really needs several sections.
- A label is `docs/en/docs/<page>.md#<anchor>` or just `docs/en/docs/<page>.md` (the whole page). An H2
  label also matches the chunks of its H3 subsections.
- **No nested labels** in one item: not a page and a section on it, not an H2 and one of its H3s. Pick
  the one that fits (the H2 if the answer spans its subsections, else the H3s).
- Label the sections as the chunker produced them: `grounded golden sections <page>` lists them. A small
  section merged into its previous or next sibling has no chunk of its own (the merged chunk keeps the
  first section's anchor), so a label on it matches nothing; label the section it was merged into, or
  the parent.
- Write questions the way a user would ask them, not by rephrasing the section title. Prefer questions
  with one clear answer in the docs; avoid ones whose answer depends on the FastAPI version beyond the
  pinned tag.
- Unanswerable items: mostly "near misses" that sound like FastAPI topics the docs don't cover (they
  test that the model refuses instead of answering from memory), plus a couple of clearly off-topic
  ones. Check the corpus first: a near miss that the docs partly answer is a bad item.

## Commands

Run from `backend/`; they chunk the pinned corpus (`FASTAPI_REF`) locally, with no API calls.

```bash
uv run grounded golden sample --n 50 --seed 20260926
```

```bash
uv run grounded golden sections tutorial/background-tasks.md
```

```bash
uv run grounded golden validate ../eval/golden/golden_set.v1.jsonl
```

```bash
uv run grounded golden validate ../eval/golden/golden_set.v1.jsonl --against-index
```

`validate` checks the schema, unique IDs and the type mix, then resolves every label against the corpus
chunks with the metrics' own matching rule (a label must match a chunk; two labels of one item must never
match the same chunk). `--against-index` repeats the resolution on the active index in the database.

## Provenance of v1

`candidates.v1.jsonl` (52 items) was drafted by the Agent on 2026-09-26 from FastAPI `0.141.1`, with no
LLM API calls: 50 sections drawn with `grounded golden sample --n 50 --seed 20260926` (items with a
`source_section`), plus a few written for type coverage and 8 unanswerable items (`source_section: null`,
the reason in `notes`).

`golden_set.v1.jsonl` (30 items): on 2026-09-28 the Author approved a selection of 30 candidates with no
edits. The mix is 8 factual, 8 how_to, 5 code, 4 multi_section and 5 unanswerable (4 near misses and 1
off-topic question). Items keep their candidate IDs, so each one traces back to its draft.

Manual label check (ticket R.08), 2026-09-29: the Author read 10 of the 25 answerable items against the text of
the chunks their labels point at and found no errors: every grade-2 chunk contains the answer, every grade-1
chunk only gives context, no section is missing, and no question is a lexical paraphrase of its section. The
sample was drawn with seed 20260929 by a one-off script over `grounded.evals.golden` (not committed):
- all 4 `multi_section` items: q036, q039, q042, q043;
- 3 items with a grade-1 label: q016, q018, q020;
- 3 other answerable items: q008, q028, q029.

The other 15 answerable items were checked by tooling only (schema, label resolution, `validate
--against-index`). The 5 unanswerable items have no labels.
