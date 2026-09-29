# R.08 — Manual check of the golden-set labels and explain-back

> **Before you start:** read [AGENTS.md](../../AGENTS.md), the rules in [tickets/README.md](../README.md) (how to work a ticket, ticket types, [standard checks](../README.md#standard-checks)) and the phase notes in [README.md](README.md). Work only on this ticket.

> **Owner:** **A** (+G worksheet) · **Type:** Author · **Branch:** none, or a small `eval:` / golden PR ·
> **Blocked by:** none (run in parallel with R.01–R.06; must finish before B.01) · **Builds on:**
> `evals/golden.py` (`load_golden_set`, `prepare_corpus`, `section_matches`)

**What to build:** review item #4. The golden set's labels were checked mostly by tooling. The Author checks a sample
by hand, so the numbers in every later baseline rest on labels a person has read.

**Agent's role:** prepares a worksheet, `.cache/review/golden-v1-labels.md` (never committed), with a one-off script
over the existing helpers. It has 10 items: all 4 `multi_section`, 3 items with grade-1 labels, and 3 drawn with a
fixed seed. Per item: the question, the reference answer, and every label with its grade and the text of the chunk it
points at. Afterwards the Agent asks interview-style questions in the explain-back session.

**Author checks**
- Does each grade-2 chunk really contain the answer, and does each grade-1 only give context?
- Is a section missing?
- Is the question a lexical paraphrase of the section (which would flatter lexical search)?
- Use `grounded golden sections <page>` to look at the sections.

**Outcome**
- Everything OK: a note in `eval/golden/README.md` ("Provenance of v1": who checked which items, and when).
- Errors found: `golden_set.v2.jsonl` (v1 is never edited in place, AGENTS.md §7) with the same IDs and a provenance
  note. Then `golden validate --against-index`, and the default `--golden` in `cli.py` and AGENTS.md §10 move to v2.

**Explain-back session:** the Author explains chunker rules 2–4 (including keep-with-next from R.07) and how the ideal
DCG is computed. No repo changes.

**Acceptance criteria**
- [ ] The 10 items are checked, and either the provenance note or v2 is committed.
- [ ] The explain-back session took place.

**Eval impact:** depends on the findings (v2 means new baselines through B.01).
