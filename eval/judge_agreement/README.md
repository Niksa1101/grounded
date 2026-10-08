# Judge–human agreement (ticket 4.11)

PRD FR-22 asks for the LLM judge's agreement with a human, measured on at least 10 verdicts and published. This folder
holds the **sheet** the Author labels by hand and how it was drawn. The procedure is in [docs/Tech.md §15.4](../../docs/Tech.md).

| File | What it is | Committed |
|---|---|---|
| `v1.csv` | the labeling sheet: 20 items with their evidence and an empty `human_label` column. **No verdict, reason, score, config name or control mark in it.** UTF-8 with a BOM, so Excel opens it as UTF-8 | yes |
| `v1.md` | the same 20 items laid out to be read (long evidence is painful in a CSV cell), with how to label and the label definitions copied verbatim from the two rubrics | yes |
| `eval/results/judge-agreement-v1.key.json` | the **key**: what each `item_id` is (kind, real or control, config, question, claim) and what the judge said (verdict, reason, rubric version, model). Gitignored on purpose: the labels are made without it | no (4.11b copies it next to the labels once they are in) |
| `README.md` | this file | yes |

The CSV and the view depend only on the seed and the results file, not on any verdict: they are the final sheet whether or
not the judge could be asked for the controls.

## The sample

**Source run.** The first generation baseline run (ticket 4.09), the only source of real items. Its promptfoo results file is
gitignored and is not in the repository; this is what identifies it.

| | |
|---|---|
| Results file | `20261008T155539Z-generation.json`, sha256 `4d571ae1a34b43d70a2ed46c38629ef81f0d5de76d433b5e160bd62269b40a31` |
| Produced at | commit `2a7d381f75f40791f6185a1e1ef80320a1cdfed8` (the `git_sha` of the baseline rows in `eval/baselines/generation.json`; promptfoo's file does not record it), 2026-10-08 15:56 UTC, promptfoo 0.123.1 |
| Golden set | `v1`, sha256 `8f0848177a5cb5a8ef649ee211ab9afb7f9bc07c73d37b37cbf3ef554e1abcb2` |
| Generator | `gemini` / `gemini-3.5-flash-lite`; configs `hybrid` (`answer_v1@08cc49e5`, index `0.141.1@4949e8a3`) and `no_rag` (`answer_no_rag_v1@5f725a9d`) |
| Judge | `groq` / `openai/gpt-oss-120b`, rubrics `judge_faithfulness_v1@84103412` and `judge_correctness_v1@1bde5fe4` |

**What the run offers** (the input of the draw, counted by `grounded eval export-verdicts`; the aggregates are also in the
baseline):

| Kind | Config | Candidates | Judge's verdicts |
|---|---|---:|---|
| faithfulness (one per claim) | `hybrid` | 53 | `SUPPORTED` 53 |
| correctness (one per answer) | `hybrid` | 30 | `CORRECT` 26, `PARTIALLY_CORRECT` 2, `INCORRECT` 2 |
| correctness (one per answer) | `no_rag` | 30 | `CORRECT` 17, `PARTIALLY_CORRECT` 4, `INCORRECT` 9 |

`no_rag` answers have no sources, so they have no faithfulness claims to judge. The 53 claims belong to the 23 `hybrid`
answers that were scored for faithfulness (the other 7 are refusals); a claim the judge decided without a call (no usable
source) would not be a candidate, and there was none.

**Which 20 items** (`grounded eval export-verdicts`, seed `411`, the ticket number, fixed before the run was looked at):

- **10 faithfulness items**: 6 real claims, each from a different question, and 4 synthetic negative controls (below).
- **10 correctness items**: 10 answers of 10 different questions, from both configs, drawn so that every grade the judge gave
  in the run appears (the run has all three). The split over grades and configs is in the key, not here: the grade of an item
  is the judge's verdict, and the labeling is blind.
- **How the draw works.** Nothing uses a random-number generator: the rank of a candidate is the sha256 of
  `seed | purpose | its key`, so the same seed and file give the same items on any Python version. The items are spread over
  the verdicts the run has as evenly as it allows (a lopsided run would otherwise give a sample on which agreement means
  little), within a verdict over the configs, and over different questions while there are any. The `item_id`s (`a01`...
  `a20`) follow a seeded shuffle of all 20 items, so they say nothing about kind, config or source.

**Negative controls.** The baseline run has no `NOT_SUPPORTED` faithfulness verdict at all (all 53 claims are `SUPPORTED`), so
a sample of real claims alone could not show whether the judge says `NOT_SUPPORTED` when it should. Four faithfulness items are
therefore **synthetic**: the claim of one question shown with the cited sources of a claim from a *different* question.

- Built only from real claims. The claim and the sources come from two questions that no real item of the sample shows, each
  question is used once, the two cite different sections (a section cited by both could support the claim) and at most 25% of
  the claim's content words (3+ letters, minus a short list of words every page shares) occur in the other sources
  (`--max-overlap`, recorded for each control in the key).
- Their **judge verdicts are real**: one call each to the real judge (`openai/gpt-oss-120b` on Groq, through the eval cache),
  made on 2026-10-08, because what is compared with the Author's labels is what the judge actually says, not what it should
  say. The 16 real items carry the verdict the judge gave in the run.
- On the sheet they are indistinguishable from the real items by their fields (the same columns are filled). A person who reads
  the evidence will see that the sources are about something else; that is what the right label is built on.
- They are **synthetic and easy**, and `grounded eval agreement` reports them separately (real items only, controls only),
  so the number can say what it means.

## Procedure (4.11b, the Author)

1. Read `v1.md` and write one label per item into the `human_label` column of `v1.csv` (the row with the same `item_id`).
   Allowed: `SUPPORTED` | `NOT_SUPPORTED` for a faithfulness item, `CORRECT` | `PARTIALLY_CORRECT` | `INCORRECT` for a
   correctness item. Case, spaces and a hyphen instead of an underscore do not matter. The definitions are in `v1.md`
   (verbatim from `backend/prompts/judge_faithfulness_v1.md` and `judge_correctness_v1.md`). Judge from the evidence on the
   page, not from your own knowledge of FastAPI.
2. **Do not open the key, the results file or the judge's reasons while labeling.**
3. Save as **CSV UTF-8** (in Excel, "Save As, CSV UTF-8"; opening the file with Data, From Text/CSV and the delimiter comma
   shows every column in its own cell). A semicolon-separated file, as Excel saves it in some locales, is read too.
4. Copy the key next to the labels (`eval/judge_agreement/v1.key.json`) and run, from `backend/`:

   ```bash
   uv run grounded eval agreement --labels ../eval/judge_agreement/v1.csv --key ../eval/judge_agreement/v1.key.json
   ```

   It refuses, listing every item at fault, a sheet with a missing or invalid label. Otherwise it prints exact agreement and
   Cohen's kappa (all items, per kind, real items only, controls only), the confusion matrices, the disagreements and the PRD
   §8 target (0.8), with `n` in every row. Kappa is reported as *undefined* when both raters used a single label.
5. The numbers go to the README (Evaluation) as the tool printed them, with `n`. If agreement is low, the README says so;
   changing a rubric afterwards is a new judge prompt version and a re-run (ticket 4.11).

To redraw the same sheet or complete a key (the controls need Groq quota; the command stops at the first quota error and a
re-run asks only for what is missing), from `backend/` and with the results file in `eval/results/`:

```bash
APP_ENV=eval GENERATOR_PROVIDERS=gemini GROQ_MODEL=openai/gpt-oss-120b JUDGE_MODEL=openai/gpt-oss-120b \
  uv run grounded eval export-verdicts --results ../eval/results/20261008T155539Z-generation.json \
  --git-sha 2a7d381f75f40791f6185a1e1ef80320a1cdfed8
```

It never overwrites a sheet that already holds labels. `--no-judge` writes the sheet, the view and a key whose controls are
pending.

## What this number can and cannot say

- **n = 20.** One item is 5 percentage points. Kappa on so few items, with labels this skewed, is unstable; read it with the
  confusion matrices.
- **Not a random sample.** The draw is stratified by the judge's own verdict, so that every grade appears. The agreement is
  the agreement on this sample, not an estimate of the judge's accuracy on the run.
- **The real faithfulness items are all `SUPPORTED` by the judge.** Agreement on them can only show that the Author also
  finds them supported (no false positives in 6 claims). Whether the judge catches an unsupported *real* claim is not measured
  by them; the four controls test it only for the easy case of clearly unrelated sources. A borderline unsupported claim
  (a detail that is not in the source) is not in this sample.
- **The controls are synthetic.** Their agreement is reported on its own and should not be averaged into a claim about real
  answers without saying so.
- **One rater, who wrote the rubrics.** The Author sees the same rubric text the judge saw, so the agreement partly measures
  a shared reading of it, not an independent human standard. The labels are made blind to the item's verdict, not to the
  baseline's aggregate numbers (they are public).
- **One run, one judge configuration** (`GROQ_REASONING_EFFORT=low`, temperature 0, `openai/gpt-oss-120b`). The PRD §12 open
  question about reasoning effort is answered only to the extent that this agreement is high.
