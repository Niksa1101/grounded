# Judge–human agreement (ticket 4.11)

PRD FR-22 asks for the LLM judge's agreement with a human, measured on at least 10 verdicts and published. This folder
holds the **sheet** the Author labeled by hand, how it was drawn, the key and the measured agreement ([Result](#result)).
The procedure is in [docs/Tech.md §15.4](../../docs/Tech.md).

| File | What it is | Committed |
|---|---|---|
| `v1.csv` | the labeling sheet: 20 items with their evidence and the Author's labels in the `human_label` column (empty in the blind sheet of 4.11a, which is what the Author labeled). **No verdict, reason, score, config name or control mark in it.** UTF-8 with a BOM, so Excel opens it as UTF-8 | yes |
| `v1.md` | the same 20 items laid out to be read (long evidence is painful in a CSV cell), with how to label and the label definitions copied verbatim from the two rubrics | yes |
| `v1.key.json` | the **key**: what each `item_id` is (kind, real or control, config, question, claim) and what the judge said (verdict, reason, rubric version, model). A byte-identical copy of the file below, committed in 4.11b once the labels were in. It holds the judge's verdicts and reasons, the control construction and the results file's sha256, and nothing secret or machine-specific | yes (4.11b) |
| `v1.agreement.md` | the output of `grounded eval agreement` on `v1.csv` and `v1.key.json`, verbatim under a one-line header. A test fails when it differs from what the command prints | yes (4.11b) |
| `eval/results/judge-agreement-v1.key.json` | where `export-verdicts` writes the key. Gitignored with the rest of `eval/results/`, and kept out of the repository while the labels were made | no |
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

## Result

Measured in 4.11b. The Author labeled the sheet, the key was copied next to it (`v1.key.json`), and
`grounded eval agreement` printed the output in [v1.agreement.md](v1.agreement.md); the same output is pasted in the
[README](../../README.md#judge-human-agreement), and a test fails if either differs from what the command prints. Every
number in this section is copied from that output; the sentences are the reading. The only cells of `v1.csv` that differ
from the blind sheet committed in 4.11a are the 20 `human_label` cells (compared cell by cell, in the 4.11b PR). The labels
were made on the blind sheet, as the procedure asks; that is the Author's account, since nothing in the repository can
show whether the key or the judge's reasons were opened first.

| Scope | n | Exact agreement | Cohen's kappa |
|---|---:|---:|---:|
| All items (labels of both kinds pooled) | 20 | 85.0% (17/20) | 0.808 |
| Faithfulness | 10 | 100.0% (10/10) | 1.000 |
| Correctness | 10 | 70.0% (7/10) | 0.538 |
| Real items only (pooled) | 16 | 81.2% (13/16) | 0.741 |
| Faithfulness, real items only | 6 | 100.0% (6/6) | undefined |
| Controls only (synthetic) | 4 | 100.0% (4/4) | undefined |

Kappa is undefined for the last two rows because both raters used one label for every item (`SUPPORTED` for the real
claims, `NOT_SUPPORTED` for the controls). Confusion matrices, rows the human, columns the judge:

| Faithfulness (n = 10) | SUPPORTED | NOT_SUPPORTED | total |
|---|---:|---:|---:|
| SUPPORTED | 6 | 0 | 6 |
| NOT_SUPPORTED | 0 | 4 | 4 |
| total | 6 | 4 | 10 |

| Correctness (n = 10) | CORRECT | PARTIALLY_CORRECT | INCORRECT | total |
|---|---:|---:|---:|---:|
| CORRECT | 4 | 1 | 0 | 5 |
| PARTIALLY_CORRECT | 0 | 1 | 1 | 2 |
| INCORRECT | 0 | 1 | 2 | 3 |
| total | 4 | 3 | 3 | 10 |

**PRD §8 (0.8 desired).** On all 20 items exact agreement (0.850) and kappa (0.808) both meet it. On the 16 real items
exact agreement (0.812) meets it and kappa (0.741) does not. PRD §8 does not say which measure the 0.8 is, or whether the
controls and the two kinds count together; the tool prints both. The faithfulness 10/10 is the part that lifts the pooled
numbers, and 4 of those 10 are the easy synthetic controls. **For correctness, the kind where the judge and the Author
actually differ, the target is not met** (70.0%, kappa 0.538).

### The three disagreements

All three are correctness grades of real answers, and each is one step apart on the three-grade scale; none is `CORRECT`
against `INCORRECT`. The judge's reason is quoted from `v1.key.json`. The last column describes why the two probably differ,
from the evidence on `v1.md`; it does not say who is right.

| Item | Author | Judge | Judge's reason | Why they probably differ |
|---|---|---|---|---|
| **a07**, correctness, `hybrid`, q043 (*return a 401 with a WWW-Authenticate header from a dependency*) | `INCORRECT` | `PARTIALLY_CORRECT` | "The candidate mentions that FastAPI security utilities can automatically return a 401 with a WWW-Authenticate header, but does not describe the explicit method of raising HTTPException with custom headers as the reference does." | The reference's answer is to raise `HTTPException(401, headers=...)`. The candidate never shows that: its code is an `HTTPBearer` subclass that returns a 403, and its prose says the built-in security classes already send the 401 and `WWW-Authenticate`, with `Bearer` as the value. The judge credits those true side facts; the Author counts the missing method (and the 403 example) as missing the main point. The judge's own reason names the missing method, which is the reference's main point, and does not mention the 403. The judge is the more lenient here. |
| **a09**, correctness, `no_rag`, q029 (*endpoint whose `q` takes several values, default `["foo", "bar"]`*) | `PARTIALLY_CORRECT` | `INCORRECT` | "The candidate's code is syntactically wrong (missing a parenthesis) and does not use the required Annotated form; it also omits the note that Query() is needed to avoid treating the list as a body." | The candidate uses `q: List[str] = Query(default=["foo", "bar"])`, an older spelling of the same idea, but the signature lacks its closing parenthesis and there is no note that `Query()` is needed. The Author treats the right idea with a typo as partly right; the judge counts the syntax error, the missing note and the absence of `Annotated` (which the reference uses and the question does not ask for) as enough for `INCORRECT`. The judge is the stricter here. |
| **a16**, correctness, `hybrid`, q036 (*PATCH endpoint that updates only the fields the client sent*) | `CORRECT` | `PARTIALLY_CORRECT` | "The candidate correctly describes using `exclude_unset=True`, `model_copy`, and `jsonable_encoder` to apply partial updates, matching the reference steps, but omits the detail that the input model must have all fields optional." | The candidate has every step of the reference (`model_dump(exclude_unset=True)`, `model_copy(update=...)`, `jsonable_encoder`, return the model) and leaves out only its last sentence, that the input model needs all fields optional. The Author treats that sentence as a caveat; the judge as an important part left out. The rubric separates `CORRECT` ("the same essential information") from `PARTIALLY_CORRECT` ("leaves out an important part") and the sentence sits on that line. The judge is the stricter here. |

The three sit on the correctness rubric's own boundaries (*the main point* against *an important part*, and *a wrong detail
that does not reverse the main point*), where two readers can resolve the same answer differently. Three items cannot say
which reader is the outlier, and the errors go both ways (the judge stricter in two, more lenient in one), so they do not
show a one-directional bias. A change to the correctness rubric would be a new judge prompt version (`judge_correctness_v2`),
a new baseline and a new sheet; that is not part of this ticket and is an open item in [PRD §12](../../docs/PRD.md).

**What this says about the baseline's faithfulness 1.000.** A little more credible, not validated. The judge is not a
rubber stamp: it said `NOT_SUPPORTED` for all 4 controls, and the Author agreed with it on the 6 real claims. But the judge
said `SUPPORTED` for all 53 real claims of the baseline run, so the 6 real items can only show agreement on supported claims,
the controls are synthetic and easy (the sources are about something else, at most 25% word overlap), and a subtly
unsupported real claim is not in the sample. Read 1.000 as "the judge found no unsupported claim".

## Procedure (4.11b, the Author; done for v1)

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
  question about reasoning effort is answered only to the extent that this agreement is high: the faithfulness items agreed
  on all 10, and the correctness disagreements (see [Result](#result)) cannot be attributed to the effort setting, because no
  second run at a higher effort exists to compare with.
