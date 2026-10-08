#!/usr/bin/env bash
# Compose the Markdown report of one eval.yml run, from what the earlier steps left behind
# (Tech.md §17). It writes $RESULTS_DIR/comment.md (the PR comment, which starts with the marker
# that eval-comment.sh finds it by), appends the same text to the job summary, and prints
# `status=<value>` to $GITHUB_OUTPUT. It never fails on a missing input: whatever happened, the
# report says so (a silent green is the one outcome this must not produce, AGENTS.md §7).
#
# Inputs, all environment variables:
#   RESULTS_DIR      where the earlier steps wrote gate-report.md, gate-errors.txt and promptfoo.log
#   SETUP_OUTCOME    outcome of the shared setup step (success | failure | cancelled | skipped)
#   PROMPTFOO_OUTCOME, PROMPTFOO_EXIT   outcome of the promptfoo step and promptfoo's own exit code
#   GATE_OUTCOME, GATE_EXIT             outcome of the gate step and the gate's exit code
#   JOB_STATUS       the job's status so far; "cancelled" when the run is being cancelled
#   RUN_URL, ARTIFACT_URL, HEAD_SHA, EVENT_NAME, GEMINI_MODEL, JUDGE_MODEL   for the footer
#   CACHE_MATCHED_KEY, CACHE_ENTRIES_BEFORE, CACHE_ENTRIES_AFTER   the eval LLM cache, for the footer
#
# `status` is one of: pass, fail, inconclusive (the gate's verdicts, exit 0 or 1), cannot-run (the
# gate's exit 2), not-run (a step before the gate did not finish), cancelled, unknown (the gate
# exited 0 with a report this script cannot classify). Only pass and inconclusive let the job
# succeed.
set -uo pipefail

results=${RESULTS_DIR:-eval/results}
mkdir -p "$results" # absent when the setup failed before promptfoo ran
report="$results/gate-report.md"
errors="$results/gate-errors.txt"
log="$results/promptfoo.log"
out="$results/comment.md"
marker='<!-- grounded-eval -->'

setup=${SETUP_OUTCOME:-skipped}
promptfoo=${PROMPTFOO_OUTCOME:-skipped}
gate=${GATE_OUTCOME:-skipped}
gate_exit=${GATE_EXIT:-}

# --- Which outcome is this? ----------------------------------------------------------------------
status=unknown
if [ "${JOB_STATUS:-}" = "cancelled" ] && [ "$gate" != "success" ]; then
  status=cancelled
elif [ "$setup" != "success" ] || [ "$promptfoo" != "success" ] || [ "$gate" != "success" ]; then
  status=not-run
elif [ "$gate_exit" = "0" ]; then
  headline=$(head -n 1 "$report" 2> /dev/null || true)
  case "$headline" in
    "### Generation gate: "*inconclusive*) status=inconclusive ;;
    "### Generation gate: "*pass*) status=pass ;;
  esac
elif [ "$gate_exit" = "1" ]; then
  status=fail
elif [ -n "$gate_exit" ]; then
  status=cannot-run
fi

case "$status" in
  pass) label="✅ pass" ;;
  fail) label="❌ fail" ;;
  inconclusive) label="⚠️ inconclusive" ;;
  cannot-run) label="❌ the gate could not run" ;;
  not-run) label="❌ the eval did not run" ;;
  cancelled) label="⚪ cancelled" ;;
  *) label="❓ unknown" ;;
esac

# --- Helpers -------------------------------------------------------------------------------------
# A fenced block of a file's last lines, without terminal colors; the fence is longer than any
# backtick run the text could contain.
tail_block() {
  local file=$1 lines=$2
  [ -s "$file" ] || return 0
  echo '````text'
  tail -n "$lines" "$file" | sed -E 's/\x1b\[[0-9;]*[A-Za-z]//g' | cut -c1-300
  echo '````'
}

# The end of the promptfoo log, folded away: it is what explains a crash, and noise otherwise.
log_details() {
  [ -s "$log" ] || return 0
  echo
  echo "<details><summary>Last lines of the promptfoo log</summary>"
  echo
  tail_block "$log" 30
  echo
  echo "</details>"
}

promptfoo_note() {
  if [ -n "${PROMPTFOO_EXIT:-}" ]; then
    echo "promptfoo exited with code ${PROMPTFOO_EXIT}; its code is ignored, the gate decides."
  fi
}

# --- The report ----------------------------------------------------------------------------------
{
  echo "## Generation eval: $label"
  echo
  case "$status" in
    pass | fail | inconclusive)
      cat "$report"
      # What the errors were (a quota, a 5xx, a timeout), which the gate's counts do not say.
      kinds=$(python3 -I "$(dirname "$0")/eval-error-kinds.py" "$results/generation.json" 2> /dev/null || true)
      if [ -n "$kinds" ]; then
        echo
        echo "Errors by kind, of the generator and of the judge:"
        echo
        echo "$kinds"
      fi
      if [ "$status" = "inconclusive" ]; then
        echo
        echo "**An inconclusive run is not a pass.** More than 20% of a config's cases ended in a provider"
        echo "error (a quota, a 5xx or a timeout), so its metrics are shown with their \`n\` but not gated."
        echo "The job does not fail on it. Re-run when the quota has reset or the provider has recovered:"
        echo "the eval cache keeps every call that was already paid for, so a re-run sends only the calls"
        echo "still missing."
      fi
      ;;
    cannot-run)
      echo "The gate produced no verdict: a missing or broken results or baseline file, a rejected key or a"
      echo "missing judge configuration. This is neither a pass nor a quality fail."
      echo
      tail_block "$errors" 40
      echo
      promptfoo_note
      log_details
      ;;
    not-run)
      if [ "$setup" != "success" ]; then
        echo "A setup step failed before the eval started (restore, migrate or ingest). When the job summary"
        echo "names an **infrastructure failure**, the embeddings were unavailable (no key on a cold cache, or"
        echo "the daily embedding quota): that is the provider's side, not a quality result."
      elif [ "$promptfoo" != "success" ]; then
        echo "The promptfoo step did not finish (it ran out of its time limit or was killed)."
        echo
        promptfoo_note
        log_details
      else
        echo "The gate step itself failed before it could report."
      fi
      ;;
    cancelled)
      echo "The run was cancelled before the gate ran (a newer push to this PR, or a manual cancel). The"
      echo "calls already paid for are in the eval cache."
      ;;
    *)
      echo "The gate exited 0 but its report does not start with a known verdict, so the run is not"
      echo "classified as a pass."
      echo
      tail_block "$report" 20
      ;;
  esac
  echo
  sha=${HEAD_SHA:-unknown}
  footer="commit \`${sha:0:7}\` · ${EVENT_NAME:-run}"
  [ -n "${GEMINI_MODEL:-}" ] && footer="$footer · generator \`${GEMINI_MODEL}\`"
  [ -n "${JUDGE_MODEL:-}" ] && footer="$footer · judge \`${JUDGE_MODEL}\`"
  [ -n "${RUN_URL:-}" ] && footer="$footer · [workflow run](${RUN_URL})"
  [ -n "${ARTIFACT_URL:-}" ] && footer="$footer · [report artifact](${ARTIFACT_URL}) (promptfoo HTML and JSON, gate report)"
  echo "<sub>$footer</sub>"
  if [ -n "${CACHE_ENTRIES_AFTER:-}" ]; then
    restored=${CACHE_MATCHED_KEY:-nothing}
    echo
    echo "<sub>Eval LLM cache: restored \`${restored}\` (${CACHE_ENTRIES_BEFORE:-0} entries), ${CACHE_ENTRIES_AFTER} after the run.</sub>"
  fi
} > "$out.body"

# A PR comment is limited to 65,536 characters.
if [ "$(wc -c < "$out.body")" -gt 60000 ]; then
  head -c 60000 "$out.body" > "$out.cut"
  printf '\n\n_(truncated; the full report is in the job summary)_\n' >> "$out.cut"
  mv "$out.cut" "$out.body"
fi

{
  echo "$marker"
  cat "$out.body"
} > "$out"

if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  cat "$out.body" >> "$GITHUB_STEP_SUMMARY"
fi
if [ -n "${GITHUB_OUTPUT:-}" ]; then
  echo "status=$status" >> "$GITHUB_OUTPUT"
fi
cat "$out.body"
rm -f "$out.body"
exit 0
