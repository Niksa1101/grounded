#!/usr/bin/env bash
# Create or update the one eval comment of a pull request (Tech.md §17, PRD FR-24).
#
#   eval-comment.sh <comment-file>
#
# The comment is found by the marker on its first line, <!-- grounded-eval -->, among the comments
# the Actions bot wrote, so a later push edits it and never adds a second one. Environment:
# GH_TOKEN (the job's token, `pull-requests: write`), REPO (owner/name) and PR_NUMBER.
#
# A failure to post is a warning, not a failed step: the same text is in the job summary, and the
# verdict of the run belongs to the gate, not to the comment.
set -uo pipefail

file=${1:?usage: eval-comment.sh <comment-file>}
: "${GH_TOKEN:?GH_TOKEN is required}" "${REPO:?REPO is required}" "${PR_NUMBER:?PR_NUMBER is required}"

if [ ! -s "$file" ]; then
  echo "::warning title=No PR comment::$file is missing or empty, so there is nothing to post."
  exit 0
fi

# `contains` on the marker, and only the Actions bot's own comments: a person who pastes the marker
# into a comment does not get theirs overwritten (or the job failing on a 403).
ids=$(gh api --paginate "repos/$REPO/issues/$PR_NUMBER/comments" \
  --jq '.[] | select(.user.login == "github-actions[bot]" and (.body | contains("<!-- grounded-eval -->"))) | .id') || {
  echo "::warning title=PR comment not posted::Could not list the comments of #$PR_NUMBER."
  exit 0
}
id=${ids%%$'\n'*}

if [ -n "$id" ]; then
  if gh api -X PATCH "repos/$REPO/issues/comments/$id" -F body=@"$file" > /dev/null; then
    echo "Updated the eval comment $id on #$PR_NUMBER."
  else
    echo "::warning title=PR comment not updated::Could not update comment $id on #$PR_NUMBER."
  fi
else
  if gh api -X POST "repos/$REPO/issues/$PR_NUMBER/comments" -F body=@"$file" > /dev/null; then
    echo "Posted a new eval comment on #$PR_NUMBER."
  else
    echo "::warning title=PR comment not posted::Could not post a comment on #$PR_NUMBER."
  fi
fi
exit 0
