#!/usr/bin/env bash
# Backport a PR that is merged into main onto a release branch, through a PR of its own.
#
# Usage:
#   scripts/backport.sh <pr-number> [<line>]     # line: 2.8 -> release/2.8 (default: the newest release/*)
#
# It cherry-picks the PR's merge commit (one commit that names where it came from) onto a new
# branch `backport/<pr>-to-<line>` cut from `origin/release/<line>`, pushes it and opens a PR into
# the release branch, where CI runs on it. On a conflict it stops and prints the commands that
# finish the job by hand. BACKPORT_DRY_RUN=1 does everything but the push and the PR. See RELEASING.md.
set -euo pipefail

die() { echo "Error: $*" >&2; exit 1; }

PR="${1:-}"
LINE="${2:-}"
[[ "$PR" =~ ^[0-9]+$ ]] || die "usage: $0 <pr-number> [<line, e.g. 2.8>]"

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
git fetch --quiet origin

if [ -z "$LINE" ]; then
    LINE=$(git for-each-ref --format='%(refname:lstrip=4)' 'refs/remotes/origin/release/' \
        | grep -E '^[0-9]+\.[0-9]+$' | sort -t. -k1,1n -k2,2n | tail -1)
    [ -n "$LINE" ] || die "no origin/release/<X.Y> branch found; pass the line explicitly."
fi
TARGET="release/$LINE"
git rev-parse --verify --quiet "origin/$TARGET" >/dev/null || die "origin/$TARGET does not exist."

read -r STATE BASE SHA < <(gh pr view "$PR" --json state,baseRefName,mergeCommit \
    --jq '"\(.state) \(.baseRefName) \(.mergeCommit.oid // "-")"')
[ "$STATE" = "MERGED" ] || die "#$PR is $STATE: only PRs merged into main are backported."
[ "$BASE" = "main" ] || die "#$PR was merged into $BASE, not main."
TITLE=$(gh pr view "$PR" --json title --jq .title)
git rev-parse --verify --quiet "$SHA^{commit}" >/dev/null || die "merge commit $SHA of #$PR is not in this clone."

if git merge-base --is-ancestor "$SHA" "origin/$TARGET"; then
    die "#$PR ($SHA) is already in $TARGET."
fi
RELEASE_ONLY=$(git log --format=%B "origin/main..origin/$TARGET")   # not `| grep -q`: with pipefail it can miss
if [[ "$RELEASE_ONLY" == *"cherry picked from commit $SHA"* ]]; then
    die "#$PR ($SHA) was already backported to $TARGET."
fi

BRANCH="backport/$PR-to-$LINE"
if git ls-remote --exit-code --heads origin "$BRANCH" >/dev/null 2>&1; then
    die "origin/$BRANCH already exists (an open backport?)."
fi

# A merge commit is picked against its first parent (main), so the commit carries the whole PR.
PARENTS=$(git rev-list --parents -n 1 "$SHA" | wc -w | tr -d ' ')
PICK=(git cherry-pick --no-commit)
if [ "$PARENTS" -gt 2 ]; then PICK+=(-m 1); fi

TMP_ROOT="${TMPDIR:-/tmp}"
WORKTREE=$(mktemp -d "${TMP_ROOT%/}/timbal-backport-$PR-XXXXXX")
git worktree add --quiet -b "$BRANCH" "$WORKTREE" "origin/$TARGET"
cleanup() { git worktree remove --force "$WORKTREE" 2>/dev/null || true; git branch --quiet -D "$BRANCH" 2>/dev/null || true; }

# The commit says what it is, not "Merge pull request #N from ...", and records its source like `-x`.
MSG_FILE="$WORKTREE.msg"
printf '%s (#%s)\n\nBackport of #%s to %s.\n\n(cherry picked from commit %s)\n' \
    "$TITLE" "$PR" "$PR" "$TARGET" "$SHA" > "$MSG_FILE"

BODY="Backport of #$PR to \`$TARGET\` (merge commit $SHA).

See RELEASING.md for the release-branch flow."
PR_CMD=(gh pr create --base "$TARGET" --head "$BRANCH" --title "[$LINE] $TITLE (backport #$PR)" --body "$BODY")
PR_CMD_TEXT=$(printf '%q ' "${PR_CMD[@]}")

if ! (cd "$WORKTREE" && "${PICK[@]}" "$SHA"); then
    cat >&2 <<EOF

The cherry-pick of #$PR onto $TARGET has conflicts. Resolve them in:
    $WORKTREE
then finish with:
    cd "$WORKTREE"
    git add -A && git commit -e -F "$MSG_FILE"     # add what you changed to resolve it, and why
    git push -u origin "$BRANCH"
    $PR_CMD_TEXT
    cd "$REPO_ROOT" && git worktree remove "$WORKTREE" && git branch -D "$BRANCH" && rm -f "$MSG_FILE"
To give up instead:
    cd "$REPO_ROOT" && git worktree remove --force "$WORKTREE" && git branch -D "$BRANCH" && rm -f "$MSG_FILE"
EOF
    exit 1
fi

if (cd "$WORKTREE" && git diff --cached --quiet); then
    cleanup
    rm -f "$MSG_FILE"
    die "#$PR changes nothing on $TARGET (already there in another form?)."
fi
(cd "$WORKTREE" && git commit --quiet -F "$MSG_FILE")
rm -f "$MSG_FILE"
if [ -n "${BACKPORT_DRY_RUN:-}" ]; then   # everything but the push and the PR
    (cd "$WORKTREE" && git log -1 --stat --format='%h %s%n%n%b' | head -40)
    echo "Dry run: would push $BRANCH and run: $PR_CMD_TEXT"
    cleanup
    exit 0
fi
(cd "$WORKTREE" && git push --quiet -u origin "$BRANCH")
URL=$("${PR_CMD[@]}")
cleanup
echo "Opened $URL"
