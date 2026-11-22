#!/usr/bin/env bash
# Build timbal from a clean checkout of tag v<VERSION> and upload it to PyPI.
#
# Usage:
#   scripts/publish-pypi.sh <VERSION> [--dry-run]
#
# The version comes from the tag (hatch-vcs), so building from a detached worktree of the tag, not
# from whatever is checked out, gives exactly <VERSION>: no .devN, no +g<hash>, no dirty marker.
# Credentials are uv's: UV_PUBLISH_TOKEN, or `uv publish`'s own prompts. --dry-run builds and checks
# the files, then runs `uv publish --dry-run`, which uploads nothing.
# PyPI never accepts a version twice, even after it is deleted. See RELEASING.md.
set -euo pipefail

die() { echo "Error: $*" >&2; exit 1; }

VERSION="${1:-}"
DRY_RUN="${2:-}"
[ -n "$VERSION" ] || die "usage: $0 <VERSION> [--dry-run]"
[ -z "$DRY_RUN" ] || [ "$DRY_RUN" = "--dry-run" ] || die "unknown option $DRY_RUN"

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
git rev-parse --verify --quiet "v$VERSION^{commit}" >/dev/null || die "tag v$VERSION does not exist."

TMP_ROOT="${TMPDIR:-/tmp}"
BUILD_ROOT=$(mktemp -d "${TMP_ROOT%/}/timbal-pypi-$VERSION-XXXXXX")
cleanup() { git worktree remove --force "$BUILD_ROOT/src" 2>/dev/null || true; rm -rf "$BUILD_ROOT"; }
trap cleanup EXIT

git worktree add --quiet --detach "$BUILD_ROOT/src" "v$VERSION"
(cd "$BUILD_ROOT/src" && uv build --quiet --out-dir "$BUILD_ROOT/dist")

SDIST="$BUILD_ROOT/dist/timbal-$VERSION.tar.gz"
[ -f "$SDIST" ] || die "expected $(basename "$SDIST"), built: $(ls "$BUILD_ROOT/dist" | tr '\n' ' ')"
WHEELS=$(ls "$BUILD_ROOT/dist" | grep -c "^timbal-$VERSION-.*\.whl$" || true)
[ "$WHEELS" -ge 1 ] || die "no timbal-$VERSION wheel built: $(ls "$BUILD_ROOT/dist" | tr '\n' ' ')"

echo "Built:"
ls -1 "$BUILD_ROOT/dist" | sed 's/^/  /'
if [ -n "$DRY_RUN" ]; then
    uv publish --dry-run "$BUILD_ROOT/dist"/*
    echo "Dry run: nothing was uploaded."
else
    uv publish "$BUILD_ROOT/dist"/*
    echo "Published timbal $VERSION: https://pypi.org/project/timbal/$VERSION/"
fi
