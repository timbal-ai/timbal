#!/usr/bin/env bash
# Upload or replace CLI binaries (and manifest.json) on an *existing* GitHub
# release v$VERSION. Does not create tags, create/delete releases, or edit notes.
#
# Usage:
#   ./scripts/upload-cli-release-assets.sh <VERSION>
# Example:
#   ./scripts/upload-cli-release-assets.sh 0.42.0
#
# Requires: git tag v<VERSION>, GitHub release v<VERSION>, gh CLI, zig.
set -euo pipefail

gh_repo_from_origin() {
    local url
    url=$(git config --get remote.origin.url) || return 1
    case "$url" in
        git@github.com:*)
            url="${url#git@github.com:}"
            ;;
        https://github.com/*)
            url="${url#https://github.com/}"
            ;;
        ssh://git@github.com/*)
            url="${url#ssh://git@github.com/}"
            ;;
        *)
            return 1
            ;;
    esac
    url="${url%.git}"
    url="${url%/}"
    echo "$url"
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GH_REPO=$(gh_repo_from_origin) || {
    echo "Error: could not derive owner/repo from git remote origin (use a github.com URL)."
    exit 1
}

VERSION="${1:-}"
if [ -z "$VERSION" ]; then
    echo "Usage: $0 <VERSION>"
    echo "  VERSION without leading v (e.g. 0.42.0). Tag v\$VERSION must exist."
    exit 1
fi

TAG="v${VERSION}"

if ! git rev-parse "${TAG}" >/dev/null 2>&1; then
    echo "Error: git tag ${TAG} does not exist."
    exit 1
fi

if ! gh release view "${TAG}" -R "${GH_REPO}" >/dev/null 2>&1; then
    echo "Error: GitHub release ${TAG} not found. Create the release first (e.g. scripts/release.sh), then run this to refresh assets only."
    exit 1
fi

COMMIT_HASH=$(git rev-parse --short "${TAG}")
COMMIT_DATE=$(git log -1 --format=%cs "${TAG}")

cd "${REPO_ROOT}/cli"

if [ -d "zig-out/$VERSION" ]; then
    echo "Build output for version $VERSION already exists."
    read -r -p "Rebuild before upload? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        zig build -Dversion="$VERSION" -Dcommit_hash="$COMMIT_HASH" -Dcommit_date="$COMMIT_DATE" -Doptimize=ReleaseSmall
    else
        echo "Using existing binaries under zig-out/$VERSION"
    fi
else
    zig build -Dversion="$VERSION" -Dcommit_hash="$COMMIT_HASH" -Dcommit_date="$COMMIT_DATE" -Doptimize=ReleaseSmall
fi

cd ..

RELEASE_URL="https://github.com/timbal-ai/timbal/releases/download/${TAG}"
ASSETS_PATH="cli/zig-out/$VERSION"

if [ "$(uname -s)" = "Darwin" ]; then
    SHASUM_CMD="shasum -a 256"
else
    SHASUM_CMD="sha256sum"
fi

LINUX_AARCH64_NAME="timbal-${VERSION}-linux-aarch64-gnu"
LINUX_AARCH64_SHA=$($SHASUM_CMD "$ASSETS_PATH/$LINUX_AARCH64_NAME" | awk '{print $1}')
LINUX_X86_64_NAME="timbal-${VERSION}-linux-x86_64-gnu"
LINUX_X86_64_SHA=$($SHASUM_CMD "$ASSETS_PATH/$LINUX_X86_64_NAME" | awk '{print $1}')
MACOS_AARCH64_NAME="timbal-${VERSION}-macos-aarch64"
MACOS_AARCH64_SHA=$($SHASUM_CMD "$ASSETS_PATH/$MACOS_AARCH64_NAME" | awk '{print $1}')
MACOS_X86_64_NAME="timbal-${VERSION}-macos-x86_64"
MACOS_X86_64_SHA=$($SHASUM_CMD "$ASSETS_PATH/$MACOS_X86_64_NAME" | awk '{print $1}')
WINDOWS_X86_64_NAME="timbal-${VERSION}-windows-x86_64-gnu.exe"
WINDOWS_X86_64_SHA=$($SHASUM_CMD "$ASSETS_PATH/$WINDOWS_X86_64_NAME" | awk '{print $1}')

MANIFEST_PATH="$ASSETS_PATH/manifest.json"
cat >"$MANIFEST_PATH" <<EOF
{
    "name": "timbal",
    "description": "Timbal is an open-source python framework for building reliable AI applications, battle-tested in production with simple, transparent architecture that eliminates complexity while delivering blazing fast performance, robust typing, and API stability in an ever-changing ecosystem.",
    "version": "$VERSION",
    "binaries": {
        "linux": {
            "aarch64": {
                "url": "$RELEASE_URL/$LINUX_AARCH64_NAME",
                "sha256": "$LINUX_AARCH64_SHA"
            },
            "x86_64": {
                "url": "$RELEASE_URL/$LINUX_X86_64_NAME",
                "sha256": "$LINUX_X86_64_SHA"
            }
        },
        "darwin": {
            "arm64": {
                "url": "$RELEASE_URL/$MACOS_AARCH64_NAME",
                "sha256": "$MACOS_AARCH64_SHA"
            },
            "x86_64": {
                "url": "$RELEASE_URL/$MACOS_X86_64_NAME",
                "sha256": "$MACOS_X86_64_SHA"
            }
        },
        "windows": {
            "x86_64": {
                "url": "$RELEASE_URL/$WINDOWS_X86_64_NAME",
                "sha256": "$WINDOWS_X86_64_SHA"
            }
        }
    }
}
EOF

ASSETS=(
    "$ASSETS_PATH/$LINUX_AARCH64_NAME"
    "$ASSETS_PATH/$LINUX_X86_64_NAME"
    "$ASSETS_PATH/$MACOS_AARCH64_NAME"
    "$ASSETS_PATH/$MACOS_X86_64_NAME"
    "$ASSETS_PATH/$WINDOWS_X86_64_NAME"
    "$MANIFEST_PATH"
)

echo "Uploading assets to existing release ${TAG} (replace if present)..."
gh release upload "${TAG}" "${ASSETS[@]}" --clobber -R "${GH_REPO}"

echo "Done. Release: https://github.com/timbal-ai/timbal/releases/tag/${TAG}"
