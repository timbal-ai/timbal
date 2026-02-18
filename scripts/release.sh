#!/bin/bash
set -e

VERSION="$1"
if [ -z "$VERSION" ]; then
    echo "Usage: $0 <new-version>"
    exit 1
fi

# Check if git tag exists and get commit hash
if ! TAG_COMMIT=$(git rev-parse "v$VERSION" 2>/dev/null); then
    echo "Error: Git tag v$VERSION does not exist. Please create the tag first."
    exit 1
fi

# Extract short commit hash and commit date for the version string.
COMMIT_HASH=$(git rev-parse --short "v$VERSION")
COMMIT_DATE=$(git log -1 --format=%cs "v$VERSION")

# TODO Add building the python package.

cd cli

# Check if build output already exists
if [ -d "zig-out/$VERSION" ]; then
    echo "Build output for version $VERSION already exists."
    read -p "Do you want to rebuild? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping build step..."
        # Skip to asset handling
    else
        echo "Rebuilding..."
        zig build -Dversion="$VERSION" -Dcommit_hash="$COMMIT_HASH" -Dcommit_date="$COMMIT_DATE" -Doptimize=ReleaseSmall
    fi
else
    zig build -Dversion="$VERSION" -Dcommit_hash="$COMMIT_HASH" -Dcommit_date="$COMMIT_DATE" -Doptimize=ReleaseSmall
fi

cd ..

# All release assets are going to be uploaded here.
RELEASE_URL="https://github.com/timbal-ai/timbal/releases/download/v$VERSION"

ASSETS_PATH="cli/zig-out/$VERSION"

# Determine the correct sha256sum command based on the OS.
if [ "$(uname -s)" = "Darwin" ]; then
    SHASUM_CMD="shasum -a 256"
else
    SHASUM_CMD="sha256sum"
fi

# ! We need to modify this if we want to add more targets.
LINUX_AARCH64_NAME="timbal-${VERSION}-linux-aarch64-gnu"
LINUX_AARCH64_SHA=$($SHASUM_CMD $ASSETS_PATH/$LINUX_AARCH64_NAME | awk '{print $1}')
LINUX_X86_64_NAME="timbal-${VERSION}-linux-x86_64-gnu"
LINUX_X86_64_SHA=$($SHASUM_CMD $ASSETS_PATH/$LINUX_X86_64_NAME | awk '{print $1}')
MACOS_AARCH64_NAME="timbal-${VERSION}-macos-aarch64"
MACOS_AARCH64_SHA=$($SHASUM_CMD $ASSETS_PATH/$MACOS_AARCH64_NAME | awk '{print $1}')
MACOS_X86_64_NAME="timbal-${VERSION}-macos-x86_64"
MACOS_X86_64_SHA=$($SHASUM_CMD $ASSETS_PATH/$MACOS_X86_64_NAME | awk '{print $1}')
WINDOWS_X86_64_NAME="timbal-${VERSION}-windows-x86_64-gnu.exe"
WINDOWS_X86_64_SHA=$($SHASUM_CMD $ASSETS_PATH/$WINDOWS_X86_64_NAME | awk '{print $1}')

# Generate the manifest file. This file will be used by the installation scripts to determine the exact binaries to download.
MANIFEST_PATH="$ASSETS_PATH/manifest.json"
cat > "$MANIFEST_PATH" <<EOF
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

# Check if GitHub release already exists
if gh release view "v$VERSION" > /dev/null 2>&1; then
    echo "GitHub release v$VERSION already exists."
    read -p "Do you want to override it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Deleting existing release v$VERSION..."
        gh release delete "v$VERSION" --yes
    else
        echo "Aborting release creation."
        exit 0
    fi
fi

echo "Creating release v$VERSION from tag (commit: $TAG_COMMIT)..."
gh release create "v$VERSION" \
    "${ASSETS[@]}" \
    --title "v$VERSION" \
    --notes "Release v$VERSION" \
    --target "$TAG_COMMIT"
