#!/bin/bash
set -e

VERSION="$1"
if [ -z "$VERSION" ]; then
    echo "Usage: $0 <new-version>"
    exit 1
fi

# TODO Add building the python package.

cd cli
zig build -Dversion="$VERSION" -Doptimize=ReleaseSmall
cd ..

# All release assets are going to be uploaded here.
RELEASE_URL="https://github.com/timbal-ai/timbal/releases/download/v$VERSION"

ASSETS_PATH="cli/zig-out/$VERSION"
# ! We need to modify this if we want to add more targets.
LINUX_AARCH64_NAME="timbal-${VERSION}-linux-aarch64-gnu"
LINUX_AARCH64_SHA=$(sha256sum $ASSETS_PATH/$LINUX_AARCH64_NAME | awk '{print $1}')
LINUX_X86_64_NAME="timbal-${VERSION}-linux-x86_64-gnu"
LINUX_X86_64_SHA=$(sha256sum $ASSETS_PATH/$LINUX_X86_64_NAME | awk '{print $1}')
MACOS_AARCH64_NAME="timbal-${VERSION}-macos-aarch64"
MACOS_AARCH64_SHA=$(sha256sum $ASSETS_PATH/$MACOS_AARCH64_NAME | awk '{print $1}')
MACOS_X86_64_NAME="timbal-${VERSION}-macos-x86_64"
MACOS_X86_64_SHA=$(sha256sum $ASSETS_PATH/$MACOS_X86_64_NAME | awk '{print $1}')
WINDOWS_X86_64_NAME="timbal-${VERSION}-windows-x86_64-gnu.exe"
WINDOWS_X86_64_SHA=$(sha256sum $ASSETS_PATH/$WINDOWS_X86_64_NAME | awk '{print $1}')

# Generate the manifest file. This file will be used by the installation scripts to determine the exact binaries to download.
MANIFEST_PATH="$ASSETS_PATH/manifest.json"
cat > "$MANIFEST_PATH" <<EOF
{
    "name": "timbal",
    "description": "Framework for building and orchestrating agentic AI applications—fast, scalable, and enterprise-ready. With flow-based execution, built-in memory and state management, it enables resilient, tool-using agents that think, plan, and act in dynamic environments.",
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

# Upsert the release on github. We'll be able to update the release notes afterwards.
if gh release view "v$VERSION" > /dev/null 2>&1; then
    echo "Cleaning up existing release v$VERSION..."
    gh release delete "v$VERSION" --yes
fi 

echo "Creating release v$VERSION..."
gh release create "v$VERSION" \
    "${ASSETS[@]}" \
    --title "v$VERSION" \
    --notes "Release v$VERSION"
