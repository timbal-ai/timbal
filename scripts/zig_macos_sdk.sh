# Sourced by the scripts that build the CLI with zig (release.sh, upload-cli-release-assets.sh).
#
# zig 0.14 cannot link against the macOS 26 SDK: its libSystem.tbd lists Apple silicon only as
# `arm64e-macos`, zig builds plain `arm64`, finds no matching entry and every libc symbol comes out
# undefined (`undefined symbol: _exit`, while linking zig's own build script). Installing the full Xcode
# makes it the selected developer dir, and its SDK the one zig gets. On such a Mac this picks the Command
# Line Tools' SDK for the build (DEVELOPER_DIR), if theirs still lists `arm64-macos`. The binaries are the
# same: they only need the libc symbols, which both SDKs have.
#
# Nothing happens off macOS, when DEVELOPER_DIR is already set, or when the selected SDK works. The lasting
# fix is zig 0.15, which reads the new SDKs.

# Whether the SDK at $1 lists plain arm64 for libSystem itself: the first document's `targets:` list (it can
# span lines). Later documents (re-exported libraries) may still list arm64-macos in an SDK whose libSystem
# does not, so the whole file is not a test.
_sdk_links_arm64() {
    [ -f "$1/usr/lib/libSystem.B.tbd" ] || return 1
    awk '/^targets:/ { on = 1 } on { print } on && /\]/ { exit }' "$1/usr/lib/libSystem.B.tbd" \
        | tr -s '[], \t' '\n' | grep -qx "arm64-macos"
}

zig_macos_sdk() {
    [ "$(uname -s)" = "Darwin" ] || return 0
    [ -z "${DEVELOPER_DIR:-}" ] || return 0

    local selected clt
    selected="$(xcrun --show-sdk-path 2>/dev/null || true)"
    if [ -n "$selected" ] && _sdk_links_arm64 "$selected"; then
        return 0
    fi

    clt="/Library/Developer/CommandLineTools"
    if _sdk_links_arm64 "$clt/SDKs/MacOSX.sdk"; then
        export DEVELOPER_DIR="$clt"
        echo "zig: the selected macOS SDK (${selected:-none}) lists no arm64-macos; building against $clt/SDKs/MacOSX.sdk."
        return 0
    fi

    echo "Error: zig 0.14 cannot link against the selected macOS SDK (${selected:-none}): its libSystem.tbd" >&2
    echo "lists no arm64-macos, and no Command Line Tools SDK that does was found. Install the Command Line" >&2
    echo "Tools with a macOS 15 SDK, or point DEVELOPER_DIR at a developer dir whose SDK lists arm64-macos." >&2
    return 1
}
