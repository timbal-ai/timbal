#!/usr/bin/env bash
#
# Integration tests for the `timbal create` exit-code / output contract.
#
# These assert the behaviour that downstream callers (e.g. the monolith) depend
# on and that cannot be covered by the in-process zig unit tests because the
# code paths call `std.process.exit` directly:
#
#   exit 0  success
#   exit 2  usage / precondition error (bad flags, missing args, non-empty
#           target, interactive without a TTY, invalid/reserved/duplicate name)
#   exit 1  runtime failure (fs / network / git)
#
#   - all errors go to stderr; stdout stays empty on failure
#   - `-q` success prints exactly one line on stdout: the absolute project path
#
# Network-dependent success cases (which fetch blueprints from GitHub) are only
# run when TIMBAL_CLI_E2E_NETWORK=1, so the default run stays hermetic.
#
# Usage:
#   cli/integration_tests.sh            # build + run offline cases
#   TIMBAL_CLI_E2E_NETWORK=1 cli/integration_tests.sh   # also run success cases

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSION="${TIMBAL_CLI_VERSION:-dev}"

PASS=0
FAIL=0

red() { printf '\033[31m%s\033[0m\n' "$1"; }
green() { printf '\033[32m%s\033[0m\n' "$1"; }

# Locate the host binary produced by `zig build` (zig-out/<version>/<triple>).
detect_bin() {
    local os arch zos zarch bin
    os="$(uname -s)"
    arch="$(uname -m)"
    case "$os" in
        Darwin) zos="macos" ;;
        Linux) zos="linux" ;;
        *) red "Unsupported OS for integration tests: $os"; exit 1 ;;
    esac
    case "$arch" in
        arm64 | aarch64) zarch="aarch64" ;;
        x86_64 | amd64) zarch="x86_64" ;;
        *) red "Unsupported arch for integration tests: $arch"; exit 1 ;;
    esac
    bin="$SCRIPT_DIR/zig-out/$VERSION/timbal-$VERSION-$zos-$zarch"
    [ "$zos" = "linux" ] && bin="$bin-gnu"
    printf '%s' "$bin"
}

# assert_usage <desc> <expected_code> <stderr_substring> -- <argv...>
# Asserts: exit code matches, stdout is empty, stderr contains the substring.
assert_cli() {
    local desc="$1" expected="$2" substr="$3"
    shift 3
    [ "$1" = "--" ] && shift

    local errfile out code
    errfile="$(mktemp)"
    out="$("$@" 2>"$errfile" </dev/null)"
    code=$?

    local ok=1
    [ "$code" -eq "$expected" ] || { ok=0; red "  exit: got $code, want $expected"; }
    if [ -n "$out" ]; then ok=0; red "  stdout not empty: '$out'"; fi
    if [ -n "$substr" ] && ! grep -qF "$substr" "$errfile"; then
        ok=0
        red "  stderr missing: '$substr' (got: '$(head -1 "$errfile")')"
    fi
    rm -f "$errfile"

    if [ "$ok" -eq 1 ]; then PASS=$((PASS + 1)); green "ok   $desc (exit $code)"; else
        FAIL=$((FAIL + 1)); red "FAIL $desc"; fi
}

main() {
    local BIN
    BIN="$(detect_bin)"
    if [ ! -x "$BIN" ]; then
        echo "Building CLI (zig build)..."
        (cd "$SCRIPT_DIR" && zig build) || { red "zig build failed"; exit 1; }
    fi
    [ -x "$BIN" ] || { red "binary not found: $BIN"; exit 1; }
    echo "Using binary: $BIN"

    local T
    T="$(mktemp -d)"
    trap 'rm -rf "$T"' EXIT

    echo
    echo "== usage errors (expect exit 2, stderr, empty stdout) =="
    assert_cli "unknown flag"          2 "Error: unknown option"            -- "$BIN" create --bogus "$T/a"
    assert_cli "missing path"          2 "Error: a target path is required" -- "$BIN" create --agent foo
    assert_cli "--agent without name"  2 "Error: --agent requires a name"   -- "$BIN" create "$T/b" --agent
    assert_cli "--workflow no name"    2 "Error: --workflow requires a name" -- "$BIN" create "$T/c" --workflow
    assert_cli "multiple paths"        2 "Error: multiple target paths"     -- "$BIN" create "$T/d" "$T/e"
    assert_cli "--with-ui workflow"    2 "Error: --with-ui requires at least one agent" -- "$BIN" create "$T/f" --workflow w --with-ui
    assert_cli "reserved name"         2 "is reserved"                      -- "$BIN" create "$T/g" --agent ui
    assert_cli "invalid name (slash)"  2 "invalid workforce name"           -- "$BIN" create "$T/h" --agent "a/b"
    assert_cli "duplicate name"        2 "duplicate workforce member name"  -- "$BIN" create "$T/i" --agent x --agent x
    assert_cli "interactive no TTY"    2 "interactive create requires a terminal" -- "$BIN" create "$T/j"

    echo
    echo "== preconditions =="
    mkdir -p "$T/nonempty" && touch "$T/nonempty/keep"
    assert_cli "existing non-empty dir" 2 "already exists and is not empty" -- "$BIN" create "$T/nonempty" --agent foo

    echo
    echo "== help (expect exit 0) =="
    assert_cli "create -h"             0 "Create a new timbal project"      -- "$BIN" create -h

    if [ "${TIMBAL_CLI_E2E_NETWORK:-0}" = "1" ]; then
        echo
        echo "== success (network: fetches blueprints) =="
        assert_success_quiet "$BIN" "$T/proj"
        assert_success_with_ui "$BIN" "$T/projui"
    else
        echo
        echo "(skipping success cases; set TIMBAL_CLI_E2E_NETWORK=1 to run them)"
    fi

    echo
    echo "----------------------------------------"
    if [ "$FAIL" -eq 0 ]; then
        green "All $PASS checks passed."
        exit 0
    fi
    red "$FAIL failed, $PASS passed."
    exit 1
}

# Success in -q mode: exit 0, stdout is exactly one line == abs path, and the
# post-create invariants hold (.git/, workforce/<name>/timbal.yaml, api/, no ui/).
assert_success_quiet() {
    local BIN="$1" dir="$2" out code lines
    out="$("$BIN" create "$dir" --agent assistant -q 2>/dev/null </dev/null)"
    code=$?
    lines="$(printf '%s\n' "$out" | grep -c .)"
    local ok=1
    [ "$code" -eq 0 ] || { ok=0; red "  exit: got $code, want 0"; }
    [ "$lines" -eq 1 ] || { ok=0; red "  stdout should be 1 line, got $lines: '$out'"; }
    [ -d "$dir/.git" ] || { ok=0; red "  missing .git/"; }
    [ -f "$dir/workforce/assistant/timbal.yaml" ] || { ok=0; red "  missing workforce/assistant/timbal.yaml"; }
    [ -d "$dir/api" ] || { ok=0; red "  missing api/"; }
    [ ! -d "$dir/ui" ] || { ok=0; red "  unexpected ui/ without --with-ui"; }
    if [ "$ok" -eq 1 ]; then PASS=$((PASS + 1)); green "ok   success -q + invariants"; else
        FAIL=$((FAIL + 1)); red "FAIL success -q + invariants"; fi
}

assert_success_with_ui() {
    local BIN="$1" dir="$2" code
    "$BIN" create "$dir" --agent assistant --with-ui -q >/dev/null 2>&1 </dev/null
    code=$?
    local ok=1
    [ "$code" -eq 0 ] || { ok=0; red "  exit: got $code, want 0"; }
    [ -d "$dir/ui" ] || { ok=0; red "  missing ui/ with --with-ui"; }
    [ -d "$dir/.git" ] || { ok=0; red "  missing .git/"; }
    if [ "$ok" -eq 1 ]; then PASS=$((PASS + 1)); green "ok   success --with-ui"; else
        FAIL=$((FAIL + 1)); red "FAIL success --with-ui"; fi
}

main "$@"
