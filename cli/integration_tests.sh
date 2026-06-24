#!/usr/bin/env bash
#
# Integration tests for the `timbal create` and `timbal add` exit-code / output
# contract.
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

# assert_add <desc> <expected_code> <stderr_substring> <project_dir> -- <add-args...>
# Runs `timbal add <args...>` with cwd = project_dir and asserts exit code +
# stderr substring. (stdout is not checked: `add` prints its success banner to
# stdout, and these cases are error paths.)
assert_add() {
    local desc="$1" expected="$2" substr="$3" dir="$4"
    shift 4
    [ "${1:-}" = "--" ] && shift

    local errfile code
    errfile="$(mktemp)"
    ( cd "$dir" && "$BIN" add "$@" ) >/dev/null 2>"$errfile" </dev/null
    code=$?

    local ok=1
    [ "$code" -eq "$expected" ] || { ok=0; red "  exit: got $code, want $expected"; }
    if [ -n "$substr" ] && ! grep -qF "$substr" "$errfile"; then
        ok=0
        red "  stderr missing: '$substr' (got: '$(head -1 "$errfile")')"
    fi
    rm -f "$errfile"

    if [ "$ok" -eq 1 ]; then PASS=$((PASS + 1)); green "ok   $desc (exit $code)"; else
        FAIL=$((FAIL + 1)); red "FAIL $desc"; fi
}

# Offline success: `add <comp> <name>` exits 0 and scaffolds the member files.
assert_add_success() {
    local dir="$1" comp="$2" name="$3" code
    ( cd "$dir" && "$BIN" add "$comp" "$name" ) >/dev/null 2>&1 </dev/null
    code=$?
    local app="agent.py"; [ "$comp" = "workflow" ] && app="workflow.py"
    local ok=1
    [ "$code" -eq 0 ] || { ok=0; red "  exit: got $code, want 0"; }
    [ -f "$dir/workforce/$name/$app" ] || { ok=0; red "  missing workforce/$name/$app"; }
    [ -f "$dir/workforce/$name/timbal.yaml" ] || { ok=0; red "  missing workforce/$name/timbal.yaml"; }
    if [ "$ok" -eq 1 ]; then PASS=$((PASS + 1)); green "ok   add $comp $name success"; else
        FAIL=$((FAIL + 1)); red "FAIL add $comp $name success"; fi
}

# --force replaces an existing member in place (exit 0) and must not leave the
# `workforce/.<name>.bak` backup behind once the replacement succeeds.
assert_add_force_replace() {
    local dir="$1" comp="$2" name="$3" code
    ( cd "$dir" && "$BIN" add "$comp" "$name" --force ) >/dev/null 2>&1 </dev/null
    code=$?
    local ok=1
    [ "$code" -eq 0 ] || { ok=0; red "  exit: got $code, want 0"; }
    [ -f "$dir/workforce/$name/timbal.yaml" ] || { ok=0; red "  missing timbal.yaml after replace"; }
    [ ! -e "$dir/workforce/.$name.bak" ] || { ok=0; red "  leftover backup workforce/.$name.bak"; }
    if [ "$ok" -eq 1 ]; then PASS=$((PASS + 1)); green "ok   add --force replace (backup cleaned)"; else
        FAIL=$((FAIL + 1)); red "FAIL add --force replace"; fi
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
    echo "== add: usage / precondition errors =="
    # `add agent/workflow` scaffolds from local templates (no network), so these
    # all run hermetically. A bare `workforce/` dir is enough of a project.
    mkdir -p "$T/proj/workforce"
    assert_add "add missing component"   2 "missing component"                 "$T/proj" --
    assert_add "add agent missing name"  2 "missing required argument: name"   "$T/proj" -- agent
    assert_add "add invalid name"        2 "invalid workforce name"            "$T/proj" -- agent "a/b"
    assert_add "add reserved name"       2 "is reserved"                       "$T/proj" -- agent ui
    assert_add "add unknown component"   2 "unknown component"                 "$T/proj" -- frobnicate x
    assert_add "add unknown option"      2 "unknown option"                    "$T/proj" -- agent foo --nope
    mkdir -p "$T/noproj"
    assert_add "add without workforce/"  1 "No 'workforce' directory found"    "$T/noproj" -- agent foo

    echo
    echo "== add: success + --force replace (offline) =="
    assert_add_success "$T/proj" agent foo
    # Re-adding without --force on a non-TTY must fail loudly (exit 1), not silently.
    assert_add "add duplicate, no --force" 1 "Re-run with --force"            "$T/proj" -- agent foo
    # --force replaces in place, exits 0, and leaves no .bak backup behind.
    assert_add_force_replace "$T/proj" agent foo

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
