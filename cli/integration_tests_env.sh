#!/usr/bin/env bash
#
# Integration tests for `timbal env pull` / `timbal env push`.
#
# These exercise the built binary end to end — exit-code contract, where files land,
# what is written commented out vs active, and what push refuses — against a temp git
# checkout. Three tiers:
#
#   default                     hermetic: usage / precondition errors. Fake $HOME, no network.
#   TIMBAL_CLI_E2E_NETWORK=1    read-only against the platform: pulls a real project from
#                               api.dev.timbal.ai (default) and checks file placement against
#                               what the API itself returns. Never POSTs (push runs --dry-run).
#   TIMBAL_CLI_E2E_WRITE=1      also exercises real push: creates uniquely named probe vars on
#                               the project via the API, pushes, verifies types/values, deletes
#                               them again (implies NETWORK). Only run against a dev project.
#
# Network tiers read the API key the same way the CLI does (~/.timbal/credentials, profile
# from TIMBAL_E2E_PROFILE / TIMBAL_PROFILE / default) unless TIMBAL_E2E_API_KEY is set.
# They need `curl` and `jq`.
#
#   TIMBAL_E2E_HOST      api.dev.timbal.ai
#   TIMBAL_E2E_ORG       1
#   TIMBAL_E2E_PROJECT   1460
#   TIMBAL_E2E_REV       main
#
# Usage:
#   cli/integration_tests_env.sh
#   TIMBAL_CLI_E2E_NETWORK=1 cli/integration_tests_env.sh
#   TIMBAL_CLI_E2E_WRITE=1 TIMBAL_E2E_PROJECT=1460 cli/integration_tests_env.sh

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSION="${TIMBAL_CLI_VERSION:-dev}"

PASS=0
FAIL=0

red() { printf '\033[31m%s\033[0m\n' "$1"; }
green() { printf '\033[32m%s\033[0m\n' "$1"; }
dim() { printf '\033[2m%s\033[0m\n' "$1"; }

pass() { PASS=$((PASS + 1)); green "ok   $1"; }
fail() { FAIL=$((FAIL + 1)); red "FAIL $1"; }

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

# Run the CLI with cwd=$1 and capture stdout/stderr/exit into globals.
# Env for the child is whatever the caller exported (HOME, TIMBAL_PROFILE).
OUT=""
ERR=""
CODE=0
run_cli() {
    local dir="$1"
    shift
    local outfile errfile
    outfile="$(mktemp)"
    errfile="$(mktemp)"
    ( cd "$dir" && "$BIN" "$@" ) >"$outfile" 2>"$errfile" </dev/null
    CODE=$?
    OUT="$(cat "$outfile")"
    ERR="$(cat "$errfile")"
    rm -f "$outfile" "$errfile"
}

# expect <desc> <exit> [stderr_substr] [stdout_substr]
expect() {
    local desc="$1" want="$2" err_sub="${3:-}" out_sub="${4:-}" ok=1
    [ "$CODE" -eq "$want" ] || { ok=0; red "  exit: got $CODE, want $want"; }
    if [ -n "$err_sub" ] && ! printf '%s' "$ERR" | grep -qF -- "$err_sub"; then
        ok=0; red "  stderr missing: '$err_sub' (got: '$(printf '%s' "$ERR" | head -3)')"
    fi
    if [ -n "$out_sub" ] && ! printf '%s' "$OUT" | grep -qF -- "$out_sub"; then
        ok=0; red "  stdout missing: '$out_sub' (got: '$(printf '%s' "$OUT" | head -5)')"
    fi
    if [ "$ok" -eq 1 ]; then pass "$desc (exit $CODE)"; else fail "$desc"; fi
}

# check <desc> <shell-condition...>
check() {
    local desc="$1"
    shift
    if "$@"; then pass "$desc"; else fail "$desc"; fi
}

out_has() { printf '%s' "$OUT" | grep -qF -- "$1"; }
err_has() { printf '%s' "$ERR" | grep -qF -- "$1"; }
err_lacks() { ! printf '%s' "$ERR" | grep -qF -- "$1"; }
# The stdout line containing <needle> also contains <substr>.
out_line_has() { printf '%s\n' "$OUT" | grep -F -- "$1" | grep -qF -- "$2"; }
not_grep() { ! grep -Eq -- "$1" "$2"; }
count_is() { [ "$(grep -cE -- "$1" "$2")" = "$3" ]; }
line_count_is() { [ "$(wc -l <"$1" | tr -d ' ')" = "$2" ]; }

# Active (non-comment) keys in a .env, one per line.
active_keys() { grep -E '^[A-Za-z_][A-Za-z0-9_]*=' "$1" 2>/dev/null | cut -d= -f1 | sort -u; }
no_active_platform_keys() { ! active_keys "$1" | grep -Eq '^(TIMBAL_|VITE_TIMBAL_)'; }
# Every active key in $1 is one of the newline-separated names in $2.
active_subset_of() {
    local extra
    extra="$(comm -23 <(active_keys "$1") <(printf '%s\n' "$2" | sort -u))"
    [ -z "$extra" ] || { red "  unexpected active keys: $(printf '%s' "$extra" | tr '\n' ' ')"; return 1; }
}
member_env_is_only_app_id() { [ "$(active_keys "$1")" = TIMBAL_APP_ID ] && grep -qx "TIMBAL_APP_ID=$2" "$1"; }
# `KEY=VALUE` line in $2 has `# type: $3` within the 3 lines above it.
tagged_as() { grep -B3 -x -- "$1" "$2" | grep -qx "# type: $3"; }

git_init_repo() {
    local dir="$1" remote="$2" branch="$3"
    mkdir -p "$dir"
    git -C "$dir" -c init.defaultBranch="$branch" init -q
    git -C "$dir" -c user.name=e2e -c user.email=e2e@example.com commit -q --allow-empty -m init
    git -C "$dir" checkout -q -B "$branch"
    git -C "$dir" remote add origin "$remote"
    printf '.env\n' >"$dir/.gitignore"
}

write_member() {
    local dir="$1" name="$2" uid="$3"
    mkdir -p "$dir/workforce/$name"
    printf '_id: "%s"\n_type: "agent"\nfqn: "agent.py::agent"\n' "$uid" >"$dir/workforce/$name/timbal.yaml"
}

# ---------------------------------------------------------------------------
# Hermetic tier
# ---------------------------------------------------------------------------
hermetic_tier() {
    echo
    echo "== env: hermetic usage / precondition contract =="

    local FAKE_HOME="$T/home"
    mkdir -p "$FAKE_HOME/.timbal"
    printf '[default]\napi_key = fake-key-for-tests\n' >"$FAKE_HOME/.timbal/credentials"

    # Not configured at all.
    local EMPTY_HOME="$T/emptyhome"
    mkdir -p "$EMPTY_HOME"
    HOME="$EMPTY_HOME" run_cli "$T" env pull --dry-run
    expect "not configured → exit 1" 1 "Timbal is not configured"

    export HOME="$FAKE_HOME"
    unset TIMBAL_PROFILE

    run_cli "$T" env
    expect "missing command" 2 "missing command (pull or push)"
    run_cli "$T" env frobnicate
    expect "unknown command" 2 "unknown env command"
    run_cli "$T" env pull --bogus
    expect "unknown option" 2 "unknown option"
    run_cli "$T" env pull --rev
    expect "--rev without value" 2 "--rev requires a branch name"
    run_cli "$T" env pull --rev main --default
    expect "--rev + --default" 2 "mutually exclusive"
    run_cli "$T" env push --force
    expect "--force with push" 2 "--force is only valid with"
    run_cli "$T" env pull --secret X
    expect "--secret with pull" 2 "--secret / --plain are only valid with"
    run_cli "$T" env push --include-platform-vars
    expect "--include-platform-vars with push" 2 "only valid with \`timbal env pull\`"
    run_cli "$T" env push --secret X --plain X
    expect "same var --secret and --plain" 2 "cannot be both"
    run_cli "$T" env pull --base-url http://api.dev.timbal.ai
    expect "--base-url http" 2 "must be https"
    run_cli "$T" env pull --base-url https://evil.example.com
    expect "--base-url lookalike host" 2 "must be https://api.timbal.ai"
    run_cli "$T" env -h
    expect "env -h" 0

    # Precondition: not a git repo.
    mkdir -p "$T/nogit"
    run_cli "$T/nogit" env pull --dry-run
    expect "outside a git repo" 1 "not inside a git repository"

    # Precondition: git repo without a Timbal remote.
    git_init_repo "$T/github" "git@github.com:acme/app.git" main
    run_cli "$T/github" env pull --dry-run
    expect "no Timbal remote" 1 "no Timbal git remote found"

    # Precondition: detached HEAD → needs --rev / --default (parsed before any network).
    git_init_repo "$T/detached" "https://api.dev.timbal.ai/orgs/1/projects/1/git" main
    git -C "$T/detached" checkout -q --detach
    run_cli "$T/detached" env pull --dry-run
    expect "detached HEAD" 1 "could not determine current git branch"

    # Push preconditions on the local file happen before any network call.
    git_init_repo "$T/proj" "https://api.dev.timbal.ai/orgs/1/projects/1/git" main
    mkdir -p "$T/proj/workforce"
    run_cli "$T/proj" env push --dry-run
    expect "push without local file" 1 "local env file not found"
    : >"$T/proj/.env"
    run_cli "$T/proj" env push --dry-run
    expect "push empty file" 1 "no variables found"
    printf '# type: hidden\nFOO=1\n' >"$T/proj/.env"
    run_cli "$T/proj" env push --dry-run
    expect "push invalid type metadata" 1 "invalid type 'hidden'"
    rm -f "$T/proj/.env"
}

# ---------------------------------------------------------------------------
# Network tier (read-only)
# ---------------------------------------------------------------------------
API_HOST="${TIMBAL_E2E_HOST:-api.dev.timbal.ai}"
API_ORG="${TIMBAL_E2E_ORG:-1}"
API_PROJECT="${TIMBAL_E2E_PROJECT:-1460}"
API_REV="${TIMBAL_E2E_REV:-main}"
API_KEY=""
API=""

resolve_api_key() {
    if [ -n "${TIMBAL_E2E_API_KEY:-}" ]; then
        API_KEY="$TIMBAL_E2E_API_KEY"
        return 0
    fi
    local profile="${TIMBAL_E2E_PROFILE:-${TIMBAL_PROFILE:-default}}" header
    if [ "$profile" = "default" ]; then header='[default]'; else header="[profile $profile]"; fi
    API_KEY="$(awk -v h="$header" '
        $0 == h { f = 1; next }
        /^\[/ { f = 0 }
        f && /^[ \t]*api_key[ \t]*=/ { sub(/^[^=]*=[ \t]*/, ""); print; exit }
    ' "$HOME/.timbal/credentials" 2>/dev/null)"
    [ -n "$API_KEY" ]
}

api_get() { curl -sS -H "Authorization: Bearer $API_KEY" -H "Accept: application/json" "$API$1"; }
api_post() { curl -sS -o /dev/null -w '%{http_code}' -X POST -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" -d "$2" "$API$1"; }
api_delete() { curl -sS -o /dev/null -w '%{http_code}' -X DELETE -H "Authorization: Bearer $API_KEY" "$API$1"; }

network_tier() {
    echo
    echo "== env: network (read-only) against https://$API_HOST org=$API_ORG project=$API_PROJECT rev=$API_REV =="

    command -v jq >/dev/null 2>&1 || { red "jq is required for the network tier"; FAIL=$((FAIL + 1)); return; }
    resolve_api_key || { red "no API key (set TIMBAL_E2E_API_KEY or run timbal configure)"; FAIL=$((FAIL + 1)); return; }
    API="https://$API_HOST/orgs/$API_ORG/projects/$API_PROJECT"
    export TIMBAL_PROFILE="${TIMBAL_E2E_PROFILE:-${TIMBAL_PROFILE:-default}}"

    # Expectations come from the API itself, so the test follows the project.
    local wf pull list
    wf="$(api_get "/workforce?rev=$API_REV")" || { fail "GET /workforce"; return; }
    pull="$(api_get "/vars/pull?rev=$API_REV")" || { fail "GET /vars/pull"; return; }
    list="$(api_get "/vars")" || { fail "GET /vars"; return; }
    printf '%s' "$pull" | jq -e '.vars' >/dev/null 2>&1 || { fail "GET /vars/pull returned no vars array: $(printf '%s' "$pull" | head -c 300)"; return; }

    MEMBER_NAME="$(printf '%s' "$wf" | jq -r '[.workforce[] | select(.uid != null and .uid != "" and (.id|tostring|startswith("-")|not))][0].name // empty')"
    MEMBER_UID="$(printf '%s' "$wf" | jq -r '[.workforce[] | select(.uid != null and .uid != "" and (.id|tostring|startswith("-")|not))][0].uid // empty')"
    MEMBER_ID="$(printf '%s' "$wf" | jq -r '[.workforce[] | select(.uid != null and .uid != "" and (.id|tostring|startswith("-")|not))][0].id // empty' | tr -d '"')"
    [ -n "$MEMBER_NAME" ] || dim "  (project has no registered workforce component with a uid; TIMBAL_APP_ID checks will be skipped)"

    # Names in the effective env that are not user-defined project vars → must land commented out.
    local managed_names user_names
    managed_names="$(jq -rn --argjson p "$pull" --argjson l "$list" '[$p.vars[].name] - [$l.vars[].name] | .[]')"
    user_names="$(jq -rn --argjson p "$pull" --argjson l "$list" '[$p.vars[].name] - ([$p.vars[].name] - [$l.vars[].name]) | .[]')"

    P="$T/net"
    git_init_repo "$P" "https://$API_HOST/orgs/$API_ORG/projects/$API_PROJECT/git" "$API_REV"
    mkdir -p "$P/workforce"
    [ -n "$MEMBER_NAME" ] && write_member "$P" "$MEMBER_NAME" "$MEMBER_UID"
    write_member "$P" e2e-orphan "deadbeefdeadbeefdeadbeefdeadbeef"

    # --- pull --dry-run writes nothing ---------------------------------------
    run_cli "$P" env pull --dry-run
    expect "pull --dry-run" 0 "" "Dry run — would write"
    check "pull --dry-run creates no .env" test ! -e "$P/.env"
    check "pull --dry-run reports the orphan member" out_has "workforce/e2e-orphan  no platform component"
    check "pull --dry-run writes no member .env" test ! -e "$P/workforce/e2e-orphan/.env"
    if [ -n "$MEMBER_NAME" ]; then
        check "pull --dry-run plans TIMBAL_APP_ID=$MEMBER_ID for $MEMBER_NAME" out_has "workforce/$MEMBER_NAME/.env  TIMBAL_APP_ID=$MEMBER_ID  would add"
        check "pull --dry-run did not write member .env" test ! -e "$P/workforce/$MEMBER_NAME/.env"
    fi

    # --- pull: placement ------------------------------------------------------
    run_cli "$P" env pull
    expect "pull" 0 "" "Pulled"
    check "pull wrote <project>/.env" test -f "$P/.env"
    check "pull: no active TIMBAL_* / VITE_TIMBAL_* in root .env" no_active_platform_keys "$P/.env"
    local n ok=1
    while IFS= read -r n; do
        [ -n "$n" ] || continue
        grep -qF "# $n=" "$P/.env" || { ok=0; red "  managed $n not present as a commented line"; }
        grep -Eq "^$n=" "$P/.env" && { ok=0; red "  managed $n is active"; }
    done <<<"$managed_names"
    [ "$ok" -eq 1 ] && pass "pull: every platform-managed var is commented out" || fail "pull: platform-managed vars"
    ok=1
    while IFS= read -r n; do
        [ -n "$n" ] || continue
        # Reserved or multi-line values are legitimately commented; anything else must be active.
        case "$n" in PORT | TIMBAL_PROJECT_SECRET | TIMBAL_APP_ID) continue ;; esac
        if ! grep -Eq "^$n=" "$P/.env" && ! grep -qF "# $n=" "$P/.env"; then ok=0; red "  user var $n missing"; fi
    done <<<"$user_names"
    [ "$ok" -eq 1 ] && pass "pull: every user-defined var is present" || fail "pull: user-defined vars"
    check "pull: active keys ⊆ user-defined names" active_subset_of "$P/.env" "$user_names"
    check "pull: file carries the rev header" grep -qF "# rev: $API_REV" "$P/.env"
    check "pull: orphan member got no .env" test ! -e "$P/workforce/e2e-orphan/.env"
    if [ -n "$MEMBER_NAME" ]; then
        check "pull: member .env has exactly TIMBAL_APP_ID=$MEMBER_ID" member_env_is_only_app_id "$P/workforce/$MEMBER_NAME/.env" "$MEMBER_ID"
        check "pull: TIMBAL_APP_ID not active in root .env" not_grep '^TIMBAL_APP_ID=' "$P/.env"
        cp "$P/workforce/$MEMBER_NAME/.env" "$T/member-before.env"
    fi

    # --- pull refuses to clobber; --force keeps member files untouched --------
    run_cli "$P" env pull
    expect "pull again without --force" 1 "already exists. Re-run with --force"
    run_cli "$P" env pull --force
    expect "pull --force" 0 "" "Pulled"
    if [ -n "$MEMBER_NAME" ]; then
        check "pull --force: member .env byte-identical" cmp -s "$T/member-before.env" "$P/workforce/$MEMBER_NAME/.env"
        check "pull --force reports member unchanged" out_has "TIMBAL_APP_ID=$MEMBER_ID  unchanged"

        # Merge: other lines survive, stale value is replaced in place, exactly once.
        printf '# my local overrides\nDEBUG=1\nTIMBAL_APP_ID=1\n\nOTHER="a b"\n' >"$P/workforce/$MEMBER_NAME/.env"
        run_cli "$P" env pull --force
        expect "pull --force merges member .env" 0 "" "updated (was 1)"
        check "merge keeps comment" grep -qxF '# my local overrides' "$P/workforce/$MEMBER_NAME/.env"
        check "merge keeps DEBUG=1" grep -qxF 'DEBUG=1' "$P/workforce/$MEMBER_NAME/.env"
        check "merge keeps quoted OTHER" grep -qxF 'OTHER="a b"' "$P/workforce/$MEMBER_NAME/.env"
        check "merge replaced TIMBAL_APP_ID exactly once" count_is '^TIMBAL_APP_ID=' "$P/workforce/$MEMBER_NAME/.env" 1
        check "merge wrote the platform id" grep -qx "TIMBAL_APP_ID=$MEMBER_ID" "$P/workforce/$MEMBER_NAME/.env"
        check "merge keeps line count" line_count_is "$P/workforce/$MEMBER_NAME/.env" 5
    fi

    # --- --include-platform-vars writes them active and the audit says so -----
    if printf '%s\n' "$managed_names" | grep -qx TIMBAL_PROJECT_ENV_ID; then
        run_cli "$P" env pull --force --include-platform-vars
        expect "pull --include-platform-vars" 0 "Placement check" "written active"
        check "include-platform-vars: TIMBAL_PROJECT_ENV_ID active" grep -Eq '^TIMBAL_PROJECT_ENV_ID=' "$P/.env"
        check "audit flags the reroute" err_has "TIMBAL_PROJECT_ENV_ID is active"
        run_cli "$P" env pull --force
        expect "pull --force restores commented layout" 0 "" "Pulled"
        check "no audit finding after normal pull" err_lacks "Placement check"
    else
        dim "  (TIMBAL_PROJECT_ENV_ID not in this project's pull set; skipping --include-platform-vars audit case)"
    fi

    # --- push --dry-run: plan, never POSTs -------------------------------------
    local probe="E2E_PROBE_$(date +%s)_$$"
    printf '%s_KEY=sk-not-a-real-key\nLOG_LEVEL=debug\nMAX_TOKENS=1024\n# secret\nCUSTOM_THING=abc\nTIMBAL_PROJECT_ENV_ID=999\nVITE_TIMBAL_ORG_ID=1\nTIMBAL_APP_ID=5\nPORT=3000\n' "$probe" >"$P/.env"
    run_cli "$P" env push --dry-run --plain LOG_LEVEL
    expect "push --dry-run" 0 "" "No changes made."
    check "plan: inferred secret for *_KEY" out_line_has "${probe}_KEY" "secret  inferred from name/value"
    check "plan: --plain wins" out_line_has "LOG_LEVEL" "plain   --secret/--plain"
    check "plan: MAX_TOKENS is plain" out_line_has "MAX_TOKENS" "plain   inferred"
    check "plan: # secret shorthand honoured" out_line_has "CUSTOM_THING" "secret  file metadata"
    check "plan: TIMBAL_PROJECT_ENV_ID never pushed" out_line_has "TIMBAL_PROJECT_ENV_ID" "never pushed"
    check "plan: VITE_TIMBAL_* never pushed" out_line_has "VITE_TIMBAL_ORG_ID" "never pushed"
    check "plan: TIMBAL_APP_ID reserved with placement hint" out_line_has "TIMBAL_APP_ID" "belongs in workforce/<name>/.env"
    check "plan: PORT reserved" out_line_has "  PORT " "reserved"
    check "audit: active TIMBAL_PROJECT_ENV_ID in root .env flagged on push" err_has "TIMBAL_PROJECT_ENV_ID is active"
    check "audit: root TIMBAL_APP_ID flagged on push" err_has "TIMBAL_APP_ID here is loaded into every service"
    check "push --dry-run did not create the probe var" var_absent "${probe}_KEY"
    if [ -n "$MEMBER_NAME" ]; then
        check "push --dry-run plans member app id" out_has "TIMBAL_APP_ID=$MEMBER_ID"
    fi

    printf 'TIMBAL_PROJECT_ENV_ID=1\nPORT=1\n' >"$P/.env"
    run_cli "$P" env push --dry-run
    expect "push with only reserved/managed vars" 1 "nothing to push"

    run_cli "$P" env pull --dry-run --rev "e2e-no-such-branch-$$"
    expect "pull unknown branch" 1 "Error"

    run_cli "$P" env pull --dry-run -f "workforce/e2e-orphan/.env.e2e"
    check "-f inside a member dir warns about scope" err_has "scoped to workforce member 'e2e-orphan'"

    rm -f "$P/.env"
}

# ---------------------------------------------------------------------------
# Write tier (real push against a dev project, probe vars only, cleaned up)
# ---------------------------------------------------------------------------
PROBE_IDS=()
cleanup_probes() {
    local id
    for id in "${PROBE_IDS[@]:-}"; do
        [ -n "$id" ] && api_delete "/vars/$id" >/dev/null 2>&1
    done
}

var_id_by_name() { api_get /vars | jq -r --arg n "$1" '.vars[] | select(.name == $n) | .id' | head -1; }
var_type_by_name() { api_get /vars | jq -r --arg n "$1" '.vars[] | select(.name == $n) | .value.type' | head -1; }
var_decrypted_by_id() { api_get "/vars/$1" | jq -r '.value.decrypted // .value.value // empty'; }
var_absent() { [ -z "$(var_id_by_name "$1")" ]; }
value_is() { [ "$(var_decrypted_by_id "$1")" = "$2" ]; }
type_is() { [ "$(var_type_by_name "$1")" = "$2" ]; }

write_tier() {
    echo
    echo "== env: WRITE tier — real push against https://$API_HOST project=$API_PROJECT (probe vars only) =="
    [ -n "$API" ] || { red "network tier did not initialise; skipping write tier"; FAIL=$((FAIL + 1)); return; }

    local stamp="$(date +%s)_$$" env_id env_ids='null'
    local S="E2E_PROBE_${stamp}_SECRET" PL="E2E_PROBE_${stamp}_PLAIN"
    env_id="$(api_get /envs | jq -r --arg b "$API_REV" '.envs[] | select(.branch == $b) | .id' | head -1)"
    [ -n "$env_id" ] && env_ids="[$env_id]"

    local code
    code="$(api_post /vars "{\"name\":\"$S\",\"type\":\"secret\",\"value\":\"s1\",\"env_ids\":$env_ids}")"
    case "$code" in 200 | 201) ;; *) fail "create probe secret (HTTP $code)"; return ;; esac
    local sid pid
    sid="$(var_id_by_name "$S")"
    PROBE_IDS=("$sid")
    code="$(api_post /vars "{\"name\":\"$PL\",\"type\":\"plain\",\"value\":\"p1\",\"env_ids\":$env_ids}")"
    case "$code" in 200 | 201) ;; *) fail "create probe plain (HTTP $code)"; cleanup_probes; return ;; esac
    pid="$(var_id_by_name "$PL")"
    PROBE_IDS=("$sid" "$pid")
    [ -n "$sid" ] && [ -n "$pid" ] || { fail "probe vars not visible in GET /vars after create"; cleanup_probes; return; }
    trap 'cleanup_probes; rm -rf "$T"' EXIT
    pass "created probe vars $S (secret) and $PL (plain)"

    P="$T/write"
    git_init_repo "$P" "https://$API_HOST/orgs/$API_ORG/projects/$API_PROJECT/git" "$API_REV"
    mkdir -p "$P/workforce"

    # Secrets come back decrypted with their type.
    run_cli "$P" env pull
    expect "write: pull with probe vars" 0 "" "Pulled"
    check "pull: secret is active and decrypted" grep -qx "$S=s1" "$P/.env"
    check "pull: secret tagged # type: secret" tagged_as "$S=s1" "$P/.env" secret
    check "pull: plain is active" grep -qx "$PL=p1" "$P/.env"
    check "pull: plain tagged # type: plain" tagged_as "$PL=p1" "$P/.env" plain

    # A file that marks the platform secret plain is refused; nothing changes.
    printf '# type: plain\n%s=s-wrong\n\n%s=p1\n' "$S" "$PL" >"$P/.env"
    run_cli "$P" env push
    expect "push: secret marked plain is BLOCKED" 1 "secrets on the platform but marked plain" "BLOCKED"
    check "blocked push left the secret value untouched" value_is "$sid" s1

    # Value updates go through; the platform keeps the stored type (backend-confirmed).
    printf '%s=s2\n# type: secret\n%s=p2\nTIMBAL_PROJECT_ENV_ID=999\nPORT=1\n' "$S" "$PL" >"$P/.env"
    run_cli "$P" env push
    expect "push: value updates" 0 "" "Pushed"
    check "push: secret value updated" value_is "$sid" s2
    check "push: secret type still secret (no metadata → platform type)" type_is "$S" secret
    check "push: plain value updated" value_is "$pid" p2
    check "push: requested type change reported as not applied" out_has "keeps the stored type on update"
    check "push: plain stays plain on the platform" type_is "$PL" plain
    check "push: TIMBAL_PROJECT_ENV_ID not created on the platform" var_absent TIMBAL_PROJECT_ENV_ID
    check "push: audit flagged active TIMBAL_PROJECT_ENV_ID" err_has "TIMBAL_PROJECT_ENV_ID is active"

    # --plain acknowledges the mismatch and pushes the value; type still cannot flip.
    printf '# type: plain\n%s=s3\n' "$S" >"$P/.env"
    run_cli "$P" env push --plain "$S"
    expect "push --plain acknowledges mismatch" 0 "" "Pushed"
    check "push --plain: value updated" value_is "$sid" s3
    check "push --plain: type still secret on the platform" type_is "$S" secret

    cleanup_probes
    PROBE_IDS=()
    trap 'rm -rf "$T"' EXIT
    check "cleanup: probe secret deleted" var_absent "$S"
    check "cleanup: probe plain deleted" var_absent "$PL"
}

main() {
    BIN="$(detect_bin)"
    if [ ! -x "$BIN" ]; then
        echo "Building CLI (zig build)..."
        (cd "$SCRIPT_DIR" && zig build) || { red "zig build failed"; exit 1; }
    fi
    [ -x "$BIN" ] || { red "binary not found: $BIN"; exit 1; }
    echo "Using binary: $BIN"

    T="$(mktemp -d)"
    trap 'rm -rf "$T"' EXIT

    local REAL_HOME="$HOME"
    hermetic_tier
    export HOME="$REAL_HOME"

    if [ "${TIMBAL_CLI_E2E_WRITE:-0}" = "1" ]; then
        export TIMBAL_CLI_E2E_NETWORK=1
    fi
    if [ "${TIMBAL_CLI_E2E_NETWORK:-0}" = "1" ]; then
        network_tier
        if [ "${TIMBAL_CLI_E2E_WRITE:-0}" = "1" ]; then
            write_tier
        else
            echo
            dim "(skipping write tier; set TIMBAL_CLI_E2E_WRITE=1 to exercise real push against a dev project)"
        fi
    else
        echo
        dim "(skipping network tier; set TIMBAL_CLI_E2E_NETWORK=1 to pull from https://$API_HOST)"
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

main "$@"
