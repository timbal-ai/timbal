const builtin = @import("builtin");
const std = @import("std");
const fs = std.fs;

const utils = @import("../utils.zig");
const Color = utils.Color;

const is_windows = builtin.os.tag == .windows;
const sep = if (is_windows) "\\" else "/";

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------

fn printUsageWithError(err: []const u8) !void {
    const stderr = std.io.getStdErr().writer();
    try stderr.print("{s}\n\n", .{err});
    try printUsage();
}

fn printUsage() !void {
    const stderr = std.io.getStdErr().writer();
    try stderr.writeAll(
        \\Sync project environment variables with the Timbal platform.
        \\
        \\
    ++
        "\x1b[1;32mUsage: \x1b[1;36mtimbal env \x1b[0;36m<pull|push> \x1b[0m[OPTIONS]\n" ++
        "\n" ++
        "\x1b[1;32mCommands:\n" ++
        "    \x1b[1;36mpull \x1b[0mDownload vars for the env tracking --rev into <project>/.env\n" ++
        "    \x1b[1;36mpush \x1b[0mUpsert <project>/.env vars into the env tracking --rev\n" ++
        "\n" ++
        "\x1b[1;32mOptions:\n" ++
        "    \x1b[1;36m--rev <BRANCH>           \x1b[0mGit branch whose env to sync (default: current branch)\n" ++
        "    \x1b[1;36m--default                \x1b[0mOmit rev; use the project's default-branch env\n" ++
        "    \x1b[1;36m-f\x1b[0m, \x1b[1;36m--file <PATH>        \x1b[0mLocal env file (default: .env at the project root)\n" ++
        "    \x1b[1;36m--force                  \x1b[0mPull: overwrite an existing local env file\n" ++
        "    \x1b[1;36m--dry-run                \x1b[0mShow what would be written/sent (values redacted); change nothing\n" ++
        "    \x1b[1;36m--secret <NAME[,NAME]>   \x1b[0mPush: treat these vars as secrets (repeatable)\n" ++
        "    \x1b[1;36m--plain <NAME[,NAME]>    \x1b[0mPush: treat these vars as plain (also acknowledges a secret-on-platform mismatch)\n" ++
        "    \x1b[1;36m--include-platform-vars  \x1b[0mPull: write the platform-managed TIMBAL_* runtime vars active (default: commented)\n" ++
        "    \x1b[1;36m--no-app-ids             \x1b[0mDo not write TIMBAL_APP_ID into workforce/<name>/.env\n" ++
        "    \x1b[1;36m--base-url <URL>         \x1b[0mOverride API host (https://api[.env].timbal.ai)\n" ++
        "\n" ++
        "\x1b[1;32mSecrets vs plain:\n" ++
        "\x1b[0m    A `# type: secret` (or `# secret`) comment directly above a var marks it secret; `# type: plain`\n" ++
        "    (or `# plain`) marks it plain. Vars without a comment keep their current platform type. New vars\n" ++
        "    are inferred from their name/value (KEY, SECRET, TOKEN, PASSWORD, sk-..., ... → secret).\n" ++
        "    --secret / --plain always win. The platform fixes a var's type at creation — an update only changes\n" ++
        "    the value — so change types in the Timbal UI. Push refuses a file that marks a platform secret plain.\n" ++
        "\n" ++
        "\x1b[1;32mFile placement:\n" ++
        "\x1b[0m    <project>/.env is what `timbal start` loads into every service, so that is the sync target.\n" ++
        "    TIMBAL_* / VITE_TIMBAL_* are the platform's namespace: resolved before every deploy/preview, so\n" ++
        "    pull writes them commented out (loading them locally reroutes service calls to the deployed\n" ++
        "    gateway) and push NEVER sends them, whatever the file or flags say.\n" ++
        "    TIMBAL_APP_ID is per workforce member: pull/push merge it into workforce/<name>/.env by matching\n" ++
        "    the timbal.yaml _id against the platform. It is never written to the project-root .env.\n" ++
        "\n" ++
        "\x1b[1;32mNotes:\n" ++
        "\x1b[0m    Org/project are resolved from the Timbal git remote in .git/config:\n" ++
        "    https://api[.env].timbal.ai/orgs/{org_id}/projects/{project_id}/git\n" ++
        "    Auth uses the configured profile API key (timbal configure).\n" ++
        "    Secrets are written in plaintext to the local file — keep it gitignored.\n" ++
        "    Push is upsert-only: platform vars not in the file are left alone (no delete / replace-all).\n" ++
        "    Reserved names (PORT, TIMBAL_PROJECT_SECRET, TIMBAL_APP_ID) are never pushed.\n" ++
        "\n" ++
        utils.global_options_help ++
        "\n");
}

// ---------------------------------------------------------------------------
// Profile / credentials (same INI shape as configure/start)
// ---------------------------------------------------------------------------

fn getHomePath(allocator: std.mem.Allocator) ![]u8 {
    return if (is_windows)
        std.process.getEnvVarOwned(allocator, "USERPROFILE")
    else
        std.process.getEnvVarOwned(allocator, "HOME");
}

fn getCredentialsPath(allocator: std.mem.Allocator) ![]u8 {
    const home = try getHomePath(allocator);
    defer allocator.free(home);
    return std.fmt.allocPrint(allocator, "{s}{s}.timbal{s}credentials", .{ home, sep, sep });
}

fn isSectionHeader(line: []const u8, profile: []const u8) bool {
    const trimmed = std.mem.trim(u8, line, " \t\r");
    if (std.mem.eql(u8, profile, "default")) {
        return std.mem.eql(u8, trimmed, "[default]");
    }
    if (!std.mem.startsWith(u8, trimmed, "[profile ")) return false;
    if (!std.mem.endsWith(u8, trimmed, "]")) return false;
    const inner = trimmed["[profile ".len .. trimmed.len - 1];
    return std.mem.eql(u8, std.mem.trim(u8, inner, " \t"), profile);
}

fn isAnySectionHeader(line: []const u8) bool {
    const trimmed = std.mem.trim(u8, line, " \t\r");
    return trimmed.len >= 2 and trimmed[0] == '[' and trimmed[trimmed.len - 1] == ']';
}

fn readValue(content: []const u8, profile: []const u8, key: []const u8) ?[]const u8 {
    var in_target = false;
    var lines = std.mem.splitScalar(u8, content, '\n');
    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");
        if (isAnySectionHeader(trimmed)) {
            in_target = isSectionHeader(trimmed, profile);
            continue;
        }
        if (in_target and std.mem.startsWith(u8, trimmed, key)) {
            const rest = trimmed[key.len..];
            const after_key = std.mem.trimLeft(u8, rest, " \t");
            if (after_key.len > 0 and after_key[0] == '=') {
                const value = std.mem.trim(u8, after_key[1..], " \t");
                if (value.len > 0) return value;
            }
        }
    }
    return null;
}

// ---------------------------------------------------------------------------
// Git remote → org / project / API host
// ---------------------------------------------------------------------------

pub const TimbalRemote = struct {
    org_id: []const u8,
    project_id: []const u8,
    base_url: []const u8,
    remote_name: []const u8,

    pub fn deinit(self: *TimbalRemote, allocator: std.mem.Allocator) void {
        allocator.free(self.org_id);
        allocator.free(self.project_id);
        allocator.free(self.base_url);
        allocator.free(self.remote_name);
    }
};

/// Hosts we will send the profile Bearer token to for vars pull/push.
/// Allows `api.timbal.ai` and `api.<label>.timbal.ai` (e.g. dev, staging).
/// Rejects lookalikes like `notimbal.ai` / `evil.timbal.ai` / `api.timbal.ai.evil.com`.
fn isTimbalApiHost(host: []const u8) bool {
    if (std.mem.eql(u8, host, "api.timbal.ai")) return true;

    const prefix = "api.";
    const suffix = ".timbal.ai";
    if (!std.mem.startsWith(u8, host, prefix)) return false;
    if (!std.mem.endsWith(u8, host, suffix)) return false;
    if (host.len <= prefix.len + suffix.len) return false;

    const label = host[prefix.len .. host.len - suffix.len];
    if (label.len == 0) return false;
    // Single DNS label only (`api.staging.timbal.ai`, not `api.foo.bar.timbal.ai`).
    if (std.mem.indexOfScalar(u8, label, '.') != null) return false;
    for (label) |c| {
        const ok = std.ascii.isAlphanumeric(c) or c == '-';
        if (!ok) return false;
    }
    return true;
}

/// Parse a Timbal platform git remote URL.
/// Accepts: https://api[.env].timbal.ai/orgs/{org}/projects/{project}/git[/]
pub fn parseTimbalRemoteUrl(allocator: std.mem.Allocator, url: []const u8, remote_name: []const u8) !?TimbalRemote {
    const trimmed = std.mem.trim(u8, url, " \t\r\n");
    if (trimmed.len == 0) return null;

    // HTTPS only — never send the profile Bearer token over cleartext.
    if (!std.mem.startsWith(u8, trimmed, "https://")) return null;
    var rest = trimmed["https://".len..];

    const slash = std.mem.indexOfScalar(u8, rest, '/') orelse return null;
    const host = rest[0..slash];
    var path = rest[slash..];
    while (path.len > 1 and path[path.len - 1] == '/') path = path[0 .. path.len - 1];

    // Optional .git suffix on the final segment.
    if (std.mem.endsWith(u8, path, ".git")) {
        path = path[0 .. path.len - ".git".len];
    }

    // /orgs/{org_id}/projects/{project_id}/git
    var parts = std.mem.splitScalar(u8, path, '/');
    _ = parts.next(); // leading empty from leading '/'
    const orgs_seg = parts.next() orelse return null;
    const org_id = parts.next() orelse return null;
    const projects_seg = parts.next() orelse return null;
    const project_id = parts.next() orelse return null;
    const git_seg = parts.next() orelse return null;
    if (parts.next() != null) return null;

    if (!std.mem.eql(u8, orgs_seg, "orgs")) return null;
    if (!std.mem.eql(u8, projects_seg, "projects")) return null;
    if (!std.mem.eql(u8, git_seg, "git")) return null;
    if (org_id.len == 0 or project_id.len == 0 or host.len == 0) return null;
    // Exact API hosts only — `endsWith("timbal.ai")` would accept lookalikes like notimbal.ai.
    if (!isTimbalApiHost(host)) return null;

    const base_url = try std.fmt.allocPrint(allocator, "https://{s}", .{host});
    errdefer allocator.free(base_url);

    return TimbalRemote{
        .org_id = try allocator.dupe(u8, org_id),
        .project_id = try allocator.dupe(u8, project_id),
        .base_url = base_url,
        .remote_name = try allocator.dupe(u8, remote_name),
    };
}

/// Extract remotes from a .git/config file and pick the Timbal one.
/// Prefers `origin` when it is a Timbal remote; otherwise the first match.
pub fn resolveTimbalRemoteFromConfig(allocator: std.mem.Allocator, config: []const u8) !?TimbalRemote {
    var origin_match: ?TimbalRemote = null;
    errdefer if (origin_match) |*r| r.deinit(allocator);
    var first_match: ?TimbalRemote = null;
    errdefer if (first_match) |*r| r.deinit(allocator);

    var current_remote: ?[]const u8 = null;
    var lines = std.mem.splitScalar(u8, config, '\n');
    while (lines.next()) |raw| {
        const line = std.mem.trim(u8, raw, " \t\r");
        if (line.len == 0 or line[0] == '#' or line[0] == ';') continue;

        if (line[0] == '[' and line[line.len - 1] == ']') {
            current_remote = null;
            // [remote "name"]
            if (std.mem.startsWith(u8, line, "[remote \"") and std.mem.endsWith(u8, line, "\"]")) {
                const inner = line["[remote \"".len .. line.len - 2];
                current_remote = inner;
            }
            continue;
        }

        const remote_name = current_remote orelse continue;
        // url = ...
        if (!std.mem.startsWith(u8, line, "url")) continue;
        const after = std.mem.trimLeft(u8, line["url".len..], " \t");
        if (after.len == 0 or after[0] != '=') continue;
        const url = std.mem.trim(u8, after[1..], " \t");

        const parsed = try parseTimbalRemoteUrl(allocator, url, remote_name) orelse continue;
        if (std.mem.eql(u8, remote_name, "origin")) {
            if (origin_match) |*old| old.deinit(allocator);
            origin_match = parsed;
        } else if (first_match == null) {
            first_match = parsed;
        } else {
            var tmp = parsed;
            tmp.deinit(allocator);
        }
    }

    if (origin_match) |r| {
        if (first_match) |*other| other.deinit(allocator);
        origin_match = null;
        return r;
    }
    if (first_match) |r| {
        first_match = null;
        return r;
    }
    return null;
}

fn runGitCapture(allocator: std.mem.Allocator, argv: []const []const u8) !?[]u8 {
    var child = std.process.Child.init(argv, allocator);
    child.stdout_behavior = .Pipe;
    child.stderr_behavior = .Ignore;
    try child.spawn();

    const stdout = child.stdout orelse return error.GitStdoutMissing;
    const out = try stdout.readToEndAlloc(allocator, 1024 * 1024);
    errdefer allocator.free(out);

    const term = try child.wait();
    switch (term) {
        .Exited => |code| {
            if (code != 0) {
                allocator.free(out);
                return null;
            }
        },
        else => {
            allocator.free(out);
            return null;
        },
    }

    const trimmed = std.mem.trim(u8, out, " \t\r\n");
    if (trimmed.len == 0) {
        allocator.free(out);
        return null;
    }
    if (trimmed.ptr == out.ptr and trimmed.len == out.len) return out;
    const owned = try allocator.dupe(u8, trimmed);
    allocator.free(out);
    return owned;
}

/// Walk up from `start_path` looking for a `.git` entry (dir or worktree file).
/// Returns an owned path to the checkout root (directory that contains `.git`).
fn findGitDir(allocator: std.mem.Allocator, start_path: []const u8) !?[]u8 {
    var current = try allocator.dupe(u8, start_path);
    errdefer allocator.free(current);

    while (true) {
        const git_path = try std.fmt.allocPrint(allocator, "{s}{s}.git", .{ current, sep });
        defer allocator.free(git_path);

        if (fs.cwd().access(git_path, .{})) |_| {
            return current;
        } else |_| {}

        const parent = fs.path.dirname(current) orelse {
            allocator.free(current);
            return null;
        };
        if (std.mem.eql(u8, parent, current)) {
            allocator.free(current);
            return null;
        }
        const next = try allocator.dupe(u8, parent);
        allocator.free(current);
        current = next;
    }
}

fn resolveGitConfigPath(allocator: std.mem.Allocator, repo_root: []const u8) ![]u8 {
    const git_path = try std.fmt.allocPrint(allocator, "{s}{s}.git", .{ repo_root, sep });
    defer allocator.free(git_path);

    const st = try fs.cwd().statFile(git_path);
    if (st.kind == .directory) {
        return std.fmt.allocPrint(allocator, "{s}{s}config", .{ git_path, sep });
    }

    // Worktree: .git file → gitdir: <path>
    const content = try fs.cwd().readFileAlloc(allocator, git_path, 4096);
    defer allocator.free(content);
    const trimmed = std.mem.trim(u8, content, " \t\r\n");
    if (!std.mem.startsWith(u8, trimmed, "gitdir:")) return error.InvalidGitDirFile;
    const gitdir = std.mem.trim(u8, trimmed["gitdir:".len..], " \t");

    // Relative gitdir is relative to the worktree root.
    const abs_gitdir = if (fs.path.isAbsolute(gitdir))
        try allocator.dupe(u8, gitdir)
    else
        try std.fmt.allocPrint(allocator, "{s}{s}{s}", .{ repo_root, sep, gitdir });
    defer allocator.free(abs_gitdir);

    // .../.git/worktrees/<name> → .../.git/config
    if (std.mem.lastIndexOf(u8, abs_gitdir, sep ++ "worktrees" ++ sep)) |idx| {
        return std.fmt.allocPrint(allocator, "{s}{s}config", .{ abs_gitdir[0..idx], sep });
    }
    return std.fmt.allocPrint(allocator, "{s}{s}config", .{ abs_gitdir, sep });
}

fn currentGitBranch(allocator: std.mem.Allocator) !?[]u8 {
    const out = try runGitCapture(allocator, &.{ "git", "rev-parse", "--abbrev-ref", "HEAD" });
    if (out) |b| {
        if (std.mem.eql(u8, b, "HEAD")) {
            // Detached HEAD — caller should require --rev or --default.
            allocator.free(b);
            return null;
        }
        return b;
    }
    return null;
}

/// The Timbal project root is the directory `timbal start` runs from and loads `.env` in.
/// Walk up from cwd (never past the git checkout root) to the nearest directory that has a
/// `workforce/` dir; fall back to the repo root, which is also what the platform builds from.
fn resolveProjectRoot(allocator: std.mem.Allocator, cwd_path: []const u8, repo_root: []const u8) ![]u8 {
    var current: []const u8 = cwd_path;
    while (true) {
        const probe = try std.fmt.allocPrint(allocator, "{s}{s}workforce", .{ current, sep });
        defer allocator.free(probe);
        if (fs.cwd().openDir(probe, .{})) |d| {
            var dir = d;
            dir.close();
            return allocator.dupe(u8, current);
        } else |_| {}

        if (std.mem.eql(u8, current, repo_root)) break;
        const parent = fs.path.dirname(current) orelse break;
        if (parent.len >= current.len) break;
        current = parent;
    }
    return allocator.dupe(u8, repo_root);
}

// ---------------------------------------------------------------------------
// Var names: reserved / platform-managed / type helpers
// ---------------------------------------------------------------------------

/// Never pushed. PORT and TIMBAL_PROJECT_SECRET are platform-assigned. TIMBAL_APP_ID is
/// per workforce member and lives in workforce/<name>/.env — a project-level value would
/// point every member at the same app.
pub fn isReservedVarName(name: []const u8) bool {
    return std.mem.eql(u8, name, "TIMBAL_PROJECT_SECRET") or
        std.mem.eql(u8, name, "PORT") or
        std.mem.eql(u8, name, "TIMBAL_APP_ID");
}

/// The platform's own namespace (TIMBAL_PROJECT_ENV_ID, TIMBAL_ORG_ID, VITE_TIMBAL_*, ...):
/// resolved by the platform before every deploy/preview, so they appear in the effective env
/// returned by pull but are never user-defined project vars and are never pushed. Loading
/// them under `timbal start` changes SDK routing (TIMBAL_PROJECT_ENV_ID points service calls
/// at the deployed gateway).
pub fn isPlatformManagedName(name: []const u8) bool {
    return std.mem.startsWith(u8, name, "TIMBAL_") or
        std.mem.startsWith(u8, name, "VITE_TIMBAL_") or
        // Injected into UI deploys alongside the VITE_TIMBAL_* mirrors; the only one outside the prefixes.
        std.mem.eql(u8, name, "VITE_AUTH_TIMBAL_IAM");
}

/// Canonical static type string, or null when the input is neither plain nor secret.
pub fn canonType(t: []const u8) ?[]const u8 {
    const trimmed = std.mem.trim(u8, t, " \t\r");
    if (std.ascii.eqlIgnoreCase(trimmed, "secret")) return "secret";
    if (std.ascii.eqlIgnoreCase(trimmed, "plain")) return "plain";
    return null;
}

fn isSecretType(t: []const u8) bool {
    return std.ascii.eqlIgnoreCase(std.mem.trim(u8, t, " \t\r"), "secret");
}

const strong_secret_tokens = [_][]const u8{
    "SECRET",     "SECRETS",     "PASSWORD", "PASSWORDS", "PASSWD", "PASSPHRASE", "PRIVATE",
    "CREDENTIAL", "CREDENTIALS", "DSN",      "TOKEN",     "TOKENS", "APIKEY",     "JWT",
    "SALT",       "SIGNING",     "BEARER",
};
const weak_secret_tokens = [_][]const u8{ "KEY", "KEYS", "PASS", "PWD" };
/// Tokens that turn TOKEN/TOKENS into a count, not a credential (MAX_TOKENS, TOKEN_LIMIT).
const count_tokens = [_][]const u8{ "MAX", "MIN", "NUM", "COUNT", "LIMIT", "BUDGET" };
const strong_secret_substrings = [_][]const u8{ "SECRET", "PASSWORD", "PASSWD", "APIKEY", "PRIVATE" };
const secret_value_prefixes = [_][]const u8{
    "sk-",   "sk_live_", "sk_test_", "rk_live_", "rk_test_", "ghp_", "gho_",       "ghu_", "github_pat_",
    "xoxb-", "xoxp-",    "xoxa-",    "xapp-",    "AKIA",     "AIza", "-----BEGIN", "eyJ",
};

fn tokenIn(tok: []const u8, set: []const []const u8) bool {
    for (set) |s| if (std.mem.eql(u8, tok, s)) return true;
    return false;
}

/// Heuristic used only for vars that are new to the platform and carry no metadata.
/// Errs towards `secret` (a secret is still injected everywhere; only the UI hides it).
pub fn inferSecret(name: []const u8, value: []const u8) bool {
    var upper_buf: [512]u8 = undefined;
    const upper = if (name.len <= upper_buf.len)
        std.ascii.upperString(upper_buf[0..name.len], name)
    else
        name;

    var has_public = false;
    var has_count = false;
    var strong = false;
    var weak = false;
    var token_word = false;
    var it = std.mem.tokenizeScalar(u8, upper, '_');
    while (it.next()) |tok| {
        if (std.mem.eql(u8, tok, "PUBLIC") or std.mem.eql(u8, tok, "PUBLISHABLE")) {
            has_public = true;
        } else if (std.mem.eql(u8, tok, "TOKEN") or std.mem.eql(u8, tok, "TOKENS")) {
            token_word = true;
        } else if (tokenIn(tok, &count_tokens)) {
            has_count = true;
        } else if (tokenIn(tok, &strong_secret_tokens)) {
            strong = true;
        } else if (tokenIn(tok, &weak_secret_tokens)) {
            weak = true;
        }
    }
    if (token_word and !has_count) strong = true;
    if (!strong) {
        for (strong_secret_substrings) |s| {
            if (std.mem.indexOf(u8, upper, s) != null) {
                strong = true;
                break;
            }
        }
    }
    if (strong) return true;
    if (weak and !has_public) return true;

    for (secret_value_prefixes) |p| {
        if (std.mem.startsWith(u8, value, p)) return true;
    }
    // Credentials embedded in a URL: scheme://user:pass@host/...
    if (std.mem.indexOf(u8, value, "://")) |i| {
        const rest = value[i + 3 ..];
        if (std.mem.indexOfScalar(u8, rest, '@')) |at| {
            const slash = std.mem.indexOfScalar(u8, rest, '/') orelse rest.len;
            if (at < slash and std.mem.indexOfScalar(u8, rest[0..at], ':') != null) return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// SyncVar + local .env store
// ---------------------------------------------------------------------------

pub const SyncVar = struct {
    name: []const u8,
    type: []const u8, // "plain" | "secret"
    value: []const u8,
    description: ?[]const u8 = null,
    /// True when the type came from explicit metadata (a `# type:` line, or the platform).
    /// Without it, push keeps the platform's current type instead of assuming plain.
    type_explicit: bool = false,
    /// Platform-computed runtime var: present in the effective deploy env but not user-defined.
    /// Written commented out on pull (active with --include-platform-vars); never pushed.
    managed: bool = false,
    /// Secret whose value the platform did not return. Written as a commented placeholder so a
    /// later push cannot overwrite the remote value with an empty string.
    value_missing: bool = false,

    pub fn deinit(self: *SyncVar, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        allocator.free(self.type);
        allocator.free(self.value);
        if (self.description) |d| allocator.free(d);
    }
};

fn freeSyncVars(allocator: std.mem.Allocator, vars: *std.ArrayList(SyncVar)) void {
    for (vars.items) |*v| v.deinit(allocator);
    vars.deinit();
}

fn findSyncVar(vars: []const SyncVar, name: []const u8) ?usize {
    for (vars, 0..) |v, i| if (std.mem.eql(u8, v.name, name)) return i;
    return null;
}

fn needsQuoting(value: []const u8) bool {
    if (value.len == 0) return true;
    for (value) |c| {
        switch (c) {
            ' ', '\t', '#', '"', '\'', '=', '\\' => return true,
            else => {},
        }
    }
    return false;
}

/// Quote a value the way `timbal start` reads it: strip matching outer quotes only,
/// no backslash escapes. Prefer double quotes; use single quotes when the value
/// contains `"` but not `'`. Values with both quote styles are wrapped in `"` and
/// rely on first/last-char stripping (same as start). Newlines are unsupported in
/// this line-oriented format — callers should not expect multiline values to round-trip.
fn appendQuotedValue(buf: *std.ArrayList(u8), value: []const u8) !void {
    const has_dq = std.mem.indexOfScalar(u8, value, '"') != null;
    const has_sq = std.mem.indexOfScalar(u8, value, '\'') != null;
    const quote: u8 = if (has_dq and !has_sq) '\'' else '"';
    try buf.append(quote);
    try buf.appendSlice(value);
    try buf.append(quote);
}

/// Append `value` quoted iff needed, flattening newlines so the line-oriented file stays valid.
fn appendEnvValue(allocator: std.mem.Allocator, buf: *std.ArrayList(u8), value: []const u8) !void {
    if (std.mem.indexOfAny(u8, value, "\n\r") != null) {
        const flat = try allocator.alloc(u8, value.len);
        defer allocator.free(flat);
        for (value, 0..) |c, i| {
            flat[i] = if (c == '\n' or c == '\r') ' ' else c;
        }
        if (needsQuoting(flat)) {
            try appendQuotedValue(buf, flat);
        } else {
            try buf.appendSlice(flat);
        }
    } else if (needsQuoting(value)) {
        try appendQuotedValue(buf, value);
    } else {
        try buf.appendSlice(value);
    }
}

pub const FormatOptions = struct {
    /// Write platform-managed vars as active lines instead of commented placeholders.
    managed_active: bool = false,
};

fn isMultiline(value: []const u8) bool {
    return std.mem.indexOfAny(u8, value, "\n\r") != null;
}

/// Key rules `timbal start`'s loader enforces: letters, digits, underscore, no leading digit.
/// Anything else is silently dropped by start, so pushing it would never work locally.
pub fn isValidEnvKey(key: []const u8) bool {
    if (key.len == 0 or std.ascii.isDigit(key[0])) return false;
    for (key) |c| {
        if (!(std.ascii.isAlphanumeric(c) or c == '_')) return false;
    }
    return true;
}

fn appendEnvEntry(allocator: std.mem.Allocator, buf: *std.ArrayList(u8), v: SyncVar, active_in: bool) !void {
    // Reserved names are never active in a project-level file, whatever the caller asked.
    const active = active_in and !isReservedVarName(v.name);
    if (v.managed) try buf.appendSlice("# managed: platform\n");
    try buf.appendSlice("# type: ");
    try buf.appendSlice(v.type);
    try buf.append('\n');
    if (v.description) |d| {
        if (d.len > 0) {
            try buf.appendSlice("# description: ");
            // Keep description on one line.
            for (d) |c| {
                try buf.append(if (c == '\n' or c == '\r') ' ' else c);
            }
            try buf.append('\n');
        }
    }
    if (v.value_missing) {
        try buf.appendSlice("# value not returned by the platform: set it here to use it locally, or leave the\n");
        try buf.appendSlice("# line commented so `timbal env push` keeps the remote value untouched.\n");
        try buf.appendSlice("# ");
        try buf.appendSlice(v.name);
        try buf.appendSlice("=\n\n");
        return;
    }
    if (isMultiline(v.value)) {
        // This line-oriented format (and `timbal start`'s loader) cannot carry newlines. Writing
        // a flattened copy would break the value locally and, if pushed back, on the platform.
        try buf.appendSlice("# multi-line value: not representable in a .env file, so it is kept on the platform only.\n");
        try buf.appendSlice("# Provide it locally via your shell or `timbal start --env`; leave this line commented.\n");
        try buf.appendSlice("# ");
        try buf.appendSlice(v.name);
        try buf.appendSlice("=\n\n");
        return;
    }
    if (std.mem.eql(u8, v.name, "TIMBAL_APP_ID")) {
        try buf.appendSlice("# TIMBAL_APP_ID is per workforce member — it belongs in workforce/<name>/.env, not here.\n");
    }
    if (!active) try buf.appendSlice("# ");
    try buf.appendSlice(v.name);
    try buf.append('=');
    try appendEnvValue(allocator, buf, v.value);
    try buf.append('\n');
    try buf.append('\n');
}

/// Serialize SyncVars to a .env file with type/description comment metadata.
/// Format is compatible with `timbal start`'s .env loader (quote strip, no escapes).
/// User-defined vars come first; platform-managed vars go in a trailing block that is
/// commented out unless `managed_active` is set.
pub fn formatEnvFile(allocator: std.mem.Allocator, rev: []const u8, vars: []const SyncVar, fmt_opts: FormatOptions) ![]u8 {
    var buf = std.ArrayList(u8).init(allocator);
    errdefer buf.deinit();

    try buf.appendSlice("# Synced by `timbal env pull`. Keep this file gitignored — secrets are plaintext.\n");
    try buf.appendSlice("# rev: ");
    try buf.appendSlice(rev);
    try buf.append('\n');
    try buf.appendSlice("#\n");
    try buf.appendSlice("# `timbal env push` reads the metadata comments directly above each var:\n");
    try buf.appendSlice("#   # type: secret | plain   (shorthand: `# secret` / `# plain`)\n");
    try buf.appendSlice("#   # description: ...\n");
    try buf.appendSlice("# A var without metadata keeps its current platform type; a brand-new var is inferred\n");
    try buf.appendSlice("# from its name/value. Override at the CLI with --secret NAME / --plain NAME. The type is\n");
    try buf.appendSlice("# fixed when a var is created on the platform; change it in the Timbal UI afterwards.\n");
    try buf.append('\n');

    var managed_count: usize = 0;
    for (vars) |v| {
        if (v.managed) {
            managed_count += 1;
            continue;
        }
        try appendEnvEntry(allocator, &buf, v, true);
    }

    if (managed_count > 0) {
        try buf.appendSlice("# ---- Platform-managed runtime vars ----\n");
        if (fmt_opts.managed_active) {
            try buf.appendSlice("# Written ACTIVE because of --include-platform-vars. The platform resolves these before every\n");
            try buf.appendSlice("# deploy/preview; under `timbal start` they can reroute service calls (TIMBAL_PROJECT_ENV_ID) to\n");
            try buf.appendSlice("# the deployed gateway. `timbal env push` never sends TIMBAL_* vars.\n");
        } else {
            try buf.appendSlice("# Resolved by the platform before every deploy/preview; not user-defined project vars. Kept\n");
            try buf.appendSlice("# commented out on purpose: loading them under `timbal start` breaks local routing\n");
            try buf.appendSlice("# (TIMBAL_PROJECT_ENV_ID sends service calls to the deployed gateway instead of localhost).\n");
            try buf.appendSlice("# Uncomment a line only if you need it locally; `timbal env push` never sends TIMBAL_* vars.\n");
        }
        try buf.append('\n');
        for (vars) |v| {
            if (!v.managed) continue;
            try appendEnvEntry(allocator, &buf, v, fmt_opts.managed_active);
        }
    }

    return buf.toOwnedSlice();
}

const Assignment = struct { key: []const u8, value: []const u8 };

/// Parse one `KEY=VALUE` line the way `timbal start` does: optional `export `, matching
/// outer quotes stripped, no escapes, no trailing comments. Null for blanks/comments.
pub fn parseEnvAssignment(raw_line: []const u8) ?Assignment {
    var line = std.mem.trim(u8, raw_line, " \t\r");
    if (line.len == 0 or line[0] == '#') return null;
    if (std.mem.startsWith(u8, line, "export ")) {
        line = std.mem.trimLeft(u8, line["export ".len..], " \t");
    }
    const eq = std.mem.indexOfScalar(u8, line, '=') orelse return null;
    const key = std.mem.trim(u8, line[0..eq], " \t");
    if (key.len == 0) return null;

    var value = std.mem.trim(u8, line[eq + 1 ..], " \t");
    if (value.len >= 2) {
        const first = value[0];
        const last = value[value.len - 1];
        if ((first == '"' and last == '"') or (first == '\'' and last == '\'')) {
            value = value[1 .. value.len - 1];
        }
    }
    return .{ .key = key, .value = value };
}

/// Parse a local .env written by pull (or a plain KEY=VALUE file).
/// Metadata comments bind to the next assignment; a blank line ends a metadata block so a
/// stray `# type: secret` left behind after deleting a var cannot retag its neighbour.
/// Type defaults to "plain" with `type_explicit = false` when metadata is missing.
pub fn parseEnvFile(allocator: std.mem.Allocator, content: []const u8) !std.ArrayList(SyncVar) {
    var out = std.ArrayList(SyncVar).init(allocator);
    errdefer freeSyncVars(allocator, &out);

    var pending_type: ?[]const u8 = null;
    var pending_desc: ?[]const u8 = null;
    var pending_managed = false;
    defer if (pending_type) |t| allocator.free(t);
    defer if (pending_desc) |d| allocator.free(d);

    var lines = std.mem.splitScalar(u8, content, '\n');
    while (lines.next()) |raw_line| {
        const line = std.mem.trim(u8, raw_line, " \t\r");
        if (line.len == 0) {
            if (pending_type) |t| allocator.free(t);
            if (pending_desc) |d| allocator.free(d);
            pending_type = null;
            pending_desc = null;
            pending_managed = false;
            continue;
        }

        if (line[0] == '#') {
            const body = std.mem.trim(u8, line[1..], " \t");
            if (std.mem.startsWith(u8, body, "type:")) {
                const t = std.mem.trim(u8, body["type:".len..], " \t");
                if (pending_type) |old| allocator.free(old);
                pending_type = try allocator.dupe(u8, canonType(t) orelse t);
            } else if (std.mem.startsWith(u8, body, "description:")) {
                const d = std.mem.trim(u8, body["description:".len..], " \t");
                if (pending_desc) |old| allocator.free(old);
                pending_desc = try allocator.dupe(u8, d);
            } else if (std.mem.startsWith(u8, body, "managed:")) {
                const m = std.mem.trim(u8, body["managed:".len..], " \t");
                pending_managed = std.ascii.eqlIgnoreCase(m, "platform");
            } else if (canonType(body)) |shorthand| {
                if (pending_type) |old| allocator.free(old);
                pending_type = try allocator.dupe(u8, shorthand);
            }
            continue;
        }

        const assignment = parseEnvAssignment(line) orelse continue;

        const value = try allocator.dupe(u8, assignment.value);
        errdefer allocator.free(value);

        const explicit = pending_type != null;
        const typ = if (pending_type) |t| blk: {
            pending_type = null;
            break :blk t;
        } else try allocator.dupe(u8, "plain");
        errdefer allocator.free(typ);

        const desc = blk: {
            if (pending_desc) |d| {
                pending_desc = null;
                break :blk d;
            }
            break :blk null;
        };
        errdefer if (desc) |d| allocator.free(d);

        const name = try allocator.dupe(u8, assignment.key);
        errdefer allocator.free(name);

        try out.append(.{
            .name = name,
            .type = typ,
            .value = value,
            .description = desc,
            .type_explicit = explicit,
            .managed = pending_managed,
        });
        pending_managed = false;
    }

    return out;
}

pub const UpsertOutcome = enum { added, updated, unchanged };

pub const UpsertResult = struct {
    content: []u8,
    outcome: UpsertOutcome,
    /// First replaced value when `outcome == .updated`.
    previous: ?[]u8 = null,

    pub fn deinit(self: *UpsertResult, allocator: std.mem.Allocator) void {
        allocator.free(self.content);
        if (self.previous) |p| allocator.free(p);
    }
};

/// Set `key=value` in .env content while preserving every other byte (comments, order,
/// blank lines, CRLF). Existing definitions are rewritten in place — all of them, so
/// last-wins loaders agree — otherwise the assignment is appended (with `comment` above it).
pub fn upsertEnvLine(
    allocator: std.mem.Allocator,
    existing: ?[]const u8,
    key: []const u8,
    value: []const u8,
    comment: ?[]const u8,
) !UpsertResult {
    var out = std.ArrayList(u8).init(allocator);
    errdefer out.deinit();
    var previous: ?[]u8 = null;
    errdefer if (previous) |p| allocator.free(p);

    const content = existing orelse "";
    const eol: []const u8 = if (std.mem.indexOf(u8, content, "\r\n") != null) "\r\n" else "\n";

    var found = false;
    var changed = false;
    var first = true;
    var it = std.mem.splitScalar(u8, content, '\n');
    while (it.next()) |raw| {
        if (!first) try out.append('\n');
        first = false;

        const has_cr = raw.len > 0 and raw[raw.len - 1] == '\r';
        if (parseEnvAssignment(raw)) |a| {
            if (std.mem.eql(u8, a.key, key)) {
                found = true;
                if (std.mem.eql(u8, a.value, value)) {
                    try out.appendSlice(raw);
                } else {
                    changed = true;
                    if (previous == null) previous = try allocator.dupe(u8, a.value);
                    try out.appendSlice(key);
                    try out.append('=');
                    try appendEnvValue(allocator, &out, value);
                    if (has_cr) try out.append('\r');
                }
                continue;
            }
        }
        try out.appendSlice(raw);
    }

    if (found) {
        return .{
            .content = try out.toOwnedSlice(),
            .outcome = if (changed) .updated else .unchanged,
            .previous = previous,
        };
    }

    if (content.len > 0 and !std.mem.endsWith(u8, content, "\n")) try out.appendSlice(eol);
    if (comment) |c| {
        try out.appendSlice("# ");
        try out.appendSlice(c);
        try out.appendSlice(eol);
    }
    try out.appendSlice(key);
    try out.append('=');
    try appendEnvValue(allocator, &out, value);
    try out.appendSlice(eol);
    return .{ .content = try out.toOwnedSlice(), .outcome = .added, .previous = null };
}

// ---------------------------------------------------------------------------
// JSON helpers (tolerant, shape-agnostic)
// ---------------------------------------------------------------------------

fn jsonStr(obj: std.json.ObjectMap, key: []const u8) ?[]const u8 {
    const v = obj.get(key) orelse return null;
    return switch (v) {
        .string => |s| s,
        else => null,
    };
}

fn jsonBool(obj: std.json.ObjectMap, key: []const u8) ?bool {
    const v = obj.get(key) orelse return null;
    return switch (v) {
        .bool => |b| b,
        else => null,
    };
}

/// Ids come back as int64 in some schemas and strings in others; normalize to an owned string.
fn jsonIdString(allocator: std.mem.Allocator, obj: std.json.ObjectMap, key: []const u8) !?[]u8 {
    const v = obj.get(key) orelse return null;
    return switch (v) {
        .string => |s| try allocator.dupe(u8, s),
        .integer => |i| try std.fmt.allocPrint(allocator, "{d}", .{i}),
        .number_string => |s| try allocator.dupe(u8, s),
        else => null,
    };
}

fn jsonObject(v: std.json.Value) ?std.json.ObjectMap {
    return switch (v) {
        .object => |o| o,
        else => null,
    };
}

fn jsonArray(v: std.json.Value) ?std.json.Array {
    return switch (v) {
        .array => |a| a,
        else => null,
    };
}

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------

const ApiResponse = struct {
    status: u16,
    body: []u8,
};

fn printApiError(status: u16, body: []const u8, rev: ?[]const u8) void {
    const stderr = std.io.getStdErr().writer();
    if (status == 403) {
        // Platform: no env tracks the branch, or missing projects.vars.manage.
        const lower_buf_len = @min(body.len, 512);
        var looks_like_branch = false;
        if (lower_buf_len > 0) {
            // Case-insensitive-ish scan for the common branch-tracking error.
            if (std.ascii.indexOfIgnoreCase(body[0..lower_buf_len], "no env tracks branch") != null or
                std.ascii.indexOfIgnoreCase(body[0..lower_buf_len], "tracks branch") != null)
            {
                looks_like_branch = true;
            }
        }
        if (looks_like_branch) {
            if (rev) |r| {
                stderr.print(
                    "Error: no project environment tracks branch '{s}'.\n" ++
                        "Create an env for that branch in the Timbal UI, pass --rev <branch>, or use --default.\n",
                    .{r},
                ) catch {};
            } else {
                stderr.writeAll(
                    "Error: no project environment tracks the requested branch.\n" ++
                        "Create an env for that branch in the Timbal UI, pass --rev <branch>, or use --default.\n",
                ) catch {};
            }
        } else {
            stderr.writeAll(
                "Error: forbidden (HTTP 403).\n" ++
                    "Your API key/session needs `projects.vars.manage` on this project.\n" ++
                    "Also check that an environment exists for the target branch (`--rev` / current branch).\n",
            ) catch {};
        }
        if (body.len > 0) {
            const snippet = if (body.len > 500) body[0..500] else body;
            stderr.print("{s}\n", .{snippet}) catch {};
        }
        return;
    }

    stderr.print("Error: API returned HTTP {d}\n", .{status}) catch {};
    if (body.len > 0) {
        const snippet = if (body.len > 500) body[0..500] else body;
        stderr.print("{s}\n", .{snippet}) catch {};
    }
}

/// Perform a request and return status + body without judging the status code.
fn apiRequestRaw(
    allocator: std.mem.Allocator,
    method: std.http.Method,
    url: []const u8,
    api_key: []const u8,
    payload: ?[]const u8,
    verbose: bool,
) !ApiResponse {
    var client: std.http.Client = .{ .allocator = allocator };
    defer client.deinit();

    const auth = try std.fmt.allocPrint(allocator, "Bearer {s}", .{api_key});
    defer allocator.free(auth);

    var headers_buf: [3]std.http.Header = undefined;
    var n: usize = 0;
    headers_buf[n] = .{ .name = "Authorization", .value = auth };
    n += 1;
    headers_buf[n] = .{ .name = "Accept", .value = "application/json" };
    n += 1;
    if (payload != null) {
        headers_buf[n] = .{ .name = "Content-Type", .value = "application/json" };
        n += 1;
    }

    if (verbose) {
        const stderr = std.io.getStdErr().writer();
        try stderr.print("→ {s} {s}\n", .{ @tagName(method), url });
    }

    var body = std.ArrayList(u8).init(allocator);
    errdefer body.deinit();

    const result = client.fetch(.{
        .location = .{ .url = url },
        .method = method,
        .payload = payload,
        .extra_headers = headers_buf[0..n],
        .response_storage = .{ .dynamic = &body },
        .max_append_size = 16 * 1024 * 1024,
    }) catch |err| {
        const stderr = std.io.getStdErr().writer();
        try stderr.print("Error: request failed: {}\n", .{err});
        return error.HttpError;
    };

    return .{ .status = @intFromEnum(result.status), .body = try body.toOwnedSlice() };
}

/// Request that must succeed: prints a friendly error and fails on non-2xx.
fn apiRequest(
    allocator: std.mem.Allocator,
    method: std.http.Method,
    url: []const u8,
    api_key: []const u8,
    payload: ?[]const u8,
    verbose: bool,
    rev: ?[]const u8,
) ![]u8 {
    const res = try apiRequestRaw(allocator, method, url, api_key, payload, verbose);
    if (res.status < 200 or res.status >= 300) {
        printApiError(res.status, res.body, rev);
        allocator.free(res.body);
        return error.HttpStatus;
    }
    return res.body;
}

fn projectUrl(allocator: std.mem.Allocator, remote: TimbalRemote, suffix: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator, "{s}/orgs/{s}/projects/{s}{s}", .{ remote.base_url, remote.org_id, remote.project_id, suffix });
}

// --- effective env (pull) ---------------------------------------------------

pub const PullResult = struct {
    rev: []u8,
    vars: std.ArrayList(SyncVar),

    pub fn deinit(self: *PullResult, allocator: std.mem.Allocator) void {
        allocator.free(self.rev);
        freeSyncVars(allocator, &self.vars);
    }
};

/// Parse `GET /vars/pull`. Accepts `value` as a string, as a `VarValue` object
/// (`{type, value|decrypted|preview}`), or missing/null → `value_missing`.
pub fn parsePullResponse(allocator: std.mem.Allocator, body: []const u8) !PullResult {
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
    defer parsed.deinit();

    const root = jsonObject(parsed.value) orelse return error.UnexpectedResponse;
    const rev = try allocator.dupe(u8, jsonStr(root, "rev") orelse "");
    errdefer allocator.free(rev);

    var vars = std.ArrayList(SyncVar).init(allocator);
    errdefer freeSyncVars(allocator, &vars);

    const arr = jsonArray(root.get("vars") orelse return error.UnexpectedResponse) orelse return error.UnexpectedResponse;
    for (arr.items) |item| {
        const obj = jsonObject(item) orelse continue;
        const name = jsonStr(obj, "name") orelse continue;

        var typ: []const u8 = jsonStr(obj, "type") orelse "plain";
        var value: ?[]const u8 = null;
        if (obj.get("value")) |vv| {
            switch (vv) {
                .string => |s| value = s,
                .object => |vo| {
                    if (jsonStr(vo, "type")) |t| typ = t;
                    value = jsonStr(vo, "value") orelse jsonStr(vo, "decrypted");
                },
                else => {},
            }
        }
        const canon = canonType(typ) orelse "plain";
        const missing = value == null;
        // A plain var can legitimately be empty; only secrets get placeholder treatment.
        const treat_missing = missing and std.mem.eql(u8, canon, "secret");
        // Forward-compatible: honour an explicit platform flag if the API starts sending one.
        const flagged_managed = (jsonBool(obj, "managed") orelse false) or
            (if (jsonStr(obj, "source")) |s| std.ascii.eqlIgnoreCase(s, "platform") else false);

        var sv = SyncVar{
            .name = try allocator.dupe(u8, name),
            .type = undefined,
            .value = undefined,
            .type_explicit = true,
            .managed = flagged_managed,
            .value_missing = treat_missing,
        };
        errdefer allocator.free(sv.name);
        sv.type = try allocator.dupe(u8, canon);
        errdefer allocator.free(sv.type);
        sv.value = try allocator.dupe(u8, value orelse "");
        errdefer allocator.free(sv.value);
        if (jsonStr(obj, "description")) |d| sv.description = try allocator.dupe(u8, d);
        errdefer if (sv.description) |d| allocator.free(d);
        try vars.append(sv);
    }

    return .{ .rev = rev, .vars = vars };
}

// --- user-defined vars (list) -----------------------------------------------

pub const RemoteVar = struct {
    id: []const u8,
    name: []const u8,
    type: []const u8, // canonical "plain" | "secret"
    description: ?[]const u8 = null,
    applies_to_all_envs: bool = false,
    env_ids: []const []const u8 = &.{},

    pub fn deinit(self: *RemoteVar, allocator: std.mem.Allocator) void {
        allocator.free(self.id);
        allocator.free(self.name);
        allocator.free(self.type);
        if (self.description) |d| allocator.free(d);
        for (self.env_ids) |e| allocator.free(e);
        allocator.free(self.env_ids);
    }

    pub fn hasEnv(self: RemoteVar, env_id: []const u8) bool {
        for (self.env_ids) |e| if (std.mem.eql(u8, e, env_id)) return true;
        return false;
    }
};

fn freeRemoteVars(allocator: std.mem.Allocator, vars: *std.ArrayList(RemoteVar)) void {
    for (vars.items) |*v| v.deinit(allocator);
    vars.deinit();
}

/// Type of a user-defined var by name. Vars can exist once per env; if any copy is a
/// secret the name is treated as secret (the conservative direction). Null when the
/// name is not user-defined on the platform.
pub fn remoteTypeFor(remote: []const RemoteVar, name: []const u8) ?[]const u8 {
    var found: ?[]const u8 = null;
    for (remote) |rv| {
        if (!std.mem.eql(u8, rv.name, name)) continue;
        if (isSecretType(rv.type)) return "secret";
        found = "plain";
    }
    return found;
}

/// Parse `GET /vars` (`ListVarsResBody`): `{vars: [{id, name, value: {type, ...}, envs, applies_to_all_envs}]}`.
pub fn parseRemoteVarList(allocator: std.mem.Allocator, body: []const u8) !std.ArrayList(RemoteVar) {
    var out = std.ArrayList(RemoteVar).init(allocator);
    errdefer freeRemoteVars(allocator, &out);

    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
    defer parsed.deinit();

    const root = jsonObject(parsed.value) orelse return error.UnexpectedResponse;
    const arr = jsonArray(root.get("vars") orelse return error.UnexpectedResponse) orelse return error.UnexpectedResponse;

    for (arr.items) |item| {
        const obj = jsonObject(item) orelse continue;
        const name = jsonStr(obj, "name") orelse continue;

        var typ: []const u8 = jsonStr(obj, "type") orelse "plain";
        if (obj.get("value")) |vv| {
            if (jsonObject(vv)) |vo| {
                if (jsonStr(vo, "type")) |t| typ = t;
            }
        }

        var env_ids = std.ArrayList([]const u8).init(allocator);
        errdefer {
            for (env_ids.items) |e| allocator.free(e);
            env_ids.deinit();
        }
        if (obj.get("envs")) |ev| {
            if (jsonArray(ev)) |ea| {
                for (ea.items) |e| {
                    const eo = jsonObject(e) orelse continue;
                    if (try jsonIdString(allocator, eo, "id")) |eid| {
                        errdefer allocator.free(eid);
                        try env_ids.append(eid);
                    }
                }
            }
        }

        var rv = RemoteVar{
            .id = (try jsonIdString(allocator, obj, "id")) orelse try allocator.dupe(u8, ""),
            .name = undefined,
            .type = undefined,
            .applies_to_all_envs = jsonBool(obj, "applies_to_all_envs") orelse false,
        };
        errdefer allocator.free(rv.id);
        rv.name = try allocator.dupe(u8, name);
        errdefer allocator.free(rv.name);
        rv.type = try allocator.dupe(u8, canonType(typ) orelse "plain");
        errdefer allocator.free(rv.type);
        if (jsonStr(obj, "description")) |d| rv.description = try allocator.dupe(u8, d);
        errdefer if (rv.description) |d| allocator.free(d);
        rv.env_ids = try env_ids.toOwnedSlice();
        try out.append(rv);
    }

    return out;
}

const RemoteVarListing = struct {
    vars: std.ArrayList(RemoteVar),
    ok: bool,
    status: u16,
    error_body: ?[]u8,

    fn deinit(self: *RemoteVarListing, allocator: std.mem.Allocator) void {
        freeRemoteVars(allocator, &self.vars);
        if (self.error_body) |b| allocator.free(b);
    }
};

/// Fetch user-defined project vars. Never fails on HTTP status — callers decide how hard
/// to fail (push needs it; pull degrades gracefully).
fn fetchRemoteVarListing(allocator: std.mem.Allocator, remote: TimbalRemote, api_key: []const u8, verbose: bool) !RemoteVarListing {
    const url = try projectUrl(allocator, remote, "/vars");
    defer allocator.free(url);

    const res = try apiRequestRaw(allocator, .GET, url, api_key, null, verbose);
    if (res.status < 200 or res.status >= 300) {
        return .{ .vars = std.ArrayList(RemoteVar).init(allocator), .ok = false, .status = res.status, .error_body = res.body };
    }
    defer allocator.free(res.body);

    const vars = parseRemoteVarList(allocator, res.body) catch {
        return .{ .vars = std.ArrayList(RemoteVar).init(allocator), .ok = false, .status = res.status, .error_body = null };
    };
    return .{ .vars = vars, .ok = true, .status = res.status, .error_body = null };
}

/// `GET /vars/{id}` → decrypted secret value when the platform returns one.
pub fn parseVarDecrypted(allocator: std.mem.Allocator, body: []const u8) !?[]u8 {
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
    defer parsed.deinit();
    const root = jsonObject(parsed.value) orelse return null;
    const vv = root.get("value") orelse return null;
    switch (vv) {
        .string => |s| return try allocator.dupe(u8, s),
        .object => |vo| {
            if (jsonStr(vo, "decrypted")) |d| return try allocator.dupe(u8, d);
            if (jsonStr(vo, "value")) |v| return try allocator.dupe(u8, v);
            return null;
        },
        else => return null,
    }
}

fn fetchVarDecrypted(allocator: std.mem.Allocator, remote: TimbalRemote, api_key: []const u8, var_id: []const u8, verbose: bool) !?[]u8 {
    if (var_id.len == 0) return null;
    const suffix = try std.fmt.allocPrint(allocator, "/vars/{s}", .{var_id});
    defer allocator.free(suffix);
    const url = try projectUrl(allocator, remote, suffix);
    defer allocator.free(url);

    const res = try apiRequestRaw(allocator, .GET, url, api_key, null, verbose);
    defer allocator.free(res.body);
    if (res.status < 200 or res.status >= 300) return null;
    return parseVarDecrypted(allocator, res.body) catch null;
}

// --- environments ------------------------------------------------------------

/// `GET /envs` → id of the environment whose `branch` equals `branch`.
pub fn parseEnvIdForBranch(allocator: std.mem.Allocator, body: []const u8, branch: []const u8) !?[]u8 {
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
    defer parsed.deinit();
    const root = jsonObject(parsed.value) orelse return null;
    const arr = jsonArray(root.get("envs") orelse return null) orelse return null;
    for (arr.items) |item| {
        const obj = jsonObject(item) orelse continue;
        const b = jsonStr(obj, "branch") orelse continue;
        if (std.mem.eql(u8, b, branch)) return try jsonIdString(allocator, obj, "id");
    }
    return null;
}

fn fetchEnvIdForBranch(allocator: std.mem.Allocator, remote: TimbalRemote, api_key: []const u8, branch: []const u8, verbose: bool) !?[]u8 {
    const url = try projectUrl(allocator, remote, "/envs");
    defer allocator.free(url);
    const res = try apiRequestRaw(allocator, .GET, url, api_key, null, verbose);
    defer allocator.free(res.body);
    if (res.status < 200 or res.status >= 300) return null;
    return parseEnvIdForBranch(allocator, res.body, branch) catch null;
}

// --- workforce components ---------------------------------------------------

pub const RemoteComponent = struct {
    id: []const u8, // numeric app id (the value the SDK reads from TIMBAL_APP_ID)
    name: []const u8,
    uid: ?[]const u8, // manifest _id from timbal.yaml, when the platform knows it

    pub fn deinit(self: *RemoteComponent, allocator: std.mem.Allocator) void {
        allocator.free(self.id);
        allocator.free(self.name);
        if (self.uid) |u| allocator.free(u);
    }

    /// A member that exists in the worktree but has no `OrgsApps` row yet is listed with a
    /// negative synthetic hash of its manifest id. That is not a real app id and must never
    /// be written as TIMBAL_APP_ID — traces sent under it would be dropped or misattributed.
    pub fn isRegistered(self: RemoteComponent) bool {
        return self.id.len > 0 and self.id[0] != '-';
    }
};

fn freeRemoteComponents(allocator: std.mem.Allocator, comps: *std.ArrayList(RemoteComponent)) void {
    for (comps.items) |*c| c.deinit(allocator);
    comps.deinit();
}

/// Parse `GET /workforce?rev=` (`ListWorkforceResBody`): `{workforce: [{id, name, type, uid}]}`.
pub fn parseWorkforceList(allocator: std.mem.Allocator, body: []const u8) !std.ArrayList(RemoteComponent) {
    var out = std.ArrayList(RemoteComponent).init(allocator);
    errdefer freeRemoteComponents(allocator, &out);

    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
    defer parsed.deinit();

    const root = jsonObject(parsed.value) orelse return error.UnexpectedResponse;
    const arr = jsonArray(root.get("workforce") orelse return error.UnexpectedResponse) orelse return error.UnexpectedResponse;
    for (arr.items) |item| {
        const obj = jsonObject(item) orelse continue;
        const id = (try jsonIdString(allocator, obj, "id")) orelse continue;
        errdefer allocator.free(id);
        if (id.len == 0) {
            allocator.free(id);
            continue;
        }
        const name = try allocator.dupe(u8, jsonStr(obj, "name") orelse "");
        errdefer allocator.free(name);
        var uid: ?[]const u8 = null;
        if (jsonStr(obj, "uid")) |u| {
            if (u.len > 0) uid = try allocator.dupe(u8, u);
        }
        errdefer if (uid) |u| allocator.free(u);
        try out.append(.{ .id = id, .name = name, .uid = uid });
    }
    return out;
}

fn fetchWorkforce(allocator: std.mem.Allocator, remote: TimbalRemote, api_key: []const u8, rev: []const u8, verbose: bool) !?std.ArrayList(RemoteComponent) {
    const encoded = try urlEncodeQuery(allocator, rev);
    defer allocator.free(encoded);
    const suffix = try std.fmt.allocPrint(allocator, "/workforce?rev={s}", .{encoded});
    defer allocator.free(suffix);
    const url = try projectUrl(allocator, remote, suffix);
    defer allocator.free(url);

    const res = try apiRequestRaw(allocator, .GET, url, api_key, null, verbose);
    defer allocator.free(res.body);
    if (res.status < 200 or res.status >= 300) {
        const stderr = std.io.getStdErr().writer();
        try stderr.print(
            "{s}Warning:{s} could not list workforce components for rev {s} (HTTP {d}); TIMBAL_APP_ID not synced.\n",
            .{ Color.bold_yellow, Color.reset, rev, res.status },
        );
        if (verbose and res.body.len > 0) {
            const snippet = if (res.body.len > 500) res.body[0..500] else res.body;
            try stderr.print("{s}\n", .{snippet});
        }
        return null;
    }
    return parseWorkforceList(allocator, res.body) catch {
        const stderr = std.io.getStdErr().writer();
        try stderr.print("{s}Warning:{s} unexpected workforce response; TIMBAL_APP_ID not synced.\n", .{ Color.bold_yellow, Color.reset });
        return null;
    };
}

pub const MatchKind = enum { uid, name };
pub const ComponentMatch = struct { index: usize, kind: MatchKind };

/// Find the platform component for a local member. The manifest uid is canonical. A name
/// match is accepted only when the platform component has no uid at all: a component
/// with a *different* uid is a different manifest identity, and pointing local traces at
/// it would be wrong even if the directory name agrees.
pub fn matchComponent(comps: []const RemoteComponent, member_name: []const u8, member_uid: []const u8, claimed: []const bool) ?ComponentMatch {
    for (comps, 0..) |c, i| {
        if (claimed[i]) continue;
        if (c.uid) |u| {
            if (std.mem.eql(u8, u, member_uid)) return .{ .index = i, .kind = .uid };
        }
    }
    var found: ?usize = null;
    var count: usize = 0;
    for (comps, 0..) |c, i| {
        if (claimed[i] or c.uid != null) continue;
        if (std.mem.eql(u8, c.name, member_name)) {
            count += 1;
            found = i;
        }
    }
    if (count == 1) return .{ .index = found.?, .kind = .name };
    return null;
}

/// Same-name component whose uid differs from the local manifest (for diagnostics only).
fn findNameOnlyConflict(comps: []const RemoteComponent, member_name: []const u8, member_uid: []const u8) ?usize {
    for (comps, 0..) |c, i| {
        if (!std.mem.eql(u8, c.name, member_name)) continue;
        if (c.uid) |u| {
            if (!std.mem.eql(u8, u, member_uid)) return i;
        }
    }
    return null;
}

// --- push -------------------------------------------------------------------

pub const TypeSource = enum { flag, file, remote, inferred };
pub const PlanAction = enum { push, skip_reserved, skip_managed, blocked_downgrade };

pub const PlanEntry = struct {
    action: PlanAction,
    type: []const u8, // static "plain" | "secret"
    source: TypeSource,
    remote_type: ?[]const u8, // static, null when not user-defined on the platform
};

fn nameIn(names: []const []const u8, name: []const u8) bool {
    for (names) |n| if (std.mem.eql(u8, n, name)) return true;
    return false;
}

/// Decide, per local var, whether/how to push it. Type precedence: --secret/--plain flag >
/// file metadata > current platform type > name/value inference (new vars only). A
/// platform secret is never downgraded to plain unless the flag says so explicitly.
/// Reserved names and TIMBAL_* / VITE_TIMBAL_* are never pushed.
pub fn planPush(
    allocator: std.mem.Allocator,
    vars: []const SyncVar,
    remote: []const RemoteVar,
    secret_names: []const []const u8,
    plain_names: []const []const u8,
) ![]PlanEntry {
    const plan = try allocator.alloc(PlanEntry, vars.len);
    errdefer allocator.free(plan);

    for (vars, 0..) |v, i| {
        const remote_type = remoteTypeFor(remote, v.name);
        const file_type = canonType(v.type) orelse "plain";

        if (isReservedVarName(v.name)) {
            plan[i] = .{ .action = .skip_reserved, .type = file_type, .source = .file, .remote_type = remote_type };
            continue;
        }
        // TIMBAL_* is the platform's namespace: those values are resolved by the platform before a
        // deploy/preview, so a pushed copy can only be stale or wrong. Never pushed, no override.
        if (v.managed or isPlatformManagedName(v.name)) {
            plan[i] = .{ .action = .skip_managed, .type = file_type, .source = .file, .remote_type = remote_type };
            continue;
        }

        var typ: []const u8 = undefined;
        var src: TypeSource = undefined;
        if (nameIn(secret_names, v.name)) {
            typ = "secret";
            src = .flag;
        } else if (nameIn(plain_names, v.name)) {
            typ = "plain";
            src = .flag;
        } else if (v.type_explicit) {
            typ = file_type;
            src = .file;
        } else if (remote_type) |rt| {
            typ = rt;
            src = .remote;
        } else {
            typ = if (inferSecret(v.name, v.value)) "secret" else "plain";
            src = .inferred;
        }

        const downgrade = remote_type != null and isSecretType(remote_type.?) and !isSecretType(typ) and src != .flag;
        plan[i] = .{
            .action = if (downgrade) .blocked_downgrade else .push,
            .type = typ,
            .source = src,
            .remote_type = remote_type,
        };
    }
    return plan;
}

pub fn buildPushPayload(allocator: std.mem.Allocator, rev: ?[]const u8, vars: []const SyncVar, plan: []const PlanEntry) ![]u8 {
    var buf = std.ArrayList(u8).init(allocator);
    errdefer buf.deinit();
    var w = buf.writer();

    try w.writeAll("{\"vars\":[");
    var written: usize = 0;
    for (vars, plan) |v, p| {
        if (p.action != .push) continue;
        if (written > 0) try w.writeAll(",");
        try w.writeAll("{\"name\":");
        try std.json.stringify(v.name, .{}, w);
        try w.writeAll(",\"type\":");
        try std.json.stringify(p.type, .{}, w);
        try w.writeAll(",\"value\":");
        try std.json.stringify(v.value, .{}, w);
        if (v.description) |d| {
            try w.writeAll(",\"description\":");
            try std.json.stringify(d, .{}, w);
        }
        try w.writeAll("}");
        written += 1;
    }
    try w.writeAll("]");
    if (rev) |r| {
        try w.writeAll(",\"rev\":");
        try std.json.stringify(r, .{}, w);
    }
    try w.writeAll("}");
    return buf.toOwnedSlice();
}

const PushResponse = struct {
    rev: []const u8,
    created: [][]const u8 = &.{},
    updated: [][]const u8 = &.{},
    skipped: [][]const u8 = &.{},
};

fn printNameList(stdout: anytype, label: []const u8, names: []const []const u8) !void {
    if (names.len == 0) return;
    try stdout.print("  {s}: ", .{label});
    for (names, 0..) |n, i| {
        if (i > 0) try stdout.writeAll(", ");
        try stdout.print("{s}", .{n});
    }
    try stdout.writeAll("\n");
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

const Action = enum { pull, push };

const Options = struct {
    action: Action,
    rev: ?[]const u8 = null, // explicit --rev
    use_default_rev: bool = false, // --default → omit rev
    file: []const u8 = ".env",
    force: bool = false, // overwrite existing local file on pull
    dry_run: bool = false, // print plan, write/send nothing
    include_platform: bool = false, // pull only: write the TIMBAL_* runtime vars active
    no_app_ids: bool = false, // skip workforce/<name>/.env TIMBAL_APP_ID sync
    secret_names: std.ArrayList([]const u8),
    plain_names: std.ArrayList([]const u8),
    base_url: ?[]u8 = null, // normalized --base-url override (owned)
    profile: ?[]const u8 = null,
    verbose: bool = false,
    quiet: bool = false,

    fn deinit(self: *Options, allocator: std.mem.Allocator) void {
        self.secret_names.deinit();
        self.plain_names.deinit();
        if (self.base_url) |b| allocator.free(b);
    }
};

/// Normalize/validate `--base-url` (https + Timbal API host). Caller owns returned slice.
fn normalizeBaseUrlOverride(allocator: std.mem.Allocator, raw: []const u8) ![]u8 {
    var url = std.mem.trim(u8, raw, " \t\r\n/");
    if (std.mem.startsWith(u8, url, "https://")) {
        url = url["https://".len..];
    } else if (std.mem.startsWith(u8, url, "http://")) {
        return error.InsecureBaseUrl;
    }
    // Strip any path the user pasted.
    if (std.mem.indexOfScalar(u8, url, '/')) |idx| {
        url = url[0..idx];
    }
    if (!isTimbalApiHost(url)) return error.InvalidBaseUrl;
    return std.fmt.allocPrint(allocator, "https://{s}", .{url});
}

fn appendNameList(list: *std.ArrayList([]const u8), raw: []const u8) !void {
    var it = std.mem.splitScalar(u8, raw, ',');
    while (it.next()) |piece| {
        const name = std.mem.trim(u8, piece, " \t");
        if (name.len == 0) continue;
        try list.append(name);
    }
}

fn parseArgs(allocator: std.mem.Allocator, args: []const []const u8) !Options {
    if (args.len == 0) {
        try printUsageWithError("Error: missing command (pull or push)");
        std.process.exit(2);
    }

    for (args) |a| {
        if (std.mem.eql(u8, a, "-h") or std.mem.eql(u8, a, "--help")) {
            try printUsage();
            std.process.exit(0);
        }
    }

    var opts: Options = .{
        .action = blk: {
            if (std.mem.eql(u8, args[0], "pull")) break :blk .pull;
            if (std.mem.eql(u8, args[0], "push")) break :blk .push;
            try printUsageWithError("Error: unknown env command (expected pull or push)");
            std.process.exit(2);
        },
        .secret_names = std.ArrayList([]const u8).init(allocator),
        .plain_names = std.ArrayList([]const u8).init(allocator),
    };
    errdefer opts.deinit(allocator);

    var raw_base_url: ?[]const u8 = null;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "-v") or std.mem.eql(u8, arg, "--verbose")) {
            opts.verbose = true;
        } else if (std.mem.eql(u8, arg, "-q") or std.mem.eql(u8, arg, "--quiet")) {
            opts.quiet = true;
        } else if (std.mem.eql(u8, arg, "--force")) {
            opts.force = true;
        } else if (std.mem.eql(u8, arg, "--dry-run")) {
            opts.dry_run = true;
        } else if (std.mem.eql(u8, arg, "--default")) {
            opts.use_default_rev = true;
        } else if (std.mem.eql(u8, arg, "--include-platform-vars")) {
            opts.include_platform = true;
        } else if (std.mem.eql(u8, arg, "--no-app-ids")) {
            opts.no_app_ids = true;
        } else if (std.mem.eql(u8, arg, "--secret")) {
            i += 1;
            if (i >= args.len) {
                try printUsageWithError("Error: --secret requires a var name");
                std.process.exit(2);
            }
            try appendNameList(&opts.secret_names, args[i]);
        } else if (std.mem.eql(u8, arg, "--plain")) {
            i += 1;
            if (i >= args.len) {
                try printUsageWithError("Error: --plain requires a var name");
                std.process.exit(2);
            }
            try appendNameList(&opts.plain_names, args[i]);
        } else if (std.mem.eql(u8, arg, "--rev")) {
            i += 1;
            if (i >= args.len) {
                try printUsageWithError("Error: --rev requires a branch name");
                std.process.exit(2);
            }
            opts.rev = args[i];
        } else if (std.mem.eql(u8, arg, "-f") or std.mem.eql(u8, arg, "--file")) {
            i += 1;
            if (i >= args.len) {
                try printUsageWithError("Error: --file requires a path");
                std.process.exit(2);
            }
            opts.file = args[i];
        } else if (std.mem.eql(u8, arg, "--base-url")) {
            i += 1;
            if (i >= args.len) {
                try printUsageWithError("Error: --base-url requires a URL");
                std.process.exit(2);
            }
            raw_base_url = args[i];
        } else if (std.mem.eql(u8, arg, "--profile")) {
            i += 1;
            if (i >= args.len) {
                try printUsageWithError("Error: --profile requires a name argument");
                std.process.exit(2);
            }
            opts.profile = args[i];
        } else {
            try printUsageWithError("Error: unknown option");
            std.process.exit(2);
        }
    }

    if (opts.use_default_rev and opts.rev != null) {
        try printUsageWithError("Error: --rev and --default are mutually exclusive");
        std.process.exit(2);
    }
    if (opts.force and opts.action != .pull) {
        try printUsageWithError("Error: --force is only valid with `timbal env pull`");
        std.process.exit(2);
    }
    if ((opts.secret_names.items.len > 0 or opts.plain_names.items.len > 0) and opts.action != .push) {
        try printUsageWithError("Error: --secret / --plain are only valid with `timbal env push`");
        std.process.exit(2);
    }
    if (opts.include_platform and opts.action != .pull) {
        try printUsageWithError("Error: --include-platform-vars is only valid with `timbal env pull` (TIMBAL_* vars are never pushed)");
        std.process.exit(2);
    }
    for (opts.secret_names.items) |s| {
        if (nameIn(opts.plain_names.items, s)) {
            try printUsageWithError("Error: the same var cannot be both --secret and --plain");
            std.process.exit(2);
        }
    }
    if (raw_base_url) |raw| {
        opts.base_url = normalizeBaseUrlOverride(allocator, raw) catch |err| {
            switch (err) {
                error.InsecureBaseUrl => try printUsageWithError("Error: --base-url must be https."),
                error.InvalidBaseUrl => try printUsageWithError(
                    "Error: --base-url must be https://api.timbal.ai or https://api.<env>.timbal.ai",
                ),
                else => return err,
            }
            std.process.exit(2);
        };
    }

    return opts;
}

pub fn run(allocator: std.mem.Allocator, args: []const []const u8) !void {
    var opts = try parseArgs(allocator, args);
    defer opts.deinit(allocator);
    const stderr = std.io.getStdErr().writer();
    const stdout = std.io.getStdOut().writer();

    // Profile: --profile > TIMBAL_PROFILE > default
    const env_profile = std.process.getEnvVarOwned(allocator, "TIMBAL_PROFILE") catch |err| blk: {
        if (err == error.EnvironmentVariableNotFound) break :blk null;
        return err;
    };
    defer if (env_profile) |p| allocator.free(p);
    const profile: []const u8 = opts.profile orelse (env_profile orelse "default");

    // API key
    const credentials_path = try getCredentialsPath(allocator);
    defer allocator.free(credentials_path);
    const credentials_content = fs.cwd().readFileAlloc(allocator, credentials_path, 1024 * 1024) catch |err| {
        if (err == error.FileNotFound) {
            try stderr.print("Error: Timbal is not configured. Run '{s}timbal configure{s}' first.\n", .{ Color.bold_cyan, Color.reset });
            std.process.exit(1);
        }
        return err;
    };
    defer allocator.free(credentials_content);
    const api_key = readValue(credentials_content, profile, "api_key") orelse {
        try stderr.print("Error: No API key found for profile '{s}'. Run '{s}timbal configure --profile {s}{s}'.\n", .{ profile, Color.bold_cyan, profile, Color.reset });
        std.process.exit(1);
    };

    // Repo root + .git/config remote
    const cwd_path = try fs.cwd().realpathAlloc(allocator, ".");
    defer allocator.free(cwd_path);

    const repo_root = (try findGitDir(allocator, cwd_path)) orelse {
        try stderr.writeAll("Error: not inside a git repository.\n");
        std.process.exit(1);
    };
    defer allocator.free(repo_root);

    const config_path = resolveGitConfigPath(allocator, repo_root) catch {
        try stderr.writeAll("Error: could not locate .git/config.\n");
        std.process.exit(1);
    };
    defer allocator.free(config_path);

    const config_content = fs.cwd().readFileAlloc(allocator, config_path, 1024 * 1024) catch |err| {
        try stderr.print("Error: failed to read {s}: {}\n", .{ config_path, err });
        std.process.exit(1);
    };
    defer allocator.free(config_content);

    var remote = (try resolveTimbalRemoteFromConfig(allocator, config_content)) orelse {
        try stderr.writeAll(
            \\Error: no Timbal git remote found in .git/config.
            \\Expected a remote URL like:
            \\  https://api.dev.timbal.ai/orgs/{org_id}/projects/{project_id}/git
            \\
        );
        std.process.exit(1);
    };
    defer remote.deinit(allocator);

    if (opts.base_url) |overridden| {
        allocator.free(remote.base_url);
        remote.base_url = try allocator.dupe(u8, overridden);
    }

    // Resolve rev
    var rev_owned: ?[]u8 = null;
    defer if (rev_owned) |r| allocator.free(r);
    const rev: ?[]const u8 = blk: {
        if (opts.use_default_rev) break :blk null;
        if (opts.rev) |r| break :blk r;
        rev_owned = try currentGitBranch(allocator);
        if (rev_owned) |b| break :blk b;
        try stderr.writeAll(
            \\Error: could not determine current git branch (detached HEAD?).
            \\Pass --rev <branch> or --default.
            \\
        );
        std.process.exit(1);
    };

    // The .env must live where `timbal start` loads it from: the project root.
    const project_root = try resolveProjectRoot(allocator, cwd_path, repo_root);
    defer allocator.free(project_root);

    if (opts.verbose or opts.dry_run) {
        try stderr.print("remote: {s} → org={s} project={s} base={s}\n", .{
            remote.remote_name,
            remote.org_id,
            remote.project_id,
            remote.base_url,
        });
        if (rev) |r| {
            try stderr.print("rev: {s}\n", .{r});
        } else {
            try stderr.writeAll("rev: (project default)\n");
        }
        try stderr.print("project root: {s}\n", .{project_root});
    }

    if (std.mem.eql(u8, project_root, repo_root) and !hasTimbalLayout(allocator, project_root)) {
        try stderr.print(
            "{s}Warning:{s} no Timbal project layout (workforce/, ui/, api/) found under {s}; using it as the project root anyway.\n" ++
                "Run this from inside the project so the .env lands where `timbal start` loads it.\n",
            .{ Color.bold_yellow, Color.reset, project_root },
        );
    }

    const file_path = if (fs.path.isAbsolute(opts.file))
        try allocator.dupe(u8, opts.file)
    else
        try std.fmt.allocPrint(allocator, "{s}{s}{s}", .{ project_root, sep, opts.file });
    defer allocator.free(file_path);

    try warnIfScopedFile(allocator, project_root, file_path, stderr);
    try warnIfNotGitignored(allocator, repo_root, file_path, stderr);

    switch (opts.action) {
        .pull => try runPull(allocator, opts, remote, rev, api_key, file_path, project_root, repo_root, stdout, stderr),
        .push => try runPush(allocator, opts, remote, rev, api_key, file_path, project_root, repo_root, stdout, stderr),
    }
}

fn hasTimbalLayout(allocator: std.mem.Allocator, root: []const u8) bool {
    for ([_][]const u8{ "workforce", "ui", "api" }) |d| {
        const p = std.fmt.allocPrint(allocator, "{s}{s}{s}", .{ root, sep, d }) catch return true;
        defer allocator.free(p);
        if (fs.cwd().openDir(p, .{})) |dir| {
            var dd = dir;
            dd.close();
            return true;
        } else |_| {}
    }
    return false;
}

/// Which `timbal start` service loads a file under the project root, or null for the shared root.
pub fn scopedFileOwner(project_root: []const u8, file_path: []const u8) ?[]const u8 {
    if (!std.mem.startsWith(u8, file_path, project_root)) return null;
    var rel = file_path[project_root.len..];
    if (rel.len == 0 or (rel[0] != '/' and rel[0] != '\\')) return null;
    rel = rel[1..];
    const first_sep = std.mem.indexOfAny(u8, rel, "/\\") orelse return null;
    const top = rel[0..first_sep];
    if (std.mem.eql(u8, top, "ui") or std.mem.eql(u8, top, "api")) return top;
    if (std.mem.eql(u8, top, "workforce")) {
        const rest = rel[first_sep + 1 ..];
        const next = std.mem.indexOfAny(u8, rest, "/\\") orelse return null;
        if (next == 0) return null;
        return rest[0..next];
    }
    return null;
}

/// A -f path inside ui/, api/ or workforce/<name>/ is only ever seen by that one service.
fn warnIfScopedFile(allocator: std.mem.Allocator, project_root: []const u8, file_path: []const u8, stderr: anytype) !void {
    _ = allocator;
    const owner = scopedFileOwner(project_root, file_path) orelse return;
    if (std.mem.eql(u8, owner, "ui") or std.mem.eql(u8, owner, "api")) {
        try stderr.print(
            "{s}Note:{s} {s} lives under {s}/ — `timbal start` does not auto-load it; only the {s} toolchain reads it.\n" ++
                "Project-wide vars belong in <project>/.env.\n",
            .{ Color.bold_yellow, Color.reset, file_path, owner, owner },
        );
    } else {
        try stderr.print(
            "{s}Note:{s} {s} is scoped to workforce member '{s}' — `timbal start` loads it into that member only.\n" ++
                "Project-wide vars belong in <project>/.env.\n",
            .{ Color.bold_yellow, Color.reset, file_path, owner },
        );
    }
}

/// Warn when the local env file is not ignored by git (secrets dump risk).
fn warnIfNotGitignored(
    allocator: std.mem.Allocator,
    repo_root: []const u8,
    file_path: []const u8,
    stderr: anytype,
) !void {
    // Prefer git's own ignore rules (covers root + nested .gitignore / excludes).
    if (try gitCheckIgnore(allocator, repo_root, file_path)) |ignored| {
        if (!ignored) {
            try stderr.print(
                "{s}Warning:{s} {s} is not gitignored. Pulled secrets could be committed.\n" ++
                    "Add this path (or a matching pattern) to .gitignore.\n",
                .{ Color.bold_yellow, Color.reset, file_path },
            );
        }
        return;
    }

    // Fallback: scan repo-root .gitignore for a pattern that covers *this* file
    // (a bare `.env` entry must not silence warnings for `-f secrets.env`).
    const gi_path = try std.fmt.allocPrint(allocator, "{s}{s}.gitignore", .{ repo_root, sep });
    defer allocator.free(gi_path);
    const gi = fs.cwd().readFileAlloc(allocator, gi_path, 1024 * 1024) catch {
        try stderr.print(
            "{s}Warning:{s} no .gitignore found; ensure {s} is not committed (contains secrets).\n",
            .{ Color.bold_yellow, Color.reset, file_path },
        );
        return;
    };
    defer allocator.free(gi);

    const base = fs.path.basename(file_path);
    if (!gitignoreCoversBasename(gi, base)) {
        try stderr.print(
            "{s}Warning:{s} {s} does not appear gitignored; secrets may be committed.\n" ++
                "Add `{s}` (or a matching pattern) to .gitignore.\n",
            .{ Color.bold_yellow, Color.reset, file_path, base },
        );
    }
}

/// Returns true/false when `git check-ignore` works; null if git unavailable/errors.
fn gitCheckIgnore(allocator: std.mem.Allocator, repo_root: []const u8, file_path: []const u8) !?bool {
    var child = std.process.Child.init(&.{ "git", "-C", repo_root, "check-ignore", "-q", "--", file_path }, allocator);
    child.stdout_behavior = .Ignore;
    child.stderr_behavior = .Ignore;
    child.spawn() catch return null;
    const term = child.wait() catch return null;
    return switch (term) {
        .Exited => |code| switch (code) {
            0 => true, // ignored
            1 => false, // not ignored
            else => null,
        },
        else => null,
    };
}

/// Best-effort match of common .gitignore patterns against a basename.
/// Only used when `git check-ignore` is unavailable.
fn gitignorePatternMatches(pattern_raw: []const u8, basename: []const u8) bool {
    var pattern = std.mem.trim(u8, pattern_raw, " \t");
    if (pattern.len == 0 or pattern[0] == '!') return false;
    // Drop directory-only marker and leading path noise we care about for basenames.
    if (std.mem.endsWith(u8, pattern, "/")) pattern = pattern[0 .. pattern.len - 1];
    if (std.mem.startsWith(u8, pattern, "./")) pattern = pattern[2..];
    if (std.mem.startsWith(u8, pattern, "/")) pattern = pattern[1..];
    if (std.mem.startsWith(u8, pattern, "**/")) pattern = pattern[3..];
    // If a remaining slash is present, only the final segment can match a basename.
    if (std.mem.lastIndexOfScalar(u8, pattern, '/')) |idx| {
        pattern = pattern[idx + 1 ..];
    }
    if (pattern.len == 0) return false;

    if (std.mem.eql(u8, pattern, basename)) return true;
    if (std.mem.eql(u8, pattern, "*")) return true;

    // prefix*
    if (std.mem.endsWith(u8, pattern, "*") and !std.mem.startsWith(u8, pattern, "*")) {
        const prefix = pattern[0 .. pattern.len - 1];
        if (std.mem.indexOfScalar(u8, prefix, '*') != null) return false;
        return std.mem.startsWith(u8, basename, prefix);
    }
    // *suffix
    if (std.mem.startsWith(u8, pattern, "*") and !std.mem.endsWith(u8, pattern, "*")) {
        const suffix = pattern[1..];
        if (std.mem.indexOfScalar(u8, suffix, '*') != null) return false;
        return std.mem.endsWith(u8, basename, suffix);
    }
    return false;
}

fn gitignoreCoversBasename(content: []const u8, basename: []const u8) bool {
    var lines = std.mem.splitScalar(u8, content, '\n');
    while (lines.next()) |raw| {
        const line = std.mem.trim(u8, raw, " \t\r");
        if (line.len == 0 or line[0] == '#') continue;
        if (gitignorePatternMatches(line, basename)) return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Placement audit: what is *already* sitting in the files `timbal start` loads
// ---------------------------------------------------------------------------

/// Active locally, these make the SDK treat the process as deployed: service calls go to the
/// deployed env gateway instead of localhost (see python/timbal/platform/utils.py).
const reroute_vars = [_][]const u8{ "TIMBAL_PROJECT_ENV_ID", "TIMBAL_PROJECT_ENV_ORIGIN", "TIMBAL_DEPLOYMENTS_DOMAIN" };
/// `timbal start` sets these in its hard runtime layer; a .env copy is dead weight.
const runtime_wired_vars = [_][]const u8{ "TIMBAL_START_WORKFORCE", "TIMBAL_WORKFORCE", "TIMBAL_START_API_PORT", "TIMBAL_START_UI_PORT", "PORT" };

pub const PlacementKind = enum { reroute, app_id_scope, dead };
pub const PlacementFinding = struct { kind: PlacementKind, key: []const u8 };

/// Scan one auto-loaded .env's *active* assignments. `member_scope` is true for
/// workforce/<name>/.env, where TIMBAL_APP_ID is exactly where it belongs.
pub fn auditEnvContent(content: []const u8, member_scope: bool, out: *std.ArrayList(PlacementFinding)) !void {
    var lines = std.mem.splitScalar(u8, content, '\n');
    while (lines.next()) |raw| {
        const a = parseEnvAssignment(raw) orelse continue;
        if (tokenIn(a.key, &reroute_vars)) {
            try out.append(.{ .kind = .reroute, .key = a.key });
        } else if (!member_scope and std.mem.eql(u8, a.key, "TIMBAL_APP_ID")) {
            try out.append(.{ .kind = .app_id_scope, .key = a.key });
        } else if (tokenIn(a.key, &runtime_wired_vars)) {
            try out.append(.{ .kind = .dead, .key = a.key });
        }
    }
}

fn auditOneEnvFile(
    allocator: std.mem.Allocator,
    path: []const u8,
    label: []const u8,
    member_scope: bool,
    printed_header: *bool,
    stderr: anytype,
) !void {
    const content = fs.cwd().readFileAlloc(allocator, path, 1024 * 1024) catch return;
    defer allocator.free(content);

    var findings = std.ArrayList(PlacementFinding).init(allocator);
    defer findings.deinit();
    try auditEnvContent(content, member_scope, &findings);

    for (findings.items) |f| {
        if (!printed_header.*) {
            try stderr.print("{s}Placement check{s} (files `timbal start` auto-loads):\n", .{ Color.bold_yellow, Color.reset });
            printed_header.* = true;
        }
        switch (f.kind) {
            .reroute => try stderr.print(
                "  ! {s}: {s} is active — `timbal start` will send service calls to the deployed gateway instead of\n" ++
                    "    localhost. Comment it out (or re-run `timbal env pull --force`).\n",
                .{ label, f.key },
            ),
            .app_id_scope => try stderr.print(
                "  ! {s}: TIMBAL_APP_ID here is loaded into every service. It belongs in workforce/<name>/.env\n" ++
                    "    (`timbal env pull` writes it there); remove it from this file.\n",
                .{label},
            ),
            .dead => try stderr.print(
                "  · {s}: {s} is overridden by `timbal start`'s runtime wiring; it has no effect locally.\n",
                .{ label, f.key },
            ),
        }
    }
}

/// Warn about hazardous active vars in `<project>/.env` and every `workforce/<name>/.env`.
/// Read-only; never edits anything. Runs on both pull and push so users who pulled with an
/// older CLI (which wrote TIMBAL_PROJECT_ENV_ID active) find out why local routing is off.
fn auditEnvPlacement(allocator: std.mem.Allocator, project_root: []const u8, stderr: anytype) !void {
    var printed_header = false;

    const root_path = try std.fmt.allocPrint(allocator, "{s}{s}.env", .{ project_root, sep });
    defer allocator.free(root_path);
    try auditOneEnvFile(allocator, root_path, ".env", false, &printed_header, stderr);

    const wf_path = try std.fmt.allocPrint(allocator, "{s}{s}workforce", .{ project_root, sep });
    defer allocator.free(wf_path);
    var wf_dir = fs.cwd().openDir(wf_path, .{ .iterate = true }) catch return;
    defer wf_dir.close();
    var iter = wf_dir.iterate();
    while (try iter.next()) |entry| {
        if (entry.kind != .directory) continue;
        if (entry.name.len == 0 or entry.name[0] == '.') continue;
        const path = try std.fmt.allocPrint(allocator, "{s}{s}{s}{s}.env", .{ wf_path, sep, entry.name, sep });
        defer allocator.free(path);
        const label = try std.fmt.allocPrint(allocator, "workforce/{s}/.env", .{entry.name});
        defer allocator.free(label);
        try auditOneEnvFile(allocator, path, label, true, &printed_header, stderr);
    }
}

fn describeVarState(v: SyncVar, active_managed: bool) []const u8 {
    if (v.value_missing) return "placeholder (value not returned by platform)";
    if (isMultiline(v.value)) return "commented out (multi-line value; not representable in .env)";
    if (isReservedVarName(v.name)) return "commented out (reserved)";
    if (v.managed) return if (active_managed) "active (platform-managed, --include-platform-vars)" else "commented out (platform-managed)";
    return "active";
}

fn runPull(
    allocator: std.mem.Allocator,
    opts: Options,
    remote: TimbalRemote,
    rev: ?[]const u8,
    api_key: []const u8,
    file_path: []const u8,
    project_root: []const u8,
    repo_root: []const u8,
    stdout: anytype,
    stderr: anytype,
) !void {
    // With --force the root file is about to be rewritten, so audit afterwards; otherwise audit
    // what is on disk now — a refused pull should still tell the user what is wrong locally.
    const overwriting = opts.force and !opts.dry_run;
    if (!overwriting) try auditEnvPlacement(allocator, project_root, stderr);

    // Refuse to clobber an existing local file unless --force (dry-run touches nothing).
    if (!opts.force and !opts.dry_run) {
        if (fs.cwd().access(file_path, .{})) |_| {
            try stderr.print(
                "Error: {s} already exists. Re-run with --force to overwrite.\n",
                .{file_path},
            );
            std.process.exit(1);
        } else |_| {}
    }

    // 1. Effective env for the rev (what a deployment would see).
    const pull_suffix = if (rev) |r| blk: {
        const encoded = try urlEncodeQuery(allocator, r);
        defer allocator.free(encoded);
        break :blk try std.fmt.allocPrint(allocator, "/vars/pull?rev={s}", .{encoded});
    } else try allocator.dupe(u8, "/vars/pull");
    defer allocator.free(pull_suffix);
    const url = try projectUrl(allocator, remote, pull_suffix);
    defer allocator.free(url);

    const body = try apiRequest(allocator, .GET, url, api_key, null, opts.verbose, rev);
    defer allocator.free(body);

    var pulled = parsePullResponse(allocator, body) catch |err| {
        try stderr.print("Error: failed to parse pull response: {}\n", .{err});
        if (opts.verbose) try stderr.print("{s}\n", .{body});
        std.process.exit(1);
    };
    defer pulled.deinit(allocator);

    const effective_rev: []const u8 = if (pulled.rev.len > 0) pulled.rev else (rev orelse "");

    // 2. User-defined project vars: the source of truth for types and for which names are
    //    platform-computed. Soft failure — the pulled file is still valid without it.
    var listing = try fetchRemoteVarListing(allocator, remote, api_key, opts.verbose);
    defer listing.deinit(allocator);
    if (!listing.ok) {
        try stderr.print(
            "{s}Warning:{s} could not list project vars (HTTP {d}). Platform-managed detection falls back to the\n" ++
                "TIMBAL_* prefix and secrets absent from the effective env cannot be recovered.\n",
            .{ Color.bold_yellow, Color.reset, listing.status },
        );
    }

    // 3. Reconcile: mark managed (anything the platform computes rather than the user defines),
    //    never let a platform secret be written as plain.
    var upgraded: usize = 0;
    for (pulled.vars.items) |*v| {
        if (listing.ok) {
            const rt = remoteTypeFor(listing.vars.items, v.name);
            if (rt == null) {
                v.managed = true;
            } else if (isSecretType(rt.?) and !isSecretType(v.type)) {
                allocator.free(v.type);
                v.type = try allocator.dupe(u8, "secret");
                upgraded += 1;
            }
        } else if (!v.managed) {
            v.managed = isPlatformManagedName(v.name);
        }
    }

    // 4. Secrets the effective env left out: fetch individually (scoped to this rev's env).
    var recovered: usize = 0;
    var placeholders: usize = 0;
    var other_env: usize = 0;
    if (listing.ok) {
        var env_id: ?[]u8 = null;
        defer if (env_id) |e| allocator.free(e);
        var env_resolved = false;
        for (listing.vars.items) |rv| {
            if (!isSecretType(rv.type)) continue;
            if (isReservedVarName(rv.name)) continue;
            if (findSyncVar(pulled.vars.items, rv.name) != null) continue;

            if (!rv.applies_to_all_envs) {
                if (!env_resolved) {
                    env_resolved = true;
                    if (effective_rev.len > 0) env_id = try fetchEnvIdForBranch(allocator, remote, api_key, effective_rev, opts.verbose);
                }
                const eid = env_id orelse {
                    other_env += 1;
                    continue;
                };
                if (!rv.hasEnv(eid)) {
                    other_env += 1;
                    continue;
                }
            }

            const decrypted = try fetchVarDecrypted(allocator, remote, api_key, rv.id, opts.verbose);
            errdefer if (decrypted) |d| allocator.free(d);
            var sv = SyncVar{
                .name = try allocator.dupe(u8, rv.name),
                .type = undefined,
                .value = undefined,
                .type_explicit = true,
                .value_missing = decrypted == null,
            };
            errdefer allocator.free(sv.name);
            sv.type = try allocator.dupe(u8, "secret");
            errdefer allocator.free(sv.type);
            sv.value = decrypted orelse try allocator.dupe(u8, "");
            errdefer if (decrypted == null) allocator.free(sv.value);
            if (rv.description) |d| sv.description = try allocator.dupe(u8, d);
            errdefer if (sv.description) |d| allocator.free(d);
            try pulled.vars.append(sv);
            if (sv.value_missing) placeholders += 1 else recovered += 1;
        }
    }

    // A project-level TIMBAL_APP_ID applies to every workforce member under `timbal start`.
    if (findSyncVar(pulled.vars.items, "TIMBAL_APP_ID")) |idx| {
        if (!pulled.vars.items[idx].managed) {
            try stderr.print(
                "{s}Warning:{s} the platform has a project-level TIMBAL_APP_ID. It is written commented out — TIMBAL_APP_ID\n" ++
                    "is per workforce member and is synced into workforce/<name>/.env instead.\n",
                .{ Color.bold_yellow, Color.reset },
            );
        }
    }

    const content = try formatEnvFile(allocator, effective_rev, pulled.vars.items, .{ .managed_active = opts.include_platform });
    defer allocator.free(content);

    var active: usize = 0;
    var managed: usize = 0;
    var multiline: usize = 0;
    for (pulled.vars.items) |v| {
        if (v.managed) {
            managed += 1;
        } else if (isMultiline(v.value)) {
            multiline += 1;
        } else if (!v.value_missing and !isReservedVarName(v.name)) {
            active += 1;
        }
    }

    if (opts.dry_run) {
        try stdout.print("Dry run — would write {s}\n", .{file_path});
        try stdout.print("rev: {s}\nvars ({d}):\n", .{ effective_rev, pulled.vars.items.len });
        for (pulled.vars.items) |v| {
            try stdout.print("  {s} ({s})  {s}\n", .{ v.name, v.type, describeVarState(v, opts.include_platform) });
        }
        if (other_env > 0) try stdout.print("  ({d} secret(s) scoped to other environments not included)\n", .{other_env});
    } else {
        try persistPulledEnvFile(allocator, file_path, content, opts.force, stderr);
        if (!opts.quiet) {
            try stdout.print(
                "{s}✓{s} Pulled {d} var(s) for rev {s}{s}{s} → {s}\n",
                .{ Color.bold_green, Color.reset, active, Color.bold_cyan, effective_rev, Color.reset, file_path },
            );
            if (managed > 0) {
                if (opts.include_platform) {
                    try stdout.print("  {d} platform-managed var(s) written active (--include-platform-vars)\n", .{managed});
                } else {
                    try stdout.print("  {d} platform-managed var(s) kept commented out (TIMBAL_*/VITE_TIMBAL_* runtime wiring)\n", .{managed});
                }
            }
            if (upgraded > 0) try stdout.print("  {d} var(s) re-tagged secret to match the platform\n", .{upgraded});
            if (recovered > 0) try stdout.print("  {d} secret(s) fetched individually\n", .{recovered});
            if (placeholders > 0) try stdout.print("  {d} secret(s) written as commented placeholders (value not returned by the platform)\n", .{placeholders});
            if (multiline > 0) try stdout.print("  {d} multi-line value(s) kept commented out (not representable in .env; set them via shell or `timbal start --env`)\n", .{multiline});
            if (other_env > 0) try stdout.print("  {d} secret(s) scoped to other environments not included\n", .{other_env});
        }
        if (overwriting) try auditEnvPlacement(allocator, project_root, stderr);
    }

    if (!opts.no_app_ids) {
        if (effective_rev.len > 0) {
            try syncAppIds(allocator, opts, remote, api_key, effective_rev, project_root, repo_root, stdout, stderr);
        } else if (opts.verbose) {
            try stderr.writeAll("note: pull response carried no rev; TIMBAL_APP_ID not synced.\n");
        }
    } else if (opts.dry_run) {
        try stdout.writeAll("No changes made.\n");
    }
}

/// Persist pulled env contents safely:
/// - without --force: O_EXCL create so a file that appears after the pre-check cannot be truncated
/// - with --force: write a temp file fully, then rename over the destination so a failed write
///   cannot destroy the previous local file
fn persistPulledEnvFile(
    allocator: std.mem.Allocator,
    file_path: []const u8,
    content: []const u8,
    force: bool,
    stderr: anytype,
) !void {
    if (!force) {
        try writeFileExclusive(file_path, content, stderr);
        return;
    }
    try writeFileAtomic(allocator, file_path, content, stderr);
}

fn ensureParentDir(file_path: []const u8, stderr: anytype) !void {
    if (fs.path.dirname(file_path)) |dir| {
        fs.cwd().makePath(dir) catch |err| {
            if (err != error.PathAlreadyExists) {
                try stderr.print("Error: could not create directory for {s}: {}\n", .{ file_path, err });
                std.process.exit(1);
            }
        };
    }
}

/// Exclusive create: a file created concurrently cannot be truncated.
fn writeFileExclusive(file_path: []const u8, content: []const u8, stderr: anytype) !void {
    try ensureParentDir(file_path, stderr);
    const file = fs.cwd().createFile(file_path, .{ .exclusive = true }) catch |err| {
        if (err == error.PathAlreadyExists) {
            try stderr.print(
                "Error: {s} already exists. Re-run with --force to overwrite.\n",
                .{file_path},
            );
            std.process.exit(1);
        }
        try stderr.print("Error: could not write {s}: {}\n", .{ file_path, err });
        std.process.exit(1);
    };
    defer file.close();
    file.writeAll(content) catch |err| {
        // Best-effort cleanup of a partial new file; there was no prior content to preserve.
        fs.cwd().deleteFile(file_path) catch {};
        try stderr.print("Error: could not write {s}: {}\n", .{ file_path, err });
        std.process.exit(1);
    };
    file.setEndPos(content.len) catch {};
}

/// Temp file + rename: the destination is never deleted until the new contents are on disk.
fn writeFileAtomic(allocator: std.mem.Allocator, file_path: []const u8, content: []const u8, stderr: anytype) !void {
    try ensureParentDir(file_path, stderr);
    const tmp_path = try std.fmt.allocPrint(allocator, "{s}.timbal-pull.tmp", .{file_path});
    defer allocator.free(tmp_path);

    {
        const tmp = fs.cwd().createFile(tmp_path, .{ .truncate = true }) catch |err| {
            try stderr.print("Error: could not write temp file {s}: {}\n", .{ tmp_path, err });
            std.process.exit(1);
        };
        var tmp_ok = false;
        defer {
            tmp.close();
            if (!tmp_ok) fs.cwd().deleteFile(tmp_path) catch {};
        }
        tmp.writeAll(content) catch |err| {
            try stderr.print("Error: could not write temp file {s}: {}\n", .{ tmp_path, err });
            std.process.exit(1);
        };
        tmp.setEndPos(content.len) catch {};
        tmp_ok = true;
    }

    fs.cwd().rename(tmp_path, file_path) catch |err| {
        // POSIX rename replaces atomically. Windows often cannot rename onto an existing path —
        // only then delete the destination after the temp write succeeded.
        if (err == error.PathAlreadyExists) {
            fs.cwd().deleteFile(file_path) catch |del_err| {
                fs.cwd().deleteFile(tmp_path) catch {};
                try stderr.print("Error: could not replace {s}: {}\n", .{ file_path, del_err });
                std.process.exit(1);
            };
            fs.cwd().rename(tmp_path, file_path) catch |ren_err| {
                // Destination is gone; leave the temp file so the pulled content is recoverable.
                try stderr.print(
                    "Error: failed to move pulled vars into {s}: {}\nPulled content left at {s}\n",
                    .{ file_path, ren_err, tmp_path },
                );
                std.process.exit(1);
            };
            return;
        }
        fs.cwd().deleteFile(tmp_path) catch {};
        try stderr.print("Error: could not replace {s}: {}\n", .{ file_path, err });
        std.process.exit(1);
    };
}

fn sourceLabel(src: TypeSource) []const u8 {
    return switch (src) {
        .flag => "--secret/--plain",
        .file => "file metadata",
        .remote => "platform type",
        .inferred => "inferred from name/value",
    };
}

fn runPush(
    allocator: std.mem.Allocator,
    opts: Options,
    remote: TimbalRemote,
    rev: ?[]const u8,
    api_key: []const u8,
    file_path: []const u8,
    project_root: []const u8,
    repo_root: []const u8,
    stdout: anytype,
    stderr: anytype,
) !void {
    const content = fs.cwd().readFileAlloc(allocator, file_path, 16 * 1024 * 1024) catch |err| {
        if (err == error.FileNotFound) {
            try stderr.print("Error: local env file not found: {s}\nRun `timbal env pull` first, or create the file.\n", .{file_path});
            std.process.exit(1);
        }
        return err;
    };
    defer allocator.free(content);

    var sync_vars = try parseEnvFile(allocator, content);
    defer freeSyncVars(allocator, &sync_vars);

    if (sync_vars.items.len == 0) {
        try stderr.print("Error: no variables found in {s}\n", .{file_path});
        std.process.exit(1);
    }

    for (sync_vars.items) |v| {
        if (canonType(v.type) == null) {
            try stderr.print("Error: var '{s}' has invalid type '{s}' (expected plain|secret)\n", .{ v.name, v.type });
            std.process.exit(1);
        }
        if (!isValidEnvKey(v.name)) {
            try stderr.print(
                "{s}Warning:{s} '{s}' is not a valid env name for `timbal start` (letters, digits, underscore; no leading digit).\n" ++
                    "It will be pushed but never loaded locally.\n",
                .{ Color.bold_yellow, Color.reset, v.name },
            );
        }
    }

    try auditEnvPlacement(allocator, project_root, stderr);
    for (opts.secret_names.items) |n| {
        if (findSyncVar(sync_vars.items, n) == null) try stderr.print("{s}Warning:{s} --secret {s}: not present in {s}\n", .{ Color.bold_yellow, Color.reset, n, file_path });
    }
    for (opts.plain_names.items) |n| {
        if (findSyncVar(sync_vars.items, n) == null) try stderr.print("{s}Warning:{s} --plain {s}: not present in {s}\n", .{ Color.bold_yellow, Color.reset, n, file_path });
    }

    // Current platform types are required: without them we cannot tell a new var from an
    // existing secret, and pushing blind is exactly how secrets get downgraded to plain.
    var listing = try fetchRemoteVarListing(allocator, remote, api_key, opts.verbose);
    defer listing.deinit(allocator);
    if (!listing.ok) {
        if (listing.error_body) |b| {
            printApiError(listing.status, b, rev);
        } else {
            try stderr.print("Error: unexpected response listing project vars (HTTP {d}).\n", .{listing.status});
        }
        try stderr.writeAll("Push needs the current platform var types to classify secrets safely; aborting.\n");
        std.process.exit(1);
    }

    const plan = try planPush(allocator, sync_vars.items, listing.vars.items, opts.secret_names.items, opts.plain_names.items);
    defer allocator.free(plan);

    var pushable: usize = 0;
    var blocked: usize = 0;
    var type_changes: usize = 0;
    var name_width: usize = 0;
    for (sync_vars.items, plan) |v, p| {
        if (p.action == .push) pushable += 1;
        if (p.action == .blocked_downgrade) blocked += 1;
        if (p.action == .push and p.remote_type != null and !std.mem.eql(u8, p.remote_type.?, p.type)) type_changes += 1;
        if (v.name.len > name_width) name_width = v.name.len;
    }

    const url = try projectUrl(allocator, remote, "/vars/push");
    defer allocator.free(url);

    if (!opts.quiet or blocked > 0) {
        if (opts.dry_run) {
            try stdout.print("Dry run — would POST {s}\n", .{url});
        }
        if (rev) |r| {
            try stdout.print("rev: {s}\n", .{r});
        } else {
            try stdout.writeAll("rev: (project default)\n");
        }
        try stdout.print("file: {s}\nvars ({d}, values redacted):\n", .{ file_path, sync_vars.items.len });
        for (sync_vars.items, plan) |v, p| {
            try stdout.print("  {s}", .{v.name});
            var pad = name_width - v.name.len + 2;
            while (pad > 0) : (pad -= 1) try stdout.writeAll(" ");
            switch (p.action) {
                .push => {
                    if (p.remote_type) |rt| {
                        if (std.mem.eql(u8, rt, p.type)) {
                            try stdout.print("{s:<7} {s} — value update\n", .{ p.type, sourceLabel(p.source) });
                        } else {
                            // The platform keeps a var's stored type on update; only the value changes.
                            try stdout.print(
                                "{s:<7} {s} requested {s}; the platform keeps the stored type on update — change it in the Timbal UI\n",
                                .{ rt, sourceLabel(p.source), p.type },
                            );
                        }
                    } else {
                        try stdout.print("{s:<7} {s} — new on the platform\n", .{ p.type, sourceLabel(p.source) });
                    }
                },
                .skip_reserved => {
                    if (std.mem.eql(u8, v.name, "TIMBAL_APP_ID")) {
                        try stdout.writeAll("skipped  reserved — per workforce member; belongs in workforce/<name>/.env, not the project root\n");
                    } else {
                        try stdout.writeAll("skipped  reserved (platform-assigned)\n");
                    }
                },
                .skip_managed => try stdout.writeAll("skipped  TIMBAL_* is platform-managed (resolved by the platform before deploy/preview); never pushed\n"),
                .blocked_downgrade => try stdout.print("{s}BLOCKED{s}  secret on the platform, marked plain here\n", .{ Color.bold_yellow, Color.reset }),
            }
        }
        if (type_changes > 0) {
            try stdout.print(
                "\n{s}Note:{s} {d} var(s) request a different type than the platform has stored. Types are fixed at creation;\n" ++
                    "an update only changes the value. Change the type in the Timbal UI (project → variables).\n",
                .{ Color.bold_yellow, Color.reset, type_changes },
            );
        }
    }

    if (blocked > 0) {
        try stderr.print(
            "\nError: {d} var(s) are secrets on the platform but marked plain in the local file. Refusing to push\n" ++
                "with mismatched metadata (the platform keeps the stored type, so a downgrade would not happen, but the\n" ++
                "file is lying about what these are). Fix the file — write `# type: secret` or drop the `# type: plain`\n" ++
                "line — or pass --plain <NAME> to acknowledge and push the value anyway.\n",
            .{blocked},
        );
        std.process.exit(1);
    }

    if (pushable == 0) {
        try stderr.print(
            "Error: nothing to push — every var in {s} is reserved or platform-managed.\n",
            .{file_path},
        );
        std.process.exit(1);
    }

    if (opts.dry_run) {
        try stdout.writeAll("No changes made.\n");
        if (!opts.no_app_ids) {
            if (rev) |r| try syncAppIds(allocator, opts, remote, api_key, r, project_root, repo_root, stdout, stderr);
        }
        return;
    }

    const payload = try buildPushPayload(allocator, rev, sync_vars.items, plan);
    defer allocator.free(payload);

    const body = try apiRequest(allocator, .POST, url, api_key, payload, opts.verbose, rev);
    defer allocator.free(body);

    const parsed = std.json.parseFromSlice(PushResponse, allocator, body, .{
        .ignore_unknown_fields = true,
        .allocate = .alloc_always,
    }) catch |err| {
        try stderr.print("Error: failed to parse push response: {}\n", .{err});
        if (opts.verbose) try stderr.print("{s}\n", .{body});
        std.process.exit(1);
    };
    defer parsed.deinit();

    if (!opts.quiet) {
        try stdout.print(
            "{s}✓{s} Pushed {d} var(s) to rev {s}{s}{s} from {s}\n",
            .{ Color.bold_green, Color.reset, pushable, Color.bold_cyan, parsed.value.rev, Color.reset, file_path },
        );
        try printNameList(stdout, "created", parsed.value.created);
        try printNameList(stdout, "updated", parsed.value.updated);
        try printNameList(stdout, "skipped", parsed.value.skipped);
    }

    if (!opts.no_app_ids) {
        const effective_rev: []const u8 = if (parsed.value.rev.len > 0) parsed.value.rev else (rev orelse "");
        if (effective_rev.len > 0) try syncAppIds(allocator, opts, remote, api_key, effective_rev, project_root, repo_root, stdout, stderr);
    }
}

// ---------------------------------------------------------------------------
// TIMBAL_APP_ID → workforce/<name>/.env
// ---------------------------------------------------------------------------

const LocalMember = struct {
    name: []u8,
    uid: []u8,

    fn deinit(self: *LocalMember, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        allocator.free(self.uid);
    }
};

/// Every `workforce/<name>/timbal.yaml` under the project root (same rules as `timbal start`:
/// dot-prefixed directories are skipped, invalid manifests are reported and skipped).
fn discoverLocalMembers(allocator: std.mem.Allocator, project_root: []const u8, stderr: anytype) !std.ArrayList(LocalMember) {
    var out = std.ArrayList(LocalMember).init(allocator);
    errdefer {
        for (out.items) |*m| m.deinit(allocator);
        out.deinit();
    }

    const wf_path = try std.fmt.allocPrint(allocator, "{s}{s}workforce", .{ project_root, sep });
    defer allocator.free(wf_path);
    var wf_dir = fs.cwd().openDir(wf_path, .{ .iterate = true }) catch return out;
    defer wf_dir.close();

    var iter = wf_dir.iterate();
    while (try iter.next()) |entry| {
        if (entry.kind != .directory) continue;
        if (entry.name.len == 0 or entry.name[0] == '.') continue;

        const yaml_rel = try std.fmt.allocPrint(allocator, "{s}{s}timbal.yaml", .{ entry.name, sep });
        defer allocator.free(yaml_rel);
        const yaml = wf_dir.readFileAlloc(allocator, yaml_rel, 64 * 1024) catch continue;
        defer allocator.free(yaml);

        var config = utils.parseTimbalYaml(allocator, yaml) orelse {
            try stderr.print("{s}Warning:{s} invalid timbal.yaml in workforce/{s}; skipping TIMBAL_APP_ID sync for it.\n", .{ Color.bold_yellow, Color.reset, entry.name });
            continue;
        };
        defer config.deinit(allocator);

        const name = try allocator.dupe(u8, entry.name);
        errdefer allocator.free(name);
        const uid = try allocator.dupe(u8, config.id);
        errdefer allocator.free(uid);
        try out.append(.{ .name = name, .uid = uid });
    }
    return out;
}

fn shortId(s: []const u8) []const u8 {
    return if (s.len > 8) s[0..8] else s;
}

/// Merge `TIMBAL_APP_ID=<platform app id>` into each matched `workforce/<name>/.env`.
/// Only that one line is ever added or rewritten; everything else in the file is preserved
/// byte for byte. The project-root .env is never touched here (it applies to every member).
fn syncAppIds(
    allocator: std.mem.Allocator,
    opts: Options,
    remote: TimbalRemote,
    api_key: []const u8,
    rev: []const u8,
    project_root: []const u8,
    repo_root: []const u8,
    stdout: anytype,
    stderr: anytype,
) !void {
    var members = try discoverLocalMembers(allocator, project_root, stderr);
    defer {
        for (members.items) |*m| m.deinit(allocator);
        members.deinit();
    }
    if (members.items.len == 0) {
        if (opts.verbose) try stderr.writeAll("note: no workforce/<name>/timbal.yaml under the project root; TIMBAL_APP_ID sync skipped.\n");
        return;
    }

    var comps = (try fetchWorkforce(allocator, remote, api_key, rev, opts.verbose)) orelse return;
    defer freeRemoteComponents(allocator, &comps);

    const claimed = try allocator.alloc(bool, comps.items.len);
    defer allocator.free(claimed);
    @memset(claimed, false);

    if (!opts.quiet) {
        if (opts.dry_run) {
            try stdout.print("Workforce app ids (rev {s}) — dry run, nothing written:\n", .{rev});
        } else {
            try stdout.print("Workforce app ids (rev {s}):\n", .{rev});
        }
    }

    const comment = "Platform app id for this workforce member; kept in sync by `timbal env pull/push`.";

    for (members.items) |m| {
        const match = matchComponent(comps.items, m.name, m.uid, claimed) orelse {
            if (opts.quiet) continue;
            if (findNameOnlyConflict(comps.items, m.name, m.uid)) |ci| {
                try stdout.print(
                    "  {s}!{s} workforce/{s}  uid mismatch — local timbal.yaml _id {s}…, platform component \"{s}\" has uid {s}… (app id {s}). Not written.\n",
                    .{ Color.bold_yellow, Color.reset, m.name, shortId(m.uid), comps.items[ci].name, shortId(comps.items[ci].uid.?), comps.items[ci].id },
                );
            } else {
                try stdout.print(
                    "  {s}!{s} workforce/{s}  no platform component on rev {s} (not deployed yet, or timbal.yaml _id differs). Not written.\n",
                    .{ Color.bold_yellow, Color.reset, m.name, rev },
                );
            }
            continue;
        };
        claimed[match.index] = true;
        const comp = comps.items[match.index];

        if (!comp.isRegistered()) {
            if (!opts.quiet) {
                try stdout.print(
                    "  {s}!{s} workforce/{s}  known to the platform but not registered on rev {s} yet (synthetic id {s}); deploy it first. Not written.\n",
                    .{ Color.bold_yellow, Color.reset, m.name, rev, comp.id },
                );
            }
            continue;
        }

        const env_path = try std.fmt.allocPrint(allocator, "{s}{s}workforce{s}{s}{s}.env", .{ project_root, sep, sep, m.name, sep });
        defer allocator.free(env_path);
        const rel_path = try std.fmt.allocPrint(allocator, "workforce{s}{s}{s}.env", .{ sep, m.name, sep });
        defer allocator.free(rel_path);

        const existing: ?[]u8 = fs.cwd().readFileAlloc(allocator, env_path, 1024 * 1024) catch |err| switch (err) {
            error.FileNotFound => null,
            else => {
                try stderr.print("{s}Warning:{s} could not read {s}: {}; skipped.\n", .{ Color.bold_yellow, Color.reset, rel_path, err });
                continue;
            },
        };
        defer if (existing) |e| allocator.free(e);

        var result = try upsertEnvLine(allocator, existing, "TIMBAL_APP_ID", comp.id, comment);
        defer result.deinit(allocator);

        const via: []const u8 = if (match.kind == .uid) "" else " (matched by name — platform component has no manifest uid)";
        if (!opts.dry_run and result.outcome != .unchanged) {
            if (existing == null) {
                try writeFileExclusive(env_path, result.content, stderr);
                if (try gitCheckIgnore(allocator, repo_root, env_path)) |ignored| {
                    if (!ignored) try stderr.print("{s}Warning:{s} {s} is not gitignored.\n", .{ Color.bold_yellow, Color.reset, rel_path });
                }
            } else {
                try writeFileAtomic(allocator, env_path, result.content, stderr);
            }
        }

        if (opts.quiet) continue;
        switch (result.outcome) {
            .unchanged => try stdout.print("  · {s}  TIMBAL_APP_ID={s}  unchanged{s}\n", .{ rel_path, comp.id, via }),
            .added => try stdout.print("  {s}✓{s} {s}  TIMBAL_APP_ID={s}  {s}{s}\n", .{ Color.bold_green, Color.reset, rel_path, comp.id, if (opts.dry_run) "would add" else "added", via }),
            .updated => try stdout.print("  {s}✓{s} {s}  TIMBAL_APP_ID={s}  {s} (was {s}){s}\n", .{ Color.bold_green, Color.reset, rel_path, comp.id, if (opts.dry_run) "would update" else "updated", result.previous orelse "?", via }),
        }
    }

    if (!opts.quiet) {
        for (comps.items, 0..) |c, i| {
            if (claimed[i] or !c.isRegistered()) continue;
            try stdout.print("  note: platform component \"{s}\" (app id {s}) has no matching workforce/ directory here\n", .{ c.name, c.id });
        }
    }
}

/// Encode a query value (branch names are usually safe; still escape reserved chars).
fn urlEncodeQuery(allocator: std.mem.Allocator, s: []const u8) ![]u8 {
    var buf = std.ArrayList(u8).init(allocator);
    errdefer buf.deinit();
    for (s) |c| {
        switch (c) {
            'A'...'Z', 'a'...'z', '0'...'9', '-', '_', '.', '~' => try buf.append(c),
            ' ' => try buf.appendSlice("%20"),
            else => {
                var tmp: [3]u8 = undefined;
                _ = try std.fmt.bufPrint(&tmp, "%{X:0>2}", .{c});
                try buf.appendSlice(&tmp);
            },
        }
    }
    return buf.toOwnedSlice();
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "parseTimbalRemoteUrl accepts platform remotes" {
    const allocator = std.testing.allocator;

    var r = (try parseTimbalRemoteUrl(
        allocator,
        "https://api.dev.timbal.ai/orgs/1/projects/1144/git",
        "origin",
    )).?;
    defer r.deinit(allocator);
    try std.testing.expectEqualStrings("1", r.org_id);
    try std.testing.expectEqualStrings("1144", r.project_id);
    try std.testing.expectEqualStrings("https://api.dev.timbal.ai", r.base_url);
    try std.testing.expectEqualStrings("origin", r.remote_name);

    var r2 = (try parseTimbalRemoteUrl(
        allocator,
        "https://api.timbal.ai/orgs/9/projects/56/git/",
        "timbal",
    )).?;
    defer r2.deinit(allocator);
    try std.testing.expectEqualStrings("9", r2.org_id);
    try std.testing.expectEqualStrings("56", r2.project_id);

    try std.testing.expect(try parseTimbalRemoteUrl(allocator, "git@github.com:foo/bar.git", "origin") == null);
    try std.testing.expect(try parseTimbalRemoteUrl(allocator, "http://api.dev.timbal.ai/orgs/1/projects/1/git", "origin") == null);

    var r3 = (try parseTimbalRemoteUrl(
        allocator,
        "https://api.staging.timbal.ai/orgs/1/projects/2/git",
        "origin",
    )).?;
    defer r3.deinit(allocator);
    try std.testing.expectEqualStrings("https://api.staging.timbal.ai", r3.base_url);

    // Lookalike / non-API hosts must not receive the Bearer token.
    try std.testing.expect(try parseTimbalRemoteUrl(allocator, "https://notimbal.ai/orgs/1/projects/1/git", "origin") == null);
    try std.testing.expect(try parseTimbalRemoteUrl(allocator, "https://evil.timbal.ai/orgs/1/projects/1/git", "origin") == null);
    try std.testing.expect(try parseTimbalRemoteUrl(allocator, "https://api.timbal.ai.evil.com/orgs/1/projects/1/git", "origin") == null);
    try std.testing.expect(try parseTimbalRemoteUrl(allocator, "https://api.foo.bar.timbal.ai/orgs/1/projects/1/git", "origin") == null);
}

test "resolveTimbalRemoteFromConfig prefers origin" {
    const allocator = std.testing.allocator;
    const config =
        \\[core]
        \\  repositoryformatversion = 0
        \\[remote "upstream"]
        \\  url = https://api.timbal.ai/orgs/1/projects/1/git
        \\[remote "origin"]
        \\  url = https://api.dev.timbal.ai/orgs/11/projects/683/git
        \\
    ;
    var r = (try resolveTimbalRemoteFromConfig(allocator, config)).?;
    defer r.deinit(allocator);
    try std.testing.expectEqualStrings("origin", r.remote_name);
    try std.testing.expectEqualStrings("11", r.org_id);
    try std.testing.expectEqualStrings("683", r.project_id);
    try std.testing.expectEqualStrings("https://api.dev.timbal.ai", r.base_url);
}

test "env file round-trip preserves type and description" {
    const allocator = std.testing.allocator;
    const vars = [_]SyncVar{
        .{ .name = "DATABASE_URL", .type = "secret", .value = "postgres://u:p@h/db", .description = "Primary DB" },
        .{ .name = "VITE_FOO", .type = "plain", .value = "bar", .description = null },
        .{ .name = "MSG", .type = "plain", .value = "hello world", .description = "has spaces" },
    };
    const content = try formatEnvFile(allocator, "main", &vars, .{});
    defer allocator.free(content);

    var parsed = try parseEnvFile(allocator, content);
    defer freeSyncVars(allocator, &parsed);

    try std.testing.expectEqual(@as(usize, 3), parsed.items.len);
    try std.testing.expectEqualStrings("DATABASE_URL", parsed.items[0].name);
    try std.testing.expectEqualStrings("secret", parsed.items[0].type);
    try std.testing.expect(parsed.items[0].type_explicit);
    try std.testing.expectEqualStrings("postgres://u:p@h/db", parsed.items[0].value);
    try std.testing.expectEqualStrings("Primary DB", parsed.items[0].description.?);
    try std.testing.expectEqualStrings("VITE_FOO", parsed.items[1].name);
    try std.testing.expectEqualStrings("plain", parsed.items[1].type);
    try std.testing.expectEqualStrings("bar", parsed.items[1].value);
    try std.testing.expectEqualStrings("hello world", parsed.items[2].value);
    try std.testing.expect(!parsed.items[0].managed);
}

test "parseEnvFile defaults type to plain without explicit metadata" {
    const allocator = std.testing.allocator;
    var parsed = try parseEnvFile(allocator, "FOO=1\nBAR=\"x y\"\n");
    defer freeSyncVars(allocator, &parsed);
    try std.testing.expectEqual(@as(usize, 2), parsed.items.len);
    try std.testing.expectEqualStrings("plain", parsed.items[0].type);
    try std.testing.expect(!parsed.items[0].type_explicit);
    try std.testing.expectEqualStrings("x y", parsed.items[1].value);
}

test "parseEnvFile accepts shorthand metadata and case-insensitive types" {
    const allocator = std.testing.allocator;
    var parsed = try parseEnvFile(allocator,
        \\# secret
        \\A=1
        \\# Plain
        \\B=2
        \\# type: SECRET
        \\C=3
        \\# managed: platform
        \\# type: plain
        \\TIMBAL_ORG_ID=1
        \\
    );
    defer freeSyncVars(allocator, &parsed);
    try std.testing.expectEqual(@as(usize, 4), parsed.items.len);
    try std.testing.expectEqualStrings("secret", parsed.items[0].type);
    try std.testing.expect(parsed.items[0].type_explicit);
    try std.testing.expectEqualStrings("plain", parsed.items[1].type);
    try std.testing.expect(parsed.items[1].type_explicit);
    try std.testing.expectEqualStrings("secret", parsed.items[2].type);
    try std.testing.expect(parsed.items[3].managed);
    try std.testing.expect(!parsed.items[2].managed);
}

test "parseEnvFile: blank line ends a metadata block; commented placeholders are ignored" {
    const allocator = std.testing.allocator;
    var parsed = try parseEnvFile(allocator,
        \\# type: secret
        \\# description: orphaned after deleting its var
        \\
        \\NEXT=1
        \\
        \\# type: secret
        \\# value not returned by the platform
        \\# HIDDEN=
        \\
        \\# note: an ordinary comment
        \\LAST=2
        \\
    );
    defer freeSyncVars(allocator, &parsed);
    try std.testing.expectEqual(@as(usize, 2), parsed.items.len);
    try std.testing.expectEqualStrings("NEXT", parsed.items[0].name);
    try std.testing.expect(!parsed.items[0].type_explicit);
    try std.testing.expect(parsed.items[0].description == null);
    try std.testing.expectEqualStrings("LAST", parsed.items[1].name);
    try std.testing.expect(!parsed.items[1].type_explicit);
}

test "formatEnvFile writes platform-managed vars commented out unless requested" {
    const allocator = std.testing.allocator;
    const vars = [_]SyncVar{
        .{ .name = "OPENAI_API_KEY", .type = "secret", .value = "sk-x" },
        .{ .name = "TIMBAL_PROJECT_ENV_ID", .type = "plain", .value = "1234", .managed = true },
        .{ .name = "VITE_TIMBAL_ORG_ID", .type = "plain", .value = "1", .managed = true },
    };
    const content = try formatEnvFile(allocator, "main", &vars, .{});
    defer allocator.free(content);
    try std.testing.expect(std.mem.indexOf(u8, content, "\nOPENAI_API_KEY=sk-x\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, content, "\n# TIMBAL_PROJECT_ENV_ID=1234\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, content, "\nTIMBAL_PROJECT_ENV_ID=") == null);
    try std.testing.expect(std.mem.indexOf(u8, content, "# managed: platform\n") != null);

    // A start-compatible parse must only see the user var.
    var parsed = try parseEnvFile(allocator, content);
    defer freeSyncVars(allocator, &parsed);
    try std.testing.expectEqual(@as(usize, 1), parsed.items.len);
    try std.testing.expectEqualStrings("OPENAI_API_KEY", parsed.items[0].name);

    const active = try formatEnvFile(allocator, "main", &vars, .{ .managed_active = true });
    defer allocator.free(active);
    try std.testing.expect(std.mem.indexOf(u8, active, "\nTIMBAL_PROJECT_ENV_ID=1234\n") != null);
    var parsed_active = try parseEnvFile(allocator, active);
    defer freeSyncVars(allocator, &parsed_active);
    try std.testing.expectEqual(@as(usize, 3), parsed_active.items.len);
    try std.testing.expect(parsed_active.items[1].managed);
}

test "formatEnvFile never writes TIMBAL_APP_ID active and writes missing secrets as placeholders" {
    const allocator = std.testing.allocator;
    const vars = [_]SyncVar{
        .{ .name = "TIMBAL_APP_ID", .type = "plain", .value = "77" },
        .{ .name = "STRIPE_KEY", .type = "secret", .value = "", .value_missing = true },
    };
    const content = try formatEnvFile(allocator, "main", &vars, .{ .managed_active = true });
    defer allocator.free(content);
    try std.testing.expect(std.mem.indexOf(u8, content, "\n# TIMBAL_APP_ID=77\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, content, "\nTIMBAL_APP_ID=") == null);
    try std.testing.expect(std.mem.indexOf(u8, content, "\n# STRIPE_KEY=\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, content, "\nSTRIPE_KEY=") == null);

    var parsed = try parseEnvFile(allocator, content);
    defer freeSyncVars(allocator, &parsed);
    try std.testing.expectEqual(@as(usize, 0), parsed.items.len);
}

test "formatEnvFile keeps multi-line values commented out so a round trip cannot corrupt them" {
    const allocator = std.testing.allocator;
    const vars = [_]SyncVar{
        .{ .name = "PEM", .type = "secret", .value = "-----BEGIN KEY-----\nabc\n-----END KEY-----" },
        .{ .name = "OK", .type = "plain", .value = "1" },
    };
    const content = try formatEnvFile(allocator, "main", &vars, .{});
    defer allocator.free(content);
    try std.testing.expect(std.mem.indexOf(u8, content, "\n# PEM=\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, content, "\nPEM=") == null);
    try std.testing.expect(std.mem.indexOf(u8, content, "BEGIN KEY") == null);

    var parsed = try parseEnvFile(allocator, content);
    defer freeSyncVars(allocator, &parsed);
    try std.testing.expectEqual(@as(usize, 1), parsed.items.len);
    try std.testing.expectEqualStrings("OK", parsed.items[0].name);
}

test "auditEnvContent flags rerouting, misplaced TIMBAL_APP_ID, and dead runtime vars" {
    const allocator = std.testing.allocator;
    const root =
        \\OPENAI_API_KEY=sk-x
        \\# TIMBAL_PROJECT_ENV_ID=1
        \\TIMBAL_PROJECT_ENV_ID=1234
        \\export TIMBAL_APP_ID="5"
        \\PORT=3000
        \\TIMBAL_ORG_ID=1
        \\
    ;
    var findings = std.ArrayList(PlacementFinding).init(allocator);
    defer findings.deinit();
    try auditEnvContent(root, false, &findings);
    try std.testing.expectEqual(@as(usize, 3), findings.items.len);
    try std.testing.expectEqual(PlacementKind.reroute, findings.items[0].kind);
    try std.testing.expectEqualStrings("TIMBAL_PROJECT_ENV_ID", findings.items[0].key);
    try std.testing.expectEqual(PlacementKind.app_id_scope, findings.items[1].kind);
    try std.testing.expectEqual(PlacementKind.dead, findings.items[2].kind);
    try std.testing.expectEqualStrings("PORT", findings.items[2].key);

    // In a member file TIMBAL_APP_ID is exactly where it belongs.
    findings.clearRetainingCapacity();
    try auditEnvContent("TIMBAL_APP_ID=2335\nTIMBAL_DEPLOYMENTS_DOMAIN=x\n", true, &findings);
    try std.testing.expectEqual(@as(usize, 1), findings.items.len);
    try std.testing.expectEqual(PlacementKind.reroute, findings.items[0].kind);
}

test "isValidEnvKey mirrors timbal start's loader" {
    try std.testing.expect(isValidEnvKey("OPENAI_API_KEY"));
    try std.testing.expect(isValidEnvKey("_x1"));
    try std.testing.expect(!isValidEnvKey("1ABC"));
    try std.testing.expect(!isValidEnvKey("my-var"));
    try std.testing.expect(!isValidEnvKey("a.b"));
    try std.testing.expect(!isValidEnvKey(""));
}

test "scopedFileOwner identifies single-service env files" {
    try std.testing.expectEqualStrings("ui", scopedFileOwner("/p", "/p/ui/.env").?);
    try std.testing.expectEqualStrings("api", scopedFileOwner("/p", "/p/api/.env.local").?);
    try std.testing.expectEqualStrings("copilot", scopedFileOwner("/p", "/p/workforce/copilot/.env").?);
    try std.testing.expect(scopedFileOwner("/p", "/p/.env") == null);
    try std.testing.expect(scopedFileOwner("/p", "/p/workforce/.env") == null);
    try std.testing.expect(scopedFileOwner("/p", "/elsewhere/.env") == null);
    try std.testing.expect(scopedFileOwner("/p", "/pp/ui/.env") == null);
}

test "formatEnvFile quoting is start-compatible (no backslash escapes)" {
    const allocator = std.testing.allocator;
    const vars = [_]SyncVar{
        .{ .name = "PATH_WIN", .type = "plain", .value = "C:\\Users\\x", .description = null },
        .{ .name = "QUOTED", .type = "secret", .value = "say \"hi\"", .description = null },
        .{ .name = "BOTH", .type = "plain", .value = "a\"b'c", .description = null },
    };
    const content = try formatEnvFile(allocator, "main", &vars, .{});
    defer allocator.free(content);

    // Must not emit shell-style escapes that start's loader would leave literal.
    try std.testing.expect(std.mem.indexOf(u8, content, "\\\\") == null);
    try std.testing.expect(std.mem.indexOf(u8, content, "\\\"") == null);
    try std.testing.expect(std.mem.indexOf(u8, content, "\\n") == null);

    var parsed = try parseEnvFile(allocator, content);
    defer freeSyncVars(allocator, &parsed);
    try std.testing.expectEqualStrings("C:\\Users\\x", parsed.items[0].value);
    try std.testing.expectEqualStrings("say \"hi\"", parsed.items[1].value);
    try std.testing.expectEqualStrings("a\"b'c", parsed.items[2].value);
}

test "inferSecret: credentials by name and value, counts and public keys are plain" {
    try std.testing.expect(inferSecret("OPENAI_API_KEY", "x"));
    try std.testing.expect(inferSecret("DATABASE_PASSWORD", "x"));
    try std.testing.expect(inferSecret("GITHUB_TOKEN", "x"));
    try std.testing.expect(inferSecret("ACCESS_TOKEN", "x"));
    try std.testing.expect(inferSecret("clientSecret", "x"));
    try std.testing.expect(inferSecret("SENTRY_DSN", "x"));
    try std.testing.expect(inferSecret("AWS_SECRET_ACCESS_KEY", "x"));
    try std.testing.expect(inferSecret("OPENAI", "sk-abc"));
    try std.testing.expect(inferSecret("SLACK", "xoxb-1-2"));
    try std.testing.expect(inferSecret("PEM", "-----BEGIN PRIVATE KEY-----"));
    try std.testing.expect(inferSecret("DATABASE_URL", "postgres://user:pass@host:5432/db"));

    try std.testing.expect(!inferSecret("MAX_TOKENS", "1024"));
    try std.testing.expect(!inferSecret("TOKEN_LIMIT", "10"));
    try std.testing.expect(!inferSecret("STRIPE_PUBLISHABLE_KEY", "pk_live_x"));
    try std.testing.expect(!inferSecret("NEXT_PUBLIC_KEY_PREFIX", "x"));
    try std.testing.expect(!inferSecret("DATABASE_URL", "postgres://host:5432/db"));
    try std.testing.expect(!inferSecret("LOG_LEVEL", "debug"));
    try std.testing.expect(!inferSecret("VITE_API_BASE", "https://api.example.com/v1"));
    try std.testing.expect(!inferSecret("TOKENIZER", "cl100k"));
}

fn testRemote(name: []const u8, typ: []const u8) RemoteVar {
    return .{ .id = "1", .name = name, .type = typ };
}

test "planPush: flag > file > remote > inferred; downgrades blocked without --plain" {
    const allocator = std.testing.allocator;
    const vars = [_]SyncVar{
        .{ .name = "A_KEY", .type = "plain", .value = "1" }, // no metadata, remote secret → secret (remote)
        .{ .name = "B", .type = "plain", .value = "1", .type_explicit = true }, // file plain, remote secret → BLOCKED
        .{ .name = "C", .type = "plain", .value = "1", .type_explicit = true }, // file plain + --plain → downgrade allowed
        .{ .name = "D_TOKEN", .type = "plain", .value = "1" }, // new, inferred secret
        .{ .name = "E", .type = "plain", .value = "1" }, // new, inferred plain
        .{ .name = "F", .type = "plain", .value = "1" }, // new + --secret
        .{ .name = "G", .type = "secret", .value = "1", .type_explicit = true }, // file secret, remote plain → upgrade ok
        .{ .name = "H", .type = "plain", .value = "1" }, // no metadata, remote plain → plain (remote)
    };
    const remote = [_]RemoteVar{
        testRemote("A_KEY", "secret"),
        testRemote("B", "secret"),
        testRemote("C", "secret"),
        testRemote("G", "plain"),
        testRemote("H", "plain"),
        testRemote("H", "secret"), // same name secret in another env → conservative secret
    };
    const secret_names = [_][]const u8{"F"};
    const plain_names = [_][]const u8{"C"};
    const plan = try planPush(allocator, &vars, &remote, &secret_names, &plain_names);
    defer allocator.free(plan);

    try std.testing.expectEqual(PlanAction.push, plan[0].action);
    try std.testing.expectEqualStrings("secret", plan[0].type);
    try std.testing.expectEqual(TypeSource.remote, plan[0].source);

    try std.testing.expectEqual(PlanAction.blocked_downgrade, plan[1].action);

    try std.testing.expectEqual(PlanAction.push, plan[2].action);
    try std.testing.expectEqualStrings("plain", plan[2].type);
    try std.testing.expectEqual(TypeSource.flag, plan[2].source);

    try std.testing.expectEqual(PlanAction.push, plan[3].action);
    try std.testing.expectEqualStrings("secret", plan[3].type);
    try std.testing.expectEqual(TypeSource.inferred, plan[3].source);

    try std.testing.expectEqualStrings("plain", plan[4].type);
    try std.testing.expectEqual(TypeSource.inferred, plan[4].source);

    try std.testing.expectEqualStrings("secret", plan[5].type);
    try std.testing.expectEqual(TypeSource.flag, plan[5].source);

    try std.testing.expectEqual(PlanAction.push, plan[6].action);
    try std.testing.expectEqualStrings("secret", plan[6].type);
    try std.testing.expectEqual(TypeSource.file, plan[6].source);

    try std.testing.expectEqualStrings("secret", plan[7].type);
    try std.testing.expectEqual(TypeSource.remote, plan[7].source);
}

test "planPush: reserved and TIMBAL_* vars are never pushed, whatever the platform or flags say" {
    const allocator = std.testing.allocator;
    const vars = [_]SyncVar{
        .{ .name = "PORT", .type = "plain", .value = "3000" },
        .{ .name = "TIMBAL_PROJECT_SECRET", .type = "secret", .value = "x" },
        .{ .name = "TIMBAL_APP_ID", .type = "plain", .value = "77" },
        .{ .name = "TIMBAL_PROJECT_ENV_ID", .type = "plain", .value = "12" }, // prefix, not user-defined
        .{ .name = "VITE_TIMBAL_ORG_ID", .type = "plain", .value = "1" },
        .{ .name = "TIMBAL_KB_ID", .type = "plain", .value = "9", .type_explicit = true }, // even user-defined on the platform
        .{ .name = "TIMBAL_LOG_EVENTS", .type = "plain", .value = "START" }, // even when named by --secret
        .{ .name = "FOO", .type = "plain", .value = "1", .managed = true }, // annotated managed
        .{ .name = "OK", .type = "plain", .value = "1" },
    };
    const remote = [_]RemoteVar{testRemote("TIMBAL_KB_ID", "plain")};
    const secret_names = [_][]const u8{"TIMBAL_LOG_EVENTS"};
    const plan = try planPush(allocator, &vars, &remote, &secret_names, &.{});
    defer allocator.free(plan);
    try std.testing.expectEqual(PlanAction.skip_reserved, plan[0].action);
    try std.testing.expectEqual(PlanAction.skip_reserved, plan[1].action);
    try std.testing.expectEqual(PlanAction.skip_reserved, plan[2].action);
    try std.testing.expectEqual(PlanAction.skip_managed, plan[3].action);
    try std.testing.expectEqual(PlanAction.skip_managed, plan[4].action);
    try std.testing.expectEqual(PlanAction.skip_managed, plan[5].action);
    try std.testing.expectEqual(PlanAction.skip_managed, plan[6].action);
    try std.testing.expectEqual(PlanAction.skip_managed, plan[7].action);
    try std.testing.expectEqual(PlanAction.push, plan[8].action);

    const payload = try buildPushPayload(allocator, "main", &vars, plan);
    defer allocator.free(payload);
    try std.testing.expect(std.mem.indexOf(u8, payload, "TIMBAL_") == null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"name\":\"OK\"") != null);
}

test "buildPushPayload includes rev and only planned vars with resolved types" {
    const allocator = std.testing.allocator;
    const vars = [_]SyncVar{
        .{ .name = "A", .type = "plain", .value = "1", .description = null },
        .{ .name = "PORT", .type = "plain", .value = "3000", .description = null },
        .{ .name = "TIMBAL_APP_ID", .type = "plain", .value = "77", .description = null },
        .{ .name = "MY_KEY", .type = "plain", .value = "1", .description = "d" },
    };
    const plan = try planPush(allocator, &vars, &.{}, &.{}, &.{});
    defer allocator.free(plan);
    const payload = try buildPushPayload(allocator, "main", &vars, plan);
    defer allocator.free(payload);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"rev\":\"main\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"name\":\"A\",\"type\":\"plain\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"name\":\"MY_KEY\",\"type\":\"secret\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"description\":\"d\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"name\":\"PORT\"") == null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"name\":\"TIMBAL_APP_ID\"") == null);
}

test "upsertEnvLine appends, updates in place, preserves everything else" {
    const allocator = std.testing.allocator;

    var r1 = try upsertEnvLine(allocator, null, "TIMBAL_APP_ID", "2335", "note");
    defer r1.deinit(allocator);
    try std.testing.expectEqual(UpsertOutcome.added, r1.outcome);
    try std.testing.expectEqualStrings("# note\nTIMBAL_APP_ID=2335\n", r1.content);

    // No trailing newline on the existing file; comments and order untouched.
    var r2 = try upsertEnvLine(allocator, "# my stuff\nFOO=bar", "TIMBAL_APP_ID", "2335", null);
    defer r2.deinit(allocator);
    try std.testing.expectEqual(UpsertOutcome.added, r2.outcome);
    try std.testing.expectEqualStrings("# my stuff\nFOO=bar\nTIMBAL_APP_ID=2335\n", r2.content);

    var r3 = try upsertEnvLine(allocator, "FOO=bar\nexport TIMBAL_APP_ID=\"999\"\n# tail\n", "TIMBAL_APP_ID", "2335", "note");
    defer r3.deinit(allocator);
    try std.testing.expectEqual(UpsertOutcome.updated, r3.outcome);
    try std.testing.expectEqualStrings("999", r3.previous.?);
    try std.testing.expectEqualStrings("FOO=bar\nTIMBAL_APP_ID=2335\n# tail\n", r3.content);

    var r4 = try upsertEnvLine(allocator, "TIMBAL_APP_ID=2335\nX=1\n", "TIMBAL_APP_ID", "2335", "note");
    defer r4.deinit(allocator);
    try std.testing.expectEqual(UpsertOutcome.unchanged, r4.outcome);
    try std.testing.expectEqualStrings("TIMBAL_APP_ID=2335\nX=1\n", r4.content);

    // CRLF files stay CRLF; the commented-out placeholder is not a definition.
    var r5 = try upsertEnvLine(allocator, "A=1\r\n# TIMBAL_APP_ID=1\r\n", "TIMBAL_APP_ID", "5", null);
    defer r5.deinit(allocator);
    try std.testing.expectEqual(UpsertOutcome.added, r5.outcome);
    try std.testing.expectEqualStrings("A=1\r\n# TIMBAL_APP_ID=1\r\nTIMBAL_APP_ID=5\r\n", r5.content);

    var r6 = try upsertEnvLine(allocator, "TIMBAL_APP_ID=1\r\nB=2\r\n", "TIMBAL_APP_ID", "5", null);
    defer r6.deinit(allocator);
    try std.testing.expectEqual(UpsertOutcome.updated, r6.outcome);
    try std.testing.expectEqualStrings("TIMBAL_APP_ID=5\r\nB=2\r\n", r6.content);
}

test "parsePullResponse tolerates string, VarValue object, and missing values" {
    const allocator = std.testing.allocator;
    const body =
        \\{"rev":"main","vars":[
        \\  {"name":"A","type":"plain","value":"1","description":null},
        \\  {"name":"B","type":"secret","value":{"type":"secret","decrypted":"s3"}},
        \\  {"name":"C","value":{"type":"plain","value":"v"}},
        \\  {"name":"D","type":"secret","value":{"type":"secret","preview":"ab…"}},
        \\  {"name":"E","type":"plain"},
        \\  {"name":"TIMBAL_ORG_ID","type":"plain","value":"1","source":"platform"},
        \\  {"name":"F","type":"plain","value":"1","managed":true}
        \\]}
    ;
    var res = try parsePullResponse(allocator, body);
    defer res.deinit(allocator);
    try std.testing.expectEqualStrings("main", res.rev);
    try std.testing.expectEqual(@as(usize, 7), res.vars.items.len);
    try std.testing.expect(!res.vars.items[0].managed);
    try std.testing.expect(res.vars.items[5].managed);
    try std.testing.expect(res.vars.items[6].managed);
    try std.testing.expectEqualStrings("1", res.vars.items[0].value);
    try std.testing.expect(res.vars.items[0].type_explicit);
    try std.testing.expectEqualStrings("secret", res.vars.items[1].type);
    try std.testing.expectEqualStrings("s3", res.vars.items[1].value);
    try std.testing.expectEqualStrings("plain", res.vars.items[2].type);
    try std.testing.expectEqualStrings("v", res.vars.items[2].value);
    try std.testing.expect(res.vars.items[3].value_missing);
    try std.testing.expect(!res.vars.items[4].value_missing); // empty plain var is legitimate
    try std.testing.expectEqualStrings("", res.vars.items[4].value);
}

test "parseRemoteVarList reads types, env scoping, and ids as strings" {
    const allocator = std.testing.allocator;
    const body =
        \\{"vars":[
        \\  {"id":41,"name":"TIMBAL_PROJECT_SECRET","value":{"type":"secret","preview":"x","decrypted":null},"envs":[{"id":7,"name":"main","color":"#fff"}],"applies_to_all_envs":false},
        \\  {"id":"42","name":"FOO","value":{"type":"plain","value":"1"},"envs":[],"applies_to_all_envs":true}
        \\]}
    ;
    var list = try parseRemoteVarList(allocator, body);
    defer freeRemoteVars(allocator, &list);
    try std.testing.expectEqual(@as(usize, 2), list.items.len);
    try std.testing.expectEqualStrings("41", list.items[0].id);
    try std.testing.expectEqualStrings("secret", list.items[0].type);
    try std.testing.expect(list.items[0].hasEnv("7"));
    try std.testing.expect(!list.items[0].hasEnv("8"));
    try std.testing.expect(!list.items[0].applies_to_all_envs);
    try std.testing.expectEqualStrings("42", list.items[1].id);
    try std.testing.expectEqualStrings("plain", list.items[1].type);
    try std.testing.expect(list.items[1].applies_to_all_envs);

    try std.testing.expectEqualStrings("secret", remoteTypeFor(list.items, "TIMBAL_PROJECT_SECRET").?);
    try std.testing.expectEqualStrings("plain", remoteTypeFor(list.items, "FOO").?);
    try std.testing.expect(remoteTypeFor(list.items, "NOPE") == null);
}

test "parseVarDecrypted and parseEnvIdForBranch" {
    const allocator = std.testing.allocator;
    const d = try parseVarDecrypted(allocator, "{\"id\":1,\"name\":\"X\",\"value\":{\"type\":\"secret\",\"preview\":\"a\",\"decrypted\":\"plaintext\"}}");
    defer if (d) |s| allocator.free(s);
    try std.testing.expectEqualStrings("plaintext", d.?);
    try std.testing.expect((try parseVarDecrypted(allocator, "{\"value\":{\"type\":\"secret\",\"decrypted\":null}}")) == null);

    const envs = "{\"envs\":[{\"id\":3,\"name\":\"prod\",\"branch\":\"main\"},{\"id\":4,\"name\":\"stg\",\"branch\":null}]}";
    const id = try parseEnvIdForBranch(allocator, envs, "main");
    defer if (id) |s| allocator.free(s);
    try std.testing.expectEqualStrings("3", id.?);
    try std.testing.expect((try parseEnvIdForBranch(allocator, envs, "feature")) == null);
}

test "parseWorkforceList and matchComponent" {
    const allocator = std.testing.allocator;
    const body =
        \\{"workforce":[
        \\  {"id":"2335","name":"copilot","type":"agent","uid":"ac673db8356ab7d319a667290179b735"},
        \\  {"id":88,"name":"legacy","type":"workflow","uid":null},
        \\  {"id":"90","name":"renamed","type":"agent","uid":"ffff"},
        \\  {"id":-731055215,"name":"fresh","type":"agent","uid":"0123"}
        \\]}
    ;
    var comps = try parseWorkforceList(allocator, body);
    defer freeRemoteComponents(allocator, &comps);
    try std.testing.expectEqual(@as(usize, 4), comps.items.len);
    try std.testing.expectEqualStrings("2335", comps.items[0].id);
    try std.testing.expectEqualStrings("88", comps.items[1].id);
    try std.testing.expect(comps.items[1].uid == null);
    try std.testing.expect(comps.items[0].isRegistered());
    // Unregistered worktree members carry a negative synthetic hash — never a TIMBAL_APP_ID.
    try std.testing.expectEqualStrings("-731055215", comps.items[3].id);
    try std.testing.expect(!comps.items[3].isRegistered());

    var claimed = [_]bool{ false, false, false, false };
    // uid wins even when the directory name differs.
    const m1 = matchComponent(comps.items, "copilot-renamed-locally", "ac673db8356ab7d319a667290179b735", &claimed).?;
    try std.testing.expectEqual(@as(usize, 0), m1.index);
    try std.testing.expectEqual(MatchKind.uid, m1.kind);
    // Name fallback only for components without a uid.
    const m2 = matchComponent(comps.items, "legacy", "0000", &claimed).?;
    try std.testing.expectEqual(@as(usize, 1), m2.index);
    try std.testing.expectEqual(MatchKind.name, m2.kind);
    // Same name but a different uid is a different manifest identity → no match.
    try std.testing.expect(matchComponent(comps.items, "renamed", "0000", &claimed) == null);
    try std.testing.expectEqual(@as(usize, 2), findNameOnlyConflict(comps.items, "renamed", "0000").?);
    // Claimed components are not matched twice.
    claimed[0] = true;
    try std.testing.expect(matchComponent(comps.items, "copilot", "ac673db8356ab7d319a667290179b735", &claimed) == null);
}

test "reserved and platform-managed name rules" {
    try std.testing.expect(isReservedVarName("TIMBAL_APP_ID"));
    try std.testing.expect(isReservedVarName("PORT"));
    try std.testing.expect(isReservedVarName("TIMBAL_PROJECT_SECRET"));
    try std.testing.expect(!isReservedVarName("TIMBAL_ORG_ID"));
    try std.testing.expect(isPlatformManagedName("TIMBAL_PROJECT_ENV_ID"));
    try std.testing.expect(isPlatformManagedName("VITE_TIMBAL_PROJECT_REV"));
    try std.testing.expect(isPlatformManagedName("TIMBAL_STUDIO"));
    try std.testing.expect(isPlatformManagedName("VITE_TIMBAL_PREVIEW_HOST"));
    try std.testing.expect(isPlatformManagedName("VITE_AUTH_TIMBAL_IAM"));
    try std.testing.expect(!isPlatformManagedName("VITE_API_URL"));
    try std.testing.expect(!isPlatformManagedName("OPENAI_API_KEY"));
    try std.testing.expectEqualStrings("secret", canonType(" Secret ").?);
    try std.testing.expect(canonType("hidden") == null);
}

test "gitignoreCoversBasename matches the actual file" {
    try std.testing.expect(gitignoreCoversBasename(".env\n", ".env"));
    try std.testing.expect(gitignoreCoversBasename("*.env\n", "secrets.env"));
    try std.testing.expect(gitignoreCoversBasename(".env*\n", ".env.local"));
    // A bare `.env` entry must not cover a custom -f path.
    try std.testing.expect(!gitignoreCoversBasename(".env\n", "secrets.env"));
    try std.testing.expect(!gitignoreCoversBasename("node_modules/\n", ".env"));
}

test "normalizeBaseUrlOverride accepts api hosts" {
    const allocator = std.testing.allocator;
    const a = try normalizeBaseUrlOverride(allocator, "https://api.staging.timbal.ai/");
    defer allocator.free(a);
    try std.testing.expectEqualStrings("https://api.staging.timbal.ai", a);

    const b = try normalizeBaseUrlOverride(allocator, "api.dev.timbal.ai");
    defer allocator.free(b);
    try std.testing.expectEqualStrings("https://api.dev.timbal.ai", b);

    try std.testing.expectError(error.InsecureBaseUrl, normalizeBaseUrlOverride(allocator, "http://api.dev.timbal.ai"));
    try std.testing.expectError(error.InvalidBaseUrl, normalizeBaseUrlOverride(allocator, "https://evil.timbal.ai"));
}
