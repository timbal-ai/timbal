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
        "    \x1b[1;36mpull \x1b[0mDownload vars for the env tracking --rev into a local .env file\n" ++
        "    \x1b[1;36mpush \x1b[0mUpsert local .env vars into the env tracking --rev\n" ++
        "\n" ++
        "\x1b[1;32mOptions:\n" ++
        "    \x1b[1;36m--rev <BRANCH>  \x1b[0mGit branch whose env to sync (default: current branch)\n" ++
        "    \x1b[1;36m--default       \x1b[0mOmit rev; use the project's default-branch env\n" ++
        "    \x1b[1;36m-f\x1b[0m, \x1b[1;36m--file <PATH> \x1b[0mLocal env file (default: .env)\n" ++
        "\n" ++
        "\x1b[1;32mNotes:\n" ++
        "\x1b[0m    Org/project are resolved from the Timbal git remote in .git/config:\n" ++
        "    https://api[.dev].timbal.ai/orgs/{org_id}/projects/{project_id}/git\n" ++
        "    Auth uses the configured profile API key (timbal configure).\n" ++
        "    Secrets are written in plaintext to the local file — keep it gitignored.\n" ++
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
fn isTimbalApiHost(host: []const u8) bool {
    return std.mem.eql(u8, host, "api.timbal.ai") or std.mem.eql(u8, host, "api.dev.timbal.ai");
}

/// Parse a Timbal platform git remote URL.
/// Accepts: https://api[.dev].timbal.ai/orgs/{org}/projects/{project}/git[/]
pub fn parseTimbalRemoteUrl(allocator: std.mem.Allocator, url: []const u8, remote_name: []const u8) !?TimbalRemote {
    const trimmed = std.mem.trim(u8, url, " \t\r\n");
    if (trimmed.len == 0) return null;

    var rest = trimmed;
    var scheme: []const u8 = "https";
    if (std.mem.startsWith(u8, rest, "https://")) {
        scheme = "https";
        rest = rest["https://".len..];
    } else if (std.mem.startsWith(u8, rest, "http://")) {
        scheme = "http";
        rest = rest["http://".len..];
    } else {
        return null;
    }

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

    const base_url = try std.fmt.allocPrint(allocator, "{s}://{s}", .{ scheme, host });
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

// ---------------------------------------------------------------------------
// SyncVar + local .env store
// ---------------------------------------------------------------------------

pub const SyncVar = struct {
    name: []const u8,
    @"type": []const u8, // "plain" | "secret"
    value: []const u8,
    description: ?[]const u8 = null,

    pub fn deinit(self: *SyncVar, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        allocator.free(self.@"type");
        allocator.free(self.value);
        if (self.description) |d| allocator.free(d);
    }
};

fn freeSyncVars(allocator: std.mem.Allocator, vars: *std.ArrayList(SyncVar)) void {
    for (vars.items) |*v| v.deinit(allocator);
    vars.deinit();
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

/// Serialize SyncVars to a .env file with type/description comment metadata.
/// Format is compatible with `timbal start`'s .env loader (quote strip, no escapes).
pub fn formatEnvFile(allocator: std.mem.Allocator, rev: []const u8, vars: []const SyncVar) ![]u8 {
    var buf = std.ArrayList(u8).init(allocator);
    errdefer buf.deinit();

    try buf.appendSlice("# Synced by `timbal env pull`. Keep this file gitignored — secrets are plaintext.\n");
    try buf.appendSlice("# rev: ");
    try buf.appendSlice(rev);
    try buf.append('\n');
    try buf.append('\n');

    for (vars) |v| {
        try buf.appendSlice("# type: ");
        try buf.appendSlice(v.@"type");
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
        try buf.appendSlice(v.name);
        try buf.append('=');
        // Flatten newlines so the line-oriented .env stays start-compatible.
        if (std.mem.indexOfAny(u8, v.value, "\n\r") != null) {
            var flat = try allocator.alloc(u8, v.value.len);
            defer allocator.free(flat);
            for (v.value, 0..) |c, i| {
                flat[i] = if (c == '\n' or c == '\r') ' ' else c;
            }
            if (needsQuoting(flat)) {
                try appendQuotedValue(&buf, flat);
            } else {
                try buf.appendSlice(flat);
            }
        } else if (needsQuoting(v.value)) {
            try appendQuotedValue(&buf, v.value);
        } else {
            try buf.appendSlice(v.value);
        }
        try buf.append('\n');
        try buf.append('\n');
    }

    return buf.toOwnedSlice();
}

/// Parse a local .env written by pull (or a plain KEY=VALUE file).
/// Type defaults to "plain" when metadata is missing.
pub fn parseEnvFile(allocator: std.mem.Allocator, content: []const u8) !std.ArrayList(SyncVar) {
    var out = std.ArrayList(SyncVar).init(allocator);
    errdefer freeSyncVars(allocator, &out);

    var pending_type: ?[]const u8 = null;
    var pending_desc: ?[]const u8 = null;
    defer if (pending_type) |t| allocator.free(t);
    defer if (pending_desc) |d| allocator.free(d);

    var lines = std.mem.splitScalar(u8, content, '\n');
    while (lines.next()) |raw_line| {
        var line = std.mem.trim(u8, raw_line, " \t\r");
        if (line.len == 0) continue;

        if (line[0] == '#') {
            const body = std.mem.trim(u8, line[1..], " \t");
            if (std.mem.startsWith(u8, body, "type:")) {
                const t = std.mem.trim(u8, body["type:".len..], " \t");
                if (pending_type) |old| allocator.free(old);
                pending_type = try allocator.dupe(u8, t);
            } else if (std.mem.startsWith(u8, body, "description:")) {
                const d = std.mem.trim(u8, body["description:".len..], " \t");
                if (pending_desc) |old| allocator.free(old);
                pending_desc = try allocator.dupe(u8, d);
            }
            continue;
        }

        if (std.mem.startsWith(u8, line, "export ")) {
            line = std.mem.trimLeft(u8, line["export ".len..], " \t");
        }

        const eq = std.mem.indexOfScalar(u8, line, '=') orelse continue;
        const key = std.mem.trim(u8, line[0..eq], " \t");
        if (key.len == 0) continue;

        // Match `timbal start`: strip matching outer quotes only; no unescape.
        var value_raw = std.mem.trim(u8, line[eq + 1 ..], " \t");
        if (value_raw.len >= 2) {
            const first = value_raw[0];
            const last = value_raw[value_raw.len - 1];
            if ((first == '"' and last == '"') or (first == '\'' and last == '\'')) {
                value_raw = value_raw[1 .. value_raw.len - 1];
            }
        }
        const value = try allocator.dupe(u8, value_raw);
        errdefer allocator.free(value);

        const typ = if (pending_type) |t| blk: {
            const owned = t;
            pending_type = null;
            break :blk owned;
        } else try allocator.dupe(u8, "plain");
        errdefer allocator.free(typ);

        const desc = blk: {
            if (pending_desc) |d| {
                pending_desc = null;
                break :blk d;
            }
            break :blk null;
        };

        try out.append(.{
            .name = try allocator.dupe(u8, key),
            .@"type" = typ,
            .value = value,
            .description = desc,
        });
    }

    return out;
}

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------

const PullResponse = struct {
    rev: []const u8,
    vars: []SyncVarJson,
};

const SyncVarJson = struct {
    name: []const u8,
    @"type": []const u8,
    value: []const u8,
    description: ?[]const u8 = null,
};

const PushResponse = struct {
    rev: []const u8,
    created: [][]const u8 = &.{},
    updated: [][]const u8 = &.{},
    skipped: [][]const u8 = &.{},
};

fn apiRequest(
    allocator: std.mem.Allocator,
    method: std.http.Method,
    url: []const u8,
    api_key: []const u8,
    payload: ?[]const u8,
    verbose: bool,
) ![]u8 {
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

    const status = @intFromEnum(result.status);
    if (status < 200 or status >= 300) {
        const stderr = std.io.getStdErr().writer();
        try stderr.print("Error: API returned HTTP {d}\n", .{status});
        if (body.items.len > 0) {
            const snippet = if (body.items.len > 500) body.items[0..500] else body.items;
            try stderr.print("{s}\n", .{snippet});
        }
        return error.HttpStatus;
    }

    return body.toOwnedSlice();
}

fn buildPushPayload(allocator: std.mem.Allocator, rev: ?[]const u8, vars: []const SyncVar) ![]u8 {
    var buf = std.ArrayList(u8).init(allocator);
    errdefer buf.deinit();
    var w = buf.writer();

    try w.writeAll("{\"vars\":[");
    for (vars, 0..) |v, i| {
        if (i > 0) try w.writeAll(",");
        try w.writeAll("{\"name\":");
        try std.json.stringify(v.name, .{}, w);
        try w.writeAll(",\"type\":");
        try std.json.stringify(v.@"type", .{}, w);
        try w.writeAll(",\"value\":");
        try std.json.stringify(v.value, .{}, w);
        if (v.description) |d| {
            try w.writeAll(",\"description\":");
            try std.json.stringify(d, .{}, w);
        }
        try w.writeAll("}");
    }
    try w.writeAll("]");
    if (rev) |r| {
        try w.writeAll(",\"rev\":");
        try std.json.stringify(r, .{}, w);
    }
    try w.writeAll("}");
    return buf.toOwnedSlice();
}

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
    profile: ?[]const u8 = null,
    verbose: bool = false,
    quiet: bool = false,
};

fn parseArgs(allocator: std.mem.Allocator, args: []const []const u8) !Options {
    _ = allocator;
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
    };

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "-v") or std.mem.eql(u8, arg, "--verbose")) {
            opts.verbose = true;
        } else if (std.mem.eql(u8, arg, "-q") or std.mem.eql(u8, arg, "--quiet")) {
            opts.quiet = true;
        } else if (std.mem.eql(u8, arg, "--default")) {
            opts.use_default_rev = true;
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

    return opts;
}

pub fn run(allocator: std.mem.Allocator, args: []const []const u8) !void {
    const opts = try parseArgs(allocator, args);
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

    if (opts.verbose) {
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
    }

    const file_path = if (fs.path.isAbsolute(opts.file))
        try allocator.dupe(u8, opts.file)
    else
        try std.fmt.allocPrint(allocator, "{s}{s}{s}", .{ repo_root, sep, opts.file });
    defer allocator.free(file_path);

    switch (opts.action) {
        .pull => try runPull(allocator, opts, remote, rev, api_key, file_path, stdout, stderr),
        .push => try runPush(allocator, opts, remote, rev, api_key, file_path, stdout, stderr),
    }
}

fn runPull(
    allocator: std.mem.Allocator,
    opts: Options,
    remote: TimbalRemote,
    rev: ?[]const u8,
    api_key: []const u8,
    file_path: []const u8,
    stdout: anytype,
    stderr: anytype,
) !void {
    const url = if (rev) |r| blk: {
        const encoded = try urlEncodeQuery(allocator, r);
        defer allocator.free(encoded);
        break :blk try std.fmt.allocPrint(
            allocator,
            "{s}/orgs/{s}/projects/{s}/vars/pull?rev={s}",
            .{ remote.base_url, remote.org_id, remote.project_id, encoded },
        );
    } else try std.fmt.allocPrint(
        allocator,
        "{s}/orgs/{s}/projects/{s}/vars/pull",
        .{ remote.base_url, remote.org_id, remote.project_id },
    );
    defer allocator.free(url);

    const body = try apiRequest(allocator, .GET, url, api_key, null, opts.verbose);
    defer allocator.free(body);

    const parsed = std.json.parseFromSlice(PullResponse, allocator, body, .{
        .ignore_unknown_fields = true,
        .allocate = .alloc_always,
    }) catch |err| {
        try stderr.print("Error: failed to parse pull response: {}\n", .{err});
        if (opts.verbose) try stderr.print("{s}\n", .{body});
        std.process.exit(1);
    };
    defer parsed.deinit();

    var sync_vars = std.ArrayList(SyncVar).init(allocator);
    defer freeSyncVars(allocator, &sync_vars);
    for (parsed.value.vars) |v| {
        try sync_vars.append(.{
            .name = try allocator.dupe(u8, v.name),
            .@"type" = try allocator.dupe(u8, v.@"type"),
            .value = try allocator.dupe(u8, v.value),
            .description = if (v.description) |d| try allocator.dupe(u8, d) else null,
        });
    }

    const content = try formatEnvFile(allocator, parsed.value.rev, sync_vars.items);
    defer allocator.free(content);

    // Ensure parent dir exists for -f nested paths.
    if (fs.path.dirname(file_path)) |dir| {
        fs.cwd().makePath(dir) catch |err| {
            if (err != error.PathAlreadyExists) {
                try stderr.print("Error: could not create directory for {s}: {}\n", .{ file_path, err });
                std.process.exit(1);
            }
        };
    }

    const file = fs.cwd().createFile(file_path, .{}) catch |err| {
        try stderr.print("Error: could not write {s}: {}\n", .{ file_path, err });
        std.process.exit(1);
    };
    defer file.close();
    try file.writeAll(content);

    if (!opts.quiet) {
        try stdout.print(
            "{s}✓{s} Pulled {d} var(s) for rev {s}{s}{s} → {s}\n",
            .{ Color.bold_green, Color.reset, sync_vars.items.len, Color.bold_cyan, parsed.value.rev, Color.reset, file_path },
        );
    }
}

fn runPush(
    allocator: std.mem.Allocator,
    opts: Options,
    remote: TimbalRemote,
    rev: ?[]const u8,
    api_key: []const u8,
    file_path: []const u8,
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

    // Validate types
    for (sync_vars.items) |v| {
        if (!std.mem.eql(u8, v.@"type", "plain") and !std.mem.eql(u8, v.@"type", "secret")) {
            try stderr.print("Error: var '{s}' has invalid type '{s}' (expected plain|secret)\n", .{ v.name, v.@"type" });
            std.process.exit(1);
        }
    }

    const payload = try buildPushPayload(allocator, rev, sync_vars.items);
    defer allocator.free(payload);

    const url = try std.fmt.allocPrint(
        allocator,
        "{s}/orgs/{s}/projects/{s}/vars/push",
        .{ remote.base_url, remote.org_id, remote.project_id },
    );
    defer allocator.free(url);

    const body = try apiRequest(allocator, .POST, url, api_key, payload, opts.verbose);
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
            "{s}✓{s} Pushed to rev {s}{s}{s} from {s}\n",
            .{ Color.bold_green, Color.reset, Color.bold_cyan, parsed.value.rev, Color.reset, file_path },
        );
        try printNameList(stdout, "created", parsed.value.created);
        try printNameList(stdout, "updated", parsed.value.updated);
        try printNameList(stdout, "skipped", parsed.value.skipped);
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
    // Lookalike / non-API hosts must not receive the Bearer token.
    try std.testing.expect(try parseTimbalRemoteUrl(allocator, "https://notimbal.ai/orgs/1/projects/1/git", "origin") == null);
    try std.testing.expect(try parseTimbalRemoteUrl(allocator, "https://evil.timbal.ai/orgs/1/projects/1/git", "origin") == null);
    try std.testing.expect(try parseTimbalRemoteUrl(allocator, "https://api.timbal.ai.evil.com/orgs/1/projects/1/git", "origin") == null);
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
        .{ .name = "DATABASE_URL", .@"type" = "secret", .value = "postgres://u:p@h/db", .description = "Primary DB" },
        .{ .name = "VITE_FOO", .@"type" = "plain", .value = "bar", .description = null },
        .{ .name = "MSG", .@"type" = "plain", .value = "hello world", .description = "has spaces" },
    };
    const content = try formatEnvFile(allocator, "main", &vars);
    defer allocator.free(content);

    var parsed = try parseEnvFile(allocator, content);
    defer freeSyncVars(allocator, &parsed);

    try std.testing.expectEqual(@as(usize, 3), parsed.items.len);
    try std.testing.expectEqualStrings("DATABASE_URL", parsed.items[0].name);
    try std.testing.expectEqualStrings("secret", parsed.items[0].@"type");
    try std.testing.expectEqualStrings("postgres://u:p@h/db", parsed.items[0].value);
    try std.testing.expectEqualStrings("Primary DB", parsed.items[0].description.?);
    try std.testing.expectEqualStrings("VITE_FOO", parsed.items[1].name);
    try std.testing.expectEqualStrings("plain", parsed.items[1].@"type");
    try std.testing.expectEqualStrings("bar", parsed.items[1].value);
    try std.testing.expectEqualStrings("hello world", parsed.items[2].value);
}

test "parseEnvFile defaults type to plain" {
    const allocator = std.testing.allocator;
    var parsed = try parseEnvFile(allocator, "FOO=1\nBAR=\"x y\"\n");
    defer freeSyncVars(allocator, &parsed);
    try std.testing.expectEqual(@as(usize, 2), parsed.items.len);
    try std.testing.expectEqualStrings("plain", parsed.items[0].@"type");
    try std.testing.expectEqualStrings("x y", parsed.items[1].value);
}

test "formatEnvFile quoting is start-compatible (no backslash escapes)" {
    const allocator = std.testing.allocator;
    const vars = [_]SyncVar{
        .{ .name = "PATH_WIN", .@"type" = "plain", .value = "C:\\Users\\x", .description = null },
        .{ .name = "QUOTED", .@"type" = "secret", .value = "say \"hi\"", .description = null },
        .{ .name = "BOTH", .@"type" = "plain", .value = "a\"b'c", .description = null },
    };
    const content = try formatEnvFile(allocator, "main", &vars);
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

test "buildPushPayload includes rev and vars" {
    const allocator = std.testing.allocator;
    const vars = [_]SyncVar{
        .{ .name = "A", .@"type" = "plain", .value = "1", .description = null },
    };
    const payload = try buildPushPayload(allocator, "main", &vars);
    defer allocator.free(payload);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"rev\":\"main\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"name\":\"A\"") != null);
}
