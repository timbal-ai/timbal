const std = @import("std");
const fs = std.fs;

const utils = @import("../utils.zig");
const Color = utils.Color;

const builtin = @import("builtin");

fn enableWindowsConsole() u32 {
    if (builtin.os.tag == .windows) {
        const windows = std.os.windows;

        // Set output codepage to UTF-8.
        const orig_cp = windows.kernel32.GetConsoleOutputCP();
        _ = windows.kernel32.SetConsoleOutputCP(65001);

        // Enable VT processing on stdout so ANSI escape codes render correctly.
        const stdout_handle = std.io.getStdOut().handle;
        var mode: windows.DWORD = 0;
        if (windows.kernel32.GetConsoleMode(stdout_handle, &mode) != 0) {
            _ = windows.kernel32.SetConsoleMode(stdout_handle, mode | windows.ENABLE_VIRTUAL_TERMINAL_PROCESSING);
        }

        return orig_cp;
    }
    return 0;
}

fn restoreWindowsConsole(orig_cp: u32) void {
    if (builtin.os.tag == .windows) {
        _ = std.os.windows.kernel32.SetConsoleOutputCP(orig_cp);
    }
}

const is_windows = builtin.os.tag == .windows;
const sep = if (is_windows) "\\" else "/";

// Terminal state saved before spawning services so we can restore it on exit.
// Child processes like `bun run dev` put the terminal in raw mode and may not
// restore it if killed by Ctrl+C — we handle that here instead.
var g_stdin_fd: i32 = -1;
var g_termios_saved: bool = false;
var g_original_termios: if (!is_windows) std.posix.termios else [0]u8 = undefined;

fn restoreTermios() void {
    if (comptime !is_windows) {
        if (g_termios_saved) {
            g_termios_saved = false;
            std.posix.tcsetattr(g_stdin_fd, .FLUSH, g_original_termios) catch {};
        }
    }
}

// ---------------------------------------------------------------------------
// CLI option types: port overrides + env overrides.
// ---------------------------------------------------------------------------

/// Port number overrides supplied via CLI flags. Values are non-null only when
/// the user explicitly set them; an explicit set turns on fail-fast collision
/// checking instead of the default auto-fallback.
const PortOverrides = struct {
    ui: ?u16 = null,
    api: ?u16 = null,
    workforce_base: ?u16 = null,
    /// Per-workforce-member port overrides keyed by member name.
    members: std.StringHashMap(u16),

    fn init(allocator: std.mem.Allocator) PortOverrides {
        return .{ .members = std.StringHashMap(u16).init(allocator) };
    }

    fn deinit(self: *PortOverrides) void {
        var it = self.members.keyIterator();
        while (it.next()) |k| self.members.allocator.free(k.*);
        self.members.deinit();
    }
};

/// Raw `--env` / `--env-file` argument before scope resolution. We can't
/// classify the scope until we've discovered the project layout (because
/// scopes match workforce member names), so we keep the original string.
const RawScopedArg = struct {
    raw: []const u8, // owned
};

/// Holds everything the user passed on the command line. Owns its strings.
const StartOptions = struct {
    project_path: ?[]const u8 = null, // borrowed (from argv)
    profile: ?[]const u8 = null, // borrowed (from argv)
    ports: PortOverrides,
    raw_env_args: std.ArrayList(RawScopedArg),
    raw_env_files: std.ArrayList(RawScopedArg),

    fn init(allocator: std.mem.Allocator) StartOptions {
        return .{
            .ports = PortOverrides.init(allocator),
            .raw_env_args = std.ArrayList(RawScopedArg).init(allocator),
            .raw_env_files = std.ArrayList(RawScopedArg).init(allocator),
        };
    }

    fn deinit(self: *StartOptions, allocator: std.mem.Allocator) void {
        self.ports.deinit();
        for (self.raw_env_args.items) |e| allocator.free(e.raw);
        for (self.raw_env_files.items) |e| allocator.free(e.raw);
        self.raw_env_args.deinit();
        self.raw_env_files.deinit();
    }
};

/// Parse an unsigned u16 port, rejecting 0 and non-numeric input.
fn parsePort(s: []const u8) ?u16 {
    if (s.len == 0) return null;
    const n = std.fmt.parseInt(u32, s, 10) catch return null;
    if (n == 0 or n > 65535) return null;
    return @intCast(n);
}

/// Parse a `--port NAME=PORT` value, returning name + port.
const PortKv = struct { name: []const u8, port: u16 };

fn parsePortKv(arg: []const u8) ?PortKv {
    const eq = std.mem.indexOfScalar(u8, arg, '=') orelse return null;
    if (eq == 0 or eq == arg.len - 1) return null;
    const name = arg[0..eq];
    const port = parsePort(arg[eq + 1 ..]) orelse return null;
    return .{ .name = name, .port = port };
}

/// Split a `--env`-like argument into (maybe scope, rest) using the algorithm:
///   - find the first `:` in the arg
///   - if the prefix before that `:` matches one of the known scopes, treat it
///     as scope and return (scope, rest)
///   - otherwise return (null, arg) — the colon was part of a value/path
fn detachScope(arg: []const u8, scopes: []const []const u8) struct { scope: ?[]const u8, rest: []const u8 } {
    const colon = std.mem.indexOfScalar(u8, arg, ':') orelse return .{ .scope = null, .rest = arg };
    const prefix = arg[0..colon];
    for (scopes) |s| {
        if (std.mem.eql(u8, prefix, s)) {
            return .{ .scope = s, .rest = arg[colon + 1 ..] };
        }
    }
    return .{ .scope = null, .rest = arg };
}

/// Parse a `KEY=VALUE` string; the *first* `=` splits the two. `KEY` must be
/// non-empty and POSIX-shell-safe-ish (letters, digits, underscore, must not
/// start with a digit). Empty VALUE is allowed.
fn parseKeyValue(arg: []const u8) ?struct { key: []const u8, value: []const u8 } {
    const eq = std.mem.indexOfScalar(u8, arg, '=') orelse return null;
    const key = arg[0..eq];
    if (key.len == 0) return null;
    if (std.ascii.isDigit(key[0])) return null;
    for (key) |c| {
        if (!(std.ascii.isAlphanumeric(c) or c == '_')) return null;
    }
    return .{ .key = key, .value = arg[eq + 1 ..] };
}

/// One parsed entry from a .env file (or the equivalent inline form).
const ParsedEnv = struct {
    key: []const u8, // owned
    value: []const u8, // owned
};

/// Positionally-derived bucket indices for env scoping. `scope_names` is built
/// in a fixed order (ui first if present, then api, then each member in order),
/// so we know which bucket index each service occupies without doing a name
/// lookup. That matters because a name lookup is ambiguous when a workforce
/// member's name happens to collide with "ui" or "api" — the previous
/// implementation used last-match for the read side and first-match for the
/// write side, which silently dropped scoped env entries.
const ScopeBuckets = struct {
    /// 0 means the UI service is not present; otherwise this is the bucket
    /// index (1-based — index 0 is the global bucket).
    ui_bucket: usize,
    api_bucket: usize,
    /// Parallel to `members.items`. Owned; caller must call `deinit`.
    member_buckets: []usize,

    fn deinit(self: *ScopeBuckets, allocator: std.mem.Allocator) void {
        allocator.free(self.member_buckets);
    }
};

fn assignScopeBuckets(
    allocator: std.mem.Allocator,
    has_ui: bool,
    has_api: bool,
    member_count: usize,
) !ScopeBuckets {
    var offset: usize = 0;
    var ui_bucket: usize = 0;
    var api_bucket: usize = 0;
    if (has_ui) {
        offset += 1;
        ui_bucket = offset;
    }
    if (has_api) {
        offset += 1;
        api_bucket = offset;
    }
    const member_buckets = try allocator.alloc(usize, member_count);
    for (member_buckets, 0..) |*b, i| b.* = offset + 1 + i;
    return .{ .ui_bucket = ui_bucket, .api_bucket = api_bucket, .member_buckets = member_buckets };
}

fn freeParsedEnvList(allocator: std.mem.Allocator, list: *std.ArrayList(ParsedEnv)) void {
    for (list.items) |e| {
        allocator.free(e.key);
        allocator.free(e.value);
    }
    list.deinit();
}

/// Dupe `key`/`value` and append the pair to `list`. Each allocation is held
/// under an errdefer, so an OOM from the second dupe or the append frees
/// anything already allocated in this call.
///
/// Centralising this avoids the easy-to-miss-by-inlining pattern where
/// `list.append(.{ .key = try dupe(...), .value = try dupe(...) })` leaks the
/// key allocation if the value dupe OOMs, or both if append OOMs.
fn appendDupedKv(
    allocator: std.mem.Allocator,
    list: *std.ArrayList(ParsedEnv),
    key: []const u8,
    value: []const u8,
) !void {
    const key_owned = try allocator.dupe(u8, key);
    errdefer allocator.free(key_owned);
    const val_owned = try allocator.dupe(u8, value);
    errdefer allocator.free(val_owned);
    try list.append(.{ .key = key_owned, .value = val_owned });
}

/// Two-pass port assignment for workforce members.
///
/// The naive single-pass approach (reserve explicit if present, otherwise
/// auto-allocate from a rolling cursor) is non-deterministic because directory
/// iteration order varies by filesystem. If member A is iterated first and
/// auto-allocates, it can grab a port number that member B (later in the
/// iteration) was about to claim via `--port B=PORT`. B then fails with a
/// misleading "already in use" pointing at our own reservation.
///
/// Pass 1: reserve every member that has an explicit `--port NAME=PORT`.
/// Pass 2: auto-allocate the rest from the rolling cursor; `reserveAvailablePort`
///         naturally skips the ports Pass 1 already holds.
///
/// Pass 0 rejects two `--port` flags assigning the same number to different
/// members, which would otherwise produce a confusing "already in use" error
/// from Pass 1's second attempt.
///
/// Errors are reported to `stderr_w` with friendly messages before returning;
/// the caller is expected to silently exit on `error.PortInUse` /
/// `error.DuplicateExplicitPort` (no stack trace).
fn reserveMemberPorts(
    allocator: std.mem.Allocator,
    members: []WorkforceMember,
    port_overrides: *const std.StringHashMap(u16),
    workforce_base: u16,
    stderr_w: anytype,
) !void {
    {
        var seen = std.AutoHashMap(u16, []const u8).init(allocator);
        defer seen.deinit();
        var it = port_overrides.iterator();
        while (it.next()) |entry| {
            const p = entry.value_ptr.*;
            if (seen.get(p)) |prev_name| {
                try stderr_w.print(
                    "Error: --port assigns the same port {d} to multiple members ('{s}' and '{s}').\n",
                    .{ p, prev_name, entry.key_ptr.* },
                );
                return error.DuplicateExplicitPort;
            }
            try seen.put(p, entry.key_ptr.*);
        }
    }

    for (members) |*member| {
        if (port_overrides.get(member.name)) |explicit| {
            member.reservation = tryReservePort(explicit) orelse {
                const holder = findPortHolderDescription(allocator, explicit);
                defer if (holder) |h| allocator.free(h);
                if (holder) |h| {
                    try stderr_w.print(
                        "Error: requested port {d} for '{s}' is already in use ({s}).\n",
                        .{ explicit, member.name, h },
                    );
                } else {
                    try stderr_w.print(
                        "Error: requested port {d} for '{s}' is already in use.\n",
                        .{ explicit, member.name },
                    );
                }
                return error.PortInUse;
            };
            member.port = explicit;
        }
    }

    var cursor: u16 = workforce_base;
    for (members) |*member| {
        if (member.reservation != null) continue;
        const r = reserveAvailablePort(cursor, 200) orelse {
            try stderr_w.print(
                "Error: no free port for workforce member '{s}' starting from {d}.\n",
                .{ member.name, cursor },
            );
            return error.PortInUse;
        };
        cursor = if (r.port == 65535) 65535 else r.port + 1;
        member.port = r.port;
        member.reservation = r;
    }
}

/// Move every ParsedEnv from `src` into `dest`, preserving order.
///
/// Each `ParsedEnv` owns its `.key`/`.value` heap strings. A naive
/// `for (src.items) |e| try dest.append(e);` is unsafe because:
///   - if append OOMs partway, the strings of un-transferred items are still
///     referenced by src.items; src's plain `deinit()` would only free the
///     slot buffer, not those strings.
/// Pre-reserve capacity on `dest` so the inner copy can't fail, then clear
/// `src` to signal that ownership has moved. The caller is expected to clean
/// up `src` via `freeParsedEnvList` so the pre-reserve OOM path also frees
/// strings still held by `src`.
fn moveParsedEnvIntoBucket(
    src: *std.ArrayList(ParsedEnv),
    dest: *std.ArrayList(ParsedEnv),
) !void {
    try dest.ensureUnusedCapacity(src.items.len);
    for (src.items) |e| dest.appendAssumeCapacity(e);
    src.clearRetainingCapacity();
}

/// Minimal .env file parser:
///   - lines starting with `#` are comments
///   - blank lines are skipped
///   - `KEY=VALUE` with optional leading `export `
///   - VALUE may be wrapped in matching single or double quotes; quotes are stripped
///   - no shell expansion, no continuation, no escape sequences (intentional)
fn parseEnvFile(allocator: std.mem.Allocator, content: []const u8) !std.ArrayList(ParsedEnv) {
    var out = std.ArrayList(ParsedEnv).init(allocator);
    errdefer freeParsedEnvList(allocator, &out);

    var lines = std.mem.splitScalar(u8, content, '\n');
    while (lines.next()) |raw_line| {
        var line = std.mem.trim(u8, raw_line, " \t\r");
        if (line.len == 0 or line[0] == '#') continue;

        if (std.mem.startsWith(u8, line, "export ")) {
            line = std.mem.trimLeft(u8, line["export ".len..], " \t");
        }

        const eq = std.mem.indexOfScalar(u8, line, '=') orelse continue;
        const key_raw = std.mem.trim(u8, line[0..eq], " \t");
        if (key_raw.len == 0) continue;
        if (std.ascii.isDigit(key_raw[0])) continue;
        var key_ok = true;
        for (key_raw) |c| {
            if (!(std.ascii.isAlphanumeric(c) or c == '_')) {
                key_ok = false;
                break;
            }
        }
        if (!key_ok) continue;

        var value = std.mem.trim(u8, line[eq + 1 ..], " \t");
        if (value.len >= 2) {
            const first = value[0];
            const last = value[value.len - 1];
            if ((first == '"' and last == '"') or (first == '\'' and last == '\'')) {
                value = value[1 .. value.len - 1];
            }
        }

        try appendDupedKv(allocator, &out, key_raw, value);
    }

    return out;
}

// ---------------------------------------------------------------------------
// CLI argument parser for `timbal start`.
// ---------------------------------------------------------------------------

const ParseResult = union(enum) {
    options: StartOptions,
    help,
    err: []const u8, // owned; caller must free with the allocator that parsed
};

fn parseStartArgs(allocator: std.mem.Allocator, args: []const []const u8) !ParseResult {
    // --help / -h wins regardless of position so users with broken invocations
    // can still discover the right flags.
    for (args) |a| {
        if (std.mem.eql(u8, a, "-h") or std.mem.eql(u8, a, "--help")) return .help;
    }

    var opts = StartOptions.init(allocator);
    // We cannot use `errdefer` here: a `ParseResult{ .err = ... }` return is a
    // normal union return, not a Zig error return, so errdefer wouldn't fire
    // and any heap state already pushed into `opts` (duped --env strings,
    // duped --port NAME keys, ArrayList/HashMap buffers) would leak. Use a
    // success flag the success path flips just before returning .options.
    var ok = false;
    defer if (!ok) opts.deinit(allocator);

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];

        if (std.mem.eql(u8, arg, "--profile")) {
            i += 1;
            if (i >= args.len) {
                return ParseResult{ .err = try allocator.dupe(u8, "Error: --profile requires a name argument") };
            }
            opts.profile = args[i];
        } else if (std.mem.eql(u8, arg, "--ui-port")) {
            i += 1;
            if (i >= args.len) {
                return ParseResult{ .err = try allocator.dupe(u8, "Error: --ui-port requires a port number") };
            }
            const p = parsePort(args[i]) orelse {
                return ParseResult{ .err = try std.fmt.allocPrint(allocator, "Error: invalid --ui-port '{s}' (must be 1-65535)", .{args[i]}) };
            };
            opts.ports.ui = p;
        } else if (std.mem.eql(u8, arg, "--api-port")) {
            i += 1;
            if (i >= args.len) {
                return ParseResult{ .err = try allocator.dupe(u8, "Error: --api-port requires a port number") };
            }
            const p = parsePort(args[i]) orelse {
                return ParseResult{ .err = try std.fmt.allocPrint(allocator, "Error: invalid --api-port '{s}' (must be 1-65535)", .{args[i]}) };
            };
            opts.ports.api = p;
        } else if (std.mem.eql(u8, arg, "--workforce-port")) {
            i += 1;
            if (i >= args.len) {
                return ParseResult{ .err = try allocator.dupe(u8, "Error: --workforce-port requires a port number") };
            }
            const p = parsePort(args[i]) orelse {
                return ParseResult{ .err = try std.fmt.allocPrint(allocator, "Error: invalid --workforce-port '{s}' (must be 1-65535)", .{args[i]}) };
            };
            opts.ports.workforce_base = p;
        } else if (std.mem.eql(u8, arg, "--port")) {
            i += 1;
            if (i >= args.len) {
                return ParseResult{ .err = try allocator.dupe(u8, "Error: --port requires NAME=PORT") };
            }
            const kv = parsePortKv(args[i]) orelse {
                return ParseResult{ .err = try std.fmt.allocPrint(allocator, "Error: invalid --port '{s}' (expected NAME=PORT)", .{args[i]}) };
            };
            // ui/api shorthand: --port ui=3000 is the same as --ui-port 3000.
            if (std.mem.eql(u8, kv.name, "ui")) {
                opts.ports.ui = kv.port;
            } else if (std.mem.eql(u8, kv.name, "api")) {
                opts.ports.api = kv.port;
            } else {
                const name_owned = try allocator.dupe(u8, kv.name);
                errdefer allocator.free(name_owned);
                // Last write wins for duplicate --port NAME=...
                if (opts.ports.members.fetchRemove(name_owned)) |old| {
                    opts.ports.members.allocator.free(old.key);
                }
                try opts.ports.members.put(name_owned, kv.port);
            }
        } else if (std.mem.eql(u8, arg, "--env")) {
            i += 1;
            if (i >= args.len) {
                return ParseResult{ .err = try allocator.dupe(u8, "Error: --env requires [SCOPE:]KEY=VALUE") };
            }
            try opts.raw_env_args.append(.{ .raw = try allocator.dupe(u8, args[i]) });
        } else if (std.mem.eql(u8, arg, "--env-file")) {
            i += 1;
            if (i >= args.len) {
                return ParseResult{ .err = try allocator.dupe(u8, "Error: --env-file requires [SCOPE:]PATH") };
            }
            try opts.raw_env_files.append(.{ .raw = try allocator.dupe(u8, args[i]) });
        } else if (!std.mem.startsWith(u8, arg, "-")) {
            if (opts.project_path != null) {
                return ParseResult{ .err = try allocator.dupe(u8, "Error: multiple paths provided") };
            }
            opts.project_path = arg;
        } else {
            return ParseResult{ .err = try std.fmt.allocPrint(allocator, "Error: unknown option '{s}'", .{arg}) };
        }
    }

    ok = true;
    return ParseResult{ .options = opts };
}

// ---------------------------------------------------------------------------
// Env composition: layered overrides per scope.
// ---------------------------------------------------------------------------

/// Apply a list of (KEY, VALUE) pairs to an EnvMap, last-write-wins. Used for
/// each precedence layer when composing the final env for a service.
fn applyParsedEnv(env: *std.process.EnvMap, entries: []const ParsedEnv) !void {
    for (entries) |e| try env.put(e.key, e.value);
}

/// Compose a fresh per-service EnvMap by stacking layers in precedence order.
/// Lowest-precedence layer first; the caller is responsible for `deinit`ing
/// the returned map.
///   1. soft_pairs         — timbal soft built-ins (TIMBAL_LOG_FORMAT, etc.)
///   2. shell              — inherited shell env (with PORT pre-scrubbed)
///   3. auto_global        — auto-loaded <project>/.env
///   4. auto_scope         — auto-loaded workforce/<member>/.env (members only)
///   5. file_global        — `--env-file PATH` values
///   6. file_scope         — `--env-file scope:PATH` values
///   7. arg_global         — `--env KEY=VAL` values
///   8. arg_scope          — `--env scope:KEY=VAL` values
///   9. hard_pairs         — runtime info that always wins (TIMBAL_START_*, PORT)
fn composeServiceEnv(
    allocator: std.mem.Allocator,
    soft_pairs: []const [2][]const u8,
    shell: *const std.process.EnvMap,
    auto_global: ?[]const ParsedEnv,
    auto_scope: ?[]const ParsedEnv,
    file_global: []const ParsedEnv,
    file_scope: []const ParsedEnv,
    arg_global: []const ParsedEnv,
    arg_scope: []const ParsedEnv,
    hard_pairs: []const [2][]const u8,
) !std.process.EnvMap {
    var env = std.process.EnvMap.init(allocator);
    errdefer env.deinit();
    for (soft_pairs) |kv| try env.put(kv[0], kv[1]);
    var it = shell.iterator();
    while (it.next()) |e| try env.put(e.key_ptr.*, e.value_ptr.*);
    if (auto_global) |entries| try applyParsedEnv(&env, entries);
    if (auto_scope) |entries| try applyParsedEnv(&env, entries);
    try applyParsedEnv(&env, file_global);
    try applyParsedEnv(&env, file_scope);
    try applyParsedEnv(&env, arg_global);
    try applyParsedEnv(&env, arg_scope);
    for (hard_pairs) |kv| try env.put(kv[0], kv[1]);
    return env;
}

var g_interrupted = std.atomic.Value(bool).init(false);

fn sigintHandler(_: c_int) callconv(.C) void {
    if (g_interrupted.swap(true, .seq_cst)) {
        // Second Ctrl+C: force exit after restoring terminal.
        restoreTermios();
        std.process.exit(130);
    }
    // First Ctrl+C: let the supervisor loop stop child process groups and
    // restore the terminal through the normal cleanup path.
}

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

fn getConfigPath(allocator: std.mem.Allocator) ![]u8 {
    const home = try getHomePath(allocator);
    defer allocator.free(home);
    return std.fmt.allocPrint(allocator, "{s}{s}.timbal{s}config", .{ home, sep, sep });
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

/// Strip protocol prefix (e.g. "https://api.timbal.ai" -> "api.timbal.ai").
fn stripProtocol(url: []const u8) []const u8 {
    if (std.mem.indexOf(u8, url, "://")) |idx| {
        return url[idx + 3 ..];
    }
    return url;
}

fn printUsageWithError(err: []const u8) !void {
    const stderr = std.io.getStdErr().writer();
    try stderr.print("{s}\n\n", .{err});
    try printUsage();
}

fn printUsage() !void {
    const stderr = std.io.getStdErr().writer();
    try stderr.writeAll("Start a Timbal project (UI, API, and all agents and workflows).\n" ++
        "\n" ++
        "\x1b[1;32mUsage: \x1b[1;36mtimbal start \x1b[0;36m[PATH] [OPTIONS]\n" ++
        "\n" ++
        "\x1b[1;32mArguments:\n" ++
        "    \x1b[1;36m[PATH] \x1b[0mPath to the project directory (default: current directory)\n" ++
        "\n" ++
        "\x1b[1;32mPort overrides:\n" ++
        "    \x1b[1;36m--ui-port <PORT>           \x1b[0mUI port (fail if busy)\n" ++
        "    \x1b[1;36m--api-port <PORT>          \x1b[0mAPI port (fail if busy)\n" ++
        "    \x1b[1;36m--workforce-port <PORT>    \x1b[0mBase port for workforce members\n" ++
        "    \x1b[1;36m--port <NAME>=<PORT>       \x1b[0mPort for a specific service (repeatable)\n" ++
        "                                  NAME is `ui`, `api`, or a workforce member name.\n" ++
        "\n" ++
        "\x1b[1;32mEnvironment overrides:\n" ++
        "    \x1b[0m<project>/.env is auto-loaded into every service when present;\n" ++
        "    workforce/<member>/.env applies to that member only. Use --env and --env-file\n" ++
        "    for extras or overrides (SCOPE: ui, api, or member name). Precedence (low → high):\n" ++
        "    built-ins, shell env, auto .env, --env-file, --env, runtime (PORT, TIMBAL_START_*).\n" ++
        "    \x1b[1;36m--env [SCOPE:]KEY=VALUE    \x1b[0mSet an env var for one or all services (repeatable)\n" ++
        "    \x1b[1;36m--env-file [SCOPE:]PATH    \x1b[0mLoad env vars from a .env file (repeatable)\n" ++
        "\n" ++
        "\x1b[1;32mInteractive commands:\n" ++
        "    \x1b[1;36mr\x1b[0m, \x1b[1;36mrestart  \x1b[0mRestart all services\n" ++
        "    \x1b[1;36ms\x1b[0m, \x1b[1;36mstatus   \x1b[0mShow service status, ports, PIDs, and uptime\n" ++
        "    \x1b[1;36mo\x1b[0m, \x1b[1;36mopen     \x1b[0mOpen the app in your browser\n" ++
        "    \x1b[1;36mu\x1b[0m, \x1b[1;36mui       \x1b[0mPrint the UI URL\n" ++
        "    \x1b[1;36ma\x1b[0m, \x1b[1;36mapi      \x1b[0mPrint the API URL\n" ++
        "    \x1b[1;36mw\x1b[0m, \x1b[1;36mworkforce\x1b[0m Print workforce service URLs\n" ++
        "    \x1b[1;36mf <target>\x1b[0m        Focus logs: all, ui, api, workforce, or member name\n" ++
        "    \x1b[1;36mm <target>\x1b[0m        Toggle mute for logs: all, ui, api, workforce, or member name\n" ++
        "    \x1b[1;36mh\x1b[0m, \x1b[1;36mhelp     \x1b[0mShow commands while services are running\n" ++
        "    \x1b[1;36mq\x1b[0m, \x1b[1;36mquit     \x1b[0mStop services and quit\n" ++
        "\n" ++
        utils.global_options_help ++
        "\n");
}

const WorkforceMember = struct {
    name: []const u8,
    config: utils.TimbalYaml,
    port: u16,
    /// Held listening socket on `port`. Released right before we spawn the child
    /// so it can re-bind the same port. Null after release. See `PortReservation`.
    reservation: ?PortReservation = null,
};

/// Run a command in a given directory, printing output indented with a prefix.
/// Forces color output via FORCE_COLOR env var so child processes keep ANSI codes when piped.
/// Returns true if the command exited successfully.
fn runCommand(allocator: std.mem.Allocator, argv: []const []const u8, cwd: []const u8) bool {
    const stdout = std.io.getStdOut().writer();
    var child = std.process.Child.init(argv, allocator);
    child.cwd = cwd;
    child.stderr_behavior = .Pipe;
    child.stdout_behavior = .Pipe;

    // Force color output from child processes.
    var env_map = std.process.getEnvMap(allocator) catch return false;
    defer env_map.deinit();
    env_map.put("FORCE_COLOR", "1") catch return false;
    child.env_map = &env_map;

    child.spawn() catch return false;

    // Read stdout and stderr.
    const child_stdout = child.stdout.?.reader().readAllAlloc(allocator, 1024 * 1024) catch null;
    defer if (child_stdout) |s| allocator.free(s);

    const child_stderr = child.stderr.?.reader().readAllAlloc(allocator, 1024 * 1024) catch null;
    defer if (child_stderr) |s| allocator.free(s);

    const term = child.wait() catch return false;

    // Print indented output.
    if (child_stdout) |output| {
        var lines = std.mem.splitScalar(u8, output, '\n');
        while (lines.next()) |line| {
            if (line.len > 0) stdout.print("  {s}\n", .{line}) catch {};
        }
    }
    if (child_stderr) |output| {
        var lines = std.mem.splitScalar(u8, output, '\n');
        while (lines.next()) |line| {
            if (line.len > 0) stdout.print("  {s}\n", .{line}) catch {};
        }
    }

    return term.Exited == 0;
}

/// Prefix colors for distinguishing services in log output.
const prefix_colors = [_][]const u8{
    "\x1b[1;33m", // bold yellow
    "\x1b[1;34m", // bold blue
    "\x1b[1;32m", // bold green
    "\x1b[1;37m", // bold white
    "\x1b[33m", // yellow
    "\x1b[34m", // blue
    "\x1b[32m", // green
    "\x1b[37m", // white
    "\x1b[38;5;208m", // orange
    "\x1b[38;5;81m", // sky blue
    "\x1b[38;5;114m", // light green
    "\x1b[38;5;180m", // tan
};

/// Context passed to each pipe reader thread.
const PipeReaderCtx = struct {
    pipe: std.fs.File,
    service_name: []const u8,
    service_kind: ServiceKind,
    prefix: []const u8,
    color: []const u8,
    mutex: *std.Thread.Mutex,
    allocator: std.mem.Allocator,
    log_filter: *LogFilterState,
};

const ServiceKind = enum {
    ui,
    api,
    workforce,
};

const LogFocus = enum {
    all,
    name,
    workforce,
};

const LogFilterState = struct {
    mutex: std.Thread.Mutex = .{},
    focus: LogFocus = .all,
    focus_name: [128]u8 = undefined,
    focus_name_len: usize = 0,
    mute_all: bool = false,
    mute_ui: bool = false,
    mute_api: bool = false,
    mute_workforce: bool = false,
    muted_names: std.StringHashMap(void),

    fn init(allocator: std.mem.Allocator) LogFilterState {
        return .{
            .muted_names = std.StringHashMap(void).init(allocator),
        };
    }

    fn deinit(self: *LogFilterState) void {
        var iter = self.muted_names.keyIterator();
        while (iter.next()) |key| {
            self.muted_names.allocator.free(key.*);
        }
        self.muted_names.deinit();
    }

    fn shouldPrint(self: *LogFilterState, service_name: []const u8, service_kind: ServiceKind) bool {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.focus == .workforce and service_kind != .workforce) return false;
        if (self.focus == .name and !std.mem.eql(u8, service_name, self.focus_name[0..self.focus_name_len])) return false;

        if (self.mute_all) return false;
        if (service_kind == .ui and self.mute_ui) return false;
        if (service_kind == .api and self.mute_api) return false;
        if (service_kind == .workforce and self.mute_workforce) return false;
        if (self.muted_names.contains(service_name)) return false;

        return true;
    }

    fn setFocus(self: *LogFilterState, target: []const u8) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (std.mem.eql(u8, target, "all")) {
            self.focus = .all;
            self.focus_name_len = 0;
        } else if (std.mem.eql(u8, target, "workforce")) {
            self.focus = .workforce;
            self.focus_name_len = 0;
        } else {
            self.focus = .name;
            self.focus_name_len = @min(target.len, self.focus_name.len);
            @memcpy(self.focus_name[0..self.focus_name_len], target[0..self.focus_name_len]);
        }
    }

    fn toggleMuteName(self: *LogFilterState, target: []const u8) !bool {
        if (self.muted_names.contains(target)) {
            if (self.muted_names.fetchRemove(target)) |entry| {
                self.muted_names.allocator.free(entry.key);
            }
            return false;
        }

        const owned_target = try self.muted_names.allocator.dupe(u8, target);
        errdefer self.muted_names.allocator.free(owned_target);
        try self.muted_names.put(owned_target, {});
        return true;
    }

    fn toggleMute(self: *LogFilterState, target: []const u8, target_is_member_name: bool) !bool {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (std.mem.eql(u8, target, "all")) {
            self.mute_all = !self.mute_all;
            return self.mute_all;
        }
        if (std.mem.eql(u8, target, "workforce")) {
            self.mute_workforce = !self.mute_workforce;
            return self.mute_workforce;
        }
        if (target_is_member_name) {
            return self.toggleMuteName(target);
        }
        if (std.mem.eql(u8, target, "ui")) {
            self.mute_ui = !self.mute_ui;
            return self.mute_ui;
        }
        if (std.mem.eql(u8, target, "api")) {
            self.mute_api = !self.mute_api;
            return self.mute_api;
        }
        return self.toggleMuteName(target);
    }
};

/// Thread function that reads from a pipe line by line, printing each with a colored prefix.
///
/// Uses a heap-allocated dynamic buffer so a single oversized log line (e.g. a structlog
/// event whose context dumps a Message or a tool result) does not kill the reader thread
/// and silently swallow all subsequent child output.
fn pipeReaderFn(ctx: PipeReaderCtx) void {
    const stdout = std.io.getStdOut().writer();
    const reader = ctx.pipe.reader();

    // 16 MiB cap per line — well above anything a sane log line should produce, but
    // bounded so a runaway producer can't OOM us. Lines longer than this are split.
    const max_line_size: usize = 16 * 1024 * 1024;

    while (true) {
        const line = reader.readUntilDelimiterAlloc(ctx.allocator, '\n', max_line_size) catch |err| switch (err) {
            error.EndOfStream => break,
            error.StreamTooLong => {
                // Single line longer than max_line_size; emit a marker and resync at next '\n'.
                ctx.mutex.lock();
                stdout.print("{s}{s}{s} <line too long, truncated>\n", .{ ctx.color, ctx.prefix, Color.reset }) catch {};
                ctx.mutex.unlock();
                reader.skipUntilDelimiterOrEof('\n') catch break;
                continue;
            },
            else => continue,
        };
        defer ctx.allocator.free(line);
        if (line.len == 0) continue;
        if (!ctx.log_filter.shouldPrint(ctx.service_name, ctx.service_kind)) continue;

        ctx.mutex.lock();
        stdout.print("{s}{s}{s} {s}\n", .{ ctx.color, ctx.prefix, Color.reset, line }) catch {};
        ctx.mutex.unlock();
    }
}

const CommandAction = enum(u8) {
    none,
    help,
    status,
    open,
    urls,
    ui_url,
    api_url,
    workforce_urls,
    focus_logs,
    toggle_mute,
    restart,
    quit,
};

const Command = struct {
    action: CommandAction = .none,
    target: [128]u8 = undefined,
    target_len: usize = 0,
};

const CommandState = struct {
    mutex: std.Thread.Mutex = .{},
    command: Command = .{},

    fn set(self: *CommandState, action: CommandAction, target: []const u8) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        self.command.action = action;
        self.command.target_len = @min(target.len, self.command.target.len);
        if (self.command.target_len > 0) {
            @memcpy(self.command.target[0..self.command.target_len], target[0..self.command.target_len]);
        }
    }

    fn take(self: *CommandState) Command {
        self.mutex.lock();
        defer self.mutex.unlock();

        const command = self.command;
        self.command = .{};
        return command;
    }
};

const CommandInputCtx = struct {
    state: *CommandState,
    stop: *std.atomic.Value(bool),
};

fn targetAfterCommand(command: []const u8, verb: []const u8) ?[]const u8 {
    if (!std.mem.startsWith(u8, command, verb)) return null;
    if (command.len == verb.len) return "";
    if (command[verb.len] != ' ') return null;
    return std.mem.trim(u8, command[verb.len + 1 ..], " \t\r\n");
}

fn targetAfterEitherCommand(command: []const u8, short_verb: []const u8, long_verb: []const u8) ?[]const u8 {
    if (targetAfterCommand(command, short_verb)) |target| return target;
    return targetAfterCommand(command, long_verb);
}

fn commandInputFn(ctx: CommandInputCtx) void {
    const stdin_file = std.io.getStdIn();
    const stdin = stdin_file.reader();

    while (!ctx.stop.load(.seq_cst)) {
        if (comptime !is_windows) {
            var poll_fds = [_]std.posix.pollfd{.{
                .fd = stdin_file.handle,
                .events = std.posix.POLL.IN,
                .revents = 0,
            }};
            const ready = std.posix.poll(&poll_fds, 150) catch continue;
            if (ready == 0) continue;
            if ((poll_fds[0].revents & (std.posix.POLL.HUP | std.posix.POLL.ERR)) != 0) break;
            if ((poll_fds[0].revents & std.posix.POLL.IN) == 0) continue;
        }

        const line = stdin.readUntilDelimiterAlloc(std.heap.page_allocator, '\n', 1024) catch |err| switch (err) {
            error.EndOfStream => break,
            error.StreamTooLong => {
                stdin.skipUntilDelimiterOrEof('\n') catch break;
                continue;
            },
            else => continue,
        };
        defer std.heap.page_allocator.free(line);
        if (ctx.stop.load(.seq_cst)) break;

        const command = std.mem.trim(u8, line, " \t\r\n");
        if (std.mem.eql(u8, command, "r") or std.mem.eql(u8, command, "restart")) {
            ctx.state.set(.restart, "");
        } else if (std.mem.eql(u8, command, "h") or std.mem.eql(u8, command, "help") or std.mem.eql(u8, command, "?")) {
            ctx.state.set(.help, "");
        } else if (std.mem.eql(u8, command, "s") or std.mem.eql(u8, command, "status")) {
            ctx.state.set(.status, "");
        } else if (std.mem.eql(u8, command, "o") or std.mem.eql(u8, command, "open")) {
            ctx.state.set(.open, "");
        } else if (std.mem.eql(u8, command, "url") or std.mem.eql(u8, command, "urls")) {
            ctx.state.set(.urls, "");
        } else if (std.mem.eql(u8, command, "u") or std.mem.eql(u8, command, "ui")) {
            ctx.state.set(.ui_url, "");
        } else if (std.mem.eql(u8, command, "a") or std.mem.eql(u8, command, "api")) {
            ctx.state.set(.api_url, "");
        } else if (std.mem.eql(u8, command, "w") or std.mem.eql(u8, command, "workforce")) {
            ctx.state.set(.workforce_urls, "");
        } else if (targetAfterEitherCommand(command, "f", "focus")) |target| {
            ctx.state.set(.focus_logs, target);
        } else if (targetAfterEitherCommand(command, "m", "mute")) |target| {
            ctx.state.set(.toggle_mute, target);
        } else if (std.mem.eql(u8, command, "q") or std.mem.eql(u8, command, "quit")) {
            ctx.state.set(.quit, "");
            break;
        }
    }
}

fn forceStopSpawnedChild(child: *std.process.Child) void {
    if (comptime !is_windows) {
        std.posix.kill(-child.id, std.posix.SIG.KILL) catch |err| switch (err) {
            error.ProcessNotFound => {},
            error.PermissionDenied => {},
            else => {},
        };
    } else {
        _ = child.kill() catch {};
    }
    _ = child.wait() catch {};
}

/// Spawn a long-running process and start threads to stream its stdout/stderr with a prefix.
/// Returns the child process. Caller is responsible for waiting/killing it.
fn spawnService(
    allocator: std.mem.Allocator,
    argv: []const []const u8,
    cwd: []const u8,
    prefix: []const u8,
    service_kind: ServiceKind,
    color: []const u8,
    mutex: *std.Thread.Mutex,
    threads: *std.ArrayList(std.Thread),
    env_map: *const std.process.EnvMap,
    log_filter: *LogFilterState,
) !std.process.Child {
    var child = std.process.Child.init(argv, allocator);
    child.cwd = cwd;
    child.stdin_behavior = .Ignore;
    child.stderr_behavior = .Pipe;
    child.stdout_behavior = .Pipe;
    child.env_map = env_map;
    if (comptime !is_windows) {
        // Put every service in its own process group so restarts also stop grandchildren
        // spawned by wrappers like `bun run` or `uv run`.
        child.pgid = 0;
    }

    try child.spawn();
    errdefer forceStopSpawnedChild(&child);

    // Spawn threads to read stdout and stderr.
    if (child.stdout) |pipe| {
        const thread = try std.Thread.spawn(.{}, pipeReaderFn, .{PipeReaderCtx{
            .pipe = pipe,
            .service_name = prefix,
            .service_kind = service_kind,
            .prefix = prefix,
            .color = color,
            .mutex = mutex,
            .allocator = allocator,
            .log_filter = log_filter,
        }});
        threads.appendAssumeCapacity(thread);
    }
    if (child.stderr) |pipe| {
        const thread = try std.Thread.spawn(.{}, pipeReaderFn, .{PipeReaderCtx{
            .pipe = pipe,
            .service_name = prefix,
            .service_kind = service_kind,
            .prefix = prefix,
            .color = color,
            .mutex = mutex,
            .allocator = allocator,
            .log_filter = log_filter,
        }});
        threads.appendAssumeCapacity(thread);
    }

    return child;
}

const ServiceStatus = struct {
    name: []const u8,
    port: ?u16,
    pid: if (is_windows) void else std.posix.pid_t,
    started_at_ms: i64,
};

const RunningServices = struct {
    children: std.ArrayList(std.process.Child),
    reader_threads: std.ArrayList(std.Thread),
    statuses: std.ArrayList(ServiceStatus),

    fn deinit(self: *RunningServices) void {
        self.children.deinit();
        self.reader_threads.deinit();
        self.statuses.deinit();
    }
};

fn signalServiceGroup(child: *std.process.Child, sig: u8) void {
    if (comptime !is_windows) {
        if (child.term != null) return;
        std.posix.kill(-child.id, sig) catch |err| switch (err) {
            error.ProcessNotFound => {},
            error.PermissionDenied => {},
            else => {},
        };
    }
}

fn stopServices(services: *RunningServices) void {
    if (services.children.items.len > 0) {
        if (comptime !is_windows) {
            for (services.children.items) |*child| signalServiceGroup(child, std.posix.SIG.INT);
            std.time.sleep(1200 * std.time.ns_per_ms);

            for (services.children.items) |*child| signalServiceGroup(child, std.posix.SIG.TERM);
            std.time.sleep(800 * std.time.ns_per_ms);

            for (services.children.items) |*child| signalServiceGroup(child, std.posix.SIG.KILL);
        } else {
            for (services.children.items) |*child| {
                _ = child.kill() catch {};
            }
        }

        for (services.children.items) |*child| {
            _ = child.wait() catch {};
        }
    }

    for (services.reader_threads.items) |thread| {
        thread.join();
    }

    services.deinit();
}

fn startServices(
    allocator: std.mem.Allocator,
    abs_project_path: []const u8,
    members: []WorkforceMember,
    member_envs: []const *const std.process.EnvMap,
    has_ui: bool,
    ui_port: ?u16,
    ui_env: ?*const std.process.EnvMap,
    ui_reservation: ?*PortReservation,
    has_api: bool,
    api_port: ?u16,
    api_env: ?*const std.process.EnvMap,
    api_reservation: ?*PortReservation,
    output_mutex: *std.Thread.Mutex,
    log_filter: *LogFilterState,
) !RunningServices {
    std.debug.assert(members.len == member_envs.len);

    const stdout = std.io.getStdOut().writer();
    try stdout.print("\n{s}Starting services...{s}\n\n", .{ Color.bold, Color.reset });

    var services = RunningServices{
        .children = std.ArrayList(std.process.Child).init(allocator),
        .reader_threads = std.ArrayList(std.Thread).init(allocator),
        .statuses = std.ArrayList(ServiceStatus).init(allocator),
    };
    errdefer stopServices(&services);

    const service_count = members.len + @as(usize, @intFromBool(has_ui)) + @as(usize, @intFromBool(has_api));
    try services.children.ensureTotalCapacity(service_count);
    try services.statuses.ensureTotalCapacity(service_count);
    try services.reader_threads.ensureTotalCapacity(service_count * 2);

    var color_idx: usize = 0;

    // Start workforce members first. We release each held listener immediately
    // before invoking spawn() so the child can bind the same port. The window
    // between our close() and the child's bind() is microseconds, which is
    // small enough that competing `timbal start` processes (who also hold their
    // own reservations) won't slip in.
    for (members, member_envs) |*member, env_ptr| {
        const member_dir = try std.fmt.allocPrint(allocator, "{s}/workforce/{s}", .{ abs_project_path, member.name });
        defer allocator.free(member_dir);
        const port_str = try std.fmt.allocPrint(allocator, "{d}", .{member.port});
        defer allocator.free(port_str);
        const color = prefix_colors[color_idx % prefix_colors.len];
        color_idx += 1;

        if (member.reservation) |*r| r.release();

        const child = spawnService(allocator, &.{ "uv", "run", "-m", "timbal.server.http", "--port", port_str, "--import_spec", member.config.fqn }, member_dir, member.name, .workforce, color, output_mutex, &services.reader_threads, env_ptr, log_filter) catch {
            std.debug.print("Error: failed to start {s}\n", .{member.name});
            continue;
        };
        const status = ServiceStatus{
            .name = member.name,
            .port = member.port,
            .pid = if (is_windows) {} else child.id,
            .started_at_ms = std.time.milliTimestamp(),
        };
        services.children.appendAssumeCapacity(child);
        services.statuses.appendAssumeCapacity(status);
    }

    if (has_ui) {
        const ui_dir = try std.fmt.allocPrint(allocator, "{s}/ui", .{abs_project_path});
        defer allocator.free(ui_dir);
        const ui_port_str = try std.fmt.allocPrint(allocator, "{d}", .{ui_port.?});
        defer allocator.free(ui_port_str);

        if (ui_reservation) |r| r.release();

        const child = spawnService(allocator, &.{ "bun", "run", "dev", "--port", ui_port_str }, ui_dir, "ui", .ui, "\x1b[1;36m", output_mutex, &services.reader_threads, ui_env.?, log_filter) catch {
            std.debug.print("Error: failed to start UI\n", .{});
            return error.ServiceStartFailed;
        };
        const status = ServiceStatus{
            .name = "ui",
            .port = ui_port.?,
            .pid = if (is_windows) {} else child.id,
            .started_at_ms = std.time.milliTimestamp(),
        };
        services.children.appendAssumeCapacity(child);
        services.statuses.appendAssumeCapacity(status);
    }

    if (has_api) {
        const api_dir = try std.fmt.allocPrint(allocator, "{s}/api", .{abs_project_path});
        defer allocator.free(api_dir);
        const api_port_str = try std.fmt.allocPrint(allocator, "{d}", .{api_port.?});
        defer allocator.free(api_port_str);

        if (api_reservation) |r| r.release();

        const child = spawnService(allocator, &.{ "bun", "run", "dev" }, api_dir, "api", .api, "\x1b[1;32m", output_mutex, &services.reader_threads, api_env.?, log_filter) catch {
            std.debug.print("Error: failed to start API\n", .{});
            return error.ServiceStartFailed;
        };
        const status = ServiceStatus{
            .name = "api",
            .port = api_port.?,
            .pid = if (is_windows) {} else child.id,
            .started_at_ms = std.time.milliTimestamp(),
        };
        services.children.appendAssumeCapacity(child);
        services.statuses.appendAssumeCapacity(status);
    }

    return services;
}

/// Check if a port can be bound on the wildcard interface.
/// Binding 0.0.0.0 catches both wildcard and 127.0.0.1 listeners — a process
/// already bound to 0.0.0.0:N or 127.0.0.1:N will make this fail with AddrInUse.
///
/// NOTE: This is racy by design (TOCTOU): the port is freed before any caller
/// can use it, so another process can grab it in between. For long-running
/// reservations use `tryReservePort` instead, which holds the listener open.
fn isPortFree(port: u16) bool {
    const addr = std.net.Address.initIp4(.{ 0, 0, 0, 0 }, port);
    var server = addr.listen(.{ .reuse_address = false }) catch return false;
    server.deinit();
    return true;
}

/// A reserved port — i.e. one we successfully bound and are *holding* in this
/// process so nothing else can steal it before we hand it off to a child.
///
/// We bind without SO_REUSEADDR so a second concurrent `timbal start` probing
/// the same port hits EADDRINUSE immediately and rolls forward to the next.
/// Right before we spawn the child that wants this port, the caller invokes
/// `release()` to free the listener; the spawn arguments include `--port N`
/// so the child re-binds the same number. The race window collapses from
/// "however long uv sync + bun install take" (~10s) to microseconds.
const PortReservation = struct {
    port: u16,
    server: std.net.Server,
    released: bool = false,

    fn release(self: *PortReservation) void {
        if (!self.released) {
            self.released = true;
            self.server.deinit();
        }
    }
};

/// Try to bind and hold a specific port. Returns null on EADDRINUSE/EACCES/etc.
fn tryReservePort(port: u16) ?PortReservation {
    const addr = std.net.Address.initIp4(.{ 0, 0, 0, 0 }, port);
    const server = addr.listen(.{ .reuse_address = false }) catch return null;
    return .{ .port = port, .server = server };
}

/// Reserve the next available port starting from `start`, scanning up to
/// `start + max_attempts`. Returns null if no port in that window is free.
/// The returned reservation must be released (or deinitialised on failure).
fn reserveAvailablePort(start: u16, max_attempts: u32) ?PortReservation {
    var attempt: u32 = 0;
    while (attempt < max_attempts) : (attempt += 1) {
        const candidate: u32 = @as(u32, start) + attempt;
        if (candidate > 65535) return null;
        const p: u16 = @intCast(candidate);
        if (tryReservePort(p)) |r| return r;
    }
    return null;
}

/// Best-effort lookup of the PID holding a TCP port, for friendlier error messages.
/// Falls back silently if the platform tool isn't available — never blocks the caller.
fn findPortHolderDescription(allocator: std.mem.Allocator, port: u16) ?[]u8 {
    const port_str = std.fmt.allocPrint(allocator, "{d}", .{port}) catch return null;
    defer allocator.free(port_str);

    if (comptime is_windows) {
        // `netstat -ano -p tcp` lists LISTENING ports and PIDs. Cheaper than tasklist.
        const port_pattern = std.fmt.allocPrint(allocator, ":{d} ", .{port}) catch return null;
        defer allocator.free(port_pattern);

        var child = std.process.Child.init(&.{ "netstat", "-ano", "-p", "tcp" }, allocator);
        child.stdout_behavior = .Pipe;
        child.stderr_behavior = .Ignore;
        child.spawn() catch return null;
        const out = child.stdout.?.reader().readAllAlloc(allocator, 1 * 1024 * 1024) catch null;
        defer if (out) |s| allocator.free(s);
        _ = child.wait() catch {};
        const haystack = out orelse return null;

        var lines = std.mem.splitScalar(u8, haystack, '\n');
        while (lines.next()) |line| {
            if (std.mem.indexOf(u8, line, port_pattern) == null) continue;
            if (std.mem.indexOf(u8, line, "LISTENING") == null) continue;
            return allocator.dupe(u8, std.mem.trim(u8, line, " \t\r")) catch null;
        }
        return null;
    } else {
        // `lsof -nP -iTCP:<port> -sTCP:LISTEN -t` prints just PIDs; -t makes it scriptable.
        const arg = std.fmt.allocPrint(allocator, "-iTCP:{d}", .{port}) catch return null;
        defer allocator.free(arg);

        var child = std.process.Child.init(&.{ "lsof", "-nP", arg, "-sTCP:LISTEN", "-t" }, allocator);
        child.stdout_behavior = .Pipe;
        child.stderr_behavior = .Ignore;
        child.spawn() catch return null;
        const out = child.stdout.?.reader().readAllAlloc(allocator, 64 * 1024) catch null;
        defer if (out) |s| allocator.free(s);
        _ = child.wait() catch {};
        const raw = out orelse return null;

        const trimmed = std.mem.trim(u8, raw, " \t\r\n");
        if (trimmed.len == 0) return null;
        var first_pid_iter = std.mem.splitScalar(u8, trimmed, '\n');
        const pid_str = first_pid_iter.next() orelse return null;
        const pid_trim = std.mem.trim(u8, pid_str, " \t\r");
        if (pid_trim.len == 0) return null;
        return std.fmt.allocPrint(allocator, "PID {s}", .{pid_trim}) catch null;
    }
}

fn openBrowserUrl(allocator: std.mem.Allocator, url: []const u8) void {
    var argv = std.ArrayList([]const u8).init(allocator);
    defer argv.deinit();

    if (builtin.os.tag == .macos) {
        argv.append("open") catch return;
        argv.append(url) catch return;
    } else if (builtin.os.tag == .windows) {
        argv.append("cmd") catch return;
        argv.append("/c") catch return;
        argv.append("start") catch return;
        argv.append("") catch return;
        argv.append(url) catch return;
    } else {
        argv.append("xdg-open") catch return;
        argv.append(url) catch return;
    }

    var child = std.process.Child.init(argv.items, allocator);
    child.stdin_behavior = .Ignore;
    child.stdout_behavior = .Ignore;
    child.stderr_behavior = .Ignore;
    child.spawn() catch return;
    _ = child.wait() catch {};
}

const BrowserContext = struct {
    allocator: std.mem.Allocator,
    url: []u8,
};

fn openBrowserThread(ctx: BrowserContext) void {
    defer ctx.allocator.free(ctx.url);
    std.time.sleep(2 * std.time.ns_per_s);
    openBrowserUrl(ctx.allocator, ctx.url);
}

fn preferredBrowserUrl(allocator: std.mem.Allocator, has_ui: bool, ui_port: ?u16, has_api: bool, api_port: ?u16) ?[]u8 {
    return if (has_ui)
        std.fmt.allocPrint(allocator, "http://localhost:{d}", .{ui_port.?}) catch null
    else if (has_api)
        std.fmt.allocPrint(allocator, "http://localhost:{d}", .{api_port.?}) catch null
    else
        null;
}

fn printCommandHelp(mutex: *std.Thread.Mutex) void {
    const stdout = std.io.getStdOut().writer();
    mutex.lock();
    defer mutex.unlock();

    stdout.writeAll("\n") catch {};
    stdout.print("{s}Available commands:{s}\n", .{ Color.bold, Color.reset }) catch {};
    stdout.print("  {s}r{s}  restart all services\n", .{ Color.bold_cyan, Color.reset }) catch {};
    stdout.print("  {s}s{s}  show service status\n", .{ Color.bold_cyan, Color.reset }) catch {};
    stdout.print("  {s}o{s}  open the app in your browser\n", .{ Color.bold_cyan, Color.reset }) catch {};
    stdout.print("  {s}u{s}  print the UI URL\n", .{ Color.bold_cyan, Color.reset }) catch {};
    stdout.print("  {s}a{s}  print the API URL\n", .{ Color.bold_cyan, Color.reset }) catch {};
    stdout.print("  {s}w{s}  print workforce service URLs\n", .{ Color.bold_cyan, Color.reset }) catch {};
    stdout.print("  {s}f <target>{s}  focus logs: all, ui, api, workforce, or member name\n", .{ Color.bold_cyan, Color.reset }) catch {};
    stdout.print("  {s}m <target>{s}  toggle mute for logs: all, ui, api, workforce, or member name\n", .{ Color.bold_cyan, Color.reset }) catch {};
    stdout.print("  {s}q{s}  stop services and quit\n", .{ Color.bold_cyan, Color.reset }) catch {};
}

fn isWorkforceMemberName(target: []const u8, members: []const WorkforceMember) bool {
    for (members) |member| {
        if (std.mem.eql(u8, target, member.name)) return true;
    }
    return false;
}

fn isKnownLogTarget(target: []const u8, members: []const WorkforceMember, has_ui: bool, has_api: bool) bool {
    if (std.mem.eql(u8, target, "all")) return true;
    if (std.mem.eql(u8, target, "workforce")) return members.len > 0;
    if (isWorkforceMemberName(target, members)) return true;
    if (std.mem.eql(u8, target, "ui")) return has_ui;
    if (std.mem.eql(u8, target, "api")) return has_api;
    return false;
}

fn printUnknownLogTarget(target: []const u8, mutex: *std.Thread.Mutex) void {
    const stdout = std.io.getStdOut().writer();
    mutex.lock();
    defer mutex.unlock();

    stdout.print("\n{s}Unknown log target `{s}`. Use all, ui, api, workforce, or a workforce member name.{s}\n", .{ Color.dim, target, Color.reset }) catch {};
}

fn handleFocusCommand(
    target: []const u8,
    members: []const WorkforceMember,
    has_ui: bool,
    has_api: bool,
    log_filter: *LogFilterState,
    mutex: *std.Thread.Mutex,
) void {
    const stdout = std.io.getStdOut().writer();
    if (target.len == 0) {
        printUnknownLogTarget(target, mutex);
        return;
    }
    if (!isKnownLogTarget(target, members, has_ui, has_api)) {
        printUnknownLogTarget(target, mutex);
        return;
    }

    log_filter.setFocus(target);

    mutex.lock();
    defer mutex.unlock();
    if (std.mem.eql(u8, target, "all")) {
        stdout.print("\n{s}Showing all non-muted logs.{s}\n", .{ Color.dim, Color.reset }) catch {};
    } else {
        stdout.print("\n{s}Showing only {s} logs. Type `f all` to restore all logs.{s}\n", .{ Color.dim, target, Color.reset }) catch {};
    }
}

fn handleMuteCommand(
    target: []const u8,
    members: []const WorkforceMember,
    has_ui: bool,
    has_api: bool,
    log_filter: *LogFilterState,
    mutex: *std.Thread.Mutex,
) void {
    const stdout = std.io.getStdOut().writer();
    if (target.len == 0) {
        printUnknownLogTarget(target, mutex);
        return;
    }
    if (!isKnownLogTarget(target, members, has_ui, has_api)) {
        printUnknownLogTarget(target, mutex);
        return;
    }

    const muted = log_filter.toggleMute(target, isWorkforceMemberName(target, members)) catch {
        mutex.lock();
        stdout.print("\n{s}Could not update log mute state for `{s}`.{s}\n", .{ Color.dim, target, Color.reset }) catch {};
        mutex.unlock();
        return;
    };

    mutex.lock();
    defer mutex.unlock();
    if (muted) {
        stdout.print("\n{s}Muted {s} logs. Type `m {s}` again to unmute.{s}\n", .{ Color.dim, target, target, Color.reset }) catch {};
    } else {
        stdout.print("\n{s}Unmuted {s} logs.{s}\n", .{ Color.dim, target, Color.reset }) catch {};
    }
}

fn printServiceStatus(services: ?*RunningServices, mutex: *std.Thread.Mutex) void {
    const stdout = std.io.getStdOut().writer();
    mutex.lock();
    defer mutex.unlock();

    stdout.writeAll("\n") catch {};
    stdout.print("{s}Service status:{s}\n", .{ Color.bold, Color.reset }) catch {};

    const running = services orelse {
        stdout.print("  {s}No services running{s}\n", .{ Color.dim, Color.reset }) catch {};
        return;
    };

    const now_ms = std.time.milliTimestamp();
    for (running.statuses.items) |status| {
        const uptime_s: i64 = @divTrunc(now_ms - status.started_at_ms, 1000);
        if (comptime is_windows) {
            if (status.port) |p| {
                stdout.print("  {s}✓{s} {s}  port={d} uptime={d}s\n", .{ Color.bold_green, Color.reset, status.name, p, uptime_s }) catch {};
            } else {
                stdout.print("  {s}✓{s} {s}  uptime={d}s\n", .{ Color.bold_green, Color.reset, status.name, uptime_s }) catch {};
            }
        } else {
            if (status.port) |p| {
                stdout.print("  {s}✓{s} {s}  pid={d} port={d} uptime={d}s\n", .{ Color.bold_green, Color.reset, status.name, status.pid, p, uptime_s }) catch {};
            } else {
                stdout.print("  {s}✓{s} {s}  pid={d} uptime={d}s\n", .{ Color.bold_green, Color.reset, status.name, status.pid, uptime_s }) catch {};
            }
        }
    }
}

fn printProjectUrls(
    members: []const WorkforceMember,
    has_ui: bool,
    ui_port: ?u16,
    has_api: bool,
    api_port: ?u16,
    action: CommandAction,
    mutex: *std.Thread.Mutex,
) void {
    const stdout = std.io.getStdOut().writer();
    mutex.lock();
    defer mutex.unlock();

    stdout.writeAll("\n") catch {};

    if ((action == .urls or action == .ui_url) and has_ui) {
        stdout.print("  UI   {s}http://localhost:{d}{s}\n", .{ Color.bold_cyan, ui_port.?, Color.reset }) catch {};
    } else if (action == .ui_url) {
        stdout.print("  {s}UI not found{s}\n", .{ Color.dim, Color.reset }) catch {};
    }

    if ((action == .urls or action == .api_url) and has_api) {
        stdout.print("  API  {s}http://localhost:{d}{s}\n", .{ Color.bold_cyan, api_port.?, Color.reset }) catch {};
    } else if (action == .api_url) {
        stdout.print("  {s}API not found{s}\n", .{ Color.dim, Color.reset }) catch {};
    }

    if (action == .urls or action == .workforce_urls) {
        if (members.len == 0) {
            stdout.print("  {s}No workforce found{s}\n", .{ Color.dim, Color.reset }) catch {};
        } else {
            for (members) |member| {
                stdout.print("  {s}  {s}http://localhost:{d}{s}\n", .{ member.name, Color.bold_cyan, member.port, Color.reset }) catch {};
            }
        }
    }
}

pub fn run(allocator: std.mem.Allocator, args: []const []const u8) !void {
    const orig_cp = enableWindowsConsole();
    defer restoreWindowsConsole(orig_cp);

    const stdout = std.io.getStdOut().writer();
    const stderr_writer = std.io.getStdErr().writer();
    g_interrupted.store(false, .seq_cst);

    const base_port: u16 = 4455;
    const default_ui_port: u16 = 3737;
    const default_api_port: u16 = 3000;

    // Parse arguments using the dedicated parser.
    const parsed = try parseStartArgs(allocator, args);
    switch (parsed) {
        .help => {
            try printUsage();
            return;
        },
        .err => |msg| {
            defer allocator.free(msg);
            try printUsageWithError(msg);
            return;
        },
        .options => {},
    }
    var opts = parsed.options;
    defer opts.deinit(allocator);

    // Resolve profile: --profile flag > TIMBAL_PROFILE env var > "default".
    const env_profile = std.process.getEnvVarOwned(allocator, "TIMBAL_PROFILE") catch |err| blk: {
        if (err == error.EnvironmentVariableNotFound) break :blk null;
        return err;
    };
    defer if (env_profile) |p| allocator.free(p);
    const profile: []const u8 = opts.profile orelse (env_profile orelse "default");

    // Load credentials and config for the profile.
    const credentials_path = try getCredentialsPath(allocator);
    defer allocator.free(credentials_path);
    const credentials_content = std.fs.cwd().readFileAlloc(allocator, credentials_path, 1024 * 1024) catch |err| {
        if (err == error.FileNotFound) {
            try stderr_writer.print("Error: Timbal is not configured. Run '{s}timbal configure{s}' first.\n", .{ Color.bold_cyan, Color.reset });
            return;
        }
        return err;
    };
    defer allocator.free(credentials_content);

    const config_path = try getConfigPath(allocator);
    defer allocator.free(config_path);
    const config_content = std.fs.cwd().readFileAlloc(allocator, config_path, 1024 * 1024) catch |err| {
        if (err == error.FileNotFound) {
            try stderr_writer.print("Error: Timbal is not configured. Run '{s}timbal configure{s}' first.\n", .{ Color.bold_cyan, Color.reset });
            return;
        }
        return err;
    };
    defer allocator.free(config_content);

    const api_key = readValue(credentials_content, profile, "api_key") orelse {
        try stderr_writer.print("Error: No API key found for profile '{s}'. Run '{s}timbal configure --profile {s}{s}' to set it up.\n", .{ profile, Color.bold_cyan, profile, Color.reset });
        return;
    };
    const org_id = readValue(config_content, profile, "org") orelse {
        try stderr_writer.print("Error: No organization ID found for profile '{s}'. Run '{s}timbal configure --profile {s}{s}' to set it up.\n", .{ profile, Color.bold_cyan, profile, Color.reset });
        return;
    };
    const base_url = readValue(config_content, profile, "base_url") orelse "https://api.timbal.ai";
    const api_host = stripProtocol(base_url);

    // Check that required tools are installed.
    var uv_check = std.process.Child.init(&.{ "uv", "--version" }, allocator);
    uv_check.stdout_behavior = .Ignore;
    uv_check.stderr_behavior = .Ignore;
    if (uv_check.spawn()) |_| {
        _ = uv_check.wait() catch {};
    } else |_| {
        try stderr_writer.writeAll("Error: 'uv' is not installed.\n" ++
            "Install it from: https://docs.astral.sh/uv/getting-started/installation/\n");
        return;
    }

    var bun_check = std.process.Child.init(&.{ "bun", "--version" }, allocator);
    bun_check.stdout_behavior = .Ignore;
    bun_check.stderr_behavior = .Ignore;
    if (bun_check.spawn()) |_| {
        _ = bun_check.wait() catch {};
    } else |_| {
        try stderr_writer.writeAll("Error: 'bun' is not installed.\n" ++
            "Install it from: https://bun.sh/docs/installation\n");
        return;
    }

    // Open the project directory (provided path or cwd).
    var project_dir = if (opts.project_path) |path|
        fs.cwd().openDir(path, .{ .iterate = true }) catch {
            std.debug.print("Error: could not open directory '{s}'\n", .{path});
            return;
        }
    else
        fs.cwd();

    defer if (opts.project_path != null) project_dir.close();

    // Resolve the absolute project path for child process cwd.
    const abs_project_path = try project_dir.realpathAlloc(allocator, ".");
    defer allocator.free(abs_project_path);

    // Detect UI directory.
    const has_ui = if (project_dir.statFile("ui/package.json")) |_| true else |_| false;
    // Detect API directory.
    const has_api = if (project_dir.statFile("api/package.json")) |_| true else |_| false;

    // Helper for resolving a port: explicit → fail fast, default → auto-fallback.
    // Returns a *held* PortReservation. The caller owns it and must release it
    // (either explicitly before spawning a child on the same port, or via the
    // defer cleanup on early exit). Two concurrent `timbal start` invocations
    // probing the same default port can no longer both win — the second one's
    // bind fails immediately and rolls forward to the next port.
    const resolvePortReservation = struct {
        fn call(
            label: []const u8,
            requested: ?u16,
            default_base: u16,
            alloc: std.mem.Allocator,
            stderr_w: anytype,
        ) !PortReservation {
            if (requested) |p| {
                return tryReservePort(p) orelse {
                    const holder = findPortHolderDescription(alloc, p);
                    defer if (holder) |h| alloc.free(h);
                    if (holder) |h| {
                        try stderr_w.print(
                            "Error: requested {s} port {d} is already in use ({s}).\n",
                            .{ label, p, h },
                        );
                    } else {
                        try stderr_w.print(
                            "Error: requested {s} port {d} is already in use.\n",
                            .{ label, p },
                        );
                    }
                    return error.PortInUse;
                };
            }
            return reserveAvailablePort(default_base, 200) orelse {
                try stderr_w.print(
                    "Error: no free port for {s} starting from {d}.\n",
                    .{ label, default_base },
                );
                return error.PortInUse;
            };
        }
    }.call;

    // resolvePortReservation already prints a friendly error before returning
    // error.PortInUse, so we silently exit here — no need to bubble the trace
    // up to main.
    var ui_reservation: ?PortReservation = if (has_ui)
        (resolvePortReservation("UI", opts.ports.ui, default_ui_port, allocator, stderr_writer) catch return)
    else
        null;
    defer if (ui_reservation) |*r| r.release();

    var api_reservation: ?PortReservation = if (has_api)
        (resolvePortReservation("API", opts.ports.api, default_api_port, allocator, stderr_writer) catch return)
    else
        null;
    defer if (api_reservation) |*r| r.release();

    const ui_port: ?u16 = if (ui_reservation) |r| r.port else null;
    const api_port: ?u16 = if (api_reservation) |r| r.port else null;

    // Discover workforce members (subdirectories of workforce/ that contain timbal.yaml).
    var members = std.ArrayList(WorkforceMember).init(allocator);
    defer {
        // The list owns each member's `name` allocation, the heap-backed strings
        // inside `config`, and any still-held PortReservation. Without this
        // loop, any early return between here and the end of run() leaks them
        // (and leaves the OS port reserved until process exit).
        for (members.items) |*member| {
            if (member.reservation) |*r| r.release();
            allocator.free(member.name);
            member.config.deinit(allocator);
        }
        members.deinit();
    }

    // Phase 1: enumerate every member without reserving a port. Pure discovery.
    if (project_dir.openDir("workforce", .{ .iterate = true })) |*workforce_dir_ptr| {
        var workforce_dir = workforce_dir_ptr.*;
        defer workforce_dir.close();

        var iter = workforce_dir.iterate();
        while (try iter.next()) |entry| {
            if (entry.kind != .directory) continue;

            const yaml_path = try std.fmt.allocPrint(allocator, "workforce/{s}/timbal.yaml", .{entry.name});
            defer allocator.free(yaml_path);

            if (project_dir.statFile(yaml_path)) |_| {
                const content = project_dir.readFileAlloc(allocator, yaml_path, 64 * 1024) catch continue;
                defer allocator.free(content);
                var config = utils.parseTimbalYaml(allocator, content) orelse {
                    std.debug.print("Warning: invalid timbal.yaml in {s}, skipping\n", .{yaml_path});
                    continue;
                };
                // Until members.append succeeds, name and config are local; if
                // either dupe/append errors we must free them. After a
                // successful append the WorkforceMember owns both and the
                // global members-defer handles cleanup.
                //
                // The config errdefer must be registered BEFORE the next
                // fallible op (`try allocator.dupe`); registering it after
                // would leak config on a dupe OOM.
                errdefer config.deinit(allocator);
                const name = try allocator.dupe(u8, entry.name);
                errdefer allocator.free(name);
                try members.append(.{
                    .name = name,
                    .config = config,
                    .port = 0,
                    .reservation = null,
                });
            } else |_| {}
        }
    } else |_| {}

    // Phase 2: reserve all ports in two passes (explicit first, auto second)
    // so directory iteration order can't cause an auto-allocation to steal a
    // port that another member explicitly requested.
    reserveMemberPorts(
        allocator,
        members.items,
        &opts.ports.members,
        opts.ports.workforce_base orelse base_port,
        stderr_writer,
    ) catch return;

    // Warn (but don't fail) if the user passed --port NAME=PORT for a member
    // that doesn't exist — most likely a typo.
    {
        var it = opts.ports.members.keyIterator();
        while (it.next()) |key| {
            var found = false;
            for (members.items) |m| {
                if (std.mem.eql(u8, m.name, key.*)) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                try stderr_writer.print(
                    "Warning: --port {s}=... but no workforce member named '{s}' was found.\n",
                    .{ key.*, key.* },
                );
            }
        }
    }

    // Print discovered project layout.
    try stdout.print("\n{s}Discovered project layout:{s}\n\n", .{ Color.bold, Color.reset });

    if (has_ui) {
        try stdout.print("  {s}✓{s} UI   →  {s}:{d}{s}\n", .{ Color.bold_green, Color.reset, Color.dim, ui_port.?, Color.reset });
    } else {
        try stdout.print("  {s}-{s} UI   {s}not found{s}\n", .{ Color.dim, Color.reset, Color.dim, Color.reset });
    }

    if (has_api) {
        try stdout.print("  {s}✓{s} API  →  {s}:{d}{s}\n", .{ Color.bold_green, Color.reset, Color.dim, api_port.?, Color.reset });
    } else {
        try stdout.print("  {s}-{s} API  {s}not found{s}\n", .{ Color.dim, Color.reset, Color.dim, Color.reset });
    }

    if (members.items.len == 0) {
        try stdout.print("\n  {s}No workforce found/{s}\n", .{ Color.dim, Color.reset });
    } else {
        try stdout.print("\n  {s}Workforce:{s}\n", .{ Color.bold, Color.reset });
        for (members.items) |member| {
            try stdout.print("  {s}✓{s} {s}  →  {s}:{d}{s}\n", .{
                Color.bold_green,
                Color.reset,
                member.name,
                Color.dim,
                member.port,
                Color.reset,
            });
        }
    }

    // Install dependencies.
    var install_failed = false;

    if (has_ui and !install_failed) {
        try stdout.print("\n{s}Installing UI dependencies...{s}\n\n", .{ Color.bold, Color.reset });
        const ui_dir = try std.fmt.allocPrint(allocator, "{s}/ui", .{abs_project_path});
        defer allocator.free(ui_dir);
        if (!runCommand(allocator, &.{ "bun", "install" }, ui_dir)) {
            std.debug.print("\nError: 'bun install' failed in ui/\n", .{});
            install_failed = true;
        }
    }

    if (has_api and !install_failed) {
        try stdout.print("\n{s}Installing API dependencies...{s}\n\n", .{ Color.bold, Color.reset });
        const api_dir = try std.fmt.allocPrint(allocator, "{s}/api", .{abs_project_path});
        defer allocator.free(api_dir);
        if (!runCommand(allocator, &.{ "bun", "install" }, api_dir)) {
            std.debug.print("\nError: 'bun install' failed in api/\n", .{});
            install_failed = true;
        }
    }

    // Install workforce dependencies.
    for (members.items) |member| {
        try stdout.print("\n{s}Syncing {s}...{s}\n\n", .{ Color.bold, member.name, Color.reset });
        const member_dir = try std.fmt.allocPrint(allocator, "{s}/workforce/{s}", .{ abs_project_path, member.name });
        defer allocator.free(member_dir);

        // Create virtualenv if it doesn't already exist.
        const venv_path = try std.fmt.allocPrint(allocator, "{s}/.venv", .{member_dir});
        defer allocator.free(venv_path);

        const venv_exists = if (fs.accessAbsolute(venv_path, .{})) |_| true else |_| false;
        if (!venv_exists) {
            if (!runCommand(allocator, &.{ "uv", "venv" }, member_dir)) {
                std.debug.print("\nError: 'uv venv' failed in workforce/{s}\n", .{member.name});
                break;
            }
        }

        // Sync dependencies.
        if (!runCommand(allocator, &.{ "uv", "sync" }, member_dir)) {
            std.debug.print("\nError: 'uv sync' failed in workforce/{s}\n", .{member.name});
            break;
        }
    }

    // Save terminal state and install SIGINT handler (POSIX only).
    // tcgetattr() returns an "unexpected errno" on macOS when stdin isn't a tty
    // (e.g. piped to `head`), and Zig's stdlib dumps a stack trace before we
    // can swallow the error. Guard with isTty() first.
    if (comptime !is_windows) {
        const stdin_file = std.io.getStdIn();
        if (stdin_file.isTty()) {
            g_stdin_fd = stdin_file.handle;
            if (std.posix.tcgetattr(g_stdin_fd)) |termios| {
                g_original_termios = termios;
                g_termios_saved = true;
            } else |_| {}
        }
        const sa = std.posix.Sigaction{
            .handler = .{ .handler = sigintHandler },
            .mask = std.posix.empty_sigset,
            .flags = 0,
        };
        std.posix.sigaction(std.posix.SIG.INT, &sa, null);
    }
    defer restoreTermios();

    var output_mutex = std.Thread.Mutex{};
    const command_state = try std.heap.page_allocator.create(CommandState);
    command_state.* = .{};
    var command_input_stop = std.atomic.Value(bool).init(false);
    var command_thread: ?std.Thread = null;
    var command_state_owned = true;
    defer {
        command_input_stop.store(true, .seq_cst);
        if (command_thread) |thread| {
            if (comptime !is_windows) {
                thread.join();
            } else {
                // Windows console reads cannot be interrupted here. Keep the
                // state process-lived if the detached reader wakes after run().
                thread.detach();
                command_state_owned = false;
            }
        }
        if (command_state_owned) std.heap.page_allocator.destroy(command_state);
    }
    var log_filter = LogFilterState.init(allocator);
    defer log_filter.deinit();

    // ---------------------------------------------------------------------
    // Layered env composition.
    // Precedence (low → high):
    //   1. Soft built-ins (FORCE_COLOR, TIMBAL_LOG_*, TIMBAL_API_KEY, etc.)
    //   2. Shell env (with PORT scrubbed so it doesn't leak into workforce)
    //   3. Auto-loaded .env (project root for global; workforce/<member>/.env for members)
    //   4. --env-file values (in order, scoped or global)
    //   5. --env values (in order, scoped or global)
    //   6. Hard runtime info (TIMBAL_START_*, TIMBAL_WORKFORCE, PORT for ui/api)
    // ---------------------------------------------------------------------

    // Pre-compute the workforce mapping string (id:port,id:port,...).
    var workforce_value: ?[]u8 = null;
    defer if (workforce_value) |s| allocator.free(s);
    {
        var workforce_buf = std.ArrayList(u8).init(allocator);
        defer workforce_buf.deinit();
        for (members.items, 0..) |member, idx| {
            if (idx > 0) try workforce_buf.append(',');
            try workforce_buf.writer().print("{s}:{d}", .{ member.config.id, member.port });
        }
        if (workforce_buf.items.len > 0) {
            workforce_value = try workforce_buf.toOwnedSlice();
        }
    }

    const api_port_str: ?[]u8 = if (api_port) |p| try std.fmt.allocPrint(allocator, "{d}", .{p}) else null;
    defer if (api_port_str) |s| allocator.free(s);
    const ui_port_str: ?[]u8 = if (ui_port) |p| try std.fmt.allocPrint(allocator, "{d}", .{p}) else null;
    defer if (ui_port_str) |s| allocator.free(s);

    // Auto-loaded .env: project root applies to ALL services; member-specific
    // .env applies to that workforce member only.
    const root_env_path = try std.fmt.allocPrint(allocator, "{s}{s}.env", .{ abs_project_path, sep });
    defer allocator.free(root_env_path);

    var root_env_entries = blk: {
        const content = std.fs.cwd().readFileAlloc(allocator, root_env_path, 1024 * 1024) catch |err| switch (err) {
            error.FileNotFound => break :blk null,
            else => return err,
        };
        defer allocator.free(content);
        break :blk try parseEnvFile(allocator, content);
    };
    defer if (root_env_entries) |*e| freeParsedEnvList(allocator, e);

    // Per-member auto-loaded .env entries (parallel to members.items).
    //
    // `allocator.alloc` returns uninitialised memory. The defer reads EVERY
    // element, so if the fallible population loop below errors at index k,
    // elements [k+1..] would still contain undefined bytes — reading them
    // as `?std.ArrayList(...)` would dereference garbage (UB in release,
    // safety panic in debug). Initialise everything to null *before* any
    // fallible operation runs.
    var member_env_entries = try allocator.alloc(?std.ArrayList(ParsedEnv), members.items.len);
    for (member_env_entries) |*slot| slot.* = null;
    defer {
        for (member_env_entries) |*maybe| {
            if (maybe.*) |*e| freeParsedEnvList(allocator, e);
        }
        allocator.free(member_env_entries);
    }
    for (members.items, 0..) |member, idx| {
        const path = try std.fmt.allocPrint(allocator, "{s}{s}workforce{s}{s}{s}.env", .{ abs_project_path, sep, sep, member.name, sep });
        defer allocator.free(path);
        const content = std.fs.cwd().readFileAlloc(allocator, path, 1024 * 1024) catch |err| switch (err) {
            error.FileNotFound => continue,
            else => return err,
        };
        defer allocator.free(content);
        member_env_entries[idx] = try parseEnvFile(allocator, content);
    }

    // Build the list of known scopes for resolving --env / --env-file prefixes.
    var scope_names = std.ArrayList([]const u8).init(allocator);
    defer scope_names.deinit();
    if (has_ui) try scope_names.append("ui");
    if (has_api) try scope_names.append("api");
    for (members.items) |m| try scope_names.append(m.name);

    // Pre-parse --env-file paths into scoped lists, loading each file once.
    // Index 0 is the global bucket; subsequent indices correspond to
    // scope_names entries (ui, api, members…).
    const total_buckets = scope_names.items.len + 1;
    var env_file_buckets = try allocator.alloc(std.ArrayList(ParsedEnv), total_buckets);
    defer {
        for (env_file_buckets) |*b| freeParsedEnvList(allocator, b);
        allocator.free(env_file_buckets);
    }
    for (env_file_buckets) |*b| b.* = std.ArrayList(ParsedEnv).init(allocator);

    for (opts.raw_env_files.items) |raw| {
        const split = detachScope(raw.raw, scope_names.items);
        if (split.rest.len == 0) {
            try stderr_writer.print("Error: --env-file '{s}' is missing a path.\n", .{raw.raw});
            return;
        }
        const content = std.fs.cwd().readFileAlloc(allocator, split.rest, 1024 * 1024) catch |err| {
            try stderr_writer.print("Error: failed to read --env-file '{s}': {s}\n", .{ split.rest, @errorName(err) });
            return;
        };
        defer allocator.free(content);
        var parsed_entries = try parseEnvFile(allocator, content);
        // freeParsedEnvList (not plain deinit) is required: if the move below
        // OOMs on ensureUnusedCapacity, parsed_entries still owns every
        // string. After a successful move, clearRetainingCapacity leaves
        // items empty and this becomes a no-op string loop plus a container
        // free.
        defer freeParsedEnvList(allocator, &parsed_entries);

        const bucket_idx: usize = blk: {
            const s = split.scope orelse break :blk 0;
            for (scope_names.items, 0..) |name, idx| {
                if (std.mem.eql(u8, name, s)) break :blk idx + 1;
            }
            break :blk 0;
        };
        try moveParsedEnvIntoBucket(&parsed_entries, &env_file_buckets[bucket_idx]);
    }

    // Pre-parse --env values into scoped lists, validating KEY=VALUE.
    var env_arg_buckets = try allocator.alloc(std.ArrayList(ParsedEnv), total_buckets);
    defer {
        for (env_arg_buckets) |*b| freeParsedEnvList(allocator, b);
        allocator.free(env_arg_buckets);
    }
    for (env_arg_buckets) |*b| b.* = std.ArrayList(ParsedEnv).init(allocator);

    for (opts.raw_env_args.items) |raw| {
        const split = detachScope(raw.raw, scope_names.items);
        const kv = parseKeyValue(split.rest) orelse {
            try stderr_writer.print(
                "Error: --env '{s}' must be KEY=VALUE (KEY: letters/digits/underscore, not starting with a digit).\n",
                .{raw.raw},
            );
            return;
        };
        const bucket_idx: usize = blk: {
            const s = split.scope orelse break :blk 0;
            for (scope_names.items, 0..) |name, idx| {
                if (std.mem.eql(u8, name, s)) break :blk idx + 1;
            }
            break :blk 0;
        };
        try appendDupedKv(allocator, &env_arg_buckets[bucket_idx], kv.key, kv.value);
    }

    // Soft built-ins shared by every service.
    const soft_builtins = [_][2][]const u8{
        .{ "FORCE_COLOR", "1" },
        .{ "TIMBAL_LOG_EVENTS", "START,OUTPUT" },
        .{ "TIMBAL_LOG_FORMAT", "dev" },
        .{ "TIMBAL_DELTA_EVENTS", "true" },
        .{ "TIMBAL_API_KEY", api_key },
        .{ "TIMBAL_ORG_ID", org_id },
        .{ "TIMBAL_API_HOST", api_host },
    };

    // Shell env, with PORT scrubbed so the user's shell PORT does not leak
    // into workforce members (or override our explicit assignments later).
    var shell_env = try std.process.getEnvMap(allocator);
    defer shell_env.deinit();
    shell_env.remove("PORT");

    // Positional bucket assignment — see ScopeBuckets for why we don't look up
    // by name. The storage path for --env/--env-file uses first-match-by-name
    // against `scope_names`, which means a `--env ui:KEY=VAL` always routes to
    // the UI service (scope_names[0] when has_ui), never to a workforce member
    // that happens to be named "ui". Warn the user so the shadowing is visible.
    for (members.items) |m| {
        if (std.mem.eql(u8, m.name, "ui") or std.mem.eql(u8, m.name, "api")) {
            try stderr_writer.print(
                "Warning: workforce member '{s}' shadows the reserved '{s}' scope; '--env {s}:KEY=VAL' will route to the {s} service, not the member.\n",
                .{ m.name, m.name, m.name, m.name },
            );
        }
    }

    var scope_buckets = try assignScopeBuckets(allocator, has_ui, has_api, members.items.len);
    defer scope_buckets.deinit(allocator);
    const ui_bucket = scope_buckets.ui_bucket;
    const api_bucket = scope_buckets.api_bucket;
    const member_buckets = scope_buckets.member_buckets;

    // -- Build per-service envs --

    // Workforce members.
    var member_envs = try allocator.alloc(std.process.EnvMap, members.items.len);
    var built_member_envs: usize = 0;
    defer {
        for (member_envs[0..built_member_envs]) |*e| e.deinit();
        allocator.free(member_envs);
    }
    for (members.items, 0..) |member, idx| {
        var hard = std.ArrayList([2][]const u8).init(allocator);
        defer hard.deinit();
        const member_port_str = try std.fmt.allocPrint(allocator, "{d}", .{member.port});
        defer allocator.free(member_port_str);
        try hard.append(.{ "PORT", member_port_str });
        if (workforce_value) |v| {
            try hard.append(.{ "TIMBAL_START_WORKFORCE", v });
            try hard.append(.{ "TIMBAL_WORKFORCE", v });
        }
        if (api_port_str) |s| try hard.append(.{ "TIMBAL_START_API_PORT", s });
        if (ui_port_str) |s| try hard.append(.{ "TIMBAL_START_UI_PORT", s });

        const auto_scope_items: ?[]const ParsedEnv =
            if (member_env_entries[idx]) |*e| e.items else null;
        const auto_global_items: ?[]const ParsedEnv =
            if (root_env_entries) |*e| e.items else null;

        member_envs[idx] = try composeServiceEnv(
            allocator,
            &soft_builtins,
            &shell_env,
            auto_global_items,
            auto_scope_items,
            env_file_buckets[0].items,
            env_file_buckets[member_buckets[idx]].items,
            env_arg_buckets[0].items,
            env_arg_buckets[member_buckets[idx]].items,
            hard.items,
        );
        built_member_envs += 1;
    }

    // UI env.
    var ui_env: ?std.process.EnvMap = null;
    defer if (ui_env) |*e| e.deinit();
    if (has_ui) {
        var hard = std.ArrayList([2][]const u8).init(allocator);
        defer hard.deinit();
        try hard.append(.{ "PORT", ui_port_str.? });
        if (workforce_value) |v| {
            try hard.append(.{ "TIMBAL_START_WORKFORCE", v });
            try hard.append(.{ "TIMBAL_WORKFORCE", v });
        }
        if (api_port_str) |s| try hard.append(.{ "TIMBAL_START_API_PORT", s });
        if (ui_port_str) |s| try hard.append(.{ "TIMBAL_START_UI_PORT", s });
        const auto_global_items: ?[]const ParsedEnv =
            if (root_env_entries) |*e| e.items else null;
        ui_env = try composeServiceEnv(
            allocator,
            &soft_builtins,
            &shell_env,
            auto_global_items,
            null,
            env_file_buckets[0].items,
            env_file_buckets[ui_bucket].items,
            env_arg_buckets[0].items,
            env_arg_buckets[ui_bucket].items,
            hard.items,
        );
    }

    // API env.
    var api_env: ?std.process.EnvMap = null;
    defer if (api_env) |*e| e.deinit();
    if (has_api) {
        var hard = std.ArrayList([2][]const u8).init(allocator);
        defer hard.deinit();
        try hard.append(.{ "PORT", api_port_str.? });
        if (workforce_value) |v| {
            try hard.append(.{ "TIMBAL_START_WORKFORCE", v });
            try hard.append(.{ "TIMBAL_WORKFORCE", v });
        }
        if (api_port_str) |s| try hard.append(.{ "TIMBAL_START_API_PORT", s });
        if (ui_port_str) |s| try hard.append(.{ "TIMBAL_START_UI_PORT", s });
        const auto_global_items: ?[]const ParsedEnv =
            if (root_env_entries) |*e| e.items else null;
        api_env = try composeServiceEnv(
            allocator,
            &soft_builtins,
            &shell_env,
            auto_global_items,
            null,
            env_file_buckets[0].items,
            env_file_buckets[api_bucket].items,
            env_arg_buckets[0].items,
            env_arg_buckets[api_bucket].items,
            hard.items,
        );
    }

    // Build the parallel slice of pointers for startServices.
    var member_env_ptrs = try allocator.alloc(*const std.process.EnvMap, members.items.len);
    defer allocator.free(member_env_ptrs);
    for (member_envs[0..members.items.len], 0..) |*e, idx| member_env_ptrs[idx] = e;

    const command_input_enabled = std.io.getStdIn().isTty();
    if (command_input_enabled) {
        command_thread = std.Thread.spawn(.{}, commandInputFn, .{CommandInputCtx{
            .state = command_state,
            .stop = &command_input_stop,
        }}) catch null;
    }

    var services: ?RunningServices = try startServices(
        allocator,
        abs_project_path,
        members.items,
        member_env_ptrs,
        has_ui,
        ui_port,
        if (ui_env) |*e| e else null,
        if (ui_reservation) |*r| r else null,
        has_api,
        api_port,
        if (api_env) |*e| e else null,
        if (api_reservation) |*r| r else null,
        &output_mutex,
        &log_filter,
    );
    defer if (services) |*running| stopServices(running);

    if (command_input_enabled) {
        try stdout.print("\n{s}Press {s}h + enter{s} for commands, {s}r + enter{s} to restart, or Ctrl+C to stop.{s}\n", .{
            Color.dim,
            Color.bold_cyan,
            Color.dim,
            Color.bold_cyan,
            Color.dim,
            Color.reset,
        });
    }

    // Open browser after a short delay (UI takes priority over API).
    {
        const browser_url = preferredBrowserUrl(allocator, has_ui, ui_port, has_api, api_port);

        if (browser_url) |url| {
            const ctx = BrowserContext{ .allocator = allocator, .url = url };
            const browser_thread = std.Thread.spawn(.{}, openBrowserThread, .{ctx}) catch null;
            if (browser_thread) |t| t.detach();
        }
    }

    while (!g_interrupted.load(.seq_cst)) {
        const command = command_state.take();
        const target = command.target[0..command.target_len];
        switch (command.action) {
            .none => {},
            .help => printCommandHelp(&output_mutex),
            .status => {
                if (services) |*running| {
                    printServiceStatus(running, &output_mutex);
                } else {
                    printServiceStatus(null, &output_mutex);
                }
            },
            .open => {
                if (preferredBrowserUrl(allocator, has_ui, ui_port, has_api, api_port)) |url| {
                    defer allocator.free(url);
                    output_mutex.lock();
                    stdout.print("\n{s}Opening {s}{s}\n", .{ Color.dim, url, Color.reset }) catch {};
                    output_mutex.unlock();
                    openBrowserUrl(allocator, url);
                } else {
                    output_mutex.lock();
                    stdout.print("\n{s}No UI or API URL to open{s}\n", .{ Color.dim, Color.reset }) catch {};
                    output_mutex.unlock();
                }
            },
            .urls, .ui_url, .api_url, .workforce_urls => |action| {
                printProjectUrls(members.items, has_ui, ui_port, has_api, api_port, action, &output_mutex);
            },
            .focus_logs => {
                handleFocusCommand(target, members.items, has_ui, has_api, &log_filter, &output_mutex);
            },
            .toggle_mute => {
                handleMuteCommand(target, members.items, has_ui, has_api, &log_filter, &output_mutex);
            },
            .restart => {
                output_mutex.lock();
                stdout.print("\n{s}Restarting services...{s}\n", .{ Color.bold, Color.reset }) catch {};
                output_mutex.unlock();

                if (services) |*running| {
                    stopServices(running);
                    services = null;
                }
                // On restart we keep the original port numbers ("stick"
                // semantics). The reservations were already released at first
                // spawn; the running children hold those ports until we kill
                // them above. There's a tiny re-bind window between kill and
                // re-spawn — out of scope to harden further.
                services = try startServices(
                    allocator,
                    abs_project_path,
                    members.items,
                    member_env_ptrs,
                    has_ui,
                    ui_port,
                    if (ui_env) |*e| e else null,
                    null,
                    has_api,
                    api_port,
                    if (api_env) |*e| e else null,
                    null,
                    &output_mutex,
                    &log_filter,
                );
            },
            .quit => break,
        }

        std.time.sleep(150 * std.time.ns_per_ms);
    }

    if (services) |*running| {
        stopServices(running);
        services = null;
    }
    // Per-member allocations are freed by the `defer` declared with `members`.
}

test "log targets include workforce members named like reserved services" {
    const members = [_]WorkforceMember{
        .{ .name = "ui", .config = undefined, .port = 8001 },
        .{ .name = "api", .config = undefined, .port = 8002 },
    };

    try std.testing.expect(isKnownLogTarget("ui", &members, false, false));
    try std.testing.expect(isKnownLogTarget("api", &members, false, false));
    try std.testing.expect(isKnownLogTarget("ui", &members, true, false));
    try std.testing.expect(isKnownLogTarget("api", &members, false, true));
}

test "mute can target workforce member named ui" {
    var log_filter = LogFilterState.init(std.testing.allocator);
    defer log_filter.deinit();

    try std.testing.expect(try log_filter.toggleMute("ui", true));
    try std.testing.expect(!log_filter.shouldPrint("ui", .workforce));
    try std.testing.expect(log_filter.shouldPrint("other", .ui));

    try std.testing.expect(!try log_filter.toggleMute("ui", true));
    try std.testing.expect(log_filter.shouldPrint("ui", .workforce));
}

test "mute still targets ui service when no member has that name" {
    var log_filter = LogFilterState.init(std.testing.allocator);
    defer log_filter.deinit();

    try std.testing.expect(try log_filter.toggleMute("ui", false));
    try std.testing.expect(!log_filter.shouldPrint("ui", .ui));
    try std.testing.expect(log_filter.shouldPrint("ui", .workforce));
}

// ===========================================================================
// Port helpers
// ===========================================================================

test "parsePort accepts 1-65535 and rejects 0 / non-numeric / out-of-range" {
    try std.testing.expectEqual(@as(?u16, 3000), parsePort("3000"));
    try std.testing.expectEqual(@as(?u16, 1), parsePort("1"));
    try std.testing.expectEqual(@as(?u16, 65535), parsePort("65535"));
    try std.testing.expectEqual(@as(?u16, null), parsePort("0"));
    try std.testing.expectEqual(@as(?u16, null), parsePort("65536"));
    try std.testing.expectEqual(@as(?u16, null), parsePort(""));
    try std.testing.expectEqual(@as(?u16, null), parsePort("abc"));
    try std.testing.expectEqual(@as(?u16, null), parsePort("-1"));
    try std.testing.expectEqual(@as(?u16, null), parsePort("3000a"));
}

test "parsePortKv splits NAME=PORT and validates both sides" {
    {
        const kv = parsePortKv("ui=3000").?;
        try std.testing.expectEqualStrings("ui", kv.name);
        try std.testing.expectEqual(@as(u16, 3000), kv.port);
    }
    {
        const kv = parsePortKv("my-agent=4500").?;
        try std.testing.expectEqualStrings("my-agent", kv.name);
        try std.testing.expectEqual(@as(u16, 4500), kv.port);
    }
    try std.testing.expectEqual(@as(?PortKv, null), parsePortKv("noequals"));
    try std.testing.expectEqual(@as(?PortKv, null), parsePortKv("=3000"));
    try std.testing.expectEqual(@as(?PortKv, null), parsePortKv("ui="));
    try std.testing.expectEqual(@as(?PortKv, null), parsePortKv("ui=0"));
    try std.testing.expectEqual(@as(?PortKv, null), parsePortKv("ui=99999"));
}

test "reserveAvailablePort returns null when given an out-of-range window" {
    // start = 65535 + offset that wraps past u16 → null.
    // Note: comparing optional unions structurally via expectEqual breaks
    // because std.net.Address is an untagged union — use isNull check.
    const r = reserveAvailablePort(65535, 0);
    try std.testing.expect(r == null);
}

test "reserveAvailablePort skips a port that's actually in use" {
    // Hold a real listener on an arbitrary high port; verify reserveAvailablePort
    // starting at that port rolls forward to the next.
    const addr = std.net.Address.initIp4(.{ 127, 0, 0, 1 }, 0);
    var server = try addr.listen(.{ .reuse_address = false });
    defer server.deinit();

    const taken: u16 = server.listen_address.in.getPort();
    if (taken == 65535) return; // can't test the rollover case here

    var next = reserveAvailablePort(taken, 50) orelse return error.NoFreePort;
    defer next.release();
    try std.testing.expect(next.port != taken);
    try std.testing.expect(next.port > taken);
}

test "isPortFree returns false for a port we're currently holding" {
    const addr = std.net.Address.initIp4(.{ 127, 0, 0, 1 }, 0);
    var server = try addr.listen(.{ .reuse_address = false });
    defer server.deinit();
    const taken: u16 = server.listen_address.in.getPort();
    try std.testing.expect(!isPortFree(taken));
}

test "tryReservePort holds the port until released" {
    // Reserve a random free port (taken inside reservation #1),
    // then verify reservation #2 on the same number fails until we release.
    const addr = std.net.Address.initIp4(.{ 127, 0, 0, 1 }, 0);
    var server = try addr.listen(.{ .reuse_address = false });
    const port: u16 = server.listen_address.in.getPort();
    server.deinit();

    var r1 = tryReservePort(port) orelse return error.NoFreePort;
    // Concurrent process probing the same port must fail.
    const second = tryReservePort(port);
    try std.testing.expect(second == null);

    r1.release();
    // After release the port becomes reservable again (modulo TIME_WAIT, which
    // is fine because we never went through accept→close — just bind→close).
    var r2 = tryReservePort(port) orelse return error.NoFreePort;
    r2.release();
}

test "PortReservation.release is idempotent" {
    // Reservations live inside a defer cleanup loop *and* get explicitly
    // released before spawn(). Calling release twice must not double-free.
    const addr = std.net.Address.initIp4(.{ 127, 0, 0, 1 }, 0);
    var server = try addr.listen(.{ .reuse_address = false });
    const port: u16 = server.listen_address.in.getPort();
    server.deinit();

    var r = tryReservePort(port) orelse return error.NoFreePort;
    r.release();
    r.release(); // would crash if not idempotent
    try std.testing.expect(r.released);
}

// reserveMemberPorts tests live on test-only helpers since the real callers
// pass real WorkforceMember instances with name/config heap allocations. The
// helper itself just touches `.name`, `.port`, and `.reservation`, so we can
// hand it a bare slice with whatever names we want.

fn makeMembersForTest(comptime names: []const []const u8) [names.len]WorkforceMember {
    var out: [names.len]WorkforceMember = undefined;
    inline for (names, 0..) |n, i| {
        out[i] = .{ .name = n, .config = undefined, .port = 0, .reservation = null };
    }
    return out;
}

fn releaseAllMemberReservations(members: []WorkforceMember) void {
    for (members) |*m| if (m.reservation) |*r| r.release();
}

test "reserveMemberPorts: explicit override wins regardless of slice order" {
    // The original bug: if member A iterates first and auto-allocates the
    // base port, member B's explicit --port=base fails with "already in use"
    // pointing at our own reservation. The two-pass helper must produce the
    // same result for either order.

    const allocator = std.testing.allocator;
    // Find a base port that's currently free so the test is robust against
    // whatever else is running on the host.
    var probe = reserveAvailablePort(40000, 5000) orelse return error.NoFreePort;
    const base = probe.port;
    probe.release();

    var overrides = std.StringHashMap(u16).init(allocator);
    defer overrides.deinit();
    try overrides.put("bob", base);

    // Order A: alice first (would have grabbed `base` under the old logic).
    {
        var members = makeMembersForTest(&.{ "alice", "bob" });
        defer releaseAllMemberReservations(&members);
        var sink = std.ArrayList(u8).init(allocator);
        defer sink.deinit();
        try reserveMemberPorts(allocator, &members, &overrides, base, sink.writer());
        try std.testing.expectEqual(base, members[1].port); // bob got its explicit port
        try std.testing.expect(members[0].port != base); // alice rolled forward
        try std.testing.expect(members[0].port > base);
    }

    // Order B: bob first (control — explicit happens to come first in iteration).
    {
        var members = makeMembersForTest(&.{ "bob", "alice" });
        defer releaseAllMemberReservations(&members);
        var sink = std.ArrayList(u8).init(allocator);
        defer sink.deinit();
        try reserveMemberPorts(allocator, &members, &overrides, base, sink.writer());
        try std.testing.expectEqual(base, members[0].port);
        try std.testing.expect(members[1].port != base);
        try std.testing.expect(members[1].port > base);
    }
}

test "reserveMemberPorts: auto-allocator skips ports held by Pass 1" {
    // members = [alice, bob, charlie], --port charlie=base+1, auto starts at base.
    // Without two passes, alice gets base, bob would grab base+1 — and then
    // charlie's explicit would fail. With the fix: charlie reserves base+1
    // first, alice gets base, bob skips to base+2.
    const allocator = std.testing.allocator;
    var probe = reserveAvailablePort(40000, 5000) orelse return error.NoFreePort;
    const base = probe.port;
    probe.release();
    if (base >= 65534) return; // need room for base+2

    var overrides = std.StringHashMap(u16).init(allocator);
    defer overrides.deinit();
    try overrides.put("charlie", base + 1);

    var members = makeMembersForTest(&.{ "alice", "bob", "charlie" });
    defer releaseAllMemberReservations(&members);
    var sink = std.ArrayList(u8).init(allocator);
    defer sink.deinit();
    try reserveMemberPorts(allocator, &members, &overrides, base, sink.writer());

    try std.testing.expectEqual(base, members[0].port);
    try std.testing.expectEqual(base + 2, members[1].port);
    try std.testing.expectEqual(base + 1, members[2].port);
}

test "reserveMemberPorts: duplicate explicit ports are rejected upfront" {
    const allocator = std.testing.allocator;
    var overrides = std.StringHashMap(u16).init(allocator);
    defer overrides.deinit();
    try overrides.put("alice", 9001);
    try overrides.put("bob", 9001);

    var members = makeMembersForTest(&.{ "alice", "bob" });
    defer releaseAllMemberReservations(&members);
    var sink = std.ArrayList(u8).init(allocator);
    defer sink.deinit();

    try std.testing.expectError(error.DuplicateExplicitPort, reserveMemberPorts(allocator, &members, &overrides, 4455, sink.writer()));
    // Nothing got reserved — Pass 0 rejects before touching Pass 1.
    try std.testing.expect(members[0].reservation == null);
    try std.testing.expect(members[1].reservation == null);
    try std.testing.expect(std.mem.indexOf(u8, sink.items, "same port 9001") != null);
}

test "reserveMemberPorts: explicit-port conflict with external holder fails Pass 1" {
    const allocator = std.testing.allocator;
    // Externally hold a port (simulating another process / our own ui/api).
    var external = reserveAvailablePort(40000, 5000) orelse return error.NoFreePort;
    defer external.release();
    const taken = external.port;

    var overrides = std.StringHashMap(u16).init(allocator);
    defer overrides.deinit();
    try overrides.put("alice", taken);

    var members = makeMembersForTest(&.{"alice"});
    defer releaseAllMemberReservations(&members);
    var sink = std.ArrayList(u8).init(allocator);
    defer sink.deinit();

    try std.testing.expectError(error.PortInUse, reserveMemberPorts(allocator, &members, &overrides, 4455, sink.writer()));
    try std.testing.expect(members[0].reservation == null);
    try std.testing.expect(std.mem.indexOf(u8, sink.items, "already in use") != null);
}

// ===========================================================================
// .env file parser
// ===========================================================================

test "parseEnvFile handles comments, blank lines, quotes, and export prefix" {
    const content =
        \\# this is a comment
        \\
        \\KEY1=value1
        \\KEY2="quoted value"
        \\KEY3='single quoted'
        \\export KEY4=exported
        \\  KEY5 = whitespace_trimmed  
        \\1BAD=ignored
        \\=ignored_blank_key
        \\KEY_OK=multi=signs=in=value
    ;
    var entries = try parseEnvFile(std.testing.allocator, content);
    defer freeParsedEnvList(std.testing.allocator, &entries);

    try std.testing.expectEqual(@as(usize, 6), entries.items.len);
    try std.testing.expectEqualStrings("KEY1", entries.items[0].key);
    try std.testing.expectEqualStrings("value1", entries.items[0].value);
    try std.testing.expectEqualStrings("KEY2", entries.items[1].key);
    try std.testing.expectEqualStrings("quoted value", entries.items[1].value);
    try std.testing.expectEqualStrings("KEY3", entries.items[2].key);
    try std.testing.expectEqualStrings("single quoted", entries.items[2].value);
    try std.testing.expectEqualStrings("KEY4", entries.items[3].key);
    try std.testing.expectEqualStrings("exported", entries.items[3].value);
    try std.testing.expectEqualStrings("KEY5", entries.items[4].key);
    try std.testing.expectEqualStrings("whitespace_trimmed", entries.items[4].value);
    try std.testing.expectEqualStrings("KEY_OK", entries.items[5].key);
    try std.testing.expectEqualStrings("multi=signs=in=value", entries.items[5].value);
}

test "appendDupedKv frees both dupes when the append OOMs" {
    // Allocation order inside appendDupedKv:
    //   0: key_owned dupe
    //   1: val_owned dupe
    //   2: ArrayList capacity grow (force this one to fail)
    //
    // testing.allocator panics on any leak at teardown; FailingAllocator
    // wraps it, so reaching the end without a panic proves both dupes were
    // freed via their errdefers.
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 2 });
    var list = std.ArrayList(ParsedEnv).init(failing.allocator());
    defer list.deinit();

    const result = appendDupedKv(failing.allocator(), &list, "KEY", "value");
    try std.testing.expectError(error.OutOfMemory, result);
    try std.testing.expectEqual(@as(usize, 0), list.items.len);
}

test "appendDupedKv frees the key when the value dupe OOMs" {
    // Order: 0 = key dupe (succeeds), 1 = value dupe (fails).
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 1 });
    var list = std.ArrayList(ParsedEnv).init(failing.allocator());
    defer list.deinit();

    const result = appendDupedKv(failing.allocator(), &list, "KEY", "value");
    try std.testing.expectError(error.OutOfMemory, result);
}

test "parseEnvFile does not leak when out.append OOMs" {
    // Integration test: same failure scenario reached through parseEnvFile.
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 2 });
    const result = parseEnvFile(failing.allocator(), "KEY=value");
    try std.testing.expectError(error.OutOfMemory, result);
}

test "moveParsedEnvIntoBucket transfers all entries and empties src" {
    const allocator = std.testing.allocator;
    var src = std.ArrayList(ParsedEnv).init(allocator);
    defer freeParsedEnvList(allocator, &src);
    var dest = std.ArrayList(ParsedEnv).init(allocator);
    defer freeParsedEnvList(allocator, &dest);

    try appendDupedKv(allocator, &src, "K1", "V1");
    try appendDupedKv(allocator, &src, "K2", "V2");
    try appendDupedKv(allocator, &src, "K3", "V3");

    try moveParsedEnvIntoBucket(&src, &dest);

    try std.testing.expectEqual(@as(usize, 0), src.items.len);
    try std.testing.expectEqual(@as(usize, 3), dest.items.len);
    // Order is preserved (relevant for env precedence: later wins).
    try std.testing.expectEqualStrings("K1", dest.items[0].key);
    try std.testing.expectEqualStrings("V1", dest.items[0].value);
    try std.testing.expectEqualStrings("K2", dest.items[1].key);
    try std.testing.expectEqualStrings("K3", dest.items[2].key);
}

test "moveParsedEnvIntoBucket: OOM on ensureUnusedCapacity does not leak src strings" {
    // Pre-populate src under the testing allocator (so its strings live in
    // the real heap), then perform the move under a FailingAllocator wired
    // to fail on the first allocation it sees (the dest's capacity grow).
    //
    // src must use the *same* underlying allocator that owns its strings,
    // otherwise the defer freeParsedEnvList would call free with the wrong
    // allocator. We satisfy that by giving src/dest the same FailingAllocator,
    // but pre-load src BEFORE arming the failure (fail_index counts from the
    // moment FailingAllocator.init returns).
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 999 });
    const fa = failing.allocator();

    var src = std.ArrayList(ParsedEnv).init(fa);
    defer freeParsedEnvList(fa, &src);
    var dest = std.ArrayList(ParsedEnv).init(fa);
    defer freeParsedEnvList(fa, &dest);

    try appendDupedKv(fa, &src, "K1", "V1");
    try appendDupedKv(fa, &src, "K2", "V2");

    // Now arm: the very next allocation (dest.ensureUnusedCapacity) fails.
    failing.fail_index = failing.alloc_index;

    try std.testing.expectError(error.OutOfMemory, moveParsedEnvIntoBucket(&src, &dest));

    // src still owns both entries; the deferred freeParsedEnvList must free
    // their strings. testing.allocator panics on any leak at teardown.
    try std.testing.expectEqual(@as(usize, 2), src.items.len);
    try std.testing.expectEqual(@as(usize, 0), dest.items.len);
}

test "parseEnvFile leaves bare equals values empty and skips invalid keys" {
    const content =
        \\EMPTY=
        \\WITH-DASH=ignored
        \\WITH.DOT=ignored
        \\1NUM=ignored
        \\OK=ok
    ;
    var entries = try parseEnvFile(std.testing.allocator, content);
    defer freeParsedEnvList(std.testing.allocator, &entries);

    try std.testing.expectEqual(@as(usize, 2), entries.items.len);
    try std.testing.expectEqualStrings("EMPTY", entries.items[0].key);
    try std.testing.expectEqualStrings("", entries.items[0].value);
    try std.testing.expectEqualStrings("OK", entries.items[1].key);
}

test "parseKeyValue rejects bad keys, accepts empty values, splits on FIRST =" {
    {
        const kv = parseKeyValue("KEY=value").?;
        try std.testing.expectEqualStrings("KEY", kv.key);
        try std.testing.expectEqualStrings("value", kv.value);
    }
    {
        const kv = parseKeyValue("EMPTY=").?;
        try std.testing.expectEqualStrings("EMPTY", kv.key);
        try std.testing.expectEqualStrings("", kv.value);
    }
    {
        // Postgres URL: first '=' splits, the ':' in value is preserved.
        const kv = parseKeyValue("DATABASE_URL=postgres://u:p@host:5432/db").?;
        try std.testing.expectEqualStrings("DATABASE_URL", kv.key);
        try std.testing.expectEqualStrings("postgres://u:p@host:5432/db", kv.value);
    }
    try std.testing.expectEqual(@as(?@TypeOf(parseKeyValue("").?), null), parseKeyValue("noeq"));
    try std.testing.expectEqual(@as(?@TypeOf(parseKeyValue("").?), null), parseKeyValue("=value"));
    try std.testing.expectEqual(@as(?@TypeOf(parseKeyValue("").?), null), parseKeyValue("1KEY=value"));
    try std.testing.expectEqual(@as(?@TypeOf(parseKeyValue("").?), null), parseKeyValue("BAD-KEY=value"));
}

// ===========================================================================
// Scope resolution
// ===========================================================================

test "detachScope recognises known scopes and ignores colons in values" {
    const scopes = &[_][]const u8{ "ui", "api", "myAgent" };

    {
        const s = detachScope("ui:KEY=val", scopes);
        try std.testing.expectEqualStrings("ui", s.scope.?);
        try std.testing.expectEqualStrings("KEY=val", s.rest);
    }
    {
        const s = detachScope("api:KEY=val", scopes);
        try std.testing.expectEqualStrings("api", s.scope.?);
        try std.testing.expectEqualStrings("KEY=val", s.rest);
    }
    {
        const s = detachScope("myAgent:KEY=val", scopes);
        try std.testing.expectEqualStrings("myAgent", s.scope.?);
        try std.testing.expectEqualStrings("KEY=val", s.rest);
    }
    {
        // Postgres URL: prefix before first ':' is "DATABASE_URL=postgres",
        // not a recognised scope → treat the whole arg as global.
        const s = detachScope("DATABASE_URL=postgres://u:p@host/db", scopes);
        try std.testing.expectEqual(@as(?[]const u8, null), s.scope);
        try std.testing.expectEqualStrings("DATABASE_URL=postgres://u:p@host/db", s.rest);
    }
    {
        // Plain KEY=VALUE with no colon stays global.
        const s = detachScope("FOO=bar", scopes);
        try std.testing.expectEqual(@as(?[]const u8, null), s.scope);
        try std.testing.expectEqualStrings("FOO=bar", s.rest);
    }
    {
        // Unknown scope name (the user did add a colon but the prefix isn't a known scope).
        // We don't misinterpret it.
        const s = detachScope("notaScope:KEY=val", scopes);
        try std.testing.expectEqual(@as(?[]const u8, null), s.scope);
        try std.testing.expectEqualStrings("notaScope:KEY=val", s.rest);
    }
    {
        // Windows-style path 'C:\foo' — 'C' isn't a known scope → not stripped.
        const s = detachScope("C:\\Users\\me\\.env", scopes);
        try std.testing.expectEqual(@as(?[]const u8, null), s.scope);
        try std.testing.expectEqualStrings("C:\\Users\\me\\.env", s.rest);
    }
}

// ===========================================================================
// Argument parser
// ===========================================================================

fn assertOptions(parsed: ParseResult) StartOptions {
    return switch (parsed) {
        .options => |o| o,
        .help => unreachable,
        .err => |msg| {
            std.debug.print("unexpected parse error: {s}\n", .{msg});
            unreachable;
        },
    };
}

test "parseStartArgs - empty args returns options with all defaults" {
    const parsed = try parseStartArgs(std.testing.allocator, &.{});
    var opts = assertOptions(parsed);
    defer opts.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(?[]const u8, null), opts.project_path);
    try std.testing.expectEqual(@as(?[]const u8, null), opts.profile);
    try std.testing.expectEqual(@as(?u16, null), opts.ports.ui);
    try std.testing.expectEqual(@as(?u16, null), opts.ports.api);
    try std.testing.expectEqual(@as(?u16, null), opts.ports.workforce_base);
    try std.testing.expectEqual(@as(usize, 0), opts.ports.members.count());
    try std.testing.expectEqual(@as(usize, 0), opts.raw_env_args.items.len);
    try std.testing.expectEqual(@as(usize, 0), opts.raw_env_files.items.len);
}

test "parseStartArgs - --help short-circuits before parsing the rest" {
    const args = [_][]const u8{ "--ui-port", "junk", "--help" };
    const parsed = try parseStartArgs(std.testing.allocator, &args);
    try std.testing.expect(parsed == .help);
}

test "parseStartArgs - positional path is captured" {
    const args = [_][]const u8{"my/project"};
    const parsed = try parseStartArgs(std.testing.allocator, &args);
    var opts = assertOptions(parsed);
    defer opts.deinit(std.testing.allocator);
    try std.testing.expectEqualStrings("my/project", opts.project_path.?);
}

test "parseStartArgs - duplicate paths is an error" {
    const args = [_][]const u8{ "a", "b" };
    const parsed = try parseStartArgs(std.testing.allocator, &args);
    try std.testing.expect(parsed == .err);
    std.testing.allocator.free(parsed.err);
}

test "parseStartArgs - .err return does not leak accumulated heap state" {
    // Previously, returning `ParseResult{ .err = ... }` was a normal union
    // return and did NOT fire `errdefer opts.deinit(...)`. Anything already
    // duped into opts (--env strings, --port keys, ArrayList/HashMap buffers)
    // was leaked. std.testing.allocator panics on leaks at teardown, so this
    // test would have crashed before the fix.
    {
        const args = [_][]const u8{ "--env", "FOO=bar", "--unknown" };
        const parsed = try parseStartArgs(std.testing.allocator, &args);
        try std.testing.expect(parsed == .err);
        std.testing.allocator.free(parsed.err);
    }
    {
        const args = [_][]const u8{ "--port", "agent=7000", "--unknown" };
        const parsed = try parseStartArgs(std.testing.allocator, &args);
        try std.testing.expect(parsed == .err);
        std.testing.allocator.free(parsed.err);
    }
    {
        const args = [_][]const u8{ "--env-file", "/tmp/x", "--ui-port", "notaport" };
        const parsed = try parseStartArgs(std.testing.allocator, &args);
        try std.testing.expect(parsed == .err);
        std.testing.allocator.free(parsed.err);
    }
    {
        // Mixed accumulation across several flag families before the .err.
        const args = [_][]const u8{
            "--env",      "A=1",
            "--env",      "B=2",
            "--env-file", "/tmp/x",
            "--port",     "agent=7000",
            "--port",     "agent=7100", // forces fetchRemove + put — exercises that path
            "--unknown",
        };
        const parsed = try parseStartArgs(std.testing.allocator, &args);
        try std.testing.expect(parsed == .err);
        std.testing.allocator.free(parsed.err);
    }
}

test "parseStartArgs - port flags populate PortOverrides" {
    const args = [_][]const u8{
        "--ui-port",        "4000",
        "--api-port",       "5000",
        "--workforce-port", "6000",
        "--port",           "my-agent=7000",
        "--port",           "another=7001",
    };
    const parsed = try parseStartArgs(std.testing.allocator, &args);
    var opts = assertOptions(parsed);
    defer opts.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(?u16, 4000), opts.ports.ui);
    try std.testing.expectEqual(@as(?u16, 5000), opts.ports.api);
    try std.testing.expectEqual(@as(?u16, 6000), opts.ports.workforce_base);
    try std.testing.expectEqual(@as(?u16, 7000), opts.ports.members.get("my-agent"));
    try std.testing.expectEqual(@as(?u16, 7001), opts.ports.members.get("another"));
}

test "parseStartArgs - --port ui=N is shorthand for --ui-port N" {
    const args = [_][]const u8{ "--port", "ui=4000", "--port", "api=5000" };
    const parsed = try parseStartArgs(std.testing.allocator, &args);
    var opts = assertOptions(parsed);
    defer opts.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(?u16, 4000), opts.ports.ui);
    try std.testing.expectEqual(@as(?u16, 5000), opts.ports.api);
    try std.testing.expectEqual(@as(usize, 0), opts.ports.members.count());
}

test "parseStartArgs - duplicate --port NAME=PORT keeps the last value" {
    const args = [_][]const u8{
        "--port", "agent=7000",
        "--port", "agent=7100",
    };
    const parsed = try parseStartArgs(std.testing.allocator, &args);
    var opts = assertOptions(parsed);
    defer opts.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(?u16, 7100), opts.ports.members.get("agent"));
    try std.testing.expectEqual(@as(usize, 1), opts.ports.members.count());
}

test "parseStartArgs - invalid port value produces a clear error" {
    const args = [_][]const u8{ "--ui-port", "notaport" };
    const parsed = try parseStartArgs(std.testing.allocator, &args);
    try std.testing.expect(parsed == .err);
    try std.testing.expect(std.mem.indexOf(u8, parsed.err, "invalid --ui-port") != null);
    std.testing.allocator.free(parsed.err);
}

test "parseStartArgs - --port without NAME=PORT is an error" {
    const args = [_][]const u8{ "--port", "noequals" };
    const parsed = try parseStartArgs(std.testing.allocator, &args);
    try std.testing.expect(parsed == .err);
    std.testing.allocator.free(parsed.err);
}

test "parseStartArgs - --env and --env-file are collected in order" {
    const args = [_][]const u8{
        "--env",      "A=1",
        "--env-file", ".env.local",
        "--env",      "ui:B=2",
        "--env-file", "api:.env.api",
    };
    const parsed = try parseStartArgs(std.testing.allocator, &args);
    var opts = assertOptions(parsed);
    defer opts.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 2), opts.raw_env_args.items.len);
    try std.testing.expectEqualStrings("A=1", opts.raw_env_args.items[0].raw);
    try std.testing.expectEqualStrings("ui:B=2", opts.raw_env_args.items[1].raw);
    try std.testing.expectEqual(@as(usize, 2), opts.raw_env_files.items.len);
    try std.testing.expectEqualStrings(".env.local", opts.raw_env_files.items[0].raw);
    try std.testing.expectEqualStrings("api:.env.api", opts.raw_env_files.items[1].raw);
}

test "parseStartArgs - unknown option produces an error" {
    const args = [_][]const u8{"--made-up-flag"};
    const parsed = try parseStartArgs(std.testing.allocator, &args);
    try std.testing.expect(parsed == .err);
    std.testing.allocator.free(parsed.err);
}

test "parseStartArgs - --profile captures the next arg" {
    const args = [_][]const u8{ "--profile", "staging" };
    const parsed = try parseStartArgs(std.testing.allocator, &args);
    var opts = assertOptions(parsed);
    defer opts.deinit(std.testing.allocator);
    try std.testing.expectEqualStrings("staging", opts.profile.?);
}

test "parseStartArgs - flag-only arg without value is an error" {
    inline for (.{ "--ui-port", "--api-port", "--workforce-port", "--port", "--env", "--env-file", "--profile" }) |flag| {
        const args = [_][]const u8{flag};
        const parsed = try parseStartArgs(std.testing.allocator, &args);
        try std.testing.expect(parsed == .err);
        std.testing.allocator.free(parsed.err);
    }
}

// ===========================================================================
// Env composition end-to-end
// ===========================================================================

test "applyParsedEnv merges entries with last-write-wins semantics" {
    var env = std.process.EnvMap.init(std.testing.allocator);
    defer env.deinit();

    try env.put("KEEP", "from_initial");

    const entries = [_]ParsedEnv{
        .{ .key = "FOO", .value = "1" },
        .{ .key = "BAR", .value = "first" },
        .{ .key = "BAR", .value = "second" },
    };
    try applyParsedEnv(&env, &entries);

    try std.testing.expectEqualStrings("from_initial", env.get("KEEP").?);
    try std.testing.expectEqualStrings("1", env.get("FOO").?);
    try std.testing.expectEqualStrings("second", env.get("BAR").?);
}

test "env precedence: hard runtime overrides --env, which overrides shell" {
    // Simulate the full layering by composing one map and applying layers in order.
    var env = std.process.EnvMap.init(std.testing.allocator);
    defer env.deinit();

    // Layer 1: soft built-ins
    try env.put("TIMBAL_LOG_FORMAT", "dev");
    // Layer 2: shell env
    try env.put("TIMBAL_LOG_FORMAT", "from_shell");
    try env.put("SHELL_ONLY", "shell_val");
    // Layer 3: auto .env
    try env.put("FROM_AUTO_ENV", "auto");
    try env.put("SHARED", "auto_value");
    // Layer 4: --env-file
    try env.put("SHARED", "envfile_value");
    // Layer 5: --env
    try env.put("SHARED", "envflag_value");
    try env.put("TIMBAL_LOG_FORMAT", "from_env_flag");
    // Layer 6: hard runtime info
    try env.put("PORT", "3737");

    try std.testing.expectEqualStrings("from_env_flag", env.get("TIMBAL_LOG_FORMAT").?);
    try std.testing.expectEqualStrings("envflag_value", env.get("SHARED").?);
    try std.testing.expectEqualStrings("shell_val", env.get("SHELL_ONLY").?);
    try std.testing.expectEqualStrings("auto", env.get("FROM_AUTO_ENV").?);
    try std.testing.expectEqualStrings("3737", env.get("PORT").?);
}

test "shell env scrubbing removes PORT before layering" {
    var env = std.process.EnvMap.init(std.testing.allocator);
    defer env.deinit();

    try env.put("PORT", "8080");
    try env.put("OTHER", "untouched");
    env.remove("PORT");

    try std.testing.expectEqual(@as(?[]const u8, null), env.get("PORT"));
    try std.testing.expectEqualStrings("untouched", env.get("OTHER").?);
}

test "assignScopeBuckets: ui+api+members get distinct positional buckets" {
    var b = try assignScopeBuckets(std.testing.allocator, true, true, 3);
    defer b.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 1), b.ui_bucket);
    try std.testing.expectEqual(@as(usize, 2), b.api_bucket);
    try std.testing.expectEqual(@as(usize, 3), b.member_buckets[0]);
    try std.testing.expectEqual(@as(usize, 4), b.member_buckets[1]);
    try std.testing.expectEqual(@as(usize, 5), b.member_buckets[2]);
}

test "assignScopeBuckets: missing services leave ui/api_bucket at 0 sentinel" {
    {
        var b = try assignScopeBuckets(std.testing.allocator, false, true, 2);
        defer b.deinit(std.testing.allocator);
        try std.testing.expectEqual(@as(usize, 0), b.ui_bucket);
        try std.testing.expectEqual(@as(usize, 1), b.api_bucket);
        try std.testing.expectEqual(@as(usize, 2), b.member_buckets[0]);
        try std.testing.expectEqual(@as(usize, 3), b.member_buckets[1]);
    }
    {
        var b = try assignScopeBuckets(std.testing.allocator, true, false, 1);
        defer b.deinit(std.testing.allocator);
        try std.testing.expectEqual(@as(usize, 1), b.ui_bucket);
        try std.testing.expectEqual(@as(usize, 0), b.api_bucket);
        try std.testing.expectEqual(@as(usize, 2), b.member_buckets[0]);
    }
    {
        var b = try assignScopeBuckets(std.testing.allocator, false, false, 2);
        defer b.deinit(std.testing.allocator);
        try std.testing.expectEqual(@as(usize, 0), b.ui_bucket);
        try std.testing.expectEqual(@as(usize, 0), b.api_bucket);
        try std.testing.expectEqual(@as(usize, 1), b.member_buckets[0]);
        try std.testing.expectEqual(@as(usize, 2), b.member_buckets[1]);
    }
}

test "scope routing: --env ui:KEY lands in UI's bucket even if a member is named 'ui'" {
    // Reproduces the previous bug: the storage side used first-match in
    // scope_names (which is ["ui", "api", "ui"] when has_ui and a member is
    // named "ui"), while the read side used last-match. The scoped --env
    // entry landed in bucket 1 (storage) while the UI service read from
    // bucket 3 (last match wins). Positional assignment fixes both.
    //
    // We simulate just the scope_names + dispatch logic here, plus the
    // positional ui_bucket computation.

    const allocator = std.testing.allocator;
    const has_ui = true;
    const has_api = true;
    // Members: ["ui", "worker"]. The first one shadows the reserved scope.
    const member_names = [_][]const u8{ "ui", "worker" };

    var scope_names = std.ArrayList([]const u8).init(allocator);
    defer scope_names.deinit();
    if (has_ui) try scope_names.append("ui");
    if (has_api) try scope_names.append("api");
    for (member_names) |n| try scope_names.append(n);

    // Storage path (replicates the dispatch loop).
    const storage_bucket: usize = blk: {
        const s: []const u8 = "ui";
        for (scope_names.items, 0..) |name, idx| {
            if (std.mem.eql(u8, name, s)) break :blk idx + 1;
        }
        break :blk 0;
    };

    // Read path uses positional assignment.
    var buckets = try assignScopeBuckets(allocator, has_ui, has_api, member_names.len);
    defer buckets.deinit(allocator);

    // The fix's guarantee: storage and the UI service's read bucket agree.
    try std.testing.expectEqual(buckets.ui_bucket, storage_bucket);
    try std.testing.expectEqual(@as(usize, 1), buckets.ui_bucket);

    // The member that happens to be named "ui" has its own distinct bucket;
    // it just can't be reached via the `ui:` scope prefix (warned at startup).
    try std.testing.expect(buckets.member_buckets[0] != buckets.ui_bucket);
    try std.testing.expectEqual(@as(usize, 3), buckets.member_buckets[0]);
}

test "composeServiceEnv: full precedence stack end-to-end" {
    const alloc = std.testing.allocator;

    var shell = std.process.EnvMap.init(alloc);
    defer shell.deinit();
    try shell.put("FROM_SHELL_ONLY", "shell_value");
    try shell.put("LAYER_TEST", "from_shell");

    const soft = [_][2][]const u8{
        .{ "FROM_SOFT_ONLY", "soft_value" },
        .{ "LAYER_TEST", "from_soft" },
    };

    const auto_global = [_]ParsedEnv{
        .{ .key = "FROM_AUTO_GLOBAL", .value = "auto_global_value" },
        .{ .key = "LAYER_TEST", .value = "from_auto_global" },
    };

    const auto_scope = [_]ParsedEnv{
        .{ .key = "FROM_AUTO_SCOPE", .value = "auto_scope_value" },
        .{ .key = "LAYER_TEST", .value = "from_auto_scope" },
    };

    const file_global = [_]ParsedEnv{
        .{ .key = "FROM_FILE_GLOBAL", .value = "file_global_value" },
        .{ .key = "LAYER_TEST", .value = "from_file_global" },
    };

    const file_scope = [_]ParsedEnv{
        .{ .key = "FROM_FILE_SCOPE", .value = "file_scope_value" },
        .{ .key = "LAYER_TEST", .value = "from_file_scope" },
    };

    const arg_global = [_]ParsedEnv{
        .{ .key = "FROM_ARG_GLOBAL", .value = "arg_global_value" },
        .{ .key = "LAYER_TEST", .value = "from_arg_global" },
    };

    const arg_scope = [_]ParsedEnv{
        .{ .key = "FROM_ARG_SCOPE", .value = "arg_scope_value" },
        .{ .key = "LAYER_TEST", .value = "from_arg_scope" },
    };

    const hard = [_][2][]const u8{
        .{ "PORT", "3737" },
        .{ "LAYER_TEST", "from_hard" },
    };

    var env = try composeServiceEnv(
        alloc,
        &soft,
        &shell,
        &auto_global,
        &auto_scope,
        &file_global,
        &file_scope,
        &arg_global,
        &arg_scope,
        &hard,
    );
    defer env.deinit();

    // Each layer's exclusive key is present and untouched.
    try std.testing.expectEqualStrings("soft_value", env.get("FROM_SOFT_ONLY").?);
    try std.testing.expectEqualStrings("shell_value", env.get("FROM_SHELL_ONLY").?);
    try std.testing.expectEqualStrings("auto_global_value", env.get("FROM_AUTO_GLOBAL").?);
    try std.testing.expectEqualStrings("auto_scope_value", env.get("FROM_AUTO_SCOPE").?);
    try std.testing.expectEqualStrings("file_global_value", env.get("FROM_FILE_GLOBAL").?);
    try std.testing.expectEqualStrings("file_scope_value", env.get("FROM_FILE_SCOPE").?);
    try std.testing.expectEqualStrings("arg_global_value", env.get("FROM_ARG_GLOBAL").?);
    try std.testing.expectEqualStrings("arg_scope_value", env.get("FROM_ARG_SCOPE").?);

    // The shared LAYER_TEST key sees the highest-precedence layer (hard).
    try std.testing.expectEqualStrings("from_hard", env.get("LAYER_TEST").?);
    // PORT is set from hard runtime info.
    try std.testing.expectEqualStrings("3737", env.get("PORT").?);
}

test "composeServiceEnv: omitting auto-loaded entries works for ui/api scopes" {
    const alloc = std.testing.allocator;

    var shell = std.process.EnvMap.init(alloc);
    defer shell.deinit();

    const soft = [_][2][]const u8{.{ "TIMBAL_LOG_FORMAT", "dev" }};
    const hard = [_][2][]const u8{.{ "PORT", "3000" }};

    var env = try composeServiceEnv(
        alloc,
        &soft,
        &shell,
        null, // no auto_global
        null, // no auto_scope (ui/api don't have per-member .env files)
        &.{},
        &.{},
        &.{},
        &.{},
        &hard,
    );
    defer env.deinit();

    try std.testing.expectEqualStrings("dev", env.get("TIMBAL_LOG_FORMAT").?);
    try std.testing.expectEqualStrings("3000", env.get("PORT").?);
}

test "composeServiceEnv: --env can override soft built-ins (per user choice 'builtins_lowest')" {
    const alloc = std.testing.allocator;

    var shell = std.process.EnvMap.init(alloc);
    defer shell.deinit();

    const soft = [_][2][]const u8{
        .{ "TIMBAL_API_KEY", "soft_default_key" },
        .{ "TIMBAL_LOG_FORMAT", "dev" },
    };
    const arg_global = [_]ParsedEnv{
        .{ .key = "TIMBAL_API_KEY", .value = "user_override_key" },
    };

    var env = try composeServiceEnv(
        alloc,
        &soft,
        &shell,
        null,
        null,
        &.{},
        &.{},
        &arg_global,
        &.{},
        &.{},
    );
    defer env.deinit();

    try std.testing.expectEqualStrings("user_override_key", env.get("TIMBAL_API_KEY").?);
    try std.testing.expectEqualStrings("dev", env.get("TIMBAL_LOG_FORMAT").?);
}

test "composeServiceEnv: hard runtime info cannot be overridden by --env" {
    const alloc = std.testing.allocator;

    var shell = std.process.EnvMap.init(alloc);
    defer shell.deinit();

    const arg_global = [_]ParsedEnv{
        .{ .key = "PORT", .value = "9999" },
        .{ .key = "TIMBAL_START_UI_PORT", .value = "9998" },
    };
    const hard = [_][2][]const u8{
        .{ "PORT", "3737" },
        .{ "TIMBAL_START_UI_PORT", "3737" },
    };

    var env = try composeServiceEnv(
        alloc,
        &.{},
        &shell,
        null,
        null,
        &.{},
        &.{},
        &arg_global,
        &.{},
        &hard,
    );
    defer env.deinit();

    try std.testing.expectEqualStrings("3737", env.get("PORT").?);
    try std.testing.expectEqualStrings("3737", env.get("TIMBAL_START_UI_PORT").?);
}

test "composeServiceEnv: scope-specific layers override their global counterparts" {
    const alloc = std.testing.allocator;

    var shell = std.process.EnvMap.init(alloc);
    defer shell.deinit();

    const file_global = [_]ParsedEnv{.{ .key = "MIXED", .value = "from_file_global" }};
    const file_scope = [_]ParsedEnv{.{ .key = "MIXED", .value = "from_file_scope" }};
    const arg_global = [_]ParsedEnv{.{ .key = "MIXED", .value = "from_arg_global" }};
    const arg_scope = [_]ParsedEnv{.{ .key = "MIXED", .value = "from_arg_scope" }};

    var env = try composeServiceEnv(
        alloc,
        &.{},
        &shell,
        null,
        null,
        &file_global,
        &file_scope,
        &arg_global,
        &arg_scope,
        &.{},
    );
    defer env.deinit();

    try std.testing.expectEqualStrings("from_arg_scope", env.get("MIXED").?);
}
