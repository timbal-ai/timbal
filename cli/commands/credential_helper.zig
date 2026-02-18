const builtin = @import("builtin");
const std = @import("std");
const utils = @import("../utils.zig");

const is_windows = builtin.os.tag == .windows;
const sep = if (is_windows) "\\" else "/";

fn printUsageWithError(err: []const u8) !void {
    const stderr = std.io.getStdErr().writer();
    try stderr.print("{s}\n\n", .{err});
    try printUsage();
}

fn printUsage() !void {
    const stderr = std.io.getStdErr().writer();
    try stderr.writeAll("Git credential helper for the Timbal Platform.\n" ++
        "\n" ++
        "\x1b[1;32mUsage: \x1b[1;36mtimbal credential-helper \x1b[0;36m<get|store|erase>\n" ++
        "\n" ++
        "\x1b[1;32mSetup:\n" ++
        "\x1b[0m    git config --global credential.https://api.timbal.ai.helper '!timbal credential-helper'\n" ++
        "    git config --global credential.https://api.dev.timbal.ai.helper '!timbal credential-helper'\n" ++
        "\n" ++
        utils.global_options_help ++
        "\n");
}

fn getCredentialsPath(allocator: std.mem.Allocator) ![]u8 {
    const home = if (is_windows)
        try std.process.getEnvVarOwned(allocator, "USERPROFILE")
    else
        try std.process.getEnvVarOwned(allocator, "HOME");
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

fn readApiKey(content: []const u8, profile: []const u8) ?[]const u8 {
    var in_target = false;
    var lines = std.mem.splitScalar(u8, content, '\n');

    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");
        if (isAnySectionHeader(trimmed)) {
            in_target = isSectionHeader(trimmed, profile);
            continue;
        }
        if (in_target and std.mem.startsWith(u8, trimmed, "api_key")) {
            const rest = trimmed["api_key".len..];
            const after_key = std.mem.trimLeft(u8, rest, " \t");
            if (after_key.len > 0 and after_key[0] == '=') {
                const value = std.mem.trim(u8, after_key[1..], " \t");
                if (value.len > 0) return value;
            }
        }
    }
    return null;
}

/// Parse the Git credential protocol input from stdin.
/// Returns the host value if found.
fn parseGitInput(allocator: std.mem.Allocator, stdin: std.io.GenericReader(std.fs.File, std.posix.ReadError, std.fs.File.read)) !?[]u8 {
    var host: ?[]u8 = null;

    while (true) {
        var buf: [1024]u8 = undefined;
        const line = stdin.readUntilDelimiter(&buf, '\n') catch |err| {
            if (err == error.EndOfStream) break;
            return err;
        };
        const trimmed = std.mem.trim(u8, line, " \t\r");
        if (trimmed.len == 0) break;

        if (std.mem.startsWith(u8, trimmed, "host=")) {
            if (host) |old| allocator.free(old);
            host = try allocator.dupe(u8, trimmed["host=".len..]);
        }
    }

    return host;
}

pub fn run(allocator: std.mem.Allocator, args: []const []const u8) !void {
    // Parse args: we expect a single action (get, store, erase) and optional --profile.
    var action: ?[]const u8 = null;
    var profile_flag: ?[]const u8 = null;

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            try printUsage();
            return;
        } else if (std.mem.eql(u8, arg, "--profile")) {
            i += 1;
            if (i >= args.len) {
                try printUsageWithError("Error: --profile requires a name argument");
                return;
            }
            profile_flag = args[i];
        } else if (!std.mem.startsWith(u8, arg, "-")) {
            action = arg;
        } else {
            try printUsageWithError("Error: unknown option");
            return;
        }
    }

    if (action == null) {
        try printUsageWithError("Error: missing action (get, store, or erase)");
        return;
    }

    // We only respond to "get". Git also calls "store" and "erase" but
    // we don't need to handle those — credentials are managed by `timbal configure`.
    if (!std.mem.eql(u8, action.?, "get")) {
        return;
    }

    const stdin = std.io.getStdIn().reader();
    const stdout = std.io.getStdOut().writer();

    // Parse Git's input to check the host.
    const host = try parseGitInput(allocator, stdin);
    if (host == null) return;
    defer allocator.free(host.?);

    // Only respond for Timbal hosts.
    if (!std.mem.endsWith(u8, host.?, "timbal.ai")) return;

    // Profile priority: --profile flag > TIMBAL_PROFILE env var > "default"
    const env_profile = std.process.getEnvVarOwned(allocator, "TIMBAL_PROFILE") catch |err| blk: {
        if (err == error.EnvironmentVariableNotFound) break :blk null;
        return err;
    };
    defer if (env_profile) |p| allocator.free(p);

    const profile: []const u8 = profile_flag orelse (env_profile orelse "default");

    // Read the credentials file.
    const credentials_path = try getCredentialsPath(allocator);
    defer allocator.free(credentials_path);

    const content = std.fs.cwd().readFileAlloc(allocator, credentials_path, 1024 * 1024) catch |err| {
        if (err == error.FileNotFound) return;
        return err;
    };
    defer allocator.free(content);

    // Look up the API key for the profile.
    const api_key = readApiKey(content, profile) orelse return;

    // Output Git credential protocol response.
    try stdout.print("protocol=https\nhost={s}\nusername=timbal\npassword={s}\n\n", .{ host.?, api_key });
}
