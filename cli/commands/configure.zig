const builtin = @import("builtin");
const std = @import("std");
const fs = std.fs;

// Embedded version.
const timbal_version = @import("../version.zig").timbal_version;

const utils = @import("../utils.zig");
const Color = utils.Color;

const is_windows = builtin.os.tag == .windows;

fn printUsageWithError(err: []const u8) !void {
    const stderr = std.io.getStdErr().writer();
    try stderr.print("{s}\n\n", .{err});
    try printUsage();
}

fn printUsage() !void {
    const stderr = std.io.getStdErr().writer();
    try stderr.writeAll("Configure Timbal credentials and settings.\n" ++
        "\n" ++
        "\x1b[1;32mUsage: \x1b[1;36mtimbal configure \x1b[0;36m[OPTIONS]\n" ++
        "\n" ++
        "\x1b[1;32mGlobal options:\n" ++
        "    \x1b[1;36m--profile <NAME>\x1b[0m  Use a named profile (overrides TIMBAL_PROFILE env var)\n" ++
        "    \x1b[1;36m-q\x1b[0m, \x1b[1;36m--quiet      \x1b[0mDo not print any output\n" ++
        "    \x1b[1;36m-v\x1b[0m, \x1b[1;36m--verbose    \x1b[0mUse verbose output\n" ++
        "    \x1b[1;36m-h\x1b[0m, \x1b[1;36m--help       \x1b[0mDisplay the concise help for this command\n" ++
        "    \x1b[1;36m-V\x1b[0m, \x1b[1;36m--version    \x1b[0mDisplay the timbal version\n" ++
        "\n");
}

// --- Path helpers ---

fn getHomePath(allocator: std.mem.Allocator) ![]u8 {
    return if (is_windows)
        std.process.getEnvVarOwned(allocator, "USERPROFILE")
    else
        std.process.getEnvVarOwned(allocator, "HOME");
}

const sep = if (is_windows) "\\" else "/";

fn getTimbalDirPath(allocator: std.mem.Allocator) ![]u8 {
    const home = try getHomePath(allocator);
    defer allocator.free(home);
    return std.fmt.allocPrint(allocator, "{s}{s}.timbal", .{ home, sep });
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

// --- INI-style file helpers ---

/// Build the section header for a profile.
/// "default" -> "[default]", anything else -> "[profile <name>]"
fn buildSectionHeader(allocator: std.mem.Allocator, profile: []const u8) ![]u8 {
    if (std.mem.eql(u8, profile, "default")) {
        return std.fmt.allocPrint(allocator, "[default]", .{});
    }
    return std.fmt.allocPrint(allocator, "[profile {s}]", .{profile});
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

/// Read a value for a given key within a profile section.
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
            // Check that the character after the key name is '=' or whitespace.
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

/// Update or insert a single key=value in the given profile section of a file's content.
/// Returns the new file content.
fn upsertValue(
    allocator: std.mem.Allocator,
    existing_content: []const u8,
    profile: []const u8,
    key: []const u8,
    value: []const u8,
) ![]u8 {
    const section_header = try buildSectionHeader(allocator, profile);
    defer allocator.free(section_header);

    var result = std.ArrayList(u8).init(allocator);

    var found_section = false;
    var replaced = false;
    var in_target = false;
    var lines = std.mem.splitScalar(u8, existing_content, '\n');
    var first = true;

    while (lines.next()) |line| {
        if (!first) try result.append('\n');
        first = false;

        const trimmed = std.mem.trim(u8, line, " \t\r");

        if (isAnySectionHeader(trimmed)) {
            if (in_target and !replaced) {
                try result.appendSlice(key);
                try result.appendSlice(" = ");
                try result.appendSlice(value);
                try result.append('\n');
                replaced = true;
            }
            in_target = isSectionHeader(trimmed, profile);
            if (in_target) found_section = true;
            try result.appendSlice(line);
            continue;
        }

        if (in_target and std.mem.startsWith(u8, trimmed, key)) {
            const rest = trimmed[key.len..];
            const after_key = std.mem.trimLeft(u8, rest, " \t");
            if (after_key.len > 0 and after_key[0] == '=') {
                try result.appendSlice(key);
                try result.appendSlice(" = ");
                try result.appendSlice(value);
                replaced = true;
                continue;
            }
        }

        try result.appendSlice(line);
    }

    if (in_target and !replaced) {
        try result.append('\n');
        try result.appendSlice(key);
        try result.appendSlice(" = ");
        try result.appendSlice(value);
        try result.append('\n');
        replaced = true;
    }

    if (!found_section) {
        if (result.items.len > 0 and result.items[result.items.len - 1] != '\n') {
            try result.append('\n');
        }
        try result.appendSlice(section_header);
        try result.append('\n');
        try result.appendSlice(key);
        try result.appendSlice(" = ");
        try result.appendSlice(value);
        try result.append('\n');
    }

    return result.toOwnedSlice();
}

/// Write content to a file, creating the parent directory if needed.
fn writeFile(allocator: std.mem.Allocator, dir_path: []const u8, file_path: []const u8, content: []const u8) !void {
    _ = allocator;
    std.fs.makeDirAbsolute(dir_path) catch |err| {
        if (err != error.PathAlreadyExists) return err;
    };
    const file = try std.fs.createFileAbsolute(file_path, .{ .truncate = true });
    defer file.close();
    try file.writeAll(content);
}

/// Read a file's contents, returning empty string if it doesn't exist.
fn readFileOrEmpty(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    return std.fs.cwd().readFileAlloc(allocator, path, 1024 * 1024) catch |err| {
        if (err == error.FileNotFound) return try allocator.dupe(u8, "");
        return err;
    };
}

/// Mask an API key for display: first 4 + **** + last 4.
fn maskKey(allocator: std.mem.Allocator, key: []const u8) ![]u8 {
    if (key.len <= 8) {
        return std.fmt.allocPrint(allocator, "****", .{});
    }
    return std.fmt.allocPrint(allocator, "{s}****{s}", .{ key[0..4], key[key.len - 4 ..] });
}

/// Prompt for a value, showing the current one in brackets. Empty input keeps the current value.
/// For secret fields, the current value is masked and input is hidden.
fn promptField(
    allocator: std.mem.Allocator,
    stdout: std.fs.File.Writer,
    stdin: std.io.GenericReader(std.fs.File, std.posix.ReadError, std.fs.File.read),
    label: []const u8,
    current: ?[]const u8,
    secret: bool,
) !?[]u8 {
    // Display the prompt with current value in brackets.
    if (current) |val| {
        if (secret) {
            const masked = try maskKey(allocator, val);
            defer allocator.free(masked);
            try stdout.print("{s} [{s}]: ", .{ label, masked });
        } else {
            try stdout.print("{s} [{s}]: ", .{ label, val });
        }
    } else {
        try stdout.print("{s} [None]: ", .{label});
    }

    // Read input, hiding it if secret on Unix.
    if (secret and !is_windows) {
        const stdin_fd = std.io.getStdIn().handle;
        const original = std.posix.tcgetattr(stdin_fd) catch {
            // Fallback to normal read.
            return readLine(allocator, stdin, stdout);
        };

        var noecho = original;
        noecho.lflag.ECHO = false;
        std.posix.tcsetattr(stdin_fd, .NOW, noecho) catch {};

        var buf: [512]u8 = undefined;
        const input = stdin.readUntilDelimiter(&buf, '\n') catch |err| {
            std.posix.tcsetattr(stdin_fd, .NOW, original) catch {};
            return err;
        };

        std.posix.tcsetattr(stdin_fd, .NOW, original) catch {};
        try stdout.print("\n", .{});

        const trimmed = std.mem.trim(u8, input, " \t\r");
        if (trimmed.len == 0) return null;
        return try allocator.dupe(u8, trimmed);
    }

    return readLine(allocator, stdin, stdout);
}

fn readLine(
    allocator: std.mem.Allocator,
    stdin: std.io.GenericReader(std.fs.File, std.posix.ReadError, std.fs.File.read),
    stdout: std.fs.File.Writer,
) !?[]u8 {
    _ = stdout;
    var buf: [512]u8 = undefined;
    const input = stdin.readUntilDelimiter(&buf, '\n') catch |err| {
        return err;
    };
    const trimmed = std.mem.trim(u8, input, " \t\r");
    if (trimmed.len == 0) return null;
    return try allocator.dupe(u8, trimmed);
}

pub fn run(allocator: std.mem.Allocator, args: []const []const u8) !void {
    var verbose: bool = false;
    _ = &verbose;
    var quiet: bool = false;
    _ = &quiet;
    // Profile priority: --profile flag > TIMBAL_PROFILE env var > "default"
    const env_profile = std.process.getEnvVarOwned(allocator, "TIMBAL_PROFILE") catch |err| blk: {
        if (err == error.EnvironmentVariableNotFound) break :blk null;
        return err;
    };
    defer if (env_profile) |p| allocator.free(p);

    var profile: []const u8 = env_profile orelse "default";

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            try printUsage();
            return;
        } else if (std.mem.eql(u8, arg, "-V") or std.mem.eql(u8, arg, "--version")) {
            std.debug.print("Timbal {s}\n", .{timbal_version});
            return;
        } else if (std.mem.eql(u8, arg, "-v") or std.mem.eql(u8, arg, "--verbose")) {
            verbose = true;
        } else if (std.mem.eql(u8, arg, "-q") or std.mem.eql(u8, arg, "--quiet")) {
            quiet = true;
        } else if (std.mem.eql(u8, arg, "--profile")) {
            i += 1;
            if (i >= args.len) {
                try printUsageWithError("Error: --profile requires a name argument");
                return;
            }
            profile = args[i];
        } else {
            try printUsageWithError("Error: unknown option");
            return;
        }
    }

    const stdout = std.io.getStdOut().writer();
    const stdin = std.io.getStdIn().reader();

    // Resolve paths.
    const timbal_dir = try getTimbalDirPath(allocator);
    defer allocator.free(timbal_dir);

    const credentials_path = try getCredentialsPath(allocator);
    defer allocator.free(credentials_path);

    const config_path = try getConfigPath(allocator);
    defer allocator.free(config_path);

    // Read existing files.
    var credentials_content = try readFileOrEmpty(allocator, credentials_path);
    defer allocator.free(credentials_content);

    var config_content = try readFileOrEmpty(allocator, config_path);
    defer allocator.free(config_content);

    // Read current values for this profile.
    const current_api_key = readValue(credentials_content, profile, "api_key");
    const current_output = readValue(config_content, profile, "output");

    // --- Prompt: API Key (secret, written to credentials) ---
    if (try promptField(allocator, stdout, stdin, "Timbal API Key", current_api_key, true)) |new_key| {
        defer allocator.free(new_key);
        const new_credentials = try upsertValue(allocator, credentials_content, profile, "api_key", new_key);
        allocator.free(credentials_content);
        credentials_content = new_credentials;
    }

    // --- Prompt: Default output format (written to config) ---
    if (try promptField(allocator, stdout, stdin, "Default output format", current_output, false)) |new_output| {
        defer allocator.free(new_output);
        // Validate the value.
        if (!std.mem.eql(u8, new_output, "json") and !std.mem.eql(u8, new_output, "text")) {
            const stderr = std.io.getStdErr().writer();
            try stderr.writeAll("Error: output format must be \"json\" or \"text\".\n");
            return;
        }
        const new_config = try upsertValue(allocator, config_content, profile, "output", new_output);
        allocator.free(config_content);
        config_content = new_config;
    }

    // Write files.
    try writeFile(allocator, timbal_dir, credentials_path, credentials_content);
    try writeFile(allocator, timbal_dir, config_path, config_content);
}
