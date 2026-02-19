const builtin = @import("builtin");
const std = @import("std");
const fs = std.fs;

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
        utils.global_options_help ++
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
        if (result.items.len > 0 and result.items[result.items.len - 1] != '\n') {
            try result.append('\n');
        }
        try result.appendSlice(key);
        try result.appendSlice(" = ");
        try result.appendSlice(value);
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
    }

    // Ensure file ends with exactly one newline.
    if (result.items.len > 0 and result.items[result.items.len - 1] != '\n') {
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

    // Read input, hiding it if secret.
    if (secret) {
        if (is_windows) {
            const stdin_handle = std.io.getStdIn().handle;
            var orig_mode: std.os.windows.DWORD = 0;
            const got_mode = std.os.windows.kernel32.GetConsoleMode(stdin_handle, &orig_mode) != 0;

            if (got_mode) {
                // Disable echo input (0x0004) and line input (0x0002) for char-by-char reading.
                _ = std.os.windows.kernel32.SetConsoleMode(stdin_handle, orig_mode & ~@as(u32, 0x0004 | 0x0002));
            }

            var buf: [512]u8 = undefined;
            var len: usize = 0;

            while (true) {
                const byte = stdin.readByte() catch |err| {
                    if (got_mode) _ = std.os.windows.kernel32.SetConsoleMode(stdin_handle, orig_mode);
                    return err;
                };
                if (byte == '\n' or byte == '\r') {
                    break;
                }
                // Handle backspace (8 = BS).
                if (byte == 8) {
                    if (len > 0) {
                        len -= 1;
                        try stdout.print("\x08 \x08", .{});
                    }
                    continue;
                }
                if (len < buf.len) {
                    buf[len] = byte;
                    len += 1;
                    try stdout.print("*", .{});
                }
            }

            if (got_mode) _ = std.os.windows.kernel32.SetConsoleMode(stdin_handle, orig_mode);
            try stdout.print("\n", .{});

            const trimmed = std.mem.trim(u8, buf[0..len], " \t\r");
            if (trimmed.len == 0) return null;
            return try allocator.dupe(u8, trimmed);
        } else {
            const stdin_fd = std.io.getStdIn().handle;
            const original = std.posix.tcgetattr(stdin_fd) catch {
                // Fallback to normal read.
                return readLine(allocator, stdin, stdout);
            };

            // Disable echo and canonical mode so we get chars one at a time.
            var raw = original;
            raw.lflag.ECHO = false;
            raw.lflag.ICANON = false;
            // Read one byte at a time.
            raw.cc[@intFromEnum(std.posix.V.MIN)] = 1;
            raw.cc[@intFromEnum(std.posix.V.TIME)] = 0;
            std.posix.tcsetattr(stdin_fd, .NOW, raw) catch {};

            var buf: [512]u8 = undefined;
            var len: usize = 0;

            while (true) {
                const byte = stdin.readByte() catch |err| {
                    std.posix.tcsetattr(stdin_fd, .NOW, original) catch {};
                    return err;
                };
                if (byte == '\n' or byte == '\r') {
                    break;
                }
                // Handle backspace (127 = DEL, 8 = BS).
                if (byte == 127 or byte == 8) {
                    if (len > 0) {
                        len -= 1;
                        // Erase the asterisk on screen.
                        try stdout.print("\x08 \x08", .{});
                    }
                    continue;
                }
                if (len < buf.len) {
                    buf[len] = byte;
                    len += 1;
                    try stdout.print("*", .{});
                }
            }

            std.posix.tcsetattr(stdin_fd, .NOW, original) catch {};
            try stdout.print("\n", .{});

            const trimmed = std.mem.trim(u8, buf[0..len], " \t\r");
            if (trimmed.len == 0) return null;
            return try allocator.dupe(u8, trimmed);
        }
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

fn runGitConfig(
    allocator: std.mem.Allocator,
    stdout: std.fs.File.Writer,
    verbose: bool,
    key: []const u8,
    value: []const u8,
) !void {
    try runGitConfigWithFlag(allocator, stdout, verbose, null, key, value);
}

fn runGitConfigWithFlag(
    allocator: std.mem.Allocator,
    stdout: std.fs.File.Writer,
    verbose: bool,
    flag: ?[]const u8,
    key: []const u8,
    value: []const u8,
) !void {
    if (verbose) {
        if (flag) |f| {
            try stdout.print("Running: git config --global {s} {s} '{s}'\n", .{ f, key, value });
        } else {
            try stdout.print("Running: git config --global {s} '{s}'\n", .{ key, value });
        }
    }

    var argv_buf: [6][]const u8 = undefined;
    var argc: usize = 0;
    argv_buf[argc] = "git";
    argc += 1;
    argv_buf[argc] = "config";
    argc += 1;
    argv_buf[argc] = "--global";
    argc += 1;
    if (flag) |f| {
        argv_buf[argc] = f;
        argc += 1;
    }
    argv_buf[argc] = key;
    argc += 1;
    argv_buf[argc] = value;
    argc += 1;
    const argv = argv_buf[0..argc];

    var child = std.process.Child.init(argv, allocator);
    child.stdin_behavior = .Ignore;
    child.stdout_behavior = .Ignore;
    child.stderr_behavior = .Ignore;

    if (child.spawn()) |_| {
        const term = child.wait() catch |err| {
            if (verbose) {
                try stdout.print("Warning: Failed to wait for git process: {}\n", .{err});
            }
            return;
        };
        if (verbose) {
            if (term.Exited == 0) {
                try stdout.print("Set {s} = {s}\n", .{ key, value });
            } else {
                try stdout.print("Warning: git config exited with code {d}.\n", .{term.Exited});
            }
        }
    } else |err| {
        if (verbose) {
            try stdout.print("Warning: Failed to spawn git process: {}\n", .{err});
        } else {
            try stdout.print("Warning: Failed to set git config {s}.\n", .{key});
        }
    }
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

    // Read current values for this profile (duped so they survive buffer frees).
    const current_api_key = if (readValue(credentials_content, profile, "api_key")) |v| try allocator.dupe(u8, v) else null;
    defer if (current_api_key) |v| allocator.free(v);
    const current_org = if (readValue(config_content, profile, "org")) |v| try allocator.dupe(u8, v) else null;
    defer if (current_org) |v| allocator.free(v);
    const current_base_url = if (readValue(config_content, profile, "base_url")) |v| try allocator.dupe(u8, v) else null;
    defer if (current_base_url) |v| allocator.free(v);
    const current_output = if (readValue(config_content, profile, "output")) |v| try allocator.dupe(u8, v) else null;
    defer if (current_output) |v| allocator.free(v);

    // --- Prompt: API Key (secret, written to credentials) ---
    const api_key_input = try promptField(allocator, stdout, stdin, "Timbal API Key (https://app.timbal.ai/profile/api-keys)", current_api_key, true);
    defer if (api_key_input) |k| allocator.free(k);
    const final_api_key = api_key_input orelse current_api_key;

    // --- Prompt: Organization ID (written to config) ---
    const org_input = try promptField(allocator, stdout, stdin, "Organization ID", current_org, false);
    defer if (org_input) |o| allocator.free(o);
    const final_org = org_input orelse current_org;

    // --- Prompt: Base URL (written to config) ---
    const base_url_input = try promptField(allocator, stdout, stdin, "Platform base URL", current_base_url orelse "https://api.timbal.ai", false);
    defer if (base_url_input) |b| allocator.free(b);
    const final_base_url = base_url_input orelse (current_base_url orelse "https://api.timbal.ai");

    // --- Prompt: Default output format (written to config) ---
    const output_input = try promptField(allocator, stdout, stdin, "Default output format", current_output orelse "json", false);
    defer if (output_input) |o| allocator.free(o);
    if (output_input) |val| {
        if (!std.mem.eql(u8, val, "json") and !std.mem.eql(u8, val, "text")) {
            const stderr = std.io.getStdErr().writer();
            try stderr.writeAll("Error: output format must be \"json\" or \"text\".\n");
            return;
        }
    }
    const final_output = output_input orelse (current_output orelse "json");

    // Always upsert all fields to keep file clean.
    if (final_api_key) |key| {
        const new_creds = try upsertValue(allocator, credentials_content, profile, "api_key", key);
        allocator.free(credentials_content);
        credentials_content = new_creds;
    }

    if (final_org) |org| {
        const new_config = try upsertValue(allocator, config_content, profile, "org", org);
        allocator.free(config_content);
        config_content = new_config;
    }
    {
        const new_config = try upsertValue(allocator, config_content, profile, "base_url", final_base_url);
        allocator.free(config_content);
        config_content = new_config;
    }
    {
        const new_config = try upsertValue(allocator, config_content, profile, "output", final_output);
        allocator.free(config_content);
        config_content = new_config;
    }

    // Write files.
    try writeFile(allocator, timbal_dir, credentials_path, credentials_content);
    try writeFile(allocator, timbal_dir, config_path, config_content);

    // Configure git credential settings for the base URL.
    if (verbose) {
        try stdout.print("Configuring git credentials for {s}...\n", .{final_base_url});
    }

    const use_http_path_key = try std.fmt.allocPrint(allocator, "credential.{s}.useHttpPath", .{final_base_url});
    defer allocator.free(use_http_path_key);
    try runGitConfig(allocator, stdout, verbose, use_http_path_key, "false");

    const helper_key = try std.fmt.allocPrint(allocator, "credential.{s}.helper", .{final_base_url});
    defer allocator.free(helper_key);
    // Set empty string first to clear any inherited system/global credential helpers
    // (e.g. Git Credential Manager on Windows), then add the timbal helper.
    try runGitConfigWithFlag(allocator, stdout, verbose, "--replace-all", helper_key, "");
    try runGitConfigWithFlag(allocator, stdout, verbose, "--add", helper_key, "!timbal credential-helper");
}
