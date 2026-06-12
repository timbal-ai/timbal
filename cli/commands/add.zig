const std = @import("std");
const fs = std.fs;

const utils = @import("../utils.zig");

const Color = utils.Color;

fn printUsageWithError(err: []const u8) !void {
    const stderr = std.io.getStdErr().writer();
    try stderr.print("{s}\n\n", .{err});
    try printUsage();
}

fn printUsage() !void {
    const stderr = std.io.getStdErr().writer();
    try stderr.writeAll("Add a component to an existing timbal project.\n" ++
        "\n" ++
        "\x1b[1;32mUsage: \x1b[1;36mtimbal add \x1b[0;36m<COMPONENT> \x1b[0;33m<name> \x1b[0m[options]\n" ++
        "\n" ++
        "\x1b[1;32mComponents:\n" ++
        "    \x1b[1;36magent    \x1b[0mAdd a new agent to the workforce\n" ++
        "    \x1b[1;36mworkflow \x1b[0mAdd a new workflow to the workforce\n" ++
        "    \x1b[1;36mui       \x1b[0mAdd the default web UI blueprint\n" ++
        "    \x1b[1;36mapi      \x1b[0mAdd the API blueprint (Elysia + Bun)\n" ++
        "\n" ++
        "\x1b[1;32mArguments:\n" ++
        "    \x1b[1;33mname     \x1b[0mRequired for agent/workflow (cannot contain = : , / \\)\n" ++
        "\n" ++
        "\x1b[1;32mOptions:\n" ++
        "    \x1b[1;36m-f\x1b[0m, \x1b[1;36m--force      \x1b[0mReplace ui/, api/, or workforce/<name>/ if it already exists\n" ++
        "\n" ++
        utils.global_options_help ++
        "\n");
}

/// Case-insensitive match against any of `yes`, `y`.
fn isAffirmative(input: []const u8) bool {
    const trimmed = std.mem.trim(u8, input, " \t\r\n");
    if (trimmed.len == 0) return false;
    if (trimmed.len > 8) return false; // longest accepted is "yes"
    var lower_buf: [8]u8 = undefined;
    for (trimmed, 0..) |c, i| lower_buf[i] = std.ascii.toLower(c);
    const lower = lower_buf[0..trimmed.len];
    return std.mem.eql(u8, lower, "y") or std.mem.eql(u8, lower, "yes");
}

fn confirmOverwrite(name: []const u8) bool {
    const stdin = std.io.getStdIn();
    const stdout = std.io.getStdOut().writer();

    stdout.print("'{s}' directory already exists. Overwrite? (y/N): ", .{name}) catch return false;

    var buf: [32]u8 = undefined;
    const line = stdin.reader().readUntilDelimiter(&buf, '\n') catch return false;
    return isAffirmative(line);
}

/// Decide whether to overwrite an existing component directory.
///
/// - `--force` → unconditional yes.
/// - Non-interactive stdin → refuse with a clear message instructing the
///   caller to pass `--force`, then exit non-zero so CI/scripts notice.
///   Previously the prompt would silently abort under piped contexts
///   because `readUntilDelimiter` failed and we'd return exit 0.
/// - Interactive stdin → prompt as before; abort stays exit 0 to match
///   the prior interactive behaviour.
///
/// Returns `true` when the caller should proceed with the overwrite.
fn shouldOverwrite(name: []const u8, force: bool) bool {
    if (force) return true;
    if (!std.io.getStdIn().isTty()) {
        std.debug.print(
            "Error: '{s}' directory already exists. Re-run with --force to overwrite, " ++
                "or remove the directory manually.\n",
            .{name},
        );
        std.process.exit(1);
    }
    if (confirmOverwrite(name)) return true;
    std.debug.print("Aborted.\n", .{});
    return false;
}

const builtin = @import("builtin");
const is_windows = builtin.os.tag == .windows;

const CP_UTF8: u32 = 65001;

fn getProjectName(path: []const u8) []const u8 {
    const last_fwd = std.mem.lastIndexOf(u8, path, "/");
    const last_back = std.mem.lastIndexOf(u8, path, "\\");
    const last_sep = if (last_fwd) |f| if (last_back) |b| @max(f, b) else f else last_back;
    if (last_sep) |idx| {
        return path[idx + 1 ..];
    }
    return path;
}

pub fn run(allocator: std.mem.Allocator, args: []const []const u8) !void {
    // Enable UTF-8 output on Windows so Unicode characters render correctly.
    const orig_cp = if (is_windows) blk: {
        const cp = std.os.windows.kernel32.GetConsoleOutputCP();
        _ = std.os.windows.kernel32.SetConsoleOutputCP(CP_UTF8);
        break :blk cp;
    } else 0;
    defer if (is_windows) {
        _ = std.os.windows.kernel32.SetConsoleOutputCP(orig_cp);
    };

    const stdout = std.io.getStdOut().writer();
    var component: ?[]const u8 = null;
    var name: ?[]const u8 = null;
    var force = false;

    for (args) |arg| {
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            try printUsage();
            return;
        } else if (std.mem.eql(u8, arg, "-f") or std.mem.eql(u8, arg, "--force")) {
            force = true;
        } else if (!std.mem.startsWith(u8, arg, "-")) {
            if (component == null) {
                component = arg;
            } else if (name == null) {
                name = arg;
            } else {
                try printUsageWithError("Error: too many arguments provided");
                return;
            }
        } else {
            try printUsageWithError("Error: unknown option");
            return;
        }
    }

    const comp = component orelse {
        try printUsageWithError("Error: missing component (agent, workflow, ui, or api)");
        return;
    };

    const cwd = fs.cwd();
    const cwd_path = try cwd.realpathAlloc(allocator, ".");
    defer allocator.free(cwd_path);
    const project_name = getProjectName(cwd_path);

    if (std.mem.eql(u8, comp, "agent") or std.mem.eql(u8, comp, "workflow")) {
        const project_type: utils.ProjectType = if (std.mem.eql(u8, comp, "agent")) .agent else .workflow;

        const member_name = name orelse {
            try printUsageWithError("Error: missing required argument: name");
            return;
        };

        // Check that workforce/ directory exists
        cwd.access("workforce", .{}) catch {
            std.debug.print("Error: No 'workforce' directory found. Are you in a timbal project?\n", .{});
            return;
        };

        const member_dir = try std.fmt.allocPrint(allocator, "workforce/{s}", .{member_name});
        defer allocator.free(member_dir);

        if (cwd.access(member_dir, .{})) |_| {
            if (!shouldOverwrite(member_dir, force)) return;
            try cwd.deleteTree(member_dir);
        } else |_| {}

        const used_name = utils.addWorkforceMember(allocator, cwd, project_name, project_type, member_name) catch |err| switch (err) {
            error.InvalidWorkforceName, error.ReservedWorkforceName, error.WorkforceMemberExists => {
                utils.printWorkforceNameError(err, member_name);
                return;
            },
            else => |e| return e,
        };
        defer allocator.free(used_name);

        try stdout.print("\n{s}✓{s} {s}Added {s} '{s}' to workforce{s}\n\n", .{
            Color.bold_green,
            Color.reset,
            Color.bold,
            comp,
            used_name,
            Color.reset,
        });
    } else if (std.mem.eql(u8, comp, "ui")) {
        if (cwd.access("ui", .{})) |_| {
            if (!shouldOverwrite("ui", force)) return;
            try cwd.deleteTree("ui");
        } else |_| {}

        try utils.fetchBlueprint(allocator, cwd_path, "ui", utils.blueprint_ui_simple_chat_url);

        try stdout.print("\n{s}✓{s} {s}Added default web UI blueprint{s}\n\n", .{
            Color.bold_green,
            Color.reset,
            Color.bold,
            Color.reset,
        });
    } else if (std.mem.eql(u8, comp, "api")) {
        if (cwd.access("api", .{})) |_| {
            if (!shouldOverwrite("api", force)) return;
            try cwd.deleteTree("api");
        } else |_| {}

        try utils.fetchBlueprint(allocator, cwd_path, "api", utils.blueprint_api_url);

        try stdout.print("\n{s}✓{s} {s}Added API blueprint{s}\n\n", .{
            Color.bold_green,
            Color.reset,
            Color.bold,
            Color.reset,
        });
    } else {
        try printUsageWithError("Error: unknown component. Expected: agent, workflow, ui, or api");
        return;
    }
}
