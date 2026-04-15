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
        "\x1b[1;32mUsage: \x1b[1;36mtimbal add \x1b[0;36m<COMPONENT> \x1b[0;33m[name]\n" ++
        "\n" ++
        "\x1b[1;32mComponents:\n" ++
        "    \x1b[1;36magent    \x1b[0mAdd a new agent to the workforce\n" ++
        "    \x1b[1;36mworkflow \x1b[0mAdd a new workflow to the workforce\n" ++
        "    \x1b[1;36mui       \x1b[0mAdd the default web UI blueprint\n" ++
        "    \x1b[1;36mapi      \x1b[0mAdd the API blueprint (Elysia + Bun)\n" ++
        "\n" ++
        "\x1b[1;32mArguments:\n" ++
        "    \x1b[1;33mname     \x1b[0mOptional name for the workforce member (agent/workflow only)\n" ++
        "\n" ++
        utils.global_options_help ++
        "\n");
}

fn confirmOverwrite(name: []const u8) bool {
    const stdin = std.io.getStdIn();
    const stdout = std.io.getStdOut().writer();

    stdout.print("'{s}' directory already exists. Overwrite? (y/N): ", .{name}) catch return false;

    var buf: [16]u8 = undefined;
    const line = stdin.reader().readUntilDelimiter(&buf, '\n') catch return false;
    const trimmed = std.mem.trim(u8, line, " \t\r");
    return trimmed.len == 1 and (trimmed[0] == 'y' or trimmed[0] == 'Y');
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

    for (args) |arg| {
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            try printUsage();
            return;
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

        // Check that workforce/ directory exists
        cwd.access("workforce", .{}) catch {
            std.debug.print("Error: No 'workforce' directory found. Are you in a timbal project?\n", .{});
            return;
        };

        const funny_name = try utils.addWorkforceMember(allocator, cwd, project_name, project_type, name);
        defer allocator.free(funny_name);

        try stdout.print("\n{s}✓{s} {s}Added {s} '{s}' to workforce{s}\n\n", .{
            Color.bold_green,
            Color.reset,
            Color.bold,
            comp,
            funny_name,
            Color.reset,
        });
    } else if (std.mem.eql(u8, comp, "ui")) {
        if (cwd.access("ui", .{})) |_| {
            if (!confirmOverwrite("ui")) {
                std.debug.print("Aborted.\n", .{});
                return;
            }
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
        // Check if api/ already exists
        if (cwd.access("api", .{})) |_| {
            if (!confirmOverwrite("api")) {
                std.debug.print("Aborted.\n", .{});
                return;
            }
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
