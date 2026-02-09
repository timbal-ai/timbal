const std = @import("std");
const fs = std.fs;

const utils = @import("../utils.zig");
const timbal_version = @import("../version.zig").timbal_version;

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
        "\x1b[1;32mUsage: \x1b[1;36mtimbal add \x1b[0;36m<COMPONENT>\n" ++
        "\n" ++
        "\x1b[1;32mComponents:\n" ++
        "    \x1b[1;36magent     \x1b[0mAdd a new agent to the workforce\n" ++
        "    \x1b[1;36mworkflow  \x1b[0mAdd a new workflow to the workforce\n" ++
        "    \x1b[1;36mui        \x1b[0mAdd the UI blueprint (React + Vite + TypeScript + shadcn)\n" ++
        "    \x1b[1;36mapi       \x1b[0mAdd the API blueprint (Elysia + Bun)\n" ++
        "\n" ++
        "\x1b[1;32mGlobal options:\n" ++
        "    \x1b[1;36m-h\x1b[0m, \x1b[1;36m--help       \x1b[0mDisplay the concise help for this command\n" ++
        "    \x1b[1;36m-V\x1b[0m, \x1b[1;36m--version    \x1b[0mDisplay the timbal version\n" ++
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

fn getProjectName(path: []const u8) []const u8 {
    if (std.mem.lastIndexOf(u8, path, "/")) |idx| {
        return path[idx + 1 ..];
    }
    return path;
}

pub fn run(allocator: std.mem.Allocator, args: []const []const u8) !void {
    const stdout = std.io.getStdOut().writer();
    var component: ?[]const u8 = null;

    for (args) |arg| {
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            try printUsage();
            return;
        } else if (std.mem.eql(u8, arg, "-V") or std.mem.eql(u8, arg, "--version")) {
            std.debug.print("Timbal {s}\n", .{timbal_version});
            return;
        } else if (!std.mem.startsWith(u8, arg, "-")) {
            if (component != null) {
                try printUsageWithError("Error: multiple components provided");
                return;
            }
            component = arg;
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

        const funny_name = try utils.addWorkforceMember(allocator, cwd, project_name, project_type);
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
        // Check if ui/ already exists
        if (cwd.access("ui", .{})) |_| {
            if (!confirmOverwrite("ui")) {
                std.debug.print("Aborted.\n", .{});
                return;
            }
            try cwd.deleteTree("ui");
        } else |_| {}

        try utils.fetchBlueprint(allocator, cwd_path, "ui", utils.blueprint_ui_url);

        try stdout.print("\n{s}✓{s} {s}Added UI blueprint{s}\n\n", .{
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
