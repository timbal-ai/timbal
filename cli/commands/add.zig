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
        "    \x1b[1;36mui        \x1b[0mAdd a UI blueprint (Chat or Blank)\n" ++
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

const UIType = enum {
    chat,
    blank,

    fn url(self: UIType) []const u8 {
        return switch (self) {
            .chat => utils.blueprint_ui_simple_chat_url,
            .blank => utils.blueprint_ui_url,
        };
    }

    fn label(self: UIType) []const u8 {
        return switch (self) {
            .chat => "Chat",
            .blank => "Blank",
        };
    }
};

const builtin = @import("builtin");
const is_windows = builtin.os.tag == .windows;
const TerminalState = if (is_windows) void else std.posix.termios;

const terminal = if (is_windows) struct {
    fn enableRawMode() !TerminalState {
        return {};
    }
    fn disableRawMode(_: TerminalState) void {}
} else struct {
    fn enableRawMode() !TerminalState {
        const stdin_fd = std.io.getStdIn().handle;
        const original = try std.posix.tcgetattr(stdin_fd);

        var raw = original;
        raw.lflag.ICANON = false;
        raw.lflag.ECHO = false;
        raw.cc[@intFromEnum(std.posix.V.MIN)] = 1;
        raw.cc[@intFromEnum(std.posix.V.TIME)] = 0;

        try std.posix.tcsetattr(stdin_fd, .NOW, raw);
        return original;
    }
    fn disableRawMode(original: TerminalState) void {
        const stdin_fd = std.io.getStdIn().handle;
        std.posix.tcsetattr(stdin_fd, .NOW, original) catch {};
    }
};

fn clearLine() void {
    const stdout = std.io.getStdOut().writer();
    stdout.writeAll("\x1b[2K\x1b[1G") catch {};
}

fn moveCursorUp(lines: usize) void {
    const stdout = std.io.getStdOut().writer();
    if (lines > 0) {
        stdout.print("\x1b[{d}A", .{lines}) catch {};
    }
}

fn hideCursor() void {
    const stdout = std.io.getStdOut().writer();
    stdout.writeAll("\x1b[?25l") catch {};
}

fn showCursor() void {
    const stdout = std.io.getStdOut().writer();
    stdout.writeAll("\x1b[?25h") catch {};
}

fn printSelector(comptime options: []const []const u8, comptime descriptions: []const []const u8, selected: usize) !void {
    const stdout = std.io.getStdOut().writer();

    for (options, 0..) |option, i| {
        if (i == selected) {
            try stdout.print("  {s}❯ {s}{s}{s}  {s}{s}{s}\n", .{
                Color.bold_cyan,
                Color.bold,
                option,
                Color.reset,
                Color.dim,
                descriptions[i],
                Color.reset,
            });
        } else {
            try stdout.print("  {s}  {s}{s}  {s}{s}{s}\n", .{
                Color.dim,
                option,
                Color.reset,
                Color.dim,
                descriptions[i],
                Color.reset,
            });
        }
    }
}

fn selectOption(comptime prompt: []const u8, comptime options: []const []const u8, comptime descriptions: []const []const u8) !usize {
    const stdout = std.io.getStdOut().writer();
    const stdin = std.io.getStdIn();

    var selected: usize = 0;
    const total_lines = options.len + 1;

    try stdout.print("{s}?{s} {s}{s}{s}\n", .{
        Color.bold_green,
        Color.reset,
        Color.bold,
        prompt,
        Color.reset,
    });

    try printSelector(options, descriptions, selected);

    hideCursor();
    defer showCursor();

    while (true) {
        var buf: [3]u8 = undefined;
        const bytes_read = stdin.read(&buf) catch break;

        if (bytes_read == 0) break;

        var needs_redraw = false;

        if (bytes_read >= 3 and buf[0] == 0x1b and buf[1] == '[') {
            switch (buf[2]) {
                'A' => {
                    if (selected > 0) {
                        selected -= 1;
                        needs_redraw = true;
                    }
                },
                'B' => {
                    if (selected < options.len - 1) {
                        selected += 1;
                        needs_redraw = true;
                    }
                },
                else => {},
            }
        } else if (buf[0] == '\n' or buf[0] == '\r') {
            break;
        } else if (buf[0] == 'j' and selected < options.len - 1) {
            selected += 1;
            needs_redraw = true;
        } else if (buf[0] == 'k' and selected > 0) {
            selected -= 1;
            needs_redraw = true;
        } else if (buf[0] == 'q' or buf[0] == 0x03) {
            return error.UserCancelled;
        }

        if (needs_redraw) {
            moveCursorUp(options.len);
            try printSelector(options, descriptions, selected);
        }
    }

    moveCursorUp(total_lines);

    for (0..total_lines) |_| {
        clearLine();
        try stdout.writeAll("\n");
    }

    moveCursorUp(total_lines);
    try stdout.print("{s}✓{s} {s}{s}{s} {s}›{s} {s}{s}{s}\n", .{
        Color.bold_green,
        Color.reset,
        Color.bold,
        prompt,
        Color.reset,
        Color.dim,
        Color.reset,
        Color.cyan,
        options[selected],
        Color.reset,
    });

    return selected;
}

fn selectUIType() !UIType {
    const options = [_][]const u8{ "Chat", "Blank" };
    const descriptions = [_][]const u8{
        "Simple chat interface for your agent",
        "Blank canvas to build your own UI",
    };

    const selected = try selectOption("What UI do you want?", &options, &descriptions);

    return switch (selected) {
        0 => .chat,
        1 => .blank,
        else => unreachable,
    };
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
        const original_termios = terminal.enableRawMode() catch {
            std.debug.print("Warning: Could not enable raw mode for interactive input\n", .{});
            return;
        };
        defer terminal.disableRawMode(original_termios);

        const ui_type = selectUIType() catch |err| {
            if (err == error.UserCancelled) {
                showCursor();
                std.debug.print("\n{s}Cancelled.{s}\n", .{ Color.dim, Color.reset });
                return;
            }
            return err;
        };

        terminal.disableRawMode(original_termios);

        // Check if ui/ already exists
        if (cwd.access("ui", .{})) |_| {
            if (!confirmOverwrite("ui")) {
                std.debug.print("Aborted.\n", .{});
                return;
            }
            try cwd.deleteTree("ui");
        } else |_| {}

        try utils.fetchBlueprint(allocator, cwd_path, "ui", ui_type.url());

        try stdout.print("\n{s}✓{s} {s}Added {s} UI blueprint{s}\n\n", .{
            Color.bold_green,
            Color.reset,
            Color.bold,
            ui_type.label(),
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
