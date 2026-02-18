const std = @import("std");
const fs = std.fs;

const utils = @import("../utils.zig");
const timbal_version = @import("../version.zig").timbal_version;

const Color = utils.Color;

const ProjectType = utils.ProjectType;

const UIType = enum {
    chat,
    blank,
    none,
};

fn printUsageWithError(err: []const u8) !void {
    const stderr = std.io.getStdErr().writer();
    try stderr.print("{s}\n\n", .{err});
    try printUsage();
}

fn printUsage() !void {
    const stderr = std.io.getStdErr().writer();
    try stderr.writeAll("Create a new timbal project with interactive setup.\n" ++
        "\n" ++
        "\x1b[1;32mUsage: \x1b[1;36mtimbal create \x1b[0;36m[OPTIONS] [PATH]\n" ++
        "\n" ++
        "\x1b[1;32mArguments:\n" ++
        "    \x1b[0;36m[PATH]\x1b[0m The path where the project will be created (default: current directory)\n" ++
        "\n" ++
        "\x1b[1;32mOptions:\n" ++
        "    \x1b[1;36m--template <URL>\x1b[0m Use a template from a URL\n" ++
        "\n" ++
        "\x1b[1;32mGlobal options:\n" ++
        "    \x1b[1;36m-q\x1b[0m, \x1b[1;36m--quiet      \x1b[0mDo not print any output\n" ++
        "    \x1b[1;36m-v\x1b[0m, \x1b[1;36m--verbose    \x1b[0mUse verbose output\n" ++
        "    \x1b[1;36m-h\x1b[0m, \x1b[1;36m--help       \x1b[0mDisplay the concise help for this command\n" ++
        "    \x1b[1;36m-V\x1b[0m, \x1b[1;36m--version    \x1b[0mDisplay the timbal version\n" ++
        "\n");
}

const Editor = enum {
    zed,
    cursor,
    windsurf,
    code,

    fn command(self: Editor) []const u8 {
        return switch (self) {
            .zed => "zed",
            .cursor => "cursor",
            .windsurf => "windsurf",
            .code => "code",
        };
    }

    fn displayName(self: Editor) []const u8 {
        return switch (self) {
            .zed => "Zed",
            .cursor => "Cursor",
            .windsurf => "Windsurf",
            .code => "VS Code",
        };
    }
};

const ProjectConfig = struct {
    project_type: ProjectType,
    ui_type: UIType,
    path: []const u8, // absolute path for display
    relative_path: []const u8, // original path for commands
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

fn printBanner() !void {
    const stdout = std.io.getStdOut().writer();
    try stdout.writeAll("\n");
    try stdout.print("  {s}Create a new Timbal project{s}\n", .{ Color.bold_cyan, Color.reset });
    try stdout.print("  {s}───────────────────────────{s}\n\n", .{ Color.dim, Color.reset });
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
    const total_lines = options.len + 1; // prompt + options

    // Print prompt
    try stdout.print("{s}?{s} {s}{s}{s}\n", .{
        Color.bold_green,
        Color.reset,
        Color.bold,
        prompt,
        Color.reset,
    });

    // Initial render
    try printSelector(options, descriptions, selected);

    hideCursor();
    defer showCursor();

    // Handle input
    while (true) {
        var buf: [3]u8 = undefined;
        const bytes_read = stdin.read(&buf) catch break;

        if (bytes_read == 0) break;

        var needs_redraw = false;

        // Check for arrow keys (escape sequences)
        if (bytes_read >= 3 and buf[0] == 0x1b and buf[1] == '[') {
            switch (buf[2]) {
                'A' => { // Up arrow
                    if (selected > 0) {
                        selected -= 1;
                        needs_redraw = true;
                    }
                },
                'B' => { // Down arrow
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
            // Move up to redraw options
            moveCursorUp(options.len);
            try printSelector(options, descriptions, selected);
        }
    }

    // Clear the selector and show final result
    moveCursorUp(total_lines);

    // Clear all lines we used
    for (0..total_lines) |_| {
        clearLine();
        try stdout.writeAll("\n");
    }

    // Move back up and print final selection
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

fn selectProjectType() !ProjectType {
    const options = [_][]const u8{ "Agent", "Workflow" };
    const descriptions = [_][]const u8{
        "LLM autonomously decides execution paths and tool usage",
        "Explicit step-by-step workflow with data mapping",
    };

    const selected = try selectOption("What do you want to build?", &options, &descriptions);

    return switch (selected) {
        0 => .agent,
        1 => .workflow,
        else => unreachable,
    };
}

fn selectUI() !UIType {
    const options = [_][]const u8{ "Chat", "Blank", "None" };
    const descriptions = [_][]const u8{
        "Simple chat interface for your agent",
        "Blank canvas to build your own UI",
        "API only, no user interface",
    };

    const selected = try selectOption("What UI do you want for your project?", &options, &descriptions);

    return switch (selected) {
        0 => .chat,
        1 => .blank,
        2 => .none,
        else => unreachable,
    };
}

fn isCommandAvailable(allocator: std.mem.Allocator, cmd: []const u8) bool {
    const which_cmd = if (is_windows) "where" else "which";
    const result = std.process.Child.run(.{
        .allocator = allocator,
        .argv = &[_][]const u8{ which_cmd, cmd },
    }) catch return false;
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    return result.term.Exited == 0;
}

fn detectAvailableEditors(allocator: std.mem.Allocator) []const Editor {
    var available = std.ArrayList(Editor).init(allocator);
    const editors = [_]Editor{ .zed, .cursor, .windsurf, .code };

    for (editors) |editor| {
        if (isCommandAvailable(allocator, editor.command())) {
            available.append(editor) catch continue;
        }
    }

    return available.toOwnedSlice() catch &[_]Editor{};
}

fn selectOpenProject(available_editors: []const Editor) !?Editor {
    if (available_editors.len == 0) {
        return null;
    }

    // Build options: each available editor + "No"
    var options: [5][]const u8 = undefined;
    var descriptions: [5][]const u8 = undefined;

    for (available_editors, 0..) |editor, i| {
        options[i] = editor.displayName();
        descriptions[i] = switch (editor) {
            .zed => "Open with Zed",
            .cursor => "Open with Cursor",
            .windsurf => "Open with Windsurf",
            .code => "Open with VS Code",
        };
    }
    options[available_editors.len] = "No";
    descriptions[available_editors.len] = "I'll open it myself";

    const total_options = available_editors.len + 1;

    const stdout = std.io.getStdOut().writer();
    const stdin = std.io.getStdIn();

    var selected: usize = 0;
    const total_lines = total_options + 1; // prompt + options

    // Print prompt
    try stdout.print("{s}?{s} {s}Open your new project?{s}\n", .{
        Color.bold_green,
        Color.reset,
        Color.bold,
        Color.reset,
    });

    // Initial render
    for (0..total_options) |i| {
        if (i == selected) {
            try stdout.print("  {s}❯ {s}{s}{s}  {s}{s}{s}\n", .{
                Color.bold_cyan,
                Color.bold,
                options[i],
                Color.reset,
                Color.dim,
                descriptions[i],
                Color.reset,
            });
        } else {
            try stdout.print("  {s}  {s}{s}  {s}{s}{s}\n", .{
                Color.dim,
                options[i],
                Color.reset,
                Color.dim,
                descriptions[i],
                Color.reset,
            });
        }
    }

    hideCursor();
    defer showCursor();

    // Handle input
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
                    if (selected < total_options - 1) {
                        selected += 1;
                        needs_redraw = true;
                    }
                },
                else => {},
            }
        } else if (buf[0] == '\n' or buf[0] == '\r') {
            break;
        } else if (buf[0] == 'j' and selected < total_options - 1) {
            selected += 1;
            needs_redraw = true;
        } else if (buf[0] == 'k' and selected > 0) {
            selected -= 1;
            needs_redraw = true;
        } else if (buf[0] == 'q' or buf[0] == 0x03) {
            return error.UserCancelled;
        }

        if (needs_redraw) {
            moveCursorUp(total_options);
            for (0..total_options) |i| {
                if (i == selected) {
                    try stdout.print("  {s}❯ {s}{s}{s}  {s}{s}{s}\n", .{
                        Color.bold_cyan,
                        Color.bold,
                        options[i],
                        Color.reset,
                        Color.dim,
                        descriptions[i],
                        Color.reset,
                    });
                } else {
                    try stdout.print("  {s}  {s}{s}  {s}{s}{s}\n", .{
                        Color.dim,
                        options[i],
                        Color.reset,
                        Color.dim,
                        descriptions[i],
                        Color.reset,
                    });
                }
            }
        }
    }

    // Clear and show final result
    moveCursorUp(total_lines);
    for (0..total_lines) |_| {
        clearLine();
        try stdout.writeAll("\n");
    }
    moveCursorUp(total_lines);

    try stdout.print("{s}✓{s} {s}Open your new project?{s} {s}›{s} {s}{s}{s}\n", .{
        Color.bold_green,
        Color.reset,
        Color.bold,
        Color.reset,
        Color.dim,
        Color.reset,
        Color.cyan,
        options[selected],
        Color.reset,
    });

    // Return selected editor or null if "No"
    if (selected == available_editors.len) {
        return null;
    }
    return available_editors[selected];
}

fn openEditor(allocator: std.mem.Allocator, editor: Editor, path: []const u8) !void {
    const readme_path = try std.fmt.allocPrint(allocator, "{s}/README.md", .{path});
    defer allocator.free(readme_path);
    const argv = [_][]const u8{ editor.command(), path, readme_path };
    var child = std.process.Child.init(&argv, allocator);
    _ = child.spawn() catch |err| {
        std.debug.print("Failed to open editor: {}\n", .{err});
        return err;
    };
}

fn printSummary(config: ProjectConfig) !void {
    const stdout = std.io.getStdOut().writer();

    const project_type_str = switch (config.project_type) {
        .agent => "Agent",
        .workflow => "Workflow",
    };

    const ui_str = switch (config.ui_type) {
        .chat => "Chat",
        .blank => "Blank",
        .none => "None",
    };

    try stdout.writeAll("\n");
    try stdout.print("{s}  Project Summary{s}\n", .{ Color.bold, Color.reset });
    try stdout.print("  {s}─────────────────{s}\n", .{ Color.dim, Color.reset });
    try stdout.print("  {s}Location:{s}  {s}\n", .{ Color.dim, Color.reset, config.path });
    try stdout.print("  {s}Type:{s}      {s}{s}{s}\n", .{ Color.dim, Color.reset, Color.cyan, project_type_str, Color.reset });
    try stdout.print("  {s}UI:{s}        {s}{s}{s}\n", .{ Color.dim, Color.reset, Color.cyan, ui_str, Color.reset });
    try stdout.writeAll("\n");
}

fn printSuccess() !void {
    const stdout = std.io.getStdOut().writer();

    try stdout.print("\n{s}✓{s} {s}Project created successfully!{s}\n", .{
        Color.bold_green,
        Color.reset,
        Color.bold,
        Color.reset,
    });
}

fn initGitRepo(allocator: std.mem.Allocator, path: []const u8) !void {
    const result = std.process.Child.run(.{
        .allocator = allocator,
        .argv = &[_][]const u8{ "git", "init" },
        .cwd = path,
    }) catch |err| {
        std.debug.print("Warning: Failed to initialize git repository: {}\n", .{err});
        return;
    };
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);

    if (result.term.Exited != 0) {
        std.debug.print("Warning: git init failed\n", .{});
    }
}

fn createProjectStructure(allocator: std.mem.Allocator, app_dir: fs.Dir, config: ProjectConfig) !void {
    // Extract project name from path
    const project_name = blk: {
        if (std.mem.lastIndexOf(u8, config.path, "/")) |idx| {
            break :blk config.path[idx + 1 ..];
        }
        break :blk config.path;
    };

    // Download blueprints
    if (config.ui_type != .none) {
        const ui_url = switch (config.ui_type) {
            .chat => utils.blueprint_ui_simple_chat_url,
            .blank => utils.blueprint_ui_url,
            .none => unreachable,
        };
        try utils.fetchBlueprint(allocator, config.path, "ui", ui_url);
    }

    try utils.fetchBlueprint(allocator, config.path, "api", utils.blueprint_api_url);

    // Create workforce member
    const funny_name = try utils.addWorkforceMember(allocator, app_dir, project_name, config.project_type);
    defer allocator.free(funny_name);

    // Create .gitignore in the project root
    const gitignore_content =
        \\# Claude
        \\.claude/
        \\CLAUDE.md
        \\
        \\# Environment
        \\.env
        \\
        \\# Logs
        \\logs/
        \\*.log
        \\npm-debug.log*
        \\yarn-debug.log*
        \\yarn-error.log*
        \\pnpm-debug.log*
        \\lerna-debug.log*
        \\
        \\# Python
        \\__pycache__/
        \\*.pyc
        \\*.pyo
        \\build/
        \\wheels/
        \\*.egg-info/
        \\.pytest_cache/
        \\.ruff_cache/
        \\_version.py
        \\.venv/
        \\
        \\# Node
        \\node_modules/
        \\dist/
        \\dist-ssr/
        \\*.local
        \\
        \\# IDEs
        \\.vscode/
        \\.idea/
        \\*.suo
        \\*.ntvs*
        \\*.njsproj
        \\*.sln
        \\*.sw?
        \\*.swp
        \\
        \\# OS
        \\.DS_Store
        \\
    ;

    const gitignore_file = try app_dir.createFile(".gitignore", .{});
    defer gitignore_file.close();
    try gitignore_file.writeAll(gitignore_content);

    // Create README.md
    const app_py_display = switch (config.project_type) {
        .agent => "agent.py",
        .workflow => "workflow.py",
    };
    const ui_section = if (config.ui_type != .none)
        \\ui/              - React + Vite + TypeScript + shadcn (Bun)
        \\
    else
        "";
    const ui_note = if (config.ui_type != .none)
        \\
        \\> Do not rename or move the `api/`, `ui/`, or `workforce/` directories.
        \\> You are free to edit the contents inside each one.
        \\
        \\- **api/** - Elysia server running on Bun
        \\- **ui/** - React + Vite + TypeScript + shadcn running on Bun
        \\- **workforce/** - Your timbal components (Python)
        \\
    else
        \\
        \\> Do not rename or move the `api/` or `workforce/` directories.
        \\> You are free to edit the contents inside each one.
        \\
        \\- **api/** - Elysia server running on Bun
        \\- **workforce/** - Your timbal components (Python)
        \\
    ;
    const readme_content = try std.fmt.allocPrint(allocator,
        \\# {s}
        \\
        \\## Project Structure
        \\
        \\```
        \\{s}api/             - Elysia API server (Bun)
        \\workforce/       - Your timbal components
        \\  {s}/
        \\    {s}  - Main app logic
        \\    timbal.yaml    - Timbal configuration
        \\    pyproject.toml - Python dependencies
        \\```
        \\{s}
        \\## Getting Started
        \\
        \\Connect the Timbal MCP to your AI-powered IDE to help you build and customize your project:
        \\
        \\https://docs.timbal.ai/mcp-integration
        \\
    , .{ project_name, ui_section, funny_name, app_py_display, ui_note });
    defer allocator.free(readme_content);
    const readme_file = try app_dir.createFile("README.md", .{});
    defer readme_file.close();
    try readme_file.writeAll(readme_content);

    // Initialize git repository
    try initGitRepo(allocator, config.path);
}

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
        // Disable canonical mode and echo
        raw.lflag.ICANON = false;
        raw.lflag.ECHO = false;
        // Set minimum bytes to read
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

pub fn run(allocator: std.mem.Allocator, args: []const []const u8) !void {
    var arg_path: ?[]const u8 = null;
    var template: ?[]const u8 = null;
    var verbose: bool = false;
    var quiet: bool = false;

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
        } else if (std.mem.eql(u8, arg, "--template")) {
            i += 1;
            if (i >= args.len) {
                try printUsageWithError("Error: --template requires a URL argument");
                return;
            }
            template = args[i];
        } else if (!std.mem.startsWith(u8, arg, "-")) {
            if (arg_path != null) {
                try printUsageWithError("Error: multiple target paths provided");
                return;
            }
            arg_path = arg;
        } else {
            try printUsageWithError("Error: unknown option");
            return;
        }
    }

    // TODO: use verbose for detailed output
    // TODO: implement template support
    if (verbose) {
        std.debug.print("Verbose mode enabled\n", .{});
    }
    if (template) |t| {
        std.debug.print("Using template: {s}\n", .{t});
    }

    const target_path = arg_path orelse ".";

    // Get absolute path for display
    const cwd = fs.cwd();
    const use_current_dir = std.mem.eql(u8, target_path, ".");

    if (quiet) {
        // In quiet mode, create dir and use defaults
        var app_dir: fs.Dir = undefined;
        if (use_current_dir) {
            app_dir = cwd;
        } else {
            cwd.makePath(target_path) catch |err| {
                std.debug.print("Error creating directory: {}\n", .{err});
                return;
            };
            app_dir = try cwd.openDir(target_path, .{});
        }
        const path = try app_dir.realpathAlloc(allocator, ".");
        defer allocator.free(path);
        std.debug.print("Creating project in {s}...\n", .{path});
        return;
    }

    // Enable raw mode for arrow key input
    const original_termios = terminal.enableRawMode() catch {
        std.debug.print("Warning: Could not enable raw mode for interactive input\n", .{});
        return;
    };
    defer terminal.disableRawMode(original_termios);

    // Show banner
    try printBanner();

    // Interactive prompts (before creating any directories)
    const project_type = selectProjectType() catch |err| {
        if (err == error.UserCancelled) {
            showCursor();
            std.debug.print("\n{s}Cancelled.{s}\n", .{ Color.dim, Color.reset });
            return;
        }
        return err;
    };

    const ui_type = selectUI() catch |err| {
        if (err == error.UserCancelled) {
            showCursor();
            std.debug.print("\n{s}Cancelled.{s}\n", .{ Color.dim, Color.reset });
            return;
        }
        return err;
    };

    // All prompts completed — now create the directory
    var app_dir: fs.Dir = undefined;
    if (use_current_dir) {
        app_dir = cwd;
    } else {
        cwd.makePath(target_path) catch |err| {
            std.debug.print("Error creating directory: {}\n", .{err});
            return;
        };
        app_dir = try cwd.openDir(target_path, .{});
    }

    const path = try app_dir.realpathAlloc(allocator, ".");
    defer allocator.free(path);

    const config = ProjectConfig{
        .project_type = project_type,
        .ui_type = ui_type,
        .path = path,
        .relative_path = target_path,
    };

    // Create the project structure
    try createProjectStructure(allocator, app_dir, config);

    // Show success message and summary
    try printSuccess();
    try printSummary(config);

    // Detect available editors
    const available_editors = detectAvailableEditors(allocator);
    defer allocator.free(available_editors);

    // Ask if user wants to open the project
    const selected_editor = selectOpenProject(available_editors) catch |err| {
        if (err == error.UserCancelled) {
            showCursor();
            std.debug.print("\n{s}Cancelled.{s}\n", .{ Color.dim, Color.reset });
            return;
        }
        return err;
    };

    // Open editor if selected
    if (selected_editor) |editor| {
        try openEditor(allocator, editor, target_path);
    }
}
