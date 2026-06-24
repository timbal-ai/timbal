const std = @import("std");
const fs = std.fs;

const utils = @import("../utils.zig");

const Color = utils.Color;

const ProjectType = utils.ProjectType;

fn printUsageWithError(err: []const u8) !void {
    const stderr = std.io.getStdErr().writer();
    try stderr.print("{s}\n\n", .{err});
    try printUsage();
}

fn printUsage() !void {
    const stderr = std.io.getStdErr().writer();
    try stderr.writeAll("Create a new timbal project (interactive by default).\n" ++
        "\n" ++
        "\x1b[1;32mUsage: \x1b[1;36mtimbal create \x1b[0;36m[OPTIONS] <PATH>\n" ++
        "\n" ++
        "\x1b[1;32mArguments:\n" ++
        "    \x1b[0;33m<PATH>\x1b[0m  Project directory (use '.' for the current directory)\n" ++
        "\n" ++
        "\x1b[1;32mOptions:\n" ++
        "    \x1b[1;36m--agent <NAME>\x1b[0m    Add an agent at workforce/<NAME>/ (repeat for multiple)\n" ++
        "    \x1b[1;36m--workflow <NAME>\x1b[0m Add a workflow at workforce/<NAME>/ (repeat for multiple)\n" ++
        "    \x1b[1;36m--with-ui\x1b[0m         Include the default web UI (requires at least one agent)\n" ++
        "\n" ++
        utils.global_options_help ++
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

const WorkforceMemberSpec = struct {
    name: []const u8,
    project_type: ProjectType,
};

const ProjectConfig = struct {
    members: []WorkforceMemberSpec,
    include_ui: bool,
    path: []const u8,
    relative_path: []const u8,

    pub fn deinit(self: *ProjectConfig, allocator: std.mem.Allocator) void {
        for (self.members) |m| allocator.free(m.name);
        allocator.free(self.members);
    }

    pub fn hasAgent(self: ProjectConfig) bool {
        for (self.members) |m| {
            if (m.project_type == .agent) return true;
        }
        return false;
    }
};

fn projectDirName(path: []const u8) []const u8 {
    const last_fwd = std.mem.lastIndexOf(u8, path, "/");
    const last_back = std.mem.lastIndexOf(u8, path, "\\");
    const last_sep = if (last_fwd) |f| if (last_back) |b| @max(f, b) else f else last_back;
    if (last_sep) |idx| {
        return path[idx + 1 ..];
    }
    return path;
}

fn promptMemberName(allocator: std.mem.Allocator, kind_label: []const u8, term_state: TerminalState) ![]u8 {
    const stdout = std.io.getStdOut().writer();
    const stderr = std.io.getStdErr().writer();
    const stdin = std.io.getStdIn().reader();

    while (true) {
        try stdout.print("\n{s}?{s} {s}{s} name{s}: ", .{
            Color.bold_green,
            Color.reset,
            Color.bold,
            kind_label,
            Color.reset,
        });

        terminal.disableRawMode(term_state);
        defer _ = terminal.enableRawMode() catch {};

        var buf: [256]u8 = undefined;
        // EOF (Ctrl+D / closed stdin) yields null. Treat it as cancellation so
        // we don't loop forever printing "Name is required." — match the other
        // interactive prompts that return UserCancelled.
        const line = try stdin.readUntilDelimiterOrEof(&buf, '\n') orelse return error.UserCancelled;
        const trimmed = std.mem.trim(u8, line, " \t\r");
        if (trimmed.len == 0) {
            try stderr.writeAll("Name is required.\n");
            continue;
        }

        return utils.prepareWorkforceMemberName(allocator, trimmed) catch |err| {
            utils.printWorkforceNameError(err, trimmed);
            try stderr.writeAll("Try again.\n");
            continue;
        };
    }
}

fn printAppendMemberError(err: anyerror, name: []const u8) void {
    switch (err) {
        error.InvalidWorkforceName, error.ReservedWorkforceName => utils.printWorkforceNameError(err, name),
        error.DuplicateMemberName => {
            const stderr = std.io.getStdErr().writer();
            stderr.print("Error: duplicate workforce member name '{s}'.\n", .{name}) catch {};
        },
        else => {},
    }
}

fn appendMember(
    allocator: std.mem.Allocator,
    list: *std.ArrayList(WorkforceMemberSpec),
    project_type: ProjectType,
    name: []const u8,
) !void {
    const owned = try utils.prepareWorkforceMemberName(allocator, name);
    errdefer allocator.free(owned);

    for (list.items) |existing| {
        if (std.mem.eql(u8, existing.name, owned)) {
            return error.DuplicateMemberName;
        }
    }
    try list.append(.{ .name = owned, .project_type = project_type });
}

fn collectMembersInteractive(allocator: std.mem.Allocator, term_state: TerminalState) ![]WorkforceMemberSpec {
    var list = std.ArrayList(WorkforceMemberSpec).init(allocator);
    errdefer {
        for (list.items) |m| allocator.free(m.name);
        list.deinit();
    }

    const options = [_][]const u8{ "Add agent", "Add workflow", "Continue" };
    const descriptions = [_][]const u8{
        "Scaffold workforce/<name>/ with agent.py",
        "Scaffold workforce/<name>/ with workflow.py",
        "Finish adding members (api/ is always created)",
    };

    while (true) {
        const selected = try selectOption("Add workforce members", &options, &descriptions);
        switch (selected) {
            0 => {
                const name = try promptMemberName(allocator, "Agent", term_state);
                defer allocator.free(name);
                appendMember(allocator, &list, .agent, name) catch |err| switch (err) {
                    // Recoverable: report and stay in the menu so the user can
                    // pick another name instead of losing the whole session.
                    error.DuplicateMemberName, error.InvalidWorkforceName, error.ReservedWorkforceName => {
                        printAppendMemberError(err, name);
                        continue;
                    },
                    else => |e| return e,
                };
            },
            1 => {
                const name = try promptMemberName(allocator, "Workflow", term_state);
                defer allocator.free(name);
                appendMember(allocator, &list, .workflow, name) catch |err| switch (err) {
                    error.DuplicateMemberName, error.InvalidWorkforceName, error.ReservedWorkforceName => {
                        printAppendMemberError(err, name);
                        continue;
                    },
                    else => |e| return e,
                };
            },
            2 => return try list.toOwnedSlice(),
            else => unreachable,
        }
    }
}

fn resolveIncludeUi(
    with_ui_flag: bool,
    with_ui_set: bool,
    members: []const WorkforceMemberSpec,
) !bool {
    const any_agent = blk: {
        for (members) |m| {
            if (m.project_type == .agent) break :blk true;
        }
        break :blk false;
    };

    if (with_ui_set and with_ui_flag and !any_agent) return error.WithUiRequiresAgent;
    if (with_ui_set) return with_ui_flag and any_agent;
    return false;
}

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

fn optionColumnWidth(comptime opts: []const []const u8) usize {
    var w: usize = 0;
    for (opts) |o| w = @max(w, o.len);
    return w;
}

fn writeSpaces(stdout: anytype, n: usize) !void {
    var i: usize = 0;
    while (i < n) : (i += 1) try stdout.writeAll(" ");
}

fn printSelector(comptime options: []const []const u8, comptime descriptions: []const []const u8, selected: usize) !void {
    const stdout = std.io.getStdOut().writer();
    const col_w = optionColumnWidth(options);

    for (options, 0..) |option, i| {
        const pad = col_w - option.len;
        if (i == selected) {
            try stdout.print("  {s}❯ {s}{s}{s}", .{
                Color.bold_cyan,
                Color.bold,
                option,
                Color.reset,
            });
            try writeSpaces(stdout, pad);
            try stdout.print("  {s}{s}{s}\n", .{
                Color.dim,
                descriptions[i],
                Color.reset,
            });
        } else {
            try stdout.print("  {s}  {s}{s}", .{
                Color.dim,
                option,
                Color.reset,
            });
            try writeSpaces(stdout, pad);
            try stdout.print("  {s}{s}{s}\n", .{
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

fn selectWantUI() !bool {
    const options = [_][]const u8{ "Yes", "No" };
    const descriptions = [_][]const u8{
        "Our default web UI—sensible defaults and config so you can customize and ship faster",
        "Skip the web UI for now (API only)",
    };

    const selected = try selectOption("Do you want a UI?", &options, &descriptions);

    return selected == 0;
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

    // Find the longest option name for padding.
    var max_name_len: usize = 0;
    for (0..total_options) |i| {
        if (options[i].len > max_name_len) max_name_len = options[i].len;
    }

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
        const padding = max_name_len - options[i].len;
        if (i == selected) {
            try stdout.print("  {s}❯ {s}{s}{s}", .{
                Color.bold_cyan,
                Color.bold,
                options[i],
                Color.reset,
            });
            try stdout.writeByteNTimes(' ', padding + 2);
            try stdout.print("{s}{s}{s}\n", .{ Color.dim, descriptions[i], Color.reset });
        } else {
            try stdout.print("  {s}  {s}{s}", .{
                Color.dim,
                options[i],
                Color.reset,
            });
            try stdout.writeByteNTimes(' ', padding + 2);
            try stdout.print("{s}{s}{s}\n", .{ Color.dim, descriptions[i], Color.reset });
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
                const padding = max_name_len - options[i].len;
                if (i == selected) {
                    try stdout.print("  {s}❯ {s}{s}{s}", .{
                        Color.bold_cyan,
                        Color.bold,
                        options[i],
                        Color.reset,
                    });
                    try stdout.writeByteNTimes(' ', padding + 2);
                    try stdout.print("{s}{s}{s}\n", .{ Color.dim, descriptions[i], Color.reset });
                } else {
                    try stdout.print("  {s}  {s}{s}", .{
                        Color.dim,
                        options[i],
                        Color.reset,
                    });
                    try stdout.writeByteNTimes(' ', padding + 2);
                    try stdout.print("{s}{s}{s}\n", .{ Color.dim, descriptions[i], Color.reset });
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

    const ui_str = if (config.include_ui) "Yes (default web UI)" else if (config.hasAgent()) "No" else "No (workflows only)";

    try stdout.writeAll("\n");
    try stdout.print("{s}  Project Summary{s}\n", .{ Color.bold, Color.reset });
    try stdout.print("  {s}─────────────────{s}\n", .{ Color.dim, Color.reset });
    try stdout.print("  {s}Location:{s}  {s}\n", .{ Color.dim, Color.reset, config.path });
    try stdout.print("  {s}UI:{s}        {s}{s}{s}\n", .{ Color.dim, Color.reset, Color.cyan, ui_str, Color.reset });
    if (config.members.len == 0) {
        try stdout.print("  {s}Workforce:{s} {s}(none — use timbal add agent|workflow <name>){s}\n", .{ Color.dim, Color.reset, Color.dim, Color.reset });
    } else {
        try stdout.print("  {s}Workforce:{s}\n", .{ Color.dim, Color.reset });
        for (config.members) |m| {
            const kind = switch (m.project_type) {
                .agent => "agent",
                .workflow => "workflow",
            };
            try stdout.print("    {s}{s}{s}  {s}{s}{s}\n", .{ Color.cyan, kind, Color.reset, Color.cyan, m.name, Color.reset });
        }
    }
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
        std.debug.print("Error: failed to initialize git repository: {}\n", .{err});
        return err;
    };
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);

    switch (result.term) {
        .Exited => |code| if (code != 0) {
            std.debug.print("Error: git init failed (exit {d}).\n", .{code});
            return error.GitInitFailed;
        },
        else => {
            std.debug.print("Error: git init terminated abnormally.\n", .{});
            return error.GitInitFailed;
        },
    }
}

const CreateScaffoldOptions = struct {
    fetch_blueprints: bool = true,
    init_git: bool = true,
};

/// Register a top-level scaffold entry for rollback, but only if it does not
/// already exist. This keeps rollback safe in `timbal create .` (or an empty
/// reused directory) where pre-existing user files must never be deleted: we
/// only ever remove entries this scaffold actually created.
fn markForRollback(
    allocator: std.mem.Allocator,
    app_dir: fs.Dir,
    created: *std.ArrayList([]u8),
    rel_path: []const u8,
) !void {
    app_dir.access(rel_path, .{}) catch |err| {
        if (err == error.FileNotFound) {
            try created.append(try allocator.dupe(u8, rel_path));
        }
    };
}

fn createProjectStructure(
    allocator: std.mem.Allocator,
    app_dir: fs.Dir,
    config: ProjectConfig,
    opts: CreateScaffoldOptions,
) !void {
    const project_name = projectDirName(config.path);

    // Track newly-created top-level entries so that if scaffolding fails
    // partway through (a member name is invalid, a blueprint fetch fails,
    // git init errors, ...) we roll back to the starting state. Otherwise the
    // half-written project trips the non-empty directory guard in openAppDir
    // and a `timbal create` retry on the same path can't finish.
    var created = std.ArrayList([]u8).init(allocator);
    defer {
        for (created.items) |p| allocator.free(p);
        created.deinit();
    }
    errdefer {
        // Best-effort, reverse order; ignore errors so the original failure
        // is what propagates.
        var idx = created.items.len;
        while (idx > 0) {
            idx -= 1;
            app_dir.deleteTree(created.items[idx]) catch {};
        }
    }

    if (opts.fetch_blueprints) {
        if (config.include_ui) {
            try markForRollback(allocator, app_dir, &created, "ui");
            try utils.fetchBlueprint(allocator, config.path, "ui", utils.blueprint_ui_simple_chat_url);
        }
        try markForRollback(allocator, app_dir, &created, "api");
        try utils.fetchBlueprint(allocator, config.path, "api", utils.blueprint_api_url);
    } else {
        if (config.include_ui) {
            try markForRollback(allocator, app_dir, &created, "ui");
            try app_dir.makePath("ui");
        }
        try markForRollback(allocator, app_dir, &created, "api");
        try app_dir.makePath("api");
    }

    try markForRollback(allocator, app_dir, &created, "workforce");
    app_dir.makePath("workforce") catch |err| {
        if (err != error.PathAlreadyExists) return err;
    };

    for (config.members) |member| {
        const used_name = utils.addWorkforceMember(allocator, app_dir, project_name, member.project_type, member.name) catch |err| switch (err) {
            error.InvalidWorkforceName, error.ReservedWorkforceName, error.WorkforceMemberExists => {
                utils.printWorkforceNameError(err, member.name);
                return err;
            },
            else => |e| return e,
        };
        defer allocator.free(used_name);
        // Track each member dir explicitly: when `workforce/` already existed
        // it isn't itself slated for rollback, so the members we add must be
        // removed individually.
        try created.append(try std.fmt.allocPrint(allocator, "workforce/{s}", .{used_name}));
    }

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

    try markForRollback(allocator, app_dir, &created, ".gitignore");
    const gitignore_file = try app_dir.createFile(".gitignore", .{});
    defer gitignore_file.close();
    try gitignore_file.writeAll(gitignore_content);

    // Create README.md
    const ui_section = if (config.include_ui)
        \\ui/              - React + Vite + TypeScript + shadcn (Bun)
        \\
    else
        "";
    const ui_note = if (config.include_ui)
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
    const readme_content = if (config.members.len == 1) blk: {
        const member = config.members[0];
        const app_py_display = switch (member.project_type) {
            .agent => "agent.py",
            .workflow => "workflow.py",
        };
        break :blk try std.fmt.allocPrint(allocator,
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
        , .{ project_name, ui_section, member.name, app_py_display, ui_note });
    } else if (config.members.len > 1) try std.fmt.allocPrint(allocator,
        \\# {s}
        \\
        \\## Project Structure
        \\
        \\```
        \\{s}api/             - Elysia API server (Bun)
        \\workforce/       - Your timbal components (one directory per member)
        \\```
        \\{s}
        \\## Getting Started
        \\
        \\Connect the Timbal MCP to your AI-powered IDE to help you build and customize your project:
        \\
        \\https://docs.timbal.ai/mcp-integration
        \\
    , .{ project_name, ui_section, ui_note }) else try std.fmt.allocPrint(allocator,
        \\# {s}
        \\
        \\## Project Structure
        \\
        \\```
        \\{s}api/             - Elysia API server (Bun)
        \\workforce/       - Add agents/workflows with `timbal add`
        \\```
        \\{s}
        \\## Getting Started
        \\
        \\Add your first component:
        \\
        \\```bash
        \\timbal add agent my-agent
        \\```
        \\
        \\Connect the Timbal MCP to your AI-powered IDE to help you build and customize your project:
        \\
        \\https://docs.timbal.ai/mcp-integration
        \\
    , .{ project_name, ui_section, ui_note });
    defer allocator.free(readme_content);
    try markForRollback(allocator, app_dir, &created, "README.md");
    const readme_file = try app_dir.createFile("README.md", .{});
    defer readme_file.close();
    try readme_file.writeAll(readme_content);

    if (opts.init_git) {
        try markForRollback(allocator, app_dir, &created, ".git");
        try initGitRepo(allocator, config.path);
    }
}

const builtin = @import("builtin");

const is_windows = builtin.os.tag == .windows;

// Windows console input mode flags not provided by Zig stdlib.
const ENABLE_PROCESSED_INPUT: u32 = 0x0001;
const ENABLE_LINE_INPUT: u32 = 0x0002;
const ENABLE_ECHO_INPUT: u32 = 0x0004;
const ENABLE_VIRTUAL_TERMINAL_INPUT: u32 = 0x0200;

const CP_UTF8: u32 = 65001;

const WindowsTerminalState = struct {
    stdin_mode: std.os.windows.DWORD,
    stdout_mode: std.os.windows.DWORD,
    output_cp: u32,
};

const TerminalState = if (is_windows) WindowsTerminalState else std.posix.termios;

const terminal = if (is_windows) struct {
    fn enableRawMode() !TerminalState {
        const windows = std.os.windows;

        const stdin_handle = std.io.getStdIn().handle;
        var orig_stdin_mode: windows.DWORD = 0;
        if (windows.kernel32.GetConsoleMode(stdin_handle, &orig_stdin_mode) == 0) {
            return error.GetConsoleModeError;
        }

        // Disable line-buffering, echo, and CTRL+C processing.
        // Enable VT input so arrow keys arrive as \x1b[A/\x1b[B escape sequences.
        var new_stdin_mode = orig_stdin_mode;
        new_stdin_mode &= ~(ENABLE_LINE_INPUT | ENABLE_ECHO_INPUT | ENABLE_PROCESSED_INPUT);
        new_stdin_mode |= ENABLE_VIRTUAL_TERMINAL_INPUT;

        if (windows.kernel32.SetConsoleMode(stdin_handle, new_stdin_mode) == 0) {
            return error.SetConsoleModeError;
        }

        // Enable ANSI/VT processing on stdout so escape codes render correctly.
        const stdout_handle = std.io.getStdOut().handle;
        var orig_stdout_mode: windows.DWORD = 0;
        if (windows.kernel32.GetConsoleMode(stdout_handle, &orig_stdout_mode) == 0) {
            _ = windows.kernel32.SetConsoleMode(stdin_handle, orig_stdin_mode);
            return error.GetConsoleModeError;
        }

        var new_stdout_mode = orig_stdout_mode;
        new_stdout_mode |= windows.ENABLE_VIRTUAL_TERMINAL_PROCESSING;

        if (windows.kernel32.SetConsoleMode(stdout_handle, new_stdout_mode) == 0) {
            _ = windows.kernel32.SetConsoleMode(stdin_handle, orig_stdin_mode);
            return error.SetConsoleModeError;
        }

        // Set console output codepage to UTF-8 so Unicode characters render correctly.
        const orig_output_cp = windows.kernel32.GetConsoleOutputCP();
        _ = windows.kernel32.SetConsoleOutputCP(CP_UTF8);

        return WindowsTerminalState{
            .stdin_mode = orig_stdin_mode,
            .stdout_mode = orig_stdout_mode,
            .output_cp = orig_output_cp,
        };
    }

    fn disableRawMode(original: TerminalState) void {
        const windows = std.os.windows;
        _ = windows.kernel32.SetConsoleMode(std.io.getStdIn().handle, original.stdin_mode);
        _ = windows.kernel32.SetConsoleMode(std.io.getStdOut().handle, original.stdout_mode);
        _ = windows.kernel32.SetConsoleOutputCP(original.output_cp);
    }
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

fn openAppDir(cwd: fs.Dir, target_path: []const u8, use_current_dir: bool) !fs.Dir {
    if (use_current_dir) return cwd;

    // Refuse to scaffold into an existing, non-empty directory: merging into a
    // populated path risks clobbering files and leaves an ambiguous result for
    // callers that pinned this path. An empty existing directory is fine to reuse.
    if (cwd.openDir(target_path, .{ .iterate = true })) |existing_dir| {
        var existing = existing_dir;
        var it = existing.iterate();
        const has_entry = (try it.next()) != null;
        if (has_entry) {
            existing.close();
            const stderr = std.io.getStdErr().writer();
            stderr.print("Error: target directory '{s}' already exists and is not empty.\n", .{target_path}) catch {};
            std.process.exit(2);
        }
        return existing;
    } else |err| switch (err) {
        error.FileNotFound => {},
        else => {
            std.debug.print("Error opening directory: {}\n", .{err});
            return err;
        },
    }

    cwd.makePath(target_path) catch |err| {
        std.debug.print("Error creating directory: {}\n", .{err});
        return err;
    };
    return cwd.openDir(target_path, .{});
}

fn finishCreate(
    allocator: std.mem.Allocator,
    app_dir: fs.Dir,
    config: *ProjectConfig,
    quiet: bool,
) !void {
    try createProjectStructure(allocator, app_dir, config.*, .{});

    if (quiet) {
        const stdout = std.io.getStdOut().writer();
        try stdout.print("{s}\n", .{config.path});
    } else {
        try printSuccess();
        try printSummary(config.*);
    }
}

pub fn run(allocator: std.mem.Allocator, args: []const []const u8) !void {
    var arg_path: ?[]const u8 = null;
    var verbose: bool = false;
    var quiet: bool = false;

    var agent_names = std.ArrayList([]const u8).init(allocator);
    defer {
        for (agent_names.items) |n| allocator.free(n);
        agent_names.deinit();
    }
    var workflow_names = std.ArrayList([]const u8).init(allocator);
    defer {
        for (workflow_names.items) |n| allocator.free(n);
        workflow_names.deinit();
    }

    var with_ui = false;
    var with_ui_set = false;
    var script_mode = false;

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            try printUsage();
            return;
        } else if (std.mem.eql(u8, arg, "-v") or std.mem.eql(u8, arg, "--verbose")) {
            verbose = true;
        } else if (std.mem.eql(u8, arg, "-q") or std.mem.eql(u8, arg, "--quiet")) {
            script_mode = true;
            quiet = true;
        } else if (std.mem.eql(u8, arg, "--with-ui")) {
            script_mode = true;
            with_ui = true;
            with_ui_set = true;
        } else if (std.mem.eql(u8, arg, "--agent")) {
            script_mode = true;
            i += 1;
            if (i >= args.len) {
                try printUsageWithError("Error: --agent requires a name");
                std.process.exit(2);
            }
            try agent_names.append(try allocator.dupe(u8, args[i]));
        } else if (std.mem.eql(u8, arg, "--workflow")) {
            script_mode = true;
            i += 1;
            if (i >= args.len) {
                try printUsageWithError("Error: --workflow requires a name");
                std.process.exit(2);
            }
            try workflow_names.append(try allocator.dupe(u8, args[i]));
        } else if (!std.mem.startsWith(u8, arg, "-")) {
            if (arg_path != null) {
                try printUsageWithError("Error: multiple target paths provided");
                std.process.exit(2);
            }
            arg_path = arg;
        } else {
            try printUsageWithError("Error: unknown option");
            std.process.exit(2);
        }
    }

    if (verbose) {
        std.debug.print("Verbose mode enabled\n", .{});
    }

    const target_path = arg_path orelse {
        try printUsageWithError("Error: a target path is required (use '.' for the current directory)");
        std.process.exit(2);
    };
    const cwd = fs.cwd();
    const use_current_dir = std.mem.eql(u8, target_path, ".");

    var app_dir = try openAppDir(cwd, target_path, use_current_dir);
    defer if (!use_current_dir) app_dir.close();

    const path = try app_dir.realpathAlloc(allocator, ".");
    defer allocator.free(path);

    if (script_mode) {
        var members = std.ArrayList(WorkforceMemberSpec).init(allocator);
        defer {
            for (members.items) |m| allocator.free(m.name);
            members.deinit();
        }

        for (agent_names.items) |name| {
            appendMember(allocator, &members, .agent, name) catch |err| {
                printAppendMemberError(err, name);
                std.process.exit(2);
            };
        }
        for (workflow_names.items) |name| {
            appendMember(allocator, &members, .workflow, name) catch |err| {
                printAppendMemberError(err, name);
                std.process.exit(2);
            };
        }

        const include_ui = resolveIncludeUi(with_ui, with_ui_set, members.items) catch |err| {
            if (err == error.WithUiRequiresAgent) {
                const stderr = std.io.getStdErr().writer();
                try stderr.writeAll("Error: --with-ui requires at least one agent (workflows alone do not use the web UI).\n");
                std.process.exit(2);
            }
            return err;
        };

        var config = ProjectConfig{
            .members = try members.toOwnedSlice(),
            .include_ui = include_ui,
            .path = path,
            .relative_path = target_path,
        };
        defer config.deinit(allocator);

        try finishCreate(allocator, app_dir, &config, quiet);
        return;
    }

    // Interactive mode requires a real terminal. If stdin isn't a TTY (piped,
    // redirected, or running under CI), there's no way to prompt — fail loudly
    // and point at the non-interactive flags instead of silently doing nothing.
    if (!std.io.getStdIn().isTty()) {
        const stderr = std.io.getStdErr().writer();
        try stderr.writeAll("Error: interactive create requires a terminal. " ++
            "Pass --agent <name> and/or --workflow <name> (optionally --with-ui) for non-interactive use.\n");
        std.process.exit(2);
    }

    const original_termios = terminal.enableRawMode() catch {
        const stderr = std.io.getStdErr().writer();
        stderr.writeAll("Error: failed to set up the terminal for interactive input.\n") catch {};
        std.process.exit(1);
    };
    defer terminal.disableRawMode(original_termios);

    try printBanner();

    const members = collectMembersInteractive(allocator, original_termios) catch |err| {
        if (err == error.UserCancelled) {
            showCursor();
            std.debug.print("\n{s}Cancelled.{s}\n", .{ Color.dim, Color.reset });
            return;
        }
        return err;
    };

    // Own `members` until it's handed off to `config` below. This covers the
    // early returns in the UI prompt (cancel/error), which would otherwise
    // leak the allocated member names and the slice.
    var members_transferred = false;
    defer if (!members_transferred) {
        for (members) |m| allocator.free(m.name);
        allocator.free(members);
    };

    const include_ui = blk: {
        var any_agent = false;
        for (members) |m| {
            if (m.project_type == .agent) {
                any_agent = true;
                break;
            }
        }
        if (!any_agent) break :blk false;

        const want_ui = selectWantUI() catch |err| {
            if (err == error.UserCancelled) {
                showCursor();
                std.debug.print("\n{s}Cancelled.{s}\n", .{ Color.dim, Color.reset });
                return;
            }
            return err;
        };
        break :blk want_ui;
    };

    var config = ProjectConfig{
        .members = members,
        .include_ui = include_ui,
        .path = path,
        .relative_path = target_path,
    };
    members_transferred = true;
    defer config.deinit(allocator);

    try finishCreate(allocator, app_dir, &config, false);

    const available_editors = detectAvailableEditors(allocator);
    defer allocator.free(available_editors);

    const selected_editor = selectOpenProject(available_editors) catch |err| {
        if (err == error.UserCancelled) {
            showCursor();
            std.debug.print("\n{s}Cancelled.{s}\n", .{ Color.dim, Color.reset });
            return;
        }
        return err;
    };

    if (selected_editor) |editor| {
        try openEditor(allocator, editor, target_path);
    }
}

test "projectDirName extracts final path segment" {
    try std.testing.expectEqualStrings("myapp", projectDirName("myapp"));
    try std.testing.expectEqualStrings("myapp", projectDirName("./myapp"));
    try std.testing.expectEqualStrings("myapp", projectDirName("/tmp/foo/myapp"));
    try std.testing.expectEqualStrings("myapp", projectDirName("C:\\Users\\me\\myapp"));
}

test "resolveIncludeUi requires an agent when --with-ui is set" {
    const workflow_only = [_]WorkforceMemberSpec{
        .{ .name = "etl", .project_type = .workflow },
    };
    try std.testing.expectError(
        error.WithUiRequiresAgent,
        resolveIncludeUi(true, true, &workflow_only),
    );
    try std.testing.expect(!try resolveIncludeUi(false, true, &workflow_only));

    const with_agent = [_]WorkforceMemberSpec{
        .{ .name = "bot", .project_type = .agent },
    };
    try std.testing.expect(try resolveIncludeUi(true, true, &with_agent));
    try std.testing.expect(!try resolveIncludeUi(false, true, &with_agent));
    try std.testing.expect(!try resolveIncludeUi(true, false, &with_agent));
}

test "appendMember prepares names and rejects duplicates" {
    const a = std.testing.allocator;
    var list = std.ArrayList(WorkforceMemberSpec).init(a);
    defer {
        for (list.items) |m| a.free(m.name);
        list.deinit();
    }

    try appendMember(a, &list, .agent, "  My Agent  ");
    try std.testing.expectEqual(@as(usize, 1), list.items.len);
    try std.testing.expectEqualStrings("My Agent", list.items[0].name);

    try std.testing.expectError(error.DuplicateMemberName, appendMember(a, &list, .agent, "My Agent"));
    try std.testing.expectEqual(@as(usize, 1), list.items.len);

    // Case is preserved; "my agent" is a distinct member from "My Agent".
    try appendMember(a, &list, .workflow, "my agent");
    try std.testing.expectEqual(@as(usize, 2), list.items.len);

    try std.testing.expectError(error.InvalidWorkforceName, appendMember(a, &list, .agent, "bad/name"));
}

fn expectEntryExists(dir: fs.Dir, rel_path: []const u8) !void {
    try dir.access(rel_path, .{});
}

fn expectEntryMissing(dir: fs.Dir, rel_path: []const u8) !void {
    dir.access(rel_path, .{}) catch |err| {
        if (err == error.FileNotFound) return;
        return err;
    };
    return error.TestUnexpectedResult;
}

// Scaffolds workforce members and project metadata without network or git (fast, deterministic).
test "createProjectStructure scaffolds workforce and project files" {
    const a = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const path = try tmp.dir.realpathAlloc(a, ".");
    defer a.free(path);

    const foo_name = try a.dupe(u8, "foo");
    defer a.free(foo_name);
    const bar_name = try a.dupe(u8, "bar");
    defer a.free(bar_name);
    var members_list = std.ArrayList(WorkforceMemberSpec).init(a);
    defer members_list.deinit();
    try members_list.append(.{ .name = foo_name, .project_type = .agent });
    try members_list.append(.{ .name = bar_name, .project_type = .workflow });

    const config = ProjectConfig{
        .members = members_list.items,
        .include_ui = false,
        .path = path,
        .relative_path = ".",
    };

    try createProjectStructure(a, tmp.dir, config, .{
        .fetch_blueprints = false,
        .init_git = false,
    });

    try expectEntryExists(tmp.dir, "api");
    try expectEntryExists(tmp.dir, "workforce/foo/agent.py");
    try expectEntryExists(tmp.dir, "workforce/foo/timbal.yaml");
    try expectEntryExists(tmp.dir, "workforce/foo/pyproject.toml");
    try expectEntryExists(tmp.dir, "workforce/bar/workflow.py");
    try expectEntryExists(tmp.dir, "workforce/bar/timbal.yaml");
    try expectEntryExists(tmp.dir, "workforce/bar/pyproject.toml");
    try expectEntryExists(tmp.dir, ".gitignore");
    try expectEntryExists(tmp.dir, "README.md");
    try expectEntryMissing(tmp.dir, "ui");

    const agent_yaml = try tmp.dir.readFileAlloc(a, "workforce/foo/timbal.yaml", 64 * 1024);
    defer a.free(agent_yaml);
    var agent_cfg = utils.parseTimbalYaml(a, agent_yaml) orelse return error.TestUnexpectedResult;
    defer agent_cfg.deinit(a);
    try std.testing.expectEqualStrings("agent", agent_cfg.type);
    try std.testing.expectEqualStrings("agent.py::agent", agent_cfg.fqn);

    const workflow_yaml = try tmp.dir.readFileAlloc(a, "workforce/bar/timbal.yaml", 64 * 1024);
    defer a.free(workflow_yaml);
    var workflow_cfg = utils.parseTimbalYaml(a, workflow_yaml) orelse return error.TestUnexpectedResult;
    defer workflow_cfg.deinit(a);
    try std.testing.expectEqualStrings("workflow", workflow_cfg.type);
    try std.testing.expectEqualStrings("workflow.py::workflow", workflow_cfg.fqn);
}

// A failure partway through must leave nothing behind, so a retry on the same
// path isn't blocked by the non-empty directory guard.
test "createProjectStructure rolls back a partial scaffold on failure" {
    const a = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const path = try tmp.dir.realpathAlloc(a, ".");
    defer a.free(path);

    // Force a quiet I/O failure late in scaffolding: a pre-existing directory
    // named "README.md" makes the README createFile fail with error.IsDir,
    // after the member, api/, and workforce/ have already been written.
    try tmp.dir.makeDir("README.md");

    const good_name = try a.dupe(u8, "foo");
    defer a.free(good_name);
    var members_list = std.ArrayList(WorkforceMemberSpec).init(a);
    defer members_list.deinit();
    try members_list.append(.{ .name = good_name, .project_type = .agent });

    const config = ProjectConfig{
        .members = members_list.items,
        .include_ui = false,
        .path = path,
        .relative_path = ".",
    };

    try std.testing.expectError(error.IsDir, createProjectStructure(a, tmp.dir, config, .{
        .fetch_blueprints = false,
        .init_git = false,
    }));

    // Everything the scaffold created must be gone. The pre-existing
    // "README.md" directory (not created by us) must be left untouched.
    try expectEntryMissing(tmp.dir, "api");
    try expectEntryMissing(tmp.dir, "workforce");
    try expectEntryMissing(tmp.dir, ".gitignore");
    try expectEntryExists(tmp.dir, "README.md");
}
