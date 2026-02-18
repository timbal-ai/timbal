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
    try stderr.writeAll("Start a Timbal project (UI, API, and all agents and workflows).\n" ++
        "\n" ++
        "\x1b[1;32mUsage: \x1b[1;36mtimbal start \x1b[0;36m[PATH] [OPTIONS]\n" ++
        "\n" ++
        "\x1b[1;32mArguments:\n" ++
        "    \x1b[1;36m[PATH]         \x1b[0mPath to the project directory (default: current directory)\n" ++
        "\n" ++
        "\x1b[1;32mGlobal options:\n" ++
        "    \x1b[1;36m-h\x1b[0m, \x1b[1;36m--help       \x1b[0mDisplay the concise help for this command\n" ++
        "    \x1b[1;36m-V\x1b[0m, \x1b[1;36m--version    \x1b[0mDisplay the timbal version\n" ++
        "\n");
}

const WorkforceMember = struct {
    name: []const u8,
    config: utils.TimbalYaml,
    port: u16,
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
    prefix: []const u8,
    color: []const u8,
    mutex: *std.Thread.Mutex,
};

/// Thread function that reads from a pipe line by line, printing each with a colored prefix.
fn pipeReaderFn(ctx: PipeReaderCtx) void {
    const stdout = std.io.getStdOut().writer();
    var buf: [4096]u8 = undefined;
    const reader = ctx.pipe.reader();

    while (true) {
        const line = reader.readUntilDelimiter(&buf, '\n') catch |err| {
            if (err == error.EndOfStream) break;
            break;
        };
        if (line.len == 0) continue;

        ctx.mutex.lock();
        stdout.print("{s}{s}{s} {s}\n", .{ ctx.color, ctx.prefix, Color.reset, line }) catch {};
        ctx.mutex.unlock();
    }
}

/// Spawn a long-running process and start threads to stream its stdout/stderr with a prefix.
/// Returns the child process. Caller is responsible for waiting/killing it.
fn spawnService(
    allocator: std.mem.Allocator,
    argv: []const []const u8,
    cwd: []const u8,
    prefix: []const u8,
    color: []const u8,
    mutex: *std.Thread.Mutex,
    threads: *std.ArrayList(std.Thread),
    env_map: *const std.process.EnvMap,
) !std.process.Child {
    var child = std.process.Child.init(argv, allocator);
    child.cwd = cwd;
    child.stderr_behavior = .Pipe;
    child.stdout_behavior = .Pipe;
    child.env_map = env_map;

    try child.spawn();

    // Spawn threads to read stdout and stderr.
    if (child.stdout) |pipe| {
        const thread = try std.Thread.spawn(.{}, pipeReaderFn, .{PipeReaderCtx{
            .pipe = pipe,
            .prefix = prefix,
            .color = color,
            .mutex = mutex,
        }});
        try threads.append(thread);
    }
    if (child.stderr) |pipe| {
        const thread = try std.Thread.spawn(.{}, pipeReaderFn, .{PipeReaderCtx{
            .pipe = pipe,
            .prefix = prefix,
            .color = color,
            .mutex = mutex,
        }});
        try threads.append(thread);
    }

    return child;
}

/// Find the next available port starting from the given one.
fn findAvailablePort(start: u16) u16 {
    var port = start;
    while (port < 65535) : (port += 1) {
        const addr = std.net.Address.initIp4(.{ 127, 0, 0, 1 }, port);
        var server = addr.listen(.{}) catch continue;
        server.deinit();
        return port;
    }
    return start;
}

pub fn run(allocator: std.mem.Allocator, args: []const []const u8) !void {
    const stdout = std.io.getStdOut().writer();

    const base_port: u16 = 4455;
    var project_path: ?[]const u8 = null;

    // Parse arguments.
    for (args) |arg| {
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            try printUsage();
            return;
        } else if (std.mem.eql(u8, arg, "-V") or std.mem.eql(u8, arg, "--version")) {
            std.debug.print("Timbal {s}\n", .{timbal_version});
            return;
        } else if (!std.mem.startsWith(u8, arg, "-")) {
            if (project_path != null) {
                try printUsageWithError("Error: multiple paths provided");
                return;
            }
            project_path = arg;
        } else {
            try printUsageWithError("Error: unknown option");
            return;
        }
    }

    // Check that required tools are installed.
    const stderr = std.io.getStdErr().writer();

    var uv_check = std.process.Child.init(&.{ "uv", "--version" }, allocator);
    uv_check.stdout_behavior = .Ignore;
    uv_check.stderr_behavior = .Ignore;
    if (uv_check.spawn()) |_| {
        _ = uv_check.wait() catch {};
    } else |_| {
        try stderr.writeAll("Error: 'uv' is not installed.\n" ++
            "Install it from: https://docs.astral.sh/uv/getting-started/installation/\n");
        return;
    }

    var bun_check = std.process.Child.init(&.{ "bun", "--version" }, allocator);
    bun_check.stdout_behavior = .Ignore;
    bun_check.stderr_behavior = .Ignore;
    if (bun_check.spawn()) |_| {
        _ = bun_check.wait() catch {};
    } else |_| {
        try stderr.writeAll("Error: 'bun' is not installed.\n" ++
            "Install it from: https://bun.sh/docs/installation\n");
        return;
    }

    // Open the project directory (provided path or cwd).
    var project_dir = if (project_path) |path|
        fs.cwd().openDir(path, .{ .iterate = true }) catch {
            std.debug.print("Error: could not open directory '{s}'\n", .{path});
            return;
        }
    else
        fs.cwd();

    defer if (project_path != null) project_dir.close();

    // Resolve the absolute project path for child process cwd.
    const abs_project_path = try project_dir.realpathAlloc(allocator, ".");
    defer allocator.free(abs_project_path);

    // Detect UI directory and assign port.
    const has_ui = if (project_dir.statFile("ui/package.json")) |_| true else |_| false;
    const ui_port: ?u16 = if (has_ui) findAvailablePort(3737) else null;

    // Detect API directory and assign port.
    const has_api = if (project_dir.statFile("api/package.json")) |_| true else |_| false;
    const api_port: ?u16 = if (has_api) findAvailablePort(3000) else null;

    // Discover workforce members (subdirectories of workforce/ that contain timbal.yaml).
    var members = std.ArrayList(WorkforceMember).init(allocator);
    defer members.deinit();

    var port = base_port;

    if (project_dir.openDir("workforce", .{ .iterate = true })) |*workforce_dir_ptr| {
        var workforce_dir = workforce_dir_ptr.*;
        defer workforce_dir.close();

        var iter = workforce_dir.iterate();
        while (try iter.next()) |entry| {
            if (entry.kind != .directory) continue;

            // Check if this subdirectory contains a timbal.yaml.
            const yaml_path = try std.fmt.allocPrint(allocator, "workforce/{s}/timbal.yaml", .{entry.name});
            defer allocator.free(yaml_path);

            if (project_dir.statFile(yaml_path)) |_| {
                const content = project_dir.readFileAlloc(allocator, yaml_path, 64 * 1024) catch continue;
                defer allocator.free(content);
                const config = utils.parseTimbalYaml(allocator, content) orelse {
                    std.debug.print("Warning: invalid timbal.yaml in {s}, skipping\n", .{yaml_path});
                    continue;
                };
                const name = try allocator.dupe(u8, entry.name);
                const available_port = findAvailablePort(port);
                try members.append(.{ .name = name, .config = config, .port = available_port });
                port = available_port + 1;
            } else |_| {}
        }
    } else |_| {}

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

    // Start services.
    try stdout.print("\n{s}Starting services...{s}\n\n", .{ Color.bold, Color.reset });

    var output_mutex = std.Thread.Mutex{};
    var reader_threads = std.ArrayList(std.Thread).init(allocator);
    defer reader_threads.deinit();

    var children = std.ArrayList(std.process.Child).init(allocator);
    defer children.deinit();

    // Shared env map with FORCE_COLOR for all services.
    var service_env = std.process.getEnvMap(allocator) catch return;
    defer service_env.deinit();
    service_env.put("FORCE_COLOR", "1") catch return;
    service_env.put("TIMBAL_LOG_EVENTS", "START,OUTPUT") catch return;
    service_env.put("TIMBAL_LOG_FORMAT", "dev") catch return;

    var color_idx: usize = 0;

    // Start workforce members first.
    for (members.items) |member| {
        const member_dir = try std.fmt.allocPrint(allocator, "{s}/workforce/{s}", .{ abs_project_path, member.name });
        defer allocator.free(member_dir);
        const port_str = try std.fmt.allocPrint(allocator, "{d}", .{member.port});
        defer allocator.free(port_str);
        const color = prefix_colors[color_idx % prefix_colors.len];
        color_idx += 1;

        var child = spawnService(allocator, &.{ "uv", "run", "-m", "timbal.server.http", "--port", port_str, "--import_spec", member.config.fqn }, member_dir, member.name, color, &output_mutex, &reader_threads, &service_env) catch {
            std.debug.print("Error: failed to start {s}\n", .{member.name});
            continue;
        };
        try children.append(child);
        _ = &child;
    }

    // Build TIMBAL_WORKFORCE env var: name:port,name:port,...
    {
        var workforce_buf = std.ArrayList(u8).init(allocator);
        defer workforce_buf.deinit();
        for (members.items, 0..) |member, idx| {
            if (idx > 0) workforce_buf.append(',') catch {};
            const entry = std.fmt.allocPrint(allocator, "{s}:{d}", .{ member.config.id, member.port }) catch continue;
            defer allocator.free(entry);
            workforce_buf.appendSlice(entry) catch {};
        }
        if (workforce_buf.items.len > 0) {
            service_env.put("TIMBAL_WORKFORCE", workforce_buf.items) catch {};
        }
    }

    if (has_ui) {
        const ui_dir = try std.fmt.allocPrint(allocator, "{s}/ui", .{abs_project_path});
        defer allocator.free(ui_dir);
        const ui_port_str = try std.fmt.allocPrint(allocator, "{d}", .{ui_port.?});
        defer allocator.free(ui_port_str);

        service_env.put("PORT", ui_port_str) catch return;
        var child = spawnService(allocator, &.{ "bun", "run", "dev", "--port", ui_port_str }, ui_dir, "ui", "\x1b[1;36m", &output_mutex, &reader_threads, &service_env) catch {
            std.debug.print("Error: failed to start UI\n", .{});
            return;
        };
        try children.append(child);
        _ = &child;
    }

    if (has_api) {
        const api_dir = try std.fmt.allocPrint(allocator, "{s}/api", .{abs_project_path});
        defer allocator.free(api_dir);
        const api_port_str = try std.fmt.allocPrint(allocator, "{d}", .{api_port.?});
        defer allocator.free(api_port_str);

        service_env.put("PORT", api_port_str) catch return;
        var child = spawnService(allocator, &.{ "bun", "run", "dev" }, api_dir, "api", "\x1b[1;32m", &output_mutex, &reader_threads, &service_env) catch {
            std.debug.print("Error: failed to start API\n", .{});
            return;
        };
        try children.append(child);
        _ = &child;
    }

    // Wait for all reader threads (blocks until all services exit).
    for (reader_threads.items) |thread| {
        thread.join();
    }

    // Wait for all child processes to finish.
    for (children.items) |*child| {
        _ = child.wait() catch {};
    }

    // Free member data.
    for (members.items) |*member| {
        allocator.free(member.name);
        member.config.deinit(allocator);
    }
}
