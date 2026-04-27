const std = @import("std");
const fs = std.fs;

const utils = @import("../utils.zig");
const Color = utils.Color;

const builtin = @import("builtin");

fn enableWindowsConsole() u32 {
    if (builtin.os.tag == .windows) {
        const windows = std.os.windows;

        // Set output codepage to UTF-8.
        const orig_cp = windows.kernel32.GetConsoleOutputCP();
        _ = windows.kernel32.SetConsoleOutputCP(65001);

        // Enable VT processing on stdout so ANSI escape codes render correctly.
        const stdout_handle = std.io.getStdOut().handle;
        var mode: windows.DWORD = 0;
        if (windows.kernel32.GetConsoleMode(stdout_handle, &mode) != 0) {
            _ = windows.kernel32.SetConsoleMode(stdout_handle, mode | windows.ENABLE_VIRTUAL_TERMINAL_PROCESSING);
        }

        return orig_cp;
    }
    return 0;
}

fn restoreWindowsConsole(orig_cp: u32) void {
    if (builtin.os.tag == .windows) {
        _ = std.os.windows.kernel32.SetConsoleOutputCP(orig_cp);
    }
}

const is_windows = builtin.os.tag == .windows;
const sep = if (is_windows) "\\" else "/";

// Terminal state saved before spawning services so we can restore it on exit.
// Child processes like `bun run dev` put the terminal in raw mode and may not
// restore it if killed by Ctrl+C — we handle that here instead.
var g_stdin_fd: i32 = -1;
var g_termios_saved: bool = false;
var g_original_termios: if (!is_windows) std.posix.termios else [0]u8 = undefined;

fn restoreTermios() void {
    if (comptime !is_windows) {
        if (g_termios_saved) {
            g_termios_saved = false;
            std.posix.tcsetattr(g_stdin_fd, .FLUSH, g_original_termios) catch {};
        }
    }
}

var g_interrupted = std.atomic.Value(bool).init(false);

fn sigintHandler(_: c_int) callconv(.C) void {
    if (g_interrupted.swap(true, .seq_cst)) {
        // Second Ctrl+C: force exit after restoring terminal.
        restoreTermios();
        std.process.exit(130);
    }
    // First Ctrl+C: let the supervisor loop stop child process groups and
    // restore the terminal through the normal cleanup path.
}

fn getHomePath(allocator: std.mem.Allocator) ![]u8 {
    return if (is_windows)
        std.process.getEnvVarOwned(allocator, "USERPROFILE")
    else
        std.process.getEnvVarOwned(allocator, "HOME");
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

/// Strip protocol prefix (e.g. "https://api.timbal.ai" -> "api.timbal.ai").
fn stripProtocol(url: []const u8) []const u8 {
    if (std.mem.indexOf(u8, url, "://")) |idx| {
        return url[idx + 3 ..];
    }
    return url;
}

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
        "    \x1b[1;36m[PATH] \x1b[0mPath to the project directory (default: current directory)\n" ++
        "\n" ++
        "\x1b[1;32mInteractive commands:\n" ++
        "    \x1b[1;36mr\x1b[0m, \x1b[1;36mrestart  \x1b[0mRestart all services\n" ++
        "    \x1b[1;36ms\x1b[0m, \x1b[1;36mstatus   \x1b[0mShow service status, ports, PIDs, and uptime\n" ++
        "    \x1b[1;36mo\x1b[0m, \x1b[1;36mopen     \x1b[0mOpen the app in your browser\n" ++
        "    \x1b[1;36mu\x1b[0m, \x1b[1;36mui       \x1b[0mPrint the UI URL\n" ++
        "    \x1b[1;36ma\x1b[0m, \x1b[1;36mapi      \x1b[0mPrint the API URL\n" ++
        "    \x1b[1;36mw\x1b[0m, \x1b[1;36mworkforce\x1b[0m Print workforce service URLs\n" ++
        "    \x1b[1;36mf <target>\x1b[0m        Focus logs: all, ui, api, workforce, or member name\n" ++
        "    \x1b[1;36mm <target>\x1b[0m        Toggle mute for logs: all, ui, api, workforce, or member name\n" ++
        "    \x1b[1;36mh\x1b[0m, \x1b[1;36mhelp     \x1b[0mShow commands while services are running\n" ++
        "    \x1b[1;36mq\x1b[0m, \x1b[1;36mquit     \x1b[0mStop services and quit\n" ++
        "\n" ++
        utils.global_options_help ++
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
    service_name: []const u8,
    service_kind: ServiceKind,
    prefix: []const u8,
    color: []const u8,
    mutex: *std.Thread.Mutex,
    allocator: std.mem.Allocator,
    log_filter: *LogFilterState,
};

const ServiceKind = enum {
    ui,
    api,
    workforce,
};

const LogFocus = enum {
    all,
    name,
    workforce,
};

const LogFilterState = struct {
    mutex: std.Thread.Mutex = .{},
    focus: LogFocus = .all,
    focus_name: [128]u8 = undefined,
    focus_name_len: usize = 0,
    mute_all: bool = false,
    mute_ui: bool = false,
    mute_api: bool = false,
    mute_workforce: bool = false,
    muted_names: std.StringHashMap(void),

    fn init(allocator: std.mem.Allocator) LogFilterState {
        return .{
            .muted_names = std.StringHashMap(void).init(allocator),
        };
    }

    fn deinit(self: *LogFilterState) void {
        var iter = self.muted_names.keyIterator();
        while (iter.next()) |key| {
            self.muted_names.allocator.free(key.*);
        }
        self.muted_names.deinit();
    }

    fn shouldPrint(self: *LogFilterState, service_name: []const u8, service_kind: ServiceKind) bool {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.focus == .workforce and service_kind != .workforce) return false;
        if (self.focus == .name and !std.mem.eql(u8, service_name, self.focus_name[0..self.focus_name_len])) return false;

        if (self.mute_all) return false;
        if (service_kind == .ui and self.mute_ui) return false;
        if (service_kind == .api and self.mute_api) return false;
        if (service_kind == .workforce and self.mute_workforce) return false;
        if (self.muted_names.contains(service_name)) return false;

        return true;
    }

    fn setFocus(self: *LogFilterState, target: []const u8) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (std.mem.eql(u8, target, "all")) {
            self.focus = .all;
            self.focus_name_len = 0;
        } else if (std.mem.eql(u8, target, "workforce")) {
            self.focus = .workforce;
            self.focus_name_len = 0;
        } else {
            self.focus = .name;
            self.focus_name_len = @min(target.len, self.focus_name.len);
            @memcpy(self.focus_name[0..self.focus_name_len], target[0..self.focus_name_len]);
        }
    }

    fn toggleMuteName(self: *LogFilterState, target: []const u8) !bool {
        if (self.muted_names.contains(target)) {
            if (self.muted_names.fetchRemove(target)) |entry| {
                self.muted_names.allocator.free(entry.key);
            }
            return false;
        }

        const owned_target = try self.muted_names.allocator.dupe(u8, target);
        errdefer self.muted_names.allocator.free(owned_target);
        try self.muted_names.put(owned_target, {});
        return true;
    }

    fn toggleMute(self: *LogFilterState, target: []const u8, target_is_member_name: bool) !bool {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (std.mem.eql(u8, target, "all")) {
            self.mute_all = !self.mute_all;
            return self.mute_all;
        }
        if (std.mem.eql(u8, target, "workforce")) {
            self.mute_workforce = !self.mute_workforce;
            return self.mute_workforce;
        }
        if (target_is_member_name) {
            return self.toggleMuteName(target);
        }
        if (std.mem.eql(u8, target, "ui")) {
            self.mute_ui = !self.mute_ui;
            return self.mute_ui;
        }
        if (std.mem.eql(u8, target, "api")) {
            self.mute_api = !self.mute_api;
            return self.mute_api;
        }
        return self.toggleMuteName(target);
    }
};

/// Thread function that reads from a pipe line by line, printing each with a colored prefix.
///
/// Uses a heap-allocated dynamic buffer so a single oversized log line (e.g. a structlog
/// event whose context dumps a Message or a tool result) does not kill the reader thread
/// and silently swallow all subsequent child output.
fn pipeReaderFn(ctx: PipeReaderCtx) void {
    const stdout = std.io.getStdOut().writer();
    const reader = ctx.pipe.reader();

    // 16 MiB cap per line — well above anything a sane log line should produce, but
    // bounded so a runaway producer can't OOM us. Lines longer than this are split.
    const max_line_size: usize = 16 * 1024 * 1024;

    while (true) {
        const line = reader.readUntilDelimiterAlloc(ctx.allocator, '\n', max_line_size) catch |err| switch (err) {
            error.EndOfStream => break,
            error.StreamTooLong => {
                // Single line longer than max_line_size; emit a marker and resync at next '\n'.
                ctx.mutex.lock();
                stdout.print("{s}{s}{s} <line too long, truncated>\n", .{ ctx.color, ctx.prefix, Color.reset }) catch {};
                ctx.mutex.unlock();
                reader.skipUntilDelimiterOrEof('\n') catch break;
                continue;
            },
            else => continue,
        };
        defer ctx.allocator.free(line);
        if (line.len == 0) continue;
        if (!ctx.log_filter.shouldPrint(ctx.service_name, ctx.service_kind)) continue;

        ctx.mutex.lock();
        stdout.print("{s}{s}{s} {s}\n", .{ ctx.color, ctx.prefix, Color.reset, line }) catch {};
        ctx.mutex.unlock();
    }
}

const CommandAction = enum(u8) {
    none,
    help,
    status,
    open,
    urls,
    ui_url,
    api_url,
    workforce_urls,
    focus_logs,
    toggle_mute,
    restart,
    quit,
};

const Command = struct {
    action: CommandAction = .none,
    target: [128]u8 = undefined,
    target_len: usize = 0,
};

const CommandState = struct {
    mutex: std.Thread.Mutex = .{},
    command: Command = .{},

    fn set(self: *CommandState, action: CommandAction, target: []const u8) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        self.command.action = action;
        self.command.target_len = @min(target.len, self.command.target.len);
        if (self.command.target_len > 0) {
            @memcpy(self.command.target[0..self.command.target_len], target[0..self.command.target_len]);
        }
    }

    fn take(self: *CommandState) Command {
        self.mutex.lock();
        defer self.mutex.unlock();

        const command = self.command;
        self.command = .{};
        return command;
    }
};

const CommandInputCtx = struct {
    state: *CommandState,
    stop: *std.atomic.Value(bool),
};

fn targetAfterCommand(command: []const u8, verb: []const u8) ?[]const u8 {
    if (!std.mem.startsWith(u8, command, verb)) return null;
    if (command.len == verb.len) return "";
    if (command[verb.len] != ' ') return null;
    return std.mem.trim(u8, command[verb.len + 1 ..], " \t\r\n");
}

fn targetAfterEitherCommand(command: []const u8, short_verb: []const u8, long_verb: []const u8) ?[]const u8 {
    if (targetAfterCommand(command, short_verb)) |target| return target;
    return targetAfterCommand(command, long_verb);
}

fn commandInputFn(ctx: CommandInputCtx) void {
    const stdin_file = std.io.getStdIn();
    const stdin = stdin_file.reader();

    while (!ctx.stop.load(.seq_cst)) {
        if (comptime !is_windows) {
            var poll_fds = [_]std.posix.pollfd{.{
                .fd = stdin_file.handle,
                .events = std.posix.POLL.IN,
                .revents = 0,
            }};
            const ready = std.posix.poll(&poll_fds, 150) catch continue;
            if (ready == 0) continue;
            if ((poll_fds[0].revents & (std.posix.POLL.HUP | std.posix.POLL.ERR)) != 0) break;
            if ((poll_fds[0].revents & std.posix.POLL.IN) == 0) continue;
        }

        const line = stdin.readUntilDelimiterAlloc(std.heap.page_allocator, '\n', 1024) catch |err| switch (err) {
            error.EndOfStream => break,
            error.StreamTooLong => {
                stdin.skipUntilDelimiterOrEof('\n') catch break;
                continue;
            },
            else => continue,
        };
        defer std.heap.page_allocator.free(line);
        if (ctx.stop.load(.seq_cst)) break;

        const command = std.mem.trim(u8, line, " \t\r\n");
        if (std.mem.eql(u8, command, "r") or std.mem.eql(u8, command, "restart")) {
            ctx.state.set(.restart, "");
        } else if (std.mem.eql(u8, command, "h") or std.mem.eql(u8, command, "help") or std.mem.eql(u8, command, "?")) {
            ctx.state.set(.help, "");
        } else if (std.mem.eql(u8, command, "s") or std.mem.eql(u8, command, "status")) {
            ctx.state.set(.status, "");
        } else if (std.mem.eql(u8, command, "o") or std.mem.eql(u8, command, "open")) {
            ctx.state.set(.open, "");
        } else if (std.mem.eql(u8, command, "url") or std.mem.eql(u8, command, "urls")) {
            ctx.state.set(.urls, "");
        } else if (std.mem.eql(u8, command, "u") or std.mem.eql(u8, command, "ui")) {
            ctx.state.set(.ui_url, "");
        } else if (std.mem.eql(u8, command, "a") or std.mem.eql(u8, command, "api")) {
            ctx.state.set(.api_url, "");
        } else if (std.mem.eql(u8, command, "w") or std.mem.eql(u8, command, "workforce")) {
            ctx.state.set(.workforce_urls, "");
        } else if (targetAfterEitherCommand(command, "f", "focus")) |target| {
            ctx.state.set(.focus_logs, target);
        } else if (targetAfterEitherCommand(command, "m", "mute")) |target| {
            ctx.state.set(.toggle_mute, target);
        } else if (std.mem.eql(u8, command, "q") or std.mem.eql(u8, command, "quit")) {
            ctx.state.set(.quit, "");
            break;
        }
    }
}

fn forceStopSpawnedChild(child: *std.process.Child) void {
    if (comptime !is_windows) {
        std.posix.kill(-child.id, std.posix.SIG.KILL) catch |err| switch (err) {
            error.ProcessNotFound => {},
            error.PermissionDenied => {},
            else => {},
        };
    } else {
        _ = child.kill() catch {};
    }
    _ = child.wait() catch {};
}

/// Spawn a long-running process and start threads to stream its stdout/stderr with a prefix.
/// Returns the child process. Caller is responsible for waiting/killing it.
fn spawnService(
    allocator: std.mem.Allocator,
    argv: []const []const u8,
    cwd: []const u8,
    prefix: []const u8,
    service_kind: ServiceKind,
    color: []const u8,
    mutex: *std.Thread.Mutex,
    threads: *std.ArrayList(std.Thread),
    env_map: *const std.process.EnvMap,
    log_filter: *LogFilterState,
) !std.process.Child {
    var child = std.process.Child.init(argv, allocator);
    child.cwd = cwd;
    child.stdin_behavior = .Ignore;
    child.stderr_behavior = .Pipe;
    child.stdout_behavior = .Pipe;
    child.env_map = env_map;
    if (comptime !is_windows) {
        // Put every service in its own process group so restarts also stop grandchildren
        // spawned by wrappers like `bun run` or `uv run`.
        child.pgid = 0;
    }

    try child.spawn();
    errdefer forceStopSpawnedChild(&child);

    // Spawn threads to read stdout and stderr.
    if (child.stdout) |pipe| {
        const thread = try std.Thread.spawn(.{}, pipeReaderFn, .{PipeReaderCtx{
            .pipe = pipe,
            .service_name = prefix,
            .service_kind = service_kind,
            .prefix = prefix,
            .color = color,
            .mutex = mutex,
            .allocator = allocator,
            .log_filter = log_filter,
        }});
        threads.appendAssumeCapacity(thread);
    }
    if (child.stderr) |pipe| {
        const thread = try std.Thread.spawn(.{}, pipeReaderFn, .{PipeReaderCtx{
            .pipe = pipe,
            .service_name = prefix,
            .service_kind = service_kind,
            .prefix = prefix,
            .color = color,
            .mutex = mutex,
            .allocator = allocator,
            .log_filter = log_filter,
        }});
        threads.appendAssumeCapacity(thread);
    }

    return child;
}

const ServiceStatus = struct {
    name: []const u8,
    port: ?u16,
    pid: if (is_windows) void else std.posix.pid_t,
    started_at_ms: i64,
};

const RunningServices = struct {
    children: std.ArrayList(std.process.Child),
    reader_threads: std.ArrayList(std.Thread),
    statuses: std.ArrayList(ServiceStatus),

    fn deinit(self: *RunningServices) void {
        self.children.deinit();
        self.reader_threads.deinit();
        self.statuses.deinit();
    }
};

fn signalServiceGroup(child: *std.process.Child, sig: u8) void {
    if (comptime !is_windows) {
        if (child.term != null) return;
        std.posix.kill(-child.id, sig) catch |err| switch (err) {
            error.ProcessNotFound => {},
            error.PermissionDenied => {},
            else => {},
        };
    }
}

fn stopServices(services: *RunningServices) void {
    if (services.children.items.len > 0) {
        if (comptime !is_windows) {
            for (services.children.items) |*child| signalServiceGroup(child, std.posix.SIG.INT);
            std.time.sleep(1200 * std.time.ns_per_ms);

            for (services.children.items) |*child| signalServiceGroup(child, std.posix.SIG.TERM);
            std.time.sleep(800 * std.time.ns_per_ms);

            for (services.children.items) |*child| signalServiceGroup(child, std.posix.SIG.KILL);
        } else {
            for (services.children.items) |*child| {
                _ = child.kill() catch {};
            }
        }

        for (services.children.items) |*child| {
            _ = child.wait() catch {};
        }
    }

    for (services.reader_threads.items) |thread| {
        thread.join();
    }

    services.deinit();
}

fn startServices(
    allocator: std.mem.Allocator,
    abs_project_path: []const u8,
    members: []const WorkforceMember,
    has_ui: bool,
    ui_port: ?u16,
    has_api: bool,
    api_port: ?u16,
    service_env: *std.process.EnvMap,
    output_mutex: *std.Thread.Mutex,
    log_filter: *LogFilterState,
) !RunningServices {
    const stdout = std.io.getStdOut().writer();
    try stdout.print("\n{s}Starting services...{s}\n\n", .{ Color.bold, Color.reset });

    var services = RunningServices{
        .children = std.ArrayList(std.process.Child).init(allocator),
        .reader_threads = std.ArrayList(std.Thread).init(allocator),
        .statuses = std.ArrayList(ServiceStatus).init(allocator),
    };
    errdefer stopServices(&services);

    const service_count = members.len + @as(usize, @intFromBool(has_ui)) + @as(usize, @intFromBool(has_api));
    try services.children.ensureTotalCapacity(service_count);
    try services.statuses.ensureTotalCapacity(service_count);
    try services.reader_threads.ensureTotalCapacity(service_count * 2);

    var color_idx: usize = 0;

    // Start workforce members first.
    for (members) |member| {
        const member_dir = try std.fmt.allocPrint(allocator, "{s}/workforce/{s}", .{ abs_project_path, member.name });
        defer allocator.free(member_dir);
        const port_str = try std.fmt.allocPrint(allocator, "{d}", .{member.port});
        defer allocator.free(port_str);
        const color = prefix_colors[color_idx % prefix_colors.len];
        color_idx += 1;

        const child = spawnService(allocator, &.{ "uv", "run", "-m", "timbal.server.http", "--port", port_str, "--import_spec", member.config.fqn }, member_dir, member.name, .workforce, color, output_mutex, &services.reader_threads, service_env, log_filter) catch {
            std.debug.print("Error: failed to start {s}\n", .{member.name});
            continue;
        };
        const status = ServiceStatus{
            .name = member.name,
            .port = member.port,
            .pid = if (is_windows) {} else child.id,
            .started_at_ms = std.time.milliTimestamp(),
        };
        services.children.appendAssumeCapacity(child);
        services.statuses.appendAssumeCapacity(status);
    }

    if (has_ui) {
        const ui_dir = try std.fmt.allocPrint(allocator, "{s}/ui", .{abs_project_path});
        defer allocator.free(ui_dir);
        const ui_port_str = try std.fmt.allocPrint(allocator, "{d}", .{ui_port.?});
        defer allocator.free(ui_port_str);

        service_env.put("PORT", ui_port_str) catch return error.OutOfMemory;
        const child = spawnService(allocator, &.{ "bun", "run", "dev", "--port", ui_port_str }, ui_dir, "ui", .ui, "\x1b[1;36m", output_mutex, &services.reader_threads, service_env, log_filter) catch {
            std.debug.print("Error: failed to start UI\n", .{});
            return error.ServiceStartFailed;
        };
        const status = ServiceStatus{
            .name = "ui",
            .port = ui_port.?,
            .pid = if (is_windows) {} else child.id,
            .started_at_ms = std.time.milliTimestamp(),
        };
        services.children.appendAssumeCapacity(child);
        services.statuses.appendAssumeCapacity(status);
    }

    if (has_api) {
        const api_dir = try std.fmt.allocPrint(allocator, "{s}/api", .{abs_project_path});
        defer allocator.free(api_dir);
        const api_port_str = try std.fmt.allocPrint(allocator, "{d}", .{api_port.?});
        defer allocator.free(api_port_str);

        service_env.put("PORT", api_port_str) catch return error.OutOfMemory;
        const child = spawnService(allocator, &.{ "bun", "run", "dev" }, api_dir, "api", .api, "\x1b[1;32m", output_mutex, &services.reader_threads, service_env, log_filter) catch {
            std.debug.print("Error: failed to start API\n", .{});
            return error.ServiceStartFailed;
        };
        const status = ServiceStatus{
            .name = "api",
            .port = api_port.?,
            .pid = if (is_windows) {} else child.id,
            .started_at_ms = std.time.milliTimestamp(),
        };
        services.children.appendAssumeCapacity(child);
        services.statuses.appendAssumeCapacity(status);
    }

    return services;
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

fn openBrowserUrl(allocator: std.mem.Allocator, url: []const u8) void {
    var argv = std.ArrayList([]const u8).init(allocator);
    defer argv.deinit();

    if (builtin.os.tag == .macos) {
        argv.append("open") catch return;
        argv.append(url) catch return;
    } else if (builtin.os.tag == .windows) {
        argv.append("cmd") catch return;
        argv.append("/c") catch return;
        argv.append("start") catch return;
        argv.append("") catch return;
        argv.append(url) catch return;
    } else {
        argv.append("xdg-open") catch return;
        argv.append(url) catch return;
    }

    var child = std.process.Child.init(argv.items, allocator);
    child.stdin_behavior = .Ignore;
    child.stdout_behavior = .Ignore;
    child.stderr_behavior = .Ignore;
    child.spawn() catch return;
    _ = child.wait() catch {};
}

const BrowserContext = struct {
    allocator: std.mem.Allocator,
    url: []u8,
};

fn openBrowserThread(ctx: BrowserContext) void {
    defer ctx.allocator.free(ctx.url);
    std.time.sleep(2 * std.time.ns_per_s);
    openBrowserUrl(ctx.allocator, ctx.url);
}

fn preferredBrowserUrl(allocator: std.mem.Allocator, has_ui: bool, ui_port: ?u16, has_api: bool, api_port: ?u16) ?[]u8 {
    return if (has_ui)
        std.fmt.allocPrint(allocator, "http://localhost:{d}", .{ui_port.?}) catch null
    else if (has_api)
        std.fmt.allocPrint(allocator, "http://localhost:{d}", .{api_port.?}) catch null
    else
        null;
}

fn printCommandHelp(mutex: *std.Thread.Mutex) void {
    const stdout = std.io.getStdOut().writer();
    mutex.lock();
    defer mutex.unlock();

    stdout.writeAll("\n") catch {};
    stdout.print("{s}Available commands:{s}\n", .{ Color.bold, Color.reset }) catch {};
    stdout.print("  {s}r{s}  restart all services\n", .{ Color.bold_cyan, Color.reset }) catch {};
    stdout.print("  {s}s{s}  show service status\n", .{ Color.bold_cyan, Color.reset }) catch {};
    stdout.print("  {s}o{s}  open the app in your browser\n", .{ Color.bold_cyan, Color.reset }) catch {};
    stdout.print("  {s}u{s}  print the UI URL\n", .{ Color.bold_cyan, Color.reset }) catch {};
    stdout.print("  {s}a{s}  print the API URL\n", .{ Color.bold_cyan, Color.reset }) catch {};
    stdout.print("  {s}w{s}  print workforce service URLs\n", .{ Color.bold_cyan, Color.reset }) catch {};
    stdout.print("  {s}f <target>{s}  focus logs: all, ui, api, workforce, or member name\n", .{ Color.bold_cyan, Color.reset }) catch {};
    stdout.print("  {s}m <target>{s}  toggle mute for logs: all, ui, api, workforce, or member name\n", .{ Color.bold_cyan, Color.reset }) catch {};
    stdout.print("  {s}q{s}  stop services and quit\n", .{ Color.bold_cyan, Color.reset }) catch {};
}

fn isWorkforceMemberName(target: []const u8, members: []const WorkforceMember) bool {
    for (members) |member| {
        if (std.mem.eql(u8, target, member.name)) return true;
    }
    return false;
}

fn isKnownLogTarget(target: []const u8, members: []const WorkforceMember, has_ui: bool, has_api: bool) bool {
    if (std.mem.eql(u8, target, "all")) return true;
    if (std.mem.eql(u8, target, "workforce")) return members.len > 0;
    if (isWorkforceMemberName(target, members)) return true;
    if (std.mem.eql(u8, target, "ui")) return has_ui;
    if (std.mem.eql(u8, target, "api")) return has_api;
    return false;
}

fn printUnknownLogTarget(target: []const u8, mutex: *std.Thread.Mutex) void {
    const stdout = std.io.getStdOut().writer();
    mutex.lock();
    defer mutex.unlock();

    stdout.print("\n{s}Unknown log target `{s}`. Use all, ui, api, workforce, or a workforce member name.{s}\n", .{ Color.dim, target, Color.reset }) catch {};
}

fn handleFocusCommand(
    target: []const u8,
    members: []const WorkforceMember,
    has_ui: bool,
    has_api: bool,
    log_filter: *LogFilterState,
    mutex: *std.Thread.Mutex,
) void {
    const stdout = std.io.getStdOut().writer();
    if (target.len == 0) {
        printUnknownLogTarget(target, mutex);
        return;
    }
    if (!isKnownLogTarget(target, members, has_ui, has_api)) {
        printUnknownLogTarget(target, mutex);
        return;
    }

    log_filter.setFocus(target);

    mutex.lock();
    defer mutex.unlock();
    if (std.mem.eql(u8, target, "all")) {
        stdout.print("\n{s}Showing all non-muted logs.{s}\n", .{ Color.dim, Color.reset }) catch {};
    } else {
        stdout.print("\n{s}Showing only {s} logs. Type `f all` to restore all logs.{s}\n", .{ Color.dim, target, Color.reset }) catch {};
    }
}

fn handleMuteCommand(
    target: []const u8,
    members: []const WorkforceMember,
    has_ui: bool,
    has_api: bool,
    log_filter: *LogFilterState,
    mutex: *std.Thread.Mutex,
) void {
    const stdout = std.io.getStdOut().writer();
    if (target.len == 0) {
        printUnknownLogTarget(target, mutex);
        return;
    }
    if (!isKnownLogTarget(target, members, has_ui, has_api)) {
        printUnknownLogTarget(target, mutex);
        return;
    }

    const muted = log_filter.toggleMute(target, isWorkforceMemberName(target, members)) catch {
        mutex.lock();
        stdout.print("\n{s}Could not update log mute state for `{s}`.{s}\n", .{ Color.dim, target, Color.reset }) catch {};
        mutex.unlock();
        return;
    };

    mutex.lock();
    defer mutex.unlock();
    if (muted) {
        stdout.print("\n{s}Muted {s} logs. Type `m {s}` again to unmute.{s}\n", .{ Color.dim, target, target, Color.reset }) catch {};
    } else {
        stdout.print("\n{s}Unmuted {s} logs.{s}\n", .{ Color.dim, target, Color.reset }) catch {};
    }
}

fn printServiceStatus(services: ?*RunningServices, mutex: *std.Thread.Mutex) void {
    const stdout = std.io.getStdOut().writer();
    mutex.lock();
    defer mutex.unlock();

    stdout.writeAll("\n") catch {};
    stdout.print("{s}Service status:{s}\n", .{ Color.bold, Color.reset }) catch {};

    const running = services orelse {
        stdout.print("  {s}No services running{s}\n", .{ Color.dim, Color.reset }) catch {};
        return;
    };

    const now_ms = std.time.milliTimestamp();
    for (running.statuses.items) |status| {
        const uptime_s: i64 = @divTrunc(now_ms - status.started_at_ms, 1000);
        if (comptime is_windows) {
            if (status.port) |p| {
                stdout.print("  {s}✓{s} {s}  port={d} uptime={d}s\n", .{ Color.bold_green, Color.reset, status.name, p, uptime_s }) catch {};
            } else {
                stdout.print("  {s}✓{s} {s}  uptime={d}s\n", .{ Color.bold_green, Color.reset, status.name, uptime_s }) catch {};
            }
        } else {
            if (status.port) |p| {
                stdout.print("  {s}✓{s} {s}  pid={d} port={d} uptime={d}s\n", .{ Color.bold_green, Color.reset, status.name, status.pid, p, uptime_s }) catch {};
            } else {
                stdout.print("  {s}✓{s} {s}  pid={d} uptime={d}s\n", .{ Color.bold_green, Color.reset, status.name, status.pid, uptime_s }) catch {};
            }
        }
    }
}

fn printProjectUrls(
    members: []const WorkforceMember,
    has_ui: bool,
    ui_port: ?u16,
    has_api: bool,
    api_port: ?u16,
    action: CommandAction,
    mutex: *std.Thread.Mutex,
) void {
    const stdout = std.io.getStdOut().writer();
    mutex.lock();
    defer mutex.unlock();

    stdout.writeAll("\n") catch {};

    if ((action == .urls or action == .ui_url) and has_ui) {
        stdout.print("  UI   {s}http://localhost:{d}{s}\n", .{ Color.bold_cyan, ui_port.?, Color.reset }) catch {};
    } else if (action == .ui_url) {
        stdout.print("  {s}UI not found{s}\n", .{ Color.dim, Color.reset }) catch {};
    }

    if ((action == .urls or action == .api_url) and has_api) {
        stdout.print("  API  {s}http://localhost:{d}{s}\n", .{ Color.bold_cyan, api_port.?, Color.reset }) catch {};
    } else if (action == .api_url) {
        stdout.print("  {s}API not found{s}\n", .{ Color.dim, Color.reset }) catch {};
    }

    if (action == .urls or action == .workforce_urls) {
        if (members.len == 0) {
            stdout.print("  {s}No workforce found{s}\n", .{ Color.dim, Color.reset }) catch {};
        } else {
            for (members) |member| {
                stdout.print("  {s}  {s}http://localhost:{d}{s}\n", .{ member.name, Color.bold_cyan, member.port, Color.reset }) catch {};
            }
        }
    }
}

pub fn run(allocator: std.mem.Allocator, args: []const []const u8) !void {
    const orig_cp = enableWindowsConsole();
    defer restoreWindowsConsole(orig_cp);

    const stdout = std.io.getStdOut().writer();
    g_interrupted.store(false, .seq_cst);

    const base_port: u16 = 4455;
    var project_path: ?[]const u8 = null;
    var profile_flag: ?[]const u8 = null;

    // Parse arguments.
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

    // Resolve profile: --profile flag > TIMBAL_PROFILE env var > "default".
    const env_profile = std.process.getEnvVarOwned(allocator, "TIMBAL_PROFILE") catch |err| blk: {
        if (err == error.EnvironmentVariableNotFound) break :blk null;
        return err;
    };
    defer if (env_profile) |p| allocator.free(p);
    const profile: []const u8 = profile_flag orelse (env_profile orelse "default");

    // Load credentials and config for the profile.
    const stderr_writer = std.io.getStdErr().writer();

    const credentials_path = try getCredentialsPath(allocator);
    defer allocator.free(credentials_path);
    const credentials_content = std.fs.cwd().readFileAlloc(allocator, credentials_path, 1024 * 1024) catch |err| {
        if (err == error.FileNotFound) {
            try stderr_writer.print("Error: Timbal is not configured. Run '{s}timbal configure{s}' first.\n", .{ Color.bold_cyan, Color.reset });
            return;
        }
        return err;
    };
    defer allocator.free(credentials_content);

    const config_path = try getConfigPath(allocator);
    defer allocator.free(config_path);
    const config_content = std.fs.cwd().readFileAlloc(allocator, config_path, 1024 * 1024) catch |err| {
        if (err == error.FileNotFound) {
            try stderr_writer.print("Error: Timbal is not configured. Run '{s}timbal configure{s}' first.\n", .{ Color.bold_cyan, Color.reset });
            return;
        }
        return err;
    };
    defer allocator.free(config_content);

    const api_key = readValue(credentials_content, profile, "api_key") orelse {
        try stderr_writer.print("Error: No API key found for profile '{s}'. Run '{s}timbal configure --profile {s}{s}' to set it up.\n", .{ profile, Color.bold_cyan, profile, Color.reset });
        return;
    };
    const org_id = readValue(config_content, profile, "org") orelse {
        try stderr_writer.print("Error: No organization ID found for profile '{s}'. Run '{s}timbal configure --profile {s}{s}' to set it up.\n", .{ profile, Color.bold_cyan, profile, Color.reset });
        return;
    };
    const base_url = readValue(config_content, profile, "base_url") orelse "https://api.timbal.ai";
    const api_host = stripProtocol(base_url);

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

    // Save terminal state and install SIGINT handler (POSIX only).
    if (comptime !is_windows) {
        g_stdin_fd = std.io.getStdIn().handle;
        if (std.posix.tcgetattr(g_stdin_fd)) |termios| {
            g_original_termios = termios;
            g_termios_saved = true;
        } else |_| {}
        const sa = std.posix.Sigaction{
            .handler = .{ .handler = sigintHandler },
            .mask = std.posix.empty_sigset,
            .flags = 0,
        };
        std.posix.sigaction(std.posix.SIG.INT, &sa, null);
    }
    defer restoreTermios();

    var output_mutex = std.Thread.Mutex{};
    const command_state = try std.heap.page_allocator.create(CommandState);
    command_state.* = .{};
    var command_input_stop = std.atomic.Value(bool).init(false);
    var command_thread: ?std.Thread = null;
    var command_state_owned = true;
    defer {
        command_input_stop.store(true, .seq_cst);
        if (command_thread) |thread| {
            if (comptime !is_windows) {
                thread.join();
            } else {
                // Windows console reads cannot be interrupted here. Keep the
                // state process-lived if the detached reader wakes after run().
                thread.detach();
                command_state_owned = false;
            }
        }
        if (command_state_owned) std.heap.page_allocator.destroy(command_state);
    }
    var log_filter = LogFilterState.init(allocator);
    defer log_filter.deinit();

    // Shared env map with FORCE_COLOR for all services.
    var service_env = std.process.getEnvMap(allocator) catch return;
    defer service_env.deinit();
    service_env.put("FORCE_COLOR", "1") catch return;
    service_env.put("TIMBAL_LOG_EVENTS", "START,OUTPUT") catch return;
    service_env.put("TIMBAL_LOG_FORMAT", "dev") catch return;
    service_env.put("TIMBAL_DELTA_EVENTS", "true") catch return;
    service_env.put("TIMBAL_API_KEY", api_key) catch return;
    service_env.put("TIMBAL_ORG_ID", org_id) catch return;
    service_env.put("TIMBAL_API_HOST", api_host) catch return;

    // Build workforce env vars: id:port,id:port,...
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
            service_env.put("TIMBAL_START_WORKFORCE", workforce_buf.items) catch {};
            // TODO: Update the API blueprint to read TIMBAL_START_WORKFORCE, then remove TIMBAL_WORKFORCE.
            service_env.put("TIMBAL_WORKFORCE", workforce_buf.items) catch {};
        }
    }

    // Set API and UI port env vars for service discovery.
    if (api_port) |p| {
        const api_port_env = try std.fmt.allocPrint(allocator, "{d}", .{p});
        defer allocator.free(api_port_env);
        service_env.put("TIMBAL_START_API_PORT", api_port_env) catch {};
    }
    if (ui_port) |p| {
        const ui_port_env = try std.fmt.allocPrint(allocator, "{d}", .{p});
        defer allocator.free(ui_port_env);
        service_env.put("TIMBAL_START_UI_PORT", ui_port_env) catch {};
    }

    const command_input_enabled = std.io.getStdIn().isTty();
    if (command_input_enabled) {
        command_thread = std.Thread.spawn(.{}, commandInputFn, .{CommandInputCtx{
            .state = command_state,
            .stop = &command_input_stop,
        }}) catch null;
    }

    var services: ?RunningServices = try startServices(
        allocator,
        abs_project_path,
        members.items,
        has_ui,
        ui_port,
        has_api,
        api_port,
        &service_env,
        &output_mutex,
        &log_filter,
    );
    defer if (services) |*running| stopServices(running);

    if (command_input_enabled) {
        try stdout.print("\n{s}Press {s}h + enter{s} for commands, {s}r + enter{s} to restart, or Ctrl+C to stop.{s}\n", .{
            Color.dim,
            Color.bold_cyan,
            Color.dim,
            Color.bold_cyan,
            Color.dim,
            Color.reset,
        });
    }

    // Open browser after a short delay (UI takes priority over API).
    {
        const browser_url = preferredBrowserUrl(allocator, has_ui, ui_port, has_api, api_port);

        if (browser_url) |url| {
            const ctx = BrowserContext{ .allocator = allocator, .url = url };
            const browser_thread = std.Thread.spawn(.{}, openBrowserThread, .{ctx}) catch null;
            if (browser_thread) |t| t.detach();
        }
    }

    while (!g_interrupted.load(.seq_cst)) {
        const command = command_state.take();
        const target = command.target[0..command.target_len];
        switch (command.action) {
            .none => {},
            .help => printCommandHelp(&output_mutex),
            .status => {
                if (services) |*running| {
                    printServiceStatus(running, &output_mutex);
                } else {
                    printServiceStatus(null, &output_mutex);
                }
            },
            .open => {
                if (preferredBrowserUrl(allocator, has_ui, ui_port, has_api, api_port)) |url| {
                    defer allocator.free(url);
                    output_mutex.lock();
                    stdout.print("\n{s}Opening {s}{s}\n", .{ Color.dim, url, Color.reset }) catch {};
                    output_mutex.unlock();
                    openBrowserUrl(allocator, url);
                } else {
                    output_mutex.lock();
                    stdout.print("\n{s}No UI or API URL to open{s}\n", .{ Color.dim, Color.reset }) catch {};
                    output_mutex.unlock();
                }
            },
            .urls, .ui_url, .api_url, .workforce_urls => |action| {
                printProjectUrls(members.items, has_ui, ui_port, has_api, api_port, action, &output_mutex);
            },
            .focus_logs => {
                handleFocusCommand(target, members.items, has_ui, has_api, &log_filter, &output_mutex);
            },
            .toggle_mute => {
                handleMuteCommand(target, members.items, has_ui, has_api, &log_filter, &output_mutex);
            },
            .restart => {
                output_mutex.lock();
                stdout.print("\n{s}Restarting services...{s}\n", .{ Color.bold, Color.reset }) catch {};
                output_mutex.unlock();

                if (services) |*running| {
                    stopServices(running);
                    services = null;
                }
                services = try startServices(
                    allocator,
                    abs_project_path,
                    members.items,
                    has_ui,
                    ui_port,
                    has_api,
                    api_port,
                    &service_env,
                    &output_mutex,
                    &log_filter,
                );
            },
            .quit => break,
        }

        std.time.sleep(150 * std.time.ns_per_ms);
    }

    if (services) |*running| {
        stopServices(running);
        services = null;
    }

    // Free member data.
    for (members.items) |*member| {
        allocator.free(member.name);
        member.config.deinit(allocator);
    }
}

test "log targets include workforce members named like reserved services" {
    const members = [_]WorkforceMember{
        .{ .name = "ui", .config = undefined, .port = 8001 },
        .{ .name = "api", .config = undefined, .port = 8002 },
    };

    try std.testing.expect(isKnownLogTarget("ui", &members, false, false));
    try std.testing.expect(isKnownLogTarget("api", &members, false, false));
    try std.testing.expect(isKnownLogTarget("ui", &members, true, false));
    try std.testing.expect(isKnownLogTarget("api", &members, false, true));
}

test "mute can target workforce member named ui" {
    var log_filter = LogFilterState.init(std.testing.allocator);
    defer log_filter.deinit();

    try std.testing.expect(try log_filter.toggleMute("ui", true));
    try std.testing.expect(!log_filter.shouldPrint("ui", .workforce));
    try std.testing.expect(log_filter.shouldPrint("other", .ui));

    try std.testing.expect(!try log_filter.toggleMute("ui", true));
    try std.testing.expect(log_filter.shouldPrint("ui", .workforce));
}

test "mute still targets ui service when no member has that name" {
    var log_filter = LogFilterState.init(std.testing.allocator);
    defer log_filter.deinit();

    try std.testing.expect(try log_filter.toggleMute("ui", false));
    try std.testing.expect(!log_filter.shouldPrint("ui", .ui));
    try std.testing.expect(log_filter.shouldPrint("ui", .workforce));
}
