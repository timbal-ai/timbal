const std = @import("std");

// ANSI color codes for terminal output.
pub const Color = struct {
    pub const reset = "\x1b[0m";
    pub const bold = "\x1b[1m";
    pub const dim = "\x1b[2m";
    pub const yellow = "\x1b[33m";
    pub const bold_yellow = "\x1b[1;33m";
    pub const cyan = "\x1b[36m";
    pub const bold_green = "\x1b[1;32m";
    pub const bold_cyan = "\x1b[1;36m";
};

// Shared help text for global options.
pub const global_options_help =
    "\x1b[1;32mGlobal options:\n" ++
    "    \x1b[1;36m-q\x1b[0m, \x1b[1;36m--quiet      \x1b[0mDo not print any output\n" ++
    "    \x1b[1;36m-v\x1b[0m, \x1b[1;36m--verbose    \x1b[0mUse verbose output\n" ++
    "    \x1b[1;36m-h\x1b[0m, \x1b[1;36m--help       \x1b[0mDisplay the concise help for this command\n" ++
    "    \x1b[1;36m--profile <NAME>\x1b[0m Use a named profile\n";

pub const ProjectType = enum {
    agent,
    workflow,
};

pub const blueprint_api_url = "https://github.com/timbal-ai/blueprint-api/archive/refs/heads/main.tar.gz";
pub const blueprint_ui_simple_chat_url = "https://github.com/timbal-ai/blueprint-ui-simple-chat/archive/refs/heads/main.tar.gz";

// Embed template files into the binary.
const tmpl_pyproject_toml = @embedFile("init-templates/pyproject.toml");
const tmpl_timbal_yaml = @embedFile("init-templates/timbal.yaml");
const tmpl_agent_py = @embedFile("init-templates/agent.py");
const tmpl_workflow_py = @embedFile("init-templates/workflow.py");

pub const reserved_workforce_names = [_][]const u8{ "ui", "api", "workforce", "all" };

pub const WorkforceNameError = error{
    InvalidWorkforceName,
    ReservedWorkforceName,
    WorkforceMemberExists,
};

/// Validates a workforce member name (directory name and `timbal start` routing key).
/// Allows any character except the routing/path-breaking set `= : , / \` and control
/// chars; spaces and mixed case are fine; must not start with `.` or `-`.
pub fn validateWorkforceMemberName(name: []const u8) WorkforceNameError!void {
    if (name.len == 0 or name.len > 64) return error.InvalidWorkforceName;

    for (reserved_workforce_names) |reserved| {
        if (std.ascii.eqlIgnoreCase(name, reserved)) return error.ReservedWorkforceName;
    }

    const first = name[0];
    if (first == '.' or first == '-') return error.InvalidWorkforceName;

    for (name) |c| {
        switch (c) {
            '=', ':', ',', '/', '\\' => return error.InvalidWorkforceName,
            else => if (c < 0x20 or c == 0x7f) return error.InvalidWorkforceName,
        }
    }
}

/// Trims surrounding whitespace, validates, and returns an owned name (case preserved).
pub fn prepareWorkforceMemberName(allocator: std.mem.Allocator, raw: []const u8) ![]u8 {
    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
    try validateWorkforceMemberName(trimmed);
    return allocator.dupe(u8, trimmed);
}

pub fn printWorkforceNameError(err: anyerror, name: []const u8) void {
    const stderr = std.io.getStdErr().writer();
    switch (err) {
        error.InvalidWorkforceName => {
            stderr.print(
                "Error: invalid workforce name '{s}'. Spaces and mixed case are fine, but it cannot " ++
                    "contain = : , / \\, cannot start with '.' or '-', and must be 1-64 characters.\n",
                .{name},
            ) catch {};
        },
        error.ReservedWorkforceName => {
            stderr.print("Error: workforce name '{s}' is reserved (ui, api, workforce, all).\n", .{name}) catch {};
        },
        error.WorkforceMemberExists => {
            stderr.print("Error: workforce member '{s}' already exists. Re-run with --force to replace it.\n", .{name}) catch {};
        },
        else => {},
    }
}

/// Parsed representation of a timbal.yaml configuration file.
pub const TimbalYaml = struct {
    id: []const u8,
    type: []const u8,
    fqn: []const u8,
    system_packages: []const []const u8,
    run_commands: []const []const u8,

    pub fn deinit(self: *TimbalYaml, allocator: std.mem.Allocator) void {
        allocator.free(self.id);
        allocator.free(self.type);
        allocator.free(self.fqn);
        for (self.system_packages) |pkg| allocator.free(pkg);
        allocator.free(self.system_packages);
        for (self.run_commands) |cmd| allocator.free(cmd);
        allocator.free(self.run_commands);
    }
};

/// Strip surrounding quotes from a YAML value.
fn stripQuotes(value: []const u8) []const u8 {
    if (value.len >= 2 and value[0] == '"' and value[value.len - 1] == '"') {
        return value[1 .. value.len - 1];
    }
    return value;
}

/// Parse a timbal.yaml file from its content string.
/// Returns a TimbalYaml struct (caller owns all memory) or null if required fields are missing.
/// Prints an error message to stderr if required fields are missing.
pub fn parseTimbalYaml(allocator: std.mem.Allocator, content: []const u8) ?TimbalYaml {
    var id: ?[]const u8 = null;
    var comp_type: ?[]const u8 = null;
    var fqn: ?[]const u8 = null;
    var system_packages = std.ArrayList([]const u8).init(allocator);
    var run_commands = std.ArrayList([]const u8).init(allocator);

    var current_section: enum { None, SystemPackages, RunCommands } = .None;

    var lines = std.mem.splitScalar(u8, content, '\n');
    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");

        // Skip empty lines and comments.
        if (trimmed.len == 0 or std.mem.startsWith(u8, trimmed, "#")) continue;

        // List items in current section.
        if (current_section == .SystemPackages and std.mem.startsWith(u8, trimmed, "-")) {
            const value = stripQuotes(std.mem.trim(u8, trimmed[1..], " \t"));
            system_packages.append(allocator.dupe(u8, value) catch return null) catch return null;
            continue;
        }
        if (current_section == .RunCommands and std.mem.startsWith(u8, trimmed, "-")) {
            const value = stripQuotes(std.mem.trim(u8, trimmed[1..], " \t"));
            run_commands.append(allocator.dupe(u8, value) catch return null) catch return null;
            continue;
        }

        // Top-level keys.
        if (std.mem.startsWith(u8, trimmed, "_id:")) {
            current_section = .None;
            const value = stripQuotes(std.mem.trim(u8, trimmed[4..], " \t"));
            id = allocator.dupe(u8, value) catch return null;
        } else if (std.mem.startsWith(u8, trimmed, "_type:")) {
            current_section = .None;
            const value = stripQuotes(std.mem.trim(u8, trimmed[6..], " \t"));
            comp_type = allocator.dupe(u8, value) catch return null;
        } else if (std.mem.startsWith(u8, trimmed, "fqn:")) {
            current_section = .None;
            const value = stripQuotes(std.mem.trim(u8, trimmed[4..], " \t"));
            fqn = allocator.dupe(u8, value) catch return null;
        } else if (std.mem.startsWith(u8, trimmed, "system_packages:")) {
            current_section = .SystemPackages;
        } else if (std.mem.startsWith(u8, trimmed, "run:")) {
            current_section = .RunCommands;
        } else {
            current_section = .None;
        }
    }

    // Check for missing required fields and print appropriate error messages
    const stderr = std.io.getStdErr().writer();
    var has_error = false;

    if (id == null) {
        stderr.print("Error: timbal.yaml is missing required field '_id'.\n", .{}) catch {};
        has_error = true;
    }

    if (comp_type == null) {
        stderr.print("Error: timbal.yaml is missing required field '_type'.\n", .{}) catch {};
        has_error = true;
    }

    if (fqn == null) {
        stderr.print("Error: timbal.yaml is missing required field 'fqn'.\n", .{}) catch {};
        has_error = true;
    }

    if (has_error) {
        stderr.print("\nYou may be using a modified or outdated manifest file.\n", .{}) catch {};
        stderr.print("Please ensure your timbal.yaml contains all required fields: _id, _type, and fqn.\n", .{}) catch {};

        // Clean up any allocated memory before returning null
        if (id) |s| allocator.free(s);
        if (comp_type) |s| allocator.free(s);
        if (fqn) |s| allocator.free(s);
        for (system_packages.items) |pkg| allocator.free(pkg);
        system_packages.deinit();
        for (run_commands.items) |cmd| allocator.free(cmd);
        run_commands.deinit();
        return null;
    }

    const owned_system_packages = system_packages.toOwnedSlice() catch {
        allocator.free(id.?);
        allocator.free(comp_type.?);
        allocator.free(fqn.?);
        for (system_packages.items) |pkg| allocator.free(pkg);
        system_packages.deinit();
        for (run_commands.items) |cmd| allocator.free(cmd);
        run_commands.deinit();
        return null;
    };

    const owned_run_commands = run_commands.toOwnedSlice() catch {
        allocator.free(id.?);
        allocator.free(comp_type.?);
        allocator.free(fqn.?);
        for (owned_system_packages) |pkg| allocator.free(pkg);
        allocator.free(owned_system_packages);
        for (run_commands.items) |cmd| allocator.free(cmd);
        run_commands.deinit();
        return null;
    };

    return TimbalYaml{
        .id = id.?,
        .type = comp_type.?,
        .fqn = fqn.?,
        .system_packages = owned_system_packages,
        .run_commands = owned_run_commands,
    };
}

/// Generates a 32-character lowercase hex string from 16 cryptographically secure random bytes.
/// The caller owns the returned memory and must free it with the provided allocator.
pub fn genSecureId(allocator: std.mem.Allocator) ![]u8 {
    var bytes: [16]u8 = undefined;
    std.crypto.random.bytes(&bytes);

    return std.fmt.allocPrint(allocator, "{}", .{std.fmt.fmtSliceHexLower(&bytes)});
}

/// Downloads and extracts a blueprint tarball into a project directory.
pub fn fetchBlueprint(allocator: std.mem.Allocator, project_path: []const u8, dest_dir: []const u8, tarball_url: []const u8) !void {
    const stdout = std.io.getStdOut().writer();
    const dest_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ project_path, dest_dir });
    defer allocator.free(dest_path);

    // Create the destination directory
    std.fs.makeDirAbsolute(dest_path) catch |err| {
        if (err != error.PathAlreadyExists) return err;
    };

    // Download the tarball using Zig's HTTP client
    try stdout.print("  {s}Downloading {s}...{s}\n", .{ Color.dim, dest_dir, Color.reset });

    var client: std.http.Client = .{ .allocator = allocator };
    defer client.deinit();

    var response_body = std.ArrayList(u8).init(allocator);
    defer response_body.deinit();

    const result = client.fetch(.{
        .location = .{ .url = tarball_url },
        .response_storage = .{ .dynamic = &response_body },
        .max_append_size = 50 * 1024 * 1024,
    }) catch {
        std.debug.print("Error: Failed to download {s} blueprint.\n", .{dest_dir});
        return error.HttpError;
    };

    if (result.status != .ok) {
        std.debug.print("Error: Failed to download {s} blueprint (HTTP {d}).\n", .{ dest_dir, @intFromEnum(result.status) });
        return error.HttpError;
    }

    try stdout.print("  {s}Extracting {s}...{s}\n", .{ Color.dim, dest_dir, Color.reset });

    // Decompress gzip and extract tar into the destination directory
    var dir = try std.fs.openDirAbsolute(dest_path, .{});
    defer dir.close();

    var fbs = std.io.fixedBufferStream(response_body.items);
    var gzip = std.compress.gzip.decompressor(fbs.reader());
    std.tar.pipeToFileSystem(dir, gzip.reader(), .{
        .strip_components = 1,
    }) catch {
        std.debug.print("Error: Failed to extract {s} blueprint.\n", .{dest_dir});
        return error.ExtractionFailed;
    };
}

/// Creates a new workforce member directory with template files.
/// Returns the name used (caller owns the memory).
pub fn addWorkforceMember(allocator: std.mem.Allocator, app_dir: std.fs.Dir, project_name: []const u8, project_type: ProjectType, name: []const u8) ![]u8 {
    const prepared = try prepareWorkforceMemberName(allocator, name);
    errdefer allocator.free(prepared);

    // Determine the fully qualified name based on project type
    const fqn = switch (project_type) {
        .agent => "agent.py::agent",
        .workflow => "workflow.py::workflow",
    };

    const member_dir_name = try std.fmt.allocPrint(allocator, "workforce/{s}", .{prepared});
    defer allocator.free(member_dir_name);

    if (app_dir.access(member_dir_name, .{})) |_| {
        return error.WorkforceMemberExists;
    } else |_| {}

    try app_dir.makePath(member_dir_name);

    // Open the member directory to write template files into it
    var member_dir = try app_dir.openDir(member_dir_name, .{});
    defer member_dir.close();

    // Replace template variables and write files into the workforce member directory
    const pyproject_content = try std.mem.replaceOwned(u8, allocator, tmpl_pyproject_toml, "{{app_name}}", project_name);
    defer allocator.free(pyproject_content);

    const component_id = try genSecureId(allocator);
    defer allocator.free(component_id);
    const timbal_yaml_with_id = try std.mem.replaceOwned(u8, allocator, tmpl_timbal_yaml, "{{id}}", component_id);
    defer allocator.free(timbal_yaml_with_id);
    const project_type_str = switch (project_type) {
        .agent => "agent",
        .workflow => "workflow",
    };
    const timbal_yaml_with_type = try std.mem.replaceOwned(u8, allocator, timbal_yaml_with_id, "{{type}}", project_type_str);
    defer allocator.free(timbal_yaml_with_type);
    const timbal_yaml_content = try std.mem.replaceOwned(u8, allocator, timbal_yaml_with_type, "{{fully_qualified_name}}", fqn);
    defer allocator.free(timbal_yaml_content);

    const app_py = switch (project_type) {
        .agent => tmpl_agent_py,
        .workflow => tmpl_workflow_py,
    };

    const app_py_name = switch (project_type) {
        .agent => "agent.py",
        .workflow => "workflow.py",
    };

    // Write template files into the member directory
    const templates = [_]struct { content: []const u8, name: []const u8 }{
        .{ .content = pyproject_content, .name = "pyproject.toml" },
        .{ .content = timbal_yaml_content, .name = "timbal.yaml" },
        .{ .content = app_py, .name = app_py_name },
    };

    for (templates) |tmpl| {
        const file = try member_dir.createFile(tmpl.name, .{});
        defer file.close();
        try file.writeAll(tmpl.content);
    }

    return prepared;
}

test "validateWorkforceMemberName accepts flexible names" {
    try validateWorkforceMemberName("support-bot");
    try validateWorkforceMemberName("ingest2");
    try validateWorkforceMemberName("support_bot");
    try validateWorkforceMemberName("agent-2");
    try validateWorkforceMemberName("bad-");
    try validateWorkforceMemberName("My Agent");
    try validateWorkforceMemberName("Agent.v2");
    try validateWorkforceMemberName("2fa-bot");
}

test "validateWorkforceMemberName rejects invalid and reserved" {
    try std.testing.expectError(error.InvalidWorkforceName, validateWorkforceMemberName(""));
    try std.testing.expectError(error.InvalidWorkforceName, validateWorkforceMemberName(".hidden"));
    try std.testing.expectError(error.InvalidWorkforceName, validateWorkforceMemberName("-leading"));
    try std.testing.expectError(error.InvalidWorkforceName, validateWorkforceMemberName("a/b"));
    try std.testing.expectError(error.InvalidWorkforceName, validateWorkforceMemberName("a=b"));
    try std.testing.expectError(error.InvalidWorkforceName, validateWorkforceMemberName("a:b"));
    try std.testing.expectError(error.InvalidWorkforceName, validateWorkforceMemberName("a,b"));
    try std.testing.expectError(error.InvalidWorkforceName, validateWorkforceMemberName("a\\b"));
    try std.testing.expectError(error.ReservedWorkforceName, validateWorkforceMemberName("ui"));
    try std.testing.expectError(error.ReservedWorkforceName, validateWorkforceMemberName("UI"));
}

test "prepareWorkforceMemberName trims and preserves case" {
    const name = try prepareWorkforceMemberName(std.testing.allocator, "  My Agent  ");
    defer std.testing.allocator.free(name);
    try std.testing.expectEqualStrings("My Agent", name);
}

test "prepareWorkforceMemberName rejects whitespace-only input" {
    try std.testing.expectError(error.InvalidWorkforceName, prepareWorkforceMemberName(std.testing.allocator, "   "));
}

test "validateWorkforceMemberName enforces max length" {
    var long: [65]u8 = undefined;
    @memset(&long, 'a');
    try std.testing.expectError(error.InvalidWorkforceName, validateWorkforceMemberName(&long));
    try validateWorkforceMemberName(long[0..64]);
}

test "validateWorkforceMemberName rejects control characters" {
    try std.testing.expectError(error.InvalidWorkforceName, validateWorkforceMemberName("agent\x01"));
    try std.testing.expectError(error.InvalidWorkforceName, validateWorkforceMemberName("agent\x7f"));
}
