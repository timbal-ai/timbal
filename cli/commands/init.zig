const std = @import("std");
const fs = std.fs;
const utils = @import("../utils.zig");

// Embed template files into the binary.
const dockerignore = @embedFile("../init-templates/.dockerignore");
const pyproject_toml = @embedFile("../init-templates/pyproject.toml");
const timbal_yaml = @embedFile("../init-templates/timbal.yaml");
const agent_py = @embedFile("../init-templates/agent.py");
const workflow_py = @embedFile("../init-templates/workflow.py");

fn printUsageWithError(err: []const u8) !void {
    const stderr = std.io.getStdErr().writer();
    try stderr.print("{s}\n\n", .{err});
    try printUsage();
}

fn printUsage() !void {
    const stderr = std.io.getStdErr().writer();
    try stderr.writeAll("Create a new project.\n" ++
        "\n" ++
        "\x1b[1;32mUsage: \x1b[1;36mtimbal init \x1b[0;36m[OPTIONS] [PATH]\n" ++
        "\n" ++
        "\x1b[1;32mArguments:\n" ++
        "    \x1b[0;36m[PATH]\x1b[0m The path to use for the project\n" ++
        "\n" ++
        "\x1b[1;32mOptions:\n" ++
        "    \x1b[1;36m--agent \x1b[0m Initialize a timbal project as an agent (default)\n" ++
        "    \x1b[1;36m--workflow \x1b[0m Initialize a timbal project as a workflow\n" ++
        "\n" ++
        utils.global_options_help ++
        "\n");
}

pub fn run(allocator: std.mem.Allocator, args: []const []const u8) !void {
    var arg_path: ?[]const u8 = null;
    var verbose: bool = false;
    var quiet: bool = false;
    var app_type: []const u8 = "agent";

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
        } else if (std.mem.eql(u8, arg, "--agent")) {
            app_type = "agent";
        } else if (std.mem.eql(u8, arg, "--workflow")) {
            app_type = "workflow";
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

    const target_path = arg_path orelse ".";

    const cwd = fs.cwd();

    const use_current_dir = std.mem.eql(u8, target_path, ".");

    if (!use_current_dir) {
        try cwd.makePath(target_path);
    }

    var app_dir = if (use_current_dir)
        cwd
    else
        try cwd.openDir(target_path, .{});

    const path = try app_dir.realpathAlloc(allocator, ".");
    defer allocator.free(path);

    if (!quiet) {
        std.debug.print("Initializing timbal project in {s}\n", .{path});
    }

    // Change the name of the app in the pyproject.toml file.
    const app_name = std.fs.path.basename(path);
    const pyproject_toml_replaced = try std.mem.replaceOwned(u8, allocator, pyproject_toml, "{{app_name}}", app_name);
    defer allocator.free(pyproject_toml_replaced);

    // Change the FQN in the timbal.yaml file.
    const fully_qualified_name = if (std.mem.eql(u8, app_type, "agent"))
        "agent.py::agent"
    else
        "workflow.py::workflow";
    const component_id = try utils.genSecureId(allocator);
    defer allocator.free(component_id);
    const timbal_yaml_with_id = try std.mem.replaceOwned(u8, allocator, timbal_yaml, "{{id}}", component_id);
    defer allocator.free(timbal_yaml_with_id);
    const timbal_yaml_replaced = try std.mem.replaceOwned(u8, allocator, timbal_yaml_with_id, "{{fully_qualified_name}}", fully_qualified_name);
    defer allocator.free(timbal_yaml_replaced);

    var init_templates = std.ArrayList(struct { content: []const u8, dist: []const u8 }).init(allocator);
    try init_templates.append(.{ .content = dockerignore, .dist = ".dockerignore" });
    try init_templates.append(.{ .content = pyproject_toml_replaced, .dist = "pyproject.toml" });
    try init_templates.append(.{ .content = timbal_yaml_replaced, .dist = "timbal.yaml" });

    if (std.mem.eql(u8, app_type, "agent")) {
        try init_templates.append(.{ .content = agent_py, .dist = "agent.py" });
    } else {
        try init_templates.append(.{ .content = workflow_py, .dist = "workflow.py" });
    }

    for (init_templates.items) |init_template| {
        const dist_file = try app_dir.createFile(init_template.dist, .{});
        defer dist_file.close();
        try dist_file.writeAll(init_template.content);
        if (verbose) {
            std.debug.print("Created {s}\n", .{init_template.dist});
        }
    }
    init_templates.deinit();

    if (!quiet) {
        std.debug.print("Setting up uv project...\n", .{});
    }

    const uv_args = [_][]const u8{
        "uv",                  "sync",
        "--python",            "3.12",
        "--python-preference", "managed",
    };

    const uv_result = try std.process.Child.run(.{
        .allocator = allocator,
        .argv = &uv_args,
        .cwd = target_path,
    });
    defer allocator.free(uv_result.stdout);
    defer allocator.free(uv_result.stderr);

    if (uv_result.stderr.len > 0 and verbose) {
        std.debug.print("{s}", .{uv_result.stderr});
    }
}
