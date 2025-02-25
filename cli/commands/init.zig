const std = @import("std");
const fs = std.fs;


// Embed template files into the binary.
const dockerignore = @embedFile("../init-templates/.dockerignore");
const pyproject_toml = @embedFile("../init-templates/pyproject.toml");
const timbal_yaml = @embedFile("../init-templates/timbal.yaml");
const flow_py = @embedFile("../init-templates/flow.py");


fn printUsageWithError(err: []const u8) !void {
    const stderr = std.io.getStdErr().writer();
    try stderr.print("{s}\n\n", .{err});
    try printUsage();
}


fn printUsage() !void {
    const stderr = std.io.getStdErr().writer();
    try stderr.writeAll(
        "Create a new project.\n" ++
        "\n" ++
        "\x1b[1;32mUsage: \x1b[1;36mtimbal init \x1b[0;36m[OPTIONS] [PATH]\n" ++
        "\n" ++
        "\x1b[1;32mArguments:\n" ++
        "    \x1b[0;36m[PATH]\x1b[0m The path to use for the project\n" ++
        "\n" ++
        "\x1b[1;32mOptions:\n" ++
        "\n" ++
        "\x1b[1;32mGlobal options:\n" ++
        "    \x1b[1;36m-q\x1b[0m, \x1b[1;36m--quiet      \x1b[0mDo not print any output\n" ++
        "    \x1b[1;36m-v\x1b[0m, \x1b[1;36m--verbose\x1b[0;36m... \x1b[0mUse verbose output\n" ++
        "    \x1b[1;36m-h\x1b[0m, \x1b[1;36m--help       \x1b[0mDisplay the concise help for this command\n" ++
        "    \x1b[1;36m-V\x1b[0m, \x1b[1;36m--version    \x1b[0mDisplay the timbal version\n" ++
        "\n"
    );
}


pub fn run(allocator: std.mem.Allocator, args: []const []const u8) !void {
    var arg_path: ?[]const u8 = null;
    var verbose: bool = false;
    var quiet: bool = false;

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            try printUsage();
            return;
        } else if (std.mem.eql(u8, arg, "-V") or std.mem.eql(u8, arg, "--version")) {
            // TODO Real versioning
            std.debug.print("timbal version 0.1.0\n", .{});
            return;
        } else if (std.mem.eql(u8, arg, "-v") or std.mem.eql(u8, arg, "--verbose")) {
            verbose = true;
        } else if (std.mem.eql(u8, arg, "-q") or std.mem.eql(u8, arg, "--quiet")) {
            quiet = true;
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
    const pyproject_toml_replaced = try std.mem.replaceOwned(
        u8, allocator, pyproject_toml, "{{app_name}}", app_name);
    defer allocator.free(pyproject_toml_replaced);

    const init_templates = [_]struct { content: []const u8, dist: []const u8 } {
        .{ .content = dockerignore, .dist = ".dockerignore" },
        .{ .content = pyproject_toml_replaced, .dist = "pyproject.toml" },
        .{ .content = timbal_yaml, .dist = "timbal.yaml" },
        .{ .content = flow_py, .dist = "flow.py" },
    };

    for (init_templates) |init_template| {
        const dist_file = try app_dir.createFile(init_template.dist, .{});
        defer dist_file.close();
        try dist_file.writeAll(init_template.content);
        if (verbose) {
            std.debug.print("Created {s}\n", .{init_template.dist});
        }
    }

    if (!quiet) {
        std.debug.print("Setting up uv project...\n", .{});
    }

    const uv_args = [_][]const u8{
        "uv", "sync",
        "--python", "3.12",
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
