const std = @import("std");
const fs = std.fs;
const utils = @import("../utils.zig");

fn printUsageWithError(err: []const u8) !void {
    const stderr = std.io.getStdErr().writer();
    try stderr.print("{s}\n\n", .{err});
    try printUsage();
}

fn printUsage() !void {
    const stderr = std.io.getStdErr().writer();
    try stderr.writeAll("Build a container for the application ready for deployment.\n" ++
        "\n" ++
        "\x1b[1;32mUsage: \x1b[1;36mtimbal build \x1b[0;36m[OPTIONS] [PATH]\n" ++
        "\n" ++
        "\x1b[1;32mArguments:\n" ++
        "    \x1b[1;36mPATH\x1b[0m The path to the project to build\n" ++
        "\n" ++
        "\x1b[1;32mOptions:\n" ++
        "    \x1b[1;36m-t\x1b[0m, \x1b[1;36m--tag      \x1b[0mThe tag to use for the container\n" ++
        "        \x1b[1;36m--progress \x1b[0mSet type of progress output (\"auto\", \"quiet\", \"plain\", \"tty\",\n" ++
        "                   \"rawjson\"). Use plain to show container output (default \"auto\")\n" ++
        "        \x1b[1;36m--no-cache \x1b[0mDo not use cache when building the image\n" ++
        "\n" ++
        utils.global_options_help ++
        "\n");
}

const TimbalBuildArgs = struct {
    quiet: bool,
    verbose: bool,
    no_cache: bool,
    tag: ?[]const u8,
    progress: ?[]const u8,
    path: ?[]const u8,
};

fn parseArgs(args: []const []const u8) !TimbalBuildArgs {
    var path: ?[]const u8 = null;
    var tag: ?[]const u8 = null;
    var progress: ?[]const u8 = null;
    var verbose: bool = false;
    var quiet: bool = false;
    var no_cache: bool = false;

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            try printUsage();
            std.process.exit(0);
        } else if (std.mem.eql(u8, arg, "-v") or std.mem.eql(u8, arg, "--verbose")) {
            verbose = true;
        } else if (std.mem.eql(u8, arg, "-q") or std.mem.eql(u8, arg, "--quiet")) {
            quiet = true;
        } else if (std.mem.eql(u8, arg, "--no-cache")) {
            no_cache = true;
        } else if (std.mem.eql(u8, arg, "-t") or std.mem.eql(u8, arg, "--tag")) {
            if (tag != null) {
                try printUsageWithError("Error: multiple tags are not supported");
                std.process.exit(1);
            }
            if (i + 1 >= args.len) {
                try printUsageWithError("Error: --tag requires a value");
                std.process.exit(1);
            }
            tag = args[i + 1];
            i += 1;
        } else if (std.mem.eql(u8, arg, "--progress")) {
            if (progress != null) {
                try printUsageWithError("Error: multiple progress options provided");
                std.process.exit(1);
            }
            if (i + 1 >= args.len) {
                try printUsageWithError("Error: --progress requires a value");
                std.process.exit(1);
            }
            progress = args[i + 1];
            i += 1;
        } else if (!std.mem.startsWith(u8, arg, "-")) {
            if (path != null) {
                try printUsageWithError("Error: multiple target paths provided");
                std.process.exit(1);
            }
            path = arg;
        } else {
            try printUsageWithError("Error: unknown option");
            std.process.exit(1);
        }
    }

    return TimbalBuildArgs{
        .quiet = quiet,
        .verbose = verbose,
        .no_cache = no_cache,
        .tag = tag,
        .progress = progress,
        .path = path,
    };
}

const TimbalYaml = struct {
    system_packages: []const []const u8,
    run_commands: []const []const u8,
    fqn: []const u8,

    pub fn deinit(self: *TimbalYaml, allocator: std.mem.Allocator) void {
        for (self.system_packages) |pkg| {
            allocator.free(pkg);
        }
        allocator.free(self.system_packages);
        for (self.run_commands) |cmd| {
            allocator.free(cmd);
        }
        allocator.free(self.run_commands);
        allocator.free(self.fqn);
    }
};

// Note on custom YAML parser:
// We're using a simple custom YAML parser here because:
// 1. Our timbal.yaml structure is very minimal and well-defined
// 2. We avoid external Zig dependencies which are often unstable and difficult to maintain
// 3. We avoid C library dependencies which would complicate CLI installation
// 4. For our limited needs, a full YAML parser would be overkill
// This approach keeps our tool lightweight and installation simple.

fn parseTimbalYaml(allocator: std.mem.Allocator, timbal_yaml_file: fs.File) !TimbalYaml {
    var buf_reader = std.io.bufferedReader(timbal_yaml_file.reader());
    var in_stream = buf_reader.reader();
    var buf: [4096]u8 = undefined;

    var timbal_yaml_system_packages: std.ArrayList([]const u8) = std.ArrayList([]const u8).init(allocator);
    var timbal_yaml_run_commands: std.ArrayList([]const u8) = std.ArrayList([]const u8).init(allocator);
    var timbal_yaml_fqn: ?[]const u8 = null;

    // Track the current section of the file we're parsing.
    var current_section: enum {
        None,
        SystemPackages,
        RunCommands,
    } = .None;

    while (try in_stream.readUntilDelimiterOrEof(&buf, '\n')) |line| {
        const trimmed = std.mem.trim(u8, line, &std.ascii.whitespace);

        // Skip empty lines and comments.
        if (trimmed.len == 0 or std.mem.startsWith(u8, trimmed, "#")) {
            continue;
        }

        if (current_section == .SystemPackages and std.mem.startsWith(u8, trimmed, "-")) {
            var system_package = std.mem.trim(u8, trimmed[1..], &std.ascii.whitespace);
            system_package = std.mem.trim(u8, system_package, "\"");
            try timbal_yaml_system_packages.append(try allocator.dupe(u8, system_package));
            continue;
        }

        if (current_section == .RunCommands and std.mem.startsWith(u8, trimmed, "-")) {
            var run_command = std.mem.trim(u8, trimmed[1..], &std.ascii.whitespace);
            run_command = std.mem.trim(u8, run_command, "\"");
            try timbal_yaml_run_commands.append(try allocator.dupe(u8, run_command));
            continue;
        }

        if (std.mem.startsWith(u8, trimmed, "system_packages:")) {
            current_section = .SystemPackages;
        } else if (std.mem.startsWith(u8, trimmed, "run:")) {
            current_section = .RunCommands;
        } else if (std.mem.startsWith(u8, trimmed, "fqn:") or
            std.mem.startsWith(u8, trimmed, "agent: ") or
            std.mem.startsWith(u8, trimmed, "flow: "))
        {
            current_section = .None;
            var fqn = std.mem.trim(u8, trimmed[5..], &std.ascii.whitespace);
            fqn = std.mem.trim(u8, fqn, "\"");
            timbal_yaml_fqn = fqn;
        } else {
            current_section = .None;
        }
    }

    if (timbal_yaml_fqn == null) {
        try printUsageWithError("Error: Fully Qualified Name (FQN) not found in timbal.yaml");
        std.process.exit(1);
    }

    // We'd need to reset the file pointer if we want to use the file again.
    // try timbal_yaml_file.seekTo(0);

    return TimbalYaml{
        .system_packages = try timbal_yaml_system_packages.toOwnedSlice(),
        .run_commands = try timbal_yaml_run_commands.toOwnedSlice(),
        .fqn = try allocator.dupe(u8, timbal_yaml_fqn.?),
    };
}

fn writeDockerfile(
    allocator: std.mem.Allocator,
    timbal_yaml: TimbalYaml,
    app_dir: fs.Dir,
) !void {
    // TODO Think default CMD. If we want the container to run indefinitely, better use "sleep infinity".
    const dockerfile_template =
        \\FROM ubuntu:22.04
        \\
        \\ENV PYTHONUNBUFFERED=1
        \\ENV PYTHONDONTWRITEBYTECODE=1
        \\ENV DEBIAN_FRONTEND=noninteractive
        \\
        \\WORKDIR /timbal
        \\
        \\RUN apt update && \
        \\    apt install -yqq --no-install-recommends \
        \\        {s} && \
        \\    apt clean && \
        \\    rm -rf /var/lib/apt/lists/* && \
        \\    curl -LsSf https://astral.sh/uv/install.sh | sh
        \\
        \\ENV PATH="/root/.local/bin:$PATH"
        \\
        \\COPY . .
        \\
        \\RUN uv sync --python 3.12 --python-preference managed && \
        \\    rm -rf $HOME/.cache/uv
        \\{s}
        \\ENV PATH=".venv/bin:$PATH"
        \\
        \\ENV TIMBAL_RUNNABLE={s}
        \\
        \\CMD ["tail", "-f", "/dev/null"]
        \\
    ;

    const dockerfile = try app_dir.createFile("Dockerfile", .{});
    defer dockerfile.close();

    // Format the system packages for the Dockerfile.
    var system_packages_buffer = std.ArrayList(u8).init(allocator);
    defer system_packages_buffer.deinit();

    try system_packages_buffer.appendSlice("curl \\\n        ca-certificates \\\n        git");

    for (timbal_yaml.system_packages) |pkg| {
        try system_packages_buffer.appendSlice(" \\\n        ");
        try system_packages_buffer.appendSlice(pkg);
    }

    // Format the run commands for the Dockerfile.
    var run_commands_buffer = std.ArrayList(u8).init(allocator);
    defer run_commands_buffer.deinit();

    for (timbal_yaml.run_commands) |cmd| {
        try run_commands_buffer.appendSlice("\nRUN ");
        try run_commands_buffer.appendSlice(cmd);
        try run_commands_buffer.appendSlice("\n");
    }

    try dockerfile.writer().print(dockerfile_template, .{
        system_packages_buffer.items,
        run_commands_buffer.items,
        timbal_yaml.fqn,
    });
}

fn buildContainer(
    allocator: std.mem.Allocator,
    tag: ?[]const u8,
    progress: ?[]const u8,
    quiet: bool,
    no_cache: bool,
    path: []const u8,
) !void {
    // TODO Print the docker tag that we're using.
    const docker_tag = tag orelse blk: {
        const app_name = std.fs.path.basename(path);
        const result = try std.fmt.allocPrint(allocator, "{s}:latest", .{app_name});
        break :blk result;
    };
    defer if (tag == null) allocator.free(docker_tag);

    const docker_progress = progress orelse "auto";

    var docker_build_args = std.ArrayList([]const u8).init(allocator);
    defer docker_build_args.deinit();

    try docker_build_args.appendSlice(&[_][]const u8{
        "docker",     "build",
        "-t",         docker_tag,
        "--progress", docker_progress,
    });

    if (quiet) {
        try docker_build_args.append("-q");
    }

    if (no_cache) {
        try docker_build_args.append("--no-cache");
    }

    try docker_build_args.append(".");

    // Don't use std.process.Child.run here. We want to inherit the stderr and stdout
    // to stream the output to the main console.
    var child = std.process.Child.init(docker_build_args.items, allocator);
    child.cwd = path;
    child.stderr_behavior = .Inherit;
    child.stdout_behavior = .Inherit;

    try child.spawn();
    _ = try child.wait();
}

pub fn run(allocator: std.mem.Allocator, args: []const []const u8) !void {
    const parsed_args = try parseArgs(args);

    const target_path = parsed_args.path orelse ".";

    const cwd = fs.cwd();

    const use_current_dir = std.mem.eql(u8, target_path, ".");

    if (!use_current_dir) {
        // Ensure that the target path exists
        const target_stat = cwd.statFile(target_path) catch |err| {
            if (err == error.FileNotFound) {
                try printUsageWithError("Error: target path does not exist");
                return;
            } else {
                return err;
            }
        };

        // Ensure it is a directory
        if (target_stat.kind != .directory) {
            try printUsageWithError("Error: target path is not a directory");
            return;
        }
    }

    var app_dir = if (use_current_dir)
        cwd
    else
        try cwd.openDir(target_path, .{});

    const path = try app_dir.realpathAlloc(allocator, ".");
    defer allocator.free(path);

    if (!parsed_args.quiet) {
        std.debug.print("Building application in {s}\n", .{path});
    }

    // Check if timbal.yaml exists in the directory (i.e. is a valid timbal project)
    const timbal_yaml_path = "timbal.yaml";
    const timbal_yaml_file = app_dir.openFile(timbal_yaml_path, .{}) catch |err| {
        if (err == error.FileNotFound) {
            try printUsageWithError("Error: timbal.yaml not found");
            return;
        } else {
            return err;
        }
    };

    var timbal_yaml = try parseTimbalYaml(allocator, timbal_yaml_file);
    defer timbal_yaml.deinit(allocator);

    // Right now we create and edit the file in the cwd. We don't create a temporary directory,
    // so that we're able to debug the Dockerfile during development.
    try writeDockerfile(allocator, timbal_yaml, app_dir);

    try buildContainer(
        allocator,
        parsed_args.tag,
        parsed_args.progress,
        parsed_args.quiet,
        parsed_args.no_cache,
        path,
    );
}
