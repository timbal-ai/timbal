const std = @import("std");
const fs = std.fs;


// Embedded version.
const timbal_version = @import("../version.zig").timbal_version;


fn printUsageWithError(err: []const u8) !void {
    const stderr = std.io.getStdErr().writer();
    try stderr.print("{s}\n\n", .{err});
    try printUsage();
}


// TODO Think if we want to add -it as options.
fn printUsage() !void {
    const stderr = std.io.getStdErr().writer();
    try stderr.writeAll(
        "Run a command inside the built container.\n" ++
        "\n" ++
        "\x1b[1;32mUsage: \x1b[1;36mtimbal run \x1b[0;36m[OPTIONS] IMAGE [COMMAND]\n" ++
        "\n" ++
        "\x1b[1;32mArguments:\n" ++
        "    \x1b[1;36mIMAGE\x1b[0m   The tag of the container to run\n" ++
        "    \x1b[1;36mCOMMAND\x1b[0m The command to run inside the container\n" ++
        "            If not provided, the container will run the default command (CMD instruction)\n" ++
        "\n" ++
        "\x1b[1;32mOptions:\n" ++
        "    \x1b[1;36m-d\x1b[0m, \x1b[1;36m--detach  \x1b[0mRun container in the background and print container ID\n" ++
        "    \x1b[1;36m-p\x1b[0m, \x1b[1;36m--publish \x1b[0mPublish a container's port to the host, e.g. -p 8000\n" ++
        "\n" ++
        "\x1b[1;32mGlobal options:\n" ++
        "    \x1b[1;36m-q\x1b[0m, \x1b[1;36m--quiet      \x1b[0mDo not print any output\n" ++
        "    \x1b[1;36m-v\x1b[0m, \x1b[1;36m--verbose\x1b[0;36m... \x1b[0mUse verbose output\n" ++
        "    \x1b[1;36m-h\x1b[0m, \x1b[1;36m--help       \x1b[0mDisplay the concise help for this command\n" ++
        "    \x1b[1;36m-V\x1b[0m, \x1b[1;36m--version    \x1b[0mDisplay the timbal version\n" ++
        "\n"
    );
}


const TimbalRunArgs = struct {
    quiet: bool,
    verbose: bool,
    detach: bool,
    image: []const u8,
    command: ?[]const u8,
    publish: ?[]const u8,
};


fn parseArgs(args: []const []const u8) !TimbalRunArgs {
    var quiet: bool = false;
    var verbose: bool = false;
    var detach: bool = false;
    var image: ?[]const u8 = null;
    var command: ?[]const u8 = null;
    var publish: ?[]const u8 = null;

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            try printUsage();
            std.process.exit(0);
        } else if (std.mem.eql(u8, arg, "-V") or std.mem.eql(u8, arg, "--version")) {
            std.debug.print("Timbal {s}\n", .{timbal_version});
            std.process.exit(0);
        } else if (std.mem.eql(u8, arg, "-v") or std.mem.eql(u8, arg, "--verbose")) {
            verbose = true;
        } else if (std.mem.eql(u8, arg, "-q") or std.mem.eql(u8, arg, "--quiet")) {
            quiet = true;
        } else if (std.mem.eql(u8, arg, "-d") or std.mem.eql(u8, arg, "--detach")) {
            detach = true;
        } else if (std.mem.eql(u8, arg, "-p") or std.mem.eql(u8, arg, "--publish")) {
            if (i + 1 >= args.len) {
                try printUsageWithError("Error: --publish requires a value");
                std.process.exit(1);
            }
            if (publish != null) {
                try printUsageWithError("Error: multiple publish options provided");
                std.process.exit(1);
            }
            publish = args[i + 1];
            i += 1;
        } else if (!std.mem.startsWith(u8, arg, "-")) {
            if (image == null) {
                image = arg;
            } else if (command == null) {
                command = arg;
            } else {
                try printUsageWithError("Error: multiple commands provided");
                std.process.exit(1);
            }
        } else {
            try printUsageWithError("Error: unknown option");
            std.process.exit(1);
        }
    }

    if (image == null) {
        try printUsageWithError("Error: image is required");
        std.process.exit(1);
    }

    return TimbalRunArgs{
        .quiet = quiet,
        .verbose = verbose,
        .detach = detach,
        .image = image.?,
        .command = command,
        .publish = publish,
    };
}


fn runContainerCmd(
    allocator: std.mem.Allocator, 
    quiet: bool,
    detach: bool,
    image: []const u8,
    publish: ?[]const u8,
    command: ?[]const u8,
) !void {
    var docker_run_args = std.ArrayList([]const u8).init(allocator);
    defer docker_run_args.deinit();

    try docker_run_args.appendSlice(&[_][]const u8{ "docker", "run", });

    if (quiet) {
        try docker_run_args.append("-q");
    }

    if (publish != null) {
        try docker_run_args.append("-p");
        try docker_run_args.append(publish.?);
    }

    if (detach) {
        try docker_run_args.append("-d");
    }

    // TODO Think if we want to specify a --name for easier identification.

    try docker_run_args.append(image);

    if (command != null) {
        // docker run sends the command directly to the container, not as a string.
        // So we need to split the command string into individual arguments.
        var cmd_iter = std.mem.tokenizeScalar(u8, command.?, ' ');
        while (cmd_iter.next()) |arg| {
            try docker_run_args.append(arg);
        }
    }

    // Don't use std.process.Child.run here. We want to inherit the stderr and stdout
    // to stream the output to the main console.
    var child = std.process.Child.init(docker_run_args.items, allocator);
    child.stderr_behavior = .Inherit;
    child.stdout_behavior = .Inherit;

    try child.spawn();
    _ = try child.wait();
}


pub fn run(allocator: std.mem.Allocator, args: []const []const u8) !void {
    const parsed_args = try parseArgs(args);

    if (!parsed_args.quiet) {
        std.debug.print("Running {s}\n", .{parsed_args.image});
        if (parsed_args.command) |cmd| {
            std.debug.print("Command: {s}\n", .{cmd});
        }
        if (parsed_args.publish) |ports| {
            std.debug.print("Published ports: {s}\n", .{ports});
        }
    }

    try runContainerCmd(
        allocator,
        parsed_args.quiet,
        parsed_args.detach,
        parsed_args.image,
        parsed_args.publish,
        parsed_args.command,
    );
}
