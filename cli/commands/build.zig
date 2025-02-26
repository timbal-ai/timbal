const std = @import("std");
const fs = std.fs;


fn printUsageWithError(err: []const u8) !void {
    const stderr = std.io.getStdErr().writer();
    try stderr.print("{s}\n\n", .{err});
    try printUsage();
}


fn printUsage() !void {
    const stderr = std.io.getStdErr().writer();
    try stderr.writeAll(
        "Build a container for the application ready for deployment.\n" ++
        "\n" ++
        "\x1b[1;32mUsage: \x1b[1;36mtimbal build \x1b[0;36m[OPTIONS] [PATH]\n" ++
        "\n" ++
        "\x1b[1;32mArguments:\n" ++
        "    \x1b[1;36mPATH\x1b[0m The path to the project to build\n" ++
        "\n" ++
        "\x1b[1;32mOptions:\n" ++
        "    \x1b[1;36m-t\x1b[0m, \x1b[1;36m--tag \x1b[0mThe tag to use for the container\n" ++
        "\n" ++
        "\x1b[1;32mGlobal options:\n" ++
        "    \x1b[1;36m-q\x1b[0m, \x1b[1;36m--quiet      \x1b[0mDo not print any output\n" ++
        "    \x1b[1;36m-v\x1b[0m, \x1b[1;36m--verbose\x1b[0;36m... \x1b[0mUse verbose output\n" ++
        "    \x1b[1;36m-h\x1b[0m, \x1b[1;36m--help       \x1b[0mDisplay the concise help for this command\n" ++
        "    \x1b[1;36m-V\x1b[0m, \x1b[1;36m--version    \x1b[0mDisplay the timbal version\n" ++
        "\n"
    );
}


const TimbalYaml = struct {
    system_packages: std.ArrayList([]const u8),
    flow: ?[]const u8 = null,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) TimbalYaml {
        return TimbalYaml{
            .system_packages = std.ArrayList([]const u8).init(allocator),
            .flow = null,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *TimbalYaml) void {
        for (self.system_packages.items) |pkg| {
            self.allocator.free(pkg);
        }
        self.system_packages.deinit();
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

    var timbal_yaml = TimbalYaml.init(allocator);

    // Track the current section of the file we're parsing.
    var current_section: enum {
        None,
        SystemPackages,
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
            try timbal_yaml.system_packages.append(try allocator.dupe(u8, system_package));
            continue;
        }

        if (std.mem.startsWith(u8, trimmed, "system_packages:")) {
            current_section = .SystemPackages;
        } else if (std.mem.startsWith(u8, trimmed, "flow:")) {
            current_section = .None;
            var flow = std.mem.trim(u8, trimmed[5..], &std.ascii.whitespace);
            flow = std.mem.trim(u8, flow, "\"");
            timbal_yaml.flow = flow;
        } else {
            current_section = .None;
        }
    }

    // We'd need to reset the file pointer if we want to use the file again.
    // try timbal_yaml_file.seekTo(0);

    return timbal_yaml;
}


pub fn run(allocator: std.mem.Allocator, args: []const []const u8) !void {
    var arg_path: ?[]const u8 = null;
    var tag: ?[]const u8 = null;
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
        } else if (std.mem.eql(u8, arg, "-t") or std.mem.eql(u8, arg, "--tag")) {
            if (tag != null) {
                try printUsageWithError("Error: multiple tags are not supported");
                return;
            }
            if (i + 1 >= args.len) {
                try printUsageWithError("Error: --tag requires a value");
                return;
            }
            tag = args[i + 1];
            i += 1;
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

    if (!quiet) {
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
    defer timbal_yaml.deinit();

    // Create the Dockerfile
    // Right now we create and edit the file in the cwd. We don't create a temporary directory,
    // so that we're able to debug the Dockerfile during development.
    const dockerfile = try app_dir.createFile("Dockerfile", .{});
    defer dockerfile.close();

    // TODO Use the timbal yaml properties to modify the Dockerfile contents.
    try dockerfile.writeAll(
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
        \\        curl \
        \\        ca-certificates \
        \\        git && \
        \\    apt clean && \
        \\    rm -rf /var/lib/apt/lists/*
        \\
        \\RUN curl -LsSf https://astral.sh/uv/install.sh | sh
        \\
        \\ENV PATH="$HOME/.local/bin:$PATH"
        \\
        \\COPY . .
        \\
        \\RUN uv sync --python 3.12 --python-preference managed
        \\
        \\CMD ["tail", "-f", "/dev/null"]
        \\
    );

    // TODO Run docker build
}
