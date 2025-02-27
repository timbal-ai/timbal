const std = @import("std");
const fs = std.fs;


fn printUsageWithError(err: []const u8) !void {
    const stderr = std.io.getStdErr().writer();
    try stderr.print("{s}\n\n", .{err});
    try printUsage();
}


// TODO Add --app
// TODO Add --version
fn printUsage() !void {
    const stderr = std.io.getStdErr().writer();
    try stderr.writeAll(
        "Push an application to the Timbal Platform.\n" ++
        "\n" ++
        "\x1b[1;32mUsage: \x1b[1;36mtimbal push \x1b[0;36m[OPTIONS] IMAGE\n" ++
        "\n" ++
        "\x1b[1;32mArguments:\n" ++
        "    \x1b[1;36mIMAGE\x1b[0m   The tag of the container to push\n" ++
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


const TimbalPushArgs = struct {
    quiet: bool,
    verbose: bool,
    image: []const u8,
};


fn parseArgs(args: []const []const u8) !TimbalPushArgs {
    var quiet: bool = false;
    var verbose: bool = false;
    var image: ?[]const u8 = null;

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            try printUsage();
            std.process.exit(0);
        } else if (std.mem.eql(u8, arg, "-V") or std.mem.eql(u8, arg, "--version")) {
            // TODO Real versioning
            std.debug.print("timbal version 0.1.0\n", .{});
            std.process.exit(0);
        } else if (std.mem.eql(u8, arg, "-v") or std.mem.eql(u8, arg, "--verbose")) {
            verbose = true;
        } else if (std.mem.eql(u8, arg, "-q") or std.mem.eql(u8, arg, "--quiet")) {
            quiet = true;
        } else if (!std.mem.startsWith(u8, arg, "-")) {
            if (image != null) {
                try printUsageWithError("Error: multiple images provided");
                std.process.exit(1);
            }
            image = arg;
        } else {
            try printUsageWithError("Error: unknown option");
            std.process.exit(1);
        }
    }

    if (image == null) {
        try printUsageWithError("Error: image not provided");
        std.process.exit(1);
    }

    return TimbalPushArgs{
        .quiet = quiet,
        .verbose = verbose,
        .image = image.?,
    };
}


const DockerLogin = struct {
    user: []const u8,
    password: []const u8,
    server: []const u8,
};


fn getDockerLogin(
    allocator: std.mem.Allocator, 
    timbal_api_token: []const u8,
    verbose: bool,
) !DockerLogin {
    // TODO Change URL.
    const auth_url = "https://dev.timbal.ai/orgs/1/apps/1/versions";

    if (verbose) {
        std.debug.print("Authenticating with Timbal Platform...\n", .{});
    }

    var client = std.http.Client{ .allocator = allocator };
    defer client.deinit();

    const auth_header = try std.fmt.allocPrint(allocator, "Bearer {s}", .{timbal_api_token});
    defer allocator.free(auth_header);

    var res_buffer = std.ArrayList(u8).init(allocator);
    defer res_buffer.deinit();

    const res = try client.fetch(.{
        .location = .{ .url = auth_url },
        .method = .POST,
        .headers = .{
            .authorization = .{ .override = auth_header },
            .content_type = .{ .override = "application/json" },
        },
        .response_storage = .{ .dynamic = &res_buffer },
    });

    if (res.status != .ok) {
        const stderr = std.io.getStdErr().writer();
        try stderr.print("Error: API request failed with status code {d}\n", .{@intFromEnum(res.status)});
        
        if (res_buffer.items.len > 0) {
            try stderr.print("Response: {s}\n", .{res_buffer.items});
        }
        
        std.process.exit(1);
    }

    var parsed = try std.json.parseFromSlice(
        DockerLogin,
        allocator,
        res_buffer.items,
        .{ .ignore_unknown_fields = true },
    );
    defer parsed.deinit();

    return DockerLogin{
        .user = try allocator.dupe(u8, parsed.value.user),
        .password = try allocator.dupe(u8, parsed.value.password),
        .server = try allocator.dupe(u8, parsed.value.server),
    };
}


fn loginToDockerRegistry(
    allocator: std.mem.Allocator,
    docker_login: DockerLogin,
    quiet: bool,
) !void {
    if (!quiet) {
        std.debug.print("Logging in to Docker server...\n", .{});
    }

    const docker_login_args = [_][]const u8{
        "docker", "login",
        "-u", docker_login.user,
        "-p", docker_login.password,
        docker_login.server,
    };

    // Don't use std.process.Child.run here. We want to inherit the stderr and stdout
    // to stream the output to the main console.
    var child = std.process.Child.init(&docker_login_args, allocator);
    child.stderr_behavior = .Ignore; // Supress -p vs --password-stdin warning.
    child.stdout_behavior = .Inherit;

    try child.spawn();
    _ = try child.wait();
}


fn tagImage(
    allocator: std.mem.Allocator,
    server: []const u8,
    image: []const u8,
    quiet: bool,
) ![]const u8 {
    if (!quiet) {
        std.debug.print("Tagging image...\n", .{});
    }

    // TODO Change this. The repo name must be something of the actual Timbal Platform app.
    const target_image = try std.fmt.allocPrint(
        allocator,
        "{s}/timbal/{s}",
        .{ server, image },
    );

    const docker_tag_args = [_][]const u8{
        "docker", "tag", image, target_image
    };

    // Don't use std.process.Child.run here. We want to inherit the stderr and stdout
    // to stream the output to the main console.
    var child = std.process.Child.init(&docker_tag_args, allocator);
    child.stderr_behavior = .Inherit;
    child.stdout_behavior = .Inherit;

    try child.spawn();
    _ = try child.wait();

    return target_image;
}


fn pushImage(
    allocator: std.mem.Allocator,
    tag: []const u8,
    quiet: bool,
) !void {
    if (!quiet) {
        std.debug.print("Pushing image: {s}\n", .{tag});
    }

    const docker_push_args = [_][]const u8{
        "docker", "push", tag,
    };

    // Don't use std.process.Child.run here. We want to inherit the stderr and stdout
    // to stream the output to the main console.
    var child = std.process.Child.init(&docker_push_args, allocator);
    child.stderr_behavior = .Inherit;
    child.stdout_behavior = .Inherit;

    try child.spawn();
    _ = try child.wait();
}


pub fn run(allocator: std.mem.Allocator, args: []const []const u8) !void {
    const parsed_args = try parseArgs(args);

    if (!parsed_args.quiet) {
        std.debug.print("Pushing image: {s}\n", .{parsed_args.image});
    }

    // Ensure TIMBAL_API_TOKEN is set.
    const timbal_api_token = std.process.getEnvVarOwned(allocator, "TIMBAL_API_TOKEN") catch |err| {
        if (err == error.EnvironmentVariableNotFound) {
            const stderr = std.io.getStdErr().writer();
            try stderr.writeAll("Error: TIMBAL_API_TOKEN is not set\n");
            try stderr.writeAll("Please set the TIMBAL_API_TOKEN environment variable before running this command.\n");
            std.process.exit(1);
        } else {
            return err;
        }
    };
    defer allocator.free(timbal_api_token);

    // Retrieve login user and pwd for aws ecr registry (this authorizes access to the app too).
    const docker_login = try getDockerLogin(
        allocator, 
        timbal_api_token,
        parsed_args.verbose,
    );
    defer {
        allocator.free(docker_login.user);
        allocator.free(docker_login.password);
        allocator.free(docker_login.server);
    }

    try loginToDockerRegistry(allocator, docker_login, parsed_args.quiet);
    
    const registry_tag = try tagImage(
        allocator, 
        docker_login.server,
        parsed_args.image, 
        parsed_args.quiet,
    );
    defer allocator.free(registry_tag);

    try pushImage(allocator, registry_tag, parsed_args.quiet);
}
