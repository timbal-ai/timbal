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
        "Push a container image to an existing Timbal Platform app.\n" ++
        "You must first create an app on the platform if you don't have one yet.\n" ++
        "\n" ++
        "\x1b[1;32mUsage: \x1b[1;36mtimbal push \x1b[0;36m[OPTIONS] IMAGE APP\n" ++
        "\n" ++
        "\x1b[1;32mArguments:\n" ++
        "    \x1b[1;36mIMAGE\x1b[0m The tag of the container to push\n" ++
        "    \x1b[1;36mAPP\x1b[0m   The URI of the app you want to push the image to (as provided in the platform)\n" ++
        "\n" ++
        "\x1b[1;32mOptions:\n" ++
        "    \x1b[1;36m--name \x1b[0m The name of the app version that will be created\n" ++
        "\n" ++
        "\x1b[1;32mGlobal options:\n" ++
        "    \x1b[1;36m-q\x1b[0m, \x1b[1;36m--quiet      \x1b[0mDo not print any output\n" ++
        "    \x1b[1;36m-v\x1b[0m, \x1b[1;36m--verbose\x1b[0;36m... \x1b[0mUse verbose output\n" ++
        "    \x1b[1;36m-h\x1b[0m, \x1b[1;36m--help       \x1b[0mDisplay the concise help for this command\n" ++
        "    \x1b[1;36m-V\x1b[0m, \x1b[1;36m--version    \x1b[0mDisplay the timbal version\n" ++
        "\n"
    );
}


const AppParts = struct {
    domain: []const u8,
    org_id: usize,
    app_id: usize,
};


fn validateApp(app: []const u8) !AppParts {
    var app_parts = std.mem.split(u8, app, "/");

    const domain = app_parts.next().?;

    const orgs_part = app_parts.next() orelse return error.InvalidAppURI;
    if (!std.mem.eql(u8, orgs_part, "orgs")) {
        return error.InvalidAppURI;
    }

    const org_id_part = app_parts.next() orelse return error.InvalidAppURI;
    const org_id = try std.fmt.parseInt(usize, org_id_part, 10);

    const apps_part = app_parts.next() orelse return error.InvalidAppURI;
    if (!std.mem.eql(u8, apps_part, "apps")) {
        return error.InvalidAppURI;
    }

    const app_id_part = app_parts.next() orelse return error.InvalidAppURI;
    const app_id = try std.fmt.parseInt(usize, app_id_part, 10);

    return AppParts{
        .domain = domain,
        .org_id = org_id,
        .app_id = app_id,
    };
}


const TimbalPushArgs = struct {
    quiet: bool,
    verbose: bool,
    image: []const u8,
    app_parts: AppParts,
    version_name: ?[]const u8,
};


fn parseArgs(args: []const []const u8) !TimbalPushArgs {
    var quiet: bool = false;
    var verbose: bool = false;
    var image: ?[]const u8 = null;
    var app: ?[]const u8 = null;
    var version_name: ?[]const u8 = null;

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
        } else if (std.mem.eql(u8, arg, "--name")) {
            if (version_name != null) {
                try printUsageWithError("Error: multiple version names provided");
                std.process.exit(1);
            }
            version_name = args[i + 1];
            i += 1;
        } else if (!std.mem.startsWith(u8, arg, "-")) {
            if (image == null) {
                image = arg;
            } else if (app == null) {
                app = arg;
            } else {
                try printUsageWithError("Error: too many arguments provided");
                std.process.exit(1);
            }
        } else {
            try printUsageWithError("Error: unknown option");
            std.process.exit(1);
        }
    }

    if (image == null) {
        try printUsageWithError("Error: image not provided");
        std.process.exit(1);
    }

    if (app == null) {
        try printUsageWithError("Error: app not provided");
        std.process.exit(1);
    }

    const app_parts = validateApp(app.?) catch {
        try printUsageWithError("Error: Invalid app URI provided");
        std.process.exit(1);
    };

    return TimbalPushArgs{
        .quiet = quiet,
        .verbose = verbose,
        .image = image.?,
        .app_parts = app_parts,
        .version_name = version_name,
    };
}


const ImageDigestInfo = struct {
    hash: []const u8,
    size: usize,
};


fn inspectImage(
    allocator: std.mem.Allocator,
    image: []const u8,
    quiet: bool,
    verbose: bool,
) !ImageDigestInfo {
    if (!quiet) {
        std.debug.print("Inspecting image: {s}\n", .{image});
    }

    // Force docker to refresh image metadata cache.
    const docker_ls_args = [_][]const u8{
        "docker", "image", "ls", image,
    };
    const ls_result = try std.process.Child.run(.{
        .allocator = allocator,
        .argv = &docker_ls_args,
    });
    allocator.free(ls_result.stdout);
    allocator.free(ls_result.stderr);

    const docker_inspect_args = [_][]const u8{
        "docker", "image", "inspect",
        "--format={{.Id}}\t{{.Size}}",
        image,
    };

    const result = try std.process.Child.run(.{
        .allocator = allocator,
        .argv = &docker_inspect_args,
    });
    defer allocator.free(result.stderr);
    defer allocator.free(result.stdout);

    if (result.term.Exited != 0) {
        const stderr = std.io.getStdErr().writer();
        try stderr.print("Error: {s}\n", .{result.stderr});
        std.process.exit(1);
    }

    var parts = std.mem.split(u8, std.mem.trim(u8, result.stdout, "\n\r"), "\t");
    const hash = parts.next().?;
    const size_str = parts.next().?;
    const size = try std.fmt.parseInt(usize, size_str, 10);

    if (verbose) {
        std.debug.print("Image hash: {s}\n", .{hash});
        std.debug.print("Image size: {d}\n", .{size});
    }

    return ImageDigestInfo{
        .hash = try allocator.dupe(u8, hash),
        .size = size,
    };
}


const DockerLoginInfo = struct {
    user: []const u8,
    password: []const u8,
    server: []const u8,

    pub fn deinit(self: *DockerLoginInfo, allocator: std.mem.Allocator) void {
        allocator.free(self.user);
        allocator.free(self.password);
        allocator.free(self.server);
    }
};


const AuthRes = struct {
    docker_login: DockerLoginInfo,
    uri: []const u8,

    pub fn deinit(self: *AuthRes, allocator: std.mem.Allocator) void {
        self.docker_login.deinit(allocator);
        allocator.free(self.uri);
    }
};


fn authenticate(
    allocator: std.mem.Allocator, 
    timbal_api_token: []const u8,
    image_digest_info: ImageDigestInfo,
    app_parts: AppParts,
    version_name: ?[]const u8,
    quiet: bool,
    verbose: bool,
) !AuthRes {
    if (!quiet) {
        std.debug.print("Authenticating with Timbal Platform...\n", .{});
    }

    const auth_url = try std.fmt.allocPrint(
        allocator,
        "https://{s}/orgs/{d}/apps/{d}/versions",
        .{ app_parts.domain, app_parts.org_id, app_parts.app_id }
    );
    defer allocator.free(auth_url);

    var client = std.http.Client{ .allocator = allocator };
    defer client.deinit();

    const auth_header = try std.fmt.allocPrint(allocator, "Bearer {s}", .{timbal_api_token});
    defer allocator.free(auth_header);

    var res_buffer = std.ArrayList(u8).init(allocator);
    defer res_buffer.deinit();

    const payload = try std.json.stringifyAlloc(allocator, .{
        .hash = image_digest_info.hash,
        .size = image_digest_info.size,
        .name = version_name,
    }, .{});
    defer allocator.free(payload);

    const res = try client.fetch(.{
        .location = .{ .url = auth_url },
        .method = .POST,
        .headers = .{
            .authorization = .{ .override = auth_header },
            .content_type = .{ .override = "application/json" },
        },
        .payload = payload,
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

    if (verbose) {
        std.debug.print("Response: {s}\n", .{res_buffer.items});
    }

    var parsed = try std.json.parseFromSlice(
        AuthRes,
        allocator,
        res_buffer.items,
        .{ .ignore_unknown_fields = true },
    );
    defer parsed.deinit();

    const auth_res = AuthRes{
        .docker_login = DockerLoginInfo{
            .user = try allocator.dupe(u8, parsed.value.docker_login.user),
            .password = try allocator.dupe(u8, parsed.value.docker_login.password),
            .server = try allocator.dupe(u8, parsed.value.docker_login.server),
        },
        .uri = try allocator.dupe(u8, parsed.value.uri),
    };

    return auth_res;
}


fn loginToCR(
    allocator: std.mem.Allocator,
    docker_login: DockerLoginInfo,
    quiet: bool,
    verbose: bool,
) !void {
    _ = verbose;

    if (!quiet) {
        std.debug.print("Logging in to container registry...\n", .{});
    }

    const docker_login_args = [_][]const u8{
        "docker", "login",
        "-u", docker_login.user,
        "-p", docker_login.password,
        docker_login.server,
    };

    const result = try std.process.Child.run(.{
        .allocator = allocator,
        .argv = &docker_login_args,
    });
    defer allocator.free(result.stderr);
    defer allocator.free(result.stdout);

    if (result.term.Exited != 0) {
        const stderr = std.io.getStdErr().writer();
        try stderr.print("Error: {s}\n", .{result.stderr});
        std.process.exit(1);
    }
}


fn tagImage(
    allocator: std.mem.Allocator,
    image: []const u8,
    tag: []const u8,
    quiet: bool,
    verbose: bool,
) !void {
    _ = verbose;

    if (!quiet) {
        std.debug.print("Tagging image...\n", .{});
    }

    const docker_tag_args = [_][]const u8{
        "docker", "tag", image, tag,
    };

    const result = try std.process.Child.run(.{
        .allocator = allocator,
        .argv = &docker_tag_args,
    });
    defer allocator.free(result.stderr);
    defer allocator.free(result.stdout);

    if (result.term.Exited != 0) {
        const stderr = std.io.getStdErr().writer();
        try stderr.print("Error: {s}\n", .{result.stderr});
        std.process.exit(1);
    }
}


fn untagImage(
    allocator: std.mem.Allocator,
    tag: []const u8,
    quiet: bool,
    verbose: bool,
) !void {
    _ = verbose;

    if (!quiet) {
        std.debug.print("Cleaning up tags...\n", .{});
    }

    const docker_untag_args = [_][]const u8{
        "docker", "rmi", tag,
    };

    const result = try std.process.Child.run(.{
        .allocator = allocator,
        .argv = &docker_untag_args,
    });
    defer allocator.free(result.stderr);
    defer allocator.free(result.stdout);

    if (result.term.Exited != 0) {
        const stderr = std.io.getStdErr().writer();
        try stderr.print("Error: {s}\n", .{result.stderr});
        std.process.exit(1);
    }
}


fn pushImage(
    allocator: std.mem.Allocator,
    tag: []const u8,
    quiet: bool,
    verbose: bool,
) !void {
    _ = verbose;

    if (!quiet) {
        std.debug.print("Pushing image...\n", .{});
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

    const parsed_args = try parseArgs(args);

    // Ensure the image exists and fetch some specs, like the hash and size.
    const image_digest_info = try inspectImage(
        allocator, 
        parsed_args.image, 
        parsed_args.quiet, 
        parsed_args.verbose
    );
    defer allocator.free(image_digest_info.hash);

    // Authenticate with the Timbal Platform.
    // Retrieve login user and pwd for the container registry.
    var auth_res = try authenticate(
        allocator, 
        timbal_api_token,
        image_digest_info,
        parsed_args.app_parts,
        parsed_args.version_name,
        parsed_args.quiet,
        parsed_args.verbose,
    );
    defer auth_res.deinit(allocator);

    try loginToCR(
        allocator, 
        auth_res.docker_login, 
        parsed_args.quiet,
        parsed_args.verbose,
    );

    try tagImage(
        allocator, 
        parsed_args.image, 
        auth_res.uri,
        parsed_args.quiet,
        parsed_args.verbose,
    );

    try pushImage(
        allocator, 
        auth_res.uri, 
        parsed_args.quiet,
        parsed_args.verbose,
    );

    try untagImage(
        allocator,
        auth_res.uri,
        parsed_args.quiet,
        parsed_args.verbose,
    );
}
