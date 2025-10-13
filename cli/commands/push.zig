const std = @import("std");
const fs = std.fs;

// Embedded version.
const timbal_version = @import("../version.zig").timbal_version;

fn printUsageWithError(err: []const u8) !void {
    const stderr = std.io.getStdErr().writer();
    try stderr.print("{s}\n\n", .{err});
    try printUsage();
}

fn printUsage() !void {
    const stderr = std.io.getStdErr().writer();
    try stderr.writeAll("Push a container image to an existing Timbal Platform app.\n" ++
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
        "\n");
}

const AppParts = struct {
    domain: []const u8,
    org_id: usize,
    app_id: usize,
};

fn validateApp(app: []const u8) !AppParts {
    var app_parts = std.mem.splitScalar(u8, app, '/');

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
            std.debug.print("Timbal {s}\n", .{timbal_version});
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
    id: []const u8,
    hash: []const u8,
    size: usize,
    platform: []const u8,

    pub fn deinit(self: *ImageDigestInfo, allocator: std.mem.Allocator) void {
        allocator.free(self.id);
        allocator.free(self.hash);
        allocator.free(self.platform);
    }
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
        "docker", "image", "ls", "-q", image,
    };
    const ls_result = try std.process.Child.run(.{
        .allocator = allocator,
        .argv = &docker_ls_args,
    });
    defer allocator.free(ls_result.stderr);
    defer allocator.free(ls_result.stdout);

    if (ls_result.term.Exited != 0) {
        const stderr = std.io.getStdErr().writer();
        try stderr.print("Error: {s}\n", .{ls_result.stderr});
        std.process.exit(1);
    }

    const image_id = std.mem.trim(u8, ls_result.stdout, "\n\r");

    const docker_inspect_args = [_][]const u8{
        "docker",
        "image",
        "inspect",
        "--format={{.Id}}\t{{.Size}}\t{{.Os}}/{{.Architecture}}",
        image_id,
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

    var parts = std.mem.splitScalar(u8, std.mem.trim(u8, result.stdout, "\n\r"), '\t');
    const hash = parts.next().?;
    const size_str = parts.next().?;
    const size = try std.fmt.parseInt(usize, size_str, 10);
    const platform = parts.next().?;

    if (verbose) {
        std.debug.print("Image ID: {s}\n", .{image_id});
        std.debug.print("Image hash: {s}\n", .{hash});
        std.debug.print("Image size: {d}\n", .{size});
        std.debug.print("Image platform: {s}\n", .{platform});
    }

    return ImageDigestInfo{
        .id = try allocator.dupe(u8, image_id),
        .hash = try allocator.dupe(u8, hash),
        .size = size,
        .platform = try allocator.dupe(u8, platform),
    };
}

const ProbeResJson = struct {
    params_model_schema: std.json.Value,
    return_model_schema: std.json.Value,
    type: []const u8,
    version: []const u8,

    pub fn deinit(self: *ProbeResJson, allocator: std.mem.Allocator) void {
        self.params_model_schema.deinit(allocator);
        self.return_model_schema.deinit(allocator);
        allocator.free(self.type);
        allocator.free(self.version);
    }
};

const ProbeRes = struct {
    params_model_schema: []const u8,
    return_model_schema: []const u8,
    type: []const u8,
    version: []const u8,

    pub fn deinit(self: *ProbeRes, allocator: std.mem.Allocator) void {
        allocator.free(self.params_model_schema);
        allocator.free(self.return_model_schema);
        allocator.free(self.type);
        allocator.free(self.version);
    }
};

fn probeImage(
    allocator: std.mem.Allocator,
    image_id: []const u8,
    quiet: bool,
    verbose: bool,
) !ProbeRes {
    if (!quiet) {
        std.debug.print("Probing image {s}...\n", .{image_id});
    }

    const docker_run_args = [_][]const u8{
        "docker", "run", "--rm", image_id,
        "uv",     "run", "-m",   "timbal.server.probe",
    };

    const result = try std.process.Child.run(.{
        .allocator = allocator,
        .argv = &docker_run_args,
    });
    defer allocator.free(result.stderr);
    defer allocator.free(result.stdout);

    if (result.term.Exited != 0) {
        // Print both for maximum debugging info
        const stderr = std.io.getStdErr().writer();
        try stderr.print("stdout: {s}\n", .{result.stdout});
        try stderr.print("stderr: {s}\n", .{result.stderr});
        std.process.exit(1);
    }

    if (verbose) {
        std.debug.print("Probe result: {s}\n", .{result.stdout});
    }

    var parsed = try std.json.parseFromSlice(
        ProbeResJson,
        allocator,
        result.stdout,
        .{ .ignore_unknown_fields = true },
    );
    defer parsed.deinit();

    var params_model_schema = std.ArrayList(u8).init(allocator);
    try std.json.stringify(parsed.value.params_model_schema, .{}, params_model_schema.writer());

    var return_model_schema = std.ArrayList(u8).init(allocator);
    try std.json.stringify(parsed.value.return_model_schema, .{}, return_model_schema.writer());

    return ProbeRes{
        .params_model_schema = try params_model_schema.toOwnedSlice(),
        .return_model_schema = try return_model_schema.toOwnedSlice(),
        .type = try allocator.dupe(u8, parsed.value.type),
        .version = try allocator.dupe(u8, parsed.value.version),
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
    timbal_api_key: []const u8,
    image_digest_info: ImageDigestInfo,
    probe_res: ProbeRes,
    app_parts: AppParts,
    version_name: ?[]const u8,
    quiet: bool,
    verbose: bool,
) !AuthRes {
    if (!quiet) {
        std.debug.print("Authenticating with Timbal Platform...\n", .{});
    }

    const auth_url = try std.fmt.allocPrint(allocator, "https://{s}/orgs/{d}/apps/{d}/versions", .{ app_parts.domain, app_parts.org_id, app_parts.app_id });
    defer allocator.free(auth_url);

    var client = std.http.Client{ .allocator = allocator };
    defer client.deinit();

    const auth_header = try std.fmt.allocPrint(allocator, "Bearer {s}", .{timbal_api_key});
    defer allocator.free(auth_header);

    var res_buffer = std.ArrayList(u8).init(allocator);
    defer res_buffer.deinit();

    const payload = try std.json.stringifyAlloc(allocator, .{
        .hash = image_digest_info.hash,
        .size = image_digest_info.size,
        .platform = image_digest_info.platform,
        .name = version_name,
        .version = probe_res.version,
        .type = probe_res.type,
        .params_model_schema = probe_res.params_model_schema,
        .return_model_schema = probe_res.return_model_schema,
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
        "docker",            "login",
        "-u",                docker_login.user,
        "-p",                docker_login.password,
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
    // Ensure TIMBAL_API_KEY or TIMBAL_API_TOKEN is set.
    const timbal_api_key = std.process.getEnvVarOwned(allocator, "TIMBAL_API_KEY") catch |key_err| blk: {
        if (key_err == error.EnvironmentVariableNotFound) {
            // Try fallback to TIMBAL_API_TOKEN
            break :blk std.process.getEnvVarOwned(allocator, "TIMBAL_API_TOKEN") catch |token_err| {
                if (token_err == error.EnvironmentVariableNotFound) {
                    const stderr = std.io.getStdErr().writer();
                    try stderr.writeAll("Error: TIMBAL_API_KEY or TIMBAL_API_TOKEN is not set\n");
                    try stderr.writeAll("Please set the TIMBAL_API_KEY or TIMBAL_API_TOKEN environment variable before running this command.\n");
                    std.process.exit(1);
                } else {
                    return token_err;
                }
            };
        } else {
            return key_err;
        }
    };
    defer allocator.free(timbal_api_key);

    const parsed_args = try parseArgs(args);

    // Ensure the image exists and fetch some specs, like the hash and size.
    var image_digest_info = try inspectImage(allocator, parsed_args.image, parsed_args.quiet, parsed_args.verbose);
    defer image_digest_info.deinit(allocator);

    var probe_res = try probeImage(
        allocator,
        image_digest_info.id,
        parsed_args.quiet,
        parsed_args.verbose,
    );
    defer probe_res.deinit(allocator);

    // Authenticate with the Timbal Platform.
    // Retrieve login user and pwd for the container registry.
    var auth_res = try authenticate(
        allocator,
        timbal_api_key,
        image_digest_info,
        probe_res,
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
