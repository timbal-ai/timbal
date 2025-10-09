const builtin = @import("builtin");
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
    try stderr.writeAll("Upgrade timbal to the latest version.\n" ++
        "\n" ++
        "\x1b[1;32mUsage: \x1b[1;36mtimbal upgrade\n" ++
        "\n" ++
        "\x1b[1;32mGlobal options:\n" ++
        "    \x1b[1;36m-q\x1b[0m, \x1b[1;36m--quiet      \x1b[0mDo not print any output\n" ++
        "    \x1b[1;36m-v\x1b[0m, \x1b[1;36m--verbose\x1b[0;36m... \x1b[0mUse verbose output\n" ++
        "    \x1b[1;36m-h\x1b[0m, \x1b[1;36m--help       \x1b[0mDisplay the concise help for this command\n" ++
        "    \x1b[1;36m-V\x1b[0m, \x1b[1;36m--version    \x1b[0mDisplay the timbal version\n" ++
        "\n");
}

fn downloadInstallScript(allocator: std.mem.Allocator, url: []const u8, quiet: bool, verbose: bool) ![]const u8 {
    if (!quiet) {
        std.debug.print("Downloading install script from {s}...\n", .{url});
    }

    var client = std.http.Client{ .allocator = allocator };
    defer client.deinit();

    var script_buffer = std.ArrayList(u8).init(allocator);
    defer script_buffer.deinit();

    const res = client.fetch(.{
        .location = .{ .url = url },
        .method = .GET,
        .response_storage = .{ .dynamic = &script_buffer },
    }) catch |err| {
        const stderr = std.io.getStdErr().writer();
        try stderr.print("Error: Failed to download install script: {}\n", .{err});
        std.process.exit(1);
    };

    if (res.status != .ok) {
        const stderr = std.io.getStdErr().writer();
        try stderr.print("Error: Failed to download install script (HTTP {d})\n", .{@intFromEnum(res.status)});
        std.process.exit(1);
    }

    if (verbose) {
        std.debug.print("Downloaded {d} bytes\n", .{script_buffer.items.len});
    }

    return script_buffer.toOwnedSlice();
}

fn upgradeUnix(allocator: std.mem.Allocator, quiet: bool, verbose: bool) !void {
    const script_url = "https://raw.githubusercontent.com/timbal-ai/timbal/main/cli/install.sh";

    // Download the install script
    const script_content = try downloadInstallScript(allocator, script_url, quiet, verbose);
    defer allocator.free(script_content);

    if (!quiet) {
        std.debug.print("Launching install script...\n", .{});
        std.debug.print("The upgrade will complete after this process exits.\n", .{});
    }

    // Execute via: sh -c 'script content' sh -y
    // This passes -y as $0 argument
    const argv = [_][]const u8{ "sh", "-c", script_content, "sh", "-y" };

    var child = std.process.Child.init(&argv, allocator);
    child.stdin_behavior = .Inherit;
    child.stdout_behavior = .Inherit;
    child.stderr_behavior = .Inherit;

    try child.spawn();

    const term = try child.wait();

    if (term.Exited != 0) {
        std.debug.print("Upgrade script exited with code: {d}\n", .{term.Exited});
    } else {
        std.debug.print("Upgrade completed successfully\n", .{});
    }
}

fn upgradeWindows(allocator: std.mem.Allocator, quiet: bool, verbose: bool) !void {
    const script_url = "https://raw.githubusercontent.com/timbal-ai/timbal/main/cli/install.ps1";

    // Download the install script
    const script_content = try downloadInstallScript(allocator, script_url, quiet, verbose);
    defer allocator.free(script_content);

    if (!quiet) {
        std.debug.print("Launching install script...\n", .{});
        std.debug.print("The upgrade will complete after this process exits.\n", .{});
    }

    // Write script to a temporary file since PowerShell -Command doesn't handle full scripts well
    const temp_dir = std.process.getEnvVarOwned(allocator, "TEMP") catch "C:\\Windows\\Temp";
    defer allocator.free(temp_dir);

    const script_path = try std.fmt.allocPrint(allocator, "{s}\\timbal-upgrade.ps1", .{temp_dir});
    defer allocator.free(script_path);

    // Write and close the file before executing
    {
        const script_file = try std.fs.createFileAbsolute(script_path, .{});
        defer script_file.close();
        try script_file.writeAll(script_content);
    } // File is closed here

    // Execute the script file (no -Yes flag needed, it overwrites by default)
    const argv = [_][]const u8{ "powershell.exe", "-ExecutionPolicy", "ByPass", "-File", script_path };

    var child = std.process.Child.init(&argv, allocator);
    child.stdin_behavior = .Inherit;
    child.stdout_behavior = .Inherit;
    child.stderr_behavior = .Inherit;

    try child.spawn();

    const term = try child.wait();

    // Clean up temp file
    std.fs.deleteFileAbsolute(script_path) catch {};

    if (term.Exited != 0) {
        std.debug.print("Upgrade script exited with code: {d}\n", .{term.Exited});
    } else {
        std.debug.print("Upgrade completed successfully\n", .{});
    }
}

pub fn run(allocator: std.mem.Allocator, args: []const []const u8) !void {
    var verbose: bool = false;
    var quiet: bool = false;

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            try printUsage();
            return;
        } else if (std.mem.eql(u8, arg, "-V") or std.mem.eql(u8, arg, "--version")) {
            std.debug.print("Timbal {s}\n", .{timbal_version});
            return;
        } else if (std.mem.eql(u8, arg, "-v") or std.mem.eql(u8, arg, "--verbose")) {
            verbose = true;
        } else if (std.mem.eql(u8, arg, "-q") or std.mem.eql(u8, arg, "--quiet")) {
            quiet = true;
        } else {
            try printUsageWithError("Error: unknown option");
            return;
        }
    }

    if (!quiet) {
        std.debug.print("Upgrading timbal from version {s}...\n", .{timbal_version});
    }

    // Detect the operating system and run the appropriate installation script
    const os_tag = builtin.os.tag;

    switch (os_tag) {
        .windows => {
            try upgradeWindows(allocator, quiet, verbose);
        },
        .linux, .macos => {
            try upgradeUnix(allocator, quiet, verbose);
        },
        else => {
            const stderr = std.io.getStdErr().writer();
            try stderr.writeAll("Error: Unsupported operating system for automatic upgrade.\n");
            try stderr.writeAll("Please visit https://github.com/timbal-ai/timbal for manual installation instructions.\n");
            std.process.exit(1);
        },
    }
}
