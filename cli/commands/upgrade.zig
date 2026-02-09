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
    const stderr = std.io.getStdErr().writer();

    // Get the path of the currently running executable
    const self_exe = std.fs.selfExePathAlloc(allocator) catch {
        try stderr.writeAll("Error: Failed to determine the path of the current executable.\n");
        std.process.exit(1);
    };
    defer allocator.free(self_exe);

    if (verbose) {
        std.debug.print("Current executable: {s}\n", .{self_exe});
    }

    // Determine install directory from the current executable path
    const install_dir = std.fs.path.dirname(self_exe) orelse {
        try stderr.writeAll("Error: Failed to determine installation directory.\n");
        std.process.exit(1);
    };

    const old_exe_path = try std.fmt.allocPrint(allocator, "{s}\\timbal.old.exe", .{install_dir});
    defer allocator.free(old_exe_path);

    const new_exe_path = try std.fmt.allocPrint(allocator, "{s}\\timbal.exe", .{install_dir});
    defer allocator.free(new_exe_path);

    // Download the manifest to get the download URL
    const manifest_url = "https://github.com/timbal-ai/timbal/releases/latest/download/manifest.json";

    if (!quiet) {
        std.debug.print("Fetching latest version info...\n", .{});
    }

    var client = std.http.Client{ .allocator = allocator };
    defer client.deinit();

    var manifest_buffer = std.ArrayList(u8).init(allocator);
    defer manifest_buffer.deinit();

    const manifest_res = client.fetch(.{
        .location = .{ .url = manifest_url },
        .method = .GET,
        .response_storage = .{ .dynamic = &manifest_buffer },
    }) catch |err| {
        try stderr.print("Error: Failed to download manifest: {}\n", .{err});
        std.process.exit(1);
    };

    if (manifest_res.status != .ok) {
        try stderr.print("Error: Failed to download manifest (HTTP {d})\n", .{@intFromEnum(manifest_res.status)});
        std.process.exit(1);
    }

    // Parse the download URL from the manifest
    const download_url = parseManifestUrl(allocator, manifest_buffer.items, "windows", "x86_64") orelse {
        try stderr.writeAll("Error: No download URL found in manifest for windows x86_64.\n");
        std.process.exit(1);
    };
    defer allocator.free(download_url);

    if (verbose) {
        std.debug.print("Download URL: {s}\n", .{download_url});
    }

    // Download the new binary to a temp file
    if (!quiet) {
        std.debug.print("Downloading new version...\n", .{});
    }

    var binary_buffer = std.ArrayList(u8).init(allocator);
    defer binary_buffer.deinit();

    const bin_res = client.fetch(.{
        .location = .{ .url = download_url },
        .method = .GET,
        .response_storage = .{ .dynamic = &binary_buffer },
        .max_append_size = 50 * 1024 * 1024,
    }) catch |err| {
        try stderr.print("Error: Failed to download new binary: {}\n", .{err});
        std.process.exit(1);
    };

    if (bin_res.status != .ok) {
        try stderr.print("Error: Failed to download new binary (HTTP {d})\n", .{@intFromEnum(bin_res.status)});
        std.process.exit(1);
    }

    if (verbose) {
        std.debug.print("Downloaded {d} bytes\n", .{binary_buffer.items.len});
    }

    // Clean up any leftover .old.exe from a previous upgrade
    std.fs.deleteFileAbsolute(old_exe_path) catch {};

    // Rename the running executable: timbal.exe -> timbal.old.exe
    if (!quiet) {
        std.debug.print("Replacing binary...\n", .{});
    }

    std.fs.renameAbsolute(self_exe, old_exe_path) catch |err| {
        try stderr.print("Error: Failed to rename current executable: {}\n", .{err});
        std.process.exit(1);
    };

    // Write the new binary as timbal.exe
    const new_file = std.fs.createFileAbsolute(new_exe_path, .{}) catch |err| {
        // Try to restore the old binary if writing fails
        std.fs.renameAbsolute(old_exe_path, self_exe) catch {};
        try stderr.print("Error: Failed to create new executable: {}\n", .{err});
        std.process.exit(1);
    };
    defer new_file.close();

    new_file.writeAll(binary_buffer.items) catch |err| {
        // Try to restore the old binary if writing fails
        std.fs.renameAbsolute(old_exe_path, self_exe) catch {};
        try stderr.print("Error: Failed to write new executable: {}\n", .{err});
        std.process.exit(1);
    };

    // Try to clean up the old binary (may fail if still locked, that's ok)
    std.fs.deleteFileAbsolute(old_exe_path) catch {};

    if (!quiet) {
        std.debug.print("Upgrade completed successfully.\n", .{});
    }
}

/// Parse a URL from the manifest JSON for a given OS and architecture.
/// Caller owns the returned memory.
fn parseManifestUrl(allocator: std.mem.Allocator, manifest: []const u8, os: []const u8, arch: []const u8) ?[]const u8 {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, manifest, .{}) catch return null;
    defer parsed.deinit();

    const binaries = parsed.value.object.get("binaries") orelse return null;
    const os_obj = binaries.object.get(os) orelse return null;
    const arch_obj = os_obj.object.get(arch) orelse return null;
    const url_value = arch_obj.object.get("url") orelse return null;

    return switch (url_value) {
        .string => |s| allocator.dupe(u8, s) catch return null,
        else => null,
    };
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
