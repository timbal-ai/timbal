const builtin = @import("builtin");
const std = @import("std");
const fs = std.fs;
const utils = @import("../utils.zig");

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
        "\x1b[1;32mUsage: \x1b[1;36mtimbal upgrade \x1b[0;36m[OPTIONS]\n" ++
        "\n" ++
        "\x1b[1;32mOptions:\n" ++
        "    \x1b[1;36m-f\x1b[0m, \x1b[1;36m--force \x1b[0mForce reinstall even if already on the latest version\n" ++
        "\n" ++
        utils.global_options_help ++
        "\n");
}

fn getOsName() ?[]const u8 {
    return switch (builtin.os.tag) {
        .linux => "linux",
        .macos => "darwin",
        .windows => "windows",
        else => null,
    };
}

fn getArchName() ?[]const u8 {
    return switch (builtin.cpu.arch) {
        .x86_64 => "x86_64",
        .aarch64 => if (builtin.os.tag == .macos) "arm64" else "aarch64",
        else => null,
    };
}

fn upgradeUnix(allocator: std.mem.Allocator, quiet: bool, verbose: bool, force: bool) !void {
    const stderr = std.io.getStdErr().writer();

    // Get the path of the currently running executable.
    const self_exe = std.fs.selfExePathAlloc(allocator) catch {
        try stderr.writeAll("Error: Failed to determine the path of the current executable.\n");
        std.process.exit(1);
    };
    defer allocator.free(self_exe);

    if (verbose) {
        std.debug.print("Current executable: {s}\n", .{self_exe});
    }

    // Download the manifest to get the download URL.
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

    // Check if we're already on the latest version.
    if (!force) {
        const latest_version = parseManifestField(allocator, manifest_buffer.items, "version");
        defer if (latest_version) |v| allocator.free(v);

        if (latest_version) |lv| {
            if (std.mem.eql(u8, timbal_version, lv)) {
                std.debug.print("Already on the latest version ({s}).\n", .{lv});
                return;
            }
            if (!quiet) {
                std.debug.print("New version available: {s}\n", .{lv});
            }
        }
    }

    // Parse the download URL from the manifest.
    const os_name = getOsName() orelse {
        try stderr.writeAll("Error: Unsupported operating system.\n");
        std.process.exit(1);
    };
    const arch_name = getArchName() orelse {
        try stderr.writeAll("Error: Unsupported architecture.\n");
        std.process.exit(1);
    };

    const download_url = parseManifestUrl(allocator, manifest_buffer.items, os_name, arch_name) orelse {
        try stderr.print("Error: No download URL found in manifest for {s} {s}.\n", .{ os_name, arch_name });
        std.process.exit(1);
    };
    defer allocator.free(download_url);

    if (verbose) {
        std.debug.print("Download URL: {s}\n", .{download_url});
    }

    // Download the new binary.
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

    // Sanity check: a valid binary should be at least 100KB.
    if (binary_buffer.items.len < 100 * 1024) {
        try stderr.print("Error: Downloaded file is too small ({d} bytes). The download may have failed.\n", .{binary_buffer.items.len});
        std.process.exit(1);
    }

    // On Unix we can simply overwrite the running binary in place.
    if (!quiet) {
        std.debug.print("Replacing binary...\n", .{});
    }

    // Remove and recreate to handle potential permission/inode issues cleanly.
    std.fs.deleteFileAbsolute(self_exe) catch |err| {
        try stderr.print("Error: Failed to remove current executable: {}\n", .{err});
        std.process.exit(1);
    };

    const new_file = std.fs.createFileAbsolute(self_exe, .{}) catch |err| {
        try stderr.print("Error: Failed to create new executable: {}\n", .{err});
        std.process.exit(1);
    };
    defer new_file.close();

    new_file.writeAll(binary_buffer.items) catch |err| {
        try stderr.print("Error: Failed to write new executable: {}\n", .{err});
        std.process.exit(1);
    };

    // Set executable permission.
    std.posix.fchmod(new_file.handle, 0o755) catch |err| {
        try stderr.print("Warning: Failed to set executable permission: {}\n", .{err});
    };

    if (!quiet) {
        std.debug.print("Upgrade completed successfully.\n", .{});
    }
}

fn upgradeWindows(allocator: std.mem.Allocator, quiet: bool, verbose: bool, force: bool) !void {
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

    const exe_basename = std.fs.path.basename(self_exe);
    const old_exe_path = try std.fmt.allocPrint(allocator, "{s}\\{s}.old", .{ install_dir, exe_basename });
    defer allocator.free(old_exe_path);

    // Write the new binary to the same path as the current executable.
    const new_exe_path = self_exe;

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

    // Check if we're already on the latest version.
    if (!force) {
        const latest_version = parseManifestField(allocator, manifest_buffer.items, "version");
        defer if (latest_version) |v| allocator.free(v);

        if (latest_version) |lv| {
            if (std.mem.eql(u8, timbal_version, lv)) {
                std.debug.print("Already on the latest version ({s}).\n", .{lv});
                return;
            }
            if (!quiet) {
                std.debug.print("New version available: {s}\n", .{lv});
            }
        }
    }

    // Parse the download URL from the manifest
    const os_name = getOsName() orelse {
        try stderr.writeAll("Error: Unsupported operating system.\n");
        std.process.exit(1);
    };
    const arch_name = getArchName() orelse {
        try stderr.writeAll("Error: Unsupported architecture.\n");
        std.process.exit(1);
    };

    const download_url = parseManifestUrl(allocator, manifest_buffer.items, os_name, arch_name) orelse {
        try stderr.print("Error: No download URL found in manifest for {s} {s}.\n", .{ os_name, arch_name });
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

    // Sanity check: a valid binary should be at least 100KB.
    if (binary_buffer.items.len < 100 * 1024) {
        try stderr.print("Error: Downloaded file is too small ({d} bytes). The download may have failed.\n", .{binary_buffer.items.len});
        std.process.exit(1);
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

/// Parse a field from the top-level manifest JSON object.
/// Caller owns the returned memory.
fn parseManifestField(allocator: std.mem.Allocator, manifest: []const u8, field: []const u8) ?[]const u8 {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, manifest, .{}) catch return null;
    defer parsed.deinit();

    const value = parsed.value.object.get(field) orelse return null;
    return switch (value) {
        .string => |s| allocator.dupe(u8, s) catch return null,
        else => null,
    };
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
    var force: bool = false;

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
        } else if (std.mem.eql(u8, arg, "-f") or std.mem.eql(u8, arg, "--force")) {
            force = true;
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
            try upgradeWindows(allocator, quiet, verbose, force);
        },
        .linux, .macos => {
            try upgradeUnix(allocator, quiet, verbose, force);
        },
        else => {
            const stderr = std.io.getStdErr().writer();
            try stderr.writeAll("Error: Unsupported operating system for automatic upgrade.\n");
            try stderr.writeAll("Please visit https://github.com/timbal-ai/timbal for manual installation instructions.\n");
            std.process.exit(1);
        },
    }
}
