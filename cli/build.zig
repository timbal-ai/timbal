const std = @import("std");

const targets = [_]std.Target.Query{
    // Linux
    .{ .cpu_arch = .x86_64, .os_tag = .linux, .abi = .gnu },
    .{ .cpu_arch = .aarch64, .os_tag = .linux, .abi = .gnu },
    // MacOS. Unlike Linux, which has multiple common C library implementations and corresponding
    // ABIs, macOS has a single, unified system ABI provided by its standard C library (part of the Darwin/XNU kernel environment).
    .{ .cpu_arch = .x86_64, .os_tag = .macos },
    .{ .cpu_arch = .aarch64, .os_tag = .macos },
    // Windows. With x86 we cover the bast majority of scenarios. Might add aarch64 in the future.
    .{ .cpu_arch = .x86_64, .os_tag = .windows, .abi = .gnu },
};

fn formatExecutableName(
    allocator: std.mem.Allocator,
    name: []const u8,
    version: []const u8,
    target_query: std.Target.Query,
) ![]u8 {
    var name_buf = std.ArrayList(u8).init(allocator);

    try name_buf.appendSlice(name);

    try name_buf.append('-');
    try name_buf.appendSlice(version);

    try name_buf.append('-');
    try name_buf.appendSlice(@tagName(target_query.os_tag orelse return error.Null));

    try name_buf.append('-');
    try name_buf.appendSlice(@tagName(target_query.cpu_arch orelse return error.Null));

    if (target_query.abi) |abi| {
        try name_buf.append('-');
        try name_buf.appendSlice(@tagName(abi));
    }

    return name_buf.toOwnedSlice();
}

pub fn build(b: *std.Build) !void {
    const optimize = b.standardOptimizeOption(.{});

    const version_opt = b.option([]const u8, "version", "The CLI version being built.");
    const version = version_opt orelse "dev";

    // Generate a version file. The CLI commands will import from this file to get the timbal version.
    // const version_file = b.pathJoin(&.{ b.build_root, "version.zig" });
    const version_file = "version.zig";
    const version_content = "pub const timbal_version = \"{s}\";";
    var file = try std.fs.cwd().createFile(version_file, .{});
    defer file.close();
    const content = try std.fmt.allocPrint(b.allocator, version_content, .{version});
    defer b.allocator.free(content);
    try file.writeAll(content);

    // const target = b.standardTargetOptions(.{});
    for (targets) |target_query| {
        const target = b.resolveTargetQuery(target_query);

        const name = try formatExecutableName(b.allocator, "timbal", version, target_query);
        defer b.allocator.free(name);

        const exe = b.addExecutable(.{
            .name = name,
            .root_source_file = b.path("main.zig"),
            .target = target,
            .optimize = optimize,
        });

        // Create an install step that places the artifact in a subdirectory named after the target triple.
        const target_install = b.addInstallArtifact(exe, .{ .dest_dir = .{ .override = .{
            .custom = version,
        } } });

        // Make the main install step depend on this target-specific install step.
        b.getInstallStep().dependOn(&target_install.step);
    }
}
