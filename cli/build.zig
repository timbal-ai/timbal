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

pub fn build(b: *std.Build) !void {
    // Get the optimization mode from command line arguments.
    const optimize = b.standardOptimizeOption(.{});

    // const target = b.standardTargetOptions(.{});
    for (targets) |target_query| {
        const target = b.resolveTargetQuery(target_query);

        const exe = b.addExecutable(.{
            .name = "timbal",
            .root_source_file = b.path("main.zig"),
            .target = target,
            .optimize = optimize,
        });

        // Create an install step that places the artifact in a subdirectory named after the target triple.
        const target_install = b.addInstallArtifact(exe, .{ .dest_dir = .{ .override = .{
            .custom = try target_query.zigTriple(b.allocator),
        } } });

        // Make the main install step depend on this target-specific install step.
        b.getInstallStep().dependOn(&target_install.step);
    }
}
