const std = @import("std");

const utils = @import("utils.zig");

const add_cmd = @import("commands/add.zig");
const create_cmd = @import("commands/create.zig");
const configure_cmd = @import("commands/configure.zig");
const credential_helper_cmd = @import("commands/credential_helper.zig");
const start_cmd = @import("commands/start.zig");
const upgrade_cmd = @import("commands/upgrade.zig");

// Embedded version.
const version_info = @import("version.zig");
const timbal_version = version_info.timbal_version;
const timbal_commit_hash = version_info.timbal_commit_hash;
const timbal_commit_date = version_info.timbal_commit_date;

fn printUsage() !void {
    const stderr = std.io.getStdErr().writer();
    try stderr.writeAll("Simple, battle-tested framework for building reliable AI applications.\n" ++
        "\n" ++
        "\x1b[1;32mUsage: \x1b[1;36mtimbal \x1b[0;36m[OPTIONS] <COMMAND>\n" ++
        "\n" ++
        "\x1b[1;32mCommands:\n" ++
        "    \x1b[1;36mconfigure \x1b[0mConfigure Timbal credentials and settings\n" ++
        "    \x1b[1;36mcreate    \x1b[0mCreate a new project (interactive or use --type + --ui)\n" ++
        "    \x1b[1;36madd       \x1b[0mAdd a component to an existing project\n" ++
        "    \x1b[1;36mstart     \x1b[0mStart a project (UI, API, agents, and workflows)\n" ++
        "    \x1b[1;36mupgrade   \x1b[0mUpgrade timbal to the latest version\n" ++
        "    \x1b[1;36mversion   \x1b[0mDisplay the current version\n" ++
        "\n" ++
        utils.global_options_help ++
        "\n");
}

fn printUsageWithError(err: []const u8) !void {
    const stderr = std.io.getStdErr().writer();
    try stderr.print("{s}\n\n", .{err});
    try printUsage();
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        try printUsageWithError("Error: missing command");
        return;
    }

    const action = args[1];

    if (std.mem.eql(u8, action, "add")) {
        try add_cmd.run(allocator, args[2..]);
    } else if (std.mem.eql(u8, action, "create")) {
        try create_cmd.run(allocator, args[2..]);
    } else if (std.mem.eql(u8, action, "start")) {
        try start_cmd.run(allocator, args[2..]);
    } else if (std.mem.eql(u8, action, "configure")) {
        try configure_cmd.run(allocator, args[2..]);
    } else if (std.mem.eql(u8, action, "credential-helper")) {
        try credential_helper_cmd.run(allocator, args[2..]);
    } else if (std.mem.eql(u8, action, "upgrade")) {
        try upgrade_cmd.run(allocator, args[2..]);
    } else if (std.mem.eql(u8, action, "version")) {
        std.debug.print("timbal {s} ({s} {s})\n", .{ timbal_version, timbal_commit_hash, timbal_commit_date });
        std.process.exit(0);
    } else if (std.mem.eql(u8, action, "-h") or std.mem.eql(u8, action, "--help")) {
        try printUsage();
    } else {
        try printUsageWithError("Error: unknown command");
    }
}
