const std = @import("std");

const add_cmd = @import("commands/add.zig");
const create_cmd = @import("commands/create.zig");
const init_cmd = @import("commands/init.zig");
const build_cmd = @import("commands/build.zig");
const push_cmd = @import("commands/push.zig");
const configure_cmd = @import("commands/configure.zig");
const credential_helper_cmd = @import("commands/credential_helper.zig");
const upgrade_cmd = @import("commands/upgrade.zig");

// Embedded version.
const timbal_version = @import("version.zig").timbal_version;

fn printUsage() !void {
    const stderr = std.io.getStdErr().writer();
    try stderr.writeAll("Simple, battle-tested framework for building reliable AI applications.\n" ++
        "\n" ++
        "\x1b[1;32mUsage: \x1b[1;36mtimbal \x1b[0;36m[OPTIONS] <COMMAND>\n" ++
        "\n" ++
        "\x1b[1;32mCommands:\n" ++
        "    \x1b[1;36madd       \x1b[0mAdd a component to an existing project\n" ++
        "    \x1b[1;36mconfigure \x1b[0mConfigure Timbal credentials and settings\n" ++
        "    \x1b[1;36mcreate    \x1b[0mCreate a new project with interactive setup\n" ++
        "    \x1b[1;36mupgrade   \x1b[0mUpgrade timbal to the latest version\n" ++
        "    \x1b[1;36mhelp      \x1b[0mDisplay this help message\n" ++
        "\n" ++
        "\x1b[1;32mDeprecated commands:\n" ++
        "    \x1b[0;90minit    Initialize a new application\n" ++
        "    \x1b[0;90mbuild   Build the application into a container\n" ++
        "    \x1b[0;90mpush    Push an application to the Timbal Platform\x1b[0m\n" ++
        "\n" ++
        "\x1b[1;32mGlobal options:\n" ++
        "    \x1b[1;36m-q\x1b[0m, \x1b[1;36m--quiet       \x1b[0mDo not print any output\n" ++
        "    \x1b[1;36m-v\x1b[0m, \x1b[1;36m--verbose     \x1b[0mUse verbose output\n" ++
        "    \x1b[1;36m-h\x1b[0m, \x1b[1;36m--help        \x1b[0mDisplay the concise help for this command\n" ++
        "    \x1b[1;36m-V\x1b[0m, \x1b[1;36m--version     \x1b[0mDisplay the timbal version\n" ++
        "    \x1b[1;36m--profile <NAME>\x1b[0m  Use a named profile (overrides TIMBAL_PROFILE env var)\n" ++
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
    } else if (std.mem.eql(u8, action, "init")) {
        std.debug.print("\x1b[1;33mWarning:\x1b[0m 'timbal init' is deprecated, use 'create' instead.\n\n", .{});
        try init_cmd.run(allocator, args[2..]);
    } else if (std.mem.eql(u8, action, "build")) {
        std.debug.print("\x1b[1;33mWarning:\x1b[0m 'timbal build' is deprecated.\n\n", .{});
        try build_cmd.run(allocator, args[2..]);
    } else if (std.mem.eql(u8, action, "push")) {
        std.debug.print("\x1b[1;33mWarning:\x1b[0m 'timbal push' is deprecated.\n\n", .{});
        try push_cmd.run(allocator, args[2..]);
    } else if (std.mem.eql(u8, action, "configure")) {
        try configure_cmd.run(allocator, args[2..]);
    } else if (std.mem.eql(u8, action, "credential-helper")) {
        try credential_helper_cmd.run(allocator, args[2..]);
    } else if (std.mem.eql(u8, action, "upgrade")) {
        try upgrade_cmd.run(allocator, args[2..]);
    } else if (std.mem.eql(u8, action, "help")) {
        try printUsage();
    } else if (std.mem.eql(u8, action, "-V") or std.mem.eql(u8, action, "--version")) {
        std.debug.print("Timbal {s}\n", .{timbal_version});
        std.process.exit(0);
    } else {
        try printUsageWithError("Error: unknown command");
    }
}
