const std = @import("std");

const init_cmd = @import("commands/init.zig");
const build_cmd = @import("commands/build.zig");
const run_cmd = @import("commands/run.zig");


fn printUsage() !void {
    const stderr = std.io.getStdErr().writer();
    try stderr.writeAll(
        "A strongly opinionated framework for building and orchestrating agentic AI applications.\n" ++
        "\n" ++
        "\x1b[1;32mUsage: \x1b[1;36mtimbal \x1b[0;36m[OPTIONS] <COMMAND>\n" ++
        "\n" ++
        "\x1b[1;32mCommands:\n" ++
        "    \x1b[1;36minit  \x1b[0mInitialize a new application\n" ++
        "    \x1b[1;36mbuild \x1b[0mBuild the application into a container\n" ++
        "    \x1b[1;36mrun   \x1b[0mRun a command inside the built container\n" ++
        "    \x1b[1;36mserve \x1b[0mBuild and Run the application at once\n" ++
        "    \x1b[1;36mlogin \x1b[0mLogin to the Timbal Platform\n" ++
        "    \x1b[1;36mpush  \x1b[0mPush an application to the Timbal Platform\n" ++
        "    \x1b[1;36mhelp  \x1b[0mDisplay this help message\n" ++
        "\n" ++
        "\x1b[1;32mGlobal options:\n" ++
        "    \x1b[1;36m-q\x1b[0m, \x1b[1;36m--quiet      \x1b[0mDo not print any output\n" ++
        "    \x1b[1;36m-v\x1b[0m, \x1b[1;36m--verbose\x1b[0;36m... \x1b[0mUse verbose output\n" ++
        "    \x1b[1;36m-h\x1b[0m, \x1b[1;36m--help       \x1b[0mDisplay the concise help for this command\n" ++
        "    \x1b[1;36m-V\x1b[0m, \x1b[1;36m--version    \x1b[0mDisplay the timbal version\n" ++
        "\n"
    );
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

    if (std.mem.eql(u8, action, "init")) {
        try init_cmd.run(allocator, args[2..]);
    } else if (std.mem.eql(u8, action, "build")) {
        try build_cmd.run(allocator, args[2..]);
    } else if (std.mem.eql(u8, action, "run")) {
        try run_cmd.run(allocator, args[2..]);
    } else if (std.mem.eql(u8, action, "help")) {
        try printUsage();
    } else {
        try printUsageWithError("Error: unknown command");
    }
}
