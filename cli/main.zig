const std = @import("std");
const init_cmd = @import("commands/init.zig");


pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        try printUsage();
        return;
    }

    const action = args[1];
    if (std.mem.eql(u8, action, "init")) {
        if (args.len < 3) {
            try printUsageWithError("Error: init command requires a path argument");
            return;
        }
        try init_cmd.run(args[2]);
    } else {
        try printUsage();
    }
}


fn printUsageWithError(err: []const u8) !void {
    const stderr = std.io.getStdErr().writer();
    try stderr.print("{s}\n\n", .{err});
    try printUsage();
}


fn printUsage() !void {
    const stderr = std.io.getStdErr().writer();
    try stderr.writeAll(
        \\Usage: timbal <command> [args...]
        \\
        \\Commands:
        \\    init <path>    Initialize a new project
        \\                   Use "." to initialize in current directory
        \\
    );
}
