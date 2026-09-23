// Offline fixture executable for integration_tests_push.py. Never shipped.
const std = @import("std");
const push = @import("commands/push.zig");
const FakeApi = struct {
    pub fn createProject(a: std.mem.Allocator, req: push.CreateRequest) ![]const u8 {
        const path = try std.process.getEnvVarOwned(a, "PUSH_TEST_EVENTS");
        const file = try std.fs.cwd().createFile(path, .{ .truncate = false });
        defer file.close();
        try file.seekFromEnd(0);
        try std.json.stringify(.{ .org = req.org, .name = req.name, .branch = req.branch, .base_url = req.base_url, .api_key = req.api_key }, .{}, file.writer());
        try file.writeAll("\n");
        if (std.process.hasEnvVarConstant("PUSH_TEST_API_FAIL")) return error.FakeApiFailure;
        return "99";
    }
};
pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const args = try std.process.argsAlloc(arena.allocator());
    const code = push.runWithApi(arena.allocator(), args[1..], FakeApi) catch |err| {
        std.debug.print("Error: {s}\n", .{@errorName(err)});
        std.process.exit(if (err == error.InvalidArguments) 2 else 1);
    };
    std.process.exit(code);
}
