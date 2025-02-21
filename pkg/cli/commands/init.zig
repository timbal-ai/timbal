const std = @import("std");
const fs = std.fs;


// Embed template files into the binary.
const dockerignore = @embedFile("../init-templates/.dockerignore");
const timbal_yaml = @embedFile("../init-templates/timbal.yaml");
const flow_py = @embedFile("../init-templates/flow.py");


pub fn run(target_path: []const u8) !void {
    const cwd = fs.cwd();

    const use_current_dir = std.mem.eql(u8, target_path, ".");

    if (!use_current_dir) { 
        try cwd.makePath(target_path);
    }

    var app_dir = if (use_current_dir)
        cwd 
    else 
        try cwd.openDir(target_path, .{});

    const path = try app_dir.realpathAlloc(std.heap.page_allocator, ".");
    std.debug.print("Initializing timbal project in {s}\n", .{path});

    const init_templates = [_]struct { content: []const u8, dist: []const u8 } {
        .{ .content = dockerignore, .dist = ".dockerignore" },
        .{ .content = timbal_yaml, .dist = "timbal.yaml" },
        .{ .content = flow_py, .dist = "flow.py" },
    };

    for (init_templates) |init_template| {
        const dist_file = try app_dir.createFile(init_template.dist, .{});
        defer dist_file.close();
        try dist_file.writeAll(init_template.content);
    }
}
