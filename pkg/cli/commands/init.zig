const std = @import("std");
const fs = std.fs;


// Embed template files into the binary.
const dockerignore = @embedFile("../init-templates/.dockerignore");
const pyproject_toml = @embedFile("../init-templates/pyproject.toml");
const timbal_yaml = @embedFile("../init-templates/timbal.yaml");
const flow_py = @embedFile("../init-templates/flow.py");


pub fn run(target_path: []const u8) !void {
    const cwd = fs.cwd();
    const allocator = std.heap.page_allocator;

    const use_current_dir = std.mem.eql(u8, target_path, ".");

    if (!use_current_dir) { 
        try cwd.makePath(target_path);
    }

    var app_dir = if (use_current_dir)
        cwd 
    else 
        try cwd.openDir(target_path, .{});

    const path = try app_dir.realpathAlloc(allocator, ".");
    std.debug.print("Initializing timbal project in {s}\n", .{path});

    // Change the name of the app in the pyproject.toml file.
    const app_name = std.fs.path.basename(path);
    const pyproject_toml_replaced = try std.mem.replaceOwned(
        u8, allocator, pyproject_toml, "{{app_name}}", app_name);

    const init_templates = [_]struct { content: []const u8, dist: []const u8 } {
        .{ .content = dockerignore, .dist = ".dockerignore" },
        .{ .content = pyproject_toml_replaced, .dist = "pyproject.toml" },
        .{ .content = timbal_yaml, .dist = "timbal.yaml" },
        .{ .content = flow_py, .dist = "flow.py" },
    };

    for (init_templates) |init_template| {
        const dist_file = try app_dir.createFile(init_template.dist, .{});
        defer dist_file.close();
        try dist_file.writeAll(init_template.content);
    }

    try uv_sync(allocator, target_path);
}


fn uv_sync(allocator: std.mem.Allocator, target_path: []const u8) !void {
    const uv_args = [_][]const u8{
        "uv", "sync",
        "--python", "3.12",
        "--python-preference", "managed",
    };

    const uv_result = try std.process.Child.run(.{
        .allocator = allocator,
        .argv = &uv_args,
        .cwd = target_path,
    });

    if (uv_result.stderr.len > 0) {
        std.debug.print("uv sync warning: {s}\n", .{uv_result.stderr});
    }
}
