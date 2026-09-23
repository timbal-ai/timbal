const std = @import("std");
const builtin = @import("builtin");
const env = @import("env.zig");

const usage =
    \\Push the current Git branch to Timbal, creating and linking a project if needed.
    \\
    \\Usage: timbal push [OPTIONS]
    \\
    \\    --name <NAME>       Name for a new project (default: repository directory)
    \\    --org <ID>          Organization for a new project (default: configured org)
    \\    --profile <NAME>    Configuration and credentials profile
    \\    -n, --dry-run       Preview without creating a project or changing Git config
    \\    -q, --quiet         Suppress progress
    \\    -v, --verbose       Show Git progress
    \\    -h, --help          Show this help
    \\
    \\Git options: --force-with-lease[=<ref>:<expect>], --force, --no-verify,
    \\             --porcelain, --follow-tags, --atomic, -u/--set-upstream
    \\
    \\Only committed files are pushed. Existing remotes and upstreams are preserved
    \\unless --set-upstream is requested. Use timbal env push to upload environment vars.
    \\
;

const Options = struct {
    name: ?[]const u8 = null,
    org: ?[]const u8 = null,
    profile: ?[]const u8 = null,
    dry_run: bool = false,
    quiet: bool = false,
    git_flags: std.ArrayList([]const u8),
};

fn parseArgs(a: std.mem.Allocator, args: []const []const u8) !Options {
    var opts = Options{ .git_flags = std.ArrayList([]const u8).init(a) };
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (eq(arg, "--name") or eq(arg, "--org") or eq(arg, "--profile")) {
            i += 1;
            if (i == args.len or args[i].len == 0 or std.mem.startsWith(u8, args[i], "--")) return error.InvalidArguments;
            if (eq(arg, "--name")) opts.name = args[i];
            if (eq(arg, "--org")) opts.org = args[i];
            if (eq(arg, "--profile")) opts.profile = args[i];
        } else if (eq(arg, "--dry-run") or eq(arg, "-n")) {
            opts.dry_run = true;
            try opts.git_flags.append("--dry-run");
        } else if (eq(arg, "--quiet") or eq(arg, "-q")) {
            opts.quiet = true;
            try opts.git_flags.append("--quiet");
        } else if (eq(arg, "--verbose") or eq(arg, "-v") or eq(arg, "--force") or eq(arg, "-f") or
            eq(arg, "--force-with-lease") or std.mem.startsWith(u8, arg, "--force-with-lease=") or
            eq(arg, "--no-verify") or eq(arg, "--porcelain") or eq(arg, "--follow-tags") or
            eq(arg, "--atomic") or eq(arg, "--set-upstream") or eq(arg, "-u"))
        {
            try opts.git_flags.append(arg);
        } else return error.InvalidArguments;
    }
    return opts;
}

fn eq(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

fn fail(message: []const u8) error{PushFailed} {
    std.io.getStdErr().writer().print("Error: {s}\n", .{message}) catch {};
    return error.PushFailed;
}

fn git(a: std.mem.Allocator, args: []const []const u8) !std.process.Child.RunResult {
    return gitWithEnv(a, args, null);
}

fn gitWithEnv(a: std.mem.Allocator, args: []const []const u8, child_env: ?*const std.process.EnvMap) !std.process.Child.RunResult {
    const argv = try a.alloc([]const u8, args.len + 1);
    argv[0] = "git";
    @memcpy(argv[1..], args);
    return std.process.Child.run(.{ .allocator = a, .argv = argv, .env_map = child_env, .max_output_bytes = 1024 * 1024 });
}

fn ok(result: std.process.Child.RunResult) bool {
    return result.term == .Exited and result.term.Exited == 0;
}

fn capture(a: std.mem.Allocator, args: []const []const u8) ![]const u8 {
    const result = try git(a, args);
    if (!ok(result)) {
        try std.io.getStdErr().writeAll(result.stderr);
        return error.GitFailed;
    }
    return std.mem.trim(u8, result.stdout, "\r\n");
}

const Remote = struct { name: []const u8, url: []const u8 };

/// Inspect effective push URLs (including pushurl/insteadOf), never just fetch URLs.
/// A remote with multiple push destinations is unsafe to implicitly select.
fn findRemote(a: std.mem.Allocator) !?Remote {
    const names = try capture(a, &.{"remote"});
    var lines = std.mem.tokenizeAny(u8, names, "\r\n");
    var selected: ?Remote = null;
    var count: usize = 0;
    var named_timbal: ?Remote = null;
    while (lines.next()) |name| {
        const urls = try capture(a, &.{ "remote", "get-url", "--push", "--all", name });
        var iter = std.mem.tokenizeAny(u8, urls, "\r\n");
        var url_count: usize = 0;
        var timbal_url: ?[]const u8 = null;
        while (iter.next()) |url| {
            url_count += 1;
            if (try env.parseTimbalRemoteUrl(a, url, name) != null) timbal_url = url;
        }
        if (timbal_url) |url| {
            if (url_count != 1) return fail("Timbal remote has multiple push URLs; configure a single destination first.");
            const remote = Remote{ .name = name, .url = url };
            selected = remote;
            count += 1;
            if (eq(name, "timbal")) named_timbal = remote;
        } else if (eq(name, "timbal")) {
            return fail("The 'timbal' remote points elsewhere. Rename it before linking a Timbal project.");
        }
    }
    if (named_timbal) |remote| return remote;
    if (count > 1) return fail("Multiple Timbal remotes found. Name the intended remote 'timbal'.");
    return selected;
}

fn readConfig(a: std.mem.Allocator, home: []const u8, name: []const u8) ![]const u8 {
    const path = try std.fs.path.join(a, &.{ home, ".timbal", name });
    return std.fs.cwd().readFileAlloc(a, path, 1024 * 1024) catch |err| switch (err) {
        error.FileNotFound => "",
        else => return err,
    };
}

fn validId(id: []const u8) bool {
    const n = std.fmt.parseInt(i64, id, 10) catch return false;
    if (n <= 0) return false;
    for (id) |c| if (!std.ascii.isDigit(c)) return false;
    return true;
}

pub const CreateRequest = struct {
    base_url: []const u8,
    org: []const u8,
    api_key: []const u8,
    name: []const u8,
    branch: []const u8,
};

fn projectId(a: std.mem.Allocator, body: []const u8) ![]const u8 {
    const json = try std.json.parseFromSlice(std.json.Value, a, body, .{ .allocate = .alloc_always });
    if (json.value != .object) return error.InvalidProjectResponse;
    const value = json.value.object.get("id") orelse return error.InvalidProjectResponse;
    const id = switch (value) {
        .string => value.string,
        .integer => try std.fmt.allocPrint(a, "{d}", .{value.integer}),
        else => return error.InvalidProjectResponse,
    };
    if (!validId(id)) return error.InvalidProjectResponse;
    return id;
}

fn configValue(a: std.mem.Allocator, key: []const u8) !?[]const u8 {
    const result = try git(a, &.{ "config", "--local", "--get", key });
    if (ok(result)) return std.mem.trim(u8, result.stdout, "\r\n");
    if (result.term == .Exited and result.term.Exited == 1) return null;
    return error.GitFailed;
}

fn validSha(sha: []const u8) bool {
    if (sha.len != 0 and sha.len != 40 and sha.len != 64) return false;
    for (sha) |c| if (!std.ascii.isHex(c)) return false;
    return true;
}

fn scaffoldSha(a: std.mem.Allocator, remote: Remote, child_env: *const std.process.EnvMap) ![]const u8 {
    if (try configValue(a, "timbal.bootstrap-sha")) |saved| {
        if (!validSha(saved)) return fail("Invalid scaffold SHA in Git config.");
        return saved; // Never refresh a lease after a rejected or uncertain push.
    }
    const refs = try gitWithEnv(a, &.{ "ls-remote", "--heads", remote.url, "refs/heads/main" }, child_env);
    if (!ok(refs)) {
        try std.io.getStdErr().writeAll(refs.stderr);
        return error.GitFailed;
    }
    const line = std.mem.trim(u8, refs.stdout, "\r\n");
    const sha = if (line.len == 0) "" else line[0 .. std.mem.indexOfScalar(u8, line, '\t') orelse return error.InvalidScaffold];
    if (!validSha(sha)) return error.InvalidScaffold;
    if (sha.len > 0) {
        const ref = "refs/timbal/cli-bootstrap";
        const fetched = try gitWithEnv(a, &.{ "fetch", "--no-tags", "--no-write-fetch-head", remote.url, "+refs/heads/main:" ++ ref }, child_env);
        if (!ok(fetched)) {
            try std.io.getStdErr().writeAll(fetched.stderr);
            return error.GitFailed;
        }
        defer _ = git(a, &.{ "update-ref", "-d", ref }) catch {};
        const fetched_sha = try capture(a, &.{ "rev-parse", ref });
        const count = try capture(a, &.{ "rev-list", "--count", ref });
        const identity = try capture(a, &.{ "log", "-1", "--format=%s%n%ae", ref });
        if (!eq(sha, fetched_sha) or !eq(count, "1") or !eq(identity, "Initial scaffold\nnoreply@timbal.ai"))
            return fail("The new project's main branch is no longer the generated scaffold. Refusing automatic replacement.");
    }
    _ = try capture(a, &.{ "config", "--local", "timbal.bootstrap-sha", sha });
    return sha;
}

const Api = struct {
    pub fn createProject(a: std.mem.Allocator, req: CreateRequest) ![]const u8 {
        var client = std.http.Client{ .allocator = a };
        defer client.deinit();
        const url = try std.fmt.allocPrint(a, "{s}/orgs/{s}/projects", .{ req.base_url, req.org });
        const payload = try std.json.stringifyAlloc(a, .{
            .name = req.name,
            .origin = .{ .type = "Scratch", .with_ui = false, .agents = &[_][]const u8{}, .workflows = &[_][]const u8{} },
        }, .{});
        const auth = try std.fmt.allocPrint(a, "Bearer {s}", .{req.api_key});
        var body = std.ArrayList(u8).init(a);
        const response = client.fetch(.{
            .location = .{ .url = url },
            .method = .POST,
            .payload = payload,
            .redirect_behavior = .not_allowed,
            .extra_headers = &.{
                .{ .name = "Authorization", .value = auth },
                .{ .name = "Content-Type", .value = "application/json" },
                .{ .name = "Accept", .value = "application/json" },
            },
            .response_storage = .{ .dynamic = &body },
            .max_append_size = 1024 * 1024,
        }) catch |err| {
            try std.io.getStdErr().writer().print("Project creation response unavailable ({s}). Check the platform for '{s}' before retrying; if created, add its Git URL as the timbal remote.\n", .{ @errorName(err), req.name });
            return error.ProjectCreationUncertain;
        };
        const status = @intFromEnum(response.status);
        if (status < 200 or status >= 300) {
            try std.io.getStdErr().writer().print("Project creation failed (HTTP {d}). Check credentials, organization permissions, and project creation permissions.\n", .{status});
            return error.ProjectCreationFailed;
        }
        return projectId(a, body.items) catch {
            return fail("Project created but its ID could not be read. Link its Git URL from the platform before retrying.");
        };
    }
};

pub fn run(a: std.mem.Allocator, args: []const []const u8) !void {
    const code = runWithApi(a, args, Api) catch |err| {
        if (err == error.InvalidArguments) {
            try std.io.getStdErr().writeAll("Invalid push options.\n\n" ++ usage);
            std.process.exit(2);
        }
        return err;
    };
    if (code != 0) std.process.exit(code);
}

/// API injection keeps the full bootstrap/Git flow testable without live projects.
pub fn runWithApi(parent: std.mem.Allocator, args: []const []const u8, comptime api: type) !u8 {
    var arena = std.heap.ArenaAllocator.init(parent);
    defer arena.deinit();
    const a = arena.allocator();
    for (args) |arg| if (eq(arg, "--help") or eq(arg, "-h")) {
        try std.io.getStdErr().writeAll(usage);
        return 0;
    };
    const opts = try parseArgs(a, args);
    const root = capture(a, &.{ "rev-parse", "--show-toplevel" }) catch return fail("Run timbal push inside a Git working tree.");
    const branch_result = try git(a, &.{ "symbolic-ref", "--quiet", "--short", "HEAD" });
    if (!ok(branch_result)) return fail("HEAD is detached. Check out a branch before pushing.");
    const branch = std.mem.trim(u8, branch_result.stdout, "\r\n");
    if (!ok(try git(a, &.{ "rev-parse", "--verify", "HEAD^{commit}" }))) {
        return fail("No commits to push. Stage your project files and run git commit first.");
    }
    const profile = opts.profile orelse (std.process.getEnvVarOwned(a, "TIMBAL_PROFILE") catch "default");
    var child_env = try std.process.getEnvMap(a);
    try child_env.put("TIMBAL_PROFILE", profile);
    // Cover creation, scaffold verification, and upload across linked worktrees.
    var lock: ?std.fs.File = null;
    defer if (lock) |file| file.close();
    if (!opts.dry_run) {
        const common = try capture(a, &.{ "rev-parse", "--git-common-dir" });
        const lock_path = try std.fs.path.join(a, &.{ common, "timbal-push.lock" });
        lock = std.fs.cwd().createFile(lock_path, .{ .truncate = false, .lock = .exclusive, .lock_nonblocking = true }) catch |err| switch (err) {
            error.WouldBlock => return fail("Another timbal push is linking this repository. Retry when it finishes."),
            else => return err,
        };
    }
    var remote = try findRemote(a);
    if (remote != null and (opts.org != null or opts.name != null)) {
        return fail("--org and --name only apply before a project is linked. This repository already has a Timbal remote.");
    }
    if (remote == null) {
        const home = try std.process.getEnvVarOwned(a, if (builtin.os.tag == .windows) "USERPROFILE" else "HOME");
        const config = try readConfig(a, home, "config");
        const org = opts.org orelse env.readValue(config, profile, "org") orelse
            return fail("No organization configured. Run timbal configure or pass --org <ID>.");
        if (!validId(org)) return fail("Organization ID must be a positive integer.");
        const base_url = try env.normalizeBaseUrlOverride(a, env.readValue(config, profile, "base_url") orelse "https://api.timbal.ai");
        const name = opts.name orelse std.fs.path.basename(root);
        if (opts.dry_run) {
            try std.io.getStdErr().writer().print("Would create project '{s}' in organization {s} at {s}, add remote 'timbal', and push branch '{s}'.\n", .{ name, org, base_url, branch });
            return 0;
        }
        const credentials = try readConfig(a, home, "credentials");
        const key = env.readValue(credentials, profile, "api_key") orelse
            return fail("No API key configured for this profile. Run timbal configure.");
        remote = try findRemote(a);
        if (remote == null) {
            // Save the URL separately before adding the remote. If Git remote add
            // fails, a retry recovers the already-created project instead of creating another.
            const saved = try git(a, &.{ "config", "--local", "--get", "timbal.pending-project-url" });
            var url: []const u8 = undefined;
            if (ok(saved)) {
                url = std.mem.trim(u8, saved.stdout, "\r\n");
                if (try env.parseTimbalRemoteUrl(a, url, "timbal") == null) return fail("Invalid pending Timbal project URL in Git config.");
            } else {
                // Exit 1 means the key is absent; other failures must not create
                // a second project when the saved link cannot be read.
                if (saved.term != .Exited or saved.term.Exited != 1) return error.GitFailed;
                if (!opts.quiet) try std.io.getStdErr().writer().print("Creating Timbal project '{s}'...\n", .{name});
                const id = try api.createProject(a, .{
                    .base_url = base_url,
                    .org = org,
                    .api_key = key,
                    .name = name,
                    .branch = branch,
                });
                if (!validId(id)) return error.InvalidProjectResponse;
                url = try std.fmt.allocPrint(a, "{s}/orgs/{s}/projects/{s}/git", .{ base_url, org, id });
                _ = capture(a, &.{ "config", "--local", "timbal.pending-project-url", url }) catch {
                    try std.io.getStdErr().writer().print("Project created. Before retrying, link it with: git remote add timbal {s}\n", .{url});
                    return error.LinkFailed;
                };
            }
            // A persisted ownership marker limits the automatic lease to a
            // project created by this command, including a failed-link retry.
            const previous = try configValue(a, "timbal.bootstrap-url");
            if (previous == null or !eq(previous.?, url)) {
                _ = try git(a, &.{ "config", "--local", "--unset", "timbal.bootstrap-sha" });
            }
            _ = try capture(a, &.{ "config", "--local", "timbal.bootstrap-url", url });
            _ = try capture(a, &.{ "remote", "add", "timbal", url });
            _ = try git(a, &.{ "config", "--local", "--unset", "timbal.pending-project-url" });
            // Re-read effective push URLs: local Git URL rewrites must not send
            // this automatic push to a different host/project.
            remote = try findRemote(a);
            if (remote == null or !eq(remote.?.url, url)) return fail("Git URL rewriting changed the new remote destination. Review Git configuration before pushing.");
            if (!opts.quiet) try std.io.getStdErr().writer().print("Linked remote 'timbal': {s}\n", .{url});
        }
    }
    const destination = remote.?;
    const bootstrap_url = try configValue(a, "timbal.bootstrap-url");
    const bootstrap = eq(branch, "main") and bootstrap_url != null and eq(bootstrap_url.?, destination.url);
    var lease: ?[]const u8 = null;
    if (bootstrap) {
        for (opts.git_flags.items) |flag| {
            if (eq(flag, "--force") or eq(flag, "-f") or std.mem.startsWith(u8, flag, "--force-with-lease"))
                return fail("First push uses a scaffold-specific lease. Omit force options until bootstrap succeeds.");
        }
        if (opts.dry_run) {
            try std.io.getStdErr().writeAll("Would verify the newly created project's scaffold and replace main using its exact commit SHA as a lease.\n");
            return 0;
        }
        const sha = try scaffoldSha(a, destination, &child_env);
        lease = try std.fmt.allocPrint(a, "--force-with-lease=refs/heads/main:{s}", .{sha});
        if (!opts.quiet) try std.io.getStdErr().writeAll("Importing local history over the generated scaffold (exact-SHA lease).\n");
    }
    var argv = std.ArrayList([]const u8).init(a);
    try argv.appendSlice(&.{ "git", "push" });
    try argv.appendSlice(opts.git_flags.items);
    if (lease) |flag| try argv.append(flag);
    try argv.appendSlice(&.{ "--", destination.name, try std.fmt.allocPrint(a, "HEAD:refs/heads/{s}", .{branch}) });
    var child = std.process.Child.init(argv.items, a);
    child.env_map = &child_env;
    child.stdin_behavior = .Inherit;
    child.stdout_behavior = .Inherit;
    child.stderr_behavior = .Inherit;
    const term = try child.spawnAndWait();
    const code: u8 = switch (term) {
        .Exited => |code| code,
        else => 1,
    };
    if (code == 0 and bootstrap and !opts.dry_run) {
        _ = try git(a, &.{ "config", "--local", "--unset", "timbal.bootstrap-url" });
        _ = try git(a, &.{ "config", "--local", "--unset", "timbal.bootstrap-sha" });
    }
    return code;
}

test "project creation response IDs must be safe path components" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectEqualStrings("123", try projectId(a, "{\"id\":\"123\"}"));
    try std.testing.expectEqualStrings("123", try projectId(a, "{\"id\":123}"));
    for ([_][]const u8{ "{}", "[]", "{\"id\":0}", "{\"id\":\"../x\"}", "{\"id\":1.5}", "{\"id\":\"+2\"}" }) |body| {
        try std.testing.expectError(error.InvalidProjectResponse, projectId(a, body));
    }
}

test "push options cannot replace destination or smuggle arbitrary git config" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectError(error.InvalidArguments, parseArgs(a, &.{"--repo=https://elsewhere"}));
    try std.testing.expectError(error.InvalidArguments, parseArgs(a, &.{ "origin", "main" }));
    try std.testing.expectError(error.InvalidArguments, parseArgs(a, &.{"--mirror"}));
    try std.testing.expectError(error.InvalidArguments, parseArgs(a, &.{"--org"}));
    const options = try parseArgs(a, &.{ "--profile", "dev", "--force-with-lease", "--dry-run" });
    try std.testing.expectEqualStrings("dev", options.profile.?);
    try std.testing.expect(options.dry_run);
    try std.testing.expectEqual(@as(usize, 2), options.git_flags.items.len);
}
