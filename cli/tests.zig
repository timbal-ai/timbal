// Top-level test aggregator. `zig build test` uses this as its root, so any
// file with tests MUST be referenced here — Zig only discovers tests in
// modules reachable via @import from the root. Plain modules used by the
// production binary (e.g. when main.zig pulls in a command) are *not*
// guaranteed to expose their tests unless we explicitly import them here.
//
// Pattern matches the upstream Zig stdlib `tests.zig` aggregator.

comptime {
    _ = @import("main.zig");
    _ = @import("utils.zig");
    _ = @import("commands/add.zig");
    _ = @import("commands/configure.zig");
    _ = @import("commands/create.zig");
    _ = @import("commands/credential_helper.zig");
    _ = @import("commands/env.zig");
    _ = @import("commands/start.zig");
    _ = @import("commands/upgrade.zig");
}
