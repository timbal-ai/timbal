pub mod app;
pub mod audio;
pub mod commands;
pub mod event;
pub mod model;
pub mod screens;
pub mod theme;
pub mod tui;
pub mod ui;
pub mod widgets;

/// Package version from Cargo.toml, used in the logo header and help screen.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
