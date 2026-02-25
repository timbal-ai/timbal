use std::fmt;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::config::credentials_path;

fn history_path() -> std::path::PathBuf {
    credentials_path().parent().unwrap().join("history.jsonl")
}

fn now_ts() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// The kind of event recorded.
#[derive(Debug, Clone)]
pub enum EntryKind {
    /// User sent a message.
    Message(String),
    /// A slash command was run.
    Command(String),
    /// /configure completed (with profile name).
    ConfigureSaved(String),
    /// /configure was cancelled.
    ConfigureCancelled,
    /// Thinking was interrupted by the user.
    Interrupted,
    /// Session started.
    SessionStart,
}

impl fmt::Display for EntryKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EntryKind::Message(text) => write!(f, "{text}"),
            EntryKind::Command(cmd) => write!(f, "{cmd}"),
            EntryKind::ConfigureSaved(profile) => {
                write!(f, "Credentials saved (profile: {profile})")
            }
            EntryKind::ConfigureCancelled => write!(f, "Config dialog dismissed"),
            EntryKind::Interrupted => write!(f, "Interrupted"),
            EntryKind::SessionStart => write!(f, "Session started"),
        }
    }
}

/// A single history entry held in memory and persisted to disk.
#[derive(Debug, Clone)]
pub struct Entry {
    pub ts: u64,
    pub kind: EntryKind,
}

impl Entry {
    pub fn new(kind: EntryKind) -> Self {
        Self { ts: now_ts(), kind }
    }

    /// Render the timestamp as HH:MM.
    pub fn time_str(&self) -> String {
        let secs = self.ts % 86400;
        let h = secs / 3600;
        let m = (secs % 3600) / 60;
        format!("{h:02}:{m:02}")
    }

    /// Serialise as a single JSON line for .jsonl.
    fn to_jsonl(&self) -> String {
        let kind_tag = match &self.kind {
            EntryKind::Message(_) => "message",
            EntryKind::Command(_) => "command",
            EntryKind::ConfigureSaved(_) => "configure_saved",
            EntryKind::ConfigureCancelled => "configure_cancelled",
            EntryKind::Interrupted => "interrupted",
            EntryKind::SessionStart => "session_start",
        };
        let content = self.kind.to_string().replace('"', "\\\"");
        format!(r#"{{"ts":{},"kind":"{}","content":"{}"}}"#, self.ts, kind_tag, content)
    }
}

/// Append an entry to ~/.timbal/history.jsonl.
pub fn append(entry: &Entry) {
    let path = history_path();
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(&path) {
        let _ = writeln!(file, "{}", entry.to_jsonl());
    }
}
