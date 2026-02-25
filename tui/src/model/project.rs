use std::fs;
use std::path::Path;

/// A workforce member parsed from workforce/<name>/timbal.yaml.
#[derive(Debug, Clone)]
pub struct WorkforceMember {
    /// Directory name (e.g. "brave-seal").
    pub name: String,
    /// The _id field from timbal.yaml.
    pub id: String,
    /// The _type field ("agent" or "workflow").
    pub kind: String,
    /// The fqn field (e.g. "agent.py::agent").
    pub fqn: Option<String>,
}

/// Context about the current directory as a Timbal project.
#[derive(Debug, Clone)]
pub struct ProjectContext {
    /// Whether this directory is a Timbal project.
    pub is_timbal_project: bool,
    /// Workforce members with valid _id and _type.
    pub members: Vec<WorkforceMember>,
    /// Legacy members (timbal.yaml exists but missing _id/_type).
    pub legacy_members: Vec<String>,
    /// Whether api/ directory exists.
    pub has_api: bool,
    /// Whether ui/ directory exists.
    pub has_ui: bool,
}

impl ProjectContext {
    /// Inspect the given directory (typically cwd) for Timbal project markers.
    pub fn detect(root: &Path) -> Self {
        let workforce_dir = root.join("workforce");

        if !workforce_dir.is_dir() {
            return Self {
                is_timbal_project: false,
                members: Vec::new(),
                legacy_members: Vec::new(),
                has_api: false,
                has_ui: false,
            };
        }

        let mut members = Vec::new();
        let mut legacy_members = Vec::new();

        if let Ok(entries) = fs::read_dir(&workforce_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if !path.is_dir() {
                    continue;
                }

                let timbal_yaml = path.join("timbal.yaml");
                if !timbal_yaml.is_file() {
                    continue;
                }

                let dir_name = entry
                    .file_name()
                    .to_string_lossy()
                    .to_string();

                let content = fs::read_to_string(&timbal_yaml).unwrap_or_default();

                let id = yaml_value(&content, "_id");
                let kind = yaml_value(&content, "_type");
                let fqn = yaml_value(&content, "fqn");

                match (id, kind) {
                    (Some(id), Some(kind)) => {
                        members.push(WorkforceMember {
                            name: dir_name,
                            id,
                            kind,
                            fqn,
                        });
                    }
                    _ => {
                        legacy_members.push(dir_name);
                    }
                }
            }
        }

        // Sort members by name for deterministic display.
        members.sort_by(|a, b| a.name.cmp(&b.name));
        legacy_members.sort();

        let is_timbal_project = !members.is_empty() || !legacy_members.is_empty();
        let has_api = root.join("api").is_dir();
        let has_ui = root.join("ui").is_dir();

        Self {
            is_timbal_project,
            members,
            legacy_members,
            has_api,
            has_ui,
        }
    }
}

/// Extract a top-level YAML value by key.
/// Handles: `key: "value"` and `key: value` (strips quotes).
fn yaml_value(content: &str, key: &str) -> Option<String> {
    let prefix = format!("{}:", key);
    for line in content.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix(&prefix) {
            let val = rest.trim();
            if val.is_empty() {
                return None;
            }
            // Strip surrounding quotes if present.
            let val = val
                .strip_prefix('"')
                .and_then(|v| v.strip_suffix('"'))
                .unwrap_or(val);
            if val.is_empty() {
                return None;
            }
            return Some(val.to_string());
        }
    }
    None
}
