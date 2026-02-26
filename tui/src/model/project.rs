use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

// ---------------------------------------------------------------------------
// timbal.yaml schema
// ---------------------------------------------------------------------------

#[derive(Deserialize, Default)]
struct TimbalYaml {
    _id: Option<String>,
    _type: Option<String>,
    fqn: Option<String>,
}

// ---------------------------------------------------------------------------
// ACE config schemas (variables.yaml / policies.yaml)
// ---------------------------------------------------------------------------

/// A single context variable from variables.yaml.
#[derive(Debug, Clone, Deserialize)]
pub struct AceVariable {
    pub description: Option<String>,
    pub allowed_values: Option<Vec<serde_yaml::Value>>,
}

/// A single policy from policies.yaml.
#[derive(Debug, Clone, Deserialize)]
pub struct AcePolicy {
    pub condition: Option<String>,
    pub action: Option<String>,
    pub context_variables: Option<serde_yaml::Value>,
    pub provides: Option<Vec<String>>,
}

/// ACE configuration found in a workforce member's .ace/ directory.
#[derive(Debug, Clone, Default)]
pub struct AceConfig {
    pub variables: Vec<(String, AceVariable)>,
    pub policies: Vec<(String, AcePolicy)>,
}

// ---------------------------------------------------------------------------
// Public models
// ---------------------------------------------------------------------------

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
    /// ACE configuration, if .ace/ directory exists.
    pub ace: Option<AceConfig>,
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

                let dir_name = entry.file_name().to_string_lossy().to_string();

                let cfg: TimbalYaml = fs::read_to_string(&timbal_yaml)
                    .ok()
                    .and_then(|s| serde_yaml::from_str(&s).ok())
                    .unwrap_or_default();

                let ace = detect_ace(&path);

                match (cfg._id, cfg._type) {
                    (Some(id), Some(kind)) => {
                        members.push(WorkforceMember {
                            name: dir_name,
                            id,
                            kind,
                            fqn: cfg.fqn,
                            ace,
                        });
                    }
                    _ => {
                        legacy_members.push(dir_name);
                    }
                }
            }
        }

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

// ---------------------------------------------------------------------------
// ACE detection
// ---------------------------------------------------------------------------

/// Detect ACE configuration in a workforce member's .ace/ directory.
fn detect_ace(member_dir: &Path) -> Option<AceConfig> {
    let ace_dir = member_dir.join(".ace");
    if !ace_dir.is_dir() {
        return None;
    }

    let variables = load_yaml_map::<AceVariable>(&ace_dir, "variables");
    let policies = load_yaml_map::<AcePolicy>(&ace_dir, "policies");

    Some(AceConfig {
        variables,
        policies,
    })
}

/// Load a YAML file as a map of name -> T. Tries .yaml then .yml.
/// Supports both dict-style (top-level keys) and array-style (items with `name` field).
fn load_yaml_map<T: for<'de> Deserialize<'de>>(dir: &Path, base: &str) -> Vec<(String, T)> {
    let yaml_path = dir.join(format!("{base}.yaml"));
    let yml_path = dir.join(format!("{base}.yml"));
    let path = if yaml_path.is_file() {
        yaml_path
    } else if yml_path.is_file() {
        yml_path
    } else {
        return Vec::new();
    };

    let content = match fs::read_to_string(&path) {
        Ok(s) if !s.trim().is_empty() => s,
        _ => return Vec::new(),
    };

    // Try dict-style first (most common).
    if let Ok(map) = serde_yaml::from_str::<HashMap<String, T>>(&content) {
        let mut entries: Vec<_> = map.into_iter().collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        return entries;
    }

    // Try array-style: each item has a "name" field.
    #[derive(Deserialize)]
    struct Named<V> {
        name: String,
        #[serde(flatten)]
        value: V,
    }
    if let Ok(arr) = serde_yaml::from_str::<Vec<Named<T>>>(&content) {
        return arr.into_iter().map(|n| (n.name, n.value)).collect();
    }

    Vec::new()
}
