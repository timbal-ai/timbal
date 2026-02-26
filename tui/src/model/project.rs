use serde::{Deserialize, Serialize};
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
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AceVariable {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_values: Option<Vec<serde_yaml::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_by: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub updated_by: Option<String>,
}

/// A single policy from policies.yaml.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AcePolicy {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub condition: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub action: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_variables: Option<serde_yaml::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provides: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_by: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub updated_by: Option<String>,
}

/// ACE configuration found in a workforce member's .ace/ directory.
#[derive(Debug, Clone, Default)]
pub struct AceConfig {
    pub variables: Vec<(String, AceVariable)>,
    pub policies: Vec<(String, AcePolicy)>,
}

// ---------------------------------------------------------------------------
// Evals schemas (evals/*.yaml)
// ---------------------------------------------------------------------------

/// A single eval test case parsed from an eval YAML file.
#[derive(Debug, Clone, Deserialize)]
pub struct EvalCase {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub timeout: Option<f64>,
    #[serde(default)]
    pub runnable: Option<String>,
    #[serde(default)]
    pub params: Option<serde_yaml::Value>,
    /// All remaining fields (validators, output, seq!, etc.) captured as-is.
    #[serde(flatten)]
    pub extra: HashMap<String, serde_yaml::Value>,
}

/// Evals discovered in a workforce member's evals/ directory.
#[derive(Debug, Clone, Default)]
pub struct EvalsConfig {
    /// All eval cases discovered, grouped by source file name.
    pub files: Vec<(String, Vec<EvalCase>)>,
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
    /// Evals configuration, if evals/ directory exists.
    pub evals: Option<EvalsConfig>,
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
                let evals = detect_evals(&path);

                match (cfg._id, cfg._type) {
                    (Some(id), Some(kind)) => {
                        members.push(WorkforceMember {
                            name: dir_name,
                            id,
                            kind,
                            fqn: cfg.fqn,
                            ace,
                            evals,
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

// ---------------------------------------------------------------------------
// Evals detection
// ---------------------------------------------------------------------------

/// Discover eval files in a workforce member's evals/ directory.
/// Follows the same pattern as the Python CLI: files matching `eval*.yaml` or `*eval.yaml`.
fn detect_evals(member_dir: &Path) -> Option<EvalsConfig> {
    let evals_dir = member_dir.join("evals");
    if !evals_dir.is_dir() {
        return None;
    }

    let entries = match fs::read_dir(&evals_dir) {
        Ok(e) => e,
        Err(_) => return None,
    };

    let mut files: Vec<(String, Vec<EvalCase>)> = Vec::new();

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        if !name.ends_with(".yaml") && !name.ends_with(".yml") {
            continue;
        }

        // Match eval*.yaml or *eval.yaml (case-insensitive stem check).
        let stem = path.file_stem().unwrap_or_default().to_string_lossy().to_lowercase();
        let is_eval_file = stem.starts_with("eval") || stem.ends_with("eval");
        if !is_eval_file {
            continue;
        }

        let content = match fs::read_to_string(&path) {
            Ok(s) if !s.trim().is_empty() => s,
            _ => continue,
        };

        if let Ok(cases) = serde_yaml::from_str::<Vec<EvalCase>>(&content) {
            if !cases.is_empty() {
                files.push((name, cases));
            }
        }
    }

    if files.is_empty() {
        return None;
    }

    files.sort_by(|a, b| a.0.cmp(&b.0));
    Some(EvalsConfig { files })
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

// ---------------------------------------------------------------------------
// ACE saving
// ---------------------------------------------------------------------------

/// Save an AceConfig back to disk for the given agent name.
/// Writes dict-style YAML to workforce/<agent>/.ace/variables.yaml and policies.yaml.
pub fn save_ace(agent_name: &str, ace: &AceConfig) -> Result<(), String> {
    let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
    let ace_dir = cwd.join("workforce").join(agent_name).join(".ace");

    if !ace_dir.is_dir() {
        return Err(format!("ACE directory not found: {}", ace_dir.display()));
    }

    // Save variables.
    let var_map: HashMap<&str, &AceVariable> = ace
        .variables
        .iter()
        .map(|(name, var)| (name.as_str(), var))
        .collect();
    let var_yaml = serde_yaml::to_string(&var_map).map_err(|e| e.to_string())?;
    fs::write(ace_dir.join("variables.yaml"), var_yaml).map_err(|e| e.to_string())?;

    // Save policies.
    let pol_map: HashMap<&str, &AcePolicy> = ace
        .policies
        .iter()
        .map(|(name, pol)| (name.as_str(), pol))
        .collect();
    let pol_yaml = serde_yaml::to_string(&pol_map).map_err(|e| e.to_string())?;
    fs::write(ace_dir.join("policies.yaml"), pol_yaml).map_err(|e| e.to_string())?;

    Ok(())
}

/// Get the current git identity as "Name <email>".
pub fn git_identity() -> String {
    let name = std::process::Command::new("git")
        .args(["config", "user.name"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();

    let email = std::process::Command::new("git")
        .args(["config", "user.email"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();

    match (name.is_empty(), email.is_empty()) {
        (false, false) => format!("{name} <{email}>"),
        (false, true) => name,
        (true, false) => email,
        (true, true) => "unknown".to_string(),
    }
}

/// Get the current UTC timestamp as ISO 8601.
pub fn now_utc() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}
