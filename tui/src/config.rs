use std::fs;
use std::path::{Path, PathBuf};

use color_eyre::Result;
use dirs::home_dir;

fn timbal_dir() -> PathBuf {
    home_dir().unwrap_or_default().join(".timbal")
}

pub fn credentials_path() -> PathBuf {
    timbal_dir().join("credentials")
}

pub fn config_path() -> PathBuf {
    timbal_dir().join("config")
}

fn read_or_empty(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_default()
}

/// Read a key from an INI section (profile).
/// "default" → [default], anything else → [profile <name>].
pub fn read_value(content: &str, profile: &str, key: &str) -> Option<String> {
    let section_header = section_header(profile);
    let mut in_target = false;

    for line in content.lines() {
        let trimmed = line.trim();
        if is_section_header(trimmed) {
            in_target = trimmed == section_header;
            continue;
        }
        if in_target {
            if let Some(rest) = trimmed.strip_prefix(key) {
                let rest = rest.trim_start();
                if let Some(val) = rest.strip_prefix('=') {
                    let val = val.trim();
                    if !val.is_empty() {
                        return Some(val.to_string());
                    }
                }
            }
        }
    }
    None
}

/// Upsert a key=value in the given profile section.
pub fn upsert_value(content: &str, profile: &str, key: &str, value: &str) -> String {
    let section_hdr = section_header(profile);
    let mut result = String::new();
    let mut found_section = false;
    let mut replaced = false;
    let mut in_target = false;

    for line in content.lines() {
        let trimmed = line.trim();
        if is_section_header(trimmed) {
            if in_target && !replaced {
                result.push_str(&format!("{key} = {value}\n"));
                replaced = true;
            }
            in_target = trimmed == section_hdr;
            if in_target {
                found_section = true;
            }
            result.push_str(line);
            result.push('\n');
            continue;
        }

        if in_target && trimmed.starts_with(key) {
            let rest = trimmed[key.len()..].trim_start();
            if rest.starts_with('=') {
                result.push_str(&format!("{key} = {value}\n"));
                replaced = true;
                continue;
            }
        }

        result.push_str(line);
        result.push('\n');
    }

    if in_target && !replaced {
        if !result.ends_with('\n') {
            result.push('\n');
        }
        result.push_str(&format!("{key} = {value}\n"));
    } else if !found_section {
        if !result.is_empty() && !result.ends_with('\n') {
            result.push('\n');
        }
        result.push_str(&format!("{section_hdr}\n{key} = {value}\n"));
    }

    result
}

fn section_header(profile: &str) -> String {
    if profile == "default" {
        "[default]".to_string()
    } else {
        format!("[profile {profile}]")
    }
}

fn is_section_header(line: &str) -> bool {
    line.starts_with('[') && line.ends_with(']')
}

fn write_file(path: &Path, content: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, content)?;
    Ok(())
}

pub struct TimbalConfig {
    pub api_key: Option<String>,
    pub org: Option<String>,
    pub base_url: Option<String>,
}

impl TimbalConfig {
    pub fn load(profile: &str) -> Self {
        let creds = read_or_empty(&credentials_path());
        let cfg = read_or_empty(&config_path());
        Self {
            api_key: read_value(&creds, profile, "api_key"),
            org: read_value(&cfg, profile, "org"),
            base_url: read_value(&cfg, profile, "base_url"),
        }
    }

    pub fn save(&self, profile: &str) -> Result<()> {
        let cred_path = credentials_path();
        let cfg_path = config_path();

        let mut creds = read_or_empty(&cred_path);
        let mut cfg = read_or_empty(&cfg_path);

        if let Some(key) = &self.api_key {
            creds = upsert_value(&creds, profile, "api_key", key);
            write_file(&cred_path, &creds)?;
        }
        if let Some(org) = &self.org {
            cfg = upsert_value(&cfg, profile, "org", org);
        }
        let base_url = self.base_url.as_deref().unwrap_or("https://api.timbal.ai");
        cfg = upsert_value(&cfg, profile, "base_url", base_url);
        write_file(&cfg_path, &cfg)?;

        Ok(())
    }
}

/// Mask an API key for display: first 4 + **** + last 4.
pub fn mask_key(key: &str) -> String {
    if key.len() <= 8 {
        "****".to_string()
    } else {
        format!("{}****{}", &key[..4], &key[key.len() - 4..])
    }
}
