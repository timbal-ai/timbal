use super::AceExplorerTab;
use crate::model::project::{AceConfig, AcePolicy, AceVariable, EvalCase, EvalsConfig};

use super::widgets::yaml_value_str;

pub struct AceExplorerState {
    pub agent_name: String,
    pub ace: AceConfig,
    pub evals: EvalsConfig,
    pub active_tab: usize,
    pub selected_item: usize,
    pub search: String,
    pub search_focused: bool,
    pub scroll_offset: usize,
    pub visible_count: usize,
    /// Vertical scroll offset for the right panel (used on read-only Evals tab).
    pub right_scroll: u16,
    /// X coordinate where the right panel starts (set during render).
    pub right_panel_x: u16,
    /// Maximum value for right_scroll (set during render).
    pub right_max_scroll: u16,
    // --- Editing state ---
    /// Focus is on the right detail panel.
    pub editing_right: bool,
    /// Which field in the right panel is selected (0-indexed).
    pub editing_field: usize,
    /// Currently editing the text of a field.
    pub field_editing: bool,
    /// Input buffer for the field being edited.
    pub field_input: String,
    /// Cursor position within field_input (byte offset).
    pub field_cursor: usize,
    /// Original value before editing (for cancel/revert).
    pub field_original: String,
    /// Whether the ace config has been modified (needs save).
    pub dirty: bool,
    // --- Playground state ---
    /// The import spec for this agent (e.g. "workforce/agent-name/agent.py::agent").
    pub import_spec: Option<String>,
    /// Input buffer for the playground prompt.
    pub playground_input: String,
    /// Cursor position within playground_input.
    pub playground_cursor: usize,
    /// Whether the playground input is focused (typing mode).
    pub playground_focused: bool,
    /// Whether a process is currently running.
    pub playground_running: bool,
    // Multi-chat
    pub playground_chats: Vec<PlaygroundChat>,
    pub playground_active_chat: usize,
    pub playground_chat_counter: usize,
    // Panel focus & scroll
    pub playground_panel: PlaygroundPanel,
    pub playground_feed_scroll: u16,
    pub playground_feed_max_scroll: u16,
    pub playground_chats_scroll: usize,
    pub playground_config_scroll: u16,
    pub playground_config_max_scroll: u16,
    // Panel x-boundaries (set during render)
    pub playground_feed_x: u16,
    pub playground_config_x: u16,
    // Config overrides (empty = use agent defaults)
    pub playground_cfg_model: String,
    pub playground_cfg_system_prompt: String,
    // Config editing
    pub playground_cfg_field: usize,
    pub playground_cfg_editing: bool,
    pub playground_cfg_input: String,
    pub playground_cfg_cursor: usize,
}

/// A single piece of playground output.
#[derive(Debug, Clone)]
pub enum PlaygroundLine {
    /// The user's prompt (displayed with "❯" indicator).
    User(String),
    /// A complete line of text (displayed on its own line).
    Text(String),
    /// A streaming text delta — appended to the last Text line.
    TextDelta(String),
    /// An error line (stderr).
    Error(String),
    /// A status/info line (e.g. tool use).
    Status(String),
    /// Thinking indicator — replaced when thinking ends.
    Thinking,
    /// Run stats: duration_ms, usage summary string.
    Stats {
        duration_ms: u64,
        usage: String,
    },
}

/// A single chat session in the playground.
pub struct PlaygroundChat {
    pub id: usize,
    pub label: String,
    pub output: Vec<PlaygroundLine>,
}

/// Which panel is currently focused in the playground.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaygroundPanel {
    Chats,
    Feed,
    Config,
}

impl PlaygroundPanel {
    pub fn next(self) -> Self {
        match self {
            Self::Chats => Self::Feed,
            Self::Feed => Self::Config,
            Self::Config => Self::Chats,
        }
    }

    pub fn prev(self) -> Self {
        match self {
            Self::Chats => Self::Config,
            Self::Feed => Self::Chats,
            Self::Config => Self::Feed,
        }
    }
}

pub const PLAYGROUND_CFG_FIELD_COUNT: usize = 2;

impl AceExplorerState {
    pub fn new(
        agent_name: String,
        ace: AceConfig,
        evals: EvalsConfig,
        import_spec: Option<String>,
    ) -> Self {
        Self {
            agent_name,
            ace,
            evals,
            active_tab: 0,
            selected_item: 0,
            search: String::new(),
            search_focused: false,
            scroll_offset: 0,
            visible_count: 20,
            right_scroll: 0,
            right_panel_x: 0,
            right_max_scroll: 0,
            editing_right: false,
            editing_field: 0,
            field_editing: false,
            field_input: String::new(),
            field_cursor: 0,
            field_original: String::new(),
            dirty: false,
            import_spec,
            playground_input: String::new(),
            playground_cursor: 0,
            playground_focused: false,
            playground_running: false,
            playground_chats: vec![PlaygroundChat {
                id: 1,
                label: "Chat 1".to_string(),
                output: Vec::new(),
            }],
            playground_active_chat: 0,
            playground_chat_counter: 1,
            playground_panel: PlaygroundPanel::Feed,
            playground_feed_scroll: 0,
            playground_feed_max_scroll: 0,
            playground_chats_scroll: 0,
            playground_config_scroll: 0,
            playground_config_max_scroll: 0,
            playground_feed_x: 0,
            playground_config_x: 0,
            playground_cfg_model: String::new(),
            playground_cfg_system_prompt: String::new(),
            playground_cfg_field: 0,
            playground_cfg_editing: false,
            playground_cfg_input: String::new(),
            playground_cfg_cursor: 0,
        }
    }

    pub fn current_tab(&self) -> AceExplorerTab {
        AceExplorerTab::ALL[self.active_tab]
    }

    pub fn next_tab(&mut self) {
        self.active_tab = (self.active_tab + 1) % AceExplorerTab::ALL.len();
        self.selected_item = 0;
        self.scroll_offset = 0;
        self.right_scroll = 0;
        self.editing_right = false;
        self.field_editing = false;
    }

    pub fn prev_tab(&mut self) {
        if self.active_tab == 0 {
            self.active_tab = AceExplorerTab::ALL.len() - 1;
        } else {
            self.active_tab -= 1;
        }
        self.selected_item = 0;
        self.scroll_offset = 0;
        self.right_scroll = 0;
        self.editing_right = false;
        self.field_editing = false;
    }

    pub fn filtered_variables(&self) -> Vec<&(String, AceVariable)> {
        let query = self.search.to_lowercase();
        self.ace
            .variables
            .iter()
            .filter(|(name, var)| {
                if query.is_empty() {
                    return true;
                }
                name.to_lowercase().contains(&query)
                    || var
                        .description
                        .as_deref()
                        .unwrap_or("")
                        .to_lowercase()
                        .contains(&query)
            })
            .collect()
    }

    pub fn filtered_policies(&self) -> Vec<&(String, AcePolicy)> {
        let query = self.search.to_lowercase();
        self.ace
            .policies
            .iter()
            .filter(|(name, pol)| {
                if query.is_empty() {
                    return true;
                }
                name.to_lowercase().contains(&query)
                    || pol
                        .condition
                        .as_deref()
                        .unwrap_or("")
                        .to_lowercase()
                        .contains(&query)
                    || pol
                        .action
                        .as_deref()
                        .unwrap_or("")
                        .to_lowercase()
                        .contains(&query)
            })
            .collect()
    }

    /// Flatten all eval cases from all files into a single list.
    pub fn flattened_evals(&self) -> Vec<(&str, &EvalCase)> {
        self.evals
            .files
            .iter()
            .flat_map(|(file, cases)| cases.iter().map(move |c| (file.as_str(), c)))
            .collect()
    }

    pub fn filtered_evals(&self) -> Vec<(&str, &EvalCase)> {
        let query = self.search.to_lowercase();
        self.flattened_evals()
            .into_iter()
            .filter(|(file, eval)| {
                if query.is_empty() {
                    return true;
                }
                eval.name.to_lowercase().contains(&query)
                    || eval
                        .description
                        .as_deref()
                        .unwrap_or("")
                        .to_lowercase()
                        .contains(&query)
                    || eval.tags.iter().any(|t| t.to_lowercase().contains(&query))
                    || file.to_lowercase().contains(&query)
            })
            .collect()
    }

    pub fn current_item_count(&self) -> usize {
        match self.current_tab() {
            AceExplorerTab::Variables => self.filtered_variables().len(),
            AceExplorerTab::Policies => self.filtered_policies().len(),
            AceExplorerTab::Evals => self.filtered_evals().len(),
            AceExplorerTab::Playground => 0,
        }
    }

    pub fn move_down(&mut self) {
        let count = self.current_item_count();
        if count > 0 {
            self.selected_item = (self.selected_item + 1).min(count - 1);
            if self.selected_item >= self.scroll_offset + self.visible_count {
                self.scroll_offset = self.selected_item + 1 - self.visible_count;
            }
            self.right_scroll = 0;
        }
    }

    pub fn move_up(&mut self) {
        self.selected_item = self.selected_item.saturating_sub(1);
        if self.selected_item < self.scroll_offset {
            self.scroll_offset = self.selected_item;
        }
        self.right_scroll = 0;
    }

    pub fn clamp_selection(&mut self) {
        let count = self.current_item_count();
        if count == 0 {
            self.selected_item = 0;
            self.scroll_offset = 0;
        } else {
            if self.selected_item >= count {
                self.selected_item = count - 1;
            }
            if self.scroll_offset > self.selected_item {
                self.scroll_offset = self.selected_item;
            }
            if self.selected_item >= self.scroll_offset + self.visible_count {
                self.scroll_offset = self.selected_item + 1 - self.visible_count;
            }
        }
    }

    /// Number of editable fields for the current tab.
    pub fn field_count(&self) -> usize {
        match self.current_tab() {
            AceExplorerTab::Variables => 2, // description, allowed_values
            AceExplorerTab::Policies => 3,  // condition, action, provides
            AceExplorerTab::Evals => 0,     // read-only
            AceExplorerTab::Playground => 0,
        }
    }

    /// Enter editing mode for the right panel.
    pub fn enter_right(&mut self) {
        self.editing_right = true;
        self.editing_field = 0;
        self.field_editing = false;
    }

    /// Start editing the current field — populate input buffer.
    pub fn start_field_edit(&mut self) {
        let val = self.get_field_value();
        self.field_original = val.clone();
        self.field_cursor = val.len();
        self.field_input = val;
        self.field_editing = true;
    }

    /// Confirm the field edit — write value back to ace config, upsert timestamps.
    pub fn confirm_field_edit(&mut self) {
        use crate::model::project::{git_identity, now_utc};

        self.set_field_value(self.field_input.clone());
        self.field_editing = false;
        self.field_cursor = 0;
        self.dirty = true;

        // Update updated_at / updated_by on the item (never touch created_at on edit).
        let now = now_utc();
        let who = git_identity();
        if let Some(idx) = self.selected_original_index() {
            match self.current_tab() {
                AceExplorerTab::Variables => {
                    let var = &mut self.ace.variables[idx].1;
                    var.updated_at = Some(now);
                    var.updated_by = Some(who);
                }
                AceExplorerTab::Policies => {
                    let pol = &mut self.ace.policies[idx].1;
                    pol.updated_at = Some(now);
                    pol.updated_by = Some(who);
                }
                AceExplorerTab::Evals | AceExplorerTab::Playground => {}
            }
        }
    }

    /// Cancel the field edit — revert to original.
    pub fn cancel_field_edit(&mut self) {
        self.field_input.clear();
        self.field_cursor = 0;
        self.field_editing = false;
    }

    /// Move cursor left within the field input.
    pub fn field_cursor_left(&mut self) {
        if self.field_cursor > 0 {
            let prev = self.field_input[..self.field_cursor]
                .char_indices()
                .next_back()
                .map(|(i, _)| i)
                .unwrap_or(0);
            self.field_cursor = prev;
        }
    }

    /// Move cursor right within the field input.
    pub fn field_cursor_right(&mut self) {
        if self.field_cursor < self.field_input.len() {
            let next = self.field_input[self.field_cursor..]
                .char_indices()
                .nth(1)
                .map(|(i, _)| self.field_cursor + i)
                .unwrap_or(self.field_input.len());
            self.field_cursor = next;
        }
    }

    /// Insert a character at the cursor position.
    pub fn field_insert(&mut self, c: char) {
        self.field_input.insert(self.field_cursor, c);
        self.field_cursor += c.len_utf8();
    }

    /// Delete the character before the cursor.
    pub fn field_backspace(&mut self) {
        if self.field_cursor > 0 {
            let prev = self.field_input[..self.field_cursor]
                .char_indices()
                .next_back()
                .map(|(i, _)| i)
                .unwrap_or(0);
            self.field_input.drain(prev..self.field_cursor);
            self.field_cursor = prev;
        }
    }

    /// Get the selected item's index in the original (unfiltered) ace list.
    fn selected_original_index(&self) -> Option<usize> {
        let filtered_names: Vec<&str> = match self.current_tab() {
            AceExplorerTab::Variables => self
                .filtered_variables()
                .iter()
                .map(|(n, _)| n.as_str())
                .collect(),
            AceExplorerTab::Policies => self
                .filtered_policies()
                .iter()
                .map(|(n, _)| n.as_str())
                .collect(),
            AceExplorerTab::Evals | AceExplorerTab::Playground => return None,
        };
        let selected_name = filtered_names.get(self.selected_item)?;
        match self.current_tab() {
            AceExplorerTab::Variables => self
                .ace
                .variables
                .iter()
                .position(|(n, _)| n == selected_name),
            AceExplorerTab::Policies => self
                .ace
                .policies
                .iter()
                .position(|(n, _)| n == selected_name),
            AceExplorerTab::Evals | AceExplorerTab::Playground => None,
        }
    }

    /// Get the current field's value as a string.
    fn get_field_value(&self) -> String {
        let orig_idx = match self.selected_original_index() {
            Some(i) => i,
            None => return String::new(),
        };
        match self.current_tab() {
            AceExplorerTab::Variables => {
                let var = &self.ace.variables[orig_idx].1;
                match self.editing_field {
                    0 => var.description.clone().unwrap_or_default(),
                    1 => var
                        .allowed_values
                        .as_ref()
                        .map(|vals| {
                            vals.iter()
                                .map(|v| yaml_value_str(v))
                                .collect::<Vec<_>>()
                                .join(", ")
                        })
                        .unwrap_or_default(),
                    _ => String::new(),
                }
            }
            AceExplorerTab::Policies => {
                let pol = &self.ace.policies[orig_idx].1;
                match self.editing_field {
                    0 => pol.condition.clone().unwrap_or_default(),
                    1 => pol.action.clone().unwrap_or_default(),
                    2 => pol
                        .provides
                        .as_ref()
                        .map(|v| v.join(", "))
                        .unwrap_or_default(),
                    _ => String::new(),
                }
            }
            AceExplorerTab::Evals | AceExplorerTab::Playground => String::new(),
        }
    }

    /// Set the current field's value from a string.
    fn set_field_value(&mut self, val: String) {
        let orig_idx = match self.selected_original_index() {
            Some(i) => i,
            None => return,
        };
        match self.current_tab() {
            AceExplorerTab::Variables => {
                let var = &mut self.ace.variables[orig_idx].1;
                match self.editing_field {
                    0 => {
                        var.description = if val.is_empty() { None } else { Some(val) };
                    }
                    1 => {
                        if val.trim().is_empty() {
                            var.allowed_values = None;
                        } else {
                            let vals: Vec<serde_yaml::Value> = val
                                .split(',')
                                .map(|s| serde_yaml::Value::String(s.trim().to_string()))
                                .collect();
                            var.allowed_values = Some(vals);
                        }
                    }
                    _ => {}
                }
            }
            AceExplorerTab::Policies => {
                let pol = &mut self.ace.policies[orig_idx].1;
                match self.editing_field {
                    0 => {
                        pol.condition = if val.is_empty() { None } else { Some(val) };
                    }
                    1 => {
                        pol.action = if val.is_empty() { None } else { Some(val) };
                    }
                    2 => {
                        if val.trim().is_empty() {
                            pol.provides = None;
                        } else {
                            pol.provides =
                                Some(val.split(',').map(|s| s.trim().to_string()).collect());
                        }
                    }
                    _ => {}
                }
            }
            AceExplorerTab::Evals | AceExplorerTab::Playground => {}
        }
    }

    // --- Playground helpers ---

    /// Get the output of the active chat.
    pub fn playground_output(&self) -> &[PlaygroundLine] {
        if let Some(chat) = self.playground_chats.get(self.playground_active_chat) {
            &chat.output
        } else {
            &[]
        }
    }

    /// Push a line to the active chat's output.
    pub fn playground_push_output(&mut self, line: PlaygroundLine) {
        if let Some(chat) = self.playground_chats.get_mut(self.playground_active_chat) {
            match &line {
                PlaygroundLine::TextDelta(delta) => {
                    // Remove any Thinking placeholder before appending text.
                    if matches!(chat.output.last(), Some(PlaygroundLine::Thinking)) {
                        chat.output.pop();
                    }
                    // Append to the last Text line, or create one.
                    if let Some(PlaygroundLine::Text(existing)) = chat.output.last_mut() {
                        existing.push_str(delta);
                    } else {
                        chat.output.push(PlaygroundLine::Text(delta.clone()));
                    }
                }
                PlaygroundLine::Thinking => {
                    // Only push if not already thinking.
                    if !matches!(chat.output.last(), Some(PlaygroundLine::Thinking)) {
                        chat.output.push(line);
                    }
                }
                _ => {
                    // Remove Thinking placeholder when real content arrives.
                    if matches!(chat.output.last(), Some(PlaygroundLine::Thinking)) {
                        chat.output.pop();
                    }
                    chat.output.push(line);
                }
            }
        }
    }

    /// Create a new chat and make it active.
    pub fn playground_new_chat(&mut self) {
        self.playground_chat_counter += 1;
        let id = self.playground_chat_counter;
        self.playground_chats.push(PlaygroundChat {
            id,
            label: format!("Chat {id}"),
            output: Vec::new(),
        });
        self.playground_active_chat = self.playground_chats.len() - 1;
        self.playground_feed_scroll = 0;
    }

    /// Get the config field value by index.
    pub fn playground_cfg_value(&self, idx: usize) -> &str {
        match idx {
            0 => &self.playground_cfg_model,
            1 => &self.playground_cfg_system_prompt,
            _ => "",
        }
    }

    /// Set the config field value by index.
    pub fn playground_set_cfg_value(&mut self, idx: usize, val: String) {
        match idx {
            0 => self.playground_cfg_model = val,
            1 => self.playground_cfg_system_prompt = val,
            _ => {}
        }
    }

    pub const PLAYGROUND_CFG_LABELS: &'static [&'static str] =
        &["MODEL", "SYSTEM PROMPT"];

    /// Start editing a config field.
    pub fn playground_start_cfg_edit(&mut self) {
        let val = self.playground_cfg_value(self.playground_cfg_field).to_string();
        self.playground_cfg_cursor = val.len();
        self.playground_cfg_input = val;
        self.playground_cfg_editing = true;
    }

    /// Confirm config field edit.
    pub fn playground_confirm_cfg_edit(&mut self) {
        let val = self.playground_cfg_input.clone();
        self.playground_set_cfg_value(self.playground_cfg_field, val);
        self.playground_cfg_editing = false;
        self.playground_cfg_cursor = 0;
    }

    /// Cancel config field edit.
    pub fn playground_cancel_cfg_edit(&mut self) {
        self.playground_cfg_input.clear();
        self.playground_cfg_cursor = 0;
        self.playground_cfg_editing = false;
    }
}
