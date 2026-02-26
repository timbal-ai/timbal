use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Paragraph, Wrap},
    Frame,
};

use crate::app::App;
use crate::model::project::{AceConfig, AcePolicy, AceVariable};
use crate::theme;

// ---------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AceExplorerTab {
    Variables,
    Policies,
}

impl AceExplorerTab {
    pub const ALL: &'static [AceExplorerTab] = &[
        AceExplorerTab::Variables,
        AceExplorerTab::Policies,
    ];

    pub fn label(self) -> &'static str {
        match self {
            AceExplorerTab::Variables => "Variables",
            AceExplorerTab::Policies => "Policies",
        }
    }
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub struct AceExplorerState {
    pub agent_name: String,
    pub ace: AceConfig,
    pub active_tab: usize,
    pub selected_item: usize,
    pub search: String,
    pub search_focused: bool,
    pub scroll_offset: usize,
    pub visible_count: usize,
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
}

impl AceExplorerState {
    pub fn new(agent_name: String, ace: AceConfig) -> Self {
        Self {
            agent_name,
            ace,
            active_tab: 0,
            selected_item: 0,
            search: String::new(),
            search_focused: false,
            scroll_offset: 0,
            visible_count: 20,
            editing_right: false,
            editing_field: 0,
            field_editing: false,
            field_input: String::new(),
            field_cursor: 0,
            field_original: String::new(),
            dirty: false,
        }
    }

    pub fn current_tab(&self) -> AceExplorerTab {
        AceExplorerTab::ALL[self.active_tab]
    }

    pub fn next_tab(&mut self) {
        self.active_tab = (self.active_tab + 1) % AceExplorerTab::ALL.len();
        self.selected_item = 0;
        self.scroll_offset = 0;
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

    pub fn current_item_count(&self) -> usize {
        match self.current_tab() {
            AceExplorerTab::Variables => self.filtered_variables().len(),
            AceExplorerTab::Policies => self.filtered_policies().len(),
        }
    }

    pub fn move_down(&mut self) {
        let count = self.current_item_count();
        if count > 0 {
            self.selected_item = (self.selected_item + 1).min(count - 1);
            if self.selected_item >= self.scroll_offset + self.visible_count {
                self.scroll_offset = self.selected_item + 1 - self.visible_count;
            }
        }
    }

    pub fn move_up(&mut self) {
        self.selected_item = self.selected_item.saturating_sub(1);
        if self.selected_item < self.scroll_offset {
            self.scroll_offset = self.selected_item;
        }
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
            // Move to previous char boundary.
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
            AceExplorerTab::Variables => {
                self.filtered_variables().iter().map(|(n, _)| n.as_str()).collect()
            }
            AceExplorerTab::Policies => {
                self.filtered_policies().iter().map(|(n, _)| n.as_str()).collect()
            }
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
                            vals.iter().map(|v| yaml_value_str(v)).collect::<Vec<_>>().join(", ")
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
                            pol.provides = Some(
                                val.split(',').map(|s| s.trim().to_string()).collect(),
                            );
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Full-screen render
// ---------------------------------------------------------------------------

pub fn render_full(app: &mut App, frame: &mut Frame, area: Rect) {
    let state = match app.ace_explorer_state.as_mut() {
        Some(s) => s,
        None => return,
    };

    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2), // tab bar
            Constraint::Length(1), // gap
            Constraint::Length(3), // search box
            Constraint::Length(1), // gap
            Constraint::Min(1),    // content
            Constraint::Length(2), // footer
        ])
        .split(area);

    let tab_area = layout[0];
    let search_area = layout[2];
    let content_area = layout[4];
    let footer_area = layout[5];

    let max_rows = content_area.height as usize;
    state.visible_count = max_rows.max(1);

    render_tab_bar(state, frame, tab_area);
    render_search_box(state, frame, search_area);

    // Content: horizontal split.
    let horiz = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(42), Constraint::Percentage(58)])
        .split(content_area);

    let visible = state.visible_count;
    let (left_lines, right_lines) = match state.current_tab() {
        AceExplorerTab::Variables => build_variables_content(state, visible),
        AceExplorerTab::Policies => build_policies_content(state, visible),
    };

    frame.render_widget(Paragraph::new(left_lines), horiz[0]);
    frame.render_widget(
        Paragraph::new(right_lines).wrap(Wrap { trim: false }),
        horiz[1],
    );

    // Footer.
    let total_items = match state.current_tab() {
        AceExplorerTab::Variables => state.filtered_variables().len(),
        AceExplorerTab::Policies => state.filtered_policies().len(),
    };
    let range_str = if total_items > 0 {
        let start = state.scroll_offset + 1;
        let end = (state.scroll_offset + visible).min(total_items);
        format!("{start}-{end} of {total_items}")
    } else {
        "0 items".to_string()
    };
    let controls = if state.field_editing {
        "Type to edit  ·  Enter confirm  ·  Esc cancel"
    } else if state.editing_right {
        "↑/↓ fields  ·  Enter edit  ·  Esc back to list"
    } else if state.search_focused {
        "Type to filter  ·  Esc to clear"
    } else {
        "↑/↓ navigate  ·  Enter edit  ·  / search  ·  Esc back"
    };
    let dirty_indicator = if state.dirty { "  [modified]" } else { "" };
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(""),
            Line::from(vec![
                Span::styled(format!("  {controls}"), Style::default().fg(theme::MUTED)),
                Span::styled(
                    format!("  {range_str}{dirty_indicator}"),
                    Style::default().fg(theme::SUBTLE),
                ),
            ]),
        ]),
        footer_area,
    );
}

// ---------------------------------------------------------------------------
// Sub-renders
// ---------------------------------------------------------------------------

fn render_tab_bar(state: &AceExplorerState, frame: &mut Frame, area: Rect) {
    let bold = Modifier::BOLD;
    let mut tab_bar: Vec<Span<'static>> = vec![
        Span::styled(
            format!("  ACE: {}", state.agent_name),
            Style::default().fg(theme::GOLD).add_modifier(bold),
        ),
        Span::raw("  "),
    ];
    for (i, tab) in AceExplorerTab::ALL.iter().enumerate() {
        if i == state.active_tab {
            let bg = match tab {
                AceExplorerTab::Variables => theme::FOAM,
                AceExplorerTab::Policies => theme::IRIS,
            };
            tab_bar.push(Span::styled(
                format!(" {} ", tab.label()),
                Style::default()
                    .fg(theme::BASE)
                    .bg(bg)
                    .add_modifier(bold),
            ));
        } else {
            tab_bar.push(Span::styled(
                format!("  {}  ", tab.label()),
                Style::default().fg(theme::SUBTLE),
            ));
        }
    }
    tab_bar.push(Span::styled(
        "  (←/→ or tab to cycle)",
        Style::default().fg(theme::MUTED),
    ));

    frame.render_widget(
        Paragraph::new(vec![Line::from(""), Line::from(tab_bar)]),
        area,
    );
}

fn render_search_box(state: &AceExplorerState, frame: &mut Frame, area: Rect) {
    let placeholder = match state.current_tab() {
        AceExplorerTab::Variables => "Search variables...",
        AceExplorerTab::Policies => "Search policies...",
    };
    let box_color = if state.search_focused {
        theme::IRIS
    } else {
        theme::SUBTLE
    };
    let box_style = Style::default().fg(box_color);
    let inner_w = (area.width as usize).saturating_sub(6);

    let top = Line::from(vec![
        Span::raw("  "),
        Span::styled(format!("┌{}┐", "─".repeat(inner_w + 2)), box_style),
    ]);

    let mut row: Vec<Span<'static>> = vec![
        Span::raw("  "),
        Span::styled("│ ⌕ ", box_style),
    ];
    let icon_len = 2;
    let cursor_style = Style::default().fg(theme::BASE).bg(theme::TEXT);
    if state.search_focused {
        if state.search.is_empty() {
            row.push(Span::styled(" ", cursor_style));
            row.push(Span::raw(" ".repeat(inner_w.saturating_sub(icon_len + 1))));
        } else {
            // Cursor at end of search text — highlight trailing space.
            row.push(Span::styled(state.search.clone(), Style::default().fg(theme::TEXT)));
            row.push(Span::styled(" ", cursor_style));
            row.push(Span::raw(
                " ".repeat(inner_w.saturating_sub(icon_len + state.search.len() + 1)),
            ));
        }
    } else if !state.search.is_empty() {
        row.push(Span::styled(state.search.clone(), Style::default().fg(theme::TEXT)));
        row.push(Span::raw(
            " ".repeat(inner_w.saturating_sub(icon_len + state.search.len())),
        ));
    } else {
        row.push(Span::styled(placeholder.to_string(), Style::default().fg(theme::SUBTLE)));
        row.push(Span::raw(
            " ".repeat(inner_w.saturating_sub(icon_len + placeholder.len())),
        ));
    }
    row.push(Span::styled(" │", box_style));
    let mid = Line::from(row);

    let bot = Line::from(vec![
        Span::raw("  "),
        Span::styled(format!("└{}┘", "─".repeat(inner_w + 2)), box_style),
    ]);

    frame.render_widget(Paragraph::new(vec![top, mid, bot]), area);
}

// ---------------------------------------------------------------------------
// Content builders
// ---------------------------------------------------------------------------

fn build_variables_content(
    state: &AceExplorerState,
    max_visible: usize,
) -> (Vec<Line<'static>>, Vec<Line<'static>>) {
    let bold = Modifier::BOLD;
    let filtered = state.filtered_variables();
    let mut left: Vec<Line<'static>> = Vec::new();
    let mut right: Vec<Line<'static>> = Vec::new();
    let left_dimmed = state.editing_right;

    if filtered.is_empty() {
        let msg = if state.search.is_empty() {
            "  No variables defined"
        } else {
            "  No matching variables"
        };
        left.push(Line::from(Span::styled(msg, Style::default().fg(theme::SUBTLE))));
        return (left, right);
    }

    let total = filtered.len();
    let end = (state.scroll_offset + max_visible).min(total);
    let visible = &filtered[state.scroll_offset..end];

    for (row, (name, _)) in visible.iter().enumerate() {
        let abs_idx = state.scroll_offset + row;
        let is_selected = abs_idx == state.selected_item;
        let indicator = if is_selected { "  > " } else { "    " };
        let (ind_style, name_style) = if left_dimmed {
            (
                Style::default().fg(theme::MUTED),
                if is_selected {
                    Style::default().fg(theme::MUTED)
                } else {
                    Style::default().fg(theme::MUTED)
                },
            )
        } else {
            (
                if is_selected {
                    Style::default().fg(theme::IRIS).add_modifier(bold)
                } else {
                    Style::default()
                },
                if is_selected {
                    Style::default().fg(theme::FOAM).add_modifier(bold)
                } else {
                    Style::default().fg(theme::FOAM)
                },
            )
        };
        left.push(Line::from(vec![
            Span::styled(indicator, ind_style),
            Span::styled(name.clone(), name_style),
        ]));
    }

    // Right panel.
    if let Some((name, var)) = filtered.get(state.selected_item) {
        right.push(Line::from(Span::styled(
            name.clone(),
            Style::default().fg(theme::FOAM).add_modifier(bold),
        )));
        right.push(Line::from(""));

        // Field 0: description
        let desc_val = var.description.clone().unwrap_or_default();
        build_field_lines(state, &mut right, 0, "DESCRIPTION", &desc_val);
        right.push(Line::from(""));

        // Field 1: allowed_values
        let allowed_val = var
            .allowed_values
            .as_ref()
            .map(|vals| vals.iter().map(|v| yaml_value_str(v)).collect::<Vec<_>>().join(", "))
            .unwrap_or_default();
        build_field_lines(state, &mut right, 1, "ALLOWED VALUES", &allowed_val);

        // Metadata (read-only).
        build_metadata_lines(
            &mut right,
            var.created_at.as_deref(),
            var.updated_at.as_deref(),
            var.updated_by.as_deref(),
        );
    }

    (left, right)
}

fn build_policies_content(
    state: &AceExplorerState,
    max_visible: usize,
) -> (Vec<Line<'static>>, Vec<Line<'static>>) {
    let bold = Modifier::BOLD;
    let filtered = state.filtered_policies();
    let mut left: Vec<Line<'static>> = Vec::new();
    let mut right: Vec<Line<'static>> = Vec::new();
    let left_dimmed = state.editing_right;

    if filtered.is_empty() {
        let msg = if state.search.is_empty() {
            "  No policies defined"
        } else {
            "  No matching policies"
        };
        left.push(Line::from(Span::styled(msg, Style::default().fg(theme::SUBTLE))));
        return (left, right);
    }

    let total = filtered.len();
    let end = (state.scroll_offset + max_visible).min(total);
    let visible = &filtered[state.scroll_offset..end];

    for (row, (name, _)) in visible.iter().enumerate() {
        let abs_idx = state.scroll_offset + row;
        let is_selected = abs_idx == state.selected_item;
        let indicator = if is_selected { "  > " } else { "    " };
        let (ind_style, name_style) = if left_dimmed {
            (
                Style::default().fg(theme::MUTED),
                Style::default().fg(theme::MUTED),
            )
        } else {
            (
                if is_selected {
                    Style::default().fg(theme::IRIS).add_modifier(bold)
                } else {
                    Style::default()
                },
                if is_selected {
                    Style::default().fg(theme::IRIS).add_modifier(bold)
                } else {
                    Style::default().fg(theme::IRIS)
                },
            )
        };
        left.push(Line::from(vec![
            Span::styled(indicator, ind_style),
            Span::styled(name.clone(), name_style),
        ]));
    }

    // Right panel.
    if let Some((name, pol)) = filtered.get(state.selected_item) {
        right.push(Line::from(Span::styled(
            name.clone(),
            Style::default().fg(theme::IRIS).add_modifier(bold),
        )));
        right.push(Line::from(""));

        // Field 0: condition
        let cond_val = pol.condition.clone().unwrap_or_default();
        build_field_lines(state, &mut right, 0, "CONDITION", &cond_val);
        right.push(Line::from(""));

        // Field 1: action
        let action_val = pol.action.clone().unwrap_or_default();
        build_field_lines(state, &mut right, 1, "ACTION", &action_val);
        right.push(Line::from(""));

        // Field 2: provides
        let provides_val = pol
            .provides
            .as_ref()
            .map(|v| v.join(", "))
            .unwrap_or_default();
        build_field_lines(state, &mut right, 2, "PROVIDES", &provides_val);

        // Metadata (read-only).
        build_metadata_lines(
            &mut right,
            pol.created_at.as_deref(),
            pol.updated_at.as_deref(),
            pol.updated_by.as_deref(),
        );
    }

    (left, right)
}

/// Render read-only metadata at the bottom of the right panel.
fn build_metadata_lines(
    lines: &mut Vec<Line<'static>>,
    created_at: Option<&str>,
    updated_at: Option<&str>,
    updated_by: Option<&str>,
) {
    // Only show if at least one field exists.
    if created_at.is_none() && updated_at.is_none() && updated_by.is_none() {
        return;
    }

    let meta = Style::default().fg(theme::MUTED);
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled("─".repeat(20), meta)));

    if let Some(at) = created_at {
        lines.push(Line::from(vec![
            Span::styled("created  ", meta),
            Span::styled(at.to_string(), meta),
        ]));
    }
    if let Some(at) = updated_at {
        lines.push(Line::from(vec![
            Span::styled("updated  ", meta),
            Span::styled(at.to_string(), meta),
        ]));
    }
    if let Some(by) = updated_by {
        lines.push(Line::from(vec![
            Span::styled("by       ", meta),
            Span::styled(by.to_string(), meta),
        ]));
    }
}

/// Render a single editable field in the right panel.
fn build_field_lines(
    state: &AceExplorerState,
    lines: &mut Vec<Line<'static>>,
    field_idx: usize,
    label: &str,
    value: &str,
) {
    let bold = Modifier::BOLD;
    let is_active = state.editing_right && state.editing_field == field_idx;
    let is_typing = is_active && state.field_editing;

    // Label line.
    let indicator = if is_active { "> " } else { "  " };
    let label_style = if is_active {
        Style::default().fg(theme::GOLD).add_modifier(bold)
    } else {
        Style::default().fg(theme::SUBTLE).add_modifier(bold)
    };
    lines.push(Line::from(vec![
        Span::styled(indicator, Style::default().fg(theme::IRIS)),
        Span::styled(label.to_string(), label_style),
    ]));

    // Value line.
    if is_typing {
        // Show editable input with cursor highlighting the character at position.
        let cursor = state.field_cursor;
        let before = &state.field_input[..cursor];
        // Extract the character under the cursor (or space if at end).
        let (cursor_ch, after) = if cursor < state.field_input.len() {
            let rest = &state.field_input[cursor..];
            let ch_len = rest.chars().next().map(|c| c.len_utf8()).unwrap_or(0);
            (
                state.field_input[cursor..cursor + ch_len].to_string(),
                state.field_input[cursor + ch_len..].to_string(),
            )
        } else {
            (" ".to_string(), String::new())
        };
        let cursor_style = Style::default().fg(theme::BASE).bg(theme::TEXT);
        lines.push(Line::from(vec![
            Span::styled("  ", Style::default()),
            Span::styled(before.to_string(), Style::default().fg(theme::TEXT)),
            Span::styled(cursor_ch, cursor_style),
            Span::styled(after, Style::default().fg(theme::TEXT)),
        ]));
    } else {
        let display = if value.is_empty() {
            "(empty)".to_string()
        } else {
            value.to_string()
        };
        let val_style = if value.is_empty() {
            Style::default().fg(theme::MUTED)
        } else {
            Style::default().fg(theme::TEXT)
        };
        lines.push(Line::from(vec![
            Span::styled("  ", Style::default()),
            Span::styled(display, val_style),
        ]));
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn yaml_value_str(v: &serde_yaml::Value) -> String {
    match v {
        serde_yaml::Value::String(s) => s.clone(),
        serde_yaml::Value::Number(n) => n.to_string(),
        serde_yaml::Value::Bool(b) => b.to_string(),
        serde_yaml::Value::Null => "null".to_string(),
        other => format!("{other:?}"),
    }
}
