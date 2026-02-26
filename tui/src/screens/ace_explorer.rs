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
    /// How many list items fit on screen — updated each render frame.
    pub visible_count: usize,
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
            visible_count: 20, // sensible default, updated on render
        }
    }

    pub fn current_tab(&self) -> AceExplorerTab {
        AceExplorerTab::ALL[self.active_tab]
    }

    pub fn next_tab(&mut self) {
        self.active_tab = (self.active_tab + 1) % AceExplorerTab::ALL.len();
        self.selected_item = 0;
        self.scroll_offset = 0;
    }

    pub fn prev_tab(&mut self) {
        if self.active_tab == 0 {
            self.active_tab = AceExplorerTab::ALL.len() - 1;
        } else {
            self.active_tab -= 1;
        }
        self.selected_item = 0;
        self.scroll_offset = 0;
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

    fn current_item_count(&self) -> usize {
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
}

// ---------------------------------------------------------------------------
// Full-screen render
// ---------------------------------------------------------------------------

/// Render the ace explorer as a full-screen takeover.
pub fn render_full(app: &mut App, frame: &mut Frame, area: Rect) {
    let state = match app.ace_explorer_state.as_mut() {
        Some(s) => s,
        None => return,
    };

    // Layout: tab bar (2) + gap (1) + search box (3) + gap (1) + content (fill) + footer (2).
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2), // tab bar
            Constraint::Length(1), // gap
            Constraint::Length(3), // search box
            Constraint::Length(1), // gap
            Constraint::Min(1),    // content (list + detail)
            Constraint::Length(2), // footer
        ])
        .split(area);

    let tab_area = layout[0];
    let search_area = layout[2];
    let content_area = layout[4];
    let footer_area = layout[5];

    // Update visible_count based on actual screen height.
    // Reserve 2 lines for the scroll indicator when there are more items than fit.
    let max_rows = content_area.height as usize;
    state.visible_count = max_rows.max(1);

    // --- Tab bar ---
    render_tab_bar(state, frame, tab_area);

    // --- Search box ---
    render_search_box(state, frame, search_area);

    // --- Content: horizontal split ---
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

    // --- Footer ---
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
    let controls = if state.search_focused {
        "Type to filter  ·  Esc to clear"
    } else {
        "↑/↓ navigate  ·  / search  ·  Esc back"
    };
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(""),
            Line::from(vec![
                Span::styled(format!("  {controls}"), Style::default().fg(theme::MUTED)),
                Span::styled(
                    format!("  {range_str}"),
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

    // Top border.
    let top = Line::from(vec![
        Span::raw("  "),
        Span::styled(
            format!("┌{}┐", "─".repeat(inner_w + 2)),
            box_style,
        ),
    ]);

    // Content row.
    let mut row: Vec<Span<'static>> = vec![
        Span::raw("  "),
        Span::styled("│ ⌕ ", box_style),
    ];
    let icon_len = 2; // "⌕ "
    if state.search_focused {
        if state.search.is_empty() {
            row.push(Span::styled("█", Style::default().fg(theme::SUBTLE)));
            row.push(Span::raw(
                " ".repeat(inner_w.saturating_sub(icon_len + 1)),
            ));
        } else {
            row.push(Span::styled(
                state.search.clone(),
                Style::default().fg(theme::TEXT),
            ));
            row.push(Span::styled("█", Style::default().fg(theme::SUBTLE)));
            row.push(Span::raw(
                " ".repeat(
                    inner_w.saturating_sub(icon_len + state.search.len() + 1),
                ),
            ));
        }
    } else if !state.search.is_empty() {
        row.push(Span::styled(
            state.search.clone(),
            Style::default().fg(theme::TEXT),
        ));
        row.push(Span::raw(
            " ".repeat(inner_w.saturating_sub(icon_len + state.search.len())),
        ));
    } else {
        row.push(Span::styled(
            placeholder.to_string(),
            Style::default().fg(theme::SUBTLE),
        ));
        row.push(Span::raw(
            " ".repeat(inner_w.saturating_sub(icon_len + placeholder.len())),
        ));
    }
    row.push(Span::styled(" │", box_style));
    let mid = Line::from(row);

    // Bottom border.
    let bot = Line::from(vec![
        Span::raw("  "),
        Span::styled(
            format!("└{}┘", "─".repeat(inner_w + 2)),
            box_style,
        ),
    ]);

    frame.render_widget(Paragraph::new(vec![top, mid, bot]), area);
}

// ---------------------------------------------------------------------------
// Content builders — return (left_lines, right_lines)
// ---------------------------------------------------------------------------

fn build_variables_content(
    state: &AceExplorerState,
    max_visible: usize,
) -> (Vec<Line<'static>>, Vec<Line<'static>>) {
    let bold = Modifier::BOLD;
    let filtered = state.filtered_variables();
    let mut left: Vec<Line<'static>> = Vec::new();
    let mut right: Vec<Line<'static>> = Vec::new();

    if filtered.is_empty() {
        let msg = if state.search.is_empty() {
            "  No variables defined"
        } else {
            "  No matching variables"
        };
        left.push(Line::from(Span::styled(
            msg,
            Style::default().fg(theme::SUBTLE),
        )));
        return (left, right);
    }

    let total = filtered.len();
    let end = (state.scroll_offset + max_visible).min(total);
    let visible = &filtered[state.scroll_offset..end];

    for (row, (name, _)) in visible.iter().enumerate() {
        let abs_idx = state.scroll_offset + row;
        let is_selected = abs_idx == state.selected_item;
        let indicator = if is_selected { "  > " } else { "    " };
        let indicator_style = if is_selected {
            Style::default().fg(theme::IRIS).add_modifier(bold)
        } else {
            Style::default()
        };
        let name_style = if is_selected {
            Style::default().fg(theme::FOAM).add_modifier(bold)
        } else {
            Style::default().fg(theme::FOAM)
        };
        left.push(Line::from(vec![
            Span::styled(indicator, indicator_style),
            Span::styled(name.clone(), name_style),
        ]));
    }

    // Right: detail of selected item.
    if let Some((name, var)) = filtered.get(state.selected_item) {
        right.push(Line::from(Span::styled(
            name.clone(),
            Style::default().fg(theme::FOAM).add_modifier(bold),
        )));
        right.push(Line::from(""));

        if let Some(desc) = &var.description {
            right.push(Line::from(Span::styled(
                desc.clone(),
                Style::default().fg(theme::TEXT),
            )));
            right.push(Line::from(""));
        }

        if let Some(allowed) = &var.allowed_values {
            right.push(Line::from(Span::styled(
                "ALLOWED VALUES",
                Style::default().fg(theme::SUBTLE).add_modifier(bold),
            )));
            for v in allowed {
                right.push(Line::from(Span::styled(
                    format!("  {}", yaml_value_str(v)),
                    Style::default().fg(theme::TEXT),
                )));
            }
        }
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

    if filtered.is_empty() {
        let msg = if state.search.is_empty() {
            "  No policies defined"
        } else {
            "  No matching policies"
        };
        left.push(Line::from(Span::styled(
            msg,
            Style::default().fg(theme::SUBTLE),
        )));
        return (left, right);
    }

    let total = filtered.len();
    let end = (state.scroll_offset + max_visible).min(total);
    let visible = &filtered[state.scroll_offset..end];

    for (row, (name, _)) in visible.iter().enumerate() {
        let abs_idx = state.scroll_offset + row;
        let is_selected = abs_idx == state.selected_item;
        let indicator = if is_selected { "  > " } else { "    " };
        let indicator_style = if is_selected {
            Style::default().fg(theme::IRIS).add_modifier(bold)
        } else {
            Style::default()
        };
        let name_style = if is_selected {
            Style::default().fg(theme::IRIS).add_modifier(bold)
        } else {
            Style::default().fg(theme::IRIS)
        };
        left.push(Line::from(vec![
            Span::styled(indicator, indicator_style),
            Span::styled(name.clone(), name_style),
        ]));
    }

    // Right: detail of selected item.
    if let Some((name, pol)) = filtered.get(state.selected_item) {
        right.push(Line::from(Span::styled(
            name.clone(),
            Style::default().fg(theme::IRIS).add_modifier(bold),
        )));
        right.push(Line::from(""));

        if let Some(cond) = &pol.condition {
            right.push(Line::from(Span::styled(
                "CONDITION",
                Style::default().fg(theme::SUBTLE).add_modifier(bold),
            )));
            right.push(Line::from(Span::styled(
                cond.clone(),
                Style::default().fg(theme::TEXT),
            )));
            right.push(Line::from(""));
        }
        if let Some(action) = &pol.action {
            right.push(Line::from(Span::styled(
                "ACTION",
                Style::default().fg(theme::SUBTLE).add_modifier(bold),
            )));
            right.push(Line::from(Span::styled(
                action.clone(),
                Style::default().fg(theme::TEXT),
            )));
            right.push(Line::from(""));
        }
        if let Some(provides) = &pol.provides {
            right.push(Line::from(Span::styled(
                "PROVIDES",
                Style::default().fg(theme::SUBTLE).add_modifier(bold),
            )));
            for p in provides {
                right.push(Line::from(Span::styled(
                    format!("  {p}"),
                    Style::default().fg(theme::TEXT),
                )));
            }
        }
    }

    (left, right)
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
