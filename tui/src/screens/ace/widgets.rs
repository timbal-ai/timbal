use ratatui::{
    Frame,
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::Paragraph,
};

use super::AceExplorerTab;
use super::state::AceExplorerState;
use crate::theme;

pub fn render_tab_bar(state: &AceExplorerState, frame: &mut Frame, area: Rect) {
    let bold = Modifier::BOLD;

    // "ACE vX.Y.Z" branding fills the left portion (15% = chats panel width).
    let left_cols = (area.width as usize * 15) / 100;
    let version = env!("CARGO_PKG_VERSION");
    let branding_len = 2 + 3 + 2 + 1 + version.len(); // "  ACE  v" + version
    let pad = left_cols.saturating_sub(branding_len);

    let mut tab_bar: Vec<Span<'static>> = vec![
        Span::styled(
            "  ACE".to_string(),
            Style::default().fg(theme::GOLD).add_modifier(bold),
        ),
        Span::styled(
            format!("  v{version}"),
            Style::default().fg(theme::MUTED),
        ),
        Span::raw(" ".repeat(pad)),
        Span::raw(" "),
    ];
    for (i, tab) in AceExplorerTab::ALL.iter().enumerate() {
        if i == state.active_tab {
            let bg = match tab {
                AceExplorerTab::Playground => theme::PINE,
                AceExplorerTab::Variables => theme::FOAM,
                AceExplorerTab::Policies => theme::IRIS,
                AceExplorerTab::Evals => theme::ROSE,
            };
            tab_bar.push(Span::styled(
                format!(" {} ", tab.label()),
                Style::default().fg(theme::BASE).bg(bg).add_modifier(bold),
            ));
        } else {
            tab_bar.push(Span::styled(
                format!("  {}  ", tab.label()),
                Style::default().fg(theme::SUBTLE),
            ));
        }
    }
    tab_bar.push(Span::styled(
        "    Tab / ⇧Tab",
        Style::default().fg(theme::MUTED),
    ));

    frame.render_widget(
        Paragraph::new(vec![Line::from(""), Line::from(tab_bar)]),
        area,
    );
}

pub fn render_search_box(state: &AceExplorerState, frame: &mut Frame, area: Rect) {
    let placeholder = match state.current_tab() {
        AceExplorerTab::Variables => "Search variables...",
        AceExplorerTab::Policies => "Search policies...",
        AceExplorerTab::Evals => "Search evals...",
        AceExplorerTab::Playground => "Search...",
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

    let mut row: Vec<Span<'static>> = vec![Span::raw("  "), Span::styled("│ ⌕ ", box_style)];
    let icon_len = 2;
    let cursor_style = Style::default().fg(theme::BASE).bg(theme::TEXT);
    if state.search_focused {
        if state.search.is_empty() {
            row.push(Span::styled(" ", cursor_style));
            row.push(Span::raw(" ".repeat(inner_w.saturating_sub(icon_len + 1))));
        } else {
            row.push(Span::styled(
                state.search.clone(),
                Style::default().fg(theme::TEXT),
            ));
            row.push(Span::styled(" ", cursor_style));
            row.push(Span::raw(" ".repeat(
                inner_w.saturating_sub(icon_len + state.search.len() + 1),
            )));
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

    let bot = Line::from(vec![
        Span::raw("  "),
        Span::styled(format!("└{}┘", "─".repeat(inner_w + 2)), box_style),
    ]);

    frame.render_widget(Paragraph::new(vec![top, mid, bot]), area);
}

/// Render a single editable field in the right panel.
pub fn build_field_lines(
    state: &AceExplorerState,
    lines: &mut Vec<Line<'static>>,
    field_idx: usize,
    label: &str,
    value: &str,
) {
    let bold = Modifier::BOLD;
    let is_active = state.editing_right && state.editing_field == field_idx;
    let is_typing = is_active && state.field_editing;

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

    if is_typing {
        let cursor = state.field_cursor;
        let before = &state.field_input[..cursor];
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

/// Render read-only metadata at the bottom of the right panel.
pub fn build_metadata_lines(
    lines: &mut Vec<Line<'static>>,
    created_at: Option<&str>,
    created_by: Option<&str>,
    updated_at: Option<&str>,
    updated_by: Option<&str>,
) {
    if created_at.is_none() && created_by.is_none() && updated_at.is_none() && updated_by.is_none()
    {
        return;
    }

    let meta = Style::default().fg(theme::MUTED);
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled("─".repeat(20), meta)));

    if let Some(at) = created_at {
        lines.push(Line::from(vec![
            Span::styled("created_at  ", meta),
            Span::styled(at.to_string(), meta),
        ]));
    }
    if let Some(by) = created_by {
        lines.push(Line::from(vec![
            Span::styled("created_by  ", meta),
            Span::styled(by.to_string(), meta),
        ]));
    }
    if let Some(at) = updated_at {
        lines.push(Line::from(vec![
            Span::styled("updated_at  ", meta),
            Span::styled(at.to_string(), meta),
        ]));
    }
    if let Some(by) = updated_by {
        lines.push(Line::from(vec![
            Span::styled("updated_by  ", meta),
            Span::styled(by.to_string(), meta),
        ]));
    }
}

/// Recursively render a serde_yaml::Value as styled lines with indentation.
pub fn render_yaml_value(
    lines: &mut Vec<Line<'static>>,
    value: &serde_yaml::Value,
    indent: usize,
) {
    let prefix = "  ".repeat(indent);
    match value {
        serde_yaml::Value::Null => {
            lines.push(Line::from(Span::styled(
                format!("{prefix}null"),
                Style::default().fg(theme::MUTED),
            )));
        }
        serde_yaml::Value::Bool(b) => {
            lines.push(Line::from(Span::styled(
                format!("{prefix}{b}"),
                Style::default().fg(theme::GOLD),
            )));
        }
        serde_yaml::Value::Number(n) => {
            lines.push(Line::from(Span::styled(
                format!("{prefix}{n}"),
                Style::default().fg(theme::GOLD),
            )));
        }
        serde_yaml::Value::String(s) => {
            lines.push(Line::from(Span::styled(
                format!("{prefix}{s}"),
                Style::default().fg(theme::TEXT),
            )));
        }
        serde_yaml::Value::Sequence(seq) => {
            for item in seq {
                match item {
                    serde_yaml::Value::String(s) => {
                        lines.push(Line::from(vec![
                            Span::styled(format!("{prefix}- "), Style::default().fg(theme::MUTED)),
                            Span::styled(s.clone(), Style::default().fg(theme::TEXT)),
                        ]));
                    }
                    serde_yaml::Value::Mapping(_) => {
                        lines.push(Line::from(Span::styled(
                            format!("{prefix}-"),
                            Style::default().fg(theme::MUTED),
                        )));
                        render_yaml_value(lines, item, indent + 1);
                    }
                    _ => {
                        lines.push(Line::from(vec![
                            Span::styled(format!("{prefix}- "), Style::default().fg(theme::MUTED)),
                            Span::styled(
                                yaml_value_str(item),
                                Style::default().fg(theme::TEXT),
                            ),
                        ]));
                    }
                }
            }
        }
        serde_yaml::Value::Mapping(map) => {
            for (k, v) in map {
                let key_str = yaml_value_str(k);
                let key_color = if key_str.ends_with('!') {
                    theme::FOAM
                } else {
                    theme::IRIS
                };
                match v {
                    serde_yaml::Value::String(_)
                    | serde_yaml::Value::Number(_)
                    | serde_yaml::Value::Bool(_)
                    | serde_yaml::Value::Null => {
                        lines.push(Line::from(vec![
                            Span::styled(
                                format!("{prefix}{key_str}: "),
                                Style::default().fg(key_color),
                            ),
                            Span::styled(yaml_value_str(v), Style::default().fg(theme::TEXT)),
                        ]));
                    }
                    _ => {
                        lines.push(Line::from(Span::styled(
                            format!("{prefix}{key_str}:"),
                            Style::default().fg(key_color),
                        )));
                        render_yaml_value(lines, v, indent + 1);
                    }
                }
            }
        }
        serde_yaml::Value::Tagged(tagged) => {
            render_yaml_value(lines, &tagged.value, indent);
        }
    }
}

pub fn yaml_value_str(v: &serde_yaml::Value) -> String {
    match v {
        serde_yaml::Value::String(s) => s.clone(),
        serde_yaml::Value::Number(n) => n.to_string(),
        serde_yaml::Value::Bool(b) => b.to_string(),
        serde_yaml::Value::Null => "null".to_string(),
        other => format!("{other:?}"),
    }
}
