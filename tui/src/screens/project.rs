use ratatui::{
    style::{Modifier, Style},
    text::{Line, Span},
};

use crate::app::App;
use crate::theme;

// ---------------------------------------------------------------------------
// Project state (cursor on workforce members)
// ---------------------------------------------------------------------------

pub struct ProjectState {
    /// Index of the currently selected workforce member.
    pub selected_member: usize,
}

impl ProjectState {
    pub fn new() -> Self {
        Self { selected_member: 0 }
    }

    pub fn move_up(&mut self) {
        self.selected_member = self.selected_member.saturating_sub(1);
    }

    pub fn move_down(&mut self, member_count: usize) {
        if member_count > 0 {
            self.selected_member = (self.selected_member + 1).min(member_count - 1);
        }
    }
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

pub fn build_lines(app: &App, width: u16) -> Vec<Line<'static>> {
    let bold = Modifier::BOLD;
    let mut lines: Vec<Line<'static>> = Vec::new();

    // Separator.
    lines.push(Line::from(Span::styled(
        "─".repeat(width as usize),
        Style::default().fg(theme::MUTED),
    )));
    lines.push(Line::from(""));

    if !app.project.is_timbal_project {
        lines.push(Line::from(vec![
            Span::styled("  ⚠ ", Style::default().fg(theme::GOLD)),
            Span::styled(
                "Not a Timbal project. Run `timbal create` to get started.",
                Style::default().fg(theme::GOLD),
            ),
        ]));
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "  Esc to dismiss",
            Style::default().fg(theme::MUTED),
        )));
        return lines;
    }

    // --- Interfaces ---
    lines.push(Line::from(Span::styled(
        "  Interfaces",
        Style::default().fg(theme::TEXT).add_modifier(bold),
    )));
    lines.push(Line::from(""));

    // ui
    if app.project.has_ui {
        lines.push(Line::from(vec![
            Span::styled("    ui  ", Style::default().fg(theme::FOAM)),
            Span::styled("React Vite", Style::default().fg(theme::MUTED)),
            Span::styled("  ·  ", Style::default().fg(theme::SUBTLE)),
            Span::styled("shadcn", Style::default().fg(theme::MUTED)),
            Span::styled("  ·  ", Style::default().fg(theme::SUBTLE)),
            Span::styled("bun", Style::default().fg(theme::MUTED)),
        ]));
    } else {
        lines.push(Line::from(vec![
            Span::styled("    ui  ", Style::default().fg(theme::SUBTLE)),
            Span::styled("not configured  ", Style::default().fg(theme::SUBTLE)),
            Span::styled("`timbal add ui`", Style::default().fg(theme::MUTED)),
        ]));
    }
    lines.push(Line::from(""));

    // api
    if app.project.has_api {
        lines.push(Line::from(vec![
            Span::styled("    api  ", Style::default().fg(theme::FOAM)),
            Span::styled("Elysia", Style::default().fg(theme::MUTED)),
            Span::styled("  ·  ", Style::default().fg(theme::SUBTLE)),
            Span::styled("bun", Style::default().fg(theme::MUTED)),
        ]));
    } else {
        lines.push(Line::from(vec![
            Span::styled("    api  ", Style::default().fg(theme::SUBTLE)),
            Span::styled("not configured  ", Style::default().fg(theme::SUBTLE)),
            Span::styled("`timbal add api`", Style::default().fg(theme::MUTED)),
        ]));
    }
    lines.push(Line::from(""));

    // --- Workforce ---
    lines.push(Line::from(Span::styled(
        "  Workforce",
        Style::default().fg(theme::TEXT).add_modifier(bold),
    )));
    lines.push(Line::from(""));

    let selected = app.project_state.selected_member;

    if app.project.members.is_empty() && app.project.legacy_members.is_empty() {
        lines.push(Line::from(vec![
            Span::styled("    ", Style::default()),
            Span::styled("none", Style::default().fg(theme::SUBTLE)),
        ]));
        lines.push(Line::from(""));
    } else {
        for (idx, m) in app.project.members.iter().enumerate() {
            let is_selected = idx == selected;
            let kind_color = match m.kind.as_str() {
                "agent" => theme::IRIS,
                "workflow" => theme::FOAM,
                _ => theme::SUBTLE,
            };

            // Selection indicator + member name + kind + fqn.
            let indicator = if is_selected { "  > " } else { "    " };
            let indicator_style = if is_selected {
                Style::default().fg(theme::IRIS).add_modifier(bold)
            } else {
                Style::default()
            };
            let name_style = if is_selected {
                Style::default().fg(theme::TEXT).add_modifier(bold)
            } else {
                Style::default().fg(theme::TEXT)
            };

            let mut member_line = vec![
                Span::styled(indicator, indicator_style),
                Span::styled(m.name.clone(), name_style),
                Span::styled("  ", Style::default()),
                Span::styled(m.kind.clone(), Style::default().fg(kind_color)),
            ];
            if let Some(fqn) = &m.fqn {
                member_line.push(Span::styled(
                    format!("  {fqn}"),
                    Style::default().fg(theme::MUTED),
                ));
            }
            lines.push(Line::from(member_line));

            // ACE — agents only.
            if m.kind == "agent" {
                if let Some(ace) = &m.ace {
                    let var_count = ace.variables.len();
                    let pol_count = ace.policies.len();
                    let ace_detail = if var_count > 0 || pol_count > 0 {
                        format!("  {var_count} variables, {pol_count} policies")
                    } else {
                        "  configured (empty)".to_string()
                    };
                    lines.push(Line::from(vec![
                        Span::styled("     └ ", Style::default().fg(theme::MUTED)),
                        Span::styled("ace", Style::default().fg(theme::GOLD)),
                        Span::styled(ace_detail, Style::default().fg(theme::MUTED)),
                    ]));
                } else {
                    lines.push(Line::from(vec![
                        Span::styled("     └ ", Style::default().fg(theme::MUTED)),
                        Span::styled("ace  ", Style::default().fg(theme::SUBTLE)),
                        Span::styled("not configured  ", Style::default().fg(theme::SUBTLE)),
                        Span::styled(
                            format!("`timbal ace init {}`", m.name),
                            Style::default().fg(theme::MUTED),
                        ),
                    ]));
                }
            }

            lines.push(Line::from(""));
        }

        for name in &app.project.legacy_members {
            lines.push(Line::from(vec![
                Span::styled("    ", Style::default()),
                Span::styled(name.clone(), Style::default().fg(theme::GOLD)),
                Span::styled("  legacy, needs migration", Style::default().fg(theme::GOLD)),
            ]));
            lines.push(Line::from(""));
        }
    }

    lines.push(Line::from(Span::styled(
        "  ↑/↓ navigate  ·  Enter select  ·  Esc dismiss",
        Style::default().fg(theme::MUTED),
    )));
    lines.push(Line::from(""));

    lines
}
