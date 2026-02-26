use ratatui::{
    style::{Modifier, Style},
    text::{Line, Span},
};

use crate::app::App;
use crate::theme;

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

    lines.push(Line::from(Span::styled(
        "  Interfaces",
        Style::default().fg(theme::TEXT).add_modifier(bold),
    )));
    lines.push(Line::from(""));

    if app.project.has_ui {
        lines.push(Line::from(vec![
            Span::styled("    ui  ", Style::default().fg(theme::PINE)),
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

    if app.project.has_api {
        lines.push(Line::from(vec![
            Span::styled("    api  ", Style::default().fg(theme::PINE)),
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

    if app.project.members.is_empty() && app.project.legacy_members.is_empty() {
        lines.push(Line::from(vec![
            Span::styled("    ", Style::default()),
            Span::styled("none", Style::default().fg(theme::SUBTLE)),
        ]));
        lines.push(Line::from(""));
    } else {
        for m in app.project.members.iter() {
            let kind_color = match m.kind.as_str() {
                "agent" => theme::PINE,
                "workflow" => theme::PINE,
                _ => theme::SUBTLE,
            };

            let mut member_line = vec![
                Span::styled("    ", Style::default()),
                Span::styled(m.name.clone(), Style::default().fg(theme::TEXT)),
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
            let has_evals = m.evals.is_some();
            if m.kind == "agent" {
                let connector = if has_evals { "├" } else { "└" };
                if let Some(ace) = &m.ace {
                    let var_count = ace.variables.len();
                    let pol_count = ace.policies.len();
                    let ace_detail = if var_count > 0 || pol_count > 0 {
                        format!("  {var_count} variables, {pol_count} policies")
                    } else {
                        "  configured (empty)".to_string()
                    };
                    lines.push(Line::from(vec![
                        Span::styled(format!("     {connector} "), Style::default().fg(theme::MUTED)),
                        Span::styled("ace", Style::default().fg(theme::GOLD)),
                        Span::styled(ace_detail, Style::default().fg(theme::MUTED)),
                    ]));
                } else {
                    lines.push(Line::from(vec![
                        Span::styled(format!("     {connector} "), Style::default().fg(theme::MUTED)),
                        Span::styled("ace  ", Style::default().fg(theme::SUBTLE)),
                        Span::styled("not configured  ", Style::default().fg(theme::SUBTLE)),
                        Span::styled(
                            format!("`timbal ace init {}`", m.name),
                            Style::default().fg(theme::MUTED),
                        ),
                    ]));
                }
            }

            // Evals.
            if let Some(evals) = &m.evals {
                let eval_count: usize = evals.files.iter().map(|(_, cases)| cases.len()).sum();
                let file_count = evals.files.len();
                let detail = format!(
                    "  {eval_count} eval{} in {file_count} file{}",
                    if eval_count == 1 { "" } else { "s" },
                    if file_count == 1 { "" } else { "s" },
                );
                lines.push(Line::from(vec![
                    Span::styled("     └ ", Style::default().fg(theme::MUTED)),
                    Span::styled("evals", Style::default().fg(theme::ROSE)),
                    Span::styled(detail, Style::default().fg(theme::MUTED)),
                ]));
            } else {
                lines.push(Line::from(vec![
                    Span::styled("     └ ", Style::default().fg(theme::MUTED)),
                    Span::styled("evals  ", Style::default().fg(theme::SUBTLE)),
                    Span::styled("no evals found", Style::default().fg(theme::SUBTLE)),
                ]));
            }

            lines.push(Line::from(""));
        }

        for name in &app.project.legacy_members {
            lines.push(Line::from(vec![
                Span::styled("    ", Style::default()),
                Span::styled(name.clone(), Style::default().fg(theme::GOLD)),
                Span::styled(
                    "  legacy, needs migration",
                    Style::default().fg(theme::GOLD),
                ),
            ]));
            lines.push(Line::from(""));
        }
    }

    lines.push(Line::from(Span::styled(
        "  Esc dismiss",
        Style::default().fg(theme::MUTED),
    )));
    lines.push(Line::from(""));

    lines
}
