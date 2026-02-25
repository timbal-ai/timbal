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

    // --- Surfaces ---
    lines.push(Line::from(Span::styled(
        "  Surfaces",
        Style::default().fg(theme::TEXT).add_modifier(bold),
    )));

    let surfaces: &[(&str, bool, &str)] = &[
        ("ui", app.project.has_ui, "timbal add ui"),
        ("api", app.project.has_api, "timbal add api"),
    ];

    for &(name, available, add_cmd) in surfaces {
        if available {
            lines.push(Line::from(vec![
                Span::styled("  ", Style::default()),
                Span::styled(
                    format!(" {} ", name),
                    Style::default().fg(theme::SURFACE).bg(theme::FOAM),
                ),
            ]));
        } else {
            lines.push(Line::from(vec![
                Span::styled(format!("  {:<8}", name), Style::default().fg(theme::SUBTLE)),
                Span::styled(
                    format!("`{}`", add_cmd),
                    Style::default().fg(theme::MUTED),
                ),
            ]));
        }
    }
    lines.push(Line::from(""));

    // --- Workforce ---
    lines.push(Line::from(Span::styled(
        "  Workforce",
        Style::default().fg(theme::TEXT).add_modifier(bold),
    )));

    if app.project.members.is_empty() && app.project.legacy_members.is_empty() {
        lines.push(Line::from(Span::styled(
            "  none",
            Style::default().fg(theme::SUBTLE),
        )));
    } else {
        for m in &app.project.members {
            let kind_color = match m.kind.as_str() {
                "agent" => theme::IRIS,
                "workflow" => theme::FOAM,
                _ => theme::SUBTLE,
            };
            let mut spans = vec![
                Span::styled(format!("  {:<18}", m.name), Style::default().fg(theme::TEXT)),
                Span::styled(
                    format!(" {} ", m.kind),
                    Style::default().fg(theme::SURFACE).bg(kind_color),
                ),
            ];
            if let Some(fqn) = &m.fqn {
                spans.push(Span::styled(
                    format!("  {}", fqn),
                    Style::default().fg(theme::MUTED),
                ));
            }
            lines.push(Line::from(spans));
        }

        for name in &app.project.legacy_members {
            lines.push(Line::from(vec![
                Span::styled(format!("  {:<18}", name), Style::default().fg(theme::GOLD)),
                Span::styled(
                    " legacy ",
                    Style::default().fg(theme::SURFACE).bg(theme::GOLD),
                ),
                Span::styled(
                    "  needs migration",
                    Style::default().fg(theme::GOLD),
                ),
            ]));
        }
    }

    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "  Esc to dismiss",
        Style::default().fg(theme::MUTED),
    )));
    lines.push(Line::from(""));

    lines
}
