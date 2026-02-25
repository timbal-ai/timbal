use ratatui::{
    Frame,
    layout::{Alignment, Margin},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Paragraph, Wrap},
};

use crate::app::App;
use crate::commands;
use crate::history::EntryKind;
use crate::screens::configure;
use crate::theme;
use crate::widgets::vectorscope::Vectorscope;

fn styled_logo() -> Vec<Line<'static>> {
    let bold = Modifier::BOLD;
    let c = Alignment::Center;
    vec![
        Line::from(""),
        Line::from(""),
        Line::from(vec![
            Span::styled(
                " ████████╗ ",
                Style::default().fg(theme::LOVE).add_modifier(bold),
            ),
            Span::styled("██╗ ", Style::default().fg(theme::GOLD).add_modifier(bold)),
            Span::styled(
                "███╗   ███╗ ",
                Style::default().fg(theme::FOAM).add_modifier(bold),
            ),
            Span::styled(
                "██████╗  ",
                Style::default().fg(theme::IRIS).add_modifier(bold),
            ),
            Span::styled(
                " █████╗  ",
                Style::default().fg(theme::LOVE).add_modifier(bold),
            ),
            Span::styled(
                "██╗     ",
                Style::default().fg(theme::GOLD).add_modifier(bold),
            ),
        ]),
        Line::from(vec![
            Span::styled(
                " ╚══██╔══╝ ",
                Style::default().fg(theme::LOVE).add_modifier(bold),
            ),
            Span::styled("██║ ", Style::default().fg(theme::GOLD).add_modifier(bold)),
            Span::styled(
                "████╗ ████║ ",
                Style::default().fg(theme::FOAM).add_modifier(bold),
            ),
            Span::styled(
                "██╔══██╗ ",
                Style::default().fg(theme::IRIS).add_modifier(bold),
            ),
            Span::styled(
                "██╔══██╗ ",
                Style::default().fg(theme::LOVE).add_modifier(bold),
            ),
            Span::styled(
                "██║     ",
                Style::default().fg(theme::GOLD).add_modifier(bold),
            ),
        ]),
        Line::from(vec![
            Span::styled(
                "    ██║    ",
                Style::default().fg(theme::LOVE).add_modifier(bold),
            ),
            Span::styled("██║ ", Style::default().fg(theme::GOLD).add_modifier(bold)),
            Span::styled(
                "██╔████╔██║ ",
                Style::default().fg(theme::FOAM).add_modifier(bold),
            ),
            Span::styled(
                "██████╔╝ ",
                Style::default().fg(theme::IRIS).add_modifier(bold),
            ),
            Span::styled(
                "███████║ ",
                Style::default().fg(theme::LOVE).add_modifier(bold),
            ),
            Span::styled(
                "██║     ",
                Style::default().fg(theme::GOLD).add_modifier(bold),
            ),
        ]),
        Line::from(vec![
            Span::styled(
                "    ██║    ",
                Style::default().fg(theme::LOVE).add_modifier(bold),
            ),
            Span::styled("██║ ", Style::default().fg(theme::GOLD).add_modifier(bold)),
            Span::styled(
                "██║╚██╔╝██║ ",
                Style::default().fg(theme::FOAM).add_modifier(bold),
            ),
            Span::styled(
                "██╔══██╗ ",
                Style::default().fg(theme::IRIS).add_modifier(bold),
            ),
            Span::styled(
                "██╔══██║ ",
                Style::default().fg(theme::LOVE).add_modifier(bold),
            ),
            Span::styled(
                "██║     ",
                Style::default().fg(theme::GOLD).add_modifier(bold),
            ),
        ]),
        Line::from(vec![
            Span::styled(
                "    ██║    ",
                Style::default().fg(theme::LOVE).add_modifier(bold),
            ),
            Span::styled("██║ ", Style::default().fg(theme::GOLD).add_modifier(bold)),
            Span::styled(
                "██║ ╚═╝ ██║ ",
                Style::default().fg(theme::FOAM).add_modifier(bold),
            ),
            Span::styled(
                "██████╔╝ ",
                Style::default().fg(theme::IRIS).add_modifier(bold),
            ),
            Span::styled(
                "██║  ██║ ",
                Style::default().fg(theme::LOVE).add_modifier(bold),
            ),
            Span::styled(
                "███████╗",
                Style::default().fg(theme::GOLD).add_modifier(bold),
            ),
        ]),
        Line::from(vec![
            Span::styled(
                "    ╚═╝    ",
                Style::default().fg(theme::LOVE).add_modifier(bold),
            ),
            Span::styled("╚═╝ ", Style::default().fg(theme::GOLD).add_modifier(bold)),
            Span::styled(
                "╚═╝     ╚═╝ ",
                Style::default().fg(theme::FOAM).add_modifier(bold),
            ),
            Span::styled(
                "╚═════╝  ",
                Style::default().fg(theme::IRIS).add_modifier(bold),
            ),
            Span::styled(
                "╚═╝  ╚═╝ ",
                Style::default().fg(theme::LOVE).add_modifier(bold),
            ),
            Span::styled(
                "╚══════╝",
                Style::default().fg(theme::GOLD).add_modifier(bold),
            ),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("🦦 "),
            Span::styled(
                "Timbal",
                Style::default().fg(theme::TEXT).add_modifier(bold),
            ),
            Span::raw("  "),
            Span::styled(
                " v1.2.3 ",
                Style::default().fg(theme::SURFACE).bg(theme::IRIS),
            ),
        ]),
        Line::from(Span::styled(
            "Build reliable, enterprise-grade AI applications",
            Style::default().fg(theme::SUBTLE),
        )),
        Line::from(Span::styled(
            shorten_home(
                &std::env::current_dir()
                    .unwrap_or_default()
                    .to_string_lossy(),
            ),
            Style::default().fg(theme::MUTED),
        )),
        Line::from(""),
        Line::from(""),
    ]
    .into_iter()
    .map(|l| l.alignment(c))
    .collect()
}

fn shorten_home(path: &str) -> String {
    if let Some(home) = dirs::home_dir() {
        let home_str = home.to_string_lossy();
        if path.starts_with(home_str.as_ref()) {
            return format!("~{}", &path[home_str.len()..]);
        }
    }
    path.to_string()
}

fn history_lines(app: &App) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = Vec::new();

    let visible: Vec<_> = app
        .history
        .iter()
        .filter(|e| !matches!(e.kind, EntryKind::SessionStart))
        .collect();

    let mut i = 0;
    while i < visible.len() {
        let entry = &visible[i];
        match &entry.kind {
            EntryKind::Message(text) => {
                lines.push(Line::from(vec![
                    Span::styled("❯ ", Style::default().fg(theme::IRIS)),
                    Span::styled(
                        text.clone(),
                        Style::default()
                            .fg(theme::TEXT)
                            .add_modifier(Modifier::BOLD),
                    ),
                ]));

                // Look ahead for child outcome events (e.g. Interrupted).
                let mut j = i + 1;
                while j < visible.len() {
                    match &visible[j].kind {
                        EntryKind::Interrupted => {
                            lines.push(Line::from(vec![
                                Span::styled("  └ ", Style::default().fg(theme::MUTED)),
                                Span::styled(
                                    "Interrupted",
                                    Style::default().fg(theme::SUBTLE),
                                ),
                            ]));
                            i = j;
                            j += 1;
                        }
                        _ => break,
                    }
                }
            }
            EntryKind::Command(cmd) => {
                lines.push(Line::from(vec![
                    Span::styled("❯ ", Style::default().fg(theme::IRIS)),
                    Span::styled(
                        cmd.clone(),
                        Style::default()
                            .fg(theme::TEXT)
                            .add_modifier(Modifier::BOLD),
                    ),
                ]));

                let mut j = i + 1;
                while j < visible.len() {
                    match &visible[j].kind {
                        EntryKind::ConfigureSaved(profile) => {
                            lines.push(Line::from(vec![
                                Span::styled("  └ ", Style::default().fg(theme::MUTED)),
                                Span::styled(
                                    format!("Credentials saved (profile: {profile})"),
                                    Style::default().fg(theme::FOAM),
                                ),
                            ]));
                            i = j;
                            j += 1;
                        }
                        EntryKind::ConfigureCancelled => {
                            lines.push(Line::from(vec![
                                Span::styled("  └ ", Style::default().fg(theme::MUTED)),
                                Span::styled(
                                    "Config dialog dismissed",
                                    Style::default().fg(theme::SUBTLE),
                                ),
                            ]));
                            i = j;
                            j += 1;
                        }
                        EntryKind::Interrupted => {
                            lines.push(Line::from(vec![
                                Span::styled("  └ ", Style::default().fg(theme::MUTED)),
                                Span::styled(
                                    "Interrupted",
                                    Style::default().fg(theme::SUBTLE),
                                ),
                            ]));
                            i = j;
                            j += 1;
                        }
                        EntryKind::Message(_) | EntryKind::Command(_) => break,
                        _ => break,
                    }
                }
            }
            EntryKind::ConfigureSaved(_) | EntryKind::ConfigureCancelled | EntryKind::Interrupted => {
                lines.push(Line::from(Span::styled(
                    format!("  └ {}", entry.kind),
                    Style::default().fg(theme::MUTED),
                )));
            }
            EntryKind::SessionStart => {}
        }
        lines.push(Line::from(""));
        i += 1;
    }

    lines
}

fn separator_line(width: u16) -> Line<'static> {
    Line::from(Span::styled(
        "─".repeat(width as usize),
        Style::default().fg(theme::MUTED),
    ))
}

fn input_line(app: &App) -> Line<'static> {
    Line::from(vec![
        Span::styled("❯ ", Style::default().fg(theme::IRIS)),
        Span::styled(app.input.clone(), Style::default().fg(theme::TEXT)),
        Span::styled("█", Style::default().fg(theme::SUBTLE)),
    ])
}

fn palette_lines(app: &App, width: u16) -> Vec<Line<'static>> {
    let cmds = commands::filter(&app.input);
    if cmds.is_empty() {
        return vec![];
    }

    let desc_col = 24_u16;

    cmds.iter()
        .enumerate()
        .map(|(i, cmd)| {
            let selected = app.palette_selected == Some(i);
            let name_style = if selected {
                Style::default()
                    .fg(theme::IRIS)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(theme::FOAM)
            };
            let desc_style = if selected {
                Style::default()
                    .fg(theme::TEXT)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(theme::SUBTLE)
            };

            let typing_args = app
                .input
                .trim_start()
                .starts_with(&format!("{} ", cmd.name));
            let right_text = if typing_args {
                cmd.usage
            } else {
                cmd.description
            };

            let name_padded = format!("{:<w$}", cmd.name, w = desc_col as usize);
            let max_right = width.saturating_sub(desc_col) as usize;
            let right_text = if right_text.len() > max_right {
                &right_text[..max_right]
            } else {
                right_text
            };

            Line::from(vec![
                Span::styled(name_padded, name_style),
                Span::styled(right_text, desc_style),
            ])
        })
        .collect()
}

const SCOPE_HEIGHT: u16 = 20;

pub fn render(app: &mut App, frame: &mut Frame) {
    let area = frame.area();
    let padded = area.inner(Margin {
        horizontal: 1,
        vertical: 0,
    });
    let width = padded.width;

    // Build one scrollable document.
    let mut doc: Vec<Line<'static>> = Vec::new();

    // Logo.
    doc.extend(styled_logo());

    // History.
    let has_history = app
        .history
        .iter()
        .any(|e| !matches!(e.kind, EntryKind::SessionStart));
    if has_history || app.thinking {
        let mut h = history_lines(app);
        if app.thinking {
            h.push(Line::from(Span::styled(
                "· Thinking…",
                Style::default().fg(theme::MUTED),
            )));
        }
        doc.extend(h);
    }

    // Vectorscope placeholder — right after "Thinking...", before the input box.
    let scope_doc_start = doc.len() as u16;
    if app.thinking {
        for _ in 0..SCOPE_HEIGHT {
            doc.push(Line::from(""));
        }
    }

    // Input (hidden when config dialog is open).
    if !app.config_open {
        doc.push(separator_line(width));
        doc.push(input_line(app));
        doc.push(separator_line(width));

        if app.thinking {
            doc.push(Line::from(""));
            doc.push(Line::from(Span::styled(
                "  esc to interrupt",
                Style::default().fg(theme::MUTED),
            )));
        }
    }

    // Palette.
    if app.palette_open() {
        doc.extend(palette_lines(app, width));
    }

    // Config panel.
    if app.config_open {
        doc.extend(configure::build_lines(&app.configure_state, width));
    }

    let total_lines = doc.len() as u16;

    // The entire padded area is the scroll area.
    let scroll_area = padded;

    // Clamp scroll.
    let visible_height = scroll_area.height;
    let max_scroll = total_lines.saturating_sub(visible_height);
    app.scroll = app.scroll.min(max_scroll);

    // Render the scrollable text document.
    frame.render_widget(
        Paragraph::new(doc)
            .wrap(Wrap { trim: false })
            .scroll((app.scroll, 0)),
        scroll_area,
    );

    // Overlay the vectorscope widget on top of the blank placeholder lines.
    if app.thinking && !app.scope_frames.is_empty() {
        // Where does the scope start on screen?
        let scope_screen_y = scope_doc_start.saturating_sub(app.scroll);

        if scope_screen_y < visible_height {
            let available = visible_height.saturating_sub(scope_screen_y);
            let h = SCOPE_HEIGHT.min(available);

            if h > 2 {
                let scope_rect = ratatui::layout::Rect::new(
                    scroll_area.x,
                    scroll_area.y + scope_screen_y,
                    scroll_area.width,
                    h,
                );

                // Center horizontally.
                let scope_size = h.min(scope_rect.width / 2);
                let x_offset = (scope_rect.width.saturating_sub(scope_size * 2)) / 2;

                let centered = ratatui::layout::Rect::new(
                    scope_rect.x + x_offset,
                    scope_rect.y,
                    scope_size * 2,
                    h,
                );

                let points = app.current_scope_frame();
                frame.render_widget(Vectorscope::new(points), centered);
            }
        }
    }
}
