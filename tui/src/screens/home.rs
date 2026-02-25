use ratatui::{
    Frame,
    layout::{Alignment, Margin},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Paragraph, Wrap},
};

use crate::app::App;
use crate::model::conversation::{OutputBlock, ToolStatus, TurnInput, TurnStatus};
use crate::theme;
use crate::widgets::spinner::Spinner;
use crate::widgets::vectorscope::Vectorscope;

// ---------------------------------------------------------------------------
// Logo
// ---------------------------------------------------------------------------

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
                format!(" v{} ", crate::VERSION),
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

// ---------------------------------------------------------------------------
// Conversation rendering (Turn-based)
// ---------------------------------------------------------------------------

fn conversation_lines(app: &App) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = Vec::new();
    let spinner = Spinner::new(app.spinner_tick);

    for turn in &app.conversation.turns {
        // User input line.
        let input_text = match &turn.input {
            TurnInput::Message(text) => text.clone(),
            TurnInput::Command(cmd) => cmd.clone(),
        };

        lines.push(Line::from(vec![
            Span::styled("❯ ", Style::default().fg(theme::IRIS)),
            Span::styled(
                input_text,
                Style::default()
                    .fg(theme::TEXT)
                    .add_modifier(Modifier::BOLD),
            ),
        ]));

        // Output blocks.
        for block in &turn.outputs {
            match block {
                OutputBlock::Text(text) => {
                    for line in text.lines() {
                        lines.push(Line::from(vec![
                            Span::styled("  ", Style::default()),
                            Span::styled(line.to_string(), Style::default().fg(theme::TEXT)),
                        ]));
                    }
                }
                OutputBlock::RichLines(rich) => {
                    lines.extend(rich.iter().cloned());
                }
                OutputBlock::ToolCall {
                    name,
                    status,
                    output,
                } => {
                    let (icon, style) = match status {
                        ToolStatus::Running => (
                            format!("{} ", spinner.frame()),
                            Style::default().fg(theme::GOLD),
                        ),
                        ToolStatus::Completed => (
                            "✓ ".to_string(),
                            Style::default().fg(theme::FOAM),
                        ),
                        ToolStatus::Failed => (
                            "✗ ".to_string(),
                            Style::default().fg(theme::LOVE),
                        ),
                    };

                    lines.push(Line::from(vec![
                        Span::styled("  ", Style::default()),
                        Span::styled(icon, style),
                        Span::styled(name.clone(), style),
                    ]));

                    if let Some(out) = output {
                        for line in out.lines() {
                            lines.push(Line::from(vec![
                                Span::styled("    ", Style::default()),
                                Span::styled(
                                    line.to_string(),
                                    Style::default().fg(theme::SUBTLE),
                                ),
                            ]));
                        }
                    }
                }
                OutputBlock::Error(msg) => {
                    lines.push(Line::from(vec![
                        Span::styled("  ✗ ", Style::default().fg(theme::LOVE)),
                        Span::styled(msg.clone(), Style::default().fg(theme::LOVE)),
                    ]));
                }
            }
        }

        // Turn status indicator.
        match &turn.status {
            TurnStatus::Streaming => {
                lines.push(Line::from(Span::styled(
                    format!("  {} Thinking…", spinner.frame()),
                    Style::default().fg(theme::MUTED),
                )));
            }
            TurnStatus::Interrupted => {
                lines.push(Line::from(vec![
                    Span::styled("  └ ", Style::default().fg(theme::MUTED)),
                    Span::styled("Interrupted", Style::default().fg(theme::SUBTLE)),
                ]));
            }
            TurnStatus::Completed(msg) => {
                lines.push(Line::from(vec![
                    Span::styled("  └ ", Style::default().fg(theme::MUTED)),
                    Span::styled(msg.clone(), Style::default().fg(theme::SUBTLE)),
                ]));
            }
            TurnStatus::Complete => {}
        }

        lines.push(Line::from(""));
    }

    lines
}

// ---------------------------------------------------------------------------
// Input / Palette / Separator
// ---------------------------------------------------------------------------

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
    let cmds = app.filter_commands();
    if cmds.is_empty() {
        return vec![];
    }

    let desc_col = 24_u16;

    cmds.iter()
        .enumerate()
        .map(|(i, cmd)| {
            let meta = cmd.meta();
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
                .starts_with(&format!("{} ", meta.name));
            let right_text = if typing_args {
                meta.usage
            } else {
                meta.description
            };

            let name_padded = format!("{:<w$}", meta.name, w = desc_col as usize);
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

// ---------------------------------------------------------------------------
// Main render
// ---------------------------------------------------------------------------

const SCOPE_HEIGHT: u16 = 20;

pub fn render(app: &mut App, frame: &mut Frame) {
    let area = frame.area();
    let padded = area.inner(Margin {
        horizontal: 1,
        vertical: 0,
    });
    let width = padded.width;

    let mut doc: Vec<Line<'static>> = Vec::new();

    // Logo.
    doc.extend(styled_logo());

    // Conversation turns.
    let conv_lines = conversation_lines(app);
    if !conv_lines.is_empty() {
        doc.extend(conv_lines);
    }

    // Vectorscope placeholder — after conversation, before input box.
    let scope_doc_start = doc.len() as u16;
    let is_busy = app.conversation.is_busy();
    if is_busy {
        for _ in 0..SCOPE_HEIGHT {
            doc.push(Line::from(""));
        }
    }

    // Input (hidden when config or help dialog is open).
    if !app.config_open && !app.help_open {
        doc.push(separator_line(width));
        doc.push(input_line(app));
        doc.push(separator_line(width));

        if is_busy {
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
        doc.extend(crate::screens::configure::build_lines(
            &app.configure_state,
            width,
        ));
    }

    // Help panel.
    if app.help_open {
        doc.extend(crate::screens::help::build_lines(&app.help_state, width));
    }

    let total_lines = doc.len() as u16;
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
    if is_busy && !app.scope_frames.is_empty() {
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
