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

fn styled_logo(app: &App) -> Vec<Line<'static>> {
    let bold = Modifier::BOLD;
    let c = Alignment::Center;
    let mut logo = vec![
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
            Span::raw("  "),
            Span::styled(
                format!(" {} ", app.profile),
                Style::default().fg(theme::SURFACE).bg(theme::FOAM),
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
    ];

    // Project detection — only show warnings; silence when everything is valid.
    if app.project.is_timbal_project {
        if !app.project.legacy_members.is_empty() {
            let names = app.project.legacy_members.join(", ");
            logo.push(
                Line::from(vec![
                    Span::styled("⚠ ", Style::default().fg(theme::GOLD)),
                    Span::styled(
                        format!("Legacy members need migration: {names}"),
                        Style::default().fg(theme::GOLD),
                    ),
                ])
                .alignment(c),
            );
            logo.push(Line::from(""));
        }
    } else {
        logo.push(
            Line::from(vec![
                Span::styled("⚠ ", Style::default().fg(theme::GOLD)),
                Span::styled(
                    "Not a Timbal project. Run `timbal create` to get started.",
                    Style::default().fg(theme::GOLD),
                ),
            ])
            .alignment(c),
        );
        logo.push(Line::from(""));
    }

    if !app.configured {
        logo.push(Line::from(vec![
            Span::styled("⚠ ", Style::default().fg(theme::GOLD)),
            Span::styled(
                "No API key configured. Run ",
                Style::default().fg(theme::GOLD),
            ),
            Span::styled(
                "/configure",
                Style::default().fg(theme::IRIS).add_modifier(bold),
            ),
            Span::styled(
                " to set up credentials.",
                Style::default().fg(theme::GOLD),
            ),
        ]));
        logo.push(Line::from(""));
    }

    logo.into_iter()
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

const MAX_COLLAPSED_LINES: usize = 5;

/// Returns (lines, clickable_lines) where each entry is (doc_line_index, turn_index).
/// `hovered_line` is the doc line the mouse is over (for underline on hover).
fn conversation_lines(app: &App, hovered_line: Option<usize>) -> (Vec<Line<'static>>, Vec<(usize, usize)>) {
    let mut lines: Vec<Line<'static>> = Vec::new();
    let mut clickable: Vec<(usize, usize)> = Vec::new();
    let spinner = Spinner::new(app.spinner_tick);

    for (turn_idx, turn) in app.conversation.turns.iter().enumerate() {
        // User input line.
        match &turn.input {
            TurnInput::Shell(cmd) => {
                lines.push(Line::from(vec![
                    Span::styled("! ", Style::default().fg(theme::LOVE)),
                    Span::styled(
                        cmd.clone(),
                        Style::default()
                            .fg(theme::TEXT)
                            .add_modifier(Modifier::BOLD),
                    ),
                ]));
            }
            _ => {
                let input_text = match &turn.input {
                    TurnInput::Message(text) => text.clone(),
                    TurnInput::Command(cmd) => cmd.clone(),
                    TurnInput::Shell(_) => unreachable!(),
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
            }
        }

        // Determine shell output color based on exit status.
        let shell_failed = matches!(&turn.status, TurnStatus::Completed(msg) if msg.starts_with('✗'));
        let shell_color = if shell_failed { theme::LOVE } else { theme::SUBTLE };

        // Collect output lines, then apply turn-level collapse.
        let mut output_lines: Vec<Line<'static>> = Vec::new();

        for block in &turn.outputs {
            match block {
                OutputBlock::Text(text) => {
                    for line in text.lines() {
                        output_lines.push(Line::from(vec![
                            Span::styled("  ", Style::default()),
                            Span::styled(line.to_string(), Style::default().fg(theme::TEXT)),
                        ]));
                    }
                }
                OutputBlock::RichLines(rich) => {
                    output_lines.extend(rich.iter().cloned());
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

                    output_lines.push(Line::from(vec![
                        Span::styled("  ", Style::default()),
                        Span::styled(icon, style),
                        Span::styled(name.clone(), style),
                    ]));

                    if let Some(out) = output {
                        for line in out.lines() {
                            output_lines.push(Line::from(vec![
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
                    output_lines.push(Line::from(vec![
                        Span::styled("  ✗ ", Style::default().fg(theme::LOVE)),
                        Span::styled(msg.clone(), Style::default().fg(theme::LOVE)),
                    ]));
                }
                OutputBlock::ShellOutput(shell_lines) => {
                    for (i, line) in shell_lines.iter().enumerate() {
                        if i == 0 {
                            output_lines.push(Line::from(vec![
                                Span::styled("  └ ", Style::default().fg(theme::MUTED)),
                                Span::styled(
                                    line.clone(),
                                    Style::default().fg(shell_color),
                                ),
                            ]));
                        } else {
                            output_lines.push(Line::from(vec![
                                Span::styled("    ", Style::default()),
                                Span::styled(
                                    line.clone(),
                                    Style::default().fg(shell_color),
                                ),
                            ]));
                        }
                    }
                }
            }
        }

        // Apply turn-level collapse.
        let total = output_lines.len();
        if turn.collapsed && total > MAX_COLLAPSED_LINES {
            lines.extend(output_lines.into_iter().take(MAX_COLLAPSED_LINES));
            let remaining = total - MAX_COLLAPSED_LINES;
            let hint_line = lines.len();
            let hovered = hovered_line == Some(hint_line);
            let mut text_style = Style::default().fg(theme::MUTED);
            if hovered {
                text_style = text_style.add_modifier(Modifier::UNDERLINED);
            }
            lines.push(Line::from(vec![
                Span::styled("  ", Style::default()),
                Span::styled(format!("… +{remaining} lines (click to expand)"), text_style),
            ]));
            clickable.push((hint_line, turn_idx));
        } else if !turn.collapsed && total > MAX_COLLAPSED_LINES {
            lines.extend(output_lines);
            let hint_line = lines.len();
            let hovered = hovered_line == Some(hint_line);
            let mut text_style = Style::default().fg(theme::MUTED);
            if hovered {
                text_style = text_style.add_modifier(Modifier::UNDERLINED);
            }
            lines.push(Line::from(vec![
                Span::styled("  ", Style::default()),
                Span::styled("▴ click to collapse", text_style),
            ]));
            clickable.push((hint_line, turn_idx));
        } else {
            lines.extend(output_lines);
        }

        // Turn status indicator (only for streaming and interrupted).
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
                // Show status for non-shell turns (commands like /help, /configure).
                if !matches!(turn.input, TurnInput::Shell(_)) {
                    lines.push(Line::from(vec![
                        Span::styled("  └ ", Style::default().fg(theme::MUTED)),
                        Span::styled(msg.clone(), Style::default().fg(theme::SUBTLE)),
                    ]));
                }
            }
            TurnStatus::Complete => {}
        }

        lines.push(Line::from(""));
    }

    (lines, clickable)
}

// ---------------------------------------------------------------------------
// Input / Palette / Separator
// ---------------------------------------------------------------------------

fn separator_line(app: &App, width: u16) -> Line<'static> {
    let color = if app.bash_mode() { theme::LOVE } else { theme::MUTED };
    Line::from(Span::styled(
        "─".repeat(width as usize),
        Style::default().fg(color),
    ))
}

fn input_line(app: &App) -> Line<'static> {
    if app.bash_mode() {
        let display = app.input.strip_prefix('!').unwrap_or(&app.input);
        Line::from(vec![
            Span::styled("! ", Style::default().fg(theme::LOVE)),
            Span::styled(display.to_string(), Style::default().fg(theme::TEXT)),
            Span::styled("█", Style::default().fg(theme::SUBTLE)),
        ])
    } else {
        Line::from(vec![
            Span::styled("❯ ", Style::default().fg(theme::IRIS)),
            Span::styled(app.input.clone(), Style::default().fg(theme::TEXT)),
            Span::styled("█", Style::default().fg(theme::SUBTLE)),
        ])
    }
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

fn shortcuts_lines(width: u16) -> Vec<Line<'static>> {
    let s = Style::default().fg(theme::MUTED);
    let col = (width / 3) as usize;

    let rows: &[(&str, &str, &str)] = &[
        ("! for bash mode", "esc to cancel", "shift + click to select"),
        ("/ for commands", "\u{2191}/\u{2193} to scroll", "ctrl + c to quit"),
    ];

    let mut lines = Vec::new();
    for &(left, mid, right) in rows {
        lines.push(Line::from(vec![
            Span::styled(format!("  {:<w$}", left, w = col), s),
            Span::styled(format!("{:<w$}", mid, w = col), s),
            Span::styled(right, s),
        ]));
    }
    lines
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

    // Ace explorer takes over the full screen when open.
    if app.ace_explorer_open {
        crate::screens::ace_explorer::render_full(app, frame, padded);
        return;
    }

    let mut doc: Vec<Line<'static>> = Vec::new();

    // Logo.
    doc.extend(styled_logo(app));

    // Conversation turns.
    let logo_offset = doc.len();
    // Compute the doc line the mouse is hovering over (relative to conversation start).
    let hovered_doc_line = app.mouse_row.map(|r| r as usize + app.scroll as usize);
    let hovered_conv_line = hovered_doc_line.and_then(|d| d.checked_sub(logo_offset));
    let (conv_lines, clickable) = conversation_lines(app, hovered_conv_line);
    // Offset clickable line positions by logo lines so they map to doc positions.
    app.turn_line_ranges = clickable
        .into_iter()
        .map(|(line, idx)| (line + logo_offset, 0, idx))
        .collect();
    if !conv_lines.is_empty() {
        doc.extend(conv_lines);
    }

    // Vectorscope placeholder — only for LLM message turns, not shell/command.
    let scope_doc_start = doc.len() as u16;
    let is_busy = app.conversation.is_busy();
    let show_scope = is_busy
        && app.conversation.turns.last().is_some_and(|t| {
            matches!(t.input, TurnInput::Message(_))
        });
    if show_scope {
        for _ in 0..SCOPE_HEIGHT {
            doc.push(Line::from(""));
        }
    }

    // Input (hidden when a dialog is open).
    if !app.config_open && !app.help_open && !app.project_open && !app.ace_explorer_open {
        doc.push(separator_line(app, width));
        doc.push(input_line(app));
        doc.push(separator_line(app, width));

        if is_busy {
            doc.push(Line::from(""));
            doc.push(Line::from(Span::styled(
                "  esc to interrupt",
                Style::default().fg(theme::MUTED),
            )));
        } else if app.bash_mode() {
            doc.push(Line::from(Span::styled(
                "  ! for bash mode",
                Style::default().fg(theme::LOVE),
            )));
        } else if app.input.is_empty() && !app.shortcuts_open {
            doc.push(Line::from(Span::styled(
                "  ? for shortcuts",
                Style::default().fg(theme::MUTED),
            )));
        }
    }

    // Shortcuts panel.
    if app.shortcuts_open {
        doc.extend(shortcuts_lines(width));
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
        doc.extend(crate::screens::help::build_lines(app, width));
    }

    // Project panel.
    if app.project_open {
        doc.extend(crate::screens::project::build_lines(app, width));
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
    if show_scope && !app.scope_frames.is_empty() {
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
