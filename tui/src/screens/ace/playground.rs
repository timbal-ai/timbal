use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Paragraph, Wrap},
};

use super::state::{AceExplorerState, PlaygroundLine, PlaygroundPanel};
use super::widgets::render_tab_bar;
use crate::theme;
use crate::widgets::spinner::Spinner;

pub fn render_playground(state: &mut AceExplorerState, frame: &mut Frame, area: Rect, spinner_tick: usize) {
    let footer_height: u16 = 2;

    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2),             // tab bar
            Constraint::Length(2),             // gap
            Constraint::Min(1),               // three-panel content
            Constraint::Length(footer_height), // footer
        ])
        .split(area);

    let tab_area = layout[0];
    let content_area = layout[2];
    let footer_area = layout[3];

    render_tab_bar(state, frame, tab_area);

    // Three-panel horizontal split: chats (14%) | feed (56%) | config (30%)
    let panels = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(15), // chats
            Constraint::Percentage(55), // feed
            Constraint::Percentage(30), // config
        ])
        .split(content_area);

    let chats_area = panels[0];
    let feed_area = panels[1];
    let config_area = panels[2];

    // Store panel boundaries for mouse scroll routing.
    state.playground_feed_x = feed_area.x;
    state.playground_config_x = config_area.x;

    render_chats_panel(state, frame, chats_area);
    render_feed_panel(state, frame, feed_area, spinner_tick);
    render_config_panel(state, frame, config_area);

    // Footer.
    let controls = if state.playground_cfg_editing {
        "Type to edit  ·  Enter confirm  ·  Esc cancel"
    } else if state.playground_focused {
        "Enter send  ·  Esc unfocus"
    } else if state.playground_running {
        "Streaming...  ·  h/l panel  ·  Tab/⇧Tab tabs  ·  Esc back"
    } else {
        match state.playground_panel {
            PlaygroundPanel::Chats => {
                "↑/↓ select  ·  n new  ·  h/l panel  ·  Tab/⇧Tab tabs  ·  Esc back"
            }
            PlaygroundPanel::Feed => {
                "Enter focus  ·  ↑/↓ scroll  ·  h/l panel  ·  Tab/⇧Tab tabs  ·  Esc back"
            }
            PlaygroundPanel::Config => {
                "↑/↓ fields  ·  Enter edit  ·  h/l panel  ·  Tab/⇧Tab tabs  ·  Esc back"
            }
        }
    };
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(""),
            Line::from(Span::styled(
                format!("  {controls}"),
                Style::default().fg(theme::MUTED),
            )),
        ]),
        footer_area,
    );
}

// ---------------------------------------------------------------------------
// Chats panel (left)
// ---------------------------------------------------------------------------

fn render_chats_panel(state: &AceExplorerState, frame: &mut Frame, area: Rect) {
    let bold = Modifier::BOLD;
    let is_focused =
        state.playground_panel == PlaygroundPanel::Chats && !state.playground_focused;
    let header_color = if is_focused {
        theme::PINE
    } else {
        theme::SUBTLE
    };

    let mut lines: Vec<Line<'static>> = Vec::new();
    lines.push(Line::from(Span::styled(
        "  Chats",
        Style::default().fg(header_color).add_modifier(bold),
    )));
    lines.push(Line::from(""));

    for (i, chat) in state.playground_chats.iter().enumerate() {
        let is_active = i == state.playground_active_chat;
        let is_selected = is_focused && is_active;

        if is_selected {
            lines.push(Line::from(vec![
                Span::styled("  > ", Style::default().fg(theme::PINE)),
                Span::styled(
                    chat.label.clone(),
                    Style::default().fg(theme::PINE).add_modifier(bold),
                ),
            ]));
        } else if is_active {
            lines.push(Line::from(vec![
                Span::styled("    ", Style::default()),
                Span::styled(chat.label.clone(), Style::default().fg(theme::TEXT)),
            ]));
        } else {
            lines.push(Line::from(vec![
                Span::styled("    ", Style::default()),
                Span::styled(
                    chat.label.clone(),
                    Style::default().fg(theme::MUTED),
                ),
            ]));
        }
    }

    frame.render_widget(Paragraph::new(lines), area);
}

// ---------------------------------------------------------------------------
// Feed panel (center)
// ---------------------------------------------------------------------------

fn render_feed_panel(state: &mut AceExplorerState, frame: &mut Frame, area: Rect, spinner_tick: usize) {
    let bold = Modifier::BOLD;
    let is_focused =
        state.playground_panel == PlaygroundPanel::Feed || state.playground_focused;
    let header_color = if is_focused {
        theme::PINE
    } else {
        theme::SUBTLE
    };

    let input_height: u16 = 3;

    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(1),              // output
            Constraint::Length(1),            // gap
            Constraint::Length(input_height), // input box
        ])
        .split(area);

    let output_area = layout[0];
    let input_area = layout[2];

    // --- Output ---
    let mut output_lines: Vec<Line<'static>> = Vec::new();

    let spec_display = state
        .import_spec
        .as_deref()
        .unwrap_or("(no import spec)");

    let chat_output = state.playground_output();
    if chat_output.is_empty() && !state.playground_running {
        output_lines.push(Line::from(Span::styled(
            format!("  {spec_display}"),
            Style::default().fg(header_color).add_modifier(bold),
        )));
        output_lines.push(Line::from(""));
        output_lines.push(Line::from(""));
        output_lines.push(Line::from(Span::styled(
            "  Type a prompt and press Enter.",
            Style::default().fg(theme::MUTED),
        )));
    } else {
        output_lines.push(Line::from(Span::styled(
            format!("  {spec_display}"),
            Style::default().fg(header_color).add_modifier(bold),
        )));
        output_lines.push(Line::from(""));

        let spinner = Spinner::new(spinner_tick);
        let is_running = state.playground_running;
        let mut in_assistant_block = false;

        for line in chat_output {
            match line {
                PlaygroundLine::User(prompt) => {
                    in_assistant_block = false;
                    output_lines.push(Line::from(vec![
                        Span::styled("  ❯ ", Style::default().fg(theme::IRIS).add_modifier(bold)),
                        Span::styled(
                            prompt.clone(),
                            Style::default().fg(theme::TEXT).add_modifier(bold),
                        ),
                    ]));
                    output_lines.push(Line::from(""));
                }
                PlaygroundLine::Text(t) => {
                    if !in_assistant_block {
                        let indicator = if is_running {
                            format!("{}", spinner.frame())
                        } else {
                            "●".to_string()
                        };
                        let color = if is_running { theme::SUBTLE } else { theme::PINE };
                        // First line: "  ● text" with indicator colored.
                        // Pre-wrap to panel width so indent is consistent.
                        let prefix = format!("  {indicator} ");
                        let content_width = output_area.width as usize;
                        let avail = content_width.saturating_sub(prefix.len());
                        if avail > 0 && t.len() > avail {
                            // First visual line with indicator.
                            output_lines.push(Line::from(vec![
                                Span::styled(prefix, Style::default().fg(color)),
                                Span::styled(
                                    t[..avail].to_string(),
                                    Style::default().fg(theme::TEXT),
                                ),
                            ]));
                            // Remaining text wrapped with 4-space indent.
                            let rest = &t[avail..];
                            let indent_avail = content_width.saturating_sub(4);
                            if indent_avail > 0 {
                                for chunk in rest.as_bytes().chunks(indent_avail) {
                                    let s = String::from_utf8_lossy(chunk);
                                    output_lines.push(Line::from(Span::styled(
                                        format!("    {s}"),
                                        Style::default().fg(theme::TEXT),
                                    )));
                                }
                            }
                        } else {
                            output_lines.push(Line::from(vec![
                                Span::styled(prefix, Style::default().fg(color)),
                                Span::styled(
                                    t.clone(),
                                    Style::default().fg(theme::TEXT),
                                ),
                            ]));
                        }
                        in_assistant_block = true;
                    } else {
                        output_lines.push(Line::from(Span::styled(
                            format!("    {t}"),
                            Style::default().fg(theme::TEXT),
                        )));
                    }
                }
                PlaygroundLine::Error(e) => {
                    in_assistant_block = false;
                    output_lines.push(Line::from(Span::styled(
                        format!("    {e}"),
                        Style::default().fg(theme::LOVE),
                    )));
                }
                PlaygroundLine::Status(s) => {
                    in_assistant_block = false;
                    output_lines.push(Line::from(Span::styled(
                        format!("  {s}"),
                        Style::default().fg(theme::SUBTLE),
                    )));
                }
                PlaygroundLine::Thinking => {
                    in_assistant_block = false;
                    output_lines.push(Line::from(Span::styled(
                        format!("  {} Computing...", spinner.frame()),
                        Style::default().fg(theme::SUBTLE),
                    )));
                }
                PlaygroundLine::Stats {
                    duration_ms,
                    usage,
                } => {
                    in_assistant_block = false;
                    let dur = if *duration_ms >= 1000 {
                        format!("{:.1}s", *duration_ms as f64 / 1000.0)
                    } else {
                        format!("{}ms", duration_ms)
                    };
                    let stats_text = if usage.is_empty() {
                        dur
                    } else {
                        format!("{dur}  ·  {usage}")
                    };
                    output_lines.push(Line::from(Span::styled(
                        format!("    {stats_text}"),
                        Style::default().fg(theme::MUTED),
                    )));
                }
                PlaygroundLine::TextDelta(_) => {}
            }
        }
    }

    let output_height = output_area.height as u16;
    let output_width = output_area.width as usize;
    let wrapped_line_count: u16 = output_lines
        .iter()
        .map(|line| {
            let w: usize = line.spans.iter().map(|s| s.content.len()).sum();
            if output_width == 0 {
                1
            } else {
                ((w.max(1) + output_width - 1) / output_width) as u16
            }
        })
        .sum();
    state.playground_feed_max_scroll = wrapped_line_count.saturating_sub(output_height);
    if state.playground_feed_scroll == u16::MAX {
        state.playground_feed_scroll = state.playground_feed_max_scroll;
    } else if state.playground_feed_scroll > state.playground_feed_max_scroll {
        state.playground_feed_scroll = state.playground_feed_max_scroll;
    }

    frame.render_widget(
        Paragraph::new(output_lines)
            .wrap(Wrap { trim: false })
            .scroll((state.playground_feed_scroll, 0)),
        output_area,
    );

    // --- Input box (2-space left padding to align with content) ---
    let box_color = if state.playground_focused {
        theme::PINE
    } else {
        theme::SUBTLE
    };
    let box_style = Style::default().fg(box_color);
    let inner_w = (input_area.width as usize).saturating_sub(6); // 2 pad + 2 border + 2 space

    let top = Line::from(vec![
        Span::raw("  "),
        Span::styled(format!("╭{}╮", "─".repeat(inner_w + 2)), box_style),
    ]);

    let placeholder = if state.playground_running {
        "Running..."
    } else {
        "Type a prompt..."
    };

    let mut row: Vec<Span<'static>> = vec![Span::raw("  "), Span::styled("│ ", box_style)];
    let cursor_style = Style::default().fg(theme::BASE).bg(theme::TEXT);

    if state.playground_focused {
        let cursor = state.playground_cursor;
        let before = &state.playground_input[..cursor];
        let (cursor_ch, after) = if cursor < state.playground_input.len() {
            let rest = &state.playground_input[cursor..];
            let ch_len = rest.chars().next().map(|c| c.len_utf8()).unwrap_or(0);
            (
                state.playground_input[cursor..cursor + ch_len].to_string(),
                state.playground_input[cursor + ch_len..].to_string(),
            )
        } else {
            (" ".to_string(), String::new())
        };
        let used = before.len() + 1 + after.len();
        row.push(Span::styled(
            before.to_string(),
            Style::default().fg(theme::TEXT),
        ));
        row.push(Span::styled(cursor_ch, cursor_style));
        row.push(Span::styled(after, Style::default().fg(theme::TEXT)));
        row.push(Span::raw(" ".repeat(inner_w.saturating_sub(used))));
    } else if !state.playground_input.is_empty() {
        let display = &state.playground_input;
        row.push(Span::styled(
            display.clone(),
            Style::default().fg(theme::TEXT),
        ));
        row.push(Span::raw(
            " ".repeat(inner_w.saturating_sub(display.len())),
        ));
    } else {
        row.push(Span::styled(
            placeholder.to_string(),
            Style::default().fg(theme::MUTED),
        ));
        row.push(Span::raw(
            " ".repeat(inner_w.saturating_sub(placeholder.len())),
        ));
    }
    row.push(Span::styled(" │", box_style));
    let mid = Line::from(row);

    let bot = Line::from(vec![
        Span::raw("  "),
        Span::styled(format!("╰{}╯", "─".repeat(inner_w + 2)), box_style),
    ]);

    frame.render_widget(Paragraph::new(vec![top, mid, bot]), input_area);
}

// ---------------------------------------------------------------------------
// Config panel (right)
// ---------------------------------------------------------------------------

fn render_config_panel(state: &AceExplorerState, frame: &mut Frame, area: Rect) {
    let bold = Modifier::BOLD;
    let is_focused =
        state.playground_panel == PlaygroundPanel::Config && !state.playground_focused;
    let header_color = if is_focused {
        theme::PINE
    } else {
        theme::SUBTLE
    };

    // Panel width in display columns. Box uses 2-col left margin, 1-col borders each side.
    // Layout: "  ╭─ LABEL ───╮" = 2 + 1 + 1 + label + dashes + 1 = width
    // Content: "  │ value     │" = 2 + 1 + 1 + content + 1 + 1 = width
    // So content area = width - 6 display columns.
    let w = area.width as usize;
    let box_content = w.saturating_sub(6);

    let mut lines: Vec<Line<'static>> = Vec::new();
    lines.push(Line::from(Span::styled(
        "  Config",
        Style::default().fg(header_color).add_modifier(bold),
    )));
    lines.push(Line::from(""));

    for (i, label) in AceExplorerState::PLAYGROUND_CFG_LABELS.iter().enumerate() {
        let is_active = is_focused && state.playground_cfg_field == i;
        let is_typing = is_active && state.playground_cfg_editing;

        let border_color = if is_active {
            theme::PINE
        } else {
            theme::MUTED
        };
        let border_style = Style::default().fg(border_color);
        let label_style = if is_active {
            Style::default().fg(theme::PINE).add_modifier(bold)
        } else {
            Style::default().fg(theme::SUBTLE)
        };

        // Top: "  ╭─ LABEL ───...───╮"
        // Fixed cols: "  ╭─" (4) + " LABEL " + "╮" (1) = 5 + label_display_len
        let label_display = format!(" {} ", label);
        let label_cols = label_display.chars().count();
        let top_dashes = w.saturating_sub(5 + label_cols);
        lines.push(Line::from(vec![
            Span::styled(format!("  ╭─"), border_style),
            Span::styled(label_display, label_style),
            Span::styled(format!("{}╮", "─".repeat(top_dashes)), border_style),
        ]));

        // Content rows: "  │ value...pad │" — wraps to multiple rows if needed.
        if is_typing {
            let cursor = state.playground_cfg_cursor;
            let input = &state.playground_cfg_input;
            // Render the full input with cursor, wrapped into box_content-wide rows.
            let full_text = format!(
                "{}{}",
                &input[..cursor],
                if cursor < input.len() { &input[cursor..] } else { "" }
            );
            let chars: Vec<char> = full_text.chars().collect();
            let cursor_char_idx = input[..cursor].chars().count();
            let row_width = box_content.max(1);
            let cursor_style = Style::default().fg(theme::BASE).bg(theme::TEXT);

            if chars.is_empty() {
                // Empty input — just show cursor.
                let pad = box_content.saturating_sub(1);
                lines.push(Line::from(vec![
                    Span::styled("  │ ", border_style),
                    Span::styled(" ", cursor_style),
                    Span::raw(" ".repeat(pad)),
                    Span::styled(" │", border_style),
                ]));
            } else {
                let total_rows = (chars.len() + row_width - 1) / row_width;
                for row in 0..total_rows {
                    let start = row * row_width;
                    let end = (start + row_width).min(chars.len());
                    let mut spans: Vec<Span<'static>> = vec![
                        Span::styled("  │ ", border_style),
                    ];
                    for ci in start..end {
                        let ch: String = chars[ci].to_string();
                        if ci == cursor_char_idx {
                            spans.push(Span::styled(ch, cursor_style));
                        } else {
                            spans.push(Span::styled(ch, Style::default().fg(theme::TEXT)));
                        }
                    }
                    // Cursor at end of text.
                    if cursor_char_idx == chars.len() && row == total_rows - 1 {
                        spans.push(Span::styled(" ", cursor_style));
                        let used = end - start + 1;
                        spans.push(Span::raw(" ".repeat(box_content.saturating_sub(used))));
                    } else {
                        let used = end - start;
                        spans.push(Span::raw(" ".repeat(box_content.saturating_sub(used))));
                    }
                    spans.push(Span::styled(" │", border_style));
                    lines.push(Line::from(spans));
                }
            }
        } else {
            let val = state.playground_cfg_value(i);
            let (display, style) = if val.is_empty() {
                ("(default)".to_string(), Style::default().fg(theme::MUTED))
            } else {
                (val.to_string(), Style::default().fg(theme::TEXT))
            };
            let chars: Vec<char> = display.chars().collect();
            let row_width = box_content.max(1);
            if chars.is_empty() {
                lines.push(Line::from(vec![
                    Span::styled("  │ ", border_style),
                    Span::raw(" ".repeat(box_content)),
                    Span::styled(" │", border_style),
                ]));
            } else {
                let total_rows = (chars.len() + row_width - 1) / row_width;
                for row in 0..total_rows {
                    let start = row * row_width;
                    let end = (start + row_width).min(chars.len());
                    let row_text: String = chars[start..end].iter().collect();
                    let pad = box_content.saturating_sub(end - start);
                    lines.push(Line::from(vec![
                        Span::styled("  │ ", border_style),
                        Span::styled(row_text, style),
                        Span::raw(" ".repeat(pad)),
                        Span::styled(" │", border_style),
                    ]));
                }
            }
        }

        // Bottom: "  ╰───...───╯"
        // Fixed cols: "  ╰" (3) + "╯" (1) = 4, dashes fill the rest
        let bot_dashes = w.saturating_sub(4);
        lines.push(Line::from(Span::styled(
            format!("  ╰{}╯", "─".repeat(bot_dashes)),
            border_style,
        )));
        lines.push(Line::from(""));
    }

    let config_height = area.height as u16;
    let config_line_count = lines.len() as u16;
    let max_scroll = config_line_count.saturating_sub(config_height);
    let scroll = state.playground_config_scroll.min(max_scroll);

    frame.render_widget(
        Paragraph::new(lines).scroll((scroll, 0)),
        area,
    );
}
