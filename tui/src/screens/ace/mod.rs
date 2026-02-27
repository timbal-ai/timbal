mod evals;
mod playground;
mod policies;
pub mod state;
mod variables;
pub mod widgets;

pub use state::{AceExplorerState, PlaygroundChat, PlaygroundLine, PlaygroundPanel};

use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::Style,
    text::{Line, Span},
    widgets::{Paragraph, Wrap},
};

use crate::app::App;
use crate::theme;
use widgets::{render_search_box, render_tab_bar};

// ---------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AceExplorerTab {
    Variables,
    Policies,
    Evals,
    Playground,
}

impl AceExplorerTab {
    pub const ALL: &'static [AceExplorerTab] = &[
        AceExplorerTab::Playground,
        AceExplorerTab::Variables,
        AceExplorerTab::Policies,
        AceExplorerTab::Evals,
    ];

    pub fn label(self) -> &'static str {
        match self {
            AceExplorerTab::Variables => "Variables",
            AceExplorerTab::Policies => "Policies",
            AceExplorerTab::Evals => "Evals",
            AceExplorerTab::Playground => "Playground",
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

    // Playground tab has a different layout.
    if state.current_tab() == AceExplorerTab::Playground {
        playground::render_playground(state, frame, area, app.spinner_tick);
        return;
    }

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

    state.right_panel_x = horiz[1].x;

    let visible = state.visible_count;
    let (left_lines, right_lines) = match state.current_tab() {
        AceExplorerTab::Variables => variables::build_variables_content(state, visible),
        AceExplorerTab::Policies => policies::build_policies_content(state, visible),
        AceExplorerTab::Evals => evals::build_evals_content(state, visible),
        AceExplorerTab::Playground => unreachable!(),
    };

    // Compute max scroll for the right panel.
    let right_panel_height = horiz[1].height as u16;
    let right_line_count = right_lines.len() as u16;
    state.right_max_scroll = right_line_count.saturating_sub(right_panel_height);
    state.right_scroll = state.right_scroll.min(state.right_max_scroll);

    frame.render_widget(Paragraph::new(left_lines), horiz[0]);
    frame.render_widget(
        Paragraph::new(right_lines)
            .wrap(Wrap { trim: false })
            .scroll((state.right_scroll, 0)),
        horiz[1],
    );

    // Footer.
    let total_items = match state.current_tab() {
        AceExplorerTab::Variables => state.filtered_variables().len(),
        AceExplorerTab::Policies => state.filtered_policies().len(),
        AceExplorerTab::Evals => state.filtered_evals().len(),
        AceExplorerTab::Playground => 0,
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
    } else if state.editing_right && state.current_tab() == AceExplorerTab::Evals {
        "↑/↓ scroll  ·  Esc back to list"
    } else if state.editing_right {
        "↑/↓ fields  ·  Enter edit  ·  Esc back to list"
    } else if state.search_focused {
        "Type to filter  ·  Esc to clear"
    } else if state.current_tab() == AceExplorerTab::Evals {
        "↑/↓ navigate  ·  Enter view  ·  / search  ·  Esc back"
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
