use ratatui::{
    style::{Modifier, Style},
    text::{Line, Span},
};

use crate::theme;

// ---------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HelpTab {
    General,
    Commands,
}

impl HelpTab {
    pub const ALL: &'static [HelpTab] = &[HelpTab::General, HelpTab::Commands];

    pub fn label(self) -> &'static str {
        match self {
            HelpTab::General => "general",
            HelpTab::Commands => "commands",
        }
    }
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub struct HelpState {
    pub active_tab: usize,
}

impl HelpState {
    pub fn new() -> Self {
        Self { active_tab: 0 }
    }

    pub fn current_tab(&self) -> HelpTab {
        HelpTab::ALL[self.active_tab]
    }

    pub fn next_tab(&mut self) {
        self.active_tab = (self.active_tab + 1) % HelpTab::ALL.len();
    }

    pub fn prev_tab(&mut self) {
        if self.active_tab == 0 {
            self.active_tab = HelpTab::ALL.len() - 1;
        } else {
            self.active_tab -= 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Build lines for embedding in the scrollable document
// ---------------------------------------------------------------------------

pub fn build_lines(state: &HelpState, width: u16) -> Vec<Line<'static>> {
    let bold = Modifier::BOLD;
    let mut lines: Vec<Line<'static>> = Vec::new();

    // Separator
    lines.push(Line::from(Span::styled(
        "─".repeat(width as usize),
        Style::default().fg(theme::MUTED),
    )));

    // Version + tab bar
    lines.push(Line::from(""));
    let mut tab_bar: Vec<Span<'static>> = vec![
        Span::styled(
            format!("  Timbal v{}", crate::VERSION),
            Style::default().fg(theme::IRIS).add_modifier(bold),
        ),
        Span::raw("  "),
    ];

    for (i, tab) in HelpTab::ALL.iter().enumerate() {
        if i == state.active_tab {
            tab_bar.push(Span::styled(
                format!(" {} ", tab.label()),
                Style::default()
                    .fg(theme::TEXT)
                    .bg(theme::SURFACE)
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
        "  (\u{2190}/\u{2192} or tab to cycle)",
        Style::default().fg(theme::MUTED),
    ));

    lines.push(Line::from(tab_bar));
    lines.push(Line::from(""));

    // Tab content
    match state.current_tab() {
        HelpTab::General => build_general(&mut lines),
        HelpTab::Commands => build_commands(&mut lines),
    }

    // Esc to cancel
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "  Esc to cancel",
        Style::default().fg(theme::MUTED),
    )));
    lines.push(Line::from(""));

    lines
}

// ---------------------------------------------------------------------------
// Tab: General
// ---------------------------------------------------------------------------

fn build_general(lines: &mut Vec<Line<'static>>) {
    let bold = Modifier::BOLD;

    lines.push(Line::from(Span::styled(
        "  Timbal understands your codebase, orchestrates AI agents, and executes",
        Style::default().fg(theme::TEXT),
    )));
    lines.push(Line::from(Span::styled(
        "  commands — right from your terminal.",
        Style::default().fg(theme::TEXT),
    )));
    lines.push(Line::from(""));

    lines.push(Line::from(Span::styled(
        "  What I can help with:",
        Style::default().fg(theme::TEXT).add_modifier(bold),
    )));
    for item in [
        "Building and running AI agents",
        "Debugging and fixing bugs",
        "Adding new features",
        "Refactoring and code review",
        "Explaining code",
        "Running tests and builds",
        "Git operations",
    ] {
        lines.push(Line::from(vec![
            Span::styled("  - ", Style::default().fg(theme::MUTED)),
            Span::styled(item, Style::default().fg(theme::TEXT)),
        ]));
    }
    lines.push(Line::from(""));

    // Shortcuts (inline in general tab)
    lines.push(Line::from(Span::styled(
        "  Shortcuts",
        Style::default().fg(theme::TEXT).add_modifier(bold),
    )));

    let shortcuts: &[(&str, &str)] = &[
        ("/ for commands", "ctrl + c to quit"),
        ("esc to interrupt", "\u{2191}/\u{2193} to navigate palette"),
    ];
    for (left, right) in shortcuts {
        lines.push(Line::from(vec![
            Span::styled(
                format!("  {:<34}", left),
                Style::default().fg(theme::FOAM),
            ),
            Span::styled(*right, Style::default().fg(theme::FOAM)),
        ]));
    }
    lines.push(Line::from(""));

    lines.push(Line::from(Span::styled(
        "  For more help: https://docs.timbal.ai",
        Style::default().fg(theme::MUTED),
    )));
}

// ---------------------------------------------------------------------------
// Tab: Commands
// ---------------------------------------------------------------------------

fn build_commands(lines: &mut Vec<Line<'static>>) {
    let bold = Modifier::BOLD;

    lines.push(Line::from(Span::styled(
        "  Available commands:",
        Style::default().fg(theme::TEXT).add_modifier(bold),
    )));
    lines.push(Line::from(""));

    let commands: &[(&str, &str)] = &[
        ("/configure", "Set up API key and credentials"),
        ("/clear", "Clear conversation history"),
        ("/help", "Show this help panel"),
        ("/quit", "Exit Timbal"),
    ];
    for (cmd, desc) in commands {
        lines.push(Line::from(vec![
            Span::styled(format!("  {:<26}", cmd), Style::default().fg(theme::IRIS)),
            Span::styled(*desc, Style::default().fg(theme::SUBTLE)),
        ]));
    }
    lines.push(Line::from(""));

    lines.push(Line::from(Span::styled(
        "  Type / to open the command palette and filter commands.",
        Style::default().fg(theme::MUTED),
    )));
}
