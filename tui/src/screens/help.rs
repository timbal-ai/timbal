use ratatui::{
    style::{Modifier, Style},
    text::{Line, Span},
};

use crate::app::App;
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

pub fn build_lines(app: &App, width: u16) -> Vec<Line<'static>> {
    let state = &app.help_state;
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
        HelpTab::Commands => build_commands(app, &mut lines),
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
        "  Build, deploy, and manage your AI workforce — right from the terminal.",
        Style::default().fg(theme::FOAM),
    )));
    lines.push(Line::from(Span::styled(
        "  Powered by ACE \u{2014} deterministic control for AI agents.",
        Style::default().fg(theme::SUBTLE),
    )));
    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::styled("  Software lovingly crafted by ", Style::default().fg(theme::SUBTLE)),
        Span::styled("Timbal", Style::default().fg(theme::GOLD)),
    ]));
    lines.push(Line::from(""));

    // Shortcuts (inline in general tab)
    lines.push(Line::from(Span::styled(
        "  Shortcuts",
        Style::default().fg(theme::TEXT).add_modifier(bold),
    )));

    let shortcuts: &[(&str, &str)] = &[
        ("/ for commands", "ctrl + c to quit"),
        ("! for bash mode", "esc to cancel"),
        ("\u{2191}/\u{2193} to navigate palette", "shift + click to select text"),
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
        "  Docs: https://docs.timbal.ai",
        Style::default().fg(theme::MUTED),
    )));
    lines.push(Line::from(Span::styled(
        "  Issues: https://github.com/timbal-ai/timbal/issues",
        Style::default().fg(theme::MUTED),
    )));
}

// ---------------------------------------------------------------------------
// Tab: Commands
// ---------------------------------------------------------------------------

fn build_commands(app: &App, lines: &mut Vec<Line<'static>>) {
    let bold = Modifier::BOLD;

    lines.push(Line::from(Span::styled(
        "  Available commands:",
        Style::default().fg(theme::TEXT).add_modifier(bold),
    )));
    lines.push(Line::from(""));

    // Pull commands from the registry so this list stays in sync automatically.
    let all = app.filter_commands_all();
    for cmd in &all {
        let meta = cmd.meta();
        lines.push(Line::from(vec![
            Span::styled(format!("  {:<26}", meta.name), Style::default().fg(theme::IRIS)),
            Span::styled(meta.description, Style::default().fg(theme::SUBTLE)),
        ]));
    }
    lines.push(Line::from(""));

    lines.push(Line::from(Span::styled(
        "  Type / to open the command palette and filter commands.",
        Style::default().fg(theme::MUTED),
    )));
}
