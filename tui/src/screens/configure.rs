use ratatui::{
    layout::{Constraint, Flex, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

use crate::config::{mask_key, TimbalConfig};
use crate::theme;

fn resolve_profile(profile_arg: Option<String>) -> String {
    profile_arg
        .or_else(|| std::env::var("TIMBAL_PROFILE").ok())
        .unwrap_or_else(|| "default".to_string())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigField {
    ApiKey,
    Org,
    BaseUrl,
}

impl ConfigField {
    fn all() -> &'static [ConfigField] {
        &[ConfigField::ApiKey, ConfigField::Org, ConfigField::BaseUrl]
    }

    fn label(self) -> &'static str {
        match self {
            ConfigField::ApiKey => "Timbal API Key",
            ConfigField::Org => "Organization ID",
            ConfigField::BaseUrl => "Platform Base URL",
        }
    }

    fn hint(self) -> &'static str {
        match self {
            ConfigField::ApiKey => "https://app.timbal.ai/profile/api-keys",
            ConfigField::Org => "Your organization identifier",
            ConfigField::BaseUrl => "default: https://api.timbal.ai",
        }
    }

    fn is_secret(self) -> bool {
        self == ConfigField::ApiKey
    }
}

pub struct ConfigureState {
    pub profile: String,
    pub current_field: usize,
    pub values: [String; 3], // ApiKey, Org, BaseUrl
    pub input: String,
    pub saved: bool,
    pub error: Option<String>,
    /// Existing values loaded from disk (for display only).
    pub existing: TimbalConfig,
}

impl ConfigureState {
    pub fn new() -> Self {
        Self::with_profile(None)
    }

    pub fn with_profile(profile: Option<String>) -> Self {
        let profile = resolve_profile(profile);
        let existing = TimbalConfig::load(&profile);
        let base_url_default = existing
            .base_url
            .clone()
            .unwrap_or_else(|| "https://api.timbal.ai".to_string());

        Self {
            profile,
            current_field: 0,
            values: [String::new(), String::new(), base_url_default],
            input: String::new(),
            saved: false,
            error: None,
            existing,
        }
    }

    pub fn current_field(&self) -> ConfigField {
        ConfigField::all()[self.current_field]
    }

    /// Commit the current input and advance to the next field (or save).
    pub fn advance(&mut self) -> bool {
        let val = self.input.trim().to_string();
        self.values[self.current_field] = val;
        self.input.clear();

        if self.current_field + 1 < ConfigField::all().len() {
            self.current_field += 1;
            // Pre-fill input with existing value for non-secret fields.
            let field = self.current_field();
            if !field.is_secret() {
                self.input = self.values[self.current_field].clone();
            }
            false
        } else {
            // All fields filled — save.
            self.save();
            true
        }
    }

    fn save(&mut self) {
        let api_key = self.values[0].trim().to_string();
        let org = self.values[1].trim().to_string();
        let base_url = self.values[2].trim().to_string();

        let cfg = TimbalConfig {
            api_key: if api_key.is_empty() { None } else { Some(api_key) },
            org: if org.is_empty() { None } else { Some(org) },
            base_url: if base_url.is_empty() { None } else { Some(base_url) },
        };

        match cfg.save(&self.profile) {
            Ok(()) => {
                self.saved = true;
                self.error = None;
            }
            Err(e) => {
                self.error = Some(format!("Failed to save: {e}"));
            }
        }
    }

    /// Display value for a field (masked if secret, placeholder if empty).
    fn display_existing(&self, field: ConfigField) -> String {
        let existing = match field {
            ConfigField::ApiKey => self.existing.api_key.as_deref(),
            ConfigField::Org => self.existing.org.as_deref(),
            ConfigField::BaseUrl => self.existing.base_url.as_deref(),
        };
        match existing {
            Some(v) if field.is_secret() => mask_key(v),
            Some(v) => v.to_string(),
            None => String::new(),
        }
    }
}

pub fn render(state: &mut ConfigureState, frame: &mut Frame) {
    let area = frame.area();

    let [form_area] = Layout::vertical([Constraint::Min(4)])
        .flex(Flex::Center)
        .areas(area);
    let [form_area] = Layout::horizontal([Constraint::Length(70)])
        .flex(Flex::Center)
        .areas(form_area);

    let title = format!(" Configure Timbal  [profile: {}] ", state.profile);
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::MUTED))
        .title(Span::styled(
            title,
            Style::default().fg(theme::IRIS).add_modifier(Modifier::BOLD),
        ));

    let inner = block.inner(form_area);
    frame.render_widget(block, form_area);

    if state.saved {
        let lines = vec![
            Line::from(""),
            Line::from(Span::styled(
                "  ✓ Credentials saved to ~/.timbal/",
                Style::default().fg(theme::FOAM).add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from(Span::styled(
                "  Press Esc to return",
                Style::default().fg(theme::MUTED),
            )),
        ];
        frame.render_widget(Paragraph::new(lines), inner);
        return;
    }

    if let Some(ref err) = state.error.clone() {
        let lines = vec![
            Line::from(""),
            Line::from(Span::styled(
                format!("  ✗ {err}"),
                Style::default().fg(theme::LOVE),
            )),
            Line::from(""),
            Line::from(Span::styled(
                "  Press Esc to return",
                Style::default().fg(theme::MUTED),
            )),
        ];
        frame.render_widget(Paragraph::new(lines), inner);
        return;
    }

    let mut lines: Vec<Line> = vec![Line::from("")];

    // Render each field.
    for (i, &field) in ConfigField::all().iter().enumerate() {
        let is_active = i == state.current_field;

        let label_style = if is_active {
            Style::default().fg(theme::TEXT).add_modifier(Modifier::BOLD)
        } else if i < state.current_field {
            Style::default().fg(theme::SUBTLE)
        } else {
            Style::default().fg(theme::MUTED)
        };

        // Label + hint.
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled(field.label(), label_style),
            Span::styled(
                format!("  {}", field.hint()),
                Style::default().fg(theme::MUTED),
            ),
        ]));

        if is_active {
            // Active input line.
            let existing = state.display_existing(field);
            let placeholder = if existing.is_empty() {
                "".to_string()
            } else {
                format!("[{existing}] ")
            };

            let display = if field.is_secret() {
                "•".repeat(state.input.len())
            } else {
                state.input.clone()
            };

            lines.push(Line::from(vec![
                Span::styled("  ❯ ", Style::default().fg(theme::IRIS)),
                Span::styled(placeholder, Style::default().fg(theme::MUTED)),
                Span::styled(display, Style::default().fg(theme::TEXT)),
                Span::styled("█", Style::default().fg(theme::SUBTLE)),
            ]));
        } else if i < state.current_field {
            // Completed field — show submitted value.
            let val = &state.values[i];
            let display = if field.is_secret() && !val.is_empty() {
                mask_key(val)
            } else {
                val.clone()
            };
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled(
                    if display.is_empty() { "(skipped)".to_string() } else { display },
                    Style::default().fg(theme::FOAM),
                ),
            ]));
        }

        lines.push(Line::from(""));
    }

    lines.push(Line::from(Span::styled(
        "  Enter to confirm · Esc to cancel",
        Style::default().fg(theme::MUTED),
    )));

    frame.render_widget(Paragraph::new(lines), inner);
}

/// Build the config panel as lines (for embedding into a scrollable document).
pub fn build_lines(state: &ConfigureState, width: u16) -> Vec<Line<'static>> {
    let title = format!(" Configure Timbal [profile: {}] ", state.profile);
    let bold = Modifier::BOLD;

    let mut out: Vec<Line<'static>> = Vec::new();

    // Title separator line.
    let sep_len = width.saturating_sub(title.len() as u16) as usize;
    out.push(Line::from(vec![
        Span::styled(title.clone(), Style::default().fg(theme::IRIS).add_modifier(bold)),
        Span::styled("─".repeat(sep_len), Style::default().fg(theme::MUTED)),
    ]));

    if state.saved {
        out.push(Line::from(Span::styled(
            " ✓ Credentials saved to ~/.timbal/",
            Style::default().fg(theme::FOAM).add_modifier(bold),
        )));
        out.push(Line::from(Span::styled(
            " Press Enter to dismiss",
            Style::default().fg(theme::MUTED),
        )));
        return out;
    }

    if let Some(ref err) = state.error {
        out.push(Line::from(Span::styled(
            format!(" ✗ {err}"),
            Style::default().fg(theme::LOVE),
        )));
        out.push(Line::from(Span::styled(
            " Press Enter to dismiss",
            Style::default().fg(theme::MUTED),
        )));
        return out;
    }

    for (i, &field) in ConfigField::all().iter().enumerate() {
        let is_active = i == state.current_field;

        let label_style = if is_active {
            Style::default().fg(theme::TEXT).add_modifier(bold)
        } else if i < state.current_field {
            Style::default().fg(theme::SUBTLE)
        } else {
            Style::default().fg(theme::MUTED)
        };

        out.push(Line::from(vec![
            Span::raw(" "),
            Span::styled(field.label(), label_style),
            Span::styled(format!("  {}", field.hint()), Style::default().fg(theme::MUTED)),
        ]));

        if is_active {
            let existing = state.display_existing(field);
            let placeholder = if existing.is_empty() {
                "".to_string()
            } else {
                format!("[{existing}] ")
            };
            let display = if field.is_secret() {
                "•".repeat(state.input.len())
            } else {
                state.input.clone()
            };
            out.push(Line::from(vec![
                Span::styled(" ❯ ", Style::default().fg(theme::IRIS)),
                Span::styled(placeholder, Style::default().fg(theme::MUTED)),
                Span::styled(display, Style::default().fg(theme::TEXT)),
                Span::styled("█", Style::default().fg(theme::SUBTLE)),
            ]));
        } else if i < state.current_field {
            let val = &state.values[i];
            let display = if field.is_secret() && !val.is_empty() {
                mask_key(val)
            } else {
                val.clone()
            };
            out.push(Line::from(vec![
                Span::raw(" "),
                Span::styled(
                    if display.is_empty() { "(skipped)".to_string() } else { display },
                    Style::default().fg(theme::FOAM),
                ),
            ]));
        }

        out.push(Line::from(""));
    }

    out.push(Line::from(Span::styled(
        " Enter to confirm · Esc to dismiss",
        Style::default().fg(theme::MUTED),
    )));

    out
}

/// Render the config form inline into the given area (no centering, no fullscreen).
pub fn render_inline(state: &mut ConfigureState, frame: &mut Frame, area: Rect) {
    let title = format!(" Configure Timbal [profile: {}] ", state.profile);
    let block = Block::default()
        .borders(Borders::TOP)
        .border_style(Style::default().fg(theme::MUTED))
        .title(Span::styled(
            title,
            Style::default().fg(theme::IRIS).add_modifier(Modifier::BOLD),
        ));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    if state.saved {
        let lines = vec![
            Line::from(Span::styled(
                " ✓ Credentials saved to ~/.timbal/",
                Style::default().fg(theme::FOAM).add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::styled(
                " Press Enter to dismiss",
                Style::default().fg(theme::MUTED),
            )),
        ];
        frame.render_widget(Paragraph::new(lines), inner);
        return;
    }

    if let Some(ref err) = state.error.clone() {
        let lines = vec![
            Line::from(Span::styled(
                format!(" ✗ {err}"),
                Style::default().fg(theme::LOVE),
            )),
            Line::from(Span::styled(
                " Press Enter to dismiss",
                Style::default().fg(theme::MUTED),
            )),
        ];
        frame.render_widget(Paragraph::new(lines), inner);
        return;
    }

    let mut lines: Vec<Line> = Vec::new();

    for (i, &field) in ConfigField::all().iter().enumerate() {
        let is_active = i == state.current_field;

        let label_style = if is_active {
            Style::default().fg(theme::TEXT).add_modifier(Modifier::BOLD)
        } else if i < state.current_field {
            Style::default().fg(theme::SUBTLE)
        } else {
            Style::default().fg(theme::MUTED)
        };

        lines.push(Line::from(vec![
            Span::raw(" "),
            Span::styled(field.label(), label_style),
            Span::styled(format!("  {}", field.hint()), Style::default().fg(theme::MUTED)),
        ]));

        if is_active {
            let existing = state.display_existing(field);
            let placeholder = if existing.is_empty() {
                "".to_string()
            } else {
                format!("[{existing}] ")
            };

            let display = if field.is_secret() {
                "•".repeat(state.input.len())
            } else {
                state.input.clone()
            };

            lines.push(Line::from(vec![
                Span::styled(" ❯ ", Style::default().fg(theme::IRIS)),
                Span::styled(placeholder, Style::default().fg(theme::MUTED)),
                Span::styled(display, Style::default().fg(theme::TEXT)),
                Span::styled("█", Style::default().fg(theme::SUBTLE)),
            ]));
        } else if i < state.current_field {
            let val = &state.values[i];
            let display = if field.is_secret() && !val.is_empty() {
                mask_key(val)
            } else {
                val.clone()
            };
            lines.push(Line::from(vec![
                Span::raw(" "),
                Span::styled(
                    if display.is_empty() { "(skipped)".to_string() } else { display },
                    Style::default().fg(theme::FOAM),
                ),
            ]));
        }

        lines.push(Line::from(""));
    }

    lines.push(Line::from(Span::styled(
        " Enter to confirm · Esc to dismiss",
        Style::default().fg(theme::MUTED),
    )));

    frame.render_widget(Paragraph::new(lines), inner);
}
