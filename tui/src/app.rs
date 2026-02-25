use color_eyre::Result;
use ratatui::DefaultTerminal;

use crate::audio::{self, Frame};
use crate::commands;
use crate::event::{self, Action};
use crate::history::{self, Entry, EntryKind};
use crate::screens::configure::ConfigureState;
use crate::ui;

pub struct App {
    pub running: bool,
    pub input: String,
    pub thinking: bool,
    pub palette_selected: Option<usize>,
    pub configure_state: ConfigureState,
    pub config_open: bool,
    pub history: Vec<Entry>,
    pub scroll: u16,
    /// Precomputed vectorscope animation frames (decoded once at startup).
    pub scope_frames: Vec<Frame>,
    /// Current animation frame index (advances every N render ticks, wraps for looping).
    pub scope_tick: usize,
    /// Render tick counter for throttling the animation.
    render_tick: usize,
}

impl App {
    pub fn new() -> Result<Self> {
        let scope_frames = audio::load_frames();

        let mut app = Self {
            running: true,
            input: String::new(),
            thinking: false,
            palette_selected: None,
            configure_state: ConfigureState::new(),
            config_open: false,
            history: Vec::new(),
            scroll: 0,
            scope_frames,
            scope_tick: 0,
            render_tick: 0,
        };

        app.log(EntryKind::SessionStart);
        Ok(app)
    }

    fn log(&mut self, kind: EntryKind) {
        let entry = Entry::new(kind);
        history::append(&entry);
        self.history.push(entry);
        self.scroll = u16::MAX;
    }

    pub fn palette_open(&self) -> bool {
        !self.config_open && self.input.starts_with('/')
    }

    /// Get the current vectorscope frame (loops automatically).
    pub fn current_scope_frame(&self) -> &[(f64, f64)] {
        if self.scope_frames.is_empty() {
            return &[];
        }
        &self.scope_frames[self.scope_tick % self.scope_frames.len()]
    }

    /// Advance the animation tick. Only moves to the next frame every 4 render ticks (~15fps).
    pub fn advance_scope(&mut self) {
        self.render_tick += 1;
        if self.render_tick % 2 == 0 && !self.scope_frames.is_empty() {
            self.scope_tick = (self.scope_tick + 1) % self.scope_frames.len();
        }
    }

    pub fn run(&mut self, terminal: &mut DefaultTerminal) -> Result<()> {
        while self.running {
            // Advance animation when thinking.
            if self.thinking {
                self.advance_scope();
            }

            terminal.draw(|frame| ui::render(self, frame))?;

            if let Some(action) = event::poll()? {
                self.update(action);
            }
        }
        Ok(())
    }

    pub fn update(&mut self, action: Action) {
        if self.config_open {
            match action {
                Action::Cancel => {
                    self.log(EntryKind::ConfigureCancelled);
                    self.config_open = false;
                    self.configure_state = ConfigureState::new();
                }
                Action::Submit => {
                    if self.configure_state.saved || self.configure_state.error.is_some() {
                        if self.configure_state.saved {
                            let profile = self.configure_state.profile.clone();
                            self.log(EntryKind::ConfigureSaved(profile));
                        }
                        self.config_open = false;
                        self.configure_state = ConfigureState::new();
                    } else {
                        self.configure_state.advance();
                    }
                }
                Action::Backspace => {
                    self.configure_state.input.pop();
                }
                Action::Type(c) => {
                    self.configure_state.input.push(c);
                }
                Action::Quit => self.running = false,
                _ => {}
            }
            return;
        }

        match action {
            Action::Quit => self.running = false,

            Action::Type(c) => {
                self.input.push(c);
                if self.input.starts_with('/') {
                    self.palette_selected = Some(0);
                } else {
                    self.palette_selected = None;
                }
            }

            Action::Backspace => {
                self.input.pop();
                self.palette_selected = None;
            }

            Action::PaletteDown => {
                if self.palette_open() {
                    let count = commands::filter(&self.input).len();
                    if count > 0 {
                        self.palette_selected = Some(match self.palette_selected {
                            None => 0,
                            Some(i) => (i + 1).min(count - 1),
                        });
                    }
                } else {
                    self.scroll = self.scroll.saturating_add(3);
                }
            }

            Action::PaletteUp => {
                if self.palette_open() {
                    self.palette_selected = Some(match self.palette_selected {
                        None | Some(0) => 0,
                        Some(i) => i - 1,
                    });
                } else {
                    self.scroll = self.scroll.saturating_sub(3);
                }
            }

            Action::ScrollUp => {
                self.scroll = self.scroll.saturating_sub(10);
            }

            Action::ScrollDown => {
                self.scroll = self.scroll.saturating_add(10);
            }

            Action::Submit => {
                if self.palette_open() {
                    let matches = commands::filter(&self.input);
                    let idx = self.palette_selected.unwrap_or(0);
                    if let Some(cmd) = matches.get(idx) {
                        let arg = commands::parse_arg(&self.input, cmd.name);
                        let cmd_text = match arg {
                            Some(a) => format!("{} {}", cmd.name, a),
                            None => cmd.name.to_string(),
                        };

                        match cmd.name {
                            "/quit" => {
                                self.log(EntryKind::Command(cmd_text));
                                self.running = false;
                            }
                            "/configure" => {
                                let profile = arg.map(str::to_string);
                                self.log(EntryKind::Command(cmd_text));
                                self.input.clear();
                                self.palette_selected = None;
                                self.configure_state = ConfigureState::with_profile(profile);
                                self.config_open = true;
                            }
                            _ => {
                                self.input = cmd.name.to_string();
                                self.palette_selected = None;
                            }
                        }
                    }
                } else if !self.input.is_empty() {
                    let text = std::mem::take(&mut self.input);
                    self.log(EntryKind::Message(text));
                    self.thinking = true;
                }
            }

            Action::Cancel => {
                if self.palette_open() {
                    self.palette_selected = None;
                    self.input.clear();
                } else if self.thinking {
                    self.thinking = false;
                    self.log(EntryKind::Interrupted);
                }
            }
        }
    }
}
