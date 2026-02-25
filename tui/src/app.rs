use std::path::PathBuf;

use color_eyre::Result;
use ratatui::DefaultTerminal;

use crate::audio::{self, SampleBuffer};
use crate::commands;
use crate::event::{self, Action};
use crate::history::{self, Entry, EntryKind};
use crate::screens::configure::ConfigureState;
use crate::ui;
use crate::widgets::vectorscope::VectorscopeState;

pub struct App {
    pub running: bool,
    pub sample_buffer: SampleBuffer,
    pub vectorscope_state: VectorscopeState,
    pub input: String,
    pub thinking: bool,
    pub palette_selected: Option<usize>,
    pub configure_state: ConfigureState,
    pub config_open: bool,
    pub history: Vec<Entry>,
    pub scroll: u16,
}

impl App {
    pub fn new(mp3_path: PathBuf) -> Result<Self> {
        let sample_buffer = audio::decode_mp3(&mp3_path)?;

        let mut app = Self {
            running: true,
            sample_buffer,
            vectorscope_state: VectorscopeState::default(),
            input: String::new(),
            thinking: false,
            palette_selected: None,
            configure_state: ConfigureState::new(),
            config_open: false,
            history: Vec::new(),
            scroll: 0,
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

    pub fn run(&mut self, terminal: &mut DefaultTerminal) -> Result<()> {
        while self.running {
            terminal.draw(|frame| ui::render(self, frame))?;

            if let Some(action) = event::poll()? {
                self.update(action);
            }
        }
        Ok(())
    }

    pub fn update(&mut self, action: Action) {
        // Route input to the inline config panel when it's open.
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
                        // Build the full command text from the matched command name + any args.
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
                                self.configure_state =
                                    ConfigureState::with_profile(profile);
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
                } else {
                    self.thinking = false;
                }
            }
        }
    }
}
