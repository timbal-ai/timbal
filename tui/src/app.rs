use color_eyre::Result;
use ratatui::DefaultTerminal;
use tokio::sync::mpsc;

use crate::audio::{self, Frame};
use crate::commands::{self, CommandRegistry};
use crate::event::{self, Action};
use crate::model::config::TimbalConfig;
use crate::model::conversation::{Conversation, OutputBlock, Turn};
use crate::model::history::{self, Entry, EntryKind};
use crate::screens::configure::ConfigureState;
use crate::screens::help::HelpState;
use crate::ui;

/// Events that mutate app state. Produced by commands, background tasks, or user input.
/// This is the single channel through which all state changes flow.
#[derive(Debug)]
pub enum AppEvent {
    /// Quit the application.
    Quit,
    /// Open the configure dialog (uses the app's active profile).
    OpenConfigure,
    /// Clear conversation history.
    ClearConversation,
    /// Open the help panel.
    OpenHelp,
    /// A command produced output to display inline.
    CommandOutput(OutputBlock),
    /// An output block from a background task (streaming response, tool output, etc.).
    PushOutput(OutputBlock),
    /// Mark the current turn as complete.
    TurnComplete,
}

pub struct App {
    pub running: bool,
    pub input: String,
    pub palette_selected: Option<usize>,
    pub configure_state: ConfigureState,
    pub config_open: bool,
    pub help_state: HelpState,
    pub help_open: bool,
    /// Active profile name (from TIMBAL_PROFILE env or "default").
    pub profile: String,
    /// Whether the active profile has credentials configured.
    pub configured: bool,
    pub conversation: Conversation,
    pub scroll: u16,
    /// Precomputed vectorscope animation frames (decoded once at startup).
    pub scope_frames: Vec<Frame>,
    /// Current animation frame index.
    pub scope_tick: usize,
    /// Spinner tick for inline spinners (advances with animation).
    pub spinner_tick: usize,
    /// Command registry — owns all slash command handlers.
    commands: CommandRegistry,
    /// Sender side of the event bus — cloned and given to background tasks.
    pub event_tx: mpsc::UnboundedSender<AppEvent>,
    /// Receiver side — polled in the main loop.
    event_rx: mpsc::UnboundedReceiver<AppEvent>,
    /// Persistent history log entries (for disk persistence).
    pub history: Vec<Entry>,
}

impl App {
    pub fn new(profile_override: Option<String>) -> Result<Self> {
        let scope_frames = audio::load_frames();
        let (event_tx, event_rx) = mpsc::unbounded_channel();

        let profile = profile_override
            .or_else(|| std::env::var("TIMBAL_PROFILE").ok())
            .unwrap_or_else(|| "default".to_string());
        let config = TimbalConfig::load(&profile);
        let configured = config.is_configured();

        let mut app = Self {
            running: true,
            input: String::new(),
            palette_selected: None,
            configure_state: ConfigureState::new(),
            config_open: false,
            help_state: HelpState::new(),
            help_open: false,
            profile,
            configured,
            conversation: Conversation::new(),
            scroll: 0,
            scope_frames,
            scope_tick: 0,
            spinner_tick: 0,
            commands: CommandRegistry::new(),
            event_tx,
            event_rx,
            history: Vec::new(),
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
        !self.config_open && !self.help_open && self.input.starts_with('/')
    }

    /// Get the current vectorscope frame (loops automatically).
    pub fn current_scope_frame(&self) -> &[(f64, f64)] {
        if self.scope_frames.is_empty() {
            return &[];
        }
        &self.scope_frames[self.scope_tick % self.scope_frames.len()]
    }

    /// Advance animation state (vectorscope frame + spinner tick).
    fn advance_animation(&mut self) {
        self.spinner_tick = self.spinner_tick.wrapping_add(1);
        if self.spinner_tick % 2 == 0 && !self.scope_frames.is_empty() {
            self.scope_tick = (self.scope_tick + 1) % self.scope_frames.len();
        }
    }

    /// Filter commands matching current input (delegates to registry).
    pub fn filter_commands(&self) -> Vec<&dyn commands::CommandHandler> {
        self.commands.filter(&self.input)
    }

    /// The async main loop. Uses tokio::select! to multiplex between:
    /// - Terminal input events
    /// - AppEvent channel (from commands, background tasks)
    /// - Animation tick interval (when busy)
    pub async fn run(&mut self, terminal: &mut DefaultTerminal) -> Result<()> {
        // Animation tick interval: ~30fps when active, paused when idle.
        let mut anim_interval = tokio::time::interval(tokio::time::Duration::from_millis(33));
        anim_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        while self.running {
            terminal.draw(|frame| ui::render(self, frame))?;

            tokio::select! {
                // Terminal input (polled in a blocking thread to avoid blocking the runtime).
                action = tokio::task::spawn_blocking(event::poll_blocking) => {
                    match action {
                        Ok(Ok(Some(a))) => self.handle_action(a),
                        Ok(Ok(None)) => {}
                        Ok(Err(e)) => return Err(e),
                        Err(e) => return Err(e.into()),
                    }
                }
                // Events from the channel (commands, background tasks).
                Some(event) = self.event_rx.recv() => {
                    self.apply(event);
                }
                // Animation tick — only runs when a turn is streaming.
                _ = anim_interval.tick(), if self.conversation.is_busy() => {
                    self.advance_animation();
                }
            }
        }

        Ok(())
    }

    /// Handle a terminal input action.
    fn handle_action(&mut self, action: Action) {
        // If help panel is open, route input there.
        if self.help_open {
            match action {
                Action::Cancel => {
                    self.help_open = false;
                    self.help_state = HelpState::new();
                    if let Some(turn) = self.conversation.turns.last_mut() {
                        turn.complete_with("Help dialog dismissed".to_string());
                    }
                }
                Action::Tab | Action::Right => {
                    self.help_state.next_tab();
                }
                Action::Left => {
                    self.help_state.prev_tab();
                }
                Action::Quit => self.running = false,
                _ => {}
            }
            return;
        }

        // If config dialog is open, route input there.
        if self.config_open {
            match action {
                Action::Cancel => {
                    self.log(EntryKind::ConfigureCancelled);
                    self.config_open = false;
                    self.configure_state = ConfigureState::new();
                    if let Some(turn) = self.conversation.turns.last_mut() {
                        turn.complete_with("Config dialog dismissed".to_string());
                    }
                }
                Action::Submit => {
                    if self.configure_state.saved || self.configure_state.error.is_some() {
                        if self.configure_state.saved {
                            let profile = self.configure_state.profile.clone();
                            self.log(EntryKind::ConfigureSaved(profile.clone()));
                            self.config_open = false;
                            self.configure_state = ConfigureState::new();
                            // Refresh configured status after saving.
                            let cfg = TimbalConfig::load(&self.profile);
                            self.configured = cfg.is_configured();
                            if let Some(turn) = self.conversation.turns.last_mut() {
                                turn.complete_with(format!(
                                    "Credentials saved (profile: {profile})"
                                ));
                            }
                        } else {
                            self.config_open = false;
                            self.configure_state = ConfigureState::new();
                            if let Some(turn) = self.conversation.turns.last_mut() {
                                turn.complete_with("Config dialog closed".to_string());
                            }
                        }
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
                    let count = self.filter_commands().len();
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
                    self.submit_command();
                } else if !self.input.is_empty() {
                    self.submit_message();
                }
            }

            Action::Cancel => {
                if self.palette_open() {
                    self.palette_selected = None;
                    self.input.clear();
                } else if self.conversation.is_busy() {
                    if let Some(turn) = self.conversation.active_turn_mut() {
                        turn.interrupt();
                    }
                    self.log(EntryKind::Interrupted);
                }
            }

            // Tab/Left/Right are only meaningful inside the help panel (handled above).
            Action::Tab | Action::Left | Action::Right => {}
        }
    }

    /// Submit a slash command from the palette.
    fn submit_command(&mut self) {
        // Gather what we need from the immutable borrow, then drop it.
        let resolved = {
            let matches = self.filter_commands();
            let idx = self.palette_selected.unwrap_or(0);
            matches.get(idx).map(|cmd| {
                let meta = cmd.meta();
                let arg = commands::parse_arg(&self.input, meta.name).map(str::to_string);
                let cmd_text = match &arg {
                    Some(a) => format!("{} {}", meta.name, a),
                    None => meta.name.to_string(),
                };
                let cmd_name = meta.name.to_string();
                (cmd_name, arg, cmd_text)
            })
        };

        if let Some((cmd_name, arg, cmd_text)) = resolved {
            self.log(EntryKind::Command(cmd_text.clone()));

            let turn = Turn::command(cmd_text);
            self.conversation.push(turn);

            let events = self.commands.execute(&cmd_name, arg.as_deref());

            self.input.clear();
            self.palette_selected = None;

            for event in events {
                self.apply(event);
            }
        }
    }

    /// Submit a regular user message.
    fn submit_message(&mut self) {
        let text = std::mem::take(&mut self.input);
        self.log(EntryKind::Message(text.clone()));

        // Create a new streaming turn in the conversation.
        let turn = Turn::message(text);
        self.conversation.push(turn);
        self.scroll = u16::MAX;

        // TODO: Here is where you'd spawn a backend task that streams
        // AppEvent::PushOutput / AppEvent::TurnComplete back through event_tx.
        // For now, the turn stays in Streaming state until interrupted.
    }

    /// Apply an AppEvent to mutate state.
    fn apply(&mut self, event: AppEvent) {
        match event {
            AppEvent::Quit => {
                self.running = false;
            }
            AppEvent::OpenConfigure => {
                self.configure_state =
                    ConfigureState::for_profile(self.profile.clone());
                self.config_open = true;
            }
            AppEvent::ClearConversation => {
                self.conversation = Conversation::new();
                self.scroll = 0;
            }
            AppEvent::OpenHelp => {
                self.help_state = HelpState::new();
                self.help_open = true;
                self.scroll = u16::MAX;
            }
            AppEvent::CommandOutput(block) => {
                // Attach output to the last turn (the command turn we just created).
                if let Some(turn) = self.conversation.turns.last_mut() {
                    turn.push_output(block);
                }
                self.scroll = u16::MAX;
            }
            AppEvent::PushOutput(block) => {
                if let Some(turn) = self.conversation.active_turn_mut() {
                    turn.push_output(block);
                    self.scroll = u16::MAX;
                }
            }
            AppEvent::TurnComplete => {
                if let Some(turn) = self.conversation.active_turn_mut() {
                    turn.complete();
                }
            }
        }
    }
}
