use color_eyre::Result;
use ratatui::DefaultTerminal;
use tokio::sync::mpsc;

use crate::audio::{self, Frame};
use crate::commands::{self, CommandRegistry};
use crate::event::{self, Action};
use crate::model::config::TimbalConfig;
use crate::model::conversation::{Conversation, OutputBlock, Turn};
use crate::model::history::{self, Entry, EntryKind};
use crate::model::project::ProjectContext;
use crate::screens::ace::{AceExplorerState, AceExplorerTab};
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
    /// Show project structure info.
    ShowProject,
    /// Run the streaming test command.
    RunStreamingTest,
    /// Run a Timbal runnable by import spec (e.g. "path/to/agent.py::agent").
    RunRunnable(String),
    /// A command produced output to display inline.
    CommandOutput(OutputBlock),
    /// An output block from a background task (streaming response, tool output, etc.).
    PushOutput(OutputBlock),
    /// Mark the current turn as complete.
    TurnComplete,
    /// Mark the current turn as complete with a visible status message.
    TurnCompleteWith(String),
    /// Re-detect project structure from the filesystem.
    RefreshProject,
    /// Open the ace explorer for the first available agent with an ACE config.
    OpenAceExplorer,
    /// A line of output from the playground process.
    PlaygroundOutput(crate::screens::ace::PlaygroundLine),
    /// The playground process finished.
    PlaygroundComplete(String),
}

pub struct App {
    pub running: bool,
    pub input: String,
    pub palette_selected: Option<usize>,
    pub configure_state: ConfigureState,
    pub config_open: bool,
    pub help_state: HelpState,
    pub help_open: bool,
    pub project_open: bool,
    pub ace_explorer_open: bool,
    pub ace_explorer_state: Option<AceExplorerState>,
    pub shortcuts_open: bool,
    /// Clickable hint lines: (doc_line, unused, turn_index). Set during render.
    pub turn_line_ranges: Vec<(usize, usize, usize)>,
    /// Current mouse row in screen coordinates (for hover effects).
    pub mouse_row: Option<u16>,
    /// Active profile name (from TIMBAL_PROFILE env or "default").
    pub profile: String,
    /// Whether the active profile has credentials configured.
    pub configured: bool,
    /// Project context detected from the current directory.
    pub project: ProjectContext,
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
        let cwd = std::env::current_dir().unwrap_or_default();
        let project = ProjectContext::detect(&cwd);

        let mut app = Self {
            running: true,
            input: String::new(),
            palette_selected: None,
            configure_state: ConfigureState::new(),
            config_open: false,
            help_state: HelpState::new(),
            help_open: false,
            project_open: false,
            ace_explorer_open: false,
            ace_explorer_state: None,
            shortcuts_open: false,
            turn_line_ranges: Vec::new(),
            mouse_row: None,
            profile,
            configured,
            project,
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
        !self.config_open
            && !self.help_open
            && !self.ace_explorer_open
            && self.input.starts_with('/')
    }

    pub fn bash_mode(&self) -> bool {
        self.input.starts_with('!')
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

    /// Return all registered commands (for help screen).
    pub fn filter_commands_all(&self) -> Vec<&dyn commands::CommandHandler> {
        self.commands.all()
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
                // Animation tick — only runs when a turn is streaming or playground is running.
                _ = anim_interval.tick(), if self.conversation.is_busy()
                    || self.ace_explorer_state.as_ref().is_some_and(|s| s.playground_running) => {
                    self.advance_animation();
                }
            }
        }

        Ok(())
    }

    /// Handle a terminal input action.
    fn handle_action(&mut self, action: Action) {
        // If ace explorer is open, route input there (highest priority).
        if self.ace_explorer_open {
            if let Some(ref mut state) = self.ace_explorer_state {
                if state.field_editing {
                    // Mode: typing into a field.
                    match action {
                        Action::Submit => {
                            state.confirm_field_edit();
                        }
                        Action::Cancel => {
                            state.cancel_field_edit();
                        }
                        Action::Left => {
                            state.field_cursor_left();
                        }
                        Action::Right => {
                            state.field_cursor_right();
                        }
                        Action::Type(c) => {
                            state.field_insert(c);
                        }
                        Action::Backspace => {
                            state.field_backspace();
                        }
                        Action::Quit => self.running = false,
                        _ => {}
                    }
                } else if state.editing_right {
                    if state.current_tab() == AceExplorerTab::Evals {
                        // Mode: scrolling the right detail panel (read-only).
                        match action {
                            Action::Cancel => {
                                state.editing_right = false;
                                state.right_scroll = 0;
                            }
                            Action::PaletteDown | Action::Type('j') => {
                                state.right_scroll = state.right_scroll.saturating_add(1);
                            }
                            Action::PaletteUp | Action::Type('k') => {
                                state.right_scroll = state.right_scroll.saturating_sub(1);
                            }
                            Action::ScrollDown(col) => {
                                if col < state.right_panel_x {
                                    state.move_down();
                                } else {
                                    state.right_scroll =
                                        state.right_scroll.saturating_add(3);
                                }
                            }
                            Action::ScrollUp(col) => {
                                if col < state.right_panel_x {
                                    state.move_up();
                                } else {
                                    state.right_scroll =
                                        state.right_scroll.saturating_sub(3);
                                }
                            }
                            Action::Quit => self.running = false,
                            _ => {}
                        }
                    } else {
                        // Mode: navigating fields in the right panel.
                        match action {
                            Action::Cancel => {
                                // Leave right panel, save if dirty.
                                state.editing_right = false;
                                if state.dirty {
                                    let name = state.agent_name.clone();
                                    let ace = state.ace.clone();
                                    if let Err(e) =
                                        crate::model::project::save_ace(&name, &ace)
                                    {
                                        // TODO: show error to user
                                        eprintln!("Failed to save ace: {e}");
                                    }
                                    state.dirty = false;
                                }
                            }
                            Action::PaletteDown | Action::Type('j') => {
                                let max = state.field_count();
                                if max > 0 {
                                    state.editing_field =
                                        (state.editing_field + 1).min(max - 1);
                                }
                            }
                            Action::PaletteUp | Action::Type('k') => {
                                state.editing_field =
                                    state.editing_field.saturating_sub(1);
                            }
                            Action::Submit => {
                                state.start_field_edit();
                            }
                            Action::Quit => self.running = false,
                            _ => {}
                        }
                    }
                } else if state.search_focused {
                    // Mode: typing into search bar.
                    match action {
                        Action::Cancel | Action::Submit => {
                            state.search_focused = false;
                        }
                        Action::Type(c) => {
                            state.search.push(c);
                            state.clamp_selection();
                        }
                        Action::Backspace => {
                            state.search.pop();
                            state.clamp_selection();
                        }
                        Action::Quit => self.running = false,
                        _ => {}
                    }
                } else if state.current_tab() == AceExplorerTab::Playground {
                    // Mode: playground tab.
                    use crate::screens::ace::PlaygroundPanel;
                    use crate::screens::ace::state::PLAYGROUND_CFG_FIELD_COUNT;

                    if state.playground_cfg_editing {
                        // Editing a config field.
                        match action {
                            Action::Submit => {
                                let field = state.playground_cfg_field;
                                let value = state.playground_cfg_input.clone();
                                let fqn = state.import_spec.clone();
                                state.playground_confirm_cfg_edit();
                                // System prompt field (index 1): persist via codegen.
                                if field == 1 {
                                    if let Some(spec) = fqn {
                                        self.spawn_codegen_set_system_prompt(spec, value);
                                    }
                                }
                            }
                            Action::Cancel => {
                                state.playground_cancel_cfg_edit();
                            }
                            Action::Left => {
                                if state.playground_cfg_cursor > 0 {
                                    let prev = state.playground_cfg_input
                                        [..state.playground_cfg_cursor]
                                        .char_indices()
                                        .next_back()
                                        .map(|(i, _)| i)
                                        .unwrap_or(0);
                                    state.playground_cfg_cursor = prev;
                                }
                            }
                            Action::Right => {
                                if state.playground_cfg_cursor
                                    < state.playground_cfg_input.len()
                                {
                                    let next = state.playground_cfg_input
                                        [state.playground_cfg_cursor..]
                                        .char_indices()
                                        .nth(1)
                                        .map(|(i, _)| state.playground_cfg_cursor + i)
                                        .unwrap_or(state.playground_cfg_input.len());
                                    state.playground_cfg_cursor = next;
                                }
                            }
                            Action::Type(c) => {
                                state
                                    .playground_cfg_input
                                    .insert(state.playground_cfg_cursor, c);
                                state.playground_cfg_cursor += c.len_utf8();
                            }
                            Action::Backspace => {
                                if state.playground_cfg_cursor > 0 {
                                    let prev = state.playground_cfg_input
                                        [..state.playground_cfg_cursor]
                                        .char_indices()
                                        .next_back()
                                        .map(|(i, _)| i)
                                        .unwrap_or(0);
                                    state
                                        .playground_cfg_input
                                        .drain(prev..state.playground_cfg_cursor);
                                    state.playground_cfg_cursor = prev;
                                }
                            }
                            Action::Quit => self.running = false,
                            _ => {}
                        }
                    } else if state.playground_focused {
                        // Typing into the playground input box.
                        match action {
                            Action::Submit => {
                                if !state.playground_input.trim().is_empty()
                                    && !state.playground_running
                                {
                                    let raw_prompt =
                                        std::mem::take(&mut state.playground_input);
                                    state.playground_cursor = 0;
                                    if let Some(chat) = state.playground_chats.get_mut(
                                        state.playground_active_chat,
                                    ) {
                                        chat.output.clear();
                                        chat.output.push(
                                            crate::screens::ace::PlaygroundLine::User(
                                                raw_prompt.clone(),
                                            ),
                                        );
                                        chat.output.push(
                                            crate::screens::ace::PlaygroundLine::Thinking,
                                        );
                                    }
                                    state.playground_running = true;
                                    state.playground_feed_scroll = 0;

                                    let input_json = serde_json::json!({
                                        "prompt": raw_prompt
                                    })
                                    .to_string();

                                    let spec = state.import_spec.clone();
                                    let tx = self.event_tx.clone();
                                    self.spawn_playground_run(
                                        spec,
                                        input_json,
                                        tx,
                                    );
                                }
                            }
                            Action::Cancel => {
                                state.playground_focused = false;
                            }
                            Action::Left => {
                                if state.playground_cursor > 0 {
                                    let prev = state.playground_input
                                        [..state.playground_cursor]
                                        .char_indices()
                                        .next_back()
                                        .map(|(i, _)| i)
                                        .unwrap_or(0);
                                    state.playground_cursor = prev;
                                }
                            }
                            Action::Right => {
                                if state.playground_cursor
                                    < state.playground_input.len()
                                {
                                    let next = state.playground_input
                                        [state.playground_cursor..]
                                        .char_indices()
                                        .nth(1)
                                        .map(|(i, _)| state.playground_cursor + i)
                                        .unwrap_or(state.playground_input.len());
                                    state.playground_cursor = next;
                                }
                            }
                            Action::Type(c) => {
                                state
                                    .playground_input
                                    .insert(state.playground_cursor, c);
                                state.playground_cursor += c.len_utf8();
                            }
                            Action::Backspace => {
                                if state.playground_cursor > 0 {
                                    let prev = state.playground_input
                                        [..state.playground_cursor]
                                        .char_indices()
                                        .next_back()
                                        .map(|(i, _)| i)
                                        .unwrap_or(0);
                                    state
                                        .playground_input
                                        .drain(prev..state.playground_cursor);
                                    state.playground_cursor = prev;
                                }
                            }
                            Action::ScrollDown(col) => {
                                self.playground_scroll_down(col, 3);
                            }
                            Action::ScrollUp(col) => {
                                self.playground_scroll_up(col, 3);
                            }
                            Action::Quit => self.running = false,
                            _ => {}
                        }
                    } else {
                        // Playground: panel navigation (not focused on input/config).
                        match action {
                            Action::Cancel => {
                                self.ace_explorer_open = false;
                                self.ace_explorer_state = None;
                                if let Some(turn) =
                                    self.conversation.turns.last_mut()
                                {
                                    turn.complete_with(
                                        "ACE explorer dismissed".to_string(),
                                    );
                                }
                                self.scroll = u16::MAX;
                            }
                            // Tab/Shift+Tab: cycle ACE tabs (like Chrome).
                            Action::Tab => {
                                state.next_tab();
                            }
                            Action::BackTab => {
                                state.prev_tab();
                            }
                            // h/l or Left/Right: cycle playground panels.
                            Action::Type('l') | Action::Right => {
                                state.playground_panel =
                                    state.playground_panel.next();
                            }
                            Action::Type('h') | Action::Left => {
                                state.playground_panel =
                                    state.playground_panel.prev();
                            }
                            Action::Submit => {
                                match state.playground_panel {
                                    PlaygroundPanel::Feed => {
                                        state.playground_focused = true;
                                    }
                                    PlaygroundPanel::Chats => {
                                        // Enter switches to selected chat
                                        // (already active by selection)
                                    }
                                    PlaygroundPanel::Config => {
                                        state.playground_start_cfg_edit();
                                    }
                                }
                            }
                            Action::PaletteDown | Action::Type('j') => {
                                match state.playground_panel {
                                    PlaygroundPanel::Chats => {
                                        let count = state.playground_chats.len();
                                        if count > 0 {
                                            state.playground_active_chat =
                                                (state.playground_active_chat + 1)
                                                    .min(count - 1);
                                            state.playground_feed_scroll = 0;
                                        }
                                    }
                                    PlaygroundPanel::Feed => {
                                        state.playground_feed_scroll =
                                            state
                                                .playground_feed_scroll
                                                .saturating_add(1);
                                    }
                                    PlaygroundPanel::Config => {
                                        state.playground_cfg_field =
                                            (state.playground_cfg_field + 1)
                                                .min(PLAYGROUND_CFG_FIELD_COUNT - 1);
                                    }
                                }
                            }
                            Action::PaletteUp | Action::Type('k') => {
                                match state.playground_panel {
                                    PlaygroundPanel::Chats => {
                                        state.playground_active_chat =
                                            state
                                                .playground_active_chat
                                                .saturating_sub(1);
                                        state.playground_feed_scroll = 0;
                                    }
                                    PlaygroundPanel::Feed => {
                                        state.playground_feed_scroll =
                                            state
                                                .playground_feed_scroll
                                                .saturating_sub(1);
                                    }
                                    PlaygroundPanel::Config => {
                                        state.playground_cfg_field =
                                            state
                                                .playground_cfg_field
                                                .saturating_sub(1);
                                    }
                                }
                            }
                            Action::Type('n') => {
                                if state.playground_panel == PlaygroundPanel::Chats {
                                    state.playground_new_chat();
                                }
                            }
                            Action::ScrollDown(col) => {
                                self.playground_scroll_down(col, 3);
                            }
                            Action::ScrollUp(col) => {
                                self.playground_scroll_up(col, 3);
                            }
                            Action::Quit => self.running = false,
                            _ => {}
                        }
                    }
                } else {
                    // Mode: list navigation (default).
                    match action {
                        Action::Cancel => {
                            self.ace_explorer_open = false;
                            self.ace_explorer_state = None;
                            if let Some(turn) = self.conversation.turns.last_mut() {
                                turn.complete_with("ACE explorer dismissed".to_string());
                            }
                            self.scroll = u16::MAX;
                        }
                        Action::Submit => {
                            if state.current_item_count() > 0 {
                                state.enter_right();
                            }
                        }
                        Action::PaletteDown | Action::Type('j') => {
                            state.move_down();
                        }
                        Action::PaletteUp | Action::Type('k') => {
                            state.move_up();
                        }
                        Action::Tab | Action::Right => {
                            state.next_tab();
                        }
                        Action::BackTab | Action::Left => {
                            state.prev_tab();
                        }
                        Action::Type('/') => {
                            state.search_focused = true;
                        }
                        Action::ScrollDown(col) => {
                            if col >= state.right_panel_x {
                                state.right_scroll =
                                    state.right_scroll.saturating_add(3);
                            } else {
                                state.move_down();
                            }
                        }
                        Action::ScrollUp(col) => {
                            if col >= state.right_panel_x {
                                state.right_scroll =
                                    state.right_scroll.saturating_sub(3);
                            } else {
                                state.move_up();
                            }
                        }
                        Action::Quit => self.running = false,
                        _ => {}
                    }
                }
            }
            return;
        }

        // If project panel is open, route input there.
        if self.project_open {
            match action {
                Action::Cancel => {
                    self.project_open = false;
                    if let Some(turn) = self.conversation.turns.last_mut() {
                        turn.complete_with("Project info dismissed".to_string());
                    }
                }
                Action::Quit => self.running = false,
                _ => {}
            }
            return;
        }

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
                    self.scroll = u16::MAX;
                }
                Action::BackTab | Action::Left => {
                    self.help_state.prev_tab();
                    self.scroll = u16::MAX;
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
                if c == '?' && self.input.is_empty() {
                    self.shortcuts_open = !self.shortcuts_open;
                } else {
                    self.shortcuts_open = false;
                    self.input.push(c);
                    if self.input.starts_with('/') {
                        self.palette_selected = Some(0);
                    } else {
                        self.palette_selected = None;
                    }
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

            Action::ScrollUp(_) => {
                self.scroll = self.scroll.saturating_sub(10);
            }

            Action::ScrollDown(_) => {
                self.scroll = self.scroll.saturating_add(10);
            }

            Action::Submit => {
                if self.palette_open() {
                    self.submit_command();
                } else if self.bash_mode() {
                    self.submit_shell();
                } else if !self.input.is_empty() {
                    self.submit_message();
                }
            }

            Action::Cancel => {
                if self.shortcuts_open {
                    self.shortcuts_open = false;
                } else if self.palette_open() {
                    self.palette_selected = None;
                    self.input.clear();
                } else if self.bash_mode() {
                    self.input.clear();
                } else if self.conversation.is_busy() {
                    if let Some(turn) = self.conversation.active_turn_mut() {
                        turn.interrupt();
                    }
                    self.log(EntryKind::Interrupted);
                }
            }

            Action::MouseMove(row) => {
                self.mouse_row = Some(row);
            }

            Action::Click(_col, row) => {
                // Map screen row to doc line, check if it's a clickable hint line.
                let doc_line = row as usize + self.scroll as usize;
                for &(line, _, turn_idx) in &self.turn_line_ranges {
                    if doc_line == line {
                        if let Some(turn) = self.conversation.turns.get_mut(turn_idx) {
                            turn.collapsed = !turn.collapsed;
                        }
                        break;
                    }
                }
            }

            // Tab/Left/Right are only meaningful inside the help/ace panels (handled above).
            Action::Tab | Action::BackTab | Action::Left | Action::Right => {}
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

    /// Submit a shell command (input starts with `!`).
    fn submit_shell(&mut self) {
        let raw = std::mem::take(&mut self.input);
        let cmd = raw.strip_prefix('!').unwrap_or(&raw).trim().to_string();
        if cmd.is_empty() {
            return;
        }

        self.log(EntryKind::Command(format!("!{}", cmd)));

        let turn = Turn::shell(cmd.clone());
        self.conversation.push(turn);
        self.scroll = u16::MAX;

        let tx = self.event_tx.clone();
        let shell = std::env::var("SHELL").unwrap_or_else(|_| "/bin/sh".to_string());

        tokio::spawn(async move {
            let result = tokio::process::Command::new(&shell)
                .arg("-c")
                .arg(&cmd)
                .output()
                .await;

            match result {
                Ok(output) => {
                    let mut text = String::from_utf8_lossy(&output.stdout).to_string();
                    if !output.stderr.is_empty() {
                        if !text.is_empty() && !text.ends_with('\n') {
                            text.push('\n');
                        }
                        text.push_str(&String::from_utf8_lossy(&output.stderr));
                    }
                    let lines: Vec<String> = text.lines().map(|l| l.to_string()).collect();
                    let _ = tx.send(AppEvent::PushOutput(OutputBlock::ShellOutput(lines)));

                    let code = output.status.code().unwrap_or(-1);
                    if output.status.success() {
                        let _ = tx.send(AppEvent::TurnCompleteWith(format!("✓ exit {code}")));
                    } else {
                        let _ = tx.send(AppEvent::TurnCompleteWith(format!("✗ exit {code}")));
                    }
                }
                Err(e) => {
                    let _ = tx.send(AppEvent::PushOutput(OutputBlock::Error(format!(
                        "Failed to run command: {e}"
                    ))));
                    let _ = tx.send(AppEvent::TurnComplete);
                }
            }
        });
    }

    /// Run the streaming test: spawns a Python process that prints lines with delays.
    fn run_streaming_test(&mut self) {
        if let Some(turn) = self.conversation.turns.last_mut() {
            turn.status = crate::model::conversation::TurnStatus::Streaming;
        }
        self.scroll = u16::MAX;

        let tx = self.event_tx.clone();

        let python_code = r#"
import asyncio, sys

async def main():
    steps = [
        "Connecting to API...",
        "Authenticating...",
        "Fetching model list...",
        "Selected: claude-sonnet-4-20250514",
        "Sending prompt...",
        "Receiving response chunk 1...",
        "Receiving response chunk 2...",
        "Receiving response chunk 3...",
        "Processing tool call: Read(src/main.rs)",
        "Tool result: 52 lines",
        "Generating final response...",
        "Done. 247 tokens used.",
    ]
    for step in steps:
        print(step, flush=True)
        await asyncio.sleep(0.4)

asyncio.run(main())
"#;

        tokio::spawn(async move {
            use tokio::io::{AsyncBufReadExt, BufReader};
            use tokio::process::Command;

            let mut child = match Command::new("uv")
                .args(["run", "python", "-c", python_code])
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()
            {
                Ok(c) => c,
                Err(e) => {
                    let _ = tx.send(AppEvent::PushOutput(OutputBlock::Error(format!(
                        "Failed to spawn: {e}"
                    ))));
                    let _ = tx.send(AppEvent::TurnComplete);
                    return;
                }
            };

            if let Some(stdout) = child.stdout.take() {
                let mut reader = BufReader::new(stdout).lines();
                while let Ok(Some(line)) = reader.next_line().await {
                    let _ = tx.send(AppEvent::PushOutput(OutputBlock::Text(format!("  {line}"))));
                }
            }

            if let Some(stderr) = child.stderr.take() {
                let mut reader = BufReader::new(stderr).lines();
                while let Ok(Some(line)) = reader.next_line().await {
                    let _ = tx.send(AppEvent::PushOutput(OutputBlock::Error(line)));
                }
            }

            match child.wait().await {
                Ok(status) => {
                    let code = status.code().unwrap_or(-1);
                    if status.success() {
                        let _ = tx.send(AppEvent::TurnCompleteWith(format!("✓ exit {code}")));
                    } else {
                        let _ = tx.send(AppEvent::TurnCompleteWith(format!("✗ exit {code}")));
                    }
                }
                Err(_) => {
                    let _ = tx.send(AppEvent::TurnComplete);
                }
            }
        });
    }

    /// Run a Timbal runnable via `python -m timbal.server.run` with streaming.
    fn run_runnable(&mut self, import_spec: &str) {
        if let Some(turn) = self.conversation.turns.last_mut() {
            turn.status = crate::model::conversation::TurnStatus::Streaming;
        }
        self.scroll = u16::MAX;

        let tx = self.event_tx.clone();
        let spec = import_spec.to_string();

        tokio::spawn(async move {
            use tokio::io::{AsyncBufReadExt, BufReader};
            use tokio::process::Command;

            let mut child = match Command::new("uv")
                .args(["run", "python", "-m", "timbal.server.run", &spec, "--stream"])
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()
            {
                Ok(c) => c,
                Err(e) => {
                    let _ = tx.send(AppEvent::PushOutput(OutputBlock::Error(format!(
                        "Failed to spawn: {e}"
                    ))));
                    let _ = tx.send(AppEvent::TurnComplete);
                    return;
                }
            };

            if let Some(stdout) = child.stdout.take() {
                let mut reader = BufReader::new(stdout).lines();
                while let Ok(Some(line)) = reader.next_line().await {
                    let _ = tx.send(AppEvent::PushOutput(OutputBlock::Text(format!("  {line}"))));
                }
            }

            if let Some(stderr) = child.stderr.take() {
                let mut reader = BufReader::new(stderr).lines();
                while let Ok(Some(line)) = reader.next_line().await {
                    let _ = tx.send(AppEvent::PushOutput(OutputBlock::Error(line)));
                }
            }

            match child.wait().await {
                Ok(status) => {
                    let code = status.code().unwrap_or(-1);
                    if status.success() {
                        let _ = tx.send(AppEvent::TurnCompleteWith(format!("✓ exit {code}")));
                    } else {
                        let _ = tx.send(AppEvent::TurnCompleteWith(format!("✗ exit {code}")));
                    }
                }
                Err(_) => {
                    let _ = tx.send(AppEvent::TurnComplete);
                }
            }
        });
    }

    /// Spawn a playground run: calls `python -m timbal.server.run` with streaming,
    /// sending output to PlaygroundOutput/PlaygroundComplete events.
    /// Route mouse scroll to the correct playground panel based on column.
    fn playground_scroll_down(&mut self, col: u16, amount: u16) {
        if let Some(ref mut state) = self.ace_explorer_state {
            if col >= state.playground_config_x {
                state.playground_config_scroll =
                    state.playground_config_scroll.saturating_add(amount);
            } else if col >= state.playground_feed_x {
                state.playground_feed_scroll =
                    state.playground_feed_scroll.saturating_add(amount);
            }
            // Chats panel: no scroll for now (list is small).
        }
    }

    fn playground_scroll_up(&mut self, col: u16, amount: u16) {
        if let Some(ref mut state) = self.ace_explorer_state {
            if col >= state.playground_config_x {
                state.playground_config_scroll =
                    state.playground_config_scroll.saturating_sub(amount);
            } else if col >= state.playground_feed_x {
                state.playground_feed_scroll =
                    state.playground_feed_scroll.saturating_sub(amount);
            }
        }
    }

    /// Persist a system prompt change via `python -m timbal.codegen <fqn> set-system-prompt <value>`.
    fn spawn_codegen_set_system_prompt(&self, fqn: String, value: String) {
        tokio::spawn(async move {
            use tokio::process::Command;
            let _ = Command::new("uv")
                .args([
                    "run",
                    "python",
                    "-m",
                    "timbal.codegen",
                    &fqn,
                    "set-system-prompt",
                    &value,
                ])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .await;
        });
    }

    fn spawn_playground_run(
        &self,
        import_spec: Option<String>,
        input_json: String,
        tx: mpsc::UnboundedSender<AppEvent>,
    ) {
        use crate::screens::ace::PlaygroundLine;

        let spec = match import_spec {
            Some(s) => s,
            None => {
                let _ = tx.send(AppEvent::PlaygroundOutput(PlaygroundLine::Error(
                    "No import spec (fqn) configured for this agent.".to_string(),
                )));
                let _ = tx.send(AppEvent::PlaygroundComplete("No runnable".to_string()));
                return;
            }
        };

        tokio::spawn(async move {
            use tokio::io::{AsyncBufReadExt, BufReader};
            use tokio::process::Command;

            let mut child = match Command::new("uv")
                .args([
                    "run",
                    "python",
                    "-m",
                    "timbal.server.run",
                    &spec,
                    "--stream",
                    "--input",
                    &input_json,
                ])
                .env("TIMBAL_DELTA_EVENTS", "true")
                .env("PYTHONUNBUFFERED", "1")
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()
            {
                Ok(c) => c,
                Err(e) => {
                    let _ = tx.send(AppEvent::PlaygroundOutput(PlaygroundLine::Error(
                        format!("Failed to spawn: {e}"),
                    )));
                    let _ = tx.send(AppEvent::PlaygroundComplete("Failed".to_string()));
                    return;
                }
            };

            // Read stdout and stderr concurrently.
            let stdout = child.stdout.take();
            let stderr = child.stderr.take();

            let tx2 = tx.clone();
            let stdout_handle = tokio::spawn(async move {
                if let Some(out) = stdout {
                    let mut reader = BufReader::new(out).lines();
                    while let Ok(Some(line)) = reader.next_line().await {
                        // Try to parse as a Timbal event JSON.
                        if let Ok(event) = serde_json::from_str::<serde_json::Value>(&line) {
                            let event_type = event
                                .get("type")
                                .and_then(|v| v.as_str())
                                .unwrap_or("");
                            match event_type {
                                "DELTA" => {
                                    if let Some(item) = event.get("item") {
                                        let item_type = item
                                            .get("type")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("");
                                        match item_type {
                                            "text_delta" => {
                                                if let Some(delta) =
                                                    item.get("text_delta").and_then(|v| v.as_str())
                                                {
                                                    let _ = tx2.send(
                                                        AppEvent::PlaygroundOutput(
                                                            PlaygroundLine::TextDelta(
                                                                delta.to_string(),
                                                            ),
                                                        ),
                                                    );
                                                }
                                            }
                                            "text" => {
                                                // Initial text block — ignore (deltas follow).
                                            }
                                            "tool_use" => {
                                                let name = item
                                                    .get("name")
                                                    .and_then(|v| v.as_str())
                                                    .unwrap_or("unknown");
                                                let _ = tx2.send(
                                                    AppEvent::PlaygroundOutput(
                                                        PlaygroundLine::Status(format!(
                                                            "+ {name}"
                                                        )),
                                                    ),
                                                );
                                            }
                                            "tool_use_delta" => {
                                                // Tool input streaming — skip for now.
                                            }
                                            "thinking_delta" => {
                                                // Show a thinking indicator (not the content).
                                                let _ = tx2.send(
                                                    AppEvent::PlaygroundOutput(
                                                        PlaygroundLine::Thinking,
                                                    ),
                                                );
                                            }
                                            "content_block_stop" => {
                                                // End of a content block — start new line.
                                                let _ = tx2.send(
                                                    AppEvent::PlaygroundOutput(
                                                        PlaygroundLine::Text(String::new()),
                                                    ),
                                                );
                                            }
                                            _ => {}
                                        }
                                    }
                                }
                                "OUTPUT" => {
                                    let status_code = event
                                        .pointer("/status/code")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("unknown");
                                    let path = event
                                        .get("path")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("");
                                    if status_code == "error" {
                                        let msg = event
                                            .pointer("/status/message")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("Unknown error");
                                        let _ = tx2.send(AppEvent::PlaygroundOutput(
                                            PlaygroundLine::Error(msg.to_string()),
                                        ));
                                    }
                                    // Extract stats from the top-level agent OUTPUT.
                                    if path == "agent" || path.is_empty() {
                                        let t0 = event.get("t0").and_then(|v| v.as_u64()).unwrap_or(0);
                                        let t1 = event.get("t1").and_then(|v| v.as_u64()).unwrap_or(0);
                                        let duration_ms = t1.saturating_sub(t0);

                                        // Build usage summary from the usage dict.
                                        let mut usage_parts: Vec<String> = Vec::new();
                                        if let Some(usage_obj) = event.get("usage").and_then(|v| v.as_object()) {
                                            // Aggregate by metric type across models.
                                            let mut input_tokens: u64 = 0;
                                            let mut output_tokens: u64 = 0;
                                            for (key, val) in usage_obj {
                                                let count = val.as_u64().unwrap_or(0);
                                                if key.ends_with(":input_tokens") || key.ends_with(":input_text_tokens") {
                                                    input_tokens += count;
                                                } else if key.ends_with(":output_tokens") || key.ends_with(":output_text_tokens") {
                                                    output_tokens += count;
                                                }
                                            }
                                            if input_tokens > 0 {
                                                usage_parts.push(format!("{input_tokens} in"));
                                            }
                                            if output_tokens > 0 {
                                                usage_parts.push(format!("{output_tokens} out"));
                                            }
                                        }

                                        let usage_str = if usage_parts.is_empty() {
                                            String::new()
                                        } else {
                                            usage_parts.join(" / ")
                                        };

                                        if duration_ms > 0 || !usage_str.is_empty() {
                                            let _ = tx2.send(AppEvent::PlaygroundOutput(
                                                PlaygroundLine::Stats {
                                                    duration_ms,
                                                    usage: usage_str,
                                                },
                                            ));
                                        }
                                    }
                                }
                                "START" => {
                                    // Ignore start events.
                                }
                                _ => {
                                    // Unknown event — show raw.
                                    let _ = tx2.send(AppEvent::PlaygroundOutput(
                                        PlaygroundLine::Text(line),
                                    ));
                                }
                            }
                        } else {
                            // Not JSON — show raw line.
                            let _ = tx2.send(AppEvent::PlaygroundOutput(
                                PlaygroundLine::Text(line),
                            ));
                        }
                    }
                }
            });

            let tx3 = tx.clone();
            let stderr_handle = tokio::spawn(async move {
                if let Some(err) = stderr {
                    let mut reader = BufReader::new(err).lines();
                    while let Ok(Some(line)) = reader.next_line().await {
                        let _ = tx3.send(AppEvent::PlaygroundOutput(PlaygroundLine::Error(line)));
                    }
                }
            });

            let _ = stdout_handle.await;
            let _ = stderr_handle.await;

            match child.wait().await {
                Ok(status) => {
                    let code = status.code().unwrap_or(-1);
                    if status.success() {
                        let _ = tx.send(AppEvent::PlaygroundComplete(format!("exit {code}")));
                    } else {
                        let _ = tx.send(AppEvent::PlaygroundComplete(format!("exit {code}")));
                    }
                }
                Err(_) => {
                    let _ = tx.send(AppEvent::PlaygroundComplete("Process error".to_string()));
                }
            }
        });
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
                self.configure_state = ConfigureState::for_profile(self.profile.clone());
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
            AppEvent::ShowProject => {
                self.project_open = true;
                self.scroll = u16::MAX;
            }
            AppEvent::RunStreamingTest => {
                self.run_streaming_test();
            }
            AppEvent::RunRunnable(spec) => {
                self.run_runnable(&spec);
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
            AppEvent::TurnCompleteWith(msg) => {
                if let Some(turn) = self.conversation.active_turn_mut() {
                    turn.complete_with(msg);
                }
            }
            AppEvent::RefreshProject => {
                let cwd = std::env::current_dir().unwrap_or_default();
                self.project = ProjectContext::detect(&cwd);
            }
            AppEvent::OpenAceExplorer => {
                if let Some(idx) = self
                    .project
                    .members
                    .iter()
                    .position(|m| m.kind == "agent" && m.ace.is_some())
                {
                    self.open_ace_explorer(idx);
                } else {
                    // No agent with ACE — show a message in the conversation.
                    let msg = if self.project.members.iter().any(|m| m.kind == "agent") {
                        "No ACE configured for any agent. Run `timbal ace init <name>` first."
                            .to_string()
                    } else if self.project.is_timbal_project {
                        "No agents found in the workforce.".to_string()
                    } else {
                        "Not a Timbal project. Run `timbal create` to get started.".to_string()
                    };
                    if let Some(turn) = self.conversation.turns.last_mut() {
                        turn.push_output(OutputBlock::Error(msg));
                    }
                }
            }
            AppEvent::PlaygroundOutput(line) => {
                if let Some(ref mut state) = self.ace_explorer_state {
                    state.playground_push_output(line);
                    // Auto-scroll feed to bottom.
                    state.playground_feed_scroll = u16::MAX;
                }
            }
            AppEvent::PlaygroundComplete(_msg) => {
                if let Some(ref mut state) = self.ace_explorer_state {
                    state.playground_running = false;
                    state.playground_feed_scroll = u16::MAX;
                }
            }
        }
    }

    /// Open the ace explorer modal for the given workforce member.
    fn open_ace_explorer(&mut self, member_idx: usize) {
        if let Some(member) = self.project.members.get(member_idx) {
            if let Some(ace) = &member.ace {
                let evals = member.evals.clone().unwrap_or_default();
                let import_spec = member.fqn.as_ref().map(|fqn| {
                    format!("workforce/{}/{}", member.name, fqn)
                });
                self.ace_explorer_state = Some(AceExplorerState::new(
                    member.name.clone(),
                    ace.clone(),
                    evals,
                    import_spec,
                ));
                self.ace_explorer_open = true;
                self.project_open = false;
                self.scroll = u16::MAX;
            }
        }
    }
}
