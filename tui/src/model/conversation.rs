/// The status of a single turn in the conversation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TurnStatus {
    /// The turn is still receiving output (streaming).
    Streaming,
    /// The turn completed successfully (no status line shown).
    Complete,
    /// The turn completed with a visible status message (e.g. "Help dialog dismissed").
    Completed(String),
    /// The user interrupted the turn.
    Interrupted,
}

/// A block of output produced during a turn.
#[derive(Debug, Clone)]
pub enum OutputBlock {
    /// Plain text content (e.g. assistant response).
    Text(String),
    /// Pre-styled rich lines (for structured output like /help).
    RichLines(Vec<ratatui::text::Line<'static>>),
    /// A tool/command invocation with its status and optional output.
    ToolCall {
        name: String,
        status: ToolStatus,
        output: Option<String>,
    },
    /// An error message.
    Error(String),
    /// Shell command output lines.
    ShellOutput(Vec<String>),
}

/// Status of a tool call within a turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolStatus {
    Running,
    Completed,
    Failed,
}

/// A single conversation turn: one user input and the resulting outputs.
#[derive(Debug, Clone)]
pub struct Turn {
    pub input: TurnInput,
    pub outputs: Vec<OutputBlock>,
    pub status: TurnStatus,
    /// Whether this turn's output is collapsed.
    pub collapsed: bool,
}

/// What the user submitted for this turn.
#[derive(Debug, Clone)]
pub enum TurnInput {
    /// A regular message.
    Message(String),
    /// A slash command (e.g. "/configure prod").
    Command(String),
    /// A shell command (e.g. "!ls").
    Shell(String),
}

impl Turn {
    pub fn message(text: String) -> Self {
        Self {
            input: TurnInput::Message(text),
            outputs: Vec::new(),
            status: TurnStatus::Streaming,
            collapsed: false,
        }
    }

    pub fn command(text: String) -> Self {
        Self {
            input: TurnInput::Command(text),
            outputs: Vec::new(),
            status: TurnStatus::Complete,
            collapsed: false,
        }
    }

    pub fn shell(cmd: String) -> Self {
        Self {
            input: TurnInput::Shell(cmd),
            outputs: Vec::new(),
            status: TurnStatus::Streaming,
            collapsed: true,
        }
    }

    /// Count total output lines for this turn.
    pub fn output_line_count(&self) -> usize {
        let mut count = 0;
        for block in &self.outputs {
            count += match block {
                OutputBlock::Text(t) => t.lines().count(),
                OutputBlock::RichLines(r) => r.len(),
                OutputBlock::ToolCall { output, .. } => {
                    1 + output.as_ref().map_or(0, |o| o.lines().count())
                }
                OutputBlock::Error(_) => 1,
                OutputBlock::ShellOutput(lines) => lines.len(),
            };
        }
        count
    }

    pub fn is_streaming(&self) -> bool {
        self.status == TurnStatus::Streaming
    }

    pub fn push_output(&mut self, block: OutputBlock) {
        self.outputs.push(block);
    }

    pub fn complete(&mut self) {
        self.status = TurnStatus::Complete;
    }

    /// Complete the turn with a visible status message.
    pub fn complete_with(&mut self, message: String) {
        self.status = TurnStatus::Completed(message);
    }

    pub fn interrupt(&mut self) {
        self.status = TurnStatus::Interrupted;
    }
}

/// The full conversation state.
#[derive(Debug, Clone, Default)]
pub struct Conversation {
    pub turns: Vec<Turn>,
}

impl Conversation {
    pub fn new() -> Self {
        Self { turns: Vec::new() }
    }

    pub fn push(&mut self, turn: Turn) {
        self.turns.push(turn);
    }

    /// Get the currently active (streaming) turn, if any.
    pub fn active_turn_mut(&mut self) -> Option<&mut Turn> {
        self.turns.last_mut().filter(|t| t.is_streaming())
    }

    /// Whether any turn is currently streaming.
    pub fn is_busy(&self) -> bool {
        self.turns.last().is_some_and(|t| t.is_streaming())
    }
}
