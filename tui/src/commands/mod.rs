pub mod clear;
pub mod configure;
pub mod help;
pub mod quit;

use crate::app::AppEvent;

/// Metadata about a command, used for palette display and filtering.
pub struct CommandMeta {
    pub name: &'static str,
    pub description: &'static str,
    pub usage: &'static str,
}

/// Every command implements this trait. Each command owns its name, description,
/// and execution logic — no central match statement needed.
pub trait CommandHandler: Send + Sync {
    fn meta(&self) -> CommandMeta;

    /// Execute the command, returning a list of events to apply to the app.
    fn execute(&self, args: Option<&str>) -> Vec<AppEvent>;
}

/// Registry that holds all available commands and provides filtering/dispatch.
pub struct CommandRegistry {
    handlers: Vec<Box<dyn CommandHandler>>,
}

impl CommandRegistry {
    pub fn new() -> Self {
        let mut reg = Self {
            handlers: Vec::new(),
        };
        reg.register(Box::new(configure::ConfigureCommand));
        reg.register(Box::new(quit::QuitCommand));
        reg.register(Box::new(clear::ClearCommand));
        reg.register(Box::new(help::HelpCommand));
        reg
    }

    fn register(&mut self, handler: Box<dyn CommandHandler>) {
        self.handlers.push(handler);
    }

    /// Return commands whose name starts with the input prefix,
    /// or whose full name matches the start of the input (for argument passing).
    pub fn filter(&self, input: &str) -> Vec<&dyn CommandHandler> {
        let lower = input.to_lowercase();
        self.handlers
            .iter()
            .filter(|h| {
                let name = h.meta().name;
                name.starts_with(&*lower) || lower.starts_with(&format!("{} ", name))
            })
            .map(|h| h.as_ref())
            .collect()
    }

    /// Look up a command by exact name and execute it.
    pub fn execute(&self, name: &str, args: Option<&str>) -> Vec<AppEvent> {
        for handler in &self.handlers {
            if handler.meta().name == name {
                return handler.execute(args);
            }
        }
        vec![]
    }
}

/// Extract the argument portion after the command name, if any.
/// "/configure prod" -> Some("prod"), "/configure" -> None
pub fn parse_arg<'a>(input: &'a str, cmd_name: &str) -> Option<&'a str> {
    let rest = input.trim().strip_prefix(cmd_name)?.trim();
    if rest.is_empty() {
        None
    } else {
        Some(rest)
    }
}
