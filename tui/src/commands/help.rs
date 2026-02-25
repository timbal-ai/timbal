use super::{CommandHandler, CommandMeta};
use crate::app::AppEvent;

pub struct HelpCommand;

impl CommandHandler for HelpCommand {
    fn meta(&self) -> CommandMeta {
        CommandMeta {
            name: "/help",
            description: "Show available commands",
            usage: "/help",
        }
    }

    fn execute(&self, _args: Option<&str>) -> Vec<AppEvent> {
        vec![AppEvent::OpenHelp]
    }
}
