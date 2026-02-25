use super::{CommandHandler, CommandMeta};
use crate::app::AppEvent;

pub struct QuitCommand;

impl CommandHandler for QuitCommand {
    fn meta(&self) -> CommandMeta {
        CommandMeta {
            name: "/quit",
            description: "Exit Timbal",
            usage: "/quit",
        }
    }

    fn execute(&self, _args: Option<&str>) -> Vec<AppEvent> {
        vec![AppEvent::Quit]
    }
}
