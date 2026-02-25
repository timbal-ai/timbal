use super::{CommandHandler, CommandMeta};
use crate::app::AppEvent;

pub struct ClearCommand;

impl CommandHandler for ClearCommand {
    fn meta(&self) -> CommandMeta {
        CommandMeta {
            name: "/clear",
            description: "Clear conversation history",
            usage: "/clear",
        }
    }

    fn execute(&self, _args: Option<&str>) -> Vec<AppEvent> {
        vec![AppEvent::ClearConversation]
    }
}
