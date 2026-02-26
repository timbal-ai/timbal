use super::{CommandHandler, CommandMeta};
use crate::app::AppEvent;

pub struct AceCommand;

impl CommandHandler for AceCommand {
    fn meta(&self) -> CommandMeta {
        CommandMeta {
            name: "/ace",
            description: "Execution-time behavioral control for AI agents",
            usage: "/ace",
        }
    }

    fn execute(&self, _args: Option<&str>) -> Vec<AppEvent> {
        vec![AppEvent::OpenAceExplorer]
    }
}
