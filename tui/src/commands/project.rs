use super::{CommandHandler, CommandMeta};
use crate::app::AppEvent;

pub struct ProjectCommand;

impl CommandHandler for ProjectCommand {
    fn meta(&self) -> CommandMeta {
        CommandMeta {
            name: "/project",
            description: "Show project structure",
            usage: "/project",
        }
    }

    fn execute(&self, _args: Option<&str>) -> Vec<AppEvent> {
        vec![AppEvent::ShowProject]
    }
}
