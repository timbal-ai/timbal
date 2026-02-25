use super::{CommandHandler, CommandMeta};
use crate::app::AppEvent;

pub struct ConfigureCommand;

impl CommandHandler for ConfigureCommand {
    fn meta(&self) -> CommandMeta {
        CommandMeta {
            name: "/configure",
            description: "Set up API key and credentials",
            usage: "/configure",
        }
    }

    fn execute(&self, _args: Option<&str>) -> Vec<AppEvent> {
        vec![AppEvent::OpenConfigure]
    }
}
