use super::{CommandHandler, CommandMeta};
use crate::app::AppEvent;

pub struct ConfigureCommand;

impl CommandHandler for ConfigureCommand {
    fn meta(&self) -> CommandMeta {
        CommandMeta {
            name: "/configure",
            description: "Set up API key and credentials",
            usage: "/configure [profile]",
        }
    }

    fn execute(&self, args: Option<&str>) -> Vec<AppEvent> {
        let profile = args.map(str::to_string);
        vec![AppEvent::OpenConfigure(profile)]
    }
}
