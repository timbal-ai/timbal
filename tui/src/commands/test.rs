use super::{CommandHandler, CommandMeta};
use crate::app::AppEvent;

pub struct TestCommand;

impl CommandHandler for TestCommand {
    fn meta(&self) -> CommandMeta {
        CommandMeta {
            name: "/test",
            description: "Run a streaming test (toy async Python)",
            usage: "/test",
        }
    }

    fn execute(&self, _args: Option<&str>) -> Vec<AppEvent> {
        vec![AppEvent::RunStreamingTest]
    }
}
