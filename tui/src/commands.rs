pub struct Command {
    pub name: &'static str,
    pub description: &'static str,
    pub usage: &'static str,
}

pub const COMMANDS: &[Command] = &[
    Command {
        name: "/configure",
        description: "Set up API key and credentials",
        usage: "/configure [profile]",
    },
    Command {
        name: "/quit",
        description: "Exit Timbal",
        usage: "/quit",
    },
];

/// Return commands whose name starts with the input prefix,
/// or whose full name matches the start of the input (for argument passing).
/// e.g. "/q" → "/quit", "/configure prod" → "/configure"
pub fn filter(input: &str) -> Vec<&'static Command> {
    let lower = input.to_lowercase();
    COMMANDS
        .iter()
        .filter(|cmd| {
            // Prefix match: "/con" matches "/configure"
            cmd.name.starts_with(&*lower)
            // Full command with args: "/configure prod" still shows "/configure"
            || lower.starts_with(&format!("{} ", cmd.name))
        })
        .collect()
}

/// Extract the argument portion after the command name, if any.
/// "/configure prod" → Some("prod"), "/configure" → None
pub fn parse_arg<'a>(input: &'a str, cmd_name: &str) -> Option<&'a str> {
    let rest = input.trim().strip_prefix(cmd_name)?.trim();
    if rest.is_empty() { None } else { Some(rest) }
}
