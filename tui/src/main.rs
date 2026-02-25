use color_eyre::Result;
use timbal_tui::{app, tui};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let profile = parse_profile();
    let mut terminal = tui::init()?;
    let result = app::App::new(profile)?.run(&mut terminal).await;
    tui::restore();
    result
}

/// Parse --profile <name> from CLI args.
fn parse_profile() -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--profile" {
            if let Some(val) = args.get(i + 1) {
                return Some(val.clone());
            }
        }
        i += 1;
    }
    None
}
