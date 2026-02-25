use color_eyre::Result;
use timbal_tui::{app, tui};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let cli = parse_cli();

    // Change working directory if a path was provided.
    if let Some(ref path) = cli.path {
        let target = std::path::Path::new(path);
        std::env::set_current_dir(target)
            .map_err(|e| color_eyre::eyre::eyre!("Cannot cd to {path}: {e}"))?;
    }

    let mut terminal = tui::init()?;
    let result = app::App::new(cli.profile)?.run(&mut terminal).await;
    tui::restore();
    result
}

struct Cli {
    profile: Option<String>,
    path: Option<String>,
}

fn parse_cli() -> Cli {
    let args: Vec<String> = std::env::args().collect();
    let mut profile = None;
    let mut path = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--profile" => {
                if let Some(val) = args.get(i + 1) {
                    profile = Some(val.clone());
                    i += 1;
                }
            }
            "--path" => {
                if let Some(val) = args.get(i + 1) {
                    path = Some(val.clone());
                    i += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }

    Cli { profile, path }
}
