use std::path::PathBuf;

use color_eyre::Result;
use timbal_tui::{app, tui};

fn main() -> Result<()> {
    // TODO: make this configurable / embed as asset.
    let mp3_path = PathBuf::from("Jerobeam Fenderson - Planets.mp3");

    let mut terminal = tui::init()?;
    let result = app::App::new(mp3_path)?.run(&mut terminal);
    tui::restore();
    result
}
