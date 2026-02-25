use color_eyre::Result;
use timbal_tui::{app, tui};

fn main() -> Result<()> {
    let mut terminal = tui::init()?;
    let result = app::App::new()?.run(&mut terminal);
    tui::restore();
    result
}
