use color_eyre::Result;
use ratatui::DefaultTerminal;

pub fn init() -> Result<DefaultTerminal> {
    color_eyre::install()?;
    let terminal = ratatui::init();
    Ok(terminal)
}

pub fn restore() {
    ratatui::restore();
}
