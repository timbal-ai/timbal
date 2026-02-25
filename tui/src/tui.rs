use color_eyre::Result;
use crossterm::execute;
use crossterm::event::{DisableMouseCapture, EnableMouseCapture};
use ratatui::DefaultTerminal;

pub fn init() -> Result<DefaultTerminal> {
    color_eyre::install()?;
    let terminal = ratatui::init();
    execute!(std::io::stdout(), EnableMouseCapture)?;
    Ok(terminal)
}

pub fn restore() {
    let _ = execute!(std::io::stdout(), DisableMouseCapture);
    ratatui::restore();
}
