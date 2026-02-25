use color_eyre::Result;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use std::time::Duration;

pub enum Action {
    Quit,
    Type(char),
    Backspace,
    Submit,
    Cancel,
    PaletteUp,
    PaletteDown,
    ScrollUp,
    ScrollDown,
}

pub fn poll() -> Result<Option<Action>> {
    if event::poll(Duration::from_millis(16))? {
        if let Event::Key(key) = event::read()? {
            if key.kind == KeyEventKind::Press {
                return Ok(handle_key(key.code, key.modifiers));
            }
        }
    }
    Ok(None)
}

fn handle_key(code: KeyCode, modifiers: KeyModifiers) -> Option<Action> {
    if modifiers.contains(KeyModifiers::CONTROL) && code == KeyCode::Char('c') {
        return Some(Action::Quit);
    }

    match code {
        KeyCode::Esc => Some(Action::Cancel),
        KeyCode::Enter => Some(Action::Submit),
        KeyCode::Backspace => Some(Action::Backspace),
        KeyCode::Up => Some(Action::PaletteUp),
        KeyCode::Down => Some(Action::PaletteDown),
        KeyCode::PageUp => Some(Action::ScrollUp),
        KeyCode::PageDown => Some(Action::ScrollDown),
        KeyCode::Char(c) => Some(Action::Type(c)),
        _ => None,
    }
}
