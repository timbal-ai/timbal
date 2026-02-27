use color_eyre::Result;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers, MouseEventKind};
use std::time::Duration;

pub enum Action {
    Quit,
    Type(char),
    Backspace,
    Submit,
    Cancel,
    PaletteUp,
    PaletteDown,
    ScrollUp(u16),
    ScrollDown(u16),
    Tab,
    BackTab,
    Left,
    Right,
    /// Mouse click at (column, row) in terminal coordinates.
    Click(u16, u16),
    /// Mouse moved to row (for hover effects).
    MouseMove(u16),
}

/// Blocking poll for terminal input with a 16ms timeout.
/// Designed to be called from tokio::task::spawn_blocking.
pub fn poll_blocking() -> Result<Option<Action>> {
    if event::poll(Duration::from_millis(16))? {
        match event::read()? {
            Event::Key(key) if key.kind == KeyEventKind::Press => {
                return Ok(handle_key(key.code, key.modifiers));
            }
            Event::Mouse(mouse) => match mouse.kind {
                MouseEventKind::Down(crossterm::event::MouseButton::Left) => {
                    return Ok(Some(Action::Click(mouse.column, mouse.row)));
                }
                MouseEventKind::ScrollUp => {
                    return Ok(Some(Action::ScrollUp(mouse.column)));
                }
                MouseEventKind::ScrollDown => {
                    return Ok(Some(Action::ScrollDown(mouse.column)));
                }
                MouseEventKind::Moved => {
                    return Ok(Some(Action::MouseMove(mouse.row)));
                }
                _ => {}
            },
            _ => {}
        }
    }
    Ok(None)
}

fn handle_key(code: KeyCode, modifiers: KeyModifiers) -> Option<Action> {
    if modifiers.contains(KeyModifiers::CONTROL) {
        return match code {
            KeyCode::Char('c') => Some(Action::Quit),
            _ => None,
        };
    }

    match code {
        KeyCode::Esc => Some(Action::Cancel),
        KeyCode::Enter => Some(Action::Submit),
        KeyCode::Backspace => Some(Action::Backspace),
        KeyCode::Up => Some(Action::PaletteUp),
        KeyCode::Down => Some(Action::PaletteDown),
        KeyCode::PageUp => Some(Action::ScrollUp(0)),
        KeyCode::PageDown => Some(Action::ScrollDown(0)),
        KeyCode::Tab => Some(Action::Tab),
        KeyCode::BackTab => Some(Action::BackTab),
        KeyCode::Left => Some(Action::Left),
        KeyCode::Right => Some(Action::Right),
        KeyCode::Char(c) => Some(Action::Type(c)),
        _ => None,
    }
}
