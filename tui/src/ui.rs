use ratatui::Frame;

use crate::app::App;
use crate::screens;

pub fn render(app: &mut App, frame: &mut Frame) {
    screens::home::render(app, frame);
}
