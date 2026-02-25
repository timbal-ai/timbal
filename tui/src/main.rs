use color_eyre::Result;
use timbal_tui::{app, tui};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let mut terminal = tui::init()?;
    let result = app::App::new()?.run(&mut terminal).await;
    tui::restore();
    result
}
