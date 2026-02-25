/// Braille spinner frames for inline loading indicators.
const FRAMES: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

/// A simple braille spinner that cycles through frames based on a tick counter.
pub struct Spinner {
    tick: usize,
}

impl Spinner {
    pub fn new(tick: usize) -> Self {
        Self { tick }
    }

    /// Get the current spinner character.
    pub fn frame(&self) -> char {
        FRAMES[self.tick % FRAMES.len()]
    }
}
