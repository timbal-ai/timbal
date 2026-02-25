use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Style},
    symbols::Marker,
    widgets::{Axis, Chart, Dataset, GraphType, StatefulWidget, Widget},
};

use crate::audio::SampleBuffer;
use crate::theme;

const FRAMES: usize = 2048;

pub struct VectorscopeState {
    pub scale: f64,
    pub scatter: bool,
}

impl Default for VectorscopeState {
    fn default() -> Self {
        Self {
            scale: 1.0,
            scatter: true,
        }
    }
}

/// Reusable vectorscope widget. Plots stereo L/R samples as a Lissajous figure.
/// Use as a loading animation or standalone visualization.
pub struct Vectorscope<'a> {
    pub sample_buffer: &'a SampleBuffer,
}

impl<'a> Vectorscope<'a> {
    pub fn new(sample_buffer: &'a SampleBuffer) -> Self {
        Self { sample_buffer }
    }
}

impl StatefulWidget for Vectorscope<'_> {
    type State = VectorscopeState;

    fn render(self, area: Rect, buf: &mut Buffer, state: &mut Self::State) {
        let raw: Vec<f32> = {
            let guard = self.sample_buffer.lock().unwrap();
            let needed = FRAMES * 2;
            if guard.len() < needed {
                return;
            }
            let start = guard.len() - needed;
            guard[start..].to_vec()
        };

        let points: Vec<(f64, f64)> = (0..FRAMES)
            .map(|i| {
                let left = raw[i * 2] as f64;
                let right = raw[i * 2 + 1] as f64;
                (left, right)
            })
            .collect();

        let s = state.scale;
        let h_line: Vec<(f64, f64)> = vec![(-s, 0.0), (s, 0.0)];
        let v_line: Vec<(f64, f64)> = vec![(0.0, -s), (0.0, s)];

        let graph_type = if state.scatter {
            GraphType::Scatter
        } else {
            GraphType::Line
        };

        let datasets = vec![
            Dataset::default()
                .marker(Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(Color::DarkGray))
                .data(&h_line),
            Dataset::default()
                .marker(Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(Color::DarkGray))
                .data(&v_line),
            Dataset::default()
                .marker(Marker::Braille)
                .graph_type(graph_type)
                .style(Style::default().fg(theme::LOVE))
                .data(&points),
        ];

        let chart = Chart::new(datasets)
            .x_axis(
                Axis::default()
                    .bounds([-s, s])
                    .style(Style::default().fg(Color::DarkGray)),
            )
            .y_axis(
                Axis::default()
                    .bounds([-s, s])
                    .style(Style::default().fg(Color::DarkGray)),
            )
            .style(Style::default());

        Widget::render(chart, area, buf);
    }
}
