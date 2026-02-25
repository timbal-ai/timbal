use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Style},
    symbols::Marker,
    widgets::{Axis, Chart, Dataset, GraphType, Widget},
};

use crate::theme;

/// Stateless vectorscope widget. Renders a precomputed frame of (left, right) points.
pub struct Vectorscope<'a> {
    points: &'a [(f64, f64)],
    scale: f64,
}

impl<'a> Vectorscope<'a> {
    pub fn new(points: &'a [(f64, f64)]) -> Self {
        Self {
            points,
            scale: 1.0,
        }
    }
}

impl Widget for Vectorscope<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if self.points.is_empty() {
            return;
        }

        let s = self.scale;
        let h_line: Vec<(f64, f64)> = vec![(-s, 0.0), (s, 0.0)];
        let v_line: Vec<(f64, f64)> = vec![(0.0, -s), (0.0, s)];

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
                .graph_type(GraphType::Scatter)
                .style(Style::default().fg(theme::LOVE))
                .data(self.points),
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

        chart.render(area, buf);
    }
}
