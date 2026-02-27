use ratatui::{
    style::{Modifier, Style},
    text::{Line, Span},
};

use super::state::AceExplorerState;
use super::widgets::render_yaml_value;
use crate::theme;

pub fn build_evals_content(
    state: &AceExplorerState,
    max_visible: usize,
) -> (Vec<Line<'static>>, Vec<Line<'static>>) {
    let bold = Modifier::BOLD;
    let filtered = state.filtered_evals();
    let mut left: Vec<Line<'static>> = Vec::new();
    let mut right: Vec<Line<'static>> = Vec::new();

    if filtered.is_empty() {
        let msg = if state.search.is_empty() {
            "  No evals defined"
        } else {
            "  No matching evals"
        };
        left.push(Line::from(Span::styled(
            msg,
            Style::default().fg(theme::SUBTLE),
        )));
        return (left, right);
    }

    let total = filtered.len();
    let end = (state.scroll_offset + max_visible).min(total);
    let visible_slice = &filtered[state.scroll_offset..end];

    // Left panel: list of eval names.
    for (row, (_, eval)) in visible_slice.iter().enumerate() {
        let abs_idx = state.scroll_offset + row;
        let is_selected = abs_idx == state.selected_item;
        let indicator = if is_selected { "  > " } else { "    " };
        let ind_style = if is_selected {
            Style::default().fg(theme::IRIS).add_modifier(bold)
        } else {
            Style::default()
        };
        let name_style = if is_selected {
            Style::default().fg(theme::ROSE).add_modifier(bold)
        } else {
            Style::default().fg(theme::ROSE)
        };
        left.push(Line::from(vec![
            Span::styled(indicator, ind_style),
            Span::styled(eval.name.clone(), name_style),
        ]));
    }

    // Right panel: detail view of selected eval.
    if let Some((file, eval)) = filtered.get(state.selected_item) {
        // Name.
        right.push(Line::from(Span::styled(
            eval.name.clone(),
            Style::default().fg(theme::ROSE).add_modifier(bold),
        )));
        right.push(Line::from(""));

        // Source file.
        right.push(Line::from(vec![
            Span::styled(
                "FILE  ",
                Style::default().fg(theme::SUBTLE).add_modifier(bold),
            ),
            Span::styled(file.to_string(), Style::default().fg(theme::TEXT)),
        ]));
        right.push(Line::from(""));

        // Description.
        if let Some(desc) = &eval.description {
            right.push(Line::from(Span::styled(
                "DESCRIPTION",
                Style::default().fg(theme::SUBTLE).add_modifier(bold),
            )));
            right.push(Line::from(vec![
                Span::styled("  ", Style::default()),
                Span::styled(desc.clone(), Style::default().fg(theme::TEXT)),
            ]));
            right.push(Line::from(""));
        }

        // Tags.
        if !eval.tags.is_empty() {
            let mut tag_spans: Vec<Span<'static>> = vec![Span::styled(
                "TAGS  ",
                Style::default().fg(theme::SUBTLE).add_modifier(bold),
            )];
            for (i, tag) in eval.tags.iter().enumerate() {
                if i > 0 {
                    tag_spans.push(Span::styled("  ", Style::default()));
                }
                tag_spans.push(Span::styled("[", Style::default().fg(theme::MUTED)));
                tag_spans.push(Span::styled(tag.clone(), Style::default().fg(theme::FOAM)));
                tag_spans.push(Span::styled("]", Style::default().fg(theme::MUTED)));
            }
            right.push(Line::from(tag_spans));
            right.push(Line::from(""));
        }

        // Timeout.
        if let Some(timeout) = eval.timeout {
            right.push(Line::from(vec![
                Span::styled(
                    "TIMEOUT  ",
                    Style::default().fg(theme::SUBTLE).add_modifier(bold),
                ),
                Span::styled(format!("{timeout}ms"), Style::default().fg(theme::TEXT)),
            ]));
            right.push(Line::from(""));
        }

        // Runnable.
        if let Some(runnable) = &eval.runnable {
            right.push(Line::from(vec![
                Span::styled(
                    "RUNNABLE  ",
                    Style::default().fg(theme::SUBTLE).add_modifier(bold),
                ),
                Span::styled(runnable.clone(), Style::default().fg(theme::PINE)),
            ]));
            right.push(Line::from(""));
        }

        // Params.
        if let Some(params) = &eval.params {
            right.push(Line::from(Span::styled(
                "PARAMS",
                Style::default().fg(theme::SUBTLE).add_modifier(bold),
            )));
            render_yaml_value(&mut right, params, 1);
            right.push(Line::from(""));
        }

        // Extra fields (output, seq!, validators, etc.).
        if !eval.extra.is_empty() {
            let mut keys: Vec<&String> = eval.extra.keys().collect();
            keys.sort_by(|a, b| {
                let order = |k: &str| -> u8 {
                    match k {
                        "output" => 0,
                        k if k.ends_with('!') => 1,
                        _ => 2,
                    }
                };
                order(a).cmp(&order(b)).then(a.cmp(b))
            });

            for key in keys {
                let value = &eval.extra[key];
                let key_color = if key.ends_with('!') {
                    theme::FOAM
                } else if key == "output" {
                    theme::GOLD
                } else {
                    theme::SUBTLE
                };
                right.push(Line::from(Span::styled(
                    key.to_uppercase(),
                    Style::default().fg(key_color).add_modifier(bold),
                )));
                render_yaml_value(&mut right, value, 1);
                right.push(Line::from(""));
            }
        }
    }

    (left, right)
}
