use ratatui::{
    style::{Modifier, Style},
    text::{Line, Span},
};

use super::state::AceExplorerState;
use super::widgets::{build_field_lines, build_metadata_lines};
use crate::theme;

pub fn build_policies_content(
    state: &AceExplorerState,
    max_visible: usize,
) -> (Vec<Line<'static>>, Vec<Line<'static>>) {
    let bold = Modifier::BOLD;
    let filtered = state.filtered_policies();
    let mut left: Vec<Line<'static>> = Vec::new();
    let mut right: Vec<Line<'static>> = Vec::new();
    let left_dimmed = state.editing_right;

    if filtered.is_empty() {
        let msg = if state.search.is_empty() {
            "  No policies defined"
        } else {
            "  No matching policies"
        };
        left.push(Line::from(Span::styled(
            msg,
            Style::default().fg(theme::SUBTLE),
        )));
        return (left, right);
    }

    let total = filtered.len();
    let end = (state.scroll_offset + max_visible).min(total);
    let visible = &filtered[state.scroll_offset..end];

    for (row, (name, _)) in visible.iter().enumerate() {
        let abs_idx = state.scroll_offset + row;
        let is_selected = abs_idx == state.selected_item;
        let indicator = if is_selected { "  > " } else { "    " };
        let (ind_style, name_style) = if left_dimmed {
            (
                Style::default().fg(theme::MUTED),
                Style::default().fg(theme::MUTED),
            )
        } else {
            (
                if is_selected {
                    Style::default().fg(theme::IRIS).add_modifier(bold)
                } else {
                    Style::default()
                },
                if is_selected {
                    Style::default().fg(theme::IRIS).add_modifier(bold)
                } else {
                    Style::default().fg(theme::IRIS)
                },
            )
        };
        left.push(Line::from(vec![
            Span::styled(indicator, ind_style),
            Span::styled(name.clone(), name_style),
        ]));
    }

    // Right panel.
    if let Some((name, pol)) = filtered.get(state.selected_item) {
        right.push(Line::from(Span::styled(
            name.clone(),
            Style::default().fg(theme::IRIS).add_modifier(bold),
        )));
        right.push(Line::from(""));

        // Field 0: condition
        let cond_val = pol.condition.clone().unwrap_or_default();
        build_field_lines(state, &mut right, 0, "CONDITION", &cond_val);
        right.push(Line::from(""));

        // Field 1: action
        let action_val = pol.action.clone().unwrap_or_default();
        build_field_lines(state, &mut right, 1, "ACTION", &action_val);
        right.push(Line::from(""));

        // Field 2: provides
        let provides_val = pol
            .provides
            .as_ref()
            .map(|v| v.join(", "))
            .unwrap_or_default();
        build_field_lines(state, &mut right, 2, "PROVIDES", &provides_val);

        // Metadata (read-only).
        build_metadata_lines(
            &mut right,
            pol.created_at.as_deref(),
            pol.created_by.as_deref(),
            pol.updated_at.as_deref(),
            pol.updated_by.as_deref(),
        );
    }

    (left, right)
}
