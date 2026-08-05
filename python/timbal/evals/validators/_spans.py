"""Shared span-timing helpers for seq! and parallel! validators."""

from ...state.tracing.span import Span


def get_span_name(span_path: str) -> str:
    """Extract the span name from a full path."""
    return span_path.rsplit(".", 1)[-1] if "." in span_path else span_path


def spans_overlap(span_a: Span, span_b: Span, tolerance_ms: int = 0) -> bool:
    """Check if two spans have overlapping time ranges.

    Two spans overlap if they share any point in time (within tolerance).
    Using <= to handle edge case where spans have identical start/end times.
    If either span hasn't completed, a large value is used for its end time.
    """
    t1_a = span_a.t1 if span_a.t1 is not None else float("inf")
    t1_b = span_b.t1 if span_b.t1 is not None else float("inf")
    return span_a.t0 <= t1_b + tolerance_ms and span_b.t0 <= t1_a + tolerance_ms


def validate_parallel_spans(spans: list[Span], tolerance_ms: int = 0, *, label: str = "") -> tuple[bool, str]:
    """Check if all spans ran in parallel (all pairs overlap).

    *label* is inserted into the error message after the span names (e.g.
    ``" in parallel!"`` for seq!'s nested parallel patterns).

    Returns:
        Tuple of (all_parallel, error_message)
    """
    if len(spans) < 2:
        return True, ""

    for i, span_a in enumerate(spans):
        for span_b in spans[i + 1 :]:
            if not spans_overlap(span_a, span_b, tolerance_ms):
                name_a = get_span_name(span_a.path)
                name_b = get_span_name(span_b.path)
                t1_a_str = f"{span_a.t1}" if span_a.t1 else "running"
                t1_b_str = f"{span_b.t1}" if span_b.t1 else "running"
                return False, (
                    f"spans '{name_a}' and '{name_b}'{label} did not run in parallel. "
                    f"'{name_a}': {span_a.t0}-{t1_a_str}, "
                    f"'{name_b}': {span_b.t0}-{t1_b_str}"
                )
    return True, ""
