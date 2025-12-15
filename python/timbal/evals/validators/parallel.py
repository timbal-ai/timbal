from typing import Literal

from ...state.tracing.span import Span
from .base import BaseValidator
from .context import ValidationContext


class ParallelValidator(BaseValidator):
    """Parallel validator - checks if spans ran in parallel (overlapping time).

    Unlike seq! which requires spans in a specific order, parallel! verifies
    that all specified spans exist AND that they ran with overlapping time ranges,
    indicating actual parallel execution.

    The value should be a list of span names that must all run in parallel.

    Example YAML:
    ```yaml
    parallel!:
      - fetch_user
      - fetch_orders
      - fetch_recommendations
    ```

    With tolerance (in milliseconds):
    ```yaml
    parallel!:
      tolerance: 100  # Allow 100ms gap between spans
      spans:
        - fetch_user
        - fetch_orders
    ```

    This validates that all three spans exist and their execution times overlap
    (within tolerance), meaning they ran concurrently.
    """

    name: Literal["parallel!"] = "parallel!"  # type: ignore

    # Default tolerance in milliseconds for parallel overlap check
    DEFAULT_TOLERANCE_MS: int = 0

    def _parse_value(self) -> tuple[list[str], int]:
        """Parse the value to extract span names and tolerance.

        Returns:
            Tuple of (span_names, tolerance_ms)
        """
        tolerance_ms = self.DEFAULT_TOLERANCE_MS

        # Handle dict format with tolerance
        if isinstance(self.value, dict):
            tolerance_ms = self.value.get("tolerance", self.DEFAULT_TOLERANCE_MS)
            spans_value = self.value.get("spans", [])
        elif isinstance(self.value, list):
            spans_value = self.value
        else:
            raise ValueError(f"expected list or dict, got {type(self.value).__name__}")

        expected_spans = []
        for item in spans_value:
            if isinstance(item, str):
                expected_spans.append(item)
            elif isinstance(item, dict):
                # Dict with span name as key (for nested validators)
                # Just extract the span name
                if len(item) != 1:
                    raise ValueError(f"expected single span name in dict, got {item}")
                span_name = next(iter(item.keys()))
                expected_spans.append(span_name)
            else:
                raise ValueError(f"invalid item in parallel!: {item}")

        return expected_spans, tolerance_ms

    def _get_span_name(self, span_path: str) -> str:
        """Extract the span name from a full path."""
        return span_path.rsplit(".", 1)[-1] if "." in span_path else span_path

    def _spans_overlap(self, span_a: Span, span_b: Span, tolerance_ms: int = 0) -> bool:
        """Check if two spans have overlapping time ranges.

        Two spans overlap if they share any point in time (within tolerance).
        Using <= to handle edge case where spans have identical start/end times.

        Args:
            span_a: First span
            span_b: Second span
            tolerance_ms: Tolerance in milliseconds - spans can have this much gap and still be considered parallel
        """
        # If either span hasn't completed, use a large value for t1
        t1_a = span_a.t1 if span_a.t1 is not None else float("inf")
        t1_b = span_b.t1 if span_b.t1 is not None else float("inf")

        # Add tolerance to the end times
        return span_a.t0 <= t1_b + tolerance_ms and span_b.t0 <= t1_a + tolerance_ms

    def _all_spans_parallel(self, spans: list[Span], tolerance_ms: int = 0) -> tuple[bool, str]:
        """Check if all spans ran in parallel (all pairs overlap).

        Returns:
            Tuple of (all_parallel, error_message)
        """
        if len(spans) < 2:
            return True, ""

        # Check all pairs
        for i, span_a in enumerate(spans):
            for span_b in spans[i + 1 :]:
                if not self._spans_overlap(span_a, span_b, tolerance_ms):
                    name_a = self._get_span_name(span_a.path)
                    name_b = self._get_span_name(span_b.path)
                    t1_a_str = f"{span_a.t1}" if span_a.t1 else "running"
                    t1_b_str = f"{span_b.t1}" if span_b.t1 else "running"
                    return False, (
                        f"spans '{name_a}' and '{name_b}' did not run in parallel. "
                        f"'{name_a}': {span_a.t0}-{t1_a_str}, "
                        f"'{name_b}': {span_b.t0}-{t1_b_str}"
                    )

        return True, ""

    async def __call__(self, ctx: ValidationContext) -> None:
        """Check if all expected spans exist and ran in parallel.

        Raises:
            AssertionError: If any expected span is missing or spans didn't run in parallel.
        """
        # Parse expected spans and tolerance from value
        expected_spans, tolerance_ms = self._parse_value()

        target_path = self.target

        # Get all spans that are direct children of the target
        all_spans = ctx.trace.get_level(target_path)

        # Build a map of span name to span
        span_map: dict[str, Span] = {}
        for span in all_spans:
            name = self._get_span_name(span.path)
            span_map[name] = span

        # Check that all expected spans exist
        missing = [name for name in expected_spans if name not in span_map]
        if missing:
            raise AssertionError(f"missing spans: {missing}. Expected {expected_spans}, got {sorted(span_map.keys())}")

        # Get the spans we care about
        matched_spans = [span_map[name] for name in expected_spans]

        # Check that all matched spans ran in parallel
        all_parallel, error_msg = self._all_spans_parallel(matched_spans, tolerance_ms)
        if not all_parallel:
            raise AssertionError(error_msg)
