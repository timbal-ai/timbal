import re
from collections import Counter
from dataclasses import dataclass
from typing import Literal

from ...state.tracing.span import Span
from .base import BaseValidator
from .context import ValidationContext


@dataclass
class AnyPattern:
    """Represents an any! pattern with optional content restrictions."""

    min: int = 1
    max: int | float = float("inf")
    contains: list[str] | None = None
    not_contains: list[str] | None = None

    def validate_spans(self, span_names: list[str]) -> bool:
        """Check if the given spans satisfy the content restrictions."""
        if self.contains:
            for required in self.contains:
                if required not in span_names:
                    return False
        if self.not_contains:
            for forbidden in self.not_contains:
                if forbidden in span_names:
                    return False
        return True


@dataclass
class ParallelPattern:
    """Represents a parallel! pattern with expected span names."""

    span_names: list[str]
    tolerance_ms: int = 0


def _parse_wildcard(s: str) -> AnyPattern | None:
    """Parse wildcard string patterns into AnyPattern.

    Supported patterns:
        - ".."      -> any!(min=1, max=1)    - match exactly 1 span
        - "..."     -> any!(min=0, max=inf)  - match any number of spans
        - "..N"     -> any!(min=0, max=N)    - match at most N spans
        - "n.."     -> any!(min=n, max=inf)  - match at least n spans
        - "n..N"    -> any!(min=n, max=N)    - match between n and N spans

    Returns:
        AnyPattern if pattern matches, None otherwise
    """
    # .. -> exactly 1
    if s == "..":
        return AnyPattern(min=1, max=1)

    # ... -> any number (0+)
    if s == "...":
        return AnyPattern(min=0, max=float("inf"))

    # ..N -> at most N (0 to N)
    match = re.match(r"^\.\.(\d+)$", s)
    if match:
        return AnyPattern(min=0, max=int(match.group(1)))

    # n.. -> at least n (n to inf)
    match = re.match(r"^(\d+)\.\.$", s)
    if match:
        return AnyPattern(min=int(match.group(1)), max=float("inf"))

    # n..N -> between n and N
    match = re.match(r"^(\d+)\.\.(\d+)$", s)
    if match:
        return AnyPattern(min=int(match.group(1)), max=int(match.group(2)))

    return None


class SeqValidator(BaseValidator):
    """Sequence validator - checks if trace spans match a sequence pattern.

    Uses dynamic programming to match patterns against trace spans,
    avoiding greedy matching pitfalls.

    The value should be a list of patterns where each pattern can be:
    - A string: exact span name match
    - A dict with "any!": wildcard pattern with min/max and restrictions

    Example YAML:
    ```yaml
    seq!:
      - llm
      - any!:
          min: 1
          max: 3
          contains: [validate_input]
          not_contains: [error]
      - search_products
      - llm
    ```
    """

    name: Literal["seq!"] = "seq!"  # type: ignore

    def _parse_patterns(self) -> list[str | AnyPattern | ParallelPattern]:
        """Parse the pattern list into pattern objects."""
        if not isinstance(self.value, list):
            raise ValueError(f"expected list of patterns, got {type(self.value).__name__}")

        patterns: list[str | AnyPattern | ParallelPattern] = []
        for item in self.value:
            if isinstance(item, str):
                # Check for wildcard patterns
                pattern = _parse_wildcard(item)
                if pattern:
                    patterns.append(pattern)
                else:
                    # Exact match pattern
                    patterns.append(item)
            elif isinstance(item, dict):
                # Could be any!, parallel!, or a span name with nested validators
                if "any!" in item:
                    any_spec = item["any!"]
                    if isinstance(any_spec, dict):
                        pattern = AnyPattern(
                            min=any_spec.get("min", 1),
                            max=any_spec.get("max", float("inf")),
                            contains=any_spec.get("contains"),
                            not_contains=any_spec.get("not_contains"),
                        )
                    else:
                        # Simple any! without restrictions
                        pattern = AnyPattern()
                    patterns.append(pattern)
                elif "parallel!" in item:
                    # Nested parallel! - extract span names and optional tolerance
                    parallel_spec = item["parallel!"]
                    tolerance_ms = 0

                    if isinstance(parallel_spec, dict):
                        # Dict format with tolerance: {tolerance: 100, spans: [...]}
                        tolerance_ms = parallel_spec.get("tolerance", 0)
                        spans_list = parallel_spec.get("spans", [])
                    elif isinstance(parallel_spec, list):
                        # Simple list format
                        spans_list = parallel_spec
                    else:
                        raise ValueError(f"invalid parallel! spec: {parallel_spec}")

                    # Extract span names from spans list
                    span_names = []
                    for p_item in spans_list:
                        if isinstance(p_item, str):
                            span_names.append(p_item)
                        elif isinstance(p_item, dict) and len(p_item) == 1:
                            span_names.append(next(iter(p_item.keys())))
                    patterns.append(ParallelPattern(span_names=span_names, tolerance_ms=tolerance_ms))
                elif len(item) == 1:
                    # Span name with nested validators (e.g., "get_datetime: {...}")
                    span_name = next(iter(item.keys()))
                    patterns.append(span_name)
                else:
                    raise ValueError(f"unknown pattern type in seq!: {item}")
            else:
                raise ValueError(f"invalid pattern in seq!: {item}")

        return patterns

    def _get_span_name(self, span_path: str) -> str:
        """Extract the span name from a full path."""
        # Path is like "agent.tool.subtool", we want "subtool"
        return span_path.rsplit(".", 1)[-1] if "." in span_path else span_path

    def _dp_match(self, patterns: list[str | AnyPattern | ParallelPattern], span_names: list[str]) -> tuple[bool, str]:
        """Use dynamic programming to match patterns against span names.

        Args:
            patterns: List of pattern objects (strings, AnyPattern, or ParallelPattern)
            span_names: List of span names to match against

        Returns:
            Tuple of (matched: bool, error_message: str)
        """
        m = len(patterns)
        n = len(span_names)

        # dp[i][j] = True if patterns[0:i] can match span_names[0:j]
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = True

        # Initialize first column (matching zero spans)
        for i in range(1, m + 1):
            pattern = patterns[i - 1]
            if isinstance(pattern, AnyPattern) and pattern.min == 0:
                # Optional any! can match zero spans
                if pattern.validate_spans([]):
                    dp[i][0] = dp[i - 1][0]
            else:
                break

        # Fill the DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                pattern = patterns[i - 1]

                if isinstance(pattern, ParallelPattern):
                    # Parallel pattern: match exactly N spans where N = len(span_names)
                    # and all specified span names must be present (order doesn't matter)
                    k = len(pattern.span_names)

                    if j >= k and dp[i - 1][j - k]:
                        # Extract the k spans that would be consumed
                        consumed_spans = span_names[j - k : j]

                        # Check if all parallel span names are present (multiset comparison)
                        # Using Counter to handle duplicate span names correctly
                        if Counter(consumed_spans) == Counter(pattern.span_names):
                            dp[i][j] = True

                elif isinstance(pattern, AnyPattern):
                    # Try consuming k spans where min <= k <= max
                    max_k = j + 1 if pattern.max == float("inf") else min(int(pattern.max) + 1, j + 1)
                    for k in range(pattern.min, max_k):
                        if j - k < 0:
                            continue
                        if not dp[i - 1][j - k]:
                            continue

                        # Extract the k spans that would be consumed
                        consumed_spans = span_names[j - k : j]

                        # Validate content restrictions
                        if pattern.validate_spans(consumed_spans):
                            dp[i][j] = True
                            break
                else:
                    # Exact match (string pattern)
                    if pattern == span_names[j - 1] and dp[i - 1][j - 1]:
                        dp[i][j] = True

        if dp[m][n]:
            return True, ""

        # Generate helpful error message with pattern details
        pattern_strs = []
        for p in patterns:
            if isinstance(p, AnyPattern):
                if p.min == p.max:
                    pattern_strs.append(f"any({p.min})")
                elif p.max == float("inf"):
                    pattern_strs.append(f"any({p.min}+)")
                else:
                    pattern_strs.append(f"any({p.min}-{p.max})")
            elif isinstance(p, ParallelPattern):
                pattern_strs.append(f"parallel({', '.join(p.span_names)})")
            else:
                pattern_strs.append(str(p))

        return (
            False,
            f"expected sequence ({' -> '.join(pattern_strs)}) but got ({' -> '.join(span_names)})",
        )

    def _spans_overlap(self, span_a: Span, span_b: Span, tolerance_ms: int = 0) -> bool:
        """Check if two spans have overlapping time ranges.

        Two spans overlap if they share any point in time (within tolerance).
        Using <= to handle edge case where spans have identical start/end times.

        Args:
            span_a: First span
            span_b: Second span
            tolerance_ms: Tolerance in milliseconds - spans can have this much gap and still be considered parallel
        """
        t1_a = span_a.t1 if span_a.t1 is not None else float("inf")
        t1_b = span_b.t1 if span_b.t1 is not None else float("inf")
        return span_a.t0 <= t1_b + tolerance_ms and span_b.t0 <= t1_a + tolerance_ms

    def _validate_parallel_spans(self, spans: list[Span], tolerance_ms: int = 0) -> tuple[bool, str]:
        """Check if all spans ran in parallel (all pairs overlap)."""
        if len(spans) < 2:
            return True, ""

        for i, span_a in enumerate(spans):
            for span_b in spans[i + 1 :]:
                if not self._spans_overlap(span_a, span_b, tolerance_ms):
                    name_a = self._get_span_name(span_a.path)
                    name_b = self._get_span_name(span_b.path)
                    t1_a_str = f"{span_a.t1}" if span_a.t1 else "running"
                    t1_b_str = f"{span_b.t1}" if span_b.t1 else "running"
                    return False, (
                        f"spans '{name_a}' and '{name_b}' in parallel! did not run in parallel. "
                        f"'{name_a}': {span_a.t0}-{t1_a_str}, "
                        f"'{name_b}': {span_b.t0}-{t1_b_str}"
                    )
        return True, ""

    def _find_parallel_matches(
        self, patterns: list[str | AnyPattern | ParallelPattern], spans: list[Span]
    ) -> list[tuple[ParallelPattern, list[Span]]]:
        """Find which spans matched each ParallelPattern.

        Uses greedy matching based on the DP result to identify span groups.
        """
        matches = []
        span_idx = 0
        span_names = [self._get_span_name(s.path) for s in spans]

        for pattern in patterns:
            if isinstance(pattern, ParallelPattern):
                k = len(pattern.span_names)
                # Find k spans starting from span_idx that match the parallel pattern
                consumed = spans[span_idx : span_idx + k]
                matches.append((pattern, consumed))
                span_idx += k
            elif isinstance(pattern, AnyPattern):
                # Find how many spans this pattern consumed
                # We need to find the minimum k that allows remaining patterns to match
                remaining_patterns = patterns[patterns.index(pattern) + 1 :]
                min_remaining = sum(
                    len(p.span_names) if isinstance(p, ParallelPattern) else (p.min if isinstance(p, AnyPattern) else 1)
                    for p in remaining_patterns
                )
                remaining_spans = len(spans) - span_idx
                k = remaining_spans - min_remaining
                k = max(pattern.min, min(k, int(pattern.max) if pattern.max != float("inf") else k))
                span_idx += k
            else:
                # Exact match - consume 1 span
                span_idx += 1

        return matches

    async def __call__(self, ctx: ValidationContext) -> None:
        """Check if trace spans match the sequence pattern.

        Raises:
            AssertionError: If sequence doesn't match.
        """
        # Parse patterns from value
        try:
            patterns = self._parse_patterns()
        except Exception as e:
            raise AssertionError(f"failed to parse patterns: {e}") from e

        # Get all spans that are direct children of the target. Should already be sorted
        spans = ctx.trace.get_level(self.target)
        span_names = [self._get_span_name(s.path) for s in spans]

        if not patterns:
            raise AssertionError(f"no patterns defined in seq! value: {self.value}")

        if not span_names:
            # Generate pattern strings for error message
            pattern_strs = []
            for p in patterns:
                if isinstance(p, AnyPattern):
                    if p.min == p.max:
                        pattern_strs.append(f"any({p.min})")
                    elif p.max == float("inf"):
                        pattern_strs.append(f"any({p.min}+)")
                    else:
                        pattern_strs.append(f"any({p.min}-{p.max})")
                elif isinstance(p, ParallelPattern):
                    pattern_strs.append(f"parallel({', '.join(p.span_names)})")
                else:
                    pattern_strs.append(str(p))
            raise AssertionError(
                f"expected sequence ({' -> '.join(pattern_strs)}) but got no spans at target '{self.target}'"
            )

        # Run DP matching
        matched, error_msg = self._dp_match(patterns, span_names)

        if not matched:
            raise AssertionError(error_msg)

        # If matched, also validate that parallel patterns actually ran in parallel
        parallel_matches = self._find_parallel_matches(patterns, spans)
        for pattern, matched_spans in parallel_matches:
            is_parallel, parallel_error = self._validate_parallel_spans(matched_spans, pattern.tolerance_ms)
            if not is_parallel:
                raise AssertionError(parallel_error)
