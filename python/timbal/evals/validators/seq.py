import re
from typing import Any, Literal

from pydantic import BaseModel, model_validator

from .base import BaseValidator
from .context import ValidationContext


class AnyPattern(BaseModel):
    """Represents an any! pattern with optional content restrictions.

    Attributes:
        min: Minimum number of spans to match
        max: Maximum number of spans to match (float('inf') for unlimited)
        contains: List of span names that must be present (exact match)
        not_contains: List of span names that must NOT be present (exact match)
    """

    min: int = 1
    max: int | float = float("inf")
    contains: list[str] = []
    not_contains: list[str] = []

    def validate_spans(self, span_names: list[str]) -> bool:
        """Check if the given spans satisfy the content restrictions.

        Args:
            span_names: The span names that would be consumed by this any! pattern

        Returns:
            True if all restrictions are satisfied
        """
        # Check contains: each required pattern must exactly match at least one span
        for required in self.contains:
            if required not in span_names:
                return False

        # Check not_contains: forbidden patterns must not exactly match any span
        for forbidden in self.not_contains:
            if forbidden in span_names:
                return False

        return True


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

    # Parsed patterns stored after validation
    _patterns: list = []

    @model_validator(mode="after")
    def validate_value(self) -> "SeqValidator":
        """Parse the pattern list into pattern objects."""
        if not isinstance(self.value, list):
            raise ValueError(f"expected list of patterns, got {type(self.value).__name__}")

        self._patterns = []
        for item in self.value:
            if isinstance(item, str):
                # Check for wildcard patterns
                pattern = _parse_wildcard(item)
                if pattern:
                    self._patterns.append(pattern)
                else:
                    # Exact match pattern
                    self._patterns.append(item)
            elif isinstance(item, dict):
                # Should be an any! pattern
                if "any!" in item:
                    any_spec = item["any!"]
                    if isinstance(any_spec, dict):
                        pattern = AnyPattern(
                            min=any_spec.get("min", 1),
                            max=any_spec.get("max", float("inf")),
                            contains=any_spec.get("contains", []),
                            not_contains=any_spec.get("not_contains", []),
                        )
                    else:
                        # Simple any! without restrictions
                        pattern = AnyPattern()
                    self._patterns.append(pattern)
                else:
                    raise ValueError(f"unknown pattern type in seq!: {item}")
            else:
                raise ValueError(f"invalid pattern in seq!: {item}")

        return self

    def _get_span_name(self, span_path: str) -> str:
        """Extract the span name from a full path."""
        # Path is like "agent.tool.subtool", we want "subtool"
        return span_path.rsplit(".", 1)[-1] if "." in span_path else span_path

    def _dp_match(self, patterns: list, span_names: list[str]) -> tuple[bool, str]:
        """Use dynamic programming to match patterns against span names.

        Args:
            patterns: List of pattern objects (strings or AnyPattern)
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

                if isinstance(pattern, AnyPattern):
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
                    # Exact match
                    if pattern == span_names[j - 1] and dp[i - 1][j - 1]:
                        dp[i][j] = True

        if dp[m][n]:
            return True, ""

        # Generate helpful error message
        return (
            False,
            f"sequence pattern did not match. Expected {len(patterns)} patterns, got {len(span_names)} spans: {span_names}",
        )

    async def __call__(self, ctx: ValidationContext) -> None:
        """Check if trace spans match the sequence pattern.

        Raises:
            AssertionError: If sequence doesn't match.
        """
        # Get all spans that are direct children of the target. Should already be sorted
        spans = ctx.trace.get_level(self.target)
        span_names = [self._get_span_name(s.path) for s in spans]

        # Run DP matching
        matched, error_msg = self._dp_match(self._patterns, span_names)

        if not matched:
            raise AssertionError(error_msg)
