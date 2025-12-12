from typing import Any


class AnyPattern:
    """Represents an any! pattern with optional content restrictions."""

    def __init__(
        self, min_count: int, max_count: int, contains: list[str] | None = None, not_contains: list[str] | None = None
    ):
        self.min = min_count
        self.max = max_count
        self.contains = contains or []
        self.not_contains = not_contains or []

    def __repr__(self):
        parts = [f"{self.min},{self.max}"]
        if self.contains:
            parts.append(f"contains={self.contains}")
        if self.not_contains:
            parts.append(f"not_contains={self.not_contains}")
        return f"any({', '.join(parts)})"

    def validate_spans(self, spans: list[str]) -> bool:
        """
        Check if the given spans satisfy the content restrictions.

        Args:
            spans: The spans that would be consumed by this any! pattern

        Returns:
            True if all restrictions are satisfied
        """
        # Check contains: each required pattern must exactly match at least one span
        for required in self.contains:
            if required not in spans:
                return False

        # Check not_contains: forbidden patterns must not exactly match any span
        for forbidden in self.not_contains:
            if forbidden in spans:
                return False

        return True


def matches(pattern: Any, span: Any) -> bool:
    """Check if a pattern matches a span (exact match for strings)."""
    if isinstance(pattern, str):
        return pattern == span
    return False


def dp_match(patterns: list, spans: list, debug: bool = False) -> bool:
    m = len(patterns)
    n = len(spans)

    # Create DP table
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True

    # Initialize first column
    for i in range(1, m + 1):
        pattern = patterns[i - 1]
        if isinstance(pattern, AnyPattern) and pattern.min == 0:
            # Optional any! can match zero spans, but must validate empty list
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
                # Handle unbounded max (infinity)
                max_k = j + 1 if pattern.max == float("inf") else min(pattern.max + 1, j + 1)
                for k in range(pattern.min, max_k):
                    if not dp[i - 1][j - k]:
                        continue

                    # Extract the k spans that would be consumed
                    consumed_spans = spans[j - k : j]

                    # Validate content restrictions
                    if pattern.validate_spans(consumed_spans):
                        dp[i][j] = True
                        if debug:
                            print(f"  ✓ dp[{i}][{j}]: {pattern} consumed {consumed_spans}")
                        break
                    elif debug:
                        print(f"  ✗ dp[{i}][{j}]: {pattern} rejected {consumed_spans}")
            else:
                # Exact match
                if matches(pattern, spans[j - 1]) and dp[i - 1][j - 1]:
                    dp[i][j] = True
                    if debug:
                        print(f"  ✓ dp[{i}][{j}]: '{pattern}' matched '{spans[j - 1]}'")

    return dp[m][n]
