"""
Unbounded Any Pattern Tests
============================

Testing any! patterns with unlimited max (represented as None or a very large number).

Syntax from YAML:
- "n.." means any(min=n, max=unlimited)
- ".." means any(min=1, max=unlimited)
- "..N" means any(min=1, max=N)  (already tested)
- "N..M" means any(min=N, max=M)  (already tested)

Use cases for unbounded patterns:
1. "At least N items, no upper limit"
2. "Any number of items (1+)"
3. "Capture all remaining items"
4. "Match everything between two markers"
"""

from seq_dp_with_restrictions import AnyPattern
from seq_dp_with_restrictions import dp_match as base_dp_match



# Wrapper to ignore debug parameter
def dp_match(patterns, spans, debug=False):
    return base_dp_match(patterns, spans)


def test_unbounded_patterns():
    """Test patterns with unlimited max values."""

    print("=" * 120)
    print("UNBOUNDED ANY PATTERN TESTS")
    print("=" * 120)

    # Test 1: Simple unbounded (1..)
    print("\n" + "=" * 120)
    print("TEST 1: any(1, unlimited) - At least 1 item, no upper limit")
    print("=" * 120)

    patterns = ["start", AnyPattern(1, float("inf")), "end"]

    # Should pass with 1 item
    spans_1 = ["start", "middle", "end"]
    result_1 = dp_match(patterns, spans_1, debug=True)
    print(f"\nWith 1 item: {result_1} - {'✓ PASS' if result_1 else '✗ FAIL'}")

    # Should pass with many items
    spans_many = ["start", "a", "b", "c", "d", "e", "f", "g", "h", "end"]
    result_many = dp_match(patterns, spans_many, debug=True)
    print(f"\nWith 8 items: {result_many} - {'✓ PASS' if result_many else '✗ FAIL'}")

    # Should fail with 0 items
    spans_0 = ["start", "end"]
    result_0 = dp_match(patterns, spans_0, debug=True)
    print(
        f"\nWith 0 items: {result_0} - {'✓ PASS (correctly failed)' if not result_0 else '✗ FAIL (should have failed)'}"
    )

    # Test 2: Unbounded with restrictions
    print("\n" + "=" * 120)
    print("TEST 2: any(2, unlimited, contains=['required']) - At least 2, must contain specific item")
    print("=" * 120)

    patterns2 = ["start", AnyPattern(2, float("inf"), contains=["required_item"]), "end"]

    # Should pass - has required_item and at least 2 items
    spans2_pass = ["start", "optional", "required_item", "more", "stuff", "end"]
    result2_pass = dp_match(patterns2, spans2_pass, debug=True)
    print(f"\nWith required item (4 total): {result2_pass} - {'✓ PASS' if result2_pass else '✗ FAIL'}")

    # Should fail - has required_item but only 1 item total (min=2)
    spans2_fail_min = ["start", "required_item", "end"]
    result2_fail_min = dp_match(patterns2, spans2_fail_min, debug=True)
    print(
        f"\nWith required but only 1 item: {result2_fail_min} - {'✓ PASS (correctly failed)' if not result2_fail_min else '✗ FAIL (should have failed)'}"
    )

    # Should fail - has 2+ items but missing required_item
    spans2_fail_contains = ["start", "a", "b", "c", "end"]
    result2_fail_contains = dp_match(patterns2, spans2_fail_contains, debug=True)
    print(
        f"\nWith 3 items but no required: {result2_fail_contains} - {'✓ PASS (correctly failed)' if not result2_fail_contains else '✗ FAIL (should have failed)'}"
    )

    # Test 3: Multiple unbounded patterns
    print("\n" + "=" * 120)
    print("TEST 3: Multiple unbounded patterns - Greedy consumption challenge")
    print("=" * 120)

    patterns3 = [
        "start",
        AnyPattern(1, float("inf")),  # Could consume everything!
        "middle",
        AnyPattern(1, float("inf")),  # This also needs at least 1
        "end",
    ]

    spans3 = ["start", "a", "b", "middle", "c", "d", "end"]
    result3 = dp_match(patterns3, spans3, debug=True)
    print(f"\nMultiple unbounded: {result3} - {'✓ PASS' if result3 else '✗ FAIL'}")
    print("Note: DP should correctly avoid first any! consuming 'middle'")

    # Test 4: Unbounded at end (capture all remaining)
    print("\n" + "=" * 120)
    print("TEST 4: Unbounded at end - Capture all remaining items")
    print("=" * 120)

    patterns4 = [
        "start",
        "initialize",
        AnyPattern(0, float("inf")),  # Everything else is optional
    ]

    # With many remaining items
    spans4_many = ["start", "initialize", "a", "b", "c", "d", "e"]
    result4_many = dp_match(patterns4, spans4_many, debug=True)
    print(f"\nWith 5 remaining items: {result4_many} - {'✓ PASS' if result4_many else '✗ FAIL'}")

    # With zero remaining items (min=0)
    spans4_zero = ["start", "initialize"]
    result4_zero = dp_match(patterns4, spans4_zero, debug=True)
    print(f"\nWith 0 remaining items: {result4_zero} - {'✓ PASS' if result4_zero else '✗ FAIL'}")

    # Test 5: Unbounded with not_contains
    print("\n" + "=" * 120)
    print("TEST 5: any(1, unlimited, not_contains=['error']) - Must not contain errors")
    print("=" * 120)

    patterns5 = ["start", AnyPattern(1, float("inf"), not_contains=["error", "crash"]), "end"]

    # Should pass - many items, no errors
    spans5_pass = ["start", "a", "b", "c", "d", "e", "end"]
    result5_pass = dp_match(patterns5, spans5_pass, debug=True)
    print(f"\nWith no errors (5 items): {result5_pass} - {'✓ PASS' if result5_pass else '✗ FAIL'}")

    # Should fail - has error
    spans5_fail = ["start", "a", "error", "c", "d", "end"]
    result5_fail = dp_match(patterns5, spans5_fail, debug=True)
    print(
        f"\nWith error present: {result5_fail} - {'✓ PASS (correctly failed)' if not result5_fail else '✗ FAIL (should have failed)'}"
    )

    # Test 6: Edge case - everything is unbounded
    print("\n" + "=" * 120)
    print("TEST 6: Only unbounded pattern - Matches entire sequence")
    print("=" * 120)

    patterns6 = [AnyPattern(1, float("inf"))]

    spans6 = ["a", "b", "c", "d"]
    result6 = dp_match(patterns6, spans6, debug=True)
    print(f"\nJust unbounded any!: {result6} - {'✓ PASS' if result6 else '✗ FAIL'}")

    # Test 7: Practical example - logs with unbounded middle
    print("\n" + "=" * 120)
    print("TEST 7: Practical - Log validation with variable middle section")
    print("=" * 120)

    patterns7 = [
        "initialize_logger",
        "log_start",
        AnyPattern(0, float("inf"), not_contains=["fatal", "critical"]),  # Any logs, but no fatals
        "log_end",
        "cleanup_logger",
    ]

    # With many log entries
    spans7_many = [
        "initialize_logger",
        "log_start",
        "log_info",
        "log_debug",
        "log_warning",
        "log_info",
        "log_debug",
        "log_end",
        "cleanup_logger",
    ]
    result7_many = dp_match(patterns7, spans7_many, debug=True)
    print(f"\nWith 5 log entries: {result7_many} - {'✓ PASS' if result7_many else '✗ FAIL'}")

    # With no middle logs (min=0)
    spans7_none = ["initialize_logger", "log_start", "log_end", "cleanup_logger"]
    result7_none = dp_match(patterns7, spans7_none, debug=True)
    print(f"\nWith 0 log entries: {result7_none} - {'✓ PASS' if result7_none else '✗ FAIL'}")

    # With fatal log (should fail)
    spans7_fatal = ["initialize_logger", "log_start", "log_info", "fatal", "log_end", "cleanup_logger"]
    result7_fatal = dp_match(patterns7, spans7_fatal, debug=True)
    print(
        f"\nWith fatal log: {result7_fatal} - {'✓ PASS (correctly failed)' if not result7_fatal else '✗ FAIL (should have failed)'}"
    )

    # Summary
    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)
    print(f"Test 1 - Simple unbounded (1 item):        {'✓' if result_1 else '✗'}")
    print(f"Test 1 - Simple unbounded (8 items):       {'✓' if result_many else '✗'}")
    print(f"Test 1 - Simple unbounded (0 items fail):  {'✓' if not result_0 else '✗'}")
    print(f"Test 2 - With contains (pass):             {'✓' if result2_pass else '✗'}")
    print(f"Test 2 - With contains (fail min):         {'✓' if not result2_fail_min else '✗'}")
    print(f"Test 2 - With contains (fail contains):    {'✓' if not result2_fail_contains else '✗'}")
    print(f"Test 3 - Multiple unbounded:               {'✓' if result3 else '✗'}")
    print(f"Test 4 - End capture (many):               {'✓' if result4_many else '✗'}")
    print(f"Test 4 - End capture (zero):               {'✓' if result4_zero else '✗'}")
    print(f"Test 5 - With not_contains (pass):         {'✓' if result5_pass else '✗'}")
    print(f"Test 5 - With not_contains (fail):         {'✓' if not result5_fail else '✗'}")
    print(f"Test 6 - Only unbounded:                   {'✓' if result6 else '✗'}")
    print(f"Test 7 - Logs with many:                   {'✓' if result7_many else '✗'}")
    print(f"Test 7 - Logs with none:                   {'✓' if result7_none else '✗'}")
    print(f"Test 7 - Logs with fatal (fail):           {'✓' if not result7_fatal else '✗'}")

    all_pass = (
        result_1
        and result_many
        and not result_0
        and result2_pass
        and not result2_fail_min
        and not result2_fail_contains
        and result3
        and result4_many
        and result4_zero
        and result5_pass
        and not result5_fail
        and result6
        and result7_many
        and result7_none
        and not result7_fatal
    )

    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_pass else '✗ SOME TESTS FAILED'}")
    print("=" * 120)


if __name__ == "__main__":
    test_unbounded_patterns()
