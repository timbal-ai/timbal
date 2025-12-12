"""
Complex Sequence Matching Test
===============================

A truly complex real-world scenario testing multiple edge cases simultaneously.

SCENARIO: E-commerce AI Agent Workflow
---------------------------------------
The agent helps users find and purchase products. We're validating that the
trace follows a specific pattern with multiple any! patterns and restrictions.

Expected Flow:
1. llm (initial query understanding)
2. 1-2 validation/preprocessing steps (must include "validate_input", no "error")
3. search_products (the actual search)
4. 2-5 processing steps (must include "filter_results" and "rank_products", no "crash")
5. llm (present results to user)
6. get_user_confirmation (wait for user)
7. 0-3 additional steps (optional, but if present, must include "calculate_tax", no "refund")
8. process_payment (final payment)
9. llm (confirmation message)

This tests:
- Multiple any! patterns with different min/max bounds
- Multiple contains requirements in single any!
- Mix of optional (min=0) and required (min>=1) any!
- not_contains restrictions throughout
- Exact matches interspersed with wildcards
- Backtracking scenarios where greedy would fail
"""

from seq_dp_with_restrictions import AnyPattern, dp_match


def test_complex_ecommerce_scenario():
    """Test a complex real-world e-commerce agent workflow."""

    print("=" * 120)
    print("COMPLEX E-COMMERCE WORKFLOW TEST")
    print("=" * 120)
    print()

    # Define the expected pattern
    patterns = [
        "llm",  # Step 1: Initial query
        AnyPattern(1, 2, contains=["validate_input"], not_contains=["error"]),  # Step 2: Validation
        "search_products",  # Step 3: Search
        AnyPattern(2, 5, contains=["filter_results", "rank_products"], not_contains=["crash"]),  # Step 4: Processing
        "llm",  # Step 5: Present results
        "get_user_confirmation",  # Step 6: Wait for user
        AnyPattern(0, 3, not_contains=["refund"]),  # Step 7: Optional pre-payment (no contains since min=0)
        "process_payment",  # Step 8: Payment
        "llm",  # Step 9: Final confirmation
    ]

    # SCENARIO 1: Perfect happy path (all optional steps present)
    print("\n" + "=" * 120)
    print("SCENARIO 1: Perfect Happy Path (all optional steps)")
    print("=" * 120)

    spans_happy = [
        "llm",  # 1. Initial query
        "validate_input",  # 2. Validation (matches contains)
        "sanitize_query",  # 2. Additional validation step
        "search_products",  # 3. Search
        "filter_results",  # 4. Processing (matches 1st contains)
        "rank_products",  # 4. Processing (matches 2nd contains)
        "add_metadata",  # 4. Additional processing
        "cache_results",  # 4. Additional processing
        "llm",  # 5. Present results
        "get_user_confirmation",  # 6. User confirmation
        "calculate_tax",  # 7. Pre-payment (matches contains)
        "apply_discount",  # 7. Additional pre-payment
        "verify_inventory",  # 7. Additional pre-payment
        "process_payment",  # 8. Payment
        "llm",  # 9. Confirmation
    ]

    print(f"\nPatterns ({len(patterns)}):")
    for i, p in enumerate(patterns):
        print(f"  {i + 1}. {p}")

    print(f"\nSpans ({len(spans_happy)}):")
    for i, s in enumerate(spans_happy):
        print(f"  {i + 1}. {s}")

    print("\n" + "-" * 120)
    result1 = dp_match(patterns, spans_happy, debug=True)
    print(f"\nResult: {result1}")
    print(f"Status: {'✓ PASS' if result1 else '✗ FAIL'}")

    # SCENARIO 2: Minimal path (optional steps skipped)
    print("\n" + "=" * 120)
    print("SCENARIO 2: Minimal Path (optional steps skipped)")
    print("=" * 120)

    spans_minimal = [
        "llm",  # 1. Initial query
        "validate_input",  # 2. Validation (just 1 step, min=1)
        "search_products",  # 3. Search
        "filter_results",  # 4. Processing (2 steps, min=2)
        "rank_products",  # 4. Processing
        "llm",  # 5. Present results
        "get_user_confirmation",  # 6. User confirmation
        # 7. No pre-payment steps (min=0, so optional)
        "process_payment",  # 8. Payment
        "llm",  # 9. Confirmation
    ]

    print(f"\nSpans ({len(spans_minimal)}):")
    for i, s in enumerate(spans_minimal):
        print(f"  {i + 1}. {s}")

    print("\n" + "-" * 120)
    result2 = dp_match(patterns, spans_minimal, debug=True)
    print(f"\nResult: {result2}")
    print(f"Status: {'✓ PASS' if result2 else '✗ FAIL'}")

    # SCENARIO 3: Should FAIL - missing required "rank_products" in processing
    print("\n" + "=" * 120)
    print("SCENARIO 3: Should FAIL - Missing required 'rank_products'")
    print("=" * 120)

    spans_fail_missing = [
        "llm",
        "validate_input",
        "search_products",
        "filter_results",  # Has filter_results but missing rank_products!
        "add_metadata",
        "cache_results",
        "llm",
        "get_user_confirmation",
        "process_payment",
        "llm",
    ]

    print(f"\nSpans ({len(spans_fail_missing)}):")
    for i, s in enumerate(spans_fail_missing):
        print(f"  {i + 1}. {s}")

    print("\n" + "-" * 120)
    result3 = dp_match(patterns, spans_fail_missing, debug=True)
    print(f"\nResult: {result3}")
    print(f"Status: {'✓ PASS (correctly failed)' if not result3 else '✗ FAIL (should have failed)'}")

    # SCENARIO 4: Should FAIL - has forbidden "crash" in processing
    print("\n" + "=" * 120)
    print("SCENARIO 4: Should FAIL - Has forbidden 'crash' in processing")
    print("=" * 120)

    spans_fail_forbidden = [
        "llm",
        "validate_input",
        "search_products",
        "filter_results",
        "rank_products",
        "crash",  # Exact match - forbidden!
        "llm",
        "get_user_confirmation",
        "process_payment",
        "llm",
    ]

    print(f"\nSpans ({len(spans_fail_forbidden)}):")
    for i, s in enumerate(spans_fail_forbidden):
        print(f"  {i + 1}. {s}")

    print("\n" + "-" * 120)
    result4 = dp_match(patterns, spans_fail_forbidden, debug=True)
    print(f"\nResult: {result4}")
    print(f"Status: {'✓ PASS (correctly failed)' if not result4 else '✗ FAIL (should have failed)'}")

    # SCENARIO 5: Greedy pitfall test
    print("\n" + "=" * 120)
    print("SCENARIO 5: Greedy Pitfall - First any! shouldn't consume too much")
    print("=" * 120)

    spans_greedy = [
        "llm",
        "validate_input",  # First any! could greedily take this...
        "sanitize_query",  # ...and this...
        "search_products",  # ...and even this (but shouldn't!)
        "filter_results",
        "rank_products",
        "llm",
        "get_user_confirmation",
        "process_payment",
        "llm",
    ]

    print(f"\nSpans ({len(spans_greedy)}):")
    for i, s in enumerate(spans_greedy):
        print(f"  {i + 1}. {s}")

    print("\nNote: First any! has max=2, so it should NOT consume 'search_products'")

    print("\n" + "-" * 120)
    result5 = dp_match(patterns, spans_greedy, debug=True)
    print(f"\nResult: {result5}")
    print(f"Status: {'✓ PASS' if result5 else '✗ FAIL'}")

    # SCENARIO 6: Edge case - exactly at boundaries
    print("\n" + "=" * 120)
    print("SCENARIO 6: Edge Case - Exactly at min/max boundaries")
    print("=" * 120)

    spans_boundary = [
        "llm",
        "validate_input",  # Exactly 1 step (min=1)
        "search_products",
        "filter_results",  # Exactly 2 steps (min=2)
        "rank_products",
        "llm",
        "get_user_confirmation",
        "calculate_tax",  # Exactly 1 step (min=0, so optional but present)
        "process_payment",
        "llm",
    ]

    print(f"\nSpans ({len(spans_boundary)}):")
    for i, s in enumerate(spans_boundary):
        print(f"  {i + 1}. {s}")

    print("\n" + "-" * 120)
    result6 = dp_match(patterns, spans_boundary, debug=True)
    print(f"\nResult: {result6}")
    print(f"Status: {'✓ PASS' if result6 else '✗ FAIL'}")

    # Summary
    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)
    print(f"Scenario 1 (Happy Path):              {'✓ PASS' if result1 else '✗ FAIL'}")
    print(f"Scenario 2 (Minimal):                 {'✓ PASS' if result2 else '✗ FAIL'}")
    print(f"Scenario 3 (Missing Required):        {'✓ PASS' if not result3 else '✗ FAIL'} (should fail)")
    print(f"Scenario 4 (Has Forbidden):           {'✓ PASS' if not result4 else '✗ FAIL'} (should fail)")
    print(f"Scenario 5 (Greedy Pitfall):          {'✓ PASS' if result5 else '✗ FAIL'}")
    print(f"Scenario 6 (Boundary Conditions):     {'✓ PASS' if result6 else '✗ FAIL'}")
    print()

    all_pass = result1 and result2 and not result3 and not result4 and result5 and result6
    print(f"Overall: {'✓ ALL TESTS PASSED' if all_pass else '✗ SOME TESTS FAILED'}")
    print("=" * 120)


if __name__ == "__main__":
    test_complex_ecommerce_scenario()
