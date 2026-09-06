"""Recovering tool calls GPT-5.x leaks into assistant text (Harmony wire form)."""
from timbal.collectors.harmony_leak import contains_leak, leak_state, parse_leaked_tool_calls

# Verbatim samples from a gpt-5.6-luna orchestration run (composer evals, 2026-09-06).
SAMPLE_ONE = ' to=functions.timbal__get_preview_logs  (json 恒一\n{"service":"ui"}ંалда'
SAMPLE_REPEATED = (
    ' to=functions.wait_for_background_tasks  code:\n{"task_ids":["o20s53ln5l51"],"timeout_seconds":120}\n\n'
    ' to=functions.wait_for_background_tasks  code:\n{"task_ids":["o20s53ln5l51"],"timeout_seconds":120}\n\n'
    ' to=functions.list_background_tasks code:\n{}\n\n'
    ' to=functions.wait_for_background_tasks code:\n{"task_ids":["o20s53ln5l51"],"timeout_seconds":30}\n\n'
    "Invoice Triage backend is built and verified:\n\n- Knowledge base: `vendors` has 5 rows"
)
SAMPLE_AFTER_PROSE = (
    "Wave 3 dependency check:\n- `UI`: depends on [API routes probed] → ready.\n\n\n\n"
    ' to=functions.builder  code:\n{"title":"UI","resume_previous":false,"prompt":"## Objective\\nBuild {the} UI"}'
)


class TestLeakState:
    def test_empty_and_whitespace_are_undecided(self):
        assert leak_state("") == "undecided"
        assert leak_state("  \n") == "undecided"

    def test_prefix_of_marker_is_undecided(self):
        for head in ("t", "to", "to=", "to=fun", " to=functions"):
            assert leak_state(head) == "undecided", head

    def test_marker_is_a_leak(self):
        assert leak_state(" to=functions.") == "leak"
        assert leak_state(SAMPLE_ONE) == "leak"

    def test_divergence_is_clean(self):
        assert leak_state("The") == "clean"
        assert leak_state("tomorrow") == "clean"
        assert leak_state("to be clear") == "clean"
        # a marker after prose is not decidable while streaming
        assert leak_state("Wave 3 dependency check: to=functions.x") == "clean"


class TestParse:
    def test_single_call_with_control_token_debris(self):
        prefix, calls = parse_leaked_tool_calls(SAMPLE_ONE)
        assert prefix == ""
        assert calls == [("timbal__get_preview_logs", {"service": "ui"})]

    def test_repeated_calls_dedupe_and_trailing_summary_is_dropped(self):
        prefix, calls = parse_leaked_tool_calls(SAMPLE_REPEATED)
        assert prefix == ""
        assert calls == [
            ("wait_for_background_tasks", {"task_ids": ["o20s53ln5l51"], "timeout_seconds": 120}),
            ("list_background_tasks", {}),
            ("wait_for_background_tasks", {"task_ids": ["o20s53ln5l51"], "timeout_seconds": 30}),
        ]

    def test_prose_before_the_leak_is_kept(self):
        prefix, calls = parse_leaked_tool_calls(SAMPLE_AFTER_PROSE)
        assert prefix == "Wave 3 dependency check:\n- `UI`: depends on [API routes probed] → ready."
        assert calls == [("builder", {"title": "UI", "resume_previous": False, "prompt": "## Objective\nBuild {the} UI"})]

    def test_braces_inside_strings_do_not_end_the_object(self):
        text = ' to=functions.run json\n{"code":"if (x) { return {a:1}; }","n":2}'
        assert parse_leaked_tool_calls(text)[1] == [("run", {"code": "if (x) { return {a:1}; }", "n": 2})]

    def test_header_without_object_is_skipped(self):
        text = " to=functions.wait_for_background_tasks  code:\n to=functions.list_background_tasks code:\n{}"
        assert parse_leaked_tool_calls(text)[1] == [("list_background_tasks", {})]

    def test_unparseable_object_is_skipped(self):
        text = ' to=functions.x json\n{"a": tru}'
        assert parse_leaked_tool_calls(text) == ("", [])

    def test_non_object_json_is_skipped(self):
        assert parse_leaked_tool_calls(" to=functions.x json\n[1,2]")[1] == []

    def test_plain_text_passes_through(self):
        assert parse_leaked_tool_calls("All done.") == ("All done.", [])
        assert not contains_leak("All done.")
        assert contains_leak(SAMPLE_ONE)

    def test_mentioning_functions_in_prose_is_not_a_leak(self):
        text = "The `functions.search` helper is documented in README."
        assert not contains_leak(text)
        assert parse_leaked_tool_calls(text) == (text, [])
