"""Tests for memory compaction strategies."""

import pytest
from timbal.core.memory_compaction import (
    _SUMMARY_MARKER,
    _format_message_for_summary,
    compact_tool_results,
    keep_last_n_messages,
    keep_last_n_turns,
    summarize,
)
from timbal.core.test_model import TestModel
from timbal.types.content import TextContent, ToolResultContent, ToolUseContent
from timbal.types.message import Message


def _msg_user(text: str) -> Message:
    return Message(role="user", content=[TextContent(text=text)])


def _msg_system(text: str) -> Message:
    return Message(role="system", content=[TextContent(text=text)])


def _msg_assistant_text(text: str) -> Message:
    return Message(role="assistant", content=[TextContent(text=text)])


def _msg_assistant_tool(uid: str, name: str = "test_tool") -> Message:
    return Message(
        role="assistant",
        content=[ToolUseContent(id=uid, name=name, input={})],
    )


def _msg_assistant_mixed(text: str, uid: str, name: str = "test_tool") -> Message:
    return Message(
        role="assistant",
        content=[
            TextContent(text=text),
            ToolUseContent(id=uid, name=name, input={}),
        ],
    )


def _msg_tool(uid: str, result_text: str = "ok") -> Message:
    return Message(
        role="tool",
        content=[ToolResultContent(id=uid, content=[TextContent(text=result_text)])],
    )


# ---------------------------------------------------------------------------
# compact_tool_results — drop mode (replacement=None)
# ---------------------------------------------------------------------------


class TestCompactToolResultsDrop:
    """Tests for compact_tool_results with replacement=None (drop mode)."""

    def test_empty_memory(self) -> None:
        compactor = compact_tool_results()
        assert compactor([]) == []

    def test_only_user_messages(self) -> None:
        memory = [_msg_user("Hi"), _msg_user("Bye")]
        compactor = compact_tool_results()
        assert compactor(memory) == memory

    def test_user_and_assistant_text_only(self) -> None:
        memory = [_msg_user("Hi"), _msg_assistant_text("Hello!")]
        compactor = compact_tool_results()
        result = compactor(memory)
        assert len(result) == 2
        assert result[0].role == "user" and result[0].content[0].text == "Hi"
        assert result[1].role == "assistant" and result[1].content[0].text == "Hello!"

    def test_drop_all_when_keep_last_n_none(self) -> None:
        """When keep_last_n is None, all tool use and results are dropped."""
        memory = [
            _msg_user("hi"),
            _msg_assistant_tool("t1"),
            _msg_tool("t1", "result1"),
            _msg_assistant_text("here is the answer"),
        ]
        compactor = compact_tool_results(threshold=0, keep_last_n=None)
        result = compactor(memory)
        assert len(result) == 2
        assert result[0].role == "user"
        assert result[1].role == "assistant"
        assert result[1].content[0].text == "here is the answer"
        assert not any(m.role == "tool" for m in result)

    def test_removes_tool_messages(self) -> None:
        memory = [
            _msg_user("Get weather"),
            _msg_assistant_tool("c1", "get_weather"),
            _msg_tool("c1", '{"temp": 72}'),
            _msg_assistant_text("It's 72°F."),
        ]
        compactor = compact_tool_results()
        result = compactor(memory)
        assert len(result) == 2
        assert result[0].role == "user"
        assert result[1].role == "assistant"
        assert result[1].content[0].text == "It's 72°F."

    def test_strips_tool_use_keeps_text(self) -> None:
        memory = [
            _msg_user("Hi"),
            _msg_assistant_mixed("Let me check.", "c1", "lookup"),
            _msg_tool("c1", "result"),
            _msg_assistant_text("Done."),
        ]
        compactor = compact_tool_results()
        result = compactor(memory)
        assert len(result) == 3
        assert result[1].role == "assistant"
        assert len(result[1].content) == 1
        assert result[1].content[0].text == "Let me check."

    def test_keep_last_n_one(self) -> None:
        """When keep_last_n=1, only the last tool use/result pair is kept."""
        memory = [
            _msg_user("first"),
            _msg_assistant_tool("t1"),
            _msg_tool("t1", "r1"),
            _msg_user("second"),
            _msg_assistant_tool("t2"),
            _msg_tool("t2", "r2"),
            _msg_assistant_text("done"),
        ]
        compactor = compact_tool_results(threshold=0, keep_last_n=1)
        result = compactor(memory)
        tool_msgs = [m for m in result if m.role == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].content[0].id == "t2"
        assistant_with_tool = [
            m for m in result if m.role == "assistant" and any(isinstance(c, ToolUseContent) for c in m.content)
        ]
        assert len(assistant_with_tool) == 1
        tool_use = next(c for c in assistant_with_tool[0].content if isinstance(c, ToolUseContent))
        assert tool_use.id == "t2"

    def test_keep_last_n_two(self) -> None:
        """When keep_last_n=2, last two tool pairs are kept."""
        memory = [
            _msg_assistant_tool("t1"),
            _msg_tool("t1", "r1"),
            _msg_assistant_tool("t2"),
            _msg_tool("t2", "r2"),
            _msg_assistant_tool("t3"),
            _msg_tool("t3", "r3"),
        ]
        compactor = compact_tool_results(threshold=0, keep_last_n=2)
        result = compactor(memory)
        kept_ids = set()
        for m in result:
            for c in m.content:
                if isinstance(c, (ToolUseContent, ToolResultContent)):
                    kept_ids.add(c.id)
        assert kept_ids == {"t2", "t3"}

    def test_threshold_short_conversation_unchanged(self) -> None:
        memory = [_msg_user("hi"), _msg_assistant_text("hello")]
        compactor = compact_tool_results(threshold=10, keep_last_n=None)
        result = compactor(memory)
        assert result == memory

    def test_threshold_below_returns_unchanged(self) -> None:
        memory = [
            _msg_user("Hi"),
            _msg_assistant_tool("c1"),
            _msg_tool("c1", "r"),
            _msg_assistant_text("Done."),
        ]
        compactor = compact_tool_results(threshold=5)
        result = compactor(memory)
        assert len(result) == 4
        assert result[2].role == "tool"

    def test_threshold_exceeded_applies_compaction(self) -> None:
        memory = [
            _msg_user("Hi"),
            _msg_assistant_tool("c1"),
            _msg_tool("c1", "r"),
            _msg_assistant_text("Done."),
        ]
        compactor = compact_tool_results(threshold=3)
        result = compactor(memory)
        assert len(result) == 2
        assert not any(m.role == "tool" for m in result)


# ---------------------------------------------------------------------------
# compact_tool_results — replacement with template string
# ---------------------------------------------------------------------------


class TestCompactToolResultsReplace:
    """Tests for compact_tool_results with a string replacement template."""

    def test_no_tool_messages(self) -> None:
        memory = [_msg_user("Hi"), _msg_assistant_text("Hello")]
        compactor = compact_tool_results(replacement="[Tool result: {tool_name}]")
        result = compactor(memory)
        assert result == memory

    def test_replaces_with_default_template(self) -> None:
        memory = [
            _msg_assistant_tool("t1", "get_weather"),
            _msg_tool("t1", "temperature is 72F"),
        ]
        compactor = compact_tool_results(replacement="[Tool result: {tool_name}]")
        result = compactor(memory)
        tool_msg = result[1]
        assert tool_msg.role == "tool"
        assert tool_msg.content[0].content[0].text == "[Tool result: get_weather]"

    def test_custom_template(self) -> None:
        memory = [
            _msg_assistant_tool("call-123", "search"),
            _msg_tool("call-123", "x" * 100),
        ]
        compactor = compact_tool_results(replacement="{tool_name} (id={call_id}, len={result_length})")
        result = compactor(memory)
        text = result[1].content[0].content[0].text
        assert "search" in text
        assert "call-123" in text
        assert "100" in text

    def test_unknown_placeholder_replaced_with_empty(self) -> None:
        memory = [
            _msg_assistant_tool("t1", "foo"),
            _msg_tool("t1", "bar"),
        ]
        compactor = compact_tool_results(replacement="{tool_name}-{unknown_placeholder}")
        result = compactor(memory)
        text = result[1].content[0].content[0].text
        assert text == "foo-"

    def test_keep_last_n_preserves_recent(self) -> None:
        """When keep_last_n=1, the last tool result is kept intact."""
        memory = [
            _msg_assistant_tool("t1", "search"),
            _msg_tool("t1", "first result data"),
            _msg_assistant_tool("t2", "search"),
            _msg_tool("t2", "second result data"),
            _msg_assistant_tool("t3", "search"),
            _msg_tool("t3", "third result data"),
        ]
        compactor = compact_tool_results(keep_last_n=1, replacement="[Tool result: {tool_name}]")
        result = compactor(memory)
        # First two tool results should be placeholders
        assert result[1].content[0].content[0].text == "[Tool result: search]"
        assert result[3].content[0].content[0].text == "[Tool result: search]"
        # Last tool result should be preserved
        assert result[5].content[0].content[0].text == "third result data"

    def test_keep_last_n_none_replaces_all(self) -> None:
        """Default keep_last_n=None replaces all tool results."""
        memory = [
            _msg_assistant_tool("t1", "search"),
            _msg_tool("t1", "data"),
        ]
        compactor = compact_tool_results(replacement="[Tool result: {tool_name}]")
        result = compactor(memory)
        assert result[1].content[0].content[0].text == "[Tool result: search]"

    def test_preserves_tool_use_in_assistant_messages(self) -> None:
        """In replace mode, tool_use content in assistant messages is preserved."""
        memory = [
            _msg_user("Hi"),
            _msg_assistant_mixed("Let me check.", "c1", "lookup"),
            _msg_tool("c1", "result"),
            _msg_assistant_text("Done."),
        ]
        compactor = compact_tool_results(replacement="[Tool result: {tool_name}]")
        result = compactor(memory)
        assert len(result) == 4
        # Assistant message with tool_use should be unchanged
        assert len(result[1].content) == 2
        assert isinstance(result[1].content[1], ToolUseContent)
        # Tool result should be replaced
        assert result[2].content[0].content[0].text == "[Tool result: lookup]"


# ---------------------------------------------------------------------------
# compact_tool_results — replacement with callable
# ---------------------------------------------------------------------------


class TestCompactToolResultsCallable:
    """Tests for compact_tool_results with a callable replacement."""

    def test_callable_receives_correct_args(self) -> None:
        captured = []

        def my_replacer(tool_name: str, call_id: str, result_text: str) -> str:
            captured.append((tool_name, call_id, result_text))
            return f"summary of {tool_name}"

        memory = [
            _msg_assistant_tool("c1", "get_weather"),
            _msg_tool("c1", "temperature is 72F"),
        ]
        compactor = compact_tool_results(replacement=my_replacer)
        result = compactor(memory)

        assert len(captured) == 1
        assert captured[0] == ("get_weather", "c1", "temperature is 72F")
        assert result[1].content[0].content[0].text == "summary of get_weather"

    def test_callable_with_keep_last_n(self) -> None:
        call_count = 0

        def my_replacer(tool_name: str, _call_id: str, _result_text: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"[{tool_name}]"

        memory = [
            _msg_assistant_tool("t1", "search"),
            _msg_tool("t1", "first"),
            _msg_assistant_tool("t2", "search"),
            _msg_tool("t2", "second"),
        ]
        compactor = compact_tool_results(keep_last_n=1, replacement=my_replacer)
        result = compactor(memory)

        # Only the first tool result should be replaced
        assert call_count == 1
        assert result[1].content[0].content[0].text == "[search]"
        # Second tool result kept intact
        assert result[3].content[0].content[0].text == "second"


# ---------------------------------------------------------------------------
# keep_last_n_messages
# ---------------------------------------------------------------------------


class TestKeepLastNMessages:
    """Tests for keep_last_n_messages."""

    def test_empty_memory(self) -> None:
        compactor = keep_last_n_messages(5)
        assert compactor([]) == []

    def test_fewer_than_n(self) -> None:
        memory = [_msg_user("A"), _msg_user("B")]
        compactor = keep_last_n_messages(5)
        assert len(compactor(memory)) == 2

    def test_exactly_n(self) -> None:
        memory = [_msg_user("A"), _msg_user("B"), _msg_user("C")]
        compactor = keep_last_n_messages(3)
        assert len(compactor(memory)) == 3

    def test_keeps_last_n(self) -> None:
        memory = [
            _msg_user("1"),
            _msg_assistant_text("2"),
            _msg_user("3"),
            _msg_assistant_text("4"),
        ]
        compactor = keep_last_n_messages(2)
        result = compactor(memory)
        assert len(result) == 2
        assert result[0].content[0].text == "3"
        assert result[1].content[0].text == "4"

    def test_removes_orphaned_tool_parts(self) -> None:
        """Truncation that would cut a tool call removes the orphaned part."""
        memory = [
            _msg_user("start"),
            _msg_assistant_tool("t1"),
            _msg_tool("t1", "r1"),
            _msg_assistant_text("end"),
        ]
        compactor = keep_last_n_messages(2)
        result = compactor(memory)
        assert len(result) == 1
        assert result[0].role == "assistant"
        assert result[0].content[0].text == "end"

    def test_removes_orphaned_tool_result_when_truncating(self) -> None:
        """When we cut before tool_use, the tool_result becomes orphaned and is removed."""
        memory = [
            _msg_user("1"),
            _msg_user("2"),
            _msg_assistant_tool("c1"),
            _msg_tool("c1", "r"),
        ]
        compactor = keep_last_n_messages(2)
        result = compactor(memory)
        assert len(result) == 2

    def test_removes_orphaned_tool_use_when_truncating(self) -> None:
        """When we cut before tool_result, the tool_use becomes orphaned and is removed."""
        memory = [
            _msg_assistant_tool("c1"),
            _msg_tool("c1", "r"),
            _msg_user("3"),
            _msg_user("4"),
        ]
        compactor = keep_last_n_messages(2)
        result = compactor(memory)
        assert len(result) == 2
        assert result[0].content[0].text == "3"
        assert result[1].content[0].text == "4"


# ---------------------------------------------------------------------------
# keep_last_n_turns
# ---------------------------------------------------------------------------


class TestKeepLastNTurns:
    """Tests for keep_last_n_turns."""

    def test_empty_memory(self) -> None:
        compactor = keep_last_n_turns(2)
        assert compactor([]) == []

    def test_single_turn(self) -> None:
        memory = [_msg_user("Hi"), _msg_assistant_text("Hello")]
        compactor = keep_last_n_turns(1)
        result = compactor(memory)
        assert len(result) == 2

    def test_keeps_last_n_turns(self) -> None:
        memory = [
            _msg_user("turn1"),
            _msg_assistant_text("reply1"),
            _msg_user("turn2"),
            _msg_assistant_text("reply2"),
            _msg_user("turn3"),
            _msg_assistant_text("reply3"),
        ]
        compactor = keep_last_n_turns(2)
        result = compactor(memory)
        assert len(result) == 4
        assert result[0].content[0].text == "turn2"
        assert result[3].content[0].text == "reply3"

    def test_turn_with_tool_calls(self) -> None:
        memory = [
            _msg_user("1"),
            _msg_assistant_tool("c1"),
            _msg_tool("c1", "r"),
            _msg_assistant_text("done"),
            _msg_user("2"),
            _msg_assistant_text("ok"),
        ]
        compactor = keep_last_n_turns(1)
        result = compactor(memory)
        assert len(result) == 2
        assert result[0].content[0].text == "2"
        assert result[1].content[0].text == "ok"

    def test_no_user_messages_returns_all(self) -> None:
        memory = [_msg_assistant_text("only assistant")]
        compactor = keep_last_n_turns(2)
        result = compactor(memory)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------


class TestSummarizeOldMessages:
    """Tests for summarize."""

    @pytest.mark.asyncio
    async def test_below_threshold_returns_unchanged(self) -> None:
        memory = [_msg_user("Hi"), _msg_assistant_text("Hello")]
        compactor = summarize(threshold=10, keep_last_n=2)
        result = await compactor(memory)
        assert result == memory

    @pytest.mark.asyncio
    async def test_at_threshold_returns_unchanged(self) -> None:
        memory = [_msg_user(f"msg{i}") for i in range(5)]
        compactor = summarize(threshold=5, keep_last_n=2)
        result = await compactor(memory)
        assert result == memory

    @pytest.mark.asyncio
    async def test_above_threshold_summarizes(self) -> None:
        """When above threshold, older messages are summarized."""
        memory = [_msg_user(f"Message {i}") for i in range(20)]
        compactor = summarize(threshold=5, keep_last_n=2, model=TestModel(responses=["The user discussed topics 0 through 17."]))
        result = await compactor(memory)

        assert len(result) < 20
        # First message should be the summary with marker
        assert result[0].role == "user"
        assert result[0].content[0].text.startswith(_SUMMARY_MARKER)
        assert "topics 0 through 17" in result[0].content[0].text
        # Last 2 messages preserved
        assert result[-1].content[0].text == "Message 19"
        assert result[-2].content[0].text == "Message 18"

    @pytest.mark.asyncio
    async def test_incremental_uses_previous_summary(self) -> None:
        """When a previous summary exists, the prompt uses incremental format."""
        captured_prompts = []

        def capture_handler(messages):
            captured_prompts.append(messages[-1].collect_text())
            return "Updated summary with new info."

        # First message is a previous summary
        summary_msg = Message.validate(
            {"role": "user", "content": f"{_SUMMARY_MARKER}\nPrevious summary content here."}
        )
        memory = [
            summary_msg,
            *[_msg_user(f"New message {i}") for i in range(10)],
        ]
        compactor = summarize(threshold=5, keep_last_n=2, model=TestModel(handler=capture_handler))
        result = await compactor(memory)

        # The prompt should contain the previous summary
        assert len(captured_prompts) == 1
        assert "Previous summary content here." in captured_prompts[0]
        assert "Current summary:" in captured_prompts[0]
        # Result should have the updated summary
        assert result[0].content[0].text.startswith(_SUMMARY_MARKER)
        assert "Updated summary" in result[0].content[0].text

    @pytest.mark.asyncio
    async def test_system_messages_preserved(self) -> None:
        """System messages are never summarized and always preserved."""

        def assert_handler(messages):
            prompt_text = messages[-1].collect_text()
            assert "system instruction" not in prompt_text.lower()
            return "Summary of user conversation."

        memory = [
            _msg_system("You are a helpful assistant"),
            *[_msg_user(f"Message {i}") for i in range(10)],
        ]
        compactor = summarize(threshold=5, keep_last_n=2, model=TestModel(handler=assert_handler))
        result = await compactor(memory)

        # System message should be first, unchanged
        system_msgs = [m for m in result if m.role == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0].content[0].text == "You are a helpful assistant"

    @pytest.mark.asyncio
    async def test_no_model_returns_memory_unchanged(self) -> None:
        """When model=None and no agent model injected, return memory as-is."""
        memory = [_msg_user(f"msg{i}") for i in range(10)]
        compactor = summarize(threshold=2, model=None, keep_last_n=2)
        # _state["agent_model"] is None by default — cannot summarize
        result = await compactor(memory)
        assert result == memory

    @pytest.mark.asyncio
    async def test_agent_model_injection(self) -> None:
        """Agent can inject its model via _compact._state['agent_model']."""
        test_model = TestModel(responses=["Summary."])
        memory = [_msg_user(f"msg{i}") for i in range(10)]
        compactor = summarize(threshold=2, model=None, keep_last_n=2)
        compactor._state["agent_model"] = test_model
        await compactor(memory)

        assert test_model.call_count == 1

    @pytest.mark.asyncio
    async def test_keep_last_n_zero_summarizes_all(self) -> None:
        """keep_last_n=0 means no messages are preserved unsummarized."""
        test_model = TestModel(responses=["Full summary."])
        memory = [_msg_user(f"msg{i}") for i in range(6)]
        compactor = summarize(threshold=3, keep_last_n=0, model=test_model)
        result = await compactor(memory)

        assert test_model.call_count == 1
        # Only summary message — no unsummarized messages kept
        assert len(result) == 1
        assert result[0].content[0].text.startswith(_SUMMARY_MARKER)

    @pytest.mark.asyncio
    async def test_nothing_to_summarize_returns_unchanged(self) -> None:
        """When keep_last_n covers all non-summary messages, return unchanged."""
        test_model = TestModel(responses=["Should not be called."])
        # 4 messages, threshold=3 (triggers), but keep_last_n=4 covers everything
        memory = [_msg_user(f"msg{i}") for i in range(4)]
        compactor = summarize(threshold=3, keep_last_n=4, model=test_model)
        result = await compactor(memory)

        assert test_model.call_count == 0
        assert result == memory

    @pytest.mark.asyncio
    async def test_previous_summary_nothing_new_returns_unchanged(self) -> None:
        """Existing summary + keep_last_n covers all remaining messages — no new call."""
        test_model = TestModel(responses=["Should not be called."])
        summary_msg = Message.validate(
            {"role": "user", "content": f"{_SUMMARY_MARKER}\nExisting summary."}
        )
        # summary + 2 new messages; keep_last_n=2 covers exactly those 2
        memory = [summary_msg, _msg_user("a"), _msg_user("b")]
        compactor = summarize(threshold=2, keep_last_n=2, model=test_model)
        result = await compactor(memory)

        assert test_model.call_count == 0
        assert result == memory

    @pytest.mark.asyncio
    async def test_tool_calls_formatted_in_prompt(self) -> None:
        """Tool call structure is preserved in the summarization prompt."""
        captured_prompts = []

        def capture_handler(messages):
            captured_prompts.append(messages[-1].collect_text())
            return "User called get_weather, got temperature 72."

        memory = [
            _msg_user("What's the weather?"),
            _msg_assistant_tool("c1", "get_weather"),
            _msg_tool("c1", '{"temperature": 72}'),
            _msg_assistant_text("It's 72 degrees."),
            *[_msg_user(f"Message {i}") for i in range(10)],
        ]
        compactor = summarize(threshold=5, keep_last_n=2, model=TestModel(handler=capture_handler))
        await compactor(memory)

        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]
        assert "get_weather" in prompt
        assert "temperature" in prompt

    # ------------------------------------------------------------------
    # Strict alternation tests (ack insertion)
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_no_ack_when_to_keep_empty(self) -> None:
        """keep_last_n=0: to_keep is empty, no alternation concern, no ack inserted."""
        memory = [_msg_user(f"msg{i}") for i in range(6)]
        compactor = summarize(threshold=3, keep_last_n=0, model=TestModel(responses=["Summary."]))
        result = await compactor(memory)

        assert len(result) == 1
        assert result[0].role == "user"
        assert result[0].content[0].text.startswith(_SUMMARY_MARKER)

    @pytest.mark.asyncio
    async def test_no_ack_when_to_keep_starts_with_assistant(self) -> None:
        """to_keep[0] is assistant: summary(user) → assistant is valid alternation, no ack inserted."""
        # 6 alternating messages ending with assistant; keep_last_n=1 → to_keep=[a3]
        memory = [
            _msg_user("u1"), _msg_assistant_text("a1"),
            _msg_user("u2"), _msg_assistant_text("a2"),
            _msg_user("u3"), _msg_assistant_text("a3"),
        ]
        compactor = summarize(threshold=3, keep_last_n=1, model=TestModel(responses=["Summary."]))
        result = await compactor(memory)

        assert result[0].role == "user"       # summary
        assert result[1].role == "assistant"  # to_keep[0] — no ack in between
        # Verify no "Understood." message inserted
        roles = [m.role for m in result]
        assert roles == ["user", "assistant"]

    @pytest.mark.asyncio
    async def test_ack_inserted_when_to_keep_starts_with_user(self) -> None:
        """to_keep[0] is user: summary(user) + user would be consecutive — ack inserted."""
        # 5 alternating messages ending with user; keep_last_n=1 → to_keep=[u3(current)]
        memory = [
            _msg_user("u1"), _msg_assistant_text("a1"),
            _msg_user("u2"), _msg_assistant_text("a2"),
            _msg_user("u3"),
        ]
        compactor = summarize(threshold=3, keep_last_n=1, model=TestModel(responses=["Summary."]))
        result = await compactor(memory)

        assert result[0].role == "user"       # summary
        assert result[1].role == "assistant"  # ack
        assert result[2].role == "user"       # to_keep[0]
        # No consecutive same-role messages anywhere
        for prev, cur in zip(result, result[1:]):
            assert prev.role != cur.role, f"Consecutive {prev.role!r} messages"

    @pytest.mark.asyncio
    async def test_ack_inserted_odd_keep_last_n(self) -> None:
        """Any odd keep_last_n over a clean alternating history ending with user triggers the ack."""
        # 9 messages: u1 a1 u2 a2 u3 a3 u4 a4 u5
        memory = []
        for i in range(1, 6):
            memory.append(_msg_user(f"u{i}"))
            if i < 5:
                memory.append(_msg_assistant_text(f"a{i}"))

        for keep_last_n in (1, 3):
            compactor = summarize(threshold=3, keep_last_n=keep_last_n, model=TestModel(responses=["Summary."]))
            result = await compactor(memory)
            # to_keep[0] is user for odd keep_last_n on history ending with user
            assert result[1].role == "assistant", f"Expected ack for keep_last_n={keep_last_n}"
            for prev, cur in zip(result, result[1:]):
                assert prev.role != cur.role, f"Consecutive {prev.role!r} for keep_last_n={keep_last_n}"

    @pytest.mark.asyncio
    async def test_no_ack_even_keep_last_n(self) -> None:
        """Even keep_last_n on a clean alternating history ending with user: to_keep[0] is
        assistant, so no ack is needed."""
        # 9 messages: u1 a1 u2 a2 u3 a3 u4 a4 u5
        memory = []
        for i in range(1, 6):
            memory.append(_msg_user(f"u{i}"))
            if i < 5:
                memory.append(_msg_assistant_text(f"a{i}"))

        for keep_last_n in (2, 4):
            compactor = summarize(threshold=3, keep_last_n=keep_last_n, model=TestModel(responses=["Summary."]))
            result = await compactor(memory)
            # to_keep[0] is assistant for even keep_last_n — no ack
            assert result[1].role == "assistant", f"Expected no ack gap for keep_last_n={keep_last_n}"
            # result[1] should be directly from to_keep, not an ack
            assert result[1].content[0].text != "Understood.", f"Unexpected ack for keep_last_n={keep_last_n}"
            for prev, cur in zip(result, result[1:]):
                assert prev.role != cur.role, f"Consecutive {prev.role!r} for keep_last_n={keep_last_n}"

    @pytest.mark.asyncio
    async def test_orphaned_tool_result_at_boundary_cleaned_up(self) -> None:
        """When the keep_last_n cut falls mid-tool-call-sequence, the orphaned tool result
        is removed by _remove_orphaned_tool_parts before the ack check.
        The summary is followed by an assistant message (not an orphaned tool result)."""
        # Memory: u1, a1(tool_use t1), tool1_result, a2, u2(current)
        # keep_last_n=3 → raw to_keep = [tool1_result, a2, u2]
        # tool1_result is orphaned (tool_use t1 is in to_summarize)
        # After orphan cleanup: to_keep = [a2, u2]
        memory = [
            _msg_user("u1"),
            _msg_assistant_tool("t1", "search"),
            _msg_tool("t1", "result data"),
            _msg_assistant_text("a2"),
            _msg_user("u2"),
        ]
        compactor = summarize(threshold=3, keep_last_n=3, model=TestModel(responses=["Summary."]))
        result = await compactor(memory)

        # After orphan cleanup to_keep = [a2, u2] → starts with assistant → no ack
        assert result[0].role == "user"       # summary
        assert result[1].role == "assistant"  # a2 (not an orphaned tool result)
        assert result[2].role == "user"       # u2
        # No tool messages in result
        assert all(m.role != "tool" for m in result)
        # No consecutive same-role messages
        for prev, cur in zip(result, result[1:]):
            assert prev.role != cur.role

    @pytest.mark.asyncio
    async def test_ack_incremental_ack_folded_into_next_summary(self) -> None:
        """The ack inserted in one run appears in to_summarize on the next run.
        It's formatted as [assistant]: Understood. — minimal noise, does not break detection."""
        captured_prompts = []

        def capture_handler(messages):
            captured_prompts.append(messages[-1].collect_text())
            return "Updated summary."

        # Run 1: produces [summary, ack, u_last] (keep_last_n=1, memory ends with user)
        memory = [_msg_user(f"msg{i}") for i in range(6)]
        compactor = summarize(threshold=3, keep_last_n=1, model=TestModel(handler=capture_handler))
        result_run1 = await compactor(memory)

        # result_run1 = [summary(user), ack(assistant), msg5(user)]
        assert result_run1[1].role == "assistant"

        # Run 2: previous result becomes the new memory, with a new user message appended
        memory_run2 = result_run1 + [_msg_user("new message")]
        # non_system = [summary, ack, msg5, new_message] → len=4 > threshold=3
        # start_idx=1 (summary detected), to_keep=[new_message] (keep_last_n=1)
        # to_summarize = [ack, msg5] — ack is in the summarization input
        result_run2 = await compactor(memory_run2)

        assert len(captured_prompts) == 2
        # The ack ("Understood.") should appear in the second summarization prompt
        assert "Understood" in captured_prompts[1]


# ---------------------------------------------------------------------------
# _format_message_for_summary
# ---------------------------------------------------------------------------


class TestFormatMessageForSummary:
    """Tests for _format_message_for_summary helper."""

    def test_text_message(self) -> None:
        msg = _msg_user("Hello world")
        result = _format_message_for_summary(msg)
        assert result == "[user]: Hello world"

    def test_tool_use_message(self) -> None:
        msg = _msg_assistant_tool("c1", "get_weather")
        result = _format_message_for_summary(msg)
        assert "get_weather" in result
        assert "Called tool" in result

    def test_tool_result_message(self) -> None:
        msg = _msg_tool("c1", '{"temp": 72}')
        result = _format_message_for_summary(msg)
        assert "Tool result" in result
        assert "temp" in result

    def test_mixed_content(self) -> None:
        msg = _msg_assistant_mixed("Let me check.", "c1", "lookup")
        result = _format_message_for_summary(msg)
        assert "Let me check." in result
        assert "lookup" in result

    def test_long_tool_result_truncated(self) -> None:
        long_result = "x" * 1000
        msg = _msg_tool("c1", long_result)
        result = _format_message_for_summary(msg)
        assert "..." in result
        assert len(result) < 1000


# ---------------------------------------------------------------------------
# compact_tool_results — multiple tool calls per message
# ---------------------------------------------------------------------------


class TestCompactToolResultsMultipleToolCalls:
    """Tests for compact_tool_results when a single assistant message contains
    multiple ToolUseContent entries (parallel tool use)."""

    def test_drop_mode_removes_all_tool_use_and_results(self) -> None:
        """All tool_use content is stripped and all tool messages removed."""
        memory = [
            _msg_user("Hi"),
            Message(
                role="assistant",
                content=[
                    ToolUseContent(id="t1", name="search", input={}),
                    ToolUseContent(id="t2", name="lookup", input={}),
                ],
            ),
            _msg_tool("t1", "result1"),
            _msg_tool("t2", "result2"),
            _msg_assistant_text("done"),
        ]
        compactor = compact_tool_results()
        result = compactor(memory)
        assert all(m.role != "tool" for m in result)
        assert all(
            not any(isinstance(c, ToolUseContent) for c in m.content) for m in result
        )

    def test_keep_last_n_one_keeps_entire_last_batch(self) -> None:
        """With keep_last_n=1, the entire last batch (all parallel calls in one assistant message) is kept."""
        memory = [
            Message(
                role="assistant",
                content=[
                    ToolUseContent(id="t1", name="search", input={}),
                    ToolUseContent(id="t2", name="lookup", input={}),
                ],
            ),
            _msg_tool("t1", "result1"),
            _msg_tool("t2", "result2"),
        ]
        compactor = compact_tool_results(keep_last_n=1)
        result = compactor(memory)
        # t1 and t2 are in the same batch — both should survive
        tool_ids = {
            c.id
            for m in result
            for c in m.content
            if isinstance(c, (ToolUseContent, ToolResultContent))
        }
        assert "t1" in tool_ids
        assert "t2" in tool_ids

    def test_replace_mode_keeps_entire_last_batch(self) -> None:
        """Replace mode: keep_last_n=1 keeps the whole last batch; only earlier batches are replaced."""
        memory = [
            Message(
                role="assistant",
                content=[
                    ToolUseContent(id="t1", name="search", input={}),
                    ToolUseContent(id="t2", name="lookup", input={}),
                ],
            ),
            _msg_tool("t1", "big data"),
            _msg_tool("t2", "more data"),
        ]
        compactor = compact_tool_results(keep_last_n=1, replacement="[{tool_name}]")
        result = compactor(memory)
        tool_msgs = [m for m in result if m.role == "tool"]
        texts = {m.content[0].content[0].text for m in tool_msgs}
        # t1 and t2 are in the same batch — both intact, neither replaced
        assert "big data" in texts
        assert "more data" in texts


# ---------------------------------------------------------------------------
# compact_tool_results — keep_last_n=0
# ---------------------------------------------------------------------------


class TestCompactToolResultsKeepLastNZero:
    """keep_last_n=0 should behave the same as keep_last_n=None (compact everything)."""

    def test_drop_mode_zero_compacts_all(self) -> None:
        memory = [
            _msg_user("hi"),
            _msg_assistant_tool("t1"),
            _msg_tool("t1", "r1"),
            _msg_assistant_tool("t2"),
            _msg_tool("t2", "r2"),
        ]
        compactor = compact_tool_results(keep_last_n=0)
        result = compactor(memory)
        assert all(m.role != "tool" for m in result)
        assert all(
            not any(isinstance(c, ToolUseContent) for c in m.content) for m in result
        )

    def test_replace_mode_zero_replaces_all(self) -> None:
        memory = [
            _msg_assistant_tool("t1", "search"),
            _msg_tool("t1", "data"),
        ]
        compactor = compact_tool_results(keep_last_n=0, replacement="[{tool_name}]")
        result = compactor(memory)
        assert result[1].content[0].content[0].text == "[search]"


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


class TestComposition:
    """Tests for composing multiple compactors."""

    def test_compact_tool_then_keep_last_n(self) -> None:
        memory = [
            _msg_user("1"),
            _msg_user("2"),
            _msg_assistant_tool("c1"),
            _msg_tool("c1", "r"),
            _msg_assistant_text("Done."),
        ]
        compacted = compact_tool_results()(memory)
        compacted = keep_last_n_messages(3)(compacted)
        assert len(compacted) == 3
        assert all(m.role != "tool" for m in compacted)

    def test_replace_then_keep_last_n(self) -> None:
        memory = [
            _msg_user("1"),
            _msg_assistant_tool("c1", "search"),
            _msg_tool("c1", "big result " * 100),
            _msg_assistant_text("Done."),
            _msg_user("2"),
            _msg_assistant_text("ok"),
        ]
        compacted = compact_tool_results(replacement="[{tool_name}]")(memory)
        compacted = keep_last_n_messages(3)(compacted)
        assert len(compacted) == 3


# ---------------------------------------------------------------------------
# Compaction metadata stored on span
# ---------------------------------------------------------------------------


class TestCompactionMetadata:
    """Test that compaction info is stored in span metadata."""

    @pytest.mark.asyncio
    async def test_metadata_set_when_compaction_triggers(self, monkeypatch) -> None:
        """When compaction runs, span.metadata['compaction'] is populated."""
        from timbal.core.agent import Agent
        from timbal.state import set_run_context
        from timbal.state.context import RunContext
        from timbal.state.tracing.providers import InMemoryTracingProvider

        monkeypatch.setattr("timbal.core.agent.get_context_window", lambda _model: 100_000)

        agent = Agent(
            name="test_agent",
            model=TestModel(),
            memory_compaction=keep_last_n_turns(1),
            memory_compaction_ratio=0.75,
        )

        ctx1 = RunContext(tracing_provider=InMemoryTracingProvider)
        set_run_context(ctx1)
        await agent(prompt="Turn 1").collect()
        root1 = ctx1.root_span()
        root1.usage["gpt-4o-mini:input_tokens"] = 80_000
        root1.usage["gpt-4o-mini:output_tokens"] = 10_000
        await ctx1._save_trace()

        ctx2 = RunContext(parent_id=ctx1.id, tracing_provider=InMemoryTracingProvider)
        set_run_context(ctx2)
        await agent(prompt="Turn 2").collect()

        agent_spans = ctx2._trace.get_path(agent._path)
        assert agent_spans
        meta = agent_spans[0].metadata.get("compaction")
        assert meta is not None
        assert meta["triggered"] is True
        assert meta["utilization"] == pytest.approx(0.9, abs=0.01)
        assert len(meta["steps"]) == 1
        assert meta["steps"][0]["before"] >= meta["steps"][0]["after"]

        InMemoryTracingProvider._storage.clear()

    @pytest.mark.asyncio
    async def test_metadata_not_set_when_compaction_skipped(self, monkeypatch) -> None:
        """When compaction does not trigger, no compaction key is added to metadata."""
        from timbal.core.agent import Agent
        from timbal.state import set_run_context
        from timbal.state.context import RunContext
        from timbal.state.tracing.providers import InMemoryTracingProvider

        monkeypatch.setattr("timbal.core.agent.get_context_window", lambda _model: 100_000)

        agent = Agent(
            name="test_agent",
            model=TestModel(),
            memory_compaction=keep_last_n_turns(1),
            memory_compaction_ratio=0.75,
        )

        ctx1 = RunContext(tracing_provider=InMemoryTracingProvider)
        set_run_context(ctx1)
        await agent(prompt="Turn 1").collect()
        root1 = ctx1.root_span()
        root1.usage["gpt-4o-mini:input_tokens"] = 5_000
        root1.usage["gpt-4o-mini:output_tokens"] = 5_000
        await ctx1._save_trace()

        ctx2 = RunContext(parent_id=ctx1.id, tracing_provider=InMemoryTracingProvider)
        set_run_context(ctx2)
        await agent(prompt="Turn 2").collect()

        agent_spans = ctx2._trace.get_path(agent._path)
        assert agent_spans
        assert "compaction" not in agent_spans[0].metadata

        InMemoryTracingProvider._storage.clear()

    @pytest.mark.asyncio
    async def test_metadata_multiple_steps(self, monkeypatch) -> None:
        """Each compactor is recorded as a separate step with before/after counts."""
        from timbal.core.agent import Agent
        from timbal.state import set_run_context
        from timbal.state.context import RunContext
        from timbal.state.tracing.providers import InMemoryTracingProvider

        monkeypatch.setattr("timbal.core.agent.get_context_window", lambda _model: 100_000)

        agent = Agent(
            name="test_agent",
            model=TestModel(),
            memory_compaction=[
                compact_tool_results(),
                keep_last_n_turns(1),
            ],
            memory_compaction_ratio=0.75,
        )

        ctx1 = RunContext(tracing_provider=InMemoryTracingProvider)
        set_run_context(ctx1)
        await agent(prompt="Turn 1").collect()
        root1 = ctx1.root_span()
        root1.usage["gpt-4o-mini:input_tokens"] = 80_000
        root1.usage["gpt-4o-mini:output_tokens"] = 10_000
        await ctx1._save_trace()

        ctx2 = RunContext(parent_id=ctx1.id, tracing_provider=InMemoryTracingProvider)
        set_run_context(ctx2)
        await agent(prompt="Turn 2").collect()

        agent_spans = ctx2._trace.get_path(agent._path)
        assert agent_spans
        meta = agent_spans[0].metadata.get("compaction")
        assert meta is not None
        assert len(meta["steps"]) == 2
        # Steps are sequential: after of step N <= before of step N
        for step in meta["steps"]:
            assert step["before"] >= step["after"]
        # The chain is consistent: after[0] == before[1]
        assert meta["steps"][0]["after"] == meta["steps"][1]["before"]

        InMemoryTracingProvider._storage.clear()

    @pytest.mark.asyncio
    async def test_metadata_utilization_none_when_always_compact(self, monkeypatch) -> None:
        """When memory_compaction_ratio=0.0, utilization is None in metadata."""
        from timbal.core.agent import Agent
        from timbal.state import set_run_context
        from timbal.state.context import RunContext
        from timbal.state.tracing.providers import InMemoryTracingProvider

        monkeypatch.setattr("timbal.core.agent.get_context_window", lambda _model: 100_000)

        agent = Agent(
            name="test_agent",
            model=TestModel(),
            memory_compaction=keep_last_n_turns(1),
            memory_compaction_ratio=0.0,
        )

        ctx1 = RunContext(tracing_provider=InMemoryTracingProvider)
        set_run_context(ctx1)
        await agent(prompt="Turn 1").collect()
        await ctx1._save_trace()

        ctx2 = RunContext(parent_id=ctx1.id, tracing_provider=InMemoryTracingProvider)
        set_run_context(ctx2)
        await agent(prompt="Turn 2").collect()

        agent_spans = ctx2._trace.get_path(agent._path)
        assert agent_spans
        meta = agent_spans[0].metadata.get("compaction")
        assert meta is not None
        assert meta["triggered"] is True
        assert meta["utilization"] is None

        InMemoryTracingProvider._storage.clear()


# ---------------------------------------------------------------------------
# Context-window-aware triggering (unit test with mocked usage)
# ---------------------------------------------------------------------------


class TestContextWindowTriggering:
    """Test that compaction triggers based on context window utilization."""

    @pytest.mark.asyncio
    async def test_compaction_triggers_at_high_utilization(self, monkeypatch) -> None:
        """When previous run used 90% of context window, compactors should fire."""
        from timbal.core.agent import Agent
        from timbal.core.memory_compaction import keep_last_n_turns
        from timbal.state import set_run_context
        from timbal.state.context import RunContext
        from timbal.state.tracing.providers import InMemoryTracingProvider

        # Monkeypatch context window lookup
        monkeypatch.setattr("timbal.core.agent.get_context_window", lambda _model: 100_000)

        agent = Agent(
            name="test_agent",
            model=TestModel(),
            memory_compaction=keep_last_n_turns(1),
            memory_compaction_ratio=0.75,
        )

        # --- Run 1: build up memory with high usage ---
        ctx1 = RunContext(tracing_provider=InMemoryTracingProvider)
        set_run_context(ctx1)
        result1 = agent(prompt="Turn 1 user message")
        output1 = await result1.collect()
        assert output1.output is not None

        # Manually inflate the span's usage to simulate 90% utilization
        root1 = ctx1.root_span()
        root1.usage["gpt-4o-mini:input_tokens"] = 80_000
        root1.usage["gpt-4o-mini:output_tokens"] = 10_000
        await ctx1._save_trace()

        # --- Run 2: should trigger compaction ---
        ctx2 = RunContext(parent_id=ctx1.id, tracing_provider=InMemoryTracingProvider)
        set_run_context(ctx2)
        result2 = agent(prompt="Turn 2 user message")
        output2 = await result2.collect()
        assert output2.output is not None

        # Memory should have been compacted to 1 turn (last user + assistant)
        agent_spans = ctx2._trace.get_path(agent._path)
        if agent_spans:
            memory = agent_spans[0].memory
            user_msgs = [m for m in memory if hasattr(m, "role") and m.role == "user"]
            # With keep_last_n_turns(1), only the last turn should remain
            assert len(user_msgs) <= 2  # At most current turn + previous turn's last user msg

        # Clean up
        InMemoryTracingProvider._storage.clear()

    @pytest.mark.asyncio
    async def test_compaction_skipped_at_low_utilization(self, monkeypatch) -> None:
        """When previous run used only 10% of context window, compactors should NOT fire."""
        from timbal.core.agent import Agent
        from timbal.state import set_run_context
        from timbal.state.context import RunContext
        from timbal.state.tracing.providers import InMemoryTracingProvider

        monkeypatch.setattr("timbal.core.agent.get_context_window", lambda _model: 100_000)

        compaction_called = False
        original_keep_last_n_turns = keep_last_n_turns

        def tracking_compactor(n):
            inner = original_keep_last_n_turns(n)

            def wrapper(memory):
                nonlocal compaction_called
                compaction_called = True
                return inner(memory)

            return wrapper

        agent = Agent(
            name="test_agent",
            model=TestModel(),
            memory_compaction=tracking_compactor(1),
            memory_compaction_ratio=0.75,
        )

        # --- Run 1: low usage ---
        ctx1 = RunContext(tracing_provider=InMemoryTracingProvider)
        set_run_context(ctx1)
        await agent(prompt="Hello").collect()

        root1 = ctx1.root_span()
        root1.usage["gpt-4o-mini:input_tokens"] = 5_000
        root1.usage["gpt-4o-mini:output_tokens"] = 5_000
        await ctx1._save_trace()

        # --- Run 2: should skip compaction (10% utilization < 75% threshold) ---
        compaction_called = False
        ctx2 = RunContext(parent_id=ctx1.id, tracing_provider=InMemoryTracingProvider)
        set_run_context(ctx2)
        await agent(prompt="World").collect()

        assert not compaction_called, "Compaction should not trigger at 10% utilization"

        InMemoryTracingProvider._storage.clear()


# ---------------------------------------------------------------------------
# Integration test: fake context window to trigger compaction
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestCompactionIntegrationFakeContextWindow:
    """Integration test with monkeypatched context window to force compaction."""

    @pytest.mark.asyncio
    async def test_compaction_triggers_with_small_context_window(self, monkeypatch) -> None:
        """Monkeypatch context window to 200 tokens so a normal conversation triggers compaction."""
        from timbal.core.agent import Agent
        from timbal.core.memory_compaction import compact_tool_results, keep_last_n_turns
        from timbal.state import set_run_context
        from timbal.state.context import RunContext
        from timbal.state.tracing.providers import InMemoryTracingProvider

        # Tiny context window — any real LLM call will exceed 75%
        monkeypatch.setattr("timbal.core.agent.get_context_window", lambda _model: 200)

        agent = Agent(
            name="compact_agent",
            model="openai/gpt-4.1-nano",
            memory_compaction=[
                compact_tool_results(),
                keep_last_n_turns(1),
            ],
            memory_compaction_ratio=0.01,
        )

        # Run 1
        ctx1 = RunContext()
        set_run_context(ctx1)
        await agent(prompt="What is 2+2? Reply in one word.").collect()
        await ctx1._save_trace()

        # Run 2: should compact because even minimal usage exceeds tiny context window
        ctx2 = RunContext(parent_id=ctx1.id)
        set_run_context(ctx2)
        await agent(prompt="What is 3+3? Reply in one word.").collect()

        # Verify memory was compacted — should have at most 1 turn
        agent_spans = ctx2._trace.get_path(agent._path)
        assert agent_spans
        memory = agent_spans[0].memory
        # After keep_last_n_turns(1), we should have only the current turn's messages
        user_msgs = [m for m in memory if hasattr(m, "role") and m.role == "user"]
        assert len(user_msgs) <= 2

        InMemoryTracingProvider._storage.clear()


# ---------------------------------------------------------------------------
# Integration test: low ratio to always trigger, real LLM summarization
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestCompactionIntegrationLowRatio:
    """Integration test with a very low ratio to always trigger compaction including summarization."""

    @pytest.mark.asyncio
    async def test_summarize_end_to_end(self, monkeypatch) -> None:
        """Run sequential agent calls with summarize and a near-zero ratio."""
        from timbal.core.agent import Agent
        from timbal.core.memory_compaction import summarize
        from timbal.state import set_run_context
        from timbal.state.context import RunContext
        from timbal.state.tracing.providers import InMemoryTracingProvider

        # Tiny context window so even a few tokens exceed the ratio
        monkeypatch.setattr("timbal.core.agent.get_context_window", lambda _model: 200)

        agent = Agent(
            name="summarize_agent",
            model="openai/gpt-4.1-nano",
            memory_compaction=summarize(
                threshold=2,
                model="openai/gpt-4.1-nano",
                keep_last_n=2,
                max_summary_tokens=200,
            ),
            memory_compaction_ratio=0.01,  # Near-zero: always trigger
        )

        # Run 1
        ctx1 = RunContext()
        set_run_context(ctx1)
        await agent(prompt="My name is Alice and I live in Barcelona. Reply briefly.").collect()
        await ctx1._save_trace()

        # Run 2: memory now has [user1, assistant1, user2] = 3 messages > threshold(2)
        ctx2 = RunContext(parent_id=ctx1.id)
        set_run_context(ctx2)
        await agent(prompt="I work as a data scientist. Reply briefly.").collect()
        await ctx2._save_trace()

        # Run 3: memory should contain summary from run 2's compaction
        ctx3 = RunContext(parent_id=ctx2.id)
        set_run_context(ctx3)
        await agent(prompt="What do you know about me? Reply briefly.").collect()

        # Check that memory contains a summary marker
        agent_spans = ctx3._trace.get_path(agent._path)
        assert agent_spans
        memory = agent_spans[0].memory
        memory_texts = []
        for m in memory:
            if hasattr(m, "content"):
                for c in m.content:
                    if hasattr(c, "text"):
                        memory_texts.append(c.text)

        has_summary = any(_SUMMARY_MARKER in t for t in memory_texts)
        assert has_summary, f"Expected summary marker in memory. Got: {memory_texts[:5]}"

        InMemoryTracingProvider._storage.clear()


# ---------------------------------------------------------------------------
# Integration test: Anthropic — alternation fix
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestCompactionIntegrationAnthropic:
    """Integration tests against the real Anthropic API.

    These tests specifically validate the strict user/assistant alternation fix:
    Anthropic rejects consecutive same-role messages with a 400, so any regression
    in the ack-insertion logic would surface here as an API error rather than a
    silent test failure.
    """

    @pytest.mark.asyncio
    async def test_summarize_odd_keep_last_n_no_400(self, monkeypatch) -> None:
        """summarize(keep_last_n=1) produces an ack when to_keep starts with a user
        message. Verify Anthropic accepts the resulting message sequence (no 400)."""
        from timbal.core.agent import Agent
        from timbal.core.memory_compaction import summarize
        from timbal.state import set_run_context
        from timbal.state.context import RunContext
        from timbal.state.tracing.providers import InMemoryTracingProvider

        # Tiny context window forces compaction on every run
        monkeypatch.setattr("timbal.core.agent.get_context_window", lambda _model: 200)

        agent = Agent(
            name="anthropic_ack_agent",
            model="anthropic/claude-haiku-4-5",
            max_tokens=256,
            memory_compaction=summarize(
                threshold=2,
                model="anthropic/claude-haiku-4-5",
                keep_last_n=1,  # Odd → to_keep=[current_user] → ack required
                max_summary_tokens=150,
            ),
            memory_compaction_ratio=0.01,
        )

        ctx1 = RunContext()
        set_run_context(ctx1)
        await agent(prompt="My name is Bob. Reply in one sentence.").collect()
        await ctx1._save_trace()

        # Run 2: triggers summarize with keep_last_n=1 — ack must be inserted or
        # Anthropic will return a 400 for consecutive user messages.
        ctx2 = RunContext(parent_id=ctx1.id)
        set_run_context(ctx2)
        result = await agent(prompt="What is my name? Reply in one sentence.").collect()

        # If we got here without an exception, Anthropic accepted the message sequence.
        assert result.output is not None

        InMemoryTracingProvider._storage.clear()

    @pytest.mark.asyncio
    async def test_summarize_even_keep_last_n_no_400(self, monkeypatch) -> None:
        """summarize(keep_last_n=2) — to_keep starts with assistant, no ack inserted.
        Verify Anthropic also accepts this (regression guard for the no-ack path)."""
        from timbal.core.agent import Agent
        from timbal.core.memory_compaction import summarize
        from timbal.state import set_run_context
        from timbal.state.context import RunContext
        from timbal.state.tracing.providers import InMemoryTracingProvider

        monkeypatch.setattr("timbal.core.agent.get_context_window", lambda _model: 200)

        agent = Agent(
            name="anthropic_no_ack_agent",
            model="anthropic/claude-haiku-4-5",
            max_tokens=256,
            memory_compaction=summarize(
                threshold=2,
                model="anthropic/claude-haiku-4-5",
                keep_last_n=2,  # Even → to_keep=[assistant, user] → no ack
                max_summary_tokens=150,
            ),
            memory_compaction_ratio=0.01,
        )

        ctx1 = RunContext()
        set_run_context(ctx1)
        await agent(prompt="My name is Carol. Reply in one sentence.").collect()
        await ctx1._save_trace()

        ctx2 = RunContext(parent_id=ctx1.id)
        set_run_context(ctx2)
        result = await agent(prompt="What is my name? Reply in one sentence.").collect()

        assert result.output is not None

        InMemoryTracingProvider._storage.clear()

    @pytest.mark.asyncio
    async def test_summarize_context_carried_forward(self, monkeypatch) -> None:
        """Verify that facts from summarized turns are still accessible to the model
        after compaction — the summary actually contains the key information."""
        from timbal.core.agent import Agent
        from timbal.core.memory_compaction import summarize
        from timbal.state import set_run_context
        from timbal.state.context import RunContext
        from timbal.state.tracing.providers import InMemoryTracingProvider

        monkeypatch.setattr("timbal.core.agent.get_context_window", lambda _model: 200)

        agent = Agent(
            name="anthropic_context_agent",
            model="anthropic/claude-haiku-4-5",
            max_tokens=256,
            memory_compaction=summarize(
                threshold=2,
                model="anthropic/claude-haiku-4-5",
                keep_last_n=1,
                max_summary_tokens=200,
            ),
            memory_compaction_ratio=0.01,
        )

        ctx1 = RunContext()
        set_run_context(ctx1)
        await agent(prompt="My favourite colour is indigo. Reply in one sentence.").collect()
        await ctx1._save_trace()

        ctx2 = RunContext(parent_id=ctx1.id)
        set_run_context(ctx2)
        await agent(prompt="I have a dog named Pepper. Reply in one sentence.").collect()
        await ctx2._save_trace()

        # Run 3: both facts were summarized — can the model still recall them?
        ctx3 = RunContext(parent_id=ctx2.id)
        set_run_context(ctx3)
        result = await agent(prompt="What is my favourite colour and my dog's name? Reply in one sentence.").collect()

        assert result.output is not None
        response_text = result.output.collect_text().lower()
        assert "indigo" in response_text, f"Expected 'indigo' in response, got: {response_text}"
        assert "pepper" in response_text, f"Expected 'Pepper' in response, got: {response_text}"

        InMemoryTracingProvider._storage.clear()
