"""Tests for memory compaction strategies."""

import pytest
from timbal.core.memory_compaction import (
    drop_tool_use_and_results,
    keep_last_n_messages,
    keep_last_n_tokens,
    keep_last_n_turns,
    replace_tool_results_with_placeholder,
    summarize_old_messages,
    truncate_message_tokens,
)
from timbal.types.content import TextContent, ToolResultContent, ToolUseContent
from timbal.types.message import Message


def _msg_user(text: str) -> Message:
    return Message(role="user", content=[TextContent(text=text)])


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
# drop_tool_use_and_results
# ---------------------------------------------------------------------------


class TestDropToolUseAndResults:
    """Tests for drop_tool_use_and_results."""

    def test_empty_memory(self) -> None:
        compactor = drop_tool_use_and_results()
        assert compactor([]) == []

    def test_only_user_messages(self) -> None:
        memory = [_msg_user("Hi"), _msg_user("Bye")]
        compactor = drop_tool_use_and_results()
        assert compactor(memory) == memory

    def test_user_and_assistant_text_only(self) -> None:
        memory = [_msg_user("Hi"), _msg_assistant_text("Hello!")]
        compactor = drop_tool_use_and_results()
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
        compactor = drop_tool_use_and_results(threshold=0, keep_last_n=None)
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
        compactor = drop_tool_use_and_results()
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
        compactor = drop_tool_use_and_results()
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
        compactor = drop_tool_use_and_results(threshold=0, keep_last_n=1)
        result = compactor(memory)
        tool_msgs = [m for m in result if m.role == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].content[0].id == "t2"
        assistant_with_tool = [
            m for m in result
            if m.role == "assistant" and any(isinstance(c, ToolUseContent) for c in m.content)
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
        compactor = drop_tool_use_and_results(threshold=0, keep_last_n=2)
        result = compactor(memory)
        kept_ids = set()
        for m in result:
            for c in m.content:
                if isinstance(c, (ToolUseContent, ToolResultContent)):
                    kept_ids.add(c.id)
        assert kept_ids == {"t2", "t3"}

    def test_threshold_short_conversation_unchanged(self) -> None:
        memory = [_msg_user("hi"), _msg_assistant_text("hello")]
        compactor = drop_tool_use_and_results(threshold=10, keep_last_n=None)
        result = compactor(memory)
        assert result == memory

    def test_threshold_below_returns_unchanged(self) -> None:
        memory = [
            _msg_user("Hi"),
            _msg_assistant_tool("c1"),
            _msg_tool("c1", "r"),
            _msg_assistant_text("Done."),
        ]
        compactor = drop_tool_use_and_results(threshold=5)
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
        compactor = drop_tool_use_and_results(threshold=3)
        result = compactor(memory)
        assert len(result) == 2
        assert not any(m.role == "tool" for m in result)


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
# truncate_message_tokens
# ---------------------------------------------------------------------------


class TestTruncateMessageTokens:
    """Tests for truncate_message_tokens."""

    def test_empty_memory(self) -> None:
        compactor = truncate_message_tokens(100)
        assert compactor([]) == []

    def test_short_messages_unchanged(self) -> None:
        memory = [_msg_user("Hi"), _msg_user("Bye")]
        compactor = truncate_message_tokens(100)
        result = compactor(memory)
        assert result[0].content[0].text == "Hi"
        assert result[1].content[0].text == "Bye"

    def test_long_message_truncated(self) -> None:
        long_text = "x" * 500
        memory = [_msg_user(long_text)]
        compactor = truncate_message_tokens(50)
        result = compactor(memory)
        assert result[0].content[0].text.startswith("...")
        assert len(result[0].content[0].text) < len(long_text)

    def test_keep_last_n_untouched(self) -> None:
        short = _msg_user("short")
        long_text = "x" * 500
        memory = [_msg_user(long_text), short]
        compactor = truncate_message_tokens(50, keep_last_n=1)
        result = compactor(memory)
        assert result[0].content[0].text.startswith("...")
        assert result[1].content[0].text == "short"


# ---------------------------------------------------------------------------
# keep_last_n_tokens
# ---------------------------------------------------------------------------


class TestKeepLastNTokens:
    """Tests for keep_last_n_tokens."""

    def test_empty_memory(self) -> None:
        compactor = keep_last_n_tokens(100)
        assert compactor([]) == []

    def test_all_fit_in_budget(self) -> None:
        memory = [_msg_user("Hi"), _msg_user("Bye")]
        compactor = keep_last_n_tokens(100)
        result = compactor(memory)
        assert len(result) == 2

    def test_budget_cuts_messages(self) -> None:
        memory = [
            _msg_user("A" * 100),
            _msg_user("B" * 100),
            _msg_user("C" * 100),
        ]
        compactor = keep_last_n_tokens(60)
        result = compactor(memory)
        assert len(result) <= 2

    def test_removes_orphaned_tool_parts(self) -> None:
        memory = [
            _msg_user("old"),
            _msg_assistant_tool("c1"),
            _msg_tool("c1", "result"),
            _msg_user("new"),
        ]
        compactor = keep_last_n_tokens(20)
        result = compactor(memory)
        tool_use_ids = {
            c.id for m in result if m.role == "assistant"
            for c in m.content if isinstance(c, ToolUseContent)
        }
        tool_result_ids = {
            c.id for m in result if m.role == "tool"
            for c in m.content if isinstance(c, ToolResultContent)
        }
        assert tool_use_ids == tool_result_ids


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
# replace_tool_results_with_placeholder
# ---------------------------------------------------------------------------


class TestReplaceToolResultsWithPlaceholder:
    """Tests for replace_tool_results_with_placeholder."""

    def test_no_tool_messages(self) -> None:
        memory = [_msg_user("Hi"), _msg_assistant_text("Hello")]
        compactor = replace_tool_results_with_placeholder()
        result = compactor(memory)
        assert result == memory

    def test_replaces_with_default_template(self) -> None:
        memory = [
            _msg_assistant_tool("t1", "get_weather"),
            _msg_tool("t1", "temperature is 72F"),
        ]
        compactor = replace_tool_results_with_placeholder()
        result = compactor(memory)
        tool_msg = result[1]
        assert tool_msg.role == "tool"
        assert tool_msg.content[0].content[0].text == "[Tool result: get_weather]"

    def test_custom_template(self) -> None:
        memory = [
            _msg_assistant_tool("call-123", "search"),
            _msg_tool("call-123", "x" * 100),
        ]
        compactor = replace_tool_results_with_placeholder(
            template="{tool_name} (id={call_id}, len={result_length})"
        )
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
        compactor = replace_tool_results_with_placeholder(
            template="{tool_name}-{unknown_placeholder}"
        )
        result = compactor(memory)
        text = result[1].content[0].content[0].text
        assert text == "foo-"


# ---------------------------------------------------------------------------
# summarize_old_messages
# ---------------------------------------------------------------------------


class TestSummarizeOldMessages:
    """Tests for summarize_old_messages."""

    @pytest.mark.asyncio
    async def test_below_threshold_returns_unchanged(self) -> None:
        memory = [_msg_user("Hi"), _msg_assistant_text("Hello")]
        compactor = summarize_old_messages(threshold=10, keep_last_n=2)
        result = await compactor(memory)
        assert result == memory

    @pytest.mark.asyncio
    async def test_at_threshold_returns_unchanged(self) -> None:
        memory = [_msg_user(f"msg{i}") for i in range(5)]
        compactor = summarize_old_messages(threshold=5, keep_last_n=2)
        result = await compactor(memory)
        assert result == memory

    @pytest.mark.asyncio
    async def test_above_threshold_summarizes(self) -> None:
        memory = [_msg_user(f"Message {i}") for i in range(20)]
        compactor = summarize_old_messages(threshold=5, keep_last_n=2)
        result = await compactor(memory)
        assert len(result) < 20
        assert result[0].role == "user"
        assert "[Previous conversation summary]" in result[0].content[0].text


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


class TestComposition:
    """Tests for composing multiple compactors."""

    def test_drop_tool_then_keep_last_n(self) -> None:
        memory = [
            _msg_user("1"),
            _msg_user("2"),
            _msg_assistant_tool("c1"),
            _msg_tool("c1", "r"),
            _msg_assistant_text("Done."),
        ]
        compacted = drop_tool_use_and_results()(memory)
        compacted = keep_last_n_messages(3)(compacted)
        assert len(compacted) == 3
        assert all(m.role != "tool" for m in compacted)
