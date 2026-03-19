"""Memory compaction strategies for reducing context window usage.

Apply compaction to conversation memory before sending to the LLM to avoid
exceeding context limits. Strategies can be composed (applied in order).
"""

from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Protocol, runtime_checkable

from ..state import (
    get_call_id,
    get_parent_call_id,
    get_run_context,
    set_call_id,
    set_parent_call_id,
    set_run_context,
)
from ..types.content import TextContent, ToolResultContent, ToolUseContent
from ..types.message import Message

__all__ = [
    "MemoryCompactor",
    "drop_tool_use_and_results",
    "keep_last_n_messages",
    "keep_last_n_tokens",
    "keep_last_n_turns",
    "replace_tool_results_with_placeholder",
    "summarize_old_messages",
    "truncate_message_tokens",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_tool_use_ids(messages: list[Message]) -> set[str]:
    """Collect all tool_use call ids from assistant messages."""
    ids = set()
    for msg in messages:
        if msg.role == "assistant":
            for c in msg.content:
                if isinstance(c, ToolUseContent):
                    ids.add(c.id)
    return ids


def _collect_tool_result_ids(messages: list[Message]) -> set[str]:
    """Collect all tool_result call ids from tool messages."""
    ids = set()
    for msg in messages:
        if msg.role == "tool":
            for c in msg.content:
                if isinstance(c, ToolResultContent):
                    ids.add(c.id)
    return ids


def _remove_orphaned_tool_parts(memory: list[Message]) -> list[Message]:
    """Remove tool_use without tool_result, and tool_result without tool_use."""
    tool_use_ids = _collect_tool_use_ids(memory)
    tool_result_ids = _collect_tool_result_ids(memory)
    valid_tool_ids = tool_use_ids & tool_result_ids

    result = []
    for msg in memory:
        if msg.role == "tool":
            kept = [
                c
                for c in msg.content
                if isinstance(c, ToolResultContent) and c.id in valid_tool_ids
            ]
            if not kept:
                continue
            result.append(Message(role=msg.role, content=kept, stop_reason=msg.stop_reason))
        elif msg.role == "assistant":
            kept = [
                c
                for c in msg.content
                if not isinstance(c, ToolUseContent) or c.id in valid_tool_ids
            ]
            if not kept:
                continue
            result.append(Message(role=msg.role, content=kept, stop_reason=msg.stop_reason))
        else:
            result.append(msg)
    return result


def _estimate_tokens(text: str, encoding: object | None) -> int:
    """Estimate token count. Uses tiktoken if encoding given, else ~4 chars/token."""
    if encoding is not None:
        try:
            return len(encoding.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)


def _count_message_tokens(msg: Message, encoding: object | None) -> int:
    """Estimate total tokens for a message (text content only)."""
    total = 0
    for c in msg.content:
        if isinstance(c, TextContent):
            total += _estimate_tokens(c.text, encoding)
        elif isinstance(c, ToolResultContent):
            for item in c.content:
                if isinstance(item, TextContent):
                    total += _estimate_tokens(item.text, encoding)
    return total


# ---------------------------------------------------------------------------
# Compactors
# ---------------------------------------------------------------------------


@runtime_checkable
class MemoryCompactor(Protocol):
    """Strategy for compacting conversation memory before LLM call.

    A compactor receives a list of messages and returns a (possibly modified)
    list. Can be a sync or async callable.
    """

    def __call__(self, memory: list[Message]) -> list[Message]: ...


def drop_tool_use_and_results(
    threshold: int = 0,
    keep_last_n: int | None = None,
) -> Callable[[list[Message]], list[Message]]:
    """Remove tool use and tool result messages to reduce token usage.

    Tool results often contain large JSON/API responses. The assistant's
    final text response already summarizes the key information, so removing
    tool details from history significantly reduces tokens with minimal
    context loss.

    - Removes messages with role="tool" (except those in the last N pairs)
    - Strips tool_use content from assistant messages (keeps text only, or
      keeps tool_use for the last N pairs when keep_last_n is set)
    - Drops assistant messages that become empty after stripping

    Args:
        threshold: Only apply when len(memory) > threshold. Use 0 (default)
            to always apply, or e.g. 20 to only drop tools when conversation
            exceeds 20 messages.
        keep_last_n: If set, keep the last N tool use/result pairs and drop
            all previous ones. Use None (default) to drop all tool use and
            results. Use 1 to keep only the most recent tool call.

    Note:
        Tool results without a matching tool_use (orphaned) are always dropped.
        Tool use without a matching tool_result is also dropped.

    Returns:
        A compactor function.
    """

    def _compact(memory: list[Message]) -> list[Message]:
        if len(memory) <= threshold:
            return memory

        # Collect tool_use ids in chronological order
        tool_use_ids_ordered: list[str] = []
        for msg in memory:
            if msg.role == "assistant":
                for c in msg.content:
                    if isinstance(c, ToolUseContent):
                        tool_use_ids_ordered.append(c.id)

        # Determine which tool ids to keep
        if keep_last_n is not None and keep_last_n > 0:
            kept_ids = set(tool_use_ids_ordered[-keep_last_n:])
        else:
            kept_ids = set()

        result: list[Message] = []
        for msg in memory:
            if msg.role == "tool":
                kept = [
                    c
                    for c in msg.content
                    if isinstance(c, ToolResultContent) and c.id in kept_ids
                ]
                if not kept:
                    continue
                result.append(Message(role=msg.role, content=kept, stop_reason=msg.stop_reason))
            elif msg.role == "assistant":
                kept = [
                    c
                    for c in msg.content
                    if not isinstance(c, ToolUseContent) or c.id in kept_ids
                ]
                if not kept:
                    continue
                result.append(Message(role=msg.role, content=kept, stop_reason=msg.stop_reason))
            else:
                result.append(msg)
        return result

    return _compact


def keep_last_n_messages(n: int) -> Callable[[list[Message]], list[Message]]:
    """Keep only the last N messages (most recent context).

    Structure-aware: never leaves orphaned tool_use or tool_result. If truncation
    would cut a tool call sequence, removes the orphaned part (tool_result without
    tool_use, or tool_use without tool_result).

    Args:
        n: Maximum number of messages to retain.

    Returns:
        A compactor function.
    """

    def _compact(memory: list[Message]) -> list[Message]:
        if len(memory) <= n:
            return _remove_orphaned_tool_parts(memory)
        truncated = memory[-n:]
        return _remove_orphaned_tool_parts(truncated)

    return _compact


def truncate_message_tokens(
    max_tokens_per_message: int,
    encoding: object | None = None,
    keep_last_n: int = 0,
) -> Callable[[list[Message]], list[Message]]:
    """Truncate each message's text to a maximum token count.

    Does not truncate the last `keep_last_n` messages (keeps current exchange full).
    For messages exceeding the limit, keeps the tail (most recent part).
    Non-text content (images, etc.) is preserved as-is.

    Args:
        max_tokens_per_message: Maximum tokens per message. Excess is truncated.
        encoding: Optional tiktoken encoding for accurate counting. If None,
            uses ~4 chars per token heuristic. Example: tiktoken.get_encoding("cl100k_base").
        keep_last_n: Do not truncate the last N messages. Keeps the most recent
            exchange full. Default: 0 (truncate all).

    Returns:
        A compactor function.
    """

    def _compact(memory: list[Message]) -> list[Message]:
        result = []
        msg_count = len(memory)
        for i, msg in enumerate(memory):
            if i >= msg_count - keep_last_n:
                result.append(msg)
                continue
            new_content = []
            for c in msg.content:
                if isinstance(c, TextContent):
                    tokens = _estimate_tokens(c.text, encoding)
                    if tokens <= max_tokens_per_message:
                        new_content.append(c)
                    else:
                        if encoding is not None:
                            try:
                                encoded = encoding.encode(c.text)
                                kept = encoded[-max_tokens_per_message:]
                                truncated = "..." + encoding.decode(kept)
                            except Exception:
                                truncated = "..." + c.text[-(max_tokens_per_message * 4):]
                        else:
                            truncated = "..." + c.text[-(max_tokens_per_message * 4):]
                        new_content.append(TextContent(text=truncated))
                else:
                    new_content.append(c)
            result.append(Message(role=msg.role, content=new_content, stop_reason=msg.stop_reason))
        return result

    return _compact


def keep_last_n_tokens(
    max_tokens: int,
    encoding: object | None = None,
) -> Callable[[list[Message]], list[Message]]:
    """Keep only the last N tokens (most recent context).

    Iterates from the end, keeping messages until the token budget is exceeded.
    Structure-aware: never leaves orphaned tool_use or tool_result.

    Args:
        max_tokens: Maximum total tokens to retain.
        encoding: Optional tiktoken encoding. If None, uses ~4 chars per token.

    Returns:
        A compactor function.
    """

    def _compact(memory: list[Message]) -> list[Message]:
        if not memory:
            return memory
        kept = []
        tokens = 0
        for msg in reversed(memory):
            msg_tokens = _count_message_tokens(msg, encoding)
            if tokens + msg_tokens > max_tokens and kept:
                break
            kept.insert(0, msg)
            tokens += msg_tokens
        return _remove_orphaned_tool_parts(kept)

    return _compact


def keep_last_n_turns(n: int) -> Callable[[list[Message]], list[Message]]:
    """Keep only the last N turns (user + assistant pairs).

    A turn starts with a user message and includes all messages until the next
    user message (assistant, tool calls, etc.). Structure-aware.

    Args:
        n: Maximum number of turns to retain.

    Returns:
        A compactor function.
    """

    def _compact(memory: list[Message]) -> list[Message]:
        if not memory:
            return memory
        user_indices = [i for i, m in enumerate(memory) if m.role == "user"]
        if len(user_indices) <= n:
            return _remove_orphaned_tool_parts(memory)
        start = user_indices[-n]
        return _remove_orphaned_tool_parts(memory[start:])

    return _compact


def replace_tool_results_with_placeholder(
    template: str = "[Tool result: {tool_name}]",
) -> Callable[[list[Message]], list[Message]]:
    """Replace tool result content with a short placeholder.

    Reduces tokens while preserving the fact that a tool was called.
    Template supports: {tool_name}, {call_id}, {result_length}.
    Unknown placeholders are replaced with empty string.

    Args:
        template: Format string with optional placeholders. Default: "[Tool result: {tool_name}]"

    Returns:
        A compactor function.
    """

    def _compact(memory: list[Message]) -> list[Message]:
        call_id_to_name = {}
        for msg in memory:
            if msg.role == "assistant":
                for c in msg.content:
                    if isinstance(c, ToolUseContent):
                        call_id_to_name[c.id] = c.name

        result = []
        for msg in memory:
            if msg.role != "tool":
                result.append(msg)
                continue
            new_content = []
            for c in msg.content:
                if isinstance(c, ToolResultContent):
                    tool_name = call_id_to_name.get(c.id, "unknown")
                    result_length = sum(
                        len(item.text) if isinstance(item, TextContent) else 0
                        for item in c.content
                    )
                    placeholder = template.format_map(
                        defaultdict(str, tool_name=tool_name, call_id=c.id, result_length=result_length)
                    )
                    new_content.append(ToolResultContent(id=c.id, content=[TextContent(text=placeholder)]))
                else:
                    new_content.append(c)
            result.append(Message(role=msg.role, content=new_content, stop_reason=msg.stop_reason))
        return result

    return _compact


def summarize_old_messages(
    threshold: int,
    model: str = "openai/gpt-4.1-nano",
    keep_last_n: int = 4,
) -> Callable[[list[Message]], Awaitable[list[Message]]]:
    """Summarize old messages when exceeding threshold. Returns async compactor.

    When len(memory) > threshold, summarizes messages before the last `keep_last_n`
    into a single user message. Uses the same Agent/LLM infrastructure as Timbal.

    Args:
        threshold: Trigger summarization when message count exceeds this.
        model: Model for summarization (e.g. "openai/gpt-4.1-mini").
        keep_last_n: Number of recent messages to keep unsummarized.

    Returns:
        An async compactor function.
    """

    async def _compact(memory: list[Message]) -> list[Message]:
        if len(memory) <= threshold:
            return memory
        to_summarize = memory[:-keep_last_n]
        to_keep = memory[-keep_last_n:]

        lines = []
        for msg in to_summarize:
            role = msg.role
            text = msg.collect_text()
            if text:
                lines.append(f"{role}: {text}")

        if not lines:
            return to_keep

        conv_text = "\n\n".join(lines)
        prompt = f"Summarize this conversation briefly in 2-4 sentences, preserving key facts and decisions:\n\n{conv_text}"

        from .agent import Agent

        saved_context = get_run_context()
        saved_call_id = get_call_id()
        saved_parent_call_id = get_parent_call_id()

        try:
            summarizer = Agent(
                name="summarizer",
                model=model,
                system_prompt="You summarize conversations concisely.",
                model_params={"max_tokens": 300},
            )
            last_event = None
            async for event in summarizer(prompt=prompt):
                last_event = event
        finally:
            if saved_context is not None:
                set_run_context(saved_context)
            set_call_id(saved_call_id)
            set_parent_call_id(saved_parent_call_id)

        summary = ""
        if last_event is not None and hasattr(last_event, "output") and last_event.output is not None:
            summary = last_event.output.collect_text()

        summary_msg = Message.validate(
            {"role": "user", "content": f"[Previous conversation summary] {summary}"}
        )
        return [summary_msg] + to_keep

    return _compact