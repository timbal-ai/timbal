"""Memory compaction strategies for reducing context window usage.

Apply compaction to conversation memory before sending to the LLM to avoid
exceeding context limits. Strategies can be composed (applied in order).
"""

import json
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any

import structlog

from ..types.content import TextContent, ToolResultContent, ToolUseContent
from ..types.message import Message

logger = structlog.get_logger("timbal.core.memory_compaction")

MemoryCompactor = Callable[[list[Message]], list[Message]] | Callable[[list[Message]], Awaitable[list[Message]]]
"""A compactor receives a list of messages and returns a (possibly modified) list.
Can be sync or async."""

__all__ = [
    "MemoryCompactor",
    "compact_tool_results",
    "keep_last_n_messages",
    "keep_last_n_turns",
    "summarize",
]

_SUMMARY_MARKER = "[Conversation Summary]"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_ids(messages: list[Message], role: str, content_type: type) -> set[str]:
    """Collect all content ids of a given type from messages with the given role."""
    return {c.id for msg in messages if msg.role == role for c in msg.content if isinstance(c, content_type)}


def _remove_orphaned_tool_parts(memory: list[Message]) -> list[Message]:
    """Remove tool_use without tool_result, and tool_result without tool_use."""
    tool_use_ids = _collect_ids(memory, "assistant", ToolUseContent)
    tool_result_ids = _collect_ids(memory, "tool", ToolResultContent)
    valid_tool_ids = tool_use_ids & tool_result_ids

    result = []
    for msg in memory:
        if msg.role == "tool":
            kept = [c for c in msg.content if isinstance(c, ToolResultContent) and c.id in valid_tool_ids]
            if not kept:
                continue
            result.append(Message(role=msg.role, content=kept, stop_reason=msg.stop_reason))
        elif msg.role == "assistant":
            kept = [c for c in msg.content if not isinstance(c, ToolUseContent) or c.id in valid_tool_ids]
            if not kept:
                continue
            result.append(Message(role=msg.role, content=kept, stop_reason=msg.stop_reason))
        else:
            result.append(msg)
    return result


def _format_message_for_summary(msg: Message) -> str | None:
    """Format a message for the summarization prompt, preserving tool call structure."""
    parts = []
    for c in msg.content:
        if isinstance(c, TextContent):
            parts.append(c.text)
        elif isinstance(c, ToolUseContent):
            input_str = json.dumps(c.input)[:200] if c.input else "{}"
            parts.append(f"[Called tool '{c.name}' with: {input_str}]")
        elif isinstance(c, ToolResultContent):
            result_text = ""
            for item in c.content:
                if isinstance(item, TextContent):
                    result_text += item.text
            if len(result_text) > 500:
                result_text = result_text[:500] + "..."
            parts.append(f"[Tool result for '{c.id}': {result_text}]")
    if parts:
        return f"[{msg.role}]: " + " ".join(parts)
    return None


_INITIAL_SUMMARY_PROMPT = """\
Summarize the following conversation, preserving:
1. All specific values, identifiers, names, URLs, dates, and numbers mentioned
2. The outcome of every tool call (what tool was called, what it returned)
3. User preferences, decisions, and explicit instructions
4. Any constraints or requirements established

Conversation:
{messages}

Provide a structured summary using this format:

## Key Facts & Decisions
- [bullet points of decisions, preferences, established facts]

## Tool Outcomes
- [tool_name]: [what it returned / key result]

## Flow
- [brief chronological narrative of the conversation progression]"""

_INCREMENTAL_SUMMARY_PROMPT = """\
Update this conversation summary with new messages.

Current summary:
{previous_summary}

New messages since last summary:
{new_messages}

Update the summary to incorporate the new messages. You must:
1. Preserve all specific values, identifiers, names, URLs, dates, and numbers
2. Record the outcome of every tool call (what tool was called, what it returned)
3. Capture user preferences, decisions, and explicit instructions
4. Drop information that has been superseded by newer messages
5. Keep the summary concise but complete

Use this format:

## Key Facts & Decisions
- [bullet points]

## Tool Outcomes
- [tool_name]: [result]

## Flow
- [chronological narrative]"""


# ---------------------------------------------------------------------------
# Summarizer LLM call
# ---------------------------------------------------------------------------


async def _call_summarizer(model: Any, prompt: str, max_summary_tokens: int) -> str | None:
    """Call _llm_router directly for summarization. No Tool/Agent overhead.

    Returns the summary text, or None if the LLM call produced no usable output.
    Callers must treat None as a signal to leave memory unchanged.
    """
    import time

    from ..collectors import get_collector_registry
    from .llm_router import _llm_router

    prompt_message = Message.validate({"role": "user", "content": prompt})
    chunks = _llm_router(
        model=model,
        messages=[prompt_message],
        system_prompt="You are a conversation summarizer. Produce structured, factual summaries. Never invent facts.",
        max_tokens=max_summary_tokens,
        temperature=0.0,
    )

    # Get first chunk to determine collector type
    start = time.perf_counter()
    first_chunk = await chunks.__anext__()
    collector_type = get_collector_registry().get_collector_type(first_chunk)
    if collector_type is None:
        return None

    collector = collector_type(async_gen=chunks, start=start)
    collector.process(first_chunk)
    result_message = await collector.collect()

    if isinstance(result_message, Message):
        return result_message.collect_text() or None
    return None


# ---------------------------------------------------------------------------
# Compactors
# ---------------------------------------------------------------------------


def compact_tool_results(
    keep_last_n: int | None = None,
    threshold: int = 0,
    replacement: str | Callable[[str, str, str], str] | None = None,
) -> Callable[[list[Message]], list[Message]]:
    """Compact tool use and tool result messages to reduce token usage.

    Unified strategy that can drop, replace, or custom-transform tool results.
    The assistant's final text response typically summarizes the key information,
    so compacting tool details from history significantly reduces tokens with
    minimal context loss.

    Behavior depends on ``replacement``:

    - ``None`` (default): drop tool results and tool_use content entirely.
      Removes role="tool" messages and strips ToolUseContent from assistant
      messages. Assistant messages that become empty are dropped.
    - ``str``: replace each tool result's content with the template string.
      Supports placeholders: ``{tool_name}``, ``{call_id}``, ``{result_length}``.
      Tool_use content in assistant messages is preserved.
    - ``callable(tool_name, call_id, result_text) -> str``: call the function
      for each tool result and use the return value as the replacement text.
      Tool_use content in assistant messages is preserved.

    Args:
        keep_last_n: If set, keep the last N tool use/result pairs intact
            and only compact earlier ones. Use None (default) to compact all.
        threshold: Only apply when len(memory) > threshold. Use 0 (default)
            to always apply.
        replacement: Controls what happens to compacted tool results. See above.

    Returns:
        A compactor function.
    """

    def _compact(memory: list[Message]) -> list[Message]:
        if len(memory) <= threshold:
            return memory

        # Collect tool_use IDs grouped by assistant message.
        # Each inner list is one "batch" — all tool calls made in a single assistant turn
        # (parallel tool use). keep_last_n refers to batches, not individual call IDs.
        tool_use_batches: list[list[str]] = []
        call_id_to_name: dict[str, str] = {}
        for msg in memory:
            if msg.role == "assistant":
                batch = []
                for c in msg.content:
                    if isinstance(c, ToolUseContent):
                        batch.append(c.id)
                        call_id_to_name[c.id] = c.name
                if batch:
                    tool_use_batches.append(batch)

        # Determine which tool ids to keep intact (last N batches)
        if keep_last_n is not None and keep_last_n > 0:
            kept_ids = {cid for batch in tool_use_batches[-keep_last_n:] for cid in batch}
        else:
            kept_ids = set()

        drop_mode = replacement is None

        result: list[Message] = []
        for msg in memory:
            if msg.role == "tool":
                new_content = []
                for c in msg.content:
                    if isinstance(c, ToolResultContent):
                        if c.id in kept_ids:
                            new_content.append(c)
                        elif drop_mode:
                            continue  # drop entirely
                        else:
                            # Replace content
                            tool_name = call_id_to_name.get(c.id, "unknown")
                            result_text = "".join(
                                item.text for item in c.content if isinstance(item, TextContent)
                            )
                            if callable(replacement):
                                placeholder = replacement(tool_name, c.id, result_text)
                            else:
                                placeholder = replacement.format_map(
                                    defaultdict(
                                        str,
                                        tool_name=tool_name,
                                        call_id=c.id,
                                        result_length=str(len(result_text)),
                                    )
                                )
                            new_content.append(
                                ToolResultContent(id=c.id, content=[TextContent(text=placeholder)])
                            )
                    else:
                        new_content.append(c)
                if not new_content:
                    continue
                result.append(Message(role=msg.role, content=new_content, stop_reason=msg.stop_reason))
            elif msg.role == "assistant":
                if drop_mode:
                    kept = [c for c in msg.content if not isinstance(c, ToolUseContent) or c.id in kept_ids]
                    if not kept:
                        continue
                    result.append(Message(role=msg.role, content=kept, stop_reason=msg.stop_reason))
                else:
                    result.append(msg)
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




def summarize(
    threshold: int,
    model: Any | None = None,
    keep_last_n: int = 4,
    max_summary_tokens: int = 500,
) -> Callable[[list[Message]], Awaitable[list[Message]]]:
    """Summarize old messages using incremental/rolling summarization.

    When len(memory) > threshold, older messages are summarized into a single
    message prefixed with a marker. On subsequent runs, the previous summary
    is detected and updated incrementally (only new overflow messages are
    sent to the summarizer), making this cheaper and more stable than
    full re-summarization.

    Calls _llm_router directly (not a full Agent) to avoid context
    save/restore overhead.

    System messages are always preserved and never included in summarization.

    Args:
        threshold: Trigger summarization when message count exceeds this.
        model: Model for summarization. Defaults to None, which uses the
            agent's own model. Override with a cheaper model like
            'openai/gpt-4.1-nano' to reduce cost.
        keep_last_n: Number of recent messages to keep unsummarized.
        max_summary_tokens: Maximum tokens for the summary response.

    Returns:
        An async compactor function. The returned function has a `_model`
        attribute that the agent sets to its own model before calling,
        used as fallback when model=None.
    """
    # Mutable state: the agent sets _compact._agent_model before calling
    _state = {"agent_model": None}

    async def _compact(memory: list[Message]) -> list[Message]:
        resolved_model = model or _state["agent_model"]
        if resolved_model is None:
            return memory  # Cannot summarize without a model
        if len(memory) <= threshold:
            return memory

        # Separate system messages — never summarize them
        system_messages = [m for m in memory if m.role == "system"]
        non_system = [m for m in memory if m.role != "system"]

        if len(non_system) <= threshold:
            return memory

        # Detect previous summary
        previous_summary = None
        start_idx = 0
        if non_system and non_system[0].collect_text().startswith(_SUMMARY_MARKER):
            full_text = non_system[0].collect_text()
            previous_summary = full_text[len(_SUMMARY_MARKER) :].strip()
            start_idx = 1  # Skip the summary message itself

        # Determine what to keep vs. what to summarize.
        # Apply orphan cleanup to to_keep: if the cut falls mid-tool-call-sequence,
        # the orphaned tool result at the boundary is removed. After cleanup,
        # to_keep[0] can only be "user" or "assistant", never "tool".
        to_keep = _remove_orphaned_tool_parts(non_system[-keep_last_n:]) if keep_last_n > 0 else []
        to_summarize = non_system[start_idx : len(non_system) - keep_last_n if keep_last_n > 0 else len(non_system)]

        if not to_summarize:
            return memory  # Nothing new to summarize

        # Format messages for the summarizer with tool call structure
        lines = []
        for msg in to_summarize:
            formatted = _format_message_for_summary(msg)
            if formatted:
                lines.append(formatted)

        if not lines:
            return memory

        new_messages_text = "\n".join(lines)

        # Build the prompt — incremental if we have a previous summary
        if previous_summary:
            prompt = _INCREMENTAL_SUMMARY_PROMPT.format(
                previous_summary=previous_summary,
                new_messages=new_messages_text,
            )
        else:
            prompt = _INITIAL_SUMMARY_PROMPT.format(
                messages=new_messages_text,
            )

        # Call the summarizer LLM
        summary_text = await _call_summarizer(resolved_model, prompt, max_summary_tokens)
        if summary_text is None:
            logger.warning("Summarizer returned no output; leaving memory unchanged.", model=resolved_model)
            return memory

        # Inject summary as first message with marker
        summary_msg = Message.validate({"role": "user", "content": f"{_SUMMARY_MARKER}\n{summary_text}"})

        # Strict alternation fix: some providers (e.g. Anthropic) reject consecutive
        # same-role messages. After orphan cleanup above, to_keep[0] is always "user"
        # or "assistant" (never "tool"). If it is "user", the summary(user) would land
        # immediately before another user message — insert a brief assistant acknowledgment
        # to bridge them. When to_keep[0] is "assistant" no ack is needed (and adding one
        # would create consecutive assistants instead). The ack is folded into the next
        # incremental summarization pass as "[assistant]: Understood." — negligible noise.
        injected = [summary_msg]
        if to_keep and to_keep[0].role == "user":
            injected.append(Message.validate({"role": "assistant", "content": "Understood."}))

        return system_messages + injected + to_keep

    _compact._state = _state  # Expose state so the agent can set the model
    return _compact
