"""Memory compaction strategies for reducing context window usage.

Apply compaction to conversation memory before sending to the LLM to avoid
exceeding context limits. Strategies can be composed (applied in order).
"""

import inspect
import json
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any

import structlog
from uuid_extensions import uuid7

from ..types.content import CustomContent, TextContent, ToolResultContent, ToolUseContent
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
_VERBATIM_MARKER = "[Verbatim User Messages]"
_TRANSCRIPT_MARKER = "[Compacted Transcripts]"
_NOTE_MARKER = "[Note]"
_REHYDRATED_MARKER = "[Rehydrated Context]"
_SECTION_MARKERS = (_SUMMARY_MARKER, _VERBATIM_MARKER, _TRANSCRIPT_MARKER, _NOTE_MARKER, _REHYDRATED_MARKER)
_VERBATIM_SEPARATOR = "\n----8<----\n"
_VERBATIM_HEADER = "The user's own messages from the compacted region, verbatim, oldest first:"
_VERBATIM_TRIMMED_NOTE = (
    "[Earlier user messages were dropped from this section; they are covered by the summary above.]"
)
_MAX_TRANSCRIPT_HANDLES = 5

# The continuation guidance is deliberately conservative: post-compaction "continue without
# asking" instructions are the most reported failure mode of lossy compaction (the model acts
# on a mischaracterized summary). The user's verbatim words always outrank the summary.
_CONTINUATION_NOTE = (
    "The earlier conversation was compacted into the summary above. Treat the user's explicit "
    "instructions (see the Verbatim User Messages section when present) as ground truth over "
    "the summary. Do not assume a next step the user has not explicitly requested; when "
    "uncertain about intent, ask the user before acting."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_ids(messages: list[Message], role: str, content_type: type) -> set[str]:
    """Collect all content ids of a given type from messages with the given role."""
    return {c.id for msg in messages if msg.role == role for c in msg.content if isinstance(c, content_type)}


def _collect_pinned_ids(memory: list[Message]) -> set[str]:
    """Collect the tool_use/result ids of every pinned tool result.

    A pinned result must never be dropped or truncated by compaction (see
    ``ToolResultContent.pinned``). The id is shared by the result and its paired tool_use, so
    this single set protects both halves."""
    return {
        c.id
        for msg in memory
        if msg.role == "tool"
        for c in msg.content
        if isinstance(c, ToolResultContent) and c.pinned
    }


def _pinned_protected_indices(memory: list[Message]) -> set[int]:
    """Indices of messages that must be preserved verbatim by positional compactors because
    they carry a pinned tool_result or the tool_use paired with one."""
    pinned_ids = _collect_pinned_ids(memory)
    if not pinned_ids:
        return set()
    keep: set[int] = set()
    for i, msg in enumerate(memory):
        if msg.role == "tool":
            if any(isinstance(c, ToolResultContent) and c.id in pinned_ids for c in msg.content):
                keep.add(i)
        elif msg.role == "assistant":
            if any(isinstance(c, ToolUseContent) and c.id in pinned_ids for c in msg.content):
                keep.add(i)
    return keep


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
            result.append(
                Message(role=msg.role, content=kept, stop_reason=msg.stop_reason, metadata=msg.metadata or None)
            )
        elif msg.role == "assistant":
            # Server tool use blocks (e.g. web search) are self-contained: both the
            # server_tool_use ToolUseContent and its paired result CustomContent live in
            # the same assistant message with no separate tool-role message — never treat
            # them as orphans. Conversely, drop any CustomContent whose tool_use_id has
            # no matching server_tool_use in this message.
            server_tool_ids = {c.id for c in msg.content if isinstance(c, ToolUseContent) and c.is_server_tool_use}
            kept = [
                c
                for c in msg.content
                if not isinstance(c, ToolUseContent) or c.id in valid_tool_ids or c.is_server_tool_use
            ]
            kept = [
                c
                for c in kept
                if not (
                    isinstance(c, CustomContent)
                    and isinstance(c.value, dict)
                    and "tool_use_id" in c.value
                    and c.value["tool_use_id"] not in server_tool_ids
                )
            ]
            if not kept:
                continue
            result.append(
                Message(role=msg.role, content=kept, stop_reason=msg.stop_reason, metadata=msg.metadata or None)
            )
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


def _format_message_for_transcript(msg: Message) -> str | None:
    """Format a message for the canonical-record transcript. Unlike
    ``_format_message_for_summary`` nothing is truncated — the transcript is the lossless
    record of what compaction removed, read back on demand via ``read_tool_result``.

    Results that were offloaded at production time hold only a placeholder inline; their
    payload lives in the offload store. The transcript records the handle from the
    structured ``offload_handle`` field — never from the placeholder prose, which a
    ``compact_tool_results(replacement=...)`` rewrite may have stripped — so the chain
    back to the full payload survives any placeholder mutation."""
    parts = []
    for c in msg.content:
        if isinstance(c, TextContent):
            parts.append(c.text)
        elif isinstance(c, ToolUseContent):
            input_str = json.dumps(c.input) if c.input else "{}"
            parts.append(f"[Called tool '{c.name}' with: {input_str}]")
        elif isinstance(c, ToolResultContent):
            result_text = "".join(item.text for item in c.content if isinstance(item, TextContent))
            if c.offload_handle:
                parts.append(
                    f"[Tool result for '{c.id}' (offloaded; full content: "
                    f'read_tool_result(handle="{c.offload_handle}")): {result_text}]'
                )
            else:
                parts.append(f"[Tool result for '{c.id}': {result_text}]")
    if parts:
        return f"[{msg.role}]: " + " ".join(parts)
    return None


def _extract_section(text: str, marker: str) -> str | None:
    """Return the content of a section: from after ``marker`` to the next section marker."""
    start = text.find(marker)
    if start == -1:
        return None
    start += len(marker)
    end = len(text)
    for other in _SECTION_MARKERS:
        if other == marker:
            continue
        pos = text.find(other, start)
        if pos != -1 and pos < end:
            end = pos
    return text[start:end].strip()


def _parse_summary_message(full_text: str) -> tuple[str, list[str], list[str]]:
    """Split a previous summary message into (summary, verbatim_entries, transcript_handles).

    The verbatim and transcript sections are mechanical (never produced by the LLM), so they
    are carried forward structurally rather than re-fed through the summarizer. The note and
    rehydrated sections are regenerated on every pass and ignored here.
    """
    summary = _extract_section(full_text, _SUMMARY_MARKER) or ""
    verbatim_section = _extract_section(full_text, _VERBATIM_MARKER) or ""
    # Drop the boilerplate header/trim-note lines, then split into entries.
    verbatim_section = "\n".join(
        line for line in verbatim_section.splitlines() if line not in (_VERBATIM_HEADER, _VERBATIM_TRIMMED_NOTE)
    )
    verbatim_entries = [e.strip() for e in verbatim_section.split(_VERBATIM_SEPARATOR) if e.strip()]
    transcript_section = _extract_section(full_text, _TRANSCRIPT_MARKER) or ""
    handles = [line[2:].strip() for line in transcript_section.splitlines() if line.startswith("- ")]
    return summary, verbatim_entries, handles


def _build_summary_message_text(
    summary: str,
    verbatim_entries: list[str],
    handles: list[str],
    rehydrated: str | None,
    max_verbatim_chars: int,
) -> str:
    parts = [f"{_SUMMARY_MARKER}\n{summary}"]

    if verbatim_entries:
        entries = list(verbatim_entries)
        trimmed = False
        while len(entries) > 1 and sum(len(e) for e in entries) > max_verbatim_chars:
            entries.pop(0)  # drop oldest first — the newest instructions matter most
            trimmed = True
        if entries and len(entries[0]) > max_verbatim_chars:
            # A single oversized message: clamp head+tail rather than dropping it entirely.
            half = max_verbatim_chars // 2
            entries[0] = entries[0][:half] + "\n[... middle elided ...]\n" + entries[0][-half:]
            trimmed = True
        lines = [_VERBATIM_HEADER]
        if trimmed:
            lines.append(_VERBATIM_TRIMMED_NOTE)
        parts.append(f"{_VERBATIM_MARKER}\n" + "\n".join(lines) + "\n" + _VERBATIM_SEPARATOR.join(entries))

    if handles:
        kept_handles = handles[-_MAX_TRANSCRIPT_HANDLES:]
        parts.append(
            f"{_TRANSCRIPT_MARKER}\n"
            "Full transcripts of the compacted messages were saved. Read them with "
            'read_tool_result(handle="..."):\n' + "\n".join(f"- {h}" for h in kept_handles)
        )

    parts.append(f"{_NOTE_MARKER}\n{_CONTINUATION_NOTE}")

    if rehydrated:
        parts.append(f"{_REHYDRATED_MARKER}\n{rehydrated}")

    return "\n\n".join(parts)


_SUMMARY_RULES = """\
Rules:
- Write the summary in the same language the user writes in.
- Only mention tools that actually appear in the messages.
- Never state or imply an instruction, request, or decision the user did not explicitly make.
- If the user's latest instruction is conditional (e.g. "review first, then implement"), \
preserve the condition exactly — never collapse it into an unconditional next step."""


_INITIAL_SUMMARY_PROMPT = f"""\
Summarize the following conversation, preserving:
1. All specific values, identifiers, names, URLs, dates, and numbers mentioned
2. The outcome of every tool call (what tool was called, what it returned)
3. User preferences, decisions, and explicit instructions
4. Any constraints or requirements established

{_SUMMARY_RULES}

Conversation:
{{messages}}

Provide a structured summary using this format:

## Key Facts & Decisions
- [bullet points of decisions, preferences, established facts]

## Tool Outcomes
- [tool_name]: [what it returned / key result]

## Flow
- [brief chronological narrative of the conversation progression]"""

_INCREMENTAL_SUMMARY_PROMPT = f"""\
Update this conversation summary with new messages.

Current summary:
{{previous_summary}}

New messages since last summary:
{{new_messages}}

Update the summary to incorporate the new messages. You must:
1. Preserve all specific values, identifiers, names, URLs, dates, and numbers
2. Record the outcome of every tool call (what tool was called, what it returned)
3. Capture user preferences, decisions, and explicit instructions
4. Drop information that has been superseded by newer messages
5. Keep the summary concise but complete

{_SUMMARY_RULES}

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
    from .llm import _llm_router

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
    keep_offloaded: bool = True,
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
            String templates additionally support ``{handle}`` — the offload handle
            when the result was offloaded (empty otherwise).
        keep_offloaded: If True (default), results already offloaded at production
            time (see ``timbal.core.tool_result_offload``) are kept intact: they are
            small placeholders whose handle keeps the full payload reachable. Set
            False to compact them like any other result.

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

        # Pinned results (and their paired tool_use) are never dropped or replaced.
        kept_ids |= _collect_pinned_ids(memory)

        # Offloaded results are already-compacted placeholders; keep them (and their paired
        # tool_use) so their handles stay dereferenceable, unless the caller opts out.
        if keep_offloaded:
            kept_ids |= {
                c.id
                for msg in memory
                if msg.role == "tool"
                for c in msg.content
                if isinstance(c, ToolResultContent) and c.offload_handle
            }

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
                            result_text = "".join(item.text for item in c.content if isinstance(item, TextContent))
                            if callable(replacement):
                                placeholder = replacement(tool_name, c.id, result_text)
                            else:
                                placeholder = replacement.format_map(
                                    defaultdict(
                                        str,
                                        tool_name=tool_name,
                                        call_id=c.id,
                                        result_length=str(len(result_text)),
                                        handle=c.offload_handle or "",
                                    )
                                )
                            new_content.append(
                                ToolResultContent(
                                    id=c.id,
                                    content=[TextContent(text=placeholder)],
                                    offload_handle=c.offload_handle,
                                )
                            )
                    else:
                        new_content.append(c)
                if not new_content:
                    continue
                result.append(
                    Message(
                        role=msg.role,
                        content=new_content,
                        stop_reason=msg.stop_reason,
                        metadata=msg.metadata or None,
                    )
                )
            elif msg.role == "assistant":
                if drop_mode:
                    # Collect IDs of server tool use blocks that are being dropped so we
                    # can also remove their paired result blocks (CustomContent with a
                    # matching tool_use_id). Both live in the same assistant message and
                    # must always travel together, regardless of the result block type.
                    dropped_server_ids = {
                        c.id
                        for c in msg.content
                        if isinstance(c, ToolUseContent) and c.is_server_tool_use and c.id not in kept_ids
                    }
                    kept = [c for c in msg.content if not isinstance(c, ToolUseContent) or c.id in kept_ids]
                    if dropped_server_ids:
                        kept = [
                            c
                            for c in kept
                            if not (
                                isinstance(c, CustomContent)
                                and isinstance(c.value, dict)
                                and c.value.get("tool_use_id") in dropped_server_ids
                            )
                        ]
                    if not kept:
                        continue
                    result.append(
                        Message(
                            role=msg.role,
                            content=kept,
                            stop_reason=msg.stop_reason,
                            metadata=msg.metadata or None,
                        )
                    )
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
        # Keep the last n messages, plus any pinned messages (and their paired tool_use) that
        # fell outside the window — "last n PLUS pinned". Order is preserved.
        keep = set(range(len(memory) - n, len(memory))) | _pinned_protected_indices(memory)
        return _remove_orphaned_tool_parts([memory[i] for i in sorted(keep)])

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
        # Runtime control messages (e.g. background completion notices) are role=user
        # for the LLM wire but must not count as turn boundaries.
        user_indices = [i for i, m in enumerate(memory) if m.role == "user" and not m.is_runtime()]
        if len(user_indices) <= n:
            return _remove_orphaned_tool_parts(memory)
        start = user_indices[-n]
        # Pull contiguous runtime notices that sit immediately before the first kept
        # human user message so [...notice, user...] stays intact.
        while start > 0 and memory[start - 1].is_runtime():
            start -= 1
        # Keep the last n turns, plus any pinned messages (and their paired tool_use) from
        # earlier turns. Order is preserved.
        keep = set(range(start, len(memory))) | _pinned_protected_indices(memory)
        return _remove_orphaned_tool_parts([memory[i] for i in sorted(keep)])

    return _compact


def summarize(
    threshold: int,
    model: Any | None = None,
    keep_last_n: int = 4,
    max_summary_tokens: int = 500,
    preserve_user_messages: bool = True,
    max_verbatim_chars: int = 10_000,
    store: Any | None = None,
    canonical_record: bool = True,
    rehydrate: Callable[[], Any] | None = None,
) -> Callable[[list[Message]], Awaitable[list[Message]]]:
    """Summarize old messages using incremental/rolling summarization.

    When len(memory) > threshold, older messages are summarized into a single
    message prefixed with a marker. On subsequent runs, the previous summary
    is detected and updated incrementally (only new overflow messages are
    sent to the summarizer), making this cheaper and more stable than
    full re-summarization.

    The summary message is structured in sections. Beyond the LLM summary itself,
    all sections are mechanical (never paraphrased by the LLM):

    - Verbatim User Messages: the user's own words from the summarized region are
      carried forward verbatim (``preserve_user_messages``). Summaries that drop or
      mischaracterize user instructions are the most damaging compaction failure —
      the user's words are ground truth, so they are never trusted to the summarizer.
    - Compacted Transcripts: when a store is available (``store=``, or shared from the
      agent's ``tool_result_limit`` offload store), the full text of every summarized
      region is persisted and its handle listed, readable via ``read_tool_result``
      (``canonical_record``). Summarization thus becomes recoverable, not destructive.
    - Note: conservative continuation guidance — the model is told to verify intent
      against the user's words instead of barreling ahead on the summary.
    - Rehydrated Context: output of the ``rehydrate`` callable, regenerated on every
      compaction pass (e.g. re-read the files being worked on, re-inject a plan).

    Calls _llm_router directly (not a full Agent) to avoid context
    save/restore overhead.

    System messages are always preserved and never included in summarization.

    Args:
        threshold: Trigger summarization when message count exceeds this.
        model: Model for summarization. Defaults to None, which uses the
            agent's own model. Override with a cheaper model like
            'openai/gpt-5.4-nano' to reduce cost.
        keep_last_n: Number of recent messages to keep unsummarized.
        max_summary_tokens: Maximum tokens for the summary response.
        preserve_user_messages: Carry the user's messages from the summarized region
            verbatim in the summary message (default True). Runtime control messages
            (``Message.is_runtime()``) are excluded — they are not human utterances.
        max_verbatim_chars: Budget for the verbatim section; oldest entries are
            dropped first when exceeded.
        store: OffloadStore for the canonical record. Defaults to None, which uses
            the agent's offload store when one exists (see Agent.tool_result_limit).
        canonical_record: Persist the full text of summarized messages to the store
            (default True; skipped silently when no store is available).
        rehydrate: Optional parameterless callable (sync or async) returning str,
            list[str], or None — extra context re-injected after every summarization.

    Returns:
        An async compactor function. The returned function has a `_state`
        attribute that the agent sets (model/store) before calling.
    """
    # Mutable state: the agent injects its model and offload store before calling.
    _state = {"agent_model": None, "store": store}

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
        verbatim_entries: list[str] = []
        transcript_handles: list[str] = []
        start_idx = 0
        if non_system and non_system[0].collect_text().startswith(_SUMMARY_MARKER):
            full_text = non_system[0].collect_text()
            previous_summary, verbatim_entries, transcript_handles = _parse_summary_message(full_text)
            start_idx = 1  # Skip the summary message itself

        # Determine what to keep vs. what to summarize.
        # Keep the last keep_last_n messages, PLUS any pinned messages (and their paired
        # tool_use) from the older region — pinned context is preserved verbatim, like system
        # messages, never fed to the summarizer. Apply orphan cleanup to to_keep: if the cut
        # falls mid-tool-call-sequence, the orphaned tool result at the boundary is removed.
        # After cleanup, to_keep[0] can only be "user" or "assistant", never "tool".
        protected_idx = {i for i in _pinned_protected_indices(non_system) if i >= start_idx}
        keep_idx = set(range(len(non_system) - keep_last_n, len(non_system))) if keep_last_n > 0 else set()
        keep_idx |= protected_idx
        keep_idx = {i for i in keep_idx if i >= start_idx}
        to_keep = _remove_orphaned_tool_parts([non_system[i] for i in sorted(keep_idx)])
        to_summarize = [non_system[i] for i in range(start_idx, len(non_system)) if i not in keep_idx]

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

        # Mechanical sections — assembled by us, never trusted to the summarizer.
        if preserve_user_messages:
            for msg in to_summarize:
                if msg.role != "user" or msg.is_runtime():
                    continue
                text = msg.collect_text().strip()
                if text and not text.startswith(_SUMMARY_MARKER):
                    verbatim_entries.append(text)

        if canonical_record:
            resolved_store = store or _state.get("store")
            if resolved_store is not None:
                transcript = "\n".join(
                    formatted for msg in to_summarize if (formatted := _format_message_for_transcript(msg))
                )
                try:
                    from ..state import get_run_context

                    run_context = get_run_context()
                    run_id = run_context.id if run_context is not None else uuid7(as_type="hex")
                    handle = await resolved_store.write(
                        f"{run_id}/compaction-{uuid7(as_type='hex')}", transcript.encode()
                    )
                    transcript_handles.append(handle)
                except Exception:
                    logger.exception("Failed to persist canonical record of summarized messages; continuing.")

        rehydrated = None
        if rehydrate is not None:
            try:
                rehydrated_value = rehydrate()
                if inspect.isawaitable(rehydrated_value):
                    rehydrated_value = await rehydrated_value
                if isinstance(rehydrated_value, str):
                    rehydrated = rehydrated_value or None
                elif isinstance(rehydrated_value, list):
                    rehydrated = "\n\n".join(str(v) for v in rehydrated_value if v) or None
                elif rehydrated_value is not None:
                    rehydrated = str(rehydrated_value)
            except Exception:
                logger.exception("Rehydrate callable failed; continuing without rehydrated context.")

        # Inject summary as first message with marker
        summary_msg = Message.validate(
            {
                "role": "user",
                "content": _build_summary_message_text(
                    summary_text, verbatim_entries, transcript_handles, rehydrated, max_verbatim_chars
                ),
            }
        )

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
