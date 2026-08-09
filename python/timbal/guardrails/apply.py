"""Helpers that apply guardrail outcomes to Timbal messages, spans, and event streams.

Kept separate from ``runner.py`` because these import ``timbal.types`` (messages,
events); the runner stays framework-agnostic text-in/text-out.
"""

import time
from typing import Any

from ..types.content import TextContent, ToolResultContent
from ..types.events.guardrail import GuardrailEvent
from ..types.message import Message
from .runner import TriggerRecord

__all__ = [
    "build_guardrail_events",
    "message_text",
    "record_guardrails_metadata",
    "replace_message_text",
    "replace_tool_result_text",
    "tool_result_text",
]


def message_text(message: Message) -> str:
    """Concatenated text content of a message (what rails check)."""
    return "".join(c.text for c in message.content if isinstance(c, TextContent))


def replace_message_text(message: Message, new_text: str) -> None:
    """Swap a message's text for ``new_text`` in place, preserving non-text content.

    The first text block carries the replacement; further text blocks are dropped
    (they were part of the same checked text).
    """
    new_content: list[Any] = []
    replaced = False
    for c in message.content:
        if isinstance(c, TextContent):
            if not replaced:
                new_content.append(TextContent(text=new_text))
                replaced = True
            continue
        new_content.append(c)
    if not replaced:
        new_content.append(TextContent(text=new_text))
    message.content = new_content


def tool_result_text(content: ToolResultContent) -> str:
    return "".join(item.text for item in content.content if isinstance(item, TextContent))


def replace_tool_result_text(content: ToolResultContent, new_text: str) -> None:
    """Swap a tool result's text in place, preserving non-text items (files)."""
    new_items: list[Any] = []
    replaced = False
    for item in content.content:
        if isinstance(item, TextContent):
            if not replaced:
                new_items.append(TextContent(text=new_text))
                replaced = True
            continue
        new_items.append(item)
    if not replaced:
        new_items.append(TextContent(text=new_text))
    content.content = new_items


def build_guardrail_events(
    records: list[TriggerRecord],
    *,
    run_context: Any,
    span: Any,
) -> list[GuardrailEvent]:
    """Turn trigger records into stream events carrying the span's identity."""
    return [
        GuardrailEvent(
            run_id=run_context.id,
            parent_run_id=run_context.parent_id,
            path=span.path,
            call_id=span.call_id,
            parent_call_id=span.parent_call_id,
            rail=r.rail,
            stage=r.stage,
            action=r.action,
            reason=r.reason,
            latency_ms=r.latency_ms,
            shadow=r.shadow,
            metadata=dict(r.metadata),
        )
        for r in records
    ]


def record_guardrails_metadata(span: Any, records: list[TriggerRecord], *, run_context: Any = None) -> None:
    """Append trigger records to the span's per-run guardrail report and usage counters.

    The report lands on ``span.metadata["guardrails"]`` and therefore on the final
    ``OutputEvent.metadata`` — a per-run audit trail of every rail that fired.
    """
    if not records:
        return
    report = span.metadata.setdefault("guardrails", {"triggered": []})
    now_ms = int(time.time() * 1000)
    for r in records:
        report["triggered"].append({**r.as_dict(), "t": now_ms})
    if run_context is not None:
        enforced = sum(1 for r in records if not r.shadow and r.action != "error")
        shadowed = sum(1 for r in records if r.shadow)
        if enforced:
            run_context.update_usage("guardrails:triggered", enforced)
        if shadowed:
            run_context.update_usage("guardrails:shadow_triggered", shadowed)
