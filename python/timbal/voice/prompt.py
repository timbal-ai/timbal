"""Cache-friendly voice system prompts.

The suffix after the static instructions is, in order: caller and customer
fields (stable for the call), the ``voice_prompt`` dynamic text, the UTC
time for this turn, the already-greeted note, then the reply-language
line. OpenAI and Groq cache an exact prefix, so per-call lines have to
precede per-turn lines or the caller block falls out of the hit. The
language line is last so a soft "match the caller" earlier in the prompt
does not outrank it. Anthropic and Bedrock put the cache breakpoint on the
static block; the suffix is a following block with no breakpoint.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime

from ..core.llm.prompt_cache import SystemPrompt, voice_prompt

__all__ = ["compose_voice_system_prompt", "voice_prompt"]


def compose_voice_system_prompt(
    resolved: str | SystemPrompt | None,
    *,
    call_context: Mapping[str, str] | None,
    now: datetime,
    greeting_note: str | None = None,
    continuation_note: str | None = None,
    reply_language: str | None = None,
) -> SystemPrompt:
    """Static instructions, then per-call lines, then per-turn lines.

    Order of the suffix: caller and customer fields, the caller's ``dynamic``
    text, the UTC clock, the already-greeted note, the reply-language line.
    ``now`` is captured once per turn so a tool loop does not move the clock
    between LLM calls. Caller and customer lines are sorted so the same call
    renders the same bytes. The clock, the greeting note, and the language
    line change between turns, so they come after the per-call lines. The
    language line is last.
    """
    if isinstance(resolved, SystemPrompt):
        static = resolved.static
        user_dynamic = resolved.dynamic
    elif isinstance(resolved, str):
        static = resolved
        user_dynamic = ""
    else:
        static = ""
        user_dynamic = ""

    facts = _caller_and_customer(call_context or {})
    clock = f"Current date and time (UTC): {_utc_stamp(now)}"
    # Caller and customer are stable for the call, so they stay ahead of the
    # clock. OpenAI and Groq stop the cached prefix at the first changed byte.
    dynamic_parts = [
        part
        for part in (facts, user_dynamic, clock, greeting_note or "", continuation_note or "", reply_language or "")
        if part
    ]
    return SystemPrompt(static=static, dynamic="\n\n".join(dynamic_parts))


def _utc_stamp(now: datetime) -> str:
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    return now.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _caller_and_customer(call_context: Mapping[str, str]) -> str:
    lines: list[str] = []
    for key in sorted(call_context):
        value = call_context[key]
        if isinstance(value, str) and value:
            lines.append(f"{key}: {value}")
    if not lines:
        return ""
    return "Caller and customer:\n" + "\n".join(lines)
