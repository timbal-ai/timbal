"""Minimal LLM call for judgment rails (classifier, topic, judge).

Calls the model router directly — no Tool/Agent overhead, mirroring
``timbal.core.memory_compaction._call_summarizer``. All ``timbal.core`` imports happen
inside the function so this package never creates an import cycle at module load.
"""

from typing import Any

import structlog

logger = structlog.get_logger("timbal.guardrails.judge_llm")

__all__ = ["call_judge"]


async def call_judge(
    *,
    model: Any,
    system_prompt: str,
    prompt: str,
    max_tokens: int = 256,
) -> str | None:
    """Run one deterministic (temperature 0) LLM call and return its text, or None.

    Callers must treat ``None`` as "no usable answer" and fail open/closed per their
    own ``strict`` policy — this helper never raises for empty output.
    """
    import time

    from ..collectors import get_collector_registry
    from ..core.llm import _llm_router
    from ..types.message import Message

    prompt_message = Message.validate({"role": "user", "content": prompt})
    chunks = _llm_router(
        model=model,
        messages=[prompt_message],
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=0.0,
    )

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
