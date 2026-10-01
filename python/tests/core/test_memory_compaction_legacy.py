"""Compatibility with persisted conversations written before context measurements."""

from pathlib import Path

import pytest
from timbal import Agent
from timbal.core.memory_compaction import keep_last_n_turns
from timbal.core.test_model import TestModel
from timbal.state import get_run_context, set_run_context
from timbal.state.context import RunContext
from timbal.state.tracing.providers import InMemoryTracingProvider
from timbal.state.tracing.trace import Trace
from timbal.types.content import FileContent, TextContent
from timbal.types.file import File
from timbal.types.message import Message


@pytest.mark.parametrize(
    ("usage", "new_prompt", "should_compact"),
    [
        ({"test/model:input_text_tokens": 80_000}, "continue", True),
        ({"test/model:input_text_tokens": 40_000}, "x" * 160_000, True),
        ({}, "x" * 320_000, True),
        ({"test/model:web_search_requests": 1}, "x" * 320_000, True),
        ({"test/model:web_search_requests": 1}, "continue", False),
        ({"test/model:input_text_tokens": 5_000}, "continue", False),
    ],
    ids=["file-usage", "usage-plus-new-content", "no-usage", "request-only-large", "request-only-small", "low-usage"],
)
async def test_legacy_trace_compaction(monkeypatch, usage, new_prompt, should_compact):
    monkeypatch.setattr("timbal.core.agent.get_context_window", lambda _: 100_000)
    compactions = []
    inner = keep_last_n_turns(1)

    def compact(memory):
        compactions.append(True)
        return inner(memory)

    agent = Agent(name="legacy", model=TestModel(), memory_compaction=compact)
    ctx = RunContext(tracing_provider=InMemoryTracingProvider)
    set_run_context(ctx)
    prompt = Message(
        role="user",
        content=[
            TextContent(text="Read this document"),
            FileContent(file=File.validate(str(Path(__file__).parents[1] / "fixtures/test.pdf"))),
        ],
    )
    assert (await agent(prompt=prompt).collect()).status.code == "success"

    # Simulate provider-reported usage, not the token size of the small PDF fixture.
    # Round-trip through the serialized trace shape used by persistent providers.
    records = ctx._trace.model_dump()
    for record in records:
        if record["path"] in (agent._path, agent._llm._path):
            record["usage"] = dict(usage)
        record["metadata"].pop("context_measurement", None)
    InMemoryTracingProvider._storage[ctx.id] = Trace(records)

    next_ctx = RunContext(parent_id=ctx.id, tracing_provider=InMemoryTracingProvider)
    set_run_context(next_ctx)
    assert (await agent(prompt=new_prompt).collect()).status.code == "success"
    assert bool(compactions) is should_compact
    if should_compact:
        assert not any(isinstance(c, FileContent) for m in next_ctx.root_span().memory for c in m.content)

    if usage.get("test/model:input_text_tokens") == 80_000:
        # A successful measured call retires the conservative legacy estimate.
        # The following small turn must not keep compacting against the old 80k total.
        await next_ctx._save_trace()
        compactions.clear()
        set_run_context(RunContext(parent_id=next_ctx.id, tracing_provider=InMemoryTracingProvider))
        assert (await agent(prompt="one more question").collect()).status.code == "success"
        assert not compactions


async def test_failed_first_call_is_not_treated_as_a_legacy_trace(monkeypatch):
    monkeypatch.setattr("timbal.core.agent.get_context_window", lambda _: 100_000)
    calls = 0
    compactions = []

    def respond(_messages):
        nonlocal calls
        calls += 1
        if calls == 1:
            get_run_context().update_usage("test/model:input_text_tokens", 40_000)
            raise RuntimeError("stream interrupted before output")
        return "done"

    def compact(memory):
        compactions.append(True)
        return memory

    agent = Agent(name="interrupted", model=TestModel(handler=respond), memory_compaction=compact)
    ctx = RunContext(tracing_provider=InMemoryTracingProvider)
    set_run_context(ctx)
    assert (await agent(prompt="x" * 160_000).collect()).status.code == "error"
    # Persist the explicit absence of a measurement across a trace round-trip.
    InMemoryTracingProvider._storage[ctx.id] = Trace(ctx._trace.model_dump())
    set_run_context(RunContext(parent_id=ctx.id, tracing_provider=InMemoryTracingProvider))
    assert (await agent(prompt="continue").collect()).status.code == "success"
    assert not compactions, "Do not add the failed request's 40k input to the same 40k content again"
