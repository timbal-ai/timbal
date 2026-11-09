"""Python bytecode instructions the framework executes per run, for a fixed set of cases.

    uv run python benchmarks/instructions/count.py            # JSON on stdout

Counts every instruction executed while the case runs (``sys.monitoring``, Python 3.12+), averaged over
several runs after a warm-up. Unlike wall time this does not depend on machine load, so two revisions can be
compared exactly; ``gate.py`` does that. Only public API is used, so the same file measures any revision put
first on ``PYTHONPATH``.

The model is a ``TestModel`` that replies from a script and reports fixed usage: ``TestModel`` itself
estimates tokens from ``str()`` of every content block, which would charge each new content field to the
framework.
"""

# ruff: noqa: T201 - prints its report
from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Callable
from typing import Any

from timbal import Agent, Tool, Workflow
from timbal.core.test_model import TestModel
from timbal.state import get_run_context, set_run_context
from timbal.state.tracing.providers import InMemoryTracingProvider
from timbal.types.content import TextContent, ToolUseContent
from timbal.types.message import Message

WARMUP = 20
RUNS = 20


class ScriptedModel(TestModel):
    """Replies with ``script(messages)``; usage is a constant, so no work scales with the content."""

    def __init__(self, script: Callable[[list[Message]], Message | str]) -> None:
        super().__init__(handler=script)

    async def stream(self, messages: list, _output_model: Any = None, **_kwargs: Any):
        from timbal.collectors import _collector_registry
        from timbal.collectors.impl.message import MessageCollector

        if not TestModel._collector_registered:
            _collector_registry.register(MessageCollector)
            TestModel._collector_registered = True
        self.call_count += 1
        raw = self.handler(messages)
        if isinstance(raw, str):
            raw = Message(role="assistant", content=[TextContent(text=raw)], stop_reason="end_turn")
        get_run_context().update_usage("test/model:input_text_tokens", 100)
        get_run_context().update_usage("test/model:output_text_tokens", 10)
        yield raw


def _calls(*names: str, start: int = 0) -> Message:
    return Message(
        role="assistant",
        content=[ToolUseContent(id=f"c{start + i}", name=n, input={"a": 2, "b": 3}) for i, n in enumerate(names)],
        stop_reason="tool_use",
    )


def _tool_results(messages: list[Message]) -> int:
    return sum(1 for m in messages if m.role == "tool")


def add(a: int, b: int) -> int:
    return a + b


def mul(a: int, b: int) -> int:
    return a * b


def sub(a: int, b: int) -> int:
    return a - b


def _agent(script: Callable[[list[Message]], Message | str], tools: list, max_iter: int = 10) -> Agent:
    return Agent(
        name="bench",
        model=ScriptedModel(script),
        tools=tools,
        max_iter=max_iter,
        tracing_provider=InMemoryTracingProvider,
    )


def _one_tool(messages: list[Message]) -> Message | str:
    return _calls("add") if _tool_results(messages) == 0 else "5"


def _three_sequential(messages: list[Message]) -> Message | str:
    done = _tool_results(messages)
    return _calls(("add", "mul", "sub")[done], start=done) if done < 3 else "done"


def _three_parallel(messages: list[Message]) -> Message | str:
    return _calls("add", "mul", "sub") if _tool_results(messages) == 0 else "done"


LOOP = 20


def _loop(messages: list[Message]) -> Message | str:
    done = _tool_results(messages)
    return _calls("add", start=done) if done < LOOP else "done"


def _fanout() -> Workflow:
    def source(x: int) -> int:
        return x

    def left(v: int) -> int:
        return v + 1

    def right(v: int) -> int:
        return v * 2

    def join(a: int, b: int) -> int:
        return a + b

    return (
        Workflow(name="bench_wf", tracing_provider=InMemoryTracingProvider)
        .step(source)
        .step(left, v=lambda: get_run_context().step_span("source").output)
        .step(right, v=lambda: get_run_context().step_span("source").output)
        .step(
            join,
            a=lambda: get_run_context().step_span("left").output,
            b=lambda: get_run_context().step_span("right").output,
        )
    )


CASES: dict[str, tuple[Callable[[], Any], dict[str, Any]]] = {
    "agent_one_tool": (lambda: _agent(_one_tool, [add]), {"prompt": "go"}),
    "agent_three_sequential": (lambda: _agent(_three_sequential, [add, mul, sub]), {"prompt": "go"}),
    "agent_three_parallel": (lambda: _agent(_three_parallel, [add, mul, sub]), {"prompt": "go"}),
    "agent_loop_20": (lambda: _agent(_loop, [add], max_iter=LOOP + 5), {"prompt": "go"}),
    "workflow_fanout": (_fanout, {"x": 5}),
    "tool_call": (lambda: Tool(name="add", handler=add, tracing_provider=InMemoryTracingProvider), {"a": 1, "b": 2}),
}


async def _run(runnable: Any, kwargs: dict[str, Any]) -> None:
    set_run_context(None)
    out = await runnable(**kwargs).collect()
    assert out.status.code == "success", (out.status, out.error)


async def _count(build: Callable[[], Any], kwargs: dict[str, Any]) -> float:
    runnable = build()
    for _ in range(WARMUP):
        await _run(runnable, kwargs)
    InMemoryTracingProvider._storage.clear()
    executed = 0

    def on_instruction(_code: Any, _offset: int) -> None:
        nonlocal executed
        executed += 1

    monitoring = sys.monitoring
    tool_id = monitoring.PROFILER_ID
    monitoring.use_tool_id(tool_id, "timbal-instructions")
    monitoring.register_callback(tool_id, monitoring.events.INSTRUCTION, on_instruction)
    try:
        monitoring.set_events(tool_id, monitoring.events.INSTRUCTION)
        for _ in range(RUNS):
            await _run(runnable, kwargs)
    finally:
        monitoring.set_events(tool_id, 0)
        monitoring.free_tool_id(tool_id)
        InMemoryTracingProvider._storage.clear()
    return executed / RUNS


async def main() -> None:
    if sys.version_info < (3, 12):
        raise SystemExit("sys.monitoring needs Python 3.12+")
    import timbal

    results = {name: await _count(build, kwargs) for name, (build, kwargs) in CASES.items()}
    print(json.dumps({"timbal": timbal.__file__, "python": sys.version.split()[0], "instructions": results}))


if __name__ == "__main__":
    asyncio.run(main())
