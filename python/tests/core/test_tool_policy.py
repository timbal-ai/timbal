"""``timbal.core.tool_policy``: restoring a callable policy from a trace."""

import asyncio

from timbal import Agent, Tool
from timbal.core.test_model import TestModel
from timbal.core.tool_policy import allowed_tool_paths
from timbal.state import set_run_context
from timbal.state.context import RunContext
from timbal.state.tracing.providers import InMemoryTracingProvider
from timbal.types.content import ToolUseContent
from timbal.types.message import Message


class TestAllowedToolPaths:
    async def test_concurrent_calls_restore_the_latest_decision_not_the_latest_started_call(self):
        """``slow`` starts first but decides last (after taking ``secret`` back): its decision is the one kept."""
        revoked: set[str] = set()
        fast_done = asyncio.Event()

        inner = Agent(
            name="inner",
            model=TestModel(responses=["ok"]),
            tools=[Tool(name="secret", handler=lambda: "s"), Tool(name="lookup", handler=lambda: "l")],
            tracing_provider=InMemoryTracingProvider,
        )

        async def slow() -> str:
            await fast_done.wait()
            revoked.add("secret")
            await inner(prompt="x").collect()
            return "slow"

        async def fast() -> str:
            await inner(prompt="x").collect()
            fast_done.set()
            return "fast"

        def handler(messages):
            if messages[-1].role == "tool":
                return "done"
            return Message(
                role="assistant",
                content=[
                    ToolUseContent(id="call-slow", name="slow", input={}),
                    ToolUseContent(id="call-fast", name="fast", input={}),
                ],
                stop_reason="tool_use",
            )

        root = Agent(
            name="root",
            model=TestModel(handler=handler),
            tools=[Tool(name="slow", handler=slow), Tool(name="fast", handler=fast)],
            tracing_provider=InMemoryTracingProvider,
        )
        context = RunContext(tracing_provider=InMemoryTracingProvider)
        context._tool_policy = lambda tool: tool.name not in revoked
        set_run_context(context)
        try:
            result = await root(prompt="go").collect()
        finally:
            set_run_context(None)
        assert result.status.code == "success"

        nested = {
            span.path: span.metadata["nested_tool_policy_allowed"]
            for span in context._trace.values()
            if "nested_tool_policy_allowed" in span.metadata
        }
        assert nested["root.slow"]["inner"]["allowed"] == ["inner.lookup"]
        assert nested["root.fast"]["inner"]["allowed"] == ["inner.lookup", "inner.secret"]
        assert nested["root.slow"]["inner"]["at"] > nested["root.fast"]["inner"]["at"]
        assert "inner.secret" not in allowed_tool_paths(context._trace)
        assert "inner.lookup" in allowed_tool_paths(context._trace)
