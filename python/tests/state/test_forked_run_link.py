"""An Agent started by a running call's code (inside a tool or a step body) forks its
own run. ``parent_id`` stays the turn chain; ``invoked_by`` on the child's root span
and ``invoked_runs`` on the caller's span link the two runs both ways.
"""

from timbal import Agent, Workflow
from timbal.core import Tool
from timbal.core.test_model import TestModel
from timbal.state import get_call_id, get_run_context
from timbal.state.tracing.providers import InMemoryTracingProvider
from timbal.types.content import ToolUseContent
from timbal.types.message import Message


def _inner(observed: dict) -> Agent:
    def capture() -> None:
        observed["inner_ctx"] = get_run_context()

    return Agent(
        name="inner",
        model=TestModel(responses=["done"]),
        pre_hook=capture,
        tracing_provider=InMemoryTracingProvider,
    )


def _calls(tool: str, input: dict) -> TestModel:
    return TestModel(
        responses=[
            Message(
                role="assistant",
                content=[ToolUseContent(id="call_1", name=tool, input=input)],
                stop_reason="tool_use",
            ),
            "finished",
        ]
    )


class TestForkedRunLink:
    async def test_an_agent_inside_a_tool_links_to_the_tool_call(self):
        observed: dict = {}
        inner = _inner(observed)

        async def ask_inner() -> str:
            observed["caller_ctx"] = get_run_context()
            observed["caller_call_id"] = get_call_id()
            result = await inner(prompt="hi").collect()
            return result.output.collect_text()

        outer = Agent(
            name="outer",
            model=_calls("ask_inner", {}),
            tools=[Tool(name="ask_inner", handler=ask_inner)],
            tracing_provider=InMemoryTracingProvider,
        )
        result = await outer(prompt="go").collect()
        assert result.status.code == "success", result.error

        caller_ctx, inner_ctx = observed["caller_ctx"], observed["inner_ctx"]
        assert inner_ctx.id != caller_ctx.id
        assert inner_ctx.parent_id is None  # not a turn of the caller's conversation
        tool_span = caller_ctx._trace[observed["caller_call_id"]]
        assert tool_span.path == "outer.ask_inner"
        assert inner_ctx.root_span().metadata["invoked_by"] == {"run_id": caller_ctx.id, "call_id": tool_span.call_id}
        assert tool_span.metadata["invoked_runs"] == [inner_ctx.id]

    async def test_an_agent_inside_a_step_body_links_to_the_step(self):
        observed: dict = {}
        inner = _inner(observed)

        async def step_handler() -> str:
            observed["caller_ctx"] = get_run_context()
            observed["caller_call_id"] = get_call_id()
            result = await inner(prompt="hi").collect()
            return result.output.collect_text()

        workflow = Workflow(name="wf", tracing_provider=InMemoryTracingProvider).step(
            Tool(name="step", handler=step_handler)
        )
        result = await workflow().collect()
        assert result.status.code == "success", result.error

        caller_ctx, inner_ctx = observed["caller_ctx"], observed["inner_ctx"]
        step_span = caller_ctx._trace[observed["caller_call_id"]]
        assert step_span.path == "wf.step"
        assert inner_ctx.root_span().metadata["invoked_by"] == {"run_id": caller_ctx.id, "call_id": step_span.call_id}
        assert step_span.metadata["invoked_runs"] == [inner_ctx.id]

    async def test_an_agent_in_tools_is_nested_not_linked(self):
        observed: dict = {}
        outer = Agent(
            name="outer",
            model=_calls("inner", {"prompt": "hi"}),
            tools=[_inner(observed)],
            tracing_provider=InMemoryTracingProvider,
        )
        result = await outer(prompt="go").collect()
        assert result.status.code == "success", result.error

        inner_ctx = observed["inner_ctx"]
        assert inner_ctx.root_span().path == "outer"  # same run as the caller
        assert inner_ctx._invoked_by is None
        assert not any("invoked_by" in s.metadata or "invoked_runs" in s.metadata for s in inner_ctx._trace.values())

    async def test_a_top_level_run_has_no_caller(self):
        observed: dict = {}
        result = await _inner(observed)(prompt="hi").collect()
        assert result.status.code == "success", result.error

        inner_ctx = observed["inner_ctx"]
        assert inner_ctx._invoked_by is None
        assert "invoked_by" not in inner_ctx.root_span().metadata
