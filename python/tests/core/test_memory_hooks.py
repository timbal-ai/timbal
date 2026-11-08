"""Framework hooks for monolith#565: salience marks and run triggers (routines)."""

import asyncio

from timbal import Agent, Tool
from timbal.core.test_model import TestModel
from timbal.state import mark_salient, set_call_id, set_parent_call_id, set_run_context
from timbal.state.context import RunContext
from timbal.state.tracing.providers import InMemoryTracingProvider
from timbal.types.content import ToolUseContent
from timbal.types.message import Message


def fresh():
    set_run_context(None)
    set_call_id(None)
    set_parent_call_id(None)


def call(name: str, **input) -> Message:
    return Message(
        role="assistant", content=[ToolUseContent(id=f"c-{name}", name=name, input=input)], stop_reason="tool_use"
    )


async def trace_of(out):
    return await RunContext(tracing_provider=InMemoryTracingProvider)._get_parent_trace(out.run_id)


class TestSalience:
    async def test_the_agent_marks_its_turn(self):
        fresh()
        agent = Agent(
            name="bot",
            model=TestModel(responses=[call("mark_salient", why="User prefers replies in Spanish."), "Entendido."]),
            salience_tool=True,
            tracing_provider=InMemoryTracingProvider,
        )
        out = await agent(prompt="Please answer in Spanish from now on.").collect()
        assert out.status.code == "success", out.error
        trace = await trace_of(out)
        root = next(s for s in trace.values() if s.parent_call_id is None)
        assert root.metadata["memory"]["salient"] == ["User prefers replies in Spanish."]
        assert out.usage["memory:salient"] == 1

    async def test_off_by_default(self):
        fresh()
        seen = []

        def handler(messages):
            seen.append(messages)
            return "ok"

        agent = Agent(name="bot", model=TestModel(handler=handler), tracing_provider=InMemoryTracingProvider)
        assert all(t.name != "mark_salient" for t in (await agent._resolve_tools(0))[0])

    async def test_the_helper_from_a_tool_and_outside_a_run(self):
        fresh()
        assert mark_salient("nothing to mark") is False  # outside a run: no-op

        def refund(order: str) -> str:
            mark_salient(f"Refund issued for {order}")
            return "refunded"

        agent = Agent(
            name="bot",
            model=TestModel(responses=[call("refund", order="A-1"), "done"]),
            tools=[Tool(name="refund", handler=refund)],
            tracing_provider=InMemoryTracingProvider,
        )
        out = await agent(prompt="refund A-1").collect()
        trace = await trace_of(out)
        tool_span = next(s for s in trace.values() if s.path == "bot.refund")
        assert tool_span.metadata["memory"]["salient"] == ["Refund issued for A-1"]


class TestTriggers:
    async def test_a_run_context_trigger_is_stamped_on_the_root(self):
        fresh()
        trigger = {"kind": "routine", "routine_id": "r1", "fire_id": "f1", "scheduled_at": "2026-11-09T08:00:00Z"}
        ctx = RunContext(tracing_provider=InMemoryTracingProvider, trigger=trigger)
        set_run_context(ctx)
        agent = Agent(
            name="bot",
            model=TestModel(responses=[call("look"), "ok"]),
            tools=[Tool(name="look", handler=lambda: "seen")],
            tracing_provider=InMemoryTracingProvider,
        )
        out = await agent(prompt="summarize my tickets").collect()
        trace = await trace_of(out)
        root = next(s for s in trace.values() if s.parent_call_id is None)
        assert root.metadata["trigger"] == trigger
        assert all("trigger" not in s.metadata for s in trace.values() if s.parent_call_id is not None)

    async def test_the_trigger_comes_through_the_run_context_body(self):
        ctx = RunContext.model_validate({"id": "run-1", "trigger": {"kind": "routine", "routine_id": "r9"}})
        assert ctx.trigger == {"kind": "routine", "routine_id": "r9"}

    async def test_a_wake_carries_a_routine_trigger(self):
        from timbal.core.chat_session import ChatSession

        fresh()
        agent = Agent(name="bot", model=TestModel(responses=["NO_REPLY"]), tracing_provider=InMemoryTracingProvider)
        async with ChatSession(agent) as session:
            session.wake("Daily check: anything due?", trigger={"kind": "routine", "routine_id": "r1", "fire_id": "f7"})
            started = None
            async for item in session.events():
                if isinstance(item, dict) and item.get("kind") == "turn_started":
                    started = item
                if isinstance(item, dict) and item.get("kind") == "turn_finished":
                    break
        trigger = started["trigger"]
        assert trigger["kind"] == "routine" and trigger["routine_id"] == "r1" and trigger["fire_id"] == "f7"
        assert trigger["wake_id"] and trigger["tool_policy"] == "read_only"  # the wake's own fields stay
        await asyncio.sleep(0)
