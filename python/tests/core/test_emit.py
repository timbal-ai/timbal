"""Tests for RunContext.emit() — ambient, fire-and-forget custom DELTA events.

emit() gives any handler code (plain coroutines, sync handlers, helpers deep in
the call stack) an out-of-band channel to the current call's event stream:

- foreground calls surface emissions on the parent event stream;
- detached background children surface them in their background log/transcript;
- emissions are always ordered before that call's OUTPUT event;
- emissions never pass through the collector, so they cannot alter the output.
"""

import asyncio
import time

import pytest
from timbal import Agent, Tool
from timbal.core.test_model import TestModel
from timbal.state import (
    emit,
    get_run_context,
    list_background_tasks,
    read_background_transcript,
)
from timbal.types.content import ToolUseContent
from timbal.types.events import OutputEvent
from timbal.types.events.delta import DeltaEvent
from timbal.types.message import Message

from ..conftest import assert_has_output_event


def _tool_call(tool_name: str, input: dict, *, id: str = "c1", run_in_background: bool = False) -> Message:
    """Return a TestModel response that calls one tool."""
    actual_input = {**input, **({"run_in_background": True} if run_in_background else {})}
    return Message(
        role="assistant",
        content=[ToolUseContent(id=id, name=tool_name, input=actual_input)],
        stop_reason="tool_use",
    )


def _custom_deltas(events: list, payload_key: str | None = None) -> list[DeltaEvent]:
    """Filter DELTA events with item.type == 'custom' (optionally by payload key)."""
    found = []
    for event in events:
        if isinstance(event, DeltaEvent) and event.item.type == "custom":
            if payload_key is None or (isinstance(event.item.data, dict) and payload_key in event.item.data):
                found.append(event)
    return found


class TestEmitForeground:
    """Foreground delivery: emissions ride the parent event stream."""

    @pytest.mark.asyncio
    async def test_plain_handler_emit_shape_and_ordering(self):
        """A plain coroutine handler's emits arrive as custom DELTAs before OUTPUT."""

        async def handler() -> dict:
            get_run_context().emit({"kind": "ui-event", "step": 1})
            get_run_context().emit({"kind": "ui-event", "step": 2})
            return {"ok": True}

        tool = Tool(name="emitter", handler=handler)
        events = [event async for event in tool()]

        output_events = [e for e in events if isinstance(e, OutputEvent)]
        assert len(output_events) == 1
        output_event = output_events[0]
        assert output_event.status.code == "success"
        # Requirement: emissions never touch the output.
        assert output_event.output == {"ok": True}

        customs = _custom_deltas(events, "kind")
        assert [c.item.data["step"] for c in customs] == [1, 2]
        # Emissions are ordered before the call's OUTPUT.
        assert all(events.index(c) < events.index(output_event) for c in customs)
        # Ambient ids: the event is attributed to the emitting call.
        for custom in customs:
            assert custom.call_id == output_event.call_id
            assert custom.path == output_event.path == "emitter"
            assert custom.run_id == output_event.run_id
            assert custom.item.id == custom.call_id

    @pytest.mark.asyncio
    async def test_plain_handler_emit_flushed_before_output(self):
        """Plain handlers are not Task-wrapped (bench hot path). Emits flush at completion, still before OUTPUT."""

        async def handler() -> str:
            get_run_context().emit({"kind": "progress", "phase": "started"})
            await asyncio.sleep(0)
            return "done"

        tool = Tool(name="midrun", handler=handler)
        events = [event async for event in tool()]

        output_event = events[-1]
        assert isinstance(output_event, OutputEvent)
        assert output_event.status.code == "success"
        assert output_event.output == "done"
        customs = _custom_deltas(events, "kind")
        assert len(customs) == 1
        assert events.index(customs[0]) < events.index(output_event)

    @pytest.mark.asyncio
    async def test_emit_visible_on_agent_stream(self):
        """Emissions from a tool handler surface on the owning agent's stream."""

        async def resolve_resource() -> dict:
            emit({"kind": "compose-ui-event", "event": "trace_ref", "resource": "res-42"})
            return {"resource_id": "res-42"}

        agent = Agent(
            name="emit_agent",
            model=TestModel(responses=[_tool_call("resolve_resource", {}), "All done."]),
            tools=[Tool(name="resolve_resource", handler=resolve_resource)],
        )

        events = [event async for event in agent(prompt="resolve it")]
        customs = _custom_deltas(events, "kind")
        assert len(customs) == 1
        custom = customs[0]
        assert custom.path == "emit_agent.resolve_resource"
        assert custom.item.data["event"] == "trace_ref"

        tool_output = next(
            e
            for e in events
            if isinstance(e, OutputEvent) and e.path == "emit_agent.resolve_resource"
        )
        # Attributed to the tool call, ordered before the tool's OUTPUT, and
        # absent from the tool's output itself.
        assert custom.call_id == tool_output.call_id
        assert events.index(custom) < events.index(tool_output)
        assert tool_output.output == {"resource_id": "res-42"}

    @pytest.mark.asyncio
    async def test_emit_from_async_gen_handler_does_not_pollute_output(self):
        """Generator handlers: emit() bypasses the collector, unlike yields."""

        async def gen_handler():
            yield "a"
            get_run_context().emit({"kind": "side-channel", "n": 1})
            yield "b"

        tool = Tool(name="gen_emitter", handler=gen_handler)
        events = [event async for event in tool()]

        output_event = next(e for e in events if isinstance(e, OutputEvent))
        # String yields concatenate via the string collector — the emitted
        # dict must not have passed through it.
        assert output_event.output == "ab"
        emitted = _custom_deltas(events, "kind")
        assert len(emitted) == 1
        assert emitted[0].item.data == {"kind": "side-channel", "n": 1}
        assert events.index(emitted[0]) < events.index(output_event)

    @pytest.mark.asyncio
    async def test_emit_from_offloaded_sync_handler(self):
        """Thread safety: emit from a sync handler running in an executor thread."""

        def blocking_handler() -> str:
            emit({"kind": "from-thread"})
            time.sleep(0.05)
            return "done"

        tool = Tool(name="offloaded", handler=blocking_handler, offload_blocking=True)
        events = [event async for event in tool()]

        output_event = next(e for e in events if isinstance(e, OutputEvent))
        assert output_event.status.code == "success"
        assert output_event.output == "done"
        emitted = _custom_deltas(events, "kind")
        assert len(emitted) == 1
        assert events.index(emitted[0]) < events.index(output_event)

    @pytest.mark.asyncio
    async def test_emit_from_sync_gen_handler(self):
        """Thread safety: sync generators run via sync_to_async_gen off-loop."""

        def sync_gen():
            yield 1
            emit({"kind": "from-sync-gen"})
            yield 2

        tool = Tool(name="sync_gen_emitter", handler=sync_gen)
        events = [event async for event in tool()]

        output_event = next(e for e in events if isinstance(e, OutputEvent))
        assert output_event.output == [1, 2]
        emitted = _custom_deltas(events, "kind")
        assert len(emitted) == 1
        assert events.index(emitted[0]) < events.index(output_event)

    @pytest.mark.asyncio
    async def test_emit_kept_out_of_default_collector_output(self):
        """Non-string yields collect into a list — emissions must not join it."""

        async def gen_handler():
            yield {"chunk": 1}
            get_run_context().emit({"kind": "side-channel"})
            yield {"chunk": 2}

        tool = Tool(name="dict_gen_emitter", handler=gen_handler)
        events = [event async for event in tool()]

        output_event = next(e for e in events if isinstance(e, OutputEvent))
        assert output_event.output == [{"chunk": 1}, {"chunk": 2}]
        assert len(_custom_deltas(events, "kind")) == 1

    @pytest.mark.asyncio
    async def test_child_emit_does_not_steal_parent_sink(self):
        """A parent that already emitted must not swallow the child's first emit."""

        def pre_hook():
            emit({"kind": "parent"})

        async def child() -> str:
            emit({"kind": "child"})
            span = get_run_context().current_span()
            assert span.path == "a.child"
            assert span._emit_sink is not None
            assert span._emit_sink.has_pending()
            return "ok"

        agent = Agent(
            name="a",
            model=TestModel(responses=[_tool_call("child", {}), "done"]),
            tools=[Tool(name="child", handler=child)],
            pre_hook=pre_hook,
        )
        events = [event async for event in agent(prompt="go")]

        parent_events = [e for e in _custom_deltas(events, "kind") if e.item.data["kind"] == "parent"]
        child_events = [e for e in _custom_deltas(events, "kind") if e.item.data["kind"] == "child"]
        assert len(parent_events) == 1
        assert parent_events[0].path == "a"
        assert len(child_events) == 1
        child_output = next(e for e in events if isinstance(e, OutputEvent) and e.path == "a.child")
        assert child_events[0].path == "a.child"
        assert child_events[0].call_id == child_output.call_id
        assert events.index(child_events[0]) < events.index(child_output)


class TestEmitBackground:
    """Detached background children: emissions land in the background log."""

    @pytest.mark.asyncio
    async def test_emit_lands_in_background_transcript(self):
        async def bg_task() -> str:
            emit({"kind": "bg-progress", "pct": 50})
            await asyncio.sleep(0.05)
            return "bg done"

        agent = Agent(
            name="bg_emit_agent",
            model=TestModel(
                responses=[
                    _tool_call("bg_task", {}, run_in_background=True),
                    "Started in the background.",
                ]
            ),
            tools=[Tool(name="bg_task", description="Emit from background", handler=bg_task, background_mode="auto")],
        )

        parent_events = [event async for event in agent(prompt="run it")]
        assert_has_output_event(parent_events[-1])
        # The emission belongs to the detached child, not the parent stream.
        assert _custom_deltas(parent_events, "kind") == []

        listed = list_background_tasks()
        assert len(listed) == 1
        task_id = listed[0]["task_id"]

        # The transcript serializes events to dicts.
        deadline = time.monotonic() + 5
        emitted = []
        while time.monotonic() < deadline and not emitted:
            transcript = read_background_transcript(task_id)
            emitted = [
                event
                for event in transcript["events"]
                if isinstance(event, dict)
                and event.get("type") == "DELTA"
                and event.get("item", {}).get("type") == "custom"
                and isinstance(event["item"].get("data"), dict)
                and "kind" in event["item"]["data"]
            ]
            if not emitted:
                await asyncio.sleep(0.01)
        assert len(emitted) == 1
        assert emitted[0]["item"]["data"] == {"kind": "bg-progress", "pct": 50}
        await asyncio.sleep(0.1)


class TestEmitNoContext:
    """No-op outside a run: tools stay unit-testable without a context harness."""

    def test_module_emit_without_context_is_noop(self):
        emit({"kind": "nobody-listening"})  # must not raise

    @pytest.mark.asyncio
    async def test_handler_calling_emit_is_directly_invocable(self):
        async def handler() -> str:
            emit({"kind": "ui-event"})
            return "ok"

        # Direct invocation, no framework, no run context.
        assert await handler() == "ok"

    def test_run_context_emit_without_active_call_is_noop(self):
        from timbal.state.context import RunContext

        RunContext().emit({"kind": "no-span"})  # must not raise
