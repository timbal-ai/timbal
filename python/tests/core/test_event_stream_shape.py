"""Golden tests for the exact shape of event streams.

These pin the number, types, order, and paths of events surfaced when
iterating tools, agents, and workflows — including nested (grandchild)
runnables and DeltaEvent forwarding. They exist so refactors of the event
plumbing (collector wrapping, internal stream consumption, multiplexing)
cannot silently drop, duplicate, or reorder events.

Conventions pinned here:
- Every runnable invocation surfaces exactly one StartEvent and one
  OutputEvent on the consumer's stream, with the runnable's full path.
- Nested runnables' events (children and grandchildren) surface through the
  parent's stream, in between the parent's Start and Output.
- Async-generator handlers surface one DeltaEvent per yielded chunk (wrapped
  as Custom items when the chunk isn't a DeltaItem), between their Start and
  Output, and they propagate through parent agents/workflows.
"""

import pytest
from timbal import Agent, Tool, Workflow
from timbal.core.test_model import TestModel
from timbal.state import get_run_context
from timbal.types.content import ToolUseContent
from timbal.types.events import DeltaEvent, OutputEvent, StartEvent
from timbal.types.events.delta import Custom
from timbal.types.message import Message


def _shape(events) -> list[tuple[str, str]]:
    return [(e.type, e.path) for e in events]


def _assert_paired(events, path: str, count: int = 1) -> None:
    """Assert `path` has exactly `count` Start and Output events, each Start before its Output."""
    starts = [i for i, e in enumerate(events) if e.type == "START" and e.path == path]
    outputs = [i for i, e in enumerate(events) if e.type == "OUTPUT" and e.path == path]
    assert len(starts) == count, f"{path}: expected {count} StartEvent(s), got {len(starts)}: {_shape(events)}"
    assert len(outputs) == count, f"{path}: expected {count} OutputEvent(s), got {len(outputs)}: {_shape(events)}"
    for s, o in zip(starts, outputs, strict=True):
        assert s < o, f"{path}: OutputEvent at {o} precedes StartEvent at {s}: {_shape(events)}"


async def _drain(stream) -> list:
    return [event async for event in stream]


def _tool_use_msg(*calls: tuple[str, str, dict]) -> Message:
    return Message(
        role="assistant",
        content=[ToolUseContent(id=c[0], name=c[1], input=c[2]) for c in calls],
        stop_reason="tool_use",
    )


def add(a: int, b: int) -> int:
    return a + b


async def stream_chunks(prefix: str = "chunk"):
    """Async generator handler: yields three plain chunks."""
    yield f"{prefix}_1"
    yield f"{prefix}_2"
    yield f"{prefix}_3"


class TestToolStreamShape:
    async def test_plain_tool_exact_shape(self):
        tool = Tool(name="add", handler=add)
        events = await _drain(tool(a=1, b=2))
        assert _shape(events) == [("START", "add"), ("OUTPUT", "add")]
        assert isinstance(events[0], StartEvent)
        assert isinstance(events[1], OutputEvent)
        assert events[1].output == 3
        assert events[1].status.code == "success"

    async def test_async_gen_tool_exact_shape_with_deltas(self):
        tool = Tool(name="streamer", handler=stream_chunks)
        events = await _drain(tool())
        assert _shape(events) == [
            ("START", "streamer"),
            ("DELTA", "streamer"),
            ("DELTA", "streamer"),
            ("DELTA", "streamer"),
            ("OUTPUT", "streamer"),
        ]
        deltas = [e for e in events if isinstance(e, DeltaEvent)]
        assert all(isinstance(e.item, Custom) for e in deltas)
        assert [e.item.data for e in deltas] == ["chunk_1", "chunk_2", "chunk_3"]


class TestAgentStreamShape:
    def _agent_single_tool_call(self) -> Agent:
        return Agent(
            name="shape_agent",
            model=TestModel(responses=[
                _tool_use_msg(("c1", "add", {"a": 1, "b": 2})),
                "The answer is 3.",
            ]),
            tools=[add],
        )

    async def test_single_tool_call_exact_sequence(self):
        events = await _drain(self._agent_single_tool_call()(prompt="add 1 and 2"))
        assert _shape(events) == [
            ("START", "shape_agent"),
            ("START", "shape_agent.llm"),
            ("OUTPUT", "shape_agent.llm"),
            ("START", "shape_agent.add"),
            ("OUTPUT", "shape_agent.add"),
            ("START", "shape_agent.llm"),
            ("OUTPUT", "shape_agent.llm"),
            ("OUTPUT", "shape_agent"),
        ]
        tool_output = events[4]
        assert tool_output.output == 3
        final = events[-1]
        assert isinstance(final, OutputEvent)
        assert final.status.code == "success"

    async def test_parallel_tool_calls_completeness(self):
        def mul(a: int, b: int) -> int:
            return a * b

        agent = Agent(
            name="par_agent",
            model=TestModel(responses=[
                _tool_use_msg(("c1", "add", {"a": 1, "b": 2}), ("c2", "mul", {"a": 3, "b": 4})),
                "Done.",
            ]),
            tools=[add, mul],
        )
        events = await _drain(agent(prompt="compute"))

        # Exact counts per path; interleaving between the two tools is unordered.
        _assert_paired(events, "par_agent", count=1)
        _assert_paired(events, "par_agent.llm", count=2)
        _assert_paired(events, "par_agent.add", count=1)
        _assert_paired(events, "par_agent.mul", count=1)
        # 2 llm pairs + agent pair + 2 tool pairs = 10 events total.
        assert len(events) == 10, _shape(events)
        # Agent's Start is first, its Output is last.
        assert _shape(events)[0] == ("START", "par_agent")
        assert _shape(events)[-1] == ("OUTPUT", "par_agent")

    async def test_delta_forwarding_from_async_gen_tool(self):
        streamer = Tool(name="streamer", handler=stream_chunks)
        agent = Agent(
            name="delta_agent",
            model=TestModel(responses=[
                _tool_use_msg(("c1", "streamer", {})),
                "Streamed.",
            ]),
            tools=[streamer],
        )
        events = await _drain(agent(prompt="stream"))

        _assert_paired(events, "delta_agent", count=1)
        _assert_paired(events, "delta_agent.llm", count=2)
        _assert_paired(events, "delta_agent.streamer", count=1)

        deltas = [e for e in events if isinstance(e, DeltaEvent) and e.path == "delta_agent.streamer"]
        assert [e.item.data for e in deltas] == ["chunk_1", "chunk_2", "chunk_3"]
        # Deltas sit strictly between the tool's Start and Output.
        start_i = next(i for i, e in enumerate(events) if e.type == "START" and e.path == "delta_agent.streamer")
        output_i = next(i for i, e in enumerate(events) if e.type == "OUTPUT" and e.path == "delta_agent.streamer")
        delta_is = [i for i, e in enumerate(events) if isinstance(e, DeltaEvent) and e.path == "delta_agent.streamer"]
        assert all(start_i < i < output_i for i in delta_is), _shape(events)


class TestWorkflowStreamShape:
    async def test_sequential_exact_sequence(self):
        def step_a(x: int) -> int:
            return x + 1

        def step_b(x: int) -> int:
            return x * 2

        wf = (
            Workflow(name="seq_wf")
            .step(step_a)
            .step(step_b, x=lambda: get_run_context().step_span("step_a").output)
        )
        events = await _drain(wf(x=1))
        assert _shape(events) == [
            ("START", "seq_wf"),
            ("START", "seq_wf.step_a"),
            ("OUTPUT", "seq_wf.step_a"),
            ("START", "seq_wf.step_b"),
            ("OUTPUT", "seq_wf.step_b"),
            ("OUTPUT", "seq_wf"),
        ]
        assert events[-1].output == 4

    async def test_parallel_steps_completeness(self):
        def left(x: int) -> int:
            return x + 1

        def right(x: int) -> int:
            return x + 2

        wf = Workflow(name="par_wf").step(left).step(right)
        events = await _drain(wf(x=1))

        _assert_paired(events, "par_wf", count=1)
        _assert_paired(events, "par_wf.left", count=1)
        _assert_paired(events, "par_wf.right", count=1)
        assert len(events) == 6, _shape(events)
        assert _shape(events)[0] == ("START", "par_wf")
        assert _shape(events)[-1] == ("OUTPUT", "par_wf")

    async def test_delta_forwarding_from_async_gen_step(self):
        wf = Workflow(name="delta_wf").step(Tool(name="streamer", handler=stream_chunks))
        events = await _drain(wf())

        _assert_paired(events, "delta_wf", count=1)
        _assert_paired(events, "delta_wf.streamer", count=1)
        deltas = [e for e in events if isinstance(e, DeltaEvent) and e.path == "delta_wf.streamer"]
        assert [e.item.data for e in deltas] == ["chunk_1", "chunk_2", "chunk_3"]
        assert len(events) == 7, _shape(events)


class TestNestedGrandchildStream:
    async def test_agent_as_tool_grandchild_events_surface(self):
        specialist = Agent(
            name="specialist",
            model=TestModel(responses=["Specialist done."]),
            tools=[],
            description="A specialist subagent.",
        )
        main = Agent(
            name="main_agent",
            model=TestModel(responses=[
                _tool_use_msg(("c1", "specialist", {"prompt": "do the thing"})),
                "All done.",
            ]),
            tools=[specialist],
        )
        events = await _drain(main(prompt="delegate"))

        _assert_paired(events, "main_agent", count=1)
        _assert_paired(events, "main_agent.llm", count=2)
        _assert_paired(events, "main_agent.specialist", count=1)
        # Grandchild: the subagent's own LLM events surface through the top stream.
        _assert_paired(events, "main_agent.specialist.llm", count=1)
        # 2 main llm pairs + main pair + specialist pair + specialist llm pair = 10.
        assert len(events) == 10, _shape(events)
        # Grandchild events are nested strictly inside the subagent's Start/Output window.
        sub_start = next(i for i, e in enumerate(events) if e.type == "START" and e.path == "main_agent.specialist")
        sub_output = next(i for i, e in enumerate(events) if e.type == "OUTPUT" and e.path == "main_agent.specialist")
        for i, e in enumerate(events):
            if e.path == "main_agent.specialist.llm":
                assert sub_start < i < sub_output, _shape(events)

    async def test_workflow_in_workflow_grandchild_events_surface(self):
        def leaf(x: int = 1) -> int:
            return x + 41

        inner = Workflow(name="inner_wf").step(leaf)
        outer = Workflow(name="outer_wf").step(inner)
        events = await _drain(outer())

        _assert_paired(events, "outer_wf", count=1)
        _assert_paired(events, "outer_wf.inner_wf", count=1)
        _assert_paired(events, "outer_wf.inner_wf.leaf", count=1)
        assert len(events) == 6, _shape(events)
        assert events[-1].output == 42
