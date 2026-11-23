"""A turn that is left between two steps keeps its history as it stands.

When a turn is cut while its model call is in flight, the agent salvages what the model had said so far
into memory (``Agent._salvage_interrupted_llm_output``): the next turn must read what the user was shown.
Between two steps there is no such call: the last model output is in memory already, with the results of
its tool calls. A turn left there (cancelled, or failed while resolving the next step's tools) must not get
that output a second time: the next turn's model would read a second call of the same tools, answered
"ERROR: There was an unexpected error executing the tool.", that never happened.

The same holds before a call's first model call: the latest model output in the run may be another call's
of the same agent.
"""

import asyncio
from collections.abc import AsyncGenerator

import pytest
from timbal import Agent, Tool
from timbal.core.chat_session import ChatSession
from timbal.core.test_model import TestModel
from timbal.core.tool_set import ToolSet
from timbal.state.context import RunContext
from timbal.state.tracing.providers import InMemoryTracingProvider
from timbal.state.tracing.providers.jsonl import JsonlTracingProvider
from timbal.state.tracing.providers.sqlite import SqliteTracingProvider
from timbal.types.content import ToolUseContent
from timbal.types.message import Message

from ..guardrails.conftest import StreamingTestModel, text_stream, tool_use_item

# A hang guard, not a timing check: these runs take milliseconds.
DEADLINE_S = 30

# This process's memory (the live list), and two stores that keep its dump: the salvage writes to both.
KINDS = ["memory", "jsonl", "sqlite"]


def _provider(kind: str, tmp_path):
    if kind == "memory":
        return InMemoryTracingProvider
    if kind == "jsonl":
        return JsonlTracingProvider.configured(_path=tmp_path / "traces.jsonl")
    return SqliteTracingProvider.configured(_path=tmp_path / "traces.db")


def _tool_call(name: str, call_id: str, **arguments) -> Message:
    return Message(
        role="assistant",
        content=[ToolUseContent(id=call_id, name=name, input=arguments)],
        stop_reason="tool_use",
    )


def _shape(messages: list) -> list[tuple[str, list[str]]]:
    messages = [Message.validate(message) for message in messages]
    return [(message.role, [content.type for content in message.content]) for message in messages]


async def _spans(provider, run_id: str, path: str) -> list:
    """The spans a run stored for ``path``, read back the way the turn after it reads them."""
    trace = await asyncio.wait_for(RunContext(tracing_provider=provider)._get_parent_trace(run_id), DEADLINE_S)
    return trace.get_path(path)


async def _run_until(agent: Agent, cut: asyncio.Event | None, **kwargs) -> str:
    """Run ``agent`` and return its run id. With ``cut``: cancel the run once that event is set, and wait
    for it to have ended."""
    run_ids: list[str] = []

    async def read() -> None:
        async for event in agent(**kwargs):
            run_ids.append(event.run_id)

    task = asyncio.create_task(read())
    if cut is not None:
        waiting = asyncio.create_task(cut.wait())
        await asyncio.wait({task, waiting}, timeout=DEADLINE_S, return_when=asyncio.FIRST_COMPLETED)
        waiting.cancel()
        task.cancel()
        assert cut.is_set(), "the run ended, or the deadline passed, before the point it was to be cut at"
    try:
        await asyncio.wait_for(task, DEADLINE_S)
    except asyncio.CancelledError:
        pass
    return run_ids[0]


async def _records(session: ChatSession, kind: str, count: int) -> list[dict]:
    """The first ``count`` session records of ``kind`` (waits for them)."""
    found: list[dict] = []

    async def collect() -> None:
        async for item in session.events():
            if isinstance(item, dict) and item.get("kind") == kind:
                found.append(item)
                if len(found) >= count:
                    return

    await asyncio.wait_for(collect(), DEADLINE_S)
    return found


class _Tools(ToolSet):
    """The tools of an agent, as a source that can be slow or down when a later step asks for them."""

    __test__ = False

    tool: Tool
    # What happens from the second time the tools are asked for: None, "block" (set ``asked`` and wait to
    # be cancelled) or "fail".
    later: str | None = None
    asked: asyncio.Event
    times: int = 0

    async def resolve(self) -> list:
        self.times += 1
        if self.times >= 2 and self.later == "block":
            self.asked.set()
            await asyncio.Event().wait()
        if self.times >= 2 and self.later == "fail":
            raise ConnectionError("the tool source is down")
        return [self.tool]


class _CutAsItArrives(StreamingTestModel):
    """Calls ``cut()`` in the middle of one of its answers (the ``call``-th), just before a chunk goes out:
    whatever ``cut`` does reaches the turn with that chunk already there for it, while that model call is
    in flight."""

    __test__ = False

    def __init__(self, scripts, *, call: int, after: int = 2):
        super().__init__(scripts)
        self.call, self.after, self.cut = call, after, None

    async def stream(self, messages: list, **kwargs) -> AsyncGenerator:
        mine, sent = self.call_count + 1 == self.call, 0
        async for chunk in super().stream(messages, **kwargs):
            if mine and sent == self.after and self.cut is not None:
                cut, self.cut = self.cut, None
                cut()
            yield chunk
            sent += 1


class TestLeftBetweenSteps:
    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("how", ["cancelled", "tools_fail"])
    async def test_the_last_model_output_is_not_stored_twice(self, kind, how, tmp_path):
        ran: list[str] = []
        seen: list[list[Message]] = []

        def work() -> str:
            ran.append("work")
            return "worked"

        def answer(messages: list[Message]) -> Message | str:
            seen.append(list(messages))
            if messages[-1].role == "user" and messages[-1].collect_text() == "start":
                return _tool_call("work", "call-1")
            return "done"

        tools = _Tools(
            tool=Tool(name="work", handler=work),
            later="block" if how == "cancelled" else "fail",
            asked=asyncio.Event(),
        )
        provider = _provider(kind, tmp_path)
        agent = Agent(name="agent", model=TestModel(handler=answer), tools=[tools], tracing_provider=provider)

        run_id = await _run_until(agent, tools.asked if how == "cancelled" else None, prompt="start")

        (left,) = await _spans(provider, run_id, "agent")
        assert ran == ["work"]
        assert _shape(left.memory) == [("user", ["text"]), ("assistant", ["tool_use"]), ("tool", ["tool_result"])]

        # The turn after it reads one call and its result. Not a second call that "failed".
        tools.later = None
        after = await asyncio.wait_for(agent(prompt="and now?", parent_id=run_id).collect(), DEADLINE_S)
        assert after.status.code == "success"
        assert _shape(seen[-1]) == [
            ("user", ["text"]),
            ("assistant", ["tool_use"]),
            ("tool", ["tool_result"]),
            ("user", ["text"]),
        ]

    async def test_before_the_last_call_of_a_turn_out_of_steps(self, monkeypatch):
        """``max_iter`` reached: one more model call follows, without tools. What is awaited before it
        starts (here the per-request attachment limit) is between two steps like any other."""
        reached = asyncio.Event()
        requests = 0

        async def limit(messages, **_kwargs):
            nonlocal requests
            requests += 1
            if requests >= 2:  # the request of the call that follows max_iter
                reached.set()
                await asyncio.Event().wait()
            return messages

        monkeypatch.setattr("timbal.core.agent.apply_attachment_limit", limit)
        agent = Agent(
            name="agent",
            model=TestModel(responses=[_tool_call("work", "call-1"), "done"]),
            tools=[Tool(name="work", handler=lambda: "worked")],
            max_iter=1,
            attachment_limit=20_000,
            tracing_provider=InMemoryTracingProvider,
        )

        run_id = await _run_until(agent, reached, prompt="start")

        (left,) = await _spans(InMemoryTracingProvider, run_id, "agent")
        assert _shape(left.memory)[:3] == [("user", ["text"]), ("assistant", ["tool_use"]), ("tool", ["tool_result"])]
        assert [role for role, _ in _shape(left.memory)].count("assistant") == 1

    @pytest.mark.parametrize("kind", KINDS)
    async def test_a_turn_cut_during_a_later_model_call_keeps_what_was_said(self, kind, tmp_path):
        """The guard for the other side: with a model call in flight, in any step, what it had said by
        then is salvaged (three chunks of seven: two streamed on, one there when the cut came)."""
        model = _CutAsItArrives(
            [[tool_use_item("work", {})], text_stream("Once upon a time there was a lighthouse keeper.")], call=2
        )
        provider = _provider(kind, tmp_path)
        agent = Agent(
            name="agent", model=model, tools=[Tool(name="work", handler=lambda: "worked")], tracing_provider=provider
        )

        # Through a session: its turns read the model's stream from a task of their own, so the stop reaches
        # the agent while it waits for the next chunk, not the stream.
        async with ChatSession(agent) as session:
            model.cut = session.interrupt
            session.send("start")
            (finished,) = await _records(session, "turn_finished", 1)
        assert finished["status"]["code"] == "cancelled"

        (left,) = await _spans(provider, finished["run_id"], "agent")
        assert _shape(left.memory) == [
            ("user", ["text"]),
            ("assistant", ["tool_use"]),
            ("tool", ["tool_result"]),
            ("assistant", ["text"]),
        ]
        assert Message.validate(left.memory[-1]).collect_text() == "Once upon a time ther"

    async def test_an_answer_taken_back_is_kept_if_the_turn_is_left_before_it_is_given_again(self):
        """A guard, true before the fix as well. A ``now`` message cuts the answer, and nothing joins the
        turn after all (the message is withdrawn): what the model had said is taken back out of memory,
        since the model is about to answer again. If the turn is left before it has (here its tools cannot
        be resolved), the user was still shown that text, and it is not in memory: it is salvaged."""
        answer = "Once upon a time there was a lighthouse keeper."
        model = _CutAsItArrives([text_stream(answer)], call=1)
        tools = _Tools(tool=Tool(name="noop", handler=lambda: "ok"), later="fail", asked=asyncio.Event())
        agent = Agent(name="agent", model=model, tools=[tools], tracing_provider=InMemoryTracingProvider)

        async with ChatSession(agent) as session:

            def now_then_nothing() -> None:
                session.withdraw(session.send("wait", mode="now").id)

            model.cut = now_then_nothing
            session.send("start")
            (finished,) = await _records(session, "turn_finished", 1)
        assert finished["status"]["code"] == "error"

        (left,) = await _spans(InMemoryTracingProvider, finished["run_id"], "agent")
        assert _shape(left.memory) == [("user", ["text"]), ("assistant", ["text"])]
        said = left.memory[-1].collect_text()
        assert said and answer.startswith(said) and said != answer


class TestAnotherCallOfTheSameAgent:
    async def test_a_call_cut_before_its_model_call_is_not_given_another_call_s_answer(self):
        """A sub-agent called twice in one run. Its second call is cut before its own model call starts:
        the latest model output of that agent in the run is the first call's."""
        tools = _Tools(tool=Tool(name="noop", handler=lambda: "ok"), later="block", asked=asyncio.Event())
        helper = Agent(
            name="helper",
            model=TestModel(handler=lambda messages: f"answer to {messages[0].collect_text()}"),
            tools=[tools],
        )

        def answer(messages: list[Message]) -> Message | str:
            asked = sum(1 for message in messages if message.role == "assistant")
            if asked < 2:
                return _tool_call("helper", f"call-{asked}", prompt=["one", "two"][asked])
            return "done"

        outer = Agent(
            name="outer", model=TestModel(handler=answer), tools=[helper], tracing_provider=InMemoryTracingProvider
        )

        run_id = await _run_until(outer, tools.asked, prompt="go")

        first, second = await _spans(InMemoryTracingProvider, run_id, "outer.helper")
        assert [message.collect_text() for message in first.memory] == ["one", "answer to one"]
        assert [message.collect_text() for message in second.memory] == ["two"]
