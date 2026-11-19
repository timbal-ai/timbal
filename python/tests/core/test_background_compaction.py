"""memory_compaction_prepare_ratio: compaction prepared in the background, swapped in mid-turn, handed to the next turn."""

import asyncio
from types import SimpleNamespace

import pytest
from timbal import Agent, Tool
from timbal.core import agent as agent_module
from timbal.core.test_model import TestModel
from timbal.state import set_call_id, set_parent_call_id, set_run_context
from timbal.state.context import RunContext
from timbal.state.tracing.providers import InMemoryTracingProvider
from timbal.types.content import TextContent, ToolUseContent
from timbal.types.message import Message

SUMMARY = "[summary]"


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    set_run_context(None)
    set_call_id(None)
    set_parent_call_id(None)
    agent_module._PREPARING.clear()
    # Each message is 5% of the window: prepare at 6+ messages (0.3), compact inline at 15+ (0.75).
    monkeypatch.setattr(
        Agent, "_context_utilization", lambda _self, _model, messages, _measurement: 0.05 * len(messages)
    )
    yield
    agent_module._PREPARING.clear()
    set_run_context(None)


class Summarizer:
    """An async compactor that waits for ``release`` (the summarizer's model call), then keeps the last 2."""

    def __init__(self, *, released=True):
        self.calls: list[int] = []
        self.release = asyncio.Event()
        self.started = asyncio.Event()
        if released:
            self.release.set()

    def compactor(self):
        async def summarize_old(memory):
            self.calls.append(len(memory))
            self.started.set()
            await self.release.wait()
            if len(memory) <= 3:
                return memory
            return [Message(role="user", content=[TextContent(text=SUMMARY)]), *memory[-2:]]

        return summarize_old


def tool_loop(steps):
    """``steps`` tool calls in one turn, then ``done``; a later turn answers ``again``. Counted by the model
    itself: compaction takes tool results out of the history it reads."""
    calls = 0

    def handler(messages):
        nonlocal calls
        if messages[-1].role == "user" and messages[-1].collect_text() == "again":
            return "again done"
        if calls >= steps:
            return "done"
        calls += 1
        return Message(
            role="assistant",
            content=[ToolUseContent(id=f"c{calls}", name="step", input={})],
            stop_reason="tool_use",
        )

    return handler


async def step() -> str:
    await asyncio.sleep(0.01)  # a real tool waits on I/O: the background summarizer runs meanwhile
    return "ok"


def make_agent(summarizer, steps, **kwargs):
    return Agent(
        name="agent",
        model=TestModel(handler=tool_loop(steps)),
        tools=[Tool(name="step", handler=step)],
        tracing_provider=InMemoryTracingProvider,
        memory_compaction=summarizer.compactor(),
        max_iter=30,
        **kwargs,
    )


async def run(agent, prompt="go", parent_id=None):
    context = RunContext(parent_id=parent_id, tracing_provider=InMemoryTracingProvider)
    set_run_context(context)
    output = await agent(prompt=prompt).collect()
    set_run_context(None)
    assert output.status.code == "success", output.error
    return context, output


def root(context):
    return context.root_span()


async def settled(context):
    """Wait for the run's after-turn compaction, if it is still running (it may have landed already)."""
    finishing = agent_module._PREPARING.get((context.id, "agent"))
    if finishing is not None:
        await finishing


async def test_the_summarizer_sends_no_temperature(monkeypatch):
    """Claude Haiku 5.5 rejects ``temperature`` ("deprecated for this model"): summarize() must not send one."""
    from timbal.core import llm
    from timbal.core.memory_compaction import _call_summarizer

    sent = {}

    async def router(**kwargs):
        sent.update(kwargs)
        yield "not a provider chunk"

    monkeypatch.setattr(llm, "_llm_router", router)
    assert await _call_summarizer("anthropic/claude-haiku-5-5", "summarize this", 100) is None
    assert "temperature" not in sent and sent["max_tokens"] == 100


class TestConfig:
    def test_the_prepare_ratio_sits_below_the_compaction_ratio(self):
        with pytest.raises(ValueError, match="memory_compaction_prepare_ratio"):
            Agent(name="a", model=TestModel(), memory_compaction_prepare_ratio=0.8)
        with pytest.raises(ValueError, match="memory_compaction_prepare_ratio"):
            Agent(name="a", model=TestModel(), memory_compaction_prepare_ratio=0.0)
        Agent(name="a", model=TestModel(), memory_compaction_prepare_ratio=0.5)

    async def test_off_by_default(self):
        summarizer = Summarizer()
        context, _ = await run(make_agent(summarizer, steps=5))
        assert summarizer.calls == []  # 11 messages: under the inline ratio, nothing prepared either


class TestMidTurn:
    async def test_a_long_turn_swaps_in_the_compaction_it_prepared(self):
        summarizer = Summarizer()
        agent = make_agent(summarizer, steps=6, memory_compaction_prepare_ratio=0.3)
        context, output = await run(agent)
        compaction = root(context).metadata["compaction"]
        assert compaction["background"] is True
        assert len(summarizer.calls) >= 1
        assert root(context).memory[0].collect_text() == SUMMARY
        assert output.usage.get("compaction:background", 0) >= 1

    async def test_reaching_the_hard_ratio_waits_for_the_one_in_progress(self, monkeypatch):
        summarizer = Summarizer(released=False)
        agent = make_agent(summarizer, steps=12, memory_compaction_prepare_ratio=0.3)

        async def release_later():
            await summarizer.started.wait()
            await asyncio.sleep(0.2)  # the turn reaches the hard ratio meanwhile
            summarizer.release.set()

        sizes = []
        real_boundary = Agent._compaction_boundary

        async def boundary(self, span, run_context, prepared):
            sizes.append(len(span.memory))
            return await real_boundary(self, span, run_context, prepared)

        monkeypatch.setattr(Agent, "_compaction_boundary", boundary)
        releaser = asyncio.create_task(release_later())
        context, _ = await run(agent)
        await releaser
        # Started past the prepare ratio, it was still running at the hard one: the turn waited for it there
        # (no inline summary of its own), and the history never grew past the hard ratio.
        assert summarizer.calls[0] < 15 and max(sizes) <= 15
        assert all(n < 15 for n in summarizer.calls)
        assert root(context).metadata["compaction"]["background"] is True

    def test_a_history_that_changed_under_it_is_not_swapped(self):
        agent = Agent(name="a", model=TestModel(), memory_compaction=lambda m: m, memory_compaction_prepare_ratio=0.3)
        prefix = [Message(role="user", content=[TextContent(text=f"m{i}")]) for i in range(4)]
        loop = asyncio.new_event_loop()
        try:
            task = loop.create_future()
            task.set_result(([Message(role="user", content=[TextContent(text=SUMMARY)])], []))
            prepared = agent_module._Prepared(task=task, prefix=prefix, utilization=0.4)
            usage = {}
            span = SimpleNamespace(
                memory=[prefix[0], Message(role="user", content=[TextContent(text="cut")]), *prefix[2:]],
                metadata={},
                input={},
            )
            run_context = SimpleNamespace(update_usage=lambda key, value: usage.__setitem__(key, value))
            assert not agent._swap_prepared(span, prepared, run_context)
            assert usage == {"compaction:prepared_stale": 1}
            span.memory = [*prefix, Message(role="assistant", content=[TextContent(text="new")])]
            assert agent._swap_prepared(span, prepared, run_context)
            assert [m.collect_text() for m in span.memory] == [SUMMARY, "new"]
        finally:
            loop.close()

    async def test_a_strategy_that_shrinks_results_in_place_is_swapped_in(self):
        """compact_tool_results keeps the message count: its result must not read as 'nothing to gain'."""
        from timbal.core.memory_compaction import compact_tool_results

        agent = Agent(
            name="agent",
            model=TestModel(handler=tool_loop(6)),
            tools=[Tool(name="step", handler=step)],
            tracing_provider=InMemoryTracingProvider,
            memory_compaction=compact_tool_results(keep_last_n=1, replacement="[compacted]"),
            memory_compaction_prepare_ratio=0.3,
            max_iter=30,
        )
        context, output = await run(agent)
        assert output.usage.get("compaction:background", 0) >= 1
        results = [m for m in root(context).memory if m.role == "tool"]
        first = results[0].content[0]
        assert first.content[0].text == "[compacted]"  # replaced in place, message count kept

    async def test_a_prefix_that_compacts_to_nothing_is_not_compacted_again(self, monkeypatch):
        calls = []

        def unchanged(memory):
            calls.append(len(memory))
            return [Message(role=m.role, content=list(m.content), stop_reason=m.stop_reason) for m in memory]

        agent = Agent(name="a", model=TestModel(), memory_compaction=unchanged, memory_compaction_prepare_ratio=0.3)

        async def no_inline(*_args, **_kwargs):
            return None

        monkeypatch.setattr(agent, "_compact_preserving_last_assistant", no_inline)
        memory = []
        for i in range(4):
            memory += [
                Message(role="user", content=[TextContent(text=f"u{i}")]),
                Message(role="assistant", content=[TextContent(text=f"a{i}")]),
            ]
        span = SimpleNamespace(memory=memory, metadata={}, input={})
        usage = {}
        run_context = SimpleNamespace(update_usage=lambda key, value: usage.__setitem__(key, value))
        prepared = await agent._compaction_boundary(span, run_context, None)
        await prepared.task
        for _ in range(3):  # same history: the futile result is remembered, not recomputed
            prepared = await agent._compaction_boundary(span, run_context, prepared)
        assert calls == [7] and prepared.futile and span.memory is memory
        span.memory = memory = [
            *memory,
            Message(role="user", content=[TextContent(text="u4")]),
            Message(role="assistant", content=[TextContent(text="a4")]),
        ]
        prepared = await agent._compaction_boundary(span, run_context, prepared)  # grown: worth another look
        await prepared.task
        assert calls == [7, 9]

    def test_a_history_with_files_is_compared_without_dumping_them(self):
        """A rebuilt message that still holds a file (a kept attachment) must not crash the swap."""
        from timbal.types.content import FileContent
        from timbal.types.file import File

        agent = Agent(name="a", model=TestModel(), memory_compaction=lambda m: m, memory_compaction_prepare_ratio=0.3)
        attachment = FileContent(file=File.validate(b"data", {"extension": ".png"}))
        prefix = [
            Message(role="user", content=[attachment, TextContent(text="look")]),
            Message(role="assistant", content=[TextContent(text="long answer")]),
        ]

        def rebuilt(second_text):
            return [
                Message(role="user", content=list(prefix[0].content)),  # same blocks, new message
                Message(role="assistant", content=[TextContent(text=second_text)]),
            ]

        loop = asyncio.new_event_loop()
        try:
            for second_text, expect_swap in (("long answer", False), ("short", True)):
                task = loop.create_future()
                task.set_result((rebuilt(second_text), []))
                prepared = agent_module._Prepared(task=task, prefix=prefix, utilization=0.4)
                span = SimpleNamespace(memory=[*prefix], metadata={}, input={})
                run_context = SimpleNamespace(update_usage=lambda *_: None)
                assert agent._swap_prepared(span, prepared, run_context) is expect_swap
                assert prepared.futile is not expect_swap
        finally:
            loop.close()

    async def test_a_failing_strategy_cannot_leave_edits_in_the_live_history(self):
        """The task works on copies: an in-place edit followed by an error stays out of the messages the model sees."""

        async def edit_then_fail(memory):
            memory[0].content[0].text = "EDITED"
            raise RuntimeError("summarizer down")

        agent = Agent(
            name="a", model=TestModel(), memory_compaction=edit_then_fail, memory_compaction_prepare_ratio=0.3
        )
        live = [
            Message(role="user", content=[TextContent(text="original")]),
            Message(role="assistant", content=[TextContent(text="answer")]),
        ]
        prepared = agent._prepare_compaction(SimpleNamespace(input={}), live, 0.4)
        with pytest.raises(RuntimeError):
            await prepared.task
        assert live[0].content[0].text == "original"


class TestNextTurn:
    async def test_a_turn_that_ends_past_the_ratio_hands_its_compaction_to_the_next(self):
        summarizer = Summarizer(released=False)
        agent = make_agent(summarizer, steps=2, memory_compaction_prepare_ratio=0.3)  # 6 messages at the end
        first, _ = await run(agent)
        assert (first.id, "agent") in agent_module._PREPARING  # still summarizing after the turn ended
        summarizer.release.set()
        await settled(first)
        second, output = await run(agent, prompt="again", parent_id=first.id)
        memory = root(second).memory
        assert memory[0].collect_text() == SUMMARY and memory[-1].collect_text() == "again done"
        assert root(second).metadata["compaction"]["prepared_by"] == first.id
        assert output.usage.get("compaction:background") == 1
        assert len(summarizer.calls) == 1  # once, after turn one: never inline at turn two's start

    async def test_taken_from_this_process_it_is_removed_from_session_data(self):
        """The record the in-process task handed over must not stay behind for a later turn to find stale."""
        summarizer = Summarizer(released=False)
        agent = make_agent(summarizer, steps=2, memory_compaction_prepare_ratio=0.3)
        first, _ = await run(agent)
        summarizer.release.set()
        await settled(first)  # landed in the session data as well
        agent_module._PREPARING[(first.id, "agent")] = asyncio.get_running_loop().create_future()
        stored = await RunContext(parent_id=first.id, tracing_provider=InMemoryTracingProvider)._get_parent_trace()
        record = stored[root(first).call_id].session[agent_module._PREPARED_KEY]["agent"]
        agent_module._PREPARING[(first.id, "agent")].set_result(record)
        second, output = await run(agent, prompt="again", parent_id=first.id)
        assert root(second).memory[0].collect_text() == SUMMARY
        assert agent_module._PREPARED_KEY not in (await second.get_session())
        third, output = await run(agent, prompt="again", parent_id=second.id)
        assert "compaction:prepared_stale" not in output.usage

    async def test_a_history_that_fits_does_not_wait_for_it(self):
        """Below the hard ratio the next turn goes on with the history it has instead of waiting on the summarizer."""
        summarizer = Summarizer(released=False)
        agent = make_agent(summarizer, steps=2, memory_compaction_prepare_ratio=0.3)
        first, _ = await run(agent)
        finishing = agent_module._PREPARING[(first.id, "agent")]
        second, _ = await asyncio.wait_for(run(agent, prompt="again", parent_id=first.id), timeout=2)
        assert not finishing.done()  # never waited for
        assert root(second).memory[0].collect_text() != SUMMARY
        summarizer.release.set()
        await finishing

    async def test_the_incoming_prompt_counts_toward_the_hard_ratio(self):
        """14 messages fit (0.70); with the prompt the turn sends, 15 do not (0.75): it waits for the compaction."""
        agent = Agent(
            name="agent", model=TestModel(), memory_compaction=lambda m: m, memory_compaction_prepare_ratio=0.3
        )
        memory = [Message(role="user", content=[TextContent(text=f"m{i}")]) for i in range(14)]
        prompt = Message(role="user", content=[TextContent(text="next")])
        session = {}

        async def get_session():
            return session

        run_context = SimpleNamespace(parent_id="parent", get_session=get_session, update_usage=lambda *_: None)
        source = SimpleNamespace(memory=memory, metadata={})
        loop = asyncio.get_running_loop()
        for incoming, waits in (([], False), ([prompt], True)):
            finishing = loop.create_future()
            agent_module._PREPARING[("parent", "agent")] = finishing
            loop.call_later(0.2, lambda f=finishing: f.done() or f.set_result(None))
            started = loop.time()
            await agent._take_prepared(run_context, source, memory, "test", incoming)
            assert (loop.time() - started >= 0.15) is waits
            await finishing

    async def test_a_history_at_the_hard_ratio_waits_for_it(self, monkeypatch):
        summarizer = Summarizer(released=False)
        agent = make_agent(summarizer, steps=2, memory_compaction_prepare_ratio=0.3)
        first, _ = await run(agent)
        real = Agent._take_prepared

        async def at_the_limit(self, run_context, source, memory, model, incoming):
            monkeypatch.setattr(Agent, "_context_utilization", lambda *_args: 0.8)  # does not fit as it is
            try:
                return await real(self, run_context, source, memory, model, incoming)
            finally:
                monkeypatch.setattr(Agent, "_context_utilization", lambda _s, _m, messages, _x: 0.05 * len(messages))

        monkeypatch.setattr(Agent, "_take_prepared", at_the_limit)
        asyncio.get_running_loop().call_later(0.1, summarizer.release.set)
        second, _ = await run(agent, prompt="again", parent_id=first.id)
        assert root(second).memory[0].collect_text() == SUMMARY  # waited, and took it

    async def test_the_handoff_is_stored_before_anyone_is_told_it_finished(self, monkeypatch):
        """The next turn waits on ``_PREPARING``: once the entry is gone the record must already be in the trace."""
        real = RunContext._end_background_work

        async def slow_store(self):
            await asyncio.sleep(0.2)  # a store that takes a while (a loaded machine, a remote provider)
            await real(self)

        monkeypatch.setattr(RunContext, "_end_background_work", slow_store)
        summarizer = Summarizer()
        agent = make_agent(summarizer, steps=2, memory_compaction_prepare_ratio=0.3)
        first, _ = await run(agent)
        while (first.id, "agent") in agent_module._PREPARING:
            await asyncio.sleep(0.01)
        stored = await RunContext(parent_id=first.id, tracing_provider=InMemoryTracingProvider)._get_parent_trace()
        assert "agent" in (stored[root(first).call_id].session or {}).get(agent_module._PREPARED_KEY, {})

    async def test_another_process_takes_it_from_the_stored_session_data(self):
        summarizer = Summarizer()
        agent = make_agent(summarizer, steps=2, memory_compaction_prepare_ratio=0.3)
        first, _ = await run(agent)
        await settled(first)
        agent_module._PREPARING.clear()  # a new process: only the trace is left
        stored = await RunContext(parent_id=first.id, tracing_provider=InMemoryTracingProvider)._get_parent_trace()
        assert "agent" in stored[root(first).call_id].session[agent_module._PREPARED_KEY]
        second, _ = await run(agent, prompt="again", parent_id=first.id)
        assert root(second).memory[0].collect_text() == SUMMARY
        assert agent_module._PREPARED_KEY not in (await second.get_session())  # used: not carried on

    async def test_a_stored_compaction_for_another_history_is_ignored(self):
        summarizer = Summarizer()
        agent = make_agent(summarizer, steps=2, memory_compaction_prepare_ratio=0.3)
        first, _ = await run(agent)
        await settled(first)
        agent_module._PREPARING.clear()
        stored = InMemoryTracingProvider._storage[first.id]
        record = stored[root(first).call_id].session[agent_module._PREPARED_KEY]["agent"]
        record["fingerprint"] = "another history"
        second, output = await run(agent, prompt="again", parent_id=first.id)
        assert root(second).memory[0].collect_text() != SUMMARY
        assert output.usage.get("compaction:prepared_stale") == 1

    async def test_across_processes_on_sqlite(self, tmp_path):
        """The history comes back as stored JSON: the fingerprint must match what the run stored, not live objects."""
        from timbal.state.tracing.providers import SqliteTracingProvider

        provider = SqliteTracingProvider.configured(_path=tmp_path / "t.db")
        summarizer = Summarizer()
        agent = make_agent(summarizer, steps=2, memory_compaction_prepare_ratio=0.3)
        agent.tracing_provider = provider

        async def on(prompt, parent_id=None):
            context = RunContext(parent_id=parent_id, tracing_provider=provider)
            set_run_context(context)
            output = await agent(prompt=prompt).collect()
            set_run_context(None)
            return context, output

        first, _ = await on("go")
        await settled(first)
        agent_module._PREPARING.clear()
        second, output = await on("again", parent_id=first.id)
        assert root(second).memory[0].collect_text() == SUMMARY
        assert output.usage.get("compaction:background") == 1

    async def test_a_pause_cancels_the_compaction_in_progress(self):
        summarizer = Summarizer(released=False)
        calls = 0

        def handler(_messages):
            nonlocal calls
            calls += 1
            name = "pay" if calls == 4 else "step"
            return Message(
                role="assistant", content=[ToolUseContent(id=f"c{calls}", name=name, input={})], stop_reason="tool_use"
            )

        agent = Agent(
            name="agent",
            model=TestModel(handler=handler),
            tools=[Tool(name="step", handler=step), Tool(name="pay", handler=lambda: "paid", requires_approval=True)],
            tracing_provider=InMemoryTracingProvider,
            memory_compaction=summarizer.compactor(),
            memory_compaction_prepare_ratio=0.3,
            max_iter=30,
        )
        context = RunContext(tracing_provider=InMemoryTracingProvider)
        set_run_context(context)
        output = await agent(prompt="go").collect()
        set_run_context(None)
        assert output.status.reason == "approval_required"
        summarizer.release.set()
        await asyncio.sleep(0.05)
        # Prepared mid-turn, cancelled by the pause: nothing is handed to a next turn (the resume compacts as it
        # finds the history), and nothing landed in the run's memory.
        assert (context.id, "agent") not in agent_module._PREPARING
        assert all(m.collect_text() != SUMMARY for m in root(context).memory)

    async def test_a_turn_under_the_ratio_prepares_nothing(self):
        summarizer = Summarizer()
        agent = make_agent(summarizer, steps=1, memory_compaction_prepare_ratio=0.3)  # 4 messages
        first, _ = await run(agent)
        assert (first.id, "agent") not in agent_module._PREPARING and summarizer.calls == []
