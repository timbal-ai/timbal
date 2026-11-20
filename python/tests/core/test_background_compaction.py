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


def stepping(steps_for):
    """A model that makes ``steps_for[prompt]`` tool calls in the turn that prompt opens, then answers ``done``."""
    left = made = 0

    def handler(messages):
        nonlocal left, made
        last = messages[-1]
        if last.role == "user" and last.collect_text() in steps_for:
            left = steps_for[last.collect_text()]
        if not left:
            return "done"
        left, made = left - 1, made + 1
        return Message(
            role="assistant", content=[ToolUseContent(id=f"c{made}", name="step", input={})], stop_reason="tool_use"
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
    handoff = agent_module._PREPARING.get((context.id, "agent"))
    if handoff is not None:
        await handoff.stored


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

    async def test_nested_metadata_is_copied_too(self):
        async def edit_then_fail(memory):
            memory[0].metadata["nested"]["seen"] = "EDITED"
            raise RuntimeError("summarizer down")

        agent = Agent(
            name="a", model=TestModel(), memory_compaction=edit_then_fail, memory_compaction_prepare_ratio=0.3
        )
        live = [Message(role="user", content=[TextContent(text="hi")], metadata={"nested": {"seen": "original"}})]
        prepared = agent._prepare_compaction(SimpleNamespace(input={}), live, 0.4)
        with pytest.raises(RuntimeError):
            await prepared.task
        assert live[0].metadata["nested"]["seen"] == "original"


class TestTheLoopAndTheTurnAreNotHeld:
    async def test_a_cancelled_background_compaction_does_not_cancel_the_turn(self):
        """At the hard ratio the turn waits for the compaction being prepared. That one being cancelled (something
        it awaited was) is a preparation that came to nothing: the turn compacts inline and goes on."""
        loop = asyncio.get_running_loop()
        dependency, sizes, steps = loop.create_future(), [], 0

        async def summarize_old(memory):
            sizes.append(len(memory))
            if len(sizes) == 1:
                await dependency  # the background one: cancelled while the turn waits for it
            if len(memory) <= 3:
                return memory
            return [Message(role="user", content=[TextContent(text=SUMMARY)]), *memory[-2:]]

        async def last_step_cancels() -> str:
            nonlocal steps
            steps += 1
            if steps == 7:  # the result that takes the turn to the hard ratio: it will wait for the compaction
                loop.call_later(0.05, dependency.cancel)
            return "ok"

        agent = Agent(
            name="agent",
            model=TestModel(handler=stepping({"go": 7})),  # 15 messages after the seventh result
            tools=[Tool(name="step", handler=last_step_cancels)],
            tracing_provider=InMemoryTracingProvider,
            memory_compaction=summarize_old,
            memory_compaction_prepare_ratio=0.3,
            max_iter=30,
        )
        context, output = await run(agent)  # asserts the turn ended in success
        assert len(sizes) == 2 and root(context).metadata["compaction"].get("background") is None
        assert any(message.collect_text() == SUMMARY for message in root(context).memory)

    async def test_a_synchronous_strategy_prepared_in_the_background_does_not_hold_the_loop(self):
        """Prepared on a task, a strategy that is a plain function would still run on the loop's thread and
        stop every other turn for as long as it takes: it goes to a worker thread."""
        import time

        ended_at = []

        def slow(memory):
            time.sleep(0.2)
            ended_at.append(time.monotonic())
            return memory[-2:]

        agent = Agent(name="agent", model=TestModel(), memory_compaction=slow, memory_compaction_prepare_ratio=0.3)
        live = [Message(role="user", content=[TextContent(text=f"m{i}")]) for i in range(6)]
        loop = asyncio.get_running_loop()
        fired = loop.create_future()
        loop.call_later(0.01, lambda: fired.set_result(time.monotonic()))
        prepared = agent._prepare_compaction(SimpleNamespace(input={}), live, 0.4)
        compacted, _steps = await prepared.task
        assert await fired < ended_at[0]  # the loop ran its timer while the strategy was still at work
        assert [message.collect_text() for message in compacted] == ["m4", "m5"]

    async def test_a_strategy_still_on_its_thread_keeps_the_resumed_turn_from_starting_another(self):
        """A turn pauses for an approval while a plain-function strategy is at work on a worker thread. The task
        waiting for it is cancelled; the thread is not, and cannot be. The resumed turn passes the prepare ratio
        again: it starts no second background compaction while the first is still running."""
        import threading

        from timbal.types.events import ApprovalEvent

        lock, release = threading.Lock(), threading.Event()
        active = peak = 0
        sizes = []

        def slow(memory):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
                sizes.append(len(memory))
            try:
                release.wait(timeout=5)
            finally:
                with lock:
                    active -= 1
            return memory if len(memory) <= 3 else memory[-2:]

        def model(messages):
            results = sum(1 for message in messages if message.role == "tool")
            if results < 4:  # the fourth step runs with 7 messages behind it: the compaction is at work by then
                return Message(
                    role="assistant",
                    content=[ToolUseContent(id=f"s{results}", name="step", input={})],
                    stop_reason="tool_use",
                )
            if results == 4:
                return Message(
                    role="assistant",
                    content=[ToolUseContent(id="pay", name="charge", input={})],
                    stop_reason="tool_use",
                )
            return "done"

        agent = Agent(
            name="agent",
            model=TestModel(handler=model),
            tools=[
                Tool(name="step", handler=step),
                Tool(name="charge", handler=lambda: "charged", requires_approval=True),
            ],
            tracing_provider=InMemoryTracingProvider,
            memory_compaction=slow,
            memory_compaction_prepare_ratio=0.3,
            max_iter=30,
        )

        async def go(**kwargs):
            context = RunContext(parent_id=kwargs.pop("parent_id", None), tracing_provider=InMemoryTracingProvider)
            set_run_context(context)
            events = [event async for event in agent(prompt="go", **kwargs)]
            set_run_context(None)
            return context, events

        try:
            paused, events = await go()
            (card,) = [event for event in events if isinstance(event, ApprovalEvent)]
            assert sizes and active == 1 and (paused.id, "agent") in agent_module._PREPARING  # still on its thread
            resumed, events = await go(parent_id=paused.id, resume={card.approval_id: True})
            await asyncio.sleep(0.05)  # anything the resumed turn started would be at work by now
            assert peak == 1 and len(sizes) == 1
        finally:
            release.set()
        for handoff in list(agent_module._PREPARING.values()):
            await handoff.stored
        assert active == 0 and not agent_module._PREPARING


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
        handoff = agent_module._Handoff(record=asyncio.get_running_loop().create_future())
        agent_module._PREPARING[(first.id, "agent")] = handoff
        stored = await RunContext(parent_id=first.id, tracing_provider=InMemoryTracingProvider)._get_parent_trace()
        record = stored[root(first).call_id].session[agent_module._PREPARED_KEY]["agent"]
        handoff.record.set_result(record)
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
        handoff = agent_module._PREPARING[(first.id, "agent")]
        second, _ = await asyncio.wait_for(run(agent, prompt="again", parent_id=first.id), timeout=2)
        assert not handoff.record.done()  # never waited for
        assert root(second).memory[0].collect_text() != SUMMARY
        summarizer.release.set()
        await handoff.stored

    async def test_quick_turns_share_one_background_compaction(self):
        """Turns that do not wait for the one before take its result on instead of starting their own."""
        summarizer = Summarizer(released=False)
        agent = make_agent(summarizer, steps=2, memory_compaction_prepare_ratio=0.3)
        context, _ = await run(agent)
        for _ in range(3):  # each below the hard ratio: none waits
            context, _ = await run(agent, prompt="again", parent_id=context.id)
        await asyncio.sleep(0.05)  # let every summarizer that was started begin
        assert len(summarizer.calls) == 1
        summarizer.release.set()
        await settled(context)
        last, _ = await run(agent, prompt="again", parent_id=context.id)
        assert root(last).memory[0].collect_text() == SUMMARY  # the one result reached the latest turn
        assert len(summarizer.calls) == 1

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
            result = loop.create_future()
            agent_module._PREPARING[("parent", "agent")] = agent_module._Handoff(record=result)
            loop.call_later(0.2, lambda f=result: f.done() or f.set_result(None))
            started = loop.time()
            await agent._take_prepared(run_context, source, memory, "test", incoming)
            assert (loop.time() - started >= 0.15) is waits
            await result

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

    @pytest.mark.parametrize(
        "prompt",
        ["go", Message(role="user", content=[TextContent(text=""), TextContent(text="go")])],
        ids=["a plain prompt", "a prompt with an empty text block"],
    )
    async def test_a_turn_crossing_the_hard_ratio_takes_the_one_before_instead_of_compacting_inline(self, prompt):
        """Turn two starts below the hard ratio, so it does not wait; its tool results then take it past. It takes
        the compaction turn one is still preparing, not a second one of the same history next to it. Also when
        loading the history cleaned a message of it (a prompt stored with an empty text block): the turn then
        holds another object than the one it loaded."""
        running = peak = 0
        release, sizes = asyncio.Event(), []

        async def summarize_old(memory):
            nonlocal running, peak
            sizes.append(len(memory))
            running += 1
            peak = max(peak, running)
            try:
                await release.wait()
            finally:
                running -= 1
            if len(memory) <= 3:
                return memory
            return [Message(role="user", content=[TextContent(text=SUMMARY)]), *memory[-2:]]

        agent = Agent(
            name="agent",
            model=TestModel(handler=stepping({"go": 2, "more": 4})),
            tools=[Tool(name="step", handler=step)],
            tracing_provider=InMemoryTracingProvider,
            memory_compaction=summarize_old,
            memory_compaction_prepare_ratio=0.3,
            max_iter=30,
        )
        first, _ = await run(agent, prompt=prompt)  # 6 messages: past the prepare ratio, held at the summarizer
        asyncio.get_running_loop().call_later(0.3, release.set)
        second, output = await run(agent, prompt="more", parent_id=first.id)  # 7, then 15 after four tool results
        assert peak == 1 and sizes[0] == 6
        assert root(second).memory[0].collect_text() == SUMMARY
        assert root(second).metadata["compaction"]["prepared_by"] == first.id
        assert output.usage.get("compaction:background") == 1
        await settled(second)

    async def test_a_turn_ending_at_the_hard_ratio_takes_the_one_before_on(self):
        """Turn two never waits and stays below the hard ratio at each step; its answer takes it there. It takes
        turn one's compaction on, so turn three waits for that one instead of compacting inline next to it."""
        running = peak = 0
        release, sizes = asyncio.Event(), []

        async def summarize_old(memory):
            nonlocal running, peak
            sizes.append(len(memory))
            running += 1
            peak = max(peak, running)
            try:
                await release.wait()
            finally:
                running -= 1
            if len(memory) <= 3:
                return memory
            return [Message(role="user", content=[TextContent(text=SUMMARY)]), memory[-1]]

        # Turn one: a batch of two calls, one more, the answer (7 messages). Turn two: three single calls.
        plan, left, made = {"go": [2, 1], "more": [1, 1, 1], "again": []}, [], 0

        def model(messages):
            nonlocal left, made
            last = messages[-1]
            if last.role == "user" and last.collect_text() in plan:
                left = list(plan[last.collect_text()])
            if not left:
                return "done"
            calls = []
            for _ in range(left.pop(0)):
                made += 1
                calls.append(ToolUseContent(id=f"c{made}", name="step", input={}))
            return Message(role="assistant", content=calls, stop_reason="tool_use")

        agent = Agent(
            name="agent",
            model=TestModel(handler=model),
            tools=[Tool(name="step", handler=step)],
            tracing_provider=InMemoryTracingProvider,
            memory_compaction=summarize_old,
            memory_compaction_prepare_ratio=0.3,
            max_iter=30,
        )
        first, _ = await run(agent)
        second, _ = await run(agent, prompt="more", parent_id=first.id)  # 8, then 10, 12, 14; 15 with the answer
        assert len(root(second).memory) == 15 and (second.id, "agent") in agent_module._PREPARING
        asyncio.get_running_loop().call_later(0.3, release.set)
        third, _ = await run(agent, prompt="again", parent_id=second.id)  # 16 at its start: waits, does not compact
        assert peak == 1 and len(sizes) == 1  # turn one's, started mid-turn; nothing next to it
        assert root(third).memory[0].collect_text() == SUMMARY
        assert root(third).metadata["compaction"]["prepared_by"] == second.id
        for handoff in list(agent_module._PREPARING.values()):
            await handoff.stored

    @pytest.mark.parametrize(
        "prompt, own",
        [("go", False), (Message(role="user", content=[TextContent(text=""), TextContent(text="go")]), True)],
        ids=["it still covers the history", "loading cleaned a message it covers"],
    )
    async def test_a_turn_holding_the_one_before_after_that_run_is_stored(self, prompt, own):
        """Turn two did not wait. During it, turn one's compaction finishes and its run is stored, so it is gone
        from ``_PREPARING``; turn two still holds it and starts none of its own meanwhile. At its end it takes
        that one on for turn three, or, when it no longer matches turn two's history as stored (a prompt with an
        empty text block, cleaned on load), prepares its own: turn three gets a prepared compaction either way."""
        summarizer = Summarizer(released=False)
        before: list = []

        async def step_and_settle() -> str:
            if before:  # turn two's step: the run before finishes its compaction and is stored meanwhile
                summarizer.release.set()
                await settled(before[0])
                assert (before[0].id, "agent") not in agent_module._PREPARING
            return "ok"

        agent = Agent(
            name="agent",
            model=TestModel(handler=stepping({"go": 2, "more": 1})),
            tools=[Tool(name="step", handler=step_and_settle)],
            tracing_provider=InMemoryTracingProvider,
            memory_compaction=summarizer.compactor(),
            memory_compaction_prepare_ratio=0.3,
            max_iter=30,
        )
        first, _ = await run(agent, prompt=prompt)  # 6 messages, its compaction held at the summarizer
        before.append(first)
        second, _ = await run(agent, prompt="more", parent_id=first.id)  # 7, then 9, 10 with the answer
        await settled(second)
        assert summarizer.calls == ([6, 10] if own else [6])
        third, _ = await run(agent, prompt="again", parent_id=second.id)
        assert root(third).memory[0].collect_text() == SUMMARY
        assert root(third).metadata["compaction"]["prepared_by"] == second.id
        assert len(summarizer.calls) == (2 if own else 1)

    async def test_a_summary_that_is_there_is_taken_before_its_run_is_stored(self, monkeypatch):
        """The turn at the hard ratio waits for the summary, not for the trace of the run that made it."""
        real_store = RunContext._end_background_work

        async def slow_store(self):
            await asyncio.sleep(0.5)
            await real_store(self)

        monkeypatch.setattr(RunContext, "_end_background_work", slow_store)
        summarizer = Summarizer()
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
        loop = asyncio.get_running_loop()
        started = loop.time()
        second, _ = await run(agent, prompt="again", parent_id=first.id)
        assert root(second).memory[0].collect_text() == SUMMARY
        assert loop.time() - started < 0.3 and (first.id, "agent") in agent_module._PREPARING  # not stored yet
        await settled(first)

    async def test_a_turn_that_took_the_summary_prepares_its_own_for_the_next(self, monkeypatch):
        """The run before is still being stored when the turn that took its summary ends past the prepare ratio:
        that summary is used up, so the turn compacts its own history for the next one."""
        real_store = RunContext._end_background_work

        async def slow_store(self):
            await asyncio.sleep(0.5)
            await real_store(self)

        monkeypatch.setattr(RunContext, "_end_background_work", slow_store)
        summarizer = Summarizer()
        agent = Agent(
            name="agent",
            model=TestModel(handler=stepping({"go": 2, "more": 2})),
            tools=[Tool(name="step", handler=step)],
            tracing_provider=InMemoryTracingProvider,
            memory_compaction=summarizer.compactor(),
            memory_compaction_prepare_ratio=0.3,
            max_iter=30,
        )
        first, _ = await run(agent)
        real = Agent._take_prepared

        async def at_the_limit(self, run_context, source, memory, model, incoming):
            monkeypatch.setattr(Agent, "_context_utilization", lambda *_args: 0.8)
            try:
                return await real(self, run_context, source, memory, model, incoming)
            finally:
                monkeypatch.setattr(Agent, "_context_utilization", lambda _s, _m, messages, _x: 0.05 * len(messages))

        monkeypatch.setattr(Agent, "_take_prepared", at_the_limit)
        second, _ = await run(agent, prompt="more", parent_id=first.id)  # the summary and 2, then 6 more
        monkeypatch.setattr(Agent, "_take_prepared", real)
        assert root(second).memory[0].collect_text() == SUMMARY
        assert (first.id, "agent") in agent_module._PREPARING  # the run before: still being stored
        assert (second.id, "agent") in agent_module._PREPARING  # and this turn's own, for the next
        await settled(second)
        assert summarizer.calls[0] == 6 and len(summarizer.calls) > 1
        third, _ = await run(agent, prompt="again", parent_id=second.id)
        assert root(third).metadata["compaction"]["prepared_by"] == second.id
        await settled(first)

    async def test_a_summary_taken_on_does_not_wait_for_the_run_before_to_be_stored(self, monkeypatch):
        """Turn two took turn one's compaction on for turn three. Turn three, at the hard ratio, gets the summary
        as soon as it exists: neither run's trace write holds it."""
        real_store = RunContext._end_background_work

        async def slow_store(self):
            await asyncio.sleep(0.5)
            await real_store(self)

        monkeypatch.setattr(RunContext, "_end_background_work", slow_store)
        summarizer = Summarizer(released=False)
        agent = make_agent(summarizer, steps=2, memory_compaction_prepare_ratio=0.3)
        first, _ = await run(agent)
        second, _ = await run(agent, prompt="again", parent_id=first.id)  # below the hard ratio: takes it on
        await asyncio.sleep(0.05)  # let every summarizer that was started begin
        assert len(summarizer.calls) == 1
        summarizer.release.set()
        real = Agent._take_prepared

        async def at_the_limit(self, run_context, source, memory, model, incoming):
            monkeypatch.setattr(Agent, "_context_utilization", lambda *_args: 0.8)
            try:
                return await real(self, run_context, source, memory, model, incoming)
            finally:
                monkeypatch.setattr(Agent, "_context_utilization", lambda _s, _m, messages, _x: 0.05 * len(messages))

        monkeypatch.setattr(Agent, "_take_prepared", at_the_limit)
        loop = asyncio.get_running_loop()
        started = loop.time()
        third, _ = await run(agent, prompt="again", parent_id=second.id)
        assert root(third).memory[0].collect_text() == SUMMARY and len(summarizer.calls) == 1
        assert loop.time() - started < 0.3
        await settled(first)
        await settled(second)

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
