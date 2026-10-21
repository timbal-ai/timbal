import asyncio
from contextlib import aclosing

import pytest
from timbal import Agent, Tool, suspend
from timbal.core.test_model import TestModel
from timbal.state import get_run_context, set_call_id, set_parent_call_id, set_run_context
from timbal.state.background import BackgroundTaskStore, store_for_run
from timbal.state.tracing.providers.in_memory import InMemoryTracingProvider
from timbal.state.tracing.providers.jsonl import JsonlTracingProvider
from timbal.types.content import ToolUseContent
from timbal.types.events import OutputEvent
from timbal.types.message import Message


@pytest.fixture(autouse=True)
def isolated_context():
    set_run_context(None)
    set_call_id(None)
    set_parent_call_id(None)
    yield
    set_run_context(None)
    set_call_id(None)
    set_parent_call_id(None)


def tool_calls(*names):
    return Message(
        role="assistant",
        content=[ToolUseContent(id=f"call-{name}", name=name, input={}) for name in names],
        stop_reason="tool_use",
    )


def make_agent(release, *, tracing_provider=InMemoryTracingProvider, failure=False):
    async def work() -> str:
        await release.wait()
        if failure:
            raise ValueError("job failed")
        return "result-42"

    model = TestModel(responses=[tool_calls("work"), "Working.", "Finished."])
    agent = Agent(
        name="agent",
        model=model,
        tools=[Tool(name="work", handler=work, background_mode="always")],
        tracing_provider=tracing_provider,
    )
    return agent, model


def root_outputs(events):
    return [event for event in events if isinstance(event, OutputEvent) and event.parent_call_id is None]


async def consume(stream):
    async with aclosing(stream):
        return [event async for event in stream]


async def through_output(stream):
    async for event in stream:
        if isinstance(event, OutputEvent) and event.parent_call_id is None:
            return event
    raise AssertionError("No root output")


async def settle(store):
    await asyncio.wait_for(asyncio.gather(*(record.task for record in store._tasks.values())), 2)
    await asyncio.sleep(0)


async def test_no_background_tasks_finishes_without_extra_model_call():
    model = TestModel(responses=["Hello"])
    stream = Agent(name="agent", model=model).stream_until_idle(prompt="hi")
    events = await consume(stream)
    assert len(root_outputs(events)) == 1
    assert model.call_count == 1
    assert stream.stop_reason == "idle"
    assert stream.continuations == 0
    assert store_for_run(stream.last_run_id) is None


@pytest.mark.parametrize("provider_kind", ["memory", "jsonl"])
@pytest.mark.parametrize("failure", [False, True])
async def test_completion_starts_new_run_without_rewriting_history(tmp_path, provider_kind, failure):
    provider = (
        InMemoryTracingProvider
        if provider_kind == "memory"
        else JsonlTracingProvider.configured(_path=tmp_path / "trace.jsonl")
    )
    release = asyncio.Event()
    agent, model = make_agent(release, tracing_provider=provider, failure=failure)
    stream = agent.stream_until_idle(prompt="start")
    initial = await through_output(stream)
    initial_context = get_run_context()
    initial_memory = [message.collect_text() for message in initial_context.root_span().memory]
    release.set()
    events = await asyncio.wait_for(consume(stream), 3)
    outputs = root_outputs(events)
    assert len(outputs) == 1
    assert outputs[0].run_id != initial.run_id
    assert outputs[0].parent_run_id == initial.run_id
    assert outputs[0].metadata["trigger"] == {"kind": "background_completion", "continuation_index": 1}
    assert model.call_count == 3
    assert stream.stop_reason == "idle"
    assert stream.continuations == 1
    context = get_run_context()
    stored = await initial_context._get_parent_trace(outputs[0].run_id)
    memory = [Message.validate(message) for message in stored[outputs[0].call_id].memory]
    assert [message.collect_text() for message in memory if message.role == "user" and not message.is_runtime()] == [
        "start"
    ]
    notices = [message for message in memory if message.is_runtime()]
    assert len(notices) == 1
    assert ("job failed" if failure else "result-42") in notices[0].collect_text()
    assert [message.collect_text() for message in initial_context.root_span().memory] == initial_memory
    assert store_for_run(initial.run_id) is store_for_run(outputs[0].run_id)
    assert context is not None


async def test_completion_during_model_response_does_not_overlap():
    release = asyncio.Event()
    response_started = asyncio.Event()
    release_response = asyncio.Event()

    class GatedModel(TestModel):
        async def stream(self, messages, **kwargs):
            if self.call_count == 1:
                response_started.set()
                await release_response.wait()
            async for message in super().stream(messages, **kwargs):
                yield message

    agent, _ = make_agent(release)
    model = GatedModel(responses=[tool_calls("work"), "Working", "Finished"])
    agent.model = model
    stream = agent.stream_until_idle(prompt="start")
    consumer = asyncio.create_task(consume(stream))
    await asyncio.wait_for(response_started.wait(), 2)
    store = store_for_run(stream.last_run_id)
    release.set()
    await settle(store)
    assert model.call_count == 1
    assert stream.continuations == 0
    release_response.set()
    events = await asyncio.wait_for(consumer, 3)
    assert len(root_outputs(events)) == 2
    assert model.call_count == 3


async def test_two_completions_coalesce_into_one_turn():
    release = asyncio.Event()

    async def first() -> str:
        await release.wait()
        return "first-result"

    async def second() -> str:
        await release.wait()
        return "second-result"

    model = TestModel(responses=[tool_calls("first", "second"), "Working", "Both finished"])
    agent = Agent(
        name="agent",
        model=model,
        tools=[Tool(handler=first, background_mode="always"), Tool(handler=second, background_mode="always")],
    )
    stream = agent.stream_until_idle(prompt="start")
    initial = await through_output(stream)
    store = store_for_run(initial.run_id)
    release.set()
    await settle(store)
    events = await consume(stream)
    assert len(root_outputs(events)) == 1
    assert model.call_count == 3
    assert not store.pending_completions()


async def test_timeout_leaves_child_running_and_notice_for_next_user_turn():
    release = asyncio.Event()
    agent, model = make_agent(release)
    stream = agent.stream_until_idle(prompt="start", max_idle_secs=0.01)
    events = await consume(stream)
    parent = root_outputs(events)[0].run_id
    store = store_for_run(parent)
    assert stream.stop_reason == "idle_timeout"
    assert store.running_count() == 1
    assert store._continuation_owner is None
    release.set()
    await settle(store)
    assert store.pending_completions()
    await agent(prompt="status", parent_id=parent).collect()
    assert model.call_count == 3
    assert not store.pending_completions()


async def test_continuation_limit_does_not_drain_pending_notices():
    release = asyncio.Event()
    agent, model = make_agent(release)
    stream = agent.stream_until_idle(prompt="start", max_continuations=0)
    initial = await through_output(stream)
    release.set()
    store = store_for_run(initial.run_id)
    await settle(store)
    await consume(stream)
    assert stream.stop_reason == "continuation_limit"
    assert model.call_count == 2
    assert len(store.pending_completions()) == 1


async def test_cancel_waiting_driver_does_not_cancel_child():
    release = asyncio.Event()
    agent, model = make_agent(release)
    stream = agent.stream_until_idle(prompt="start")
    initial = await through_output(stream)
    consumer = asyncio.create_task(consume(stream))
    await asyncio.sleep(0)
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer
    store = store_for_run(initial.run_id)
    assert stream.stop_reason == "cancelled"
    assert store.running_count() == 1
    assert store._continuation_owner is None
    release.set()
    await settle(store)
    assert model.call_count == 2
    assert store.pending_completions()


async def test_close_after_output_releases_owner_without_cancelling_child():
    release = asyncio.Event()
    agent, _ = make_agent(release)
    stream = agent.stream_until_idle(prompt="start")
    initial = await through_output(stream)
    await stream.aclose()
    store = store_for_run(initial.run_id)
    assert stream.stop_reason == "closed"
    assert store._continuation_owner is None
    assert store.running_count() == 1
    release.set()
    await settle(store)


async def test_duplicate_driver_rejected_without_releasing_first_owner():
    release = asyncio.Event()
    agent, model = make_agent(release)
    first = agent.stream_until_idle(prompt="start")
    initial = await through_output(first)
    second = agent.stream_until_idle(prompt="other", parent_id=initial.run_id)
    with pytest.raises(RuntimeError, match="active continuation stream"):
        await consume(second)
    store = store_for_run(initial.run_id)
    assert store._continuation_owner is first._owner
    assert model.call_count == 2
    await first.aclose()
    release.set()
    await settle(store)


async def test_direct_turn_rejected_while_driver_owns_session():
    release = asyncio.Event()
    agent, model = make_agent(release)
    hook_calls = []

    def before_run():
        hook_calls.append("called")

    agent = Agent(name="agent", model=model, tools=agent.tools, pre_hook=before_run)
    stream = agent.stream_until_idle(prompt="start")
    initial = await through_output(stream)
    output = await agent(prompt="racing user", parent_id=initial.run_id).collect()
    assert output.status.code == "error"
    assert "Stop the session" in output.error["message"]
    assert model.call_count == 2
    assert hook_calls == ["called"]
    await stream.aclose()
    release.set()
    await settle(store_for_run(initial.run_id))


async def test_pause_stops_driver_without_auto_resuming_tools():
    calls = []

    def ask() -> str:
        calls.append("ask")
        return suspend({"question": "Continue?"})

    model = TestModel(responses=[tool_calls("ask"), "Resumed"])
    agent = Agent(name="agent", model=model, tools=[Tool(handler=ask)])
    stream = agent.stream_until_idle(prompt="start")
    events = await consume(stream)
    assert root_outputs(events)[0].status.reason == "input_required"
    assert stream.stop_reason == "paused"
    assert calls == ["ask"]
    assert model.call_count == 1


async def test_error_stops_driver():
    def fail(_messages):
        raise ValueError("model failed")

    stream = Agent(name="agent", model=TestModel(handler=fail)).stream_until_idle(prompt="start")
    events = await consume(stream)
    assert root_outputs(events)[0].status.code == "error"
    assert stream.stop_reason == "run_stopped"


async def test_missing_history_leaves_notice_pending():
    class MissingHistory(InMemoryTracingProvider):
        @classmethod
        async def get(cls, _context):
            return None

    release = asyncio.Event()
    agent, model = make_agent(release, tracing_provider=MissingHistory)
    stream = agent.stream_until_idle(prompt="start")
    initial = await through_output(stream)
    release.set()
    await consume(stream)
    assert stream.stop_reason == "history_unavailable"
    assert model.call_count == 2
    assert store_for_run(initial.run_id).pending_completions()


async def test_disabled_tracing_rejected_before_model_execution():
    model = TestModel(responses=["unused"])
    stream = Agent(name="agent", model=model, tracing_provider=None).stream_until_idle(prompt="start")
    with pytest.raises(ValueError, match="tracing provider"):
        await consume(stream)
    assert model.call_count == 0


async def test_acknowledged_completion_does_not_start_followup():
    release = asyncio.Event()
    agent, model = make_agent(release)
    stream = agent.stream_until_idle(prompt="start")
    initial = await through_output(stream)
    store = store_for_run(initial.run_id)
    release.set()
    await settle(store)
    task_id = store.list()[0]["task_id"]
    store.ack_completion(task_id)
    await consume(stream)
    assert stream.stop_reason == "idle"
    assert model.call_count == 2


async def test_waiter_wakes_when_already_acknowledged_task_finishes():
    release = asyncio.Event()
    agent, _ = make_agent(release)
    stream = agent.stream_until_idle(prompt="start")
    initial = await through_output(stream)
    store = store_for_run(initial.run_id)
    task_id = store.list()[0]["task_id"]
    store.ack_completion(task_id)
    waiter = asyncio.create_task(store.wait_for_completion())
    await asyncio.sleep(0)
    release.set()
    assert await asyncio.wait_for(waiter, 2) is False
    await stream.aclose()


async def test_empty_store_wait_returns_immediately():
    assert await BackgroundTaskStore().wait_for_completion() is False


async def test_cancellation_during_continuation_is_not_swallowed():
    release = asyncio.Event()
    continuation_started = asyncio.Event()

    class HangingModel(TestModel):
        async def stream(self, messages, **kwargs):
            if self.call_count == 2:
                continuation_started.set()
                await asyncio.Event().wait()
            async for message in super().stream(messages, **kwargs):
                yield message

    agent, _ = make_agent(release)
    agent.model = HangingModel(responses=[tool_calls("work"), "Working", "Finished"])
    stream = agent.stream_until_idle(prompt="start")

    async def run():
        async with aclosing(stream):
            async for event in stream:
                if isinstance(event, OutputEvent) and event.parent_call_id is None:
                    release.set()

    consumer = asyncio.create_task(run())
    await asyncio.wait_for(continuation_started.wait(), 3)
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(consumer, 3)
    assert stream.stop_reason == "cancelled"
    assert store_for_run(stream.last_run_id)._continuation_owner is None


async def test_two_sessions_using_one_agent_are_isolated():
    release = asyncio.Event()
    agent, model = make_agent(release)

    async def run(prompt):
        set_run_context(None)
        stream = agent.stream_until_idle(prompt=prompt)
        first = await through_output(stream)
        store = store_for_run(first.run_id)
        release.set()
        events = await consume(stream)
        return store, first.run_id, root_outputs(events)[0].parent_run_id

    first, second = await asyncio.wait_for(asyncio.gather(run("first"), run("second")), 3)
    assert first[0] is not second[0]
    assert first[1] == first[2]
    assert second[1] == second[2]
    assert model.call_count == 6


async def test_approval_with_background_work_does_not_autoresume():
    release = asyncio.Event()

    async def work() -> str:
        await release.wait()
        return "done"

    def gated() -> str:
        raise AssertionError("Approval was bypassed")

    model = TestModel(responses=[tool_calls("work"), tool_calls("gated"), "Unexpected"])
    agent = Agent(
        name="agent",
        model=model,
        tools=[Tool(handler=work, background_mode="always"), Tool(handler=gated, requires_approval=True)],
    )
    stream = agent.stream_until_idle(prompt="start")
    events = await consume(stream)
    assert stream.stop_reason == "paused"
    assert root_outputs(events)[0].status.reason == "approval_required"
    release.set()
    store = store_for_run(stream.last_run_id)
    await settle(store)
    assert store.pending_completions()
    assert model.call_count == 2


async def test_completion_acknowledged_during_history_lookup_does_not_repeat_prompt():
    class AckOnRead(InMemoryTracingProvider):
        @classmethod
        async def get(cls, context):
            store = store_for_run(context.parent_id)
            if store is not None:
                for notice in store.pending_completions():
                    store.ack_completion(notice["task_id"])
            return await super().get(context)

    release = asyncio.Event()
    agent, model = make_agent(release, tracing_provider=AckOnRead)
    stream = agent.stream_until_idle(prompt="start")
    await through_output(stream)
    release.set()
    await consume(stream)
    assert model.call_count == 2
    assert stream.stop_reason == "idle"


async def test_followup_can_start_more_background_work_but_limit_bounds_loop():
    release = asyncio.Event()
    release.set()
    agent, _ = make_agent(release)
    agent.model = TestModel(
        responses=[tool_calls("work"), "Working", tool_calls("work"), "Working again", "Unexpected"]
    )
    stream = agent.stream_until_idle(prompt="start", max_continuations=1)
    events = await consume(stream)
    assert len(root_outputs(events)) == 2
    assert stream.stop_reason == "continuation_limit"
    assert agent.model.call_count == 4
    store = store_for_run(stream.last_run_id)
    await settle(store)
    assert store.pending_completions()


async def test_invalid_parent_rejected_before_model_execution():
    model = TestModel(responses=["Unused"])
    stream = Agent(name="agent", model=model).stream_until_idle(prompt="start", parent_id="missing")
    with pytest.raises(ValueError, match="completed, readable parent"):
        await consume(stream)
    assert model.call_count == 0


async def test_ack_between_admission_and_context_attachment_skips_model_call():
    release = asyncio.Event()
    agent, model = make_agent(release)

    def acknowledge_before_model():
        context = get_run_context()
        if getattr(context, "_background_continuation", False):
            store = store_for_run(context.id)
            for notice in store.pending_completions():
                store.ack_completion(notice["task_id"])

    agent = Agent(name="agent", model=model, tools=agent.tools, pre_hook=acknowledge_before_model)
    stream = agent.stream_until_idle(prompt="start")
    await through_output(stream)
    release.set()
    events = await consume(stream)
    assert root_outputs(events)[0].metadata["continuation_skipped"] == "no_pending_completion"
    assert model.call_count == 2
    assert stream.stop_reason == "idle"


@pytest.mark.parametrize("seconds", [0, -1, float("inf"), float("nan")])
def test_invalid_idle_limit(seconds):
    with pytest.raises(ValueError, match="max_idle_secs"):
        Agent(name="agent", model=TestModel()).stream_until_idle(prompt="start", max_idle_secs=seconds)


@pytest.mark.parametrize("limit", [-1, 1.5, True])
def test_invalid_continuation_limit(limit):
    with pytest.raises(ValueError, match="max_continuations"):
        Agent(name="agent", model=TestModel()).stream_until_idle(prompt="start", max_continuations=limit)
