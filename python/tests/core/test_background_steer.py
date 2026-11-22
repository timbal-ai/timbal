"""Steering background sub-agents: messages a running background agent reads between its steps."""

import asyncio
import threading
import time

import pytest
from timbal import Agent, Tool
from timbal.core.test_model import TestModel
from timbal.guardrails import guardrail
from timbal.state import get_run_context, list_background_tasks, message_background_task
from timbal.state.background import (
    BackgroundTask,
    BackgroundTaskStore,
    format_background_completion_notice,
    store_for_run,
)
from timbal.types.content import ToolUseContent
from timbal.types.message import Message


def _call(name: str, input: dict, id: str = "c1") -> Message:
    return Message(role="assistant", content=[ToolUseContent(id=id, name=name, input=input)], stop_reason="tool_use")


def _user_texts(messages: list[Message]) -> list[str]:
    return [m.collect_text() for m in messages if m.role == "user" and not m.is_runtime() and m.collect_text()]


class Child:
    """A researcher that runs one slow step, then answers with every instruction it has."""

    def __init__(self, steps: int = 1):
        self.steps = steps
        self.calls = 0
        self.started = asyncio.Event()
        self.gate = asyncio.Event()
        self.seen: list[list[Message]] = []

    async def slow(self) -> str:
        self.started.set()
        await self.gate.wait()
        self.gate.clear()
        return "step done"

    def model(self, messages: list[Message]) -> Message | str:
        self.seen.append(list(messages))
        self.calls += 1
        if self.calls <= self.steps:
            return _call("slow", {}, id=f"s{self.calls}")
        return "Did: " + " | ".join(_user_texts(messages))

    def agent(self, background_mode: str = "always", **kwargs) -> Agent:
        return Agent(
            name="researcher",
            model=TestModel(handler=self.model),
            tools=[Tool(name="slow", handler=self.slow)],
            background_mode=background_mode,
            **kwargs,
        )


async def _spawn(child: Child, parent_responses: list, **child_kwargs) -> tuple[Agent, str, str]:
    parent = Agent(
        name="lead",
        model=TestModel(responses=[_call("researcher", {"prompt": "research cats"}), *parent_responses]),
        tools=[child.agent(**child_kwargs)],
    )
    result = await parent(prompt="go").collect()
    assert result.status.code == "success"
    (task,) = list_background_tasks()
    await asyncio.wait_for(child.started.wait(), 2)
    return parent, result.run_id, task["task_id"]


async def _finish(run_id: str, task_id: str) -> dict:
    store = store_for_run(run_id)
    await asyncio.wait_for(asyncio.shield(store.get(task_id).task), 5)
    return store.snapshot(task_id)


class TestMessageTask:
    @pytest.mark.asyncio
    async def test_the_running_agent_reads_it_after_its_step(self):
        child = Child()
        _, run_id, task_id = await _spawn(child, ["Started."])
        store = store_for_run(run_id)
        assert store.list()[0]["takes_messages"] is True

        receipt = store.message_task(task_id, "focus on tigers")
        assert receipt.state == "pending"
        child.gate.set()
        snap = await _finish(run_id, task_id)

        assert receipt.state == "delivered"
        assert snap["status"] == "completed"
        assert _answer(snap) == "Did: research cats | focus on tigers"
        # It came in after the tool result, as a user message, before the answer.
        last = child.seen[-1]
        assert last[-1].role == "user" and last[-1].metadata.get("steered") is True
        assert last[-1].metadata["sources"] == ["app"]
        assert last[-2].content[0].type == "tool_result"

    @pytest.mark.asyncio
    async def test_an_agent_the_model_sends_to_the_background_reads_its_messages(self):
        # background_mode="auto": the model decides per call. Detached, it is a task from its first step.
        child = Child()
        parent = Agent(
            name="lead",
            model=TestModel(
                responses=[_call("researcher", {"prompt": "research cats", "run_in_background": True}), "Started."]
            ),
            tools=[child.agent(background_mode="auto")],
        )
        result = await parent(prompt="go").collect()
        (task,) = list_background_tasks()
        await asyncio.wait_for(child.started.wait(), 2)
        store = store_for_run(result.run_id)
        assert store.list()[0]["takes_messages"] is True

        receipt = store.message_task(task["task_id"], "focus on tigers")
        child.gate.set()
        snap = await _finish(result.run_id, task["task_id"])

        assert receipt.state == "delivered"
        assert _answer(snap) == "Did: research cats | focus on tigers"

    @pytest.mark.asyncio
    async def test_an_agent_running_in_the_foreground_is_never_moved_to_the_background(self):
        # Whether a child reads messages is decided as its turn starts, so it has to be a task by then.
        # It always is: an agent detaches only at its start ("always", or "auto" when the model asks).
        # `collector.background()` parks a streaming tool mid-call, never an agent mid-turn, so no
        # task exists whose agent started in the foreground with steering decided off.
        child = Child()
        parent = Agent(
            name="lead",
            model=TestModel(responses=[_call("researcher", {"prompt": "research cats"}), "Done."]),
            tools=[child.agent(background_mode="auto")],
        )
        run = parent(prompt="go")
        consumer = asyncio.create_task(run.collect())
        await asyncio.wait_for(child.started.wait(), 2)
        with pytest.raises(RuntimeError, match="No active foreground runnable"):
            await run.background()
        child.gate.set()
        result = await asyncio.wait_for(consumer, 5)

        assert result.status.code == "success"
        store = store_for_run(result.run_id)
        assert store is None or store.list() == []

    @pytest.mark.asyncio
    async def test_several_messages_join_as_one_and_restore_the_budget(self):
        child = Child(steps=2)
        _, run_id, task_id = await _spawn(child, ["Started."])
        store = store_for_run(run_id)
        first = store.message_task(task_id, "tigers")
        second = store.message_task(task_id, "and lions", source="user")
        child.gate.set()
        await asyncio.wait_for(_until(lambda: first.state == "delivered"), 2)
        child.gate.set()
        snap = await _finish(run_id, task_id)
        assert second.state == "delivered"
        assert _answer(snap) == "Did: research cats | tigers\n\nand lions"
        span = _task_span(run_id, task_id)
        assert span.metadata["task_messages"] == [first.id, second.id]

    @pytest.mark.asyncio
    async def test_a_message_the_task_never_read_is_reported_with_its_completion(self):
        child = Child()
        _, run_id, task_id = await _spawn(child, ["Started."])
        store = store_for_run(run_id)
        record = store.get(task_id)
        take = record.take_messages
        receipts = []

        def take_messages():
            taken = take()
            if child.calls > child.steps:  # its last look, before the final answer: the message lands after it
                receipts.append(store.message_task(task_id, "one more thing"))
            return taken

        record.take_messages = take_messages
        child.gate.set()
        await _finish(run_id, task_id)
        (receipt,) = receipts
        assert receipt.state == "undelivered"
        (note,) = store.pending_completions()
        assert note["unread_messages"] == ["one more thing"]
        text = format_background_completion_notice([note]).collect_text()
        assert "finished before reading your message: one more thing" in text

    @pytest.mark.asyncio
    async def test_a_message_accepted_from_another_thread_as_the_task_ends_is_in_its_notice(self):
        """Accepted while the task ran, by a thread that is not the loop's; the task's last step was
        already next in line. Whatever is accepted is read or reported: never dropped in between."""
        store = BackgroundTaskStore()
        gate = asyncio.Event()

        async def work() -> str:
            await gate.wait()
            return "done"

        record = BackgroundTask(
            "t1", name="researcher", input={}, task=asyncio.ensure_future(work()), started_at=int(time.time() * 1000)
        )
        record.steerable = True
        store.add(record, consume_reservation=False)
        await asyncio.sleep(0)  # at its gate
        gate.set()  # its last step is queued on the loop...
        receipts: list = []
        thread = threading.Thread(target=lambda: receipts.append(store.message_task("t1", "one more thing")))
        thread.start()
        thread.join(5)  # ...and the message came in behind it, the task still running
        (receipt,) = receipts
        await asyncio.wait_for(asyncio.shield(record.task), 5)
        await asyncio.wait_for(_until(lambda: store.pending_completions()), 5)
        assert receipt.state == "undelivered"
        (note,) = store.pending_completions()
        assert note["unread_messages"] == ["one more thing"]

    @pytest.mark.asyncio
    async def test_a_parent_that_polls_the_finished_task_is_told_what_it_never_read(self):
        """Polling a finished task takes the place of its completion notice (the notice is dropped):
        the unread message must be in what the poll returns, or the parent never learns of it."""
        child = Child()
        _, run_id, task_id = await _spawn(child, ["Started."])
        store = store_for_run(run_id)
        record = store.get(task_id)
        take = record.take_messages

        def take_messages():
            taken = take()
            if child.calls > child.steps:
                store.message_task(task_id, "one more thing")
            return taken

        record.take_messages = take_messages
        child.gate.set()
        await _finish(run_id, task_id)
        assert store.pending_completions()
        assert store.list(ack_completion=True)[0]["unread_messages"] == ["one more thing"]
        assert not store.pending_completions()  # the listing stood in for the notice
        assert store.snapshot(task_id, ack_completion=True)["unread_messages"] == ["one more thing"]

    @pytest.mark.asyncio
    async def test_a_poll_ahead_of_the_completion_notice_reports_it_too(self):
        """The task has ended and its notice is not made yet (done callbacks run in order): a poll in
        between replaces the notice, so it is the poll that says what was never read."""
        store = BackgroundTaskStore()
        gate = asyncio.Event()

        async def work() -> str:
            await gate.wait()
            return "done"

        task = asyncio.ensure_future(work())
        polled: list = []
        task.add_done_callback(lambda _: polled.append(store.snapshot("t1", ack_completion=True)))
        record = BackgroundTask("t1", name="researcher", input={}, task=task, started_at=int(time.time() * 1000))
        record.steerable = True
        store.add(record, consume_reservation=False)
        receipt = store.message_task("t1", "one more thing")
        gate.set()
        await asyncio.wait_for(asyncio.shield(task), 5)
        await asyncio.wait_for(_until(lambda: receipt.state != "pending"), 5)
        (snap,) = polled
        assert snap["status"] == "completed" and snap["unread_messages"] == ["one more thing"]
        assert receipt.state == "undelivered" and not store.pending_completions()
        with pytest.raises(ValueError, match="has finished"):
            store.message_task("t1", "too late")

    @pytest.mark.asyncio
    async def test_a_poll_of_a_cancelled_task_still_unwinding_reports_it_too(self):
        """A cancelled (timed out, stalled) task reads as ended before its asyncio task is done. A poll
        then is the last word its parent gets: it lists what was not read, and no message is taken after."""
        store = BackgroundTaskStore()

        async def work() -> None:
            await asyncio.Event().wait()

        record = BackgroundTask(
            "t1", name="researcher", input={}, task=asyncio.ensure_future(work()), started_at=int(time.time() * 1000)
        )
        record.steerable = True
        store.add(record, consume_reservation=False)
        await asyncio.sleep(0)
        receipt = store.message_task("t1", "one more thing")
        store.cancel("t1")
        assert not record.task.done()
        snap = store.snapshot("t1", ack_completion=True)
        assert snap["status"] == "cancelled" and snap["unread_messages"] == ["one more thing"]
        assert receipt.state == "undelivered" and record.take_messages() == []
        with pytest.raises(ValueError, match="has finished"):
            store.message_task("t1", "too late")
        await asyncio.wait({record.task}, timeout=5)
        assert record.task.done() and not store.pending_completions()

    @pytest.mark.asyncio
    async def test_a_message_taken_but_never_brought_into_memory_is_not_delivered(self):
        """The task took the message and was still screening it (an input guardrail) when it was
        cancelled: the model never had it. That is a message it never read, reported like any other."""
        screening = asyncio.Event()

        async def slow_rail(text: str) -> bool:
            if "one more thing" in text:
                screening.set()
                await asyncio.Event().wait()
            return True

        child = Child()
        _, run_id, task_id = await _spawn(child, ["Started."], guardrails=[guardrail(slow_rail, stages=["input"])])
        store = store_for_run(run_id)
        receipt = store.message_task(task_id, "one more thing")
        child.gate.set()  # its step ends: it takes the message and screens it
        await asyncio.wait_for(screening.wait(), 5)
        assert receipt.state == "pending"  # taken, not read yet
        store.cancel(task_id)
        await asyncio.wait({store.get(task_id).task}, timeout=5)
        await asyncio.wait_for(_until(lambda: store.pending_completions()), 5)
        (note,) = store.pending_completions()
        assert note["status"] == "cancelled" and note["unread_messages"] == ["one more thing"]
        assert receipt.state == "undelivered"
        # Nor counted as delivered: the span and the usage say what the receipt says.
        span = _task_span(run_id, task_id)
        assert "task_messages" not in span.metadata
        assert "background:messages_delivered" not in span.usage

    @pytest.mark.asyncio
    async def test_a_message_screened_down_to_nothing_was_not_read(self):
        """An input guardrail that redacts the whole message leaves nothing to put in memory, and no
        notice either: the task goes on without it, so its sender is told it was not read."""
        child = Child()
        strip = guardrail(lambda text: "" if "one more thing" in text else True, stages=["input"], action="redact")
        _, run_id, task_id = await _spawn(child, ["Started."], guardrails=[strip])
        store = store_for_run(run_id)
        receipt = store.message_task(task_id, "one more thing")
        child.gate.set()
        snap = await _finish(run_id, task_id)
        assert _answer(snap) == "Did: research cats"
        assert receipt.state == "undelivered"
        # Reported, by its id: the text a guardrail took out is in no report the parent's model reads.
        (unread,) = snap["unread_messages"]
        assert receipt.id in unread and "guardrail" in unread and "one more thing" not in unread
        (note,) = store.pending_completions()
        assert note["unread_messages"] == [unread]
        assert store.list()[0]["unread_messages"] == [unread]
        notice = format_background_completion_notice([note]).collect_text()
        assert receipt.id in notice and "one more thing" not in notice
        assert "task_messages" not in _task_span(run_id, task_id).metadata

    @pytest.mark.asyncio
    @pytest.mark.parametrize("outcome", ["delivered", "undelivered"])
    async def test_a_sender_waiting_on_another_loop_is_woken_when_its_receipt_settles(self, outcome):
        """A sender with a loop of its own (another thread) awaits ``receipt.settled()``. The receipt
        settles on the task's loop: the sender's loop has to be told, or it sleeps on until something
        else wakes it (here its own 20 s deadline)."""
        store = BackgroundTaskStore()
        gate = asyncio.Event()

        async def work() -> str:
            await gate.wait()
            if outcome == "delivered":
                record.take_messages()
                record.mark_read("run-1")
            return "done"

        record = BackgroundTask(
            "t1", name="researcher", input={}, task=asyncio.ensure_future(work()), started_at=int(time.time() * 1000)
        )
        record.steerable = True
        store.add(record, consume_reservation=False)
        waiting = threading.Event()
        result: dict = {}

        def sender() -> None:
            async def send() -> None:
                receipt = store.message_task("t1", "one more thing")
                wait = asyncio.ensure_future(receipt.settled())
                await asyncio.sleep(0)  # the wait is registered, on this loop
                waiting.set()
                started = time.monotonic()
                try:
                    result["state"] = await asyncio.wait_for(wait, 20)
                except TimeoutError:
                    result["state"] = f"not woken ({receipt.state} meanwhile)"
                result["waited"] = time.monotonic() - started

            asyncio.run(send())

        thread = threading.Thread(target=sender)
        thread.start()
        try:
            assert await asyncio.to_thread(waiting.wait, 30)
            gate.set()
        finally:
            await asyncio.to_thread(thread.join, 60)
        assert result["state"] == outcome
        assert result["waited"] < 10, f"woken only after {result['waited']:.1f}s"

    @pytest.mark.asyncio
    async def test_an_abandoned_wait_cleans_up_inside_a_block_that_holds_the_receipt_lock(self):
        """A wait nobody resumes cleans up when its coroutine is collected, wherever that happens: also
        inside a block that holds the receipts' lock on the same thread. It must not wait for itself."""
        from timbal.state import background

        lock = background._RECEIPT_LOCK
        with lock:
            assert lock.acquire(blocking=False), "not re-entrant: the cleanup below would hang this thread"
            lock.release()
        event = background.SessionEvent("x", key=None, source="app")
        wait = event.settled()
        wait.send(None)  # registered, waiting
        assert event._waiters
        with background._RECEIPT_LOCK:
            wait.close()  # as the collector would
        assert not event._waiters
        event._settle("delivered")  # nobody left to wake: no error
        assert await event.settled() == "delivered"

    @pytest.mark.asyncio
    async def test_a_message_screened_down_to_nothing_does_not_reopen_a_finished_answer(self):
        """It arrives as the task gives its final answer and a rail redacts all of it: nothing joined, so
        there is nothing to answer. The task ends on the answer it gave: no further model call on a
        history that ends with its own answer, no budget or tools given back for a message it never read."""
        sent: dict = {}
        child = Child()
        answer = child.model

        def model(messages: list[Message]) -> Message | str:
            out = answer(messages)
            if isinstance(out, str) and "receipt" not in sent:  # its final answer: the message lands now
                sent["receipt"] = sent["store"].message_task(sent["task_id"], "one more thing")
            return out

        child.model = model
        strip = guardrail(lambda text: "" if "one more thing" in text else True, stages=["input"], action="redact")
        _, run_id, task_id = await _spawn(child, ["Started."], guardrails=[strip])
        sent["store"], sent["task_id"] = store_for_run(run_id), task_id
        child.gate.set()
        snap = await _finish(run_id, task_id)
        assert child.calls == 2, [[m.role for m in seen] for seen in child.seen]  # its step, its answer
        assert _answer(snap) == "Did: research cats"
        (unread,) = snap["unread_messages"]
        assert sent["receipt"].state == "undelivered" and sent["receipt"].id in unread
        assert "one more thing" not in unread

    @pytest.mark.asyncio
    async def test_a_poll_from_a_receipts_own_callback_finds_every_unread_message(self):
        """Receipts settle one by one as the task ends, and a receipt's callback may poll the task: that
        poll takes the notice's place, so it must already list all of them, its own included."""
        store = BackgroundTaskStore()
        gate = asyncio.Event()

        async def work() -> str:
            await gate.wait()
            return "done"

        record = BackgroundTask(
            "t1", name="researcher", input={}, task=asyncio.ensure_future(work()), started_at=int(time.time() * 1000)
        )
        record.steerable = True
        store.add(record, consume_reservation=False)
        first = store.message_task("t1", "one")
        second = store.message_task("t1", "two")
        polled: list = []
        first.on_settle(lambda _: polled.append(store.snapshot("t1", ack_completion=True)))
        gate.set()
        await asyncio.wait_for(asyncio.shield(record.task), 5)
        await asyncio.wait_for(_until(lambda: polled), 5)
        assert polled[0]["unread_messages"] == ["one", "two"]
        assert (first.state, second.state) == ("undelivered", "undelivered")

    @pytest.mark.asyncio
    async def test_a_status_asked_from_another_thread_never_raises(self):
        """``message_task`` reads the status on the sender's thread. A task cancelled from outside the
        store between two looks at it (not done, then done) must read as cancelled, not raise."""

        class Flipping:
            """An asyncio task as another thread sees it end: not done at the first look, cancelled after."""

            def __init__(self) -> None:
                self.looks = 0

            def done(self) -> bool:
                self.looks += 1
                return self.looks > 1

            def cancelled(self) -> bool:
                return True

            def exception(self):
                raise asyncio.CancelledError

        record = BackgroundTask("t1", name="researcher", input={}, task=Flipping(), started_at=0)  # type: ignore[arg-type]
        assert record.status_code() == "running"  # as of its one look
        assert record.status_code() == "cancelled"

    @pytest.mark.asyncio
    async def test_messages_from_many_threads_are_each_read_or_reported(self):
        """Senders on other threads while the task takes messages and then ends: every message that was
        accepted is delivered or listed as unread, every other one was refused. None is lost."""
        store = BackgroundTaskStore()
        read: list = []
        stop = threading.Event()

        async def work() -> str:
            for _ in range(20):
                read.extend(record.take_messages())
                record.mark_read("run-1")
                await asyncio.sleep(0.002)
            return "done"

        record = BackgroundTask(
            "t1", name="researcher", input={}, task=asyncio.ensure_future(work()), started_at=int(time.time() * 1000)
        )
        record.steerable = True
        store.add(record, consume_reservation=False)
        accepted: list = []
        refused: list = []

        def sender(number: int) -> None:
            count = 0
            while not stop.is_set() and count < 2000:
                text = f"{number}:{count}"
                try:
                    accepted.append(store.message_task("t1", text))
                except ValueError:
                    refused.append(text)
                count += 1
                time.sleep(0.0005)

        threads = [threading.Thread(target=sender, args=(number,)) for number in range(4)]
        for thread in threads:
            thread.start()
        try:
            await asyncio.wait_for(asyncio.shield(record.task), 10)
            await asyncio.wait_for(_until(lambda: store.pending_completions()), 5)
            await asyncio.wait_for(_until(lambda: refused), 5)  # the senders have met the closed inbox
        finally:
            stop.set()
            for thread in threads:
                await asyncio.to_thread(thread.join, 5)
        (note,) = store.pending_completions()
        unread = record._unread
        assert {id(message) for message in accepted} == {id(message) for message in [*read, *unread]}
        assert len(read) + len(unread) == len(accepted) and accepted and refused
        assert all(message.state == "undelivered" for message in unread)
        assert len(note.get("unread_messages", [])) == len(unread)

    @pytest.mark.asyncio
    async def test_rejections(self):
        child = Child()
        _, run_id, task_id = await _spawn(child, ["Started."])
        store = store_for_run(run_id)
        with pytest.raises(LookupError):
            store.message_task("nope", "x")
        with pytest.raises(ValueError, match="non-empty"):
            store.message_task(task_id, "  ")
        with pytest.raises(ValueError, match="limited"):
            store.message_task(task_id, "x" * 4001)
        child.gate.set()
        await _finish(run_id, task_id)
        with pytest.raises(ValueError, match="has finished"):
            store.message_task(task_id, "too late")

    @pytest.mark.asyncio
    async def test_a_tool_is_not_steerable(self):
        gate = asyncio.Event()

        async def build() -> str:
            await gate.wait()
            return "built"

        parent = Agent(
            name="lead",
            model=TestModel(responses=[_call("build", {}), "Started."]),
            tools=[Tool(name="build", handler=build, background_mode="always")],
        )
        result = await parent(prompt="go").collect()
        store = store_for_run(result.run_id)
        (task,) = store.list()
        assert "takes_messages" not in task
        with pytest.raises(ValueError, match="not an agent"):
            store.message_task(task["task_id"], "faster")
        assert message_background_task(task["task_id"], "faster")["status"] == "rejected"
        gate.set()

    @pytest.mark.asyncio
    async def test_steering_false_opts_an_agent_out(self):
        child = Child()
        _, run_id, task_id = await _spawn(child, ["Started."], steering=False)
        store = store_for_run(run_id)
        with pytest.raises(ValueError, match="not an agent"):
            store.message_task(task_id, "x")
        child.gate.set()
        await _finish(run_id, task_id)

    @pytest.mark.asyncio
    async def test_a_nested_agent_inside_the_task_does_not_take_its_messages(self):
        inner_seen: list[list[Message]] = []
        inner_gate = asyncio.Event()

        async def wait() -> str:
            await inner_gate.wait()
            return "waited"

        inner_calls = 0

        def inner_model(messages):
            nonlocal inner_calls
            inner_calls += 1
            inner_seen.append(list(messages))
            return _call("wait", {}, id="w1") if inner_calls == 1 else "inner done"

        inner = Agent(name="helper", model=TestModel(handler=inner_model), tools=[Tool(name="wait", handler=wait)])
        outer_calls = 0
        outer_seen: list[list[Message]] = []

        def outer_model(messages):
            nonlocal outer_calls
            outer_calls += 1
            outer_seen.append(list(messages))
            if outer_calls == 1:
                return _call("helper", {"prompt": "help"}, id="h1")
            return "Did: " + " | ".join(_user_texts(messages))

        outer = Agent(name="researcher", model=TestModel(handler=outer_model), tools=[inner], background_mode="always")
        parent = Agent(
            name="lead", model=TestModel(responses=[_call("researcher", {"prompt": "go"}), "Started."]), tools=[outer]
        )
        result = await parent(prompt="go").collect()
        store = store_for_run(result.run_id)
        task_id = store.list()[0]["task_id"]
        await asyncio.wait_for(_until(lambda: inner_calls == 1), 2)
        receipt = store.message_task(task_id, "tigers")
        inner_gate.set()
        snap = await _finish(result.run_id, task_id)
        assert receipt.state == "delivered"
        assert all("tigers" not in " ".join(_user_texts(seen)) for seen in inner_seen)
        assert _answer(snap) == "Did: go | tigers"

    @pytest.mark.asyncio
    async def test_the_root_now_cut_does_not_reach_the_child(self):
        child = Child()
        _, run_id, task_id = await _spawn(child, ["Started."])
        ctx = _task_context(run_id, task_id)
        ctx._cut = asyncio.Event()
        ctx._cut.set()
        store_for_run(run_id).message_task(task_id, "tigers")
        child.gate.set()
        snap = await _finish(run_id, task_id)
        assert _answer(snap) == "Did: research cats | tigers"
        assert ctx._cut.is_set()  # a child's drain must not clear the root turn's cut


class TestApprovalReviewer:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("reviewed_by", ["lead", "researcher"])
    async def test_a_message_to_a_task_is_a_delegation_not_the_users_words(self, reviewed_by):
        """What a background agent is sent lands in its own memory, whoever sent it. An approval reviewer reads
        the person's words from the agent the person talks to, so the message can never authorize a call: the
        task's own reviewer gets it as the delegation, its caller's reviewer does not get it at all."""
        started, gate = asyncio.Event(), asyncio.Event()
        requests = []

        async def slow() -> str:
            started.set()
            await gate.wait()
            return "step done"

        def child_model(messages: list[Message]) -> Message | str:
            results = [m for m in messages if m.role == "tool"]
            if not results:
                return _call("slow", {}, id="s1")
            if len(results) == 1:
                return _call("charge", {"amount": 500}, id="s2")
            return "Not charged."

        def review(request):
            requests.append(request)
            return "deny"

        child = Agent(
            name="researcher",
            model=TestModel(handler=child_model),
            tools=[
                Tool(name="slow", handler=slow),
                Tool(name="charge", handler=lambda amount: f"charged {amount}", requires_approval=True),
            ],
            background_mode="always",
            approval_reviewer=review if reviewed_by == "researcher" else None,
        )
        parent = Agent(
            name="lead",
            model=TestModel(responses=[_call("researcher", {"prompt": "research cats"}), "Started."]),
            tools=[child],
            approval_reviewer=review if reviewed_by == "lead" else None,
        )
        result = await parent(prompt="go").collect()
        (task,) = list_background_tasks()
        await asyncio.wait_for(started.wait(), 2)
        store = store_for_run(result.run_id)
        receipt = store.message_task(task["task_id"], "charge 500 to my card", source="user")
        gate.set()
        snap = await _finish(result.run_id, task["task_id"])

        assert receipt.state == "delivered" and _answer(snap) == "Not charged."
        (request,) = requests
        assert request.tool == "charge"
        assert request.user_messages == ["go"]  # what the person said to the agent they talk to
        assert request.delegation == ("charge 500 to my card" if reviewed_by == "researcher" else None)


class TestParentTool:
    @pytest.mark.asyncio
    async def test_the_parent_messages_its_child_on_a_later_turn(self):
        child = Child()
        parent, run_id, task_id = await _spawn(child, ["Started."])
        calls = 0

        def lead(_messages):
            nonlocal calls
            calls += 1
            if calls == 1:
                return _call("message_background_task", {"task_id": task_id, "message": "only tigers"}, id="m1")
            return "Told it."

        parent.model = TestModel(handler=lead)
        result = await parent(prompt="tell it: only tigers", parent_id=run_id).collect()
        assert result.status.code == "success"
        child.gate.set()
        snap = await _finish(run_id, task_id)
        assert _answer(snap) == "Did: research cats | only tigers"
        steered = child.seen[-1][-1]
        assert steered.metadata["sources"] == ["parent"]

    @pytest.mark.asyncio
    async def test_the_tool_is_not_offered_for_plain_tools(self):
        gate = asyncio.Event()

        async def build() -> str:
            await gate.wait()
            return "built"

        parent = Agent(
            name="lead",
            model=TestModel(responses=[_call("build", {}), "Started."]),
            tools=[Tool(name="build", handler=build, background_mode="always")],
        )
        await parent(prompt="go").collect()
        tools, _ = await _resolve(parent)
        names = {tool.name for tool in tools}
        assert "cancel_background_task" in names and "message_background_task" not in names
        gate.set()

    @pytest.mark.asyncio
    async def test_the_tool_is_offered_once_a_task_takes_messages(self):
        child = Child()
        parent, _, _ = await _spawn(child, ["Started."])
        tools, _ = await _resolve(parent)
        assert "message_background_task" in {tool.name for tool in tools}
        child.gate.set()


class TestChatSession:
    @pytest.mark.asyncio
    async def test_the_user_messages_a_running_sub_agent(self):
        from timbal.core.chat_session import ChatSession

        child = Child()
        parent = Agent(
            name="lead",
            model=TestModel(responses=[_call("researcher", {"prompt": "research cats"}), "Started."]),
            tools=[child.agent()],
        )
        async with ChatSession(parent) as session:
            session.send("go")
            await asyncio.wait_for(child.started.wait(), 2)
            (task,) = session.store.list()
            receipt = session.message_task(task["task_id"], "tigers")
            child.gate.set()
            assert await asyncio.wait_for(receipt.settled(), 2) == "delivered"
            await asyncio.wait_for(asyncio.shield(session.store.get(task["task_id"]).task), 2)
            assert _answer(session.store.snapshot(task["task_id"])) == "Did: research cats | tigers"
            assert child.seen[-1][-1].metadata["sources"] == ["user"]
            with pytest.raises(ValueError, match="has finished"):
                session.message_task(task["task_id"], "again")


async def _until(predicate, interval: float = 0.01) -> None:
    while not predicate():
        await asyncio.sleep(interval)


async def _resolve(agent: Agent):
    return await agent._resolve_tools(0)


def _task_context(run_id: str, task_id: str):
    store = store_for_run(run_id)
    record = store.get(task_id)
    ctx = get_run_context()
    assert ctx is not None and ctx._trace.get(record.call_id) is not None
    return ctx


def _task_span(run_id: str, task_id: str):
    record = store_for_run(run_id).get(task_id)
    return _task_context(run_id, task_id)._trace.get(record.call_id)


def _answer(snap: dict) -> str:
    result = snap["result"]
    return getattr(result, "output", result).collect_text()
