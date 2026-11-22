"""Saves of one run land in the order their snapshots were taken.

A save takes its snapshot when the provider's ``_store`` starts and commits later (a thread, a request), so two
saves of one run can be on their way at once: the end of the root run and the end of background work that
outlives it, each a barrier of its own. The later snapshot holds what the background work wrote (a compaction
for the next turn, a detached child's real output over its placeholder). Stored first and overwritten by the
earlier one, that is lost for good: nothing saves the run again.

Two things keep the order. ``uploads.write_through`` sends a run's ``sync`` saves to the provider one at a
time (unless the provider sets ``overlapping_saves``), and the SQLite provider, whose writes run on worker
threads, drops a snapshot older than the one it has committed.
"""

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest
from timbal import Agent, Tool
from timbal.core import agent as agent_module
from timbal.core.test_model import TestModel
from timbal.state import set_call_id, set_parent_call_id, set_run_context
from timbal.state.context import RunContext
from timbal.state.tracing import uploads
from timbal.state.tracing.providers import SqliteTracingProvider
from timbal.state.tracing.providers import sqlite as sqlite_module
from timbal.state.tracing.providers.base import TracingProvider
from timbal.state.tracing.trace import Trace
from timbal.types.content import TextContent, ToolUseContent
from timbal.types.message import Message

DEADLINE = 10
SUMMARY = "[summary]"


@pytest.fixture(autouse=True)
def isolated():
    set_run_context(None)
    set_call_id(None)
    set_parent_call_id(None)
    agent_module._PREPARING.clear()
    yield
    agent_module._PREPARING.clear()
    set_run_context(None)


async def until(predicate):
    """Wait for a state, however many steps it takes; the caller sets the deadline."""
    while not predicate():
        await asyncio.sleep(0)


class WriteGate:
    """Stands in for ``asyncio.to_thread`` in the SQLite provider, where a save commits.

    The first write ``stale()`` is true for is held: its snapshot was taken just before, in the same step, so
    what ``stale`` reads from the live run is what that snapshot holds. A write that starts while it is held
    commits ahead of it (``overtakers``, ``overtaken``).
    """

    def __init__(self, stale):
        self.stale = stale
        self.holding = asyncio.Event()
        self.release = asyncio.Event()
        self.overtakers = 0
        self.overtaken = asyncio.Event()

    async def to_thread(self, fn, *args):
        if fn.__name__ != "_write":
            return await asyncio.to_thread(fn, *args)
        if not self.holding.is_set() and self.stale():
            self.holding.set()
            await self.release.wait()
            return await asyncio.to_thread(fn, *args)
        ahead = self.holding.is_set() and not self.release.is_set()
        self.overtakers += ahead
        try:
            return await asyncio.to_thread(fn, *args)
        finally:
            if ahead:
                self.overtaken.set()


def gated(monkeypatch, stale):
    gate = WriteGate(stale)
    monkeypatch.setattr(sqlite_module, "asyncio", SimpleNamespace(to_thread=gate.to_thread))
    return gate


@pytest.fixture
def background_end(monkeypatch):
    """``requested``: a piece of background work asked for its end to be stored. ``stored``: that returned."""
    seen = SimpleNamespace(requested=asyncio.Event(), stored=asyncio.Event())
    real = RunContext._end_background_work

    async def traced(self):
        seen.requested.set()
        try:
            await real(self)
        finally:
            seen.stored.set()

    monkeypatch.setattr(RunContext, "_end_background_work", traced)
    return seen


async def overlap(gate, background_end, let_background_finish, *, queued):
    """The end-of-run save has its snapshot and has not committed: the background work now writes its result
    and asks for its own save. Whatever of that save can commit while the first is held does; then the first.

    ``queued``: the framework sends this provider one save of a run at a time, so the second is waiting for the
    first. If it is not (nothing queues them), its write starts in the step that asks for it. Left to the
    provider (``overlapping_saves``), it always commits first.
    """
    await gate.holding.wait()
    let_background_finish()
    await background_end.requested.wait()
    if not queued or gate.overtakers:
        await gate.overtaken.wait()
    gate.release.set()
    await background_end.stored.wait()


async def stored_trace(provider, run_id):
    return await provider.get_strict(RunContext(parent_id=run_id, tracing_provider=provider))


def span_at(trace, path):
    (span,) = trace.get_path(path)
    return span


def calls(name):
    return Message(role="assistant", content=[ToolUseContent(id="c1", name=name, input={})], stop_reason="tool_use")


def builder():
    """An agent that starts ``build`` in the background and answers; ``build`` returns once the event is set."""
    build_now = asyncio.Event()

    async def build() -> str:
        await build_now.wait()
        return "built"

    def agent(provider):
        return Agent(
            name="agent",
            model=TestModel(responses=[calls("build"), "started"]),
            tools=[Tool(name="build", handler=build, background_mode="always")],
            tracing_provider=provider,
        )

    return agent, build_now


def not_built(context):
    """The root has ended and the child's span still holds its placeholder: what the root's end save stores."""

    def stale():
        root = context.root_span()
        return root.t1 is not None and span_at(context._trace, "agent.build").output != "built"

    return stale


async def turn(agent, context, prompt="go"):
    set_run_context(context)
    output = await agent(prompt=prompt).collect()
    set_run_context(None)
    assert output.status.code == "success", output.error
    return output


QUEUED = pytest.mark.parametrize("queued", [True, False], ids=["one at a time", "left to the provider"])


class TestTheRootsEndAndBackgroundWorksEnd:
    """The root's end save takes its snapshot, the background work writes its result and saves, and that second
    save commits first if anything lets it. Either the saves go one at a time, or the writer drops the stale one."""

    @QUEUED
    async def test_a_compaction_prepared_for_the_next_turn_is_stored(
        self, tmp_path, monkeypatch, background_end, queued
    ):
        """``memory_compaction_prepare_ratio``: the turn ends, its compaction lands in session data afterwards."""
        monkeypatch.setattr(
            Agent, "_context_utilization", lambda _self, _model, messages, _measurement: 0.05 * len(messages)
        )
        provider = SqliteTracingProvider.configured(_path=tmp_path / "t.db", overlapping_saves=not queued)
        summarize = asyncio.Event()

        async def summarize_old(memory):
            await summarize.wait()
            return [Message(role="user", content=[TextContent(text=SUMMARY)]), *memory[-2:]]

        steps = 0

        def model(messages):
            nonlocal steps
            if messages[-1].role == "user" and messages[-1].collect_text() == "again":
                return "again done"
            if steps >= 2:
                return "done"
            steps += 1
            return Message(
                role="assistant",
                content=[ToolUseContent(id=f"c{steps}", name="step", input={})],
                stop_reason="tool_use",
            )

        agent = Agent(
            name="agent",
            model=TestModel(handler=model),
            tools=[Tool(name="step", handler=lambda: "ok")],
            tracing_provider=provider,
            memory_compaction=summarize_old,
            memory_compaction_prepare_ratio=0.3,  # 6 messages when the turn ends: prepared then, not before
        )
        first = RunContext(tracing_provider=provider)

        def stale():
            root = first.root_span()
            return root.t1 is not None and agent_module._PREPARED_KEY not in (first._session_data or {})

        gate = gated(monkeypatch, stale)
        running = asyncio.create_task(turn(agent, first))
        await asyncio.wait_for(overlap(gate, background_end, summarize.set, queued=queued), DEADLINE)
        await asyncio.wait_for(running, DEADLINE)

        root = (await stored_trace(provider, first.id))[first.root_span().call_id]
        assert "agent" in (root.session or {}).get(agent_module._PREPARED_KEY, {})
        agent_module._PREPARING.clear()  # a new process: only the trace is left
        second = RunContext(parent_id=first.id, tracing_provider=provider)
        output = await asyncio.wait_for(turn(agent, second, "again"), DEADLINE)
        assert second.root_span().memory[0].collect_text() == SUMMARY
        assert output.usage.get("compaction:background") == 1
        assert gate.overtakers == (0 if queued else 1)

    @QUEUED
    async def test_a_detached_childs_output_is_stored_over_its_placeholder(
        self, tmp_path, monkeypatch, background_end, queued
    ):
        """``background_mode="always"``: the launching call ends with a placeholder, the real output comes later."""
        provider = SqliteTracingProvider.configured(_path=tmp_path / "t.db", overlapping_saves=not queued)
        agent, build_now = builder()
        context = RunContext(tracing_provider=provider)
        gate = gated(monkeypatch, not_built(context))
        running = asyncio.create_task(turn(agent(provider), context))
        await asyncio.wait_for(overlap(gate, background_end, build_now.set, queued=queued), DEADLINE)
        await asyncio.wait_for(running, DEADLINE)

        trace = await stored_trace(provider, context.id)
        assert span_at(trace, "agent.build").output == "built"
        assert trace[context.root_span().call_id].t1 is not None
        assert gate.overtakers == (0 if queued else 1)


class TestACancelledSave:
    async def test_its_write_does_not_land_over_the_runs_end(self, tmp_path, monkeypatch):
        """A turn cancelled while a save is on its worker thread: the thread goes on, the run's end is saved
        meanwhile, and the abandoned write then finds a newer snapshot stored. Written all the same, it would
        leave the run stored as one that never ended."""
        provider = SqliteTracingProvider.configured(_path=tmp_path / "t.db")
        parked, go, written = threading.Event(), threading.Event(), threading.Event()
        held = False

        async def to_thread(fn, *args):
            nonlocal held
            if fn.__name__ != "_write" or held:
                return await asyncio.to_thread(fn, *args)
            held = True  # the first save of the run: the model call's end, the root still running

            def late():
                parked.set()
                go.wait(DEADLINE)
                try:
                    fn(*args)
                finally:
                    written.set()

            return await asyncio.to_thread(late)

        monkeypatch.setattr(sqlite_module, "asyncio", SimpleNamespace(to_thread=to_thread))
        agent = Agent(name="agent", model=TestModel(responses=["hello"]), tracing_provider=provider)
        context = RunContext(tracing_provider=provider)

        async def cut_turn():
            set_run_context(context)
            await agent(prompt="go").collect()

        running = asyncio.create_task(cut_turn())
        try:
            assert await asyncio.to_thread(parked.wait, DEADLINE)
            running.cancel()
            await asyncio.wait([running], timeout=DEADLINE)
            assert running.done()
            ended = await stored_trace(provider, context.id)
            assert ended[context.root_span().call_id].t1 is not None  # the run's end is stored
        finally:
            go.set()  # whatever happened above, the parked thread is let go
        assert await asyncio.to_thread(written.wait, DEADLINE)
        root = (await stored_trace(provider, context.id))[context.root_span().call_id]
        assert root.t1 is not None
        assert root.status["code"] == "cancelled"


class TestANewerWriteThatFails:
    async def test_an_older_one_still_lands(self, tmp_path, monkeypatch):
        """Only a snapshot that is stored keeps older ones out: one whose transaction was rolled back does not."""
        provider = SqliteTracingProvider.configured(_path=tmp_path / "t.db", overlapping_saves=True)
        context = RunContext(tracing_provider=provider)
        await asyncio.wait_for(
            turn(Agent(name="agent", model=TestModel(responses=["hello"]), tracing_provider=provider), context),
            DEADLINE,
        )
        root = context.root_span()
        db = sqlite_module.database(tmp_path / "t.db")

        class Refusing:
            """The connection, except that the span rows cannot be written."""

            def __init__(self, conn):
                self.conn = conn

            def __enter__(self):
                return self.conn.__enter__()

            def __exit__(self, *error):
                return self.conn.__exit__(*error)

            def execute(self, *args):
                return self.conn.execute(*args)

            def executemany(self, *_args):
                raise sqlite_module.sqlite3.OperationalError("disk I/O error")

        older_lands = asyncio.Event()
        writes = 0

        async def to_thread(fn, *args):
            nonlocal writes
            if fn.__name__ != "_write":
                return await asyncio.to_thread(fn, *args)
            writes += 1
            if writes == 1:
                await older_lands.wait()
                return await asyncio.to_thread(fn, *args)

            def refused():
                real, db.conn = db.conn, Refusing(db.conn)
                try:
                    fn(*args)
                finally:
                    db.conn = real

            return await asyncio.to_thread(refused)

        monkeypatch.setattr(sqlite_module, "asyncio", SimpleNamespace(to_thread=to_thread))
        root.metadata["note"] = "older"
        older = asyncio.create_task(provider.put(context))
        await asyncio.wait_for(until(lambda: writes == 1), DEADLINE)
        root.metadata["note"] = "newer"
        with pytest.raises(sqlite_module.sqlite3.OperationalError):
            await asyncio.wait_for(provider.put(context), DEADLINE)
        older_lands.set()
        await asyncio.wait_for(older, DEADLINE)

        stored = (await stored_trace(provider, context.id))[root.call_id]
        assert stored.metadata["note"] == "older"


class Remote(TracingProvider):
    """Copies the trace when asked to store it, as a request body would, and commits when ``reply`` lets it:
    ``reply(n)`` is an event the n-th write waits for before it lands, or None."""

    stored: dict
    log: list
    holds: dict

    @classmethod
    def reply(cls, n):
        return cls.holds.get(n)

    @classmethod
    def fresh(cls, **kwargs):
        return cls.configured(stored={}, log=[], holds={}, **kwargs)

    @classmethod
    def hold(cls, n):
        """The n-th write lands once the returned event is set."""
        event = cls.holds[n] = asyncio.Event()
        return event

    @classmethod
    async def get(cls, run_context):
        records = cls.stored.get(run_context.parent_id)
        return Trace(json.loads(records)) if records is not None else None

    @classmethod
    async def _store(cls, run_context):
        body = json.dumps(run_context._trace.model_dump(), default=str)
        n = sum(1 for kind, _ in cls.log if kind == "sent") + 1
        cls.log.append(("sent", n))
        wait = cls.reply(n)
        if wait is not None:
            await wait.wait()
        cls.stored[run_context.id] = body
        cls.log.append(("stored", n))


class Saves:
    """Saves of one run asked for from tasks of their own; ``asked`` counts those that reached ``write_through``
    (each runs on to where it waits, or to its end, in the same step)."""

    def __init__(self, context):
        self.context = context
        self.asked = 0

    def ask(self, put):
        async def save():
            self.asked += 1
            await uploads.write_through(self.context, put)

        return asyncio.create_task(save())

    async def all_asked(self, n):
        await asyncio.wait_for(until(lambda: self.asked == n), DEADLINE)


class TestAnyProvider:
    """``write_through`` is what every ``sync`` save goes through: a provider needs no ordering of its own."""

    async def test_a_detached_childs_output_reaches_a_provider_that_copies_then_waits(self, background_end):
        provider = Remote.fresh()
        agent, build_now = builder()
        context = RunContext(tracing_provider=provider)
        stale, root_end = not_built(context), asyncio.Event()
        held = []

        def reply(n):
            if not held and stale():
                held.append(n)
                return root_end
            return None

        provider.reply = staticmethod(reply)
        running = asyncio.create_task(turn(agent(provider), context))
        await asyncio.wait_for(until(lambda: held), DEADLINE)
        build_now.set()
        await asyncio.wait_for(background_end.requested.wait(), DEADLINE)
        (n,) = held
        assert provider.log[-1] == ("sent", n)  # the child's save has not started: it waits for the root's end
        root_end.set()
        await asyncio.wait_for(background_end.stored.wait(), DEADLINE)
        await asyncio.wait_for(running, DEADLINE)

        assert provider.log[-4:] == [("sent", n), ("stored", n), ("sent", n + 1), ("stored", n + 1)]
        trace = await provider.get(RunContext(parent_id=context.id, tracing_provider=provider))
        assert span_at(trace, "agent.build").output == "built"

    async def test_saves_asked_for_during_a_write_wait_and_share_the_next_one(self):
        """Calls that end together: the write that follows the one in flight holds all of their states."""
        provider = Remote.fresh()
        saves = Saves(RunContext(tracing_provider=provider))
        first_lands = provider.hold(1)
        tasks = [saves.ask(provider.put) for _ in range(4)]
        await saves.all_asked(4)
        assert provider.log == [("sent", 1)]  # the others wait: none has taken its snapshot
        first_lands.set()
        await asyncio.wait_for(asyncio.gather(*tasks), DEADLINE)
        assert provider.log == [("sent", 1), ("stored", 1), ("sent", 2), ("stored", 2)]

    async def test_a_save_asked_for_once_that_next_write_began_gets_one_of_its_own(self):
        provider = Remote.fresh()
        saves = Saves(RunContext(tracing_provider=provider))
        first_lands, second_lands = provider.hold(1), provider.hold(2)
        first, second = saves.ask(provider.put), saves.ask(provider.put)
        await saves.all_asked(2)
        first_lands.set()
        await asyncio.wait_for(until(lambda: ("sent", 2) in provider.log), DEADLINE)
        third = saves.ask(provider.put)  # the second write has its snapshot: it cannot hold this one
        await saves.all_asked(3)
        second_lands.set()
        await asyncio.wait_for(asyncio.gather(first, second, third), DEADLINE)
        assert provider.log == [(kind, n) for n in (1, 2, 3) for kind in ("sent", "stored")]

    async def test_a_shared_write_that_fails_leaves_the_next_waiter_to_write(self):
        provider = Remote.fresh()
        saves = Saves(RunContext(tracing_provider=provider))
        first_lands = provider.hold(1)

        async def refused(_context):
            raise OSError("disk full")

        first, second, third = saves.ask(provider.put), saves.ask(refused), saves.ask(provider.put)
        await saves.all_asked(3)
        first_lands.set()
        with pytest.raises(OSError):
            await asyncio.wait_for(second, DEADLINE)
        await asyncio.wait_for(asyncio.gather(first, third), DEADLINE)
        assert provider.log == [("sent", 1), ("stored", 1), ("sent", 2), ("stored", 2)]  # the third's own

    async def test_saves_that_never_overlap_take_no_lock(self):
        provider = Remote.fresh()
        context = RunContext(tracing_provider=provider)
        for _ in range(3):
            await uploads.write_through(context, provider.put)
        assert context.__dict__["_save_turn"] is None

    async def test_a_failed_write_lets_the_next_one_go(self):
        provider = Remote.fresh()
        saves = Saves(RunContext(tracing_provider=provider))
        refuse = asyncio.Event()

        async def refused(_context):
            await refuse.wait()
            raise OSError("disk full")

        first, second = saves.ask(refused), saves.ask(provider.put)
        await saves.all_asked(2)
        assert provider.log == []
        refuse.set()
        with pytest.raises(OSError):
            await asyncio.wait_for(first, DEADLINE)
        await asyncio.wait_for(second, DEADLINE)
        assert provider.log == [("sent", 1), ("stored", 1)]
        await asyncio.wait_for(uploads.write_through(saves.context, provider.put), DEADLINE)  # and the one after

    async def test_a_save_cancelled_while_it_waits_leaves_the_queue_whole(self):
        provider = Remote.fresh()
        saves = Saves(RunContext(tracing_provider=provider))
        first_lands = provider.hold(1)
        first, second, third = (saves.ask(provider.put) for _ in range(3))
        await saves.all_asked(3)
        second.cancel()
        await asyncio.wait([second], timeout=DEADLINE)
        assert second.cancelled()
        first_lands.set()
        await asyncio.wait_for(asyncio.gather(first, third), DEADLINE)
        assert provider.log == [("sent", 1), ("stored", 1), ("sent", 2), ("stored", 2)]

    async def test_a_save_cancelled_while_it_writes_lets_the_next_one_go(self):
        provider = Remote.fresh()
        saves = Saves(RunContext(tracing_provider=provider))
        provider.hold(1)  # never set: the first write is cancelled where it waits
        first, second = saves.ask(provider.put), saves.ask(provider.put)
        await saves.all_asked(2)
        first.cancel()
        await asyncio.wait_for(second, DEADLINE)
        await asyncio.wait([first], timeout=DEADLINE)
        assert first.cancelled()
        assert provider.log == [("sent", 1), ("sent", 2), ("stored", 2)]

    async def test_runs_do_not_wait_for_each_other(self):
        provider = Remote.fresh()
        one, other = Saves(RunContext(tracing_provider=provider)), RunContext(tracing_provider=provider)
        first_lands = provider.hold(1)
        held = one.ask(provider.put)
        await one.all_asked(1)
        await asyncio.wait_for(uploads.write_through(other, provider.put), DEADLINE)
        assert provider.log == [("sent", 1), ("sent", 2), ("stored", 2)]
        first_lands.set()
        await asyncio.wait_for(held, DEADLINE)

    async def test_a_provider_that_orders_its_own_saves_gets_them_as_they_come(self):
        provider = Remote.fresh(overlapping_saves=True)
        saves = Saves(RunContext(tracing_provider=provider))
        first_lands = provider.hold(1)
        first, second = saves.ask(provider.put), saves.ask(provider.put)
        await asyncio.wait_for(second, DEADLINE)
        assert provider.log == [("sent", 1), ("sent", 2), ("stored", 2)]
        first_lands.set()
        await asyncio.wait_for(first, DEADLINE)
        assert "_save_turn" not in saves.context.__dict__
