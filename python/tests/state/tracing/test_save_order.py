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
import gc
import json
import sys
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


class HeldLock:
    """Stands in for a database's connection lock: the first worker to ask for it once ``armed`` waits for ``go``
    before it takes the real one. A write stopped there is on its worker and has begun: only the order of the
    run's snapshots can still keep it out."""

    def __init__(self, db, monkeypatch):
        self.lock = db.lock
        self.armed = False
        self.parked, self.go, self.left = threading.Event(), threading.Event(), threading.Event()
        self._arming = threading.Lock()
        self._worker = None
        monkeypatch.setattr(db, "lock", self)

    def __enter__(self):
        with self._arming:
            mine, self.armed = self.armed, False
        if mine:
            self._worker = threading.get_ident()
            self.parked.set()
            self.go.wait(DEADLINE)
        return self.lock.__enter__()

    def __exit__(self, *error):
        try:
            return self.lock.__exit__(*error)
        finally:
            if threading.get_ident() == self._worker:
                self.left.set()


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
    """A save whose caller is cancelled while its write is on a worker thread: the thread goes on."""

    async def test_its_write_does_not_land_over_the_runs_end(self, tmp_path, monkeypatch):
        """The run's end is saved meanwhile, and the abandoned write then finds a newer snapshot stored. Written
        all the same, it would leave the run stored as one that never ended."""
        provider = SqliteTracingProvider.configured(_path=tmp_path / "t.db")
        db = sqlite_module.database(tmp_path / "t.db")
        lock = HeldLock(db, monkeypatch)
        lock.armed = True  # the first save of the run: the model call's end, the root still running
        agent = Agent(name="agent", model=TestModel(responses=["hello"]), tracing_provider=provider)
        context = RunContext(tracing_provider=provider)

        async def cut_turn():
            set_run_context(context)
            await agent(prompt="go").collect()

        running = asyncio.create_task(cut_turn())
        try:
            assert await asyncio.to_thread(lock.parked.wait, DEADLINE)
            running.cancel()
            await asyncio.wait([running], timeout=DEADLINE)
            assert running.done()
            ended = await stored_trace(provider, context.id)
            assert ended[context.root_span().call_id].t1 is not None  # the run's end is stored
        finally:
            lock.go.set()  # whatever happened above, the parked thread is let go
        assert await asyncio.to_thread(lock.left.wait, DEADLINE)
        root = (await stored_trace(provider, context.id))[context.root_span().call_id]
        assert root.t1 is not None
        assert root.status["code"] == "cancelled"
        await asyncio.wait_for(until(lambda: not db.writing), DEADLINE)  # nothing kept once its writes are over

    async def test_nor_after_its_runs_cached_state_was_dropped(self, tmp_path, monkeypatch):
        """The file keeps per-run state in a bounded cache. What orders a run's writes must not go with it: a
        write parked on its worker while other runs push this one out of the cache is still an older snapshot
        when the run's end has been stored."""
        monkeypatch.setattr(sqlite_module, "_RUN_STATE_MAX", 1)
        provider = SqliteTracingProvider.configured(_path=tmp_path / "t.db")
        db = sqlite_module.database(tmp_path / "t.db")
        lock = HeldLock(db, monkeypatch)
        working, finish = asyncio.Event(), asyncio.Event()

        async def work() -> str:
            working.set()
            await finish.wait()
            return "ok"

        agent = Agent(
            name="agent",
            model=TestModel(responses=[calls("work"), "done"]),
            tools=[Tool(name="work", handler=work)],
            tracing_provider=provider,
        )
        context = RunContext(tracing_provider=provider)
        running = asyncio.create_task(turn(agent, context))
        try:
            await asyncio.wait_for(working.wait(), DEADLINE)
            lock.armed = True
            mid_run = asyncio.create_task(provider.put(context))  # the run as it is now: its root still going
            assert await asyncio.to_thread(lock.parked.wait, DEADLINE)
            mid_run.cancel()
            await asyncio.wait([mid_run], timeout=DEADLINE)
            assert mid_run.cancelled()
            for _ in range(2):
                other = Agent(name="other", model=TestModel(responses=["hi"]), tracing_provider=provider)
                await asyncio.wait_for(turn(other, RunContext(tracing_provider=provider)), DEADLINE)
            assert context.id not in db.runs  # pushed out of the cache
            finish.set()
            await asyncio.wait_for(running, DEADLINE)
        finally:
            lock.go.set()
        assert await asyncio.to_thread(lock.left.wait, DEADLINE)
        root = (await stored_trace(provider, context.id))[context.root_span().call_id]
        assert root.t1 is not None
        assert root.status["code"] == "success"
        await asyncio.wait_for(until(lambda: not db.writing), DEADLINE)

    async def test_one_whose_worker_had_not_begun_is_never_written(self, tmp_path, monkeypatch):
        """Nothing has to order it against the saves that follow: it is dropped, and it keeps nothing of the
        run's state alive."""
        provider = SqliteTracingProvider.configured(_path=tmp_path / "t.db")
        db = sqlite_module.database(tmp_path / "t.db")
        context = RunContext(tracing_provider=provider)
        agent = Agent(name="agent", model=TestModel(responses=["hello"]), tracing_provider=provider)
        await asyncio.wait_for(turn(agent, context), DEADLINE)
        parked, go, returned = threading.Event(), threading.Event(), threading.Event()

        async def to_thread(fn, *args):
            if fn.__name__ != "_write":
                return await asyncio.to_thread(fn, *args)

            def late():  # the worker is busy: it gets to the write after its caller has gone
                parked.set()
                go.wait(DEADLINE)
                try:
                    fn(*args)
                finally:
                    returned.set()

            return await asyncio.to_thread(late)

        monkeypatch.setattr(sqlite_module, "asyncio", SimpleNamespace(to_thread=to_thread))
        root = context.root_span()
        root.metadata["note"] = "never stored"
        save = asyncio.create_task(provider.put(context))
        try:
            assert await asyncio.to_thread(parked.wait, DEADLINE)
            save.cancel()
            await asyncio.wait([save], timeout=DEADLINE)
            assert save.cancelled()
        finally:
            go.set()
        assert await asyncio.to_thread(returned.wait, DEADLINE)
        stored = (await stored_trace(provider, context.id))[root.call_id]
        assert "note" not in stored.metadata
        assert not db.writing


class TestASaveLeftBehind:
    def test_collected_while_another_save_registers_it_does_not_wait_for_itself(self, tmp_path, monkeypatch):
        """A save left suspended when its loop was closed is closed by the garbage collector, on whatever
        thread a collection happens to run: also on one that is registering another save, with the lock both
        of them need in its hands."""
        provider = SqliteTracingProvider.configured(_path=tmp_path / "t.db")
        db = sqlite_module.database(tmp_path / "t.db")
        agent = Agent(name="agent", model=TestModel(responses=["hello"]), tracing_provider=provider)
        left, next_one = RunContext(tracing_provider=provider), RunContext(tracing_provider=provider)
        for context in (left, next_one):
            asyncio.run(asyncio.wait_for(turn(agent, context), DEADLINE))

        class Counted:
            """The lock itself, counting who asks for it."""

            def __init__(self, lock):
                self.lock, self.asked = lock, 0

            def __enter__(self):
                self.asked += 1
                return self.lock.__enter__()

            def __exit__(self, *error):
                return self.lock.__exit__(*error)

        asked_meanwhile = []

        class Collecting(sqlite_module._Writes):
            """A collection that lands on this allocation, which is made with the lock held."""

            __slots__ = ()

            def __init__(self):
                before = lock.asked
                left_behind.clear()  # the last reference to it goes here
                gc.collect()
                asked_meanwhile.append(lock.asked - before)
                super().__init__()

        async def no_worker(_fn, *_args):
            await asyncio.get_running_loop().create_future()  # the write never reaches a worker

        gc.collect()
        gc.disable()  # nothing is collected below but where the test says
        try:
            monkeypatch.setattr(sqlite_module, "asyncio", SimpleNamespace(to_thread=no_worker))
            loop = asyncio.new_event_loop()
            left_behind = [loop.create_task(provider.put(left))]
            # To where the save waits for its worker: counted, and nothing will say it is over.
            loop.run_until_complete(asyncio.wait_for(until(lambda: left.id in db.writing), DEADLINE))
            loop.close()
            del loop
            lock = Counted(db.writing_lock)
            monkeypatch.setattr(db, "writing_lock", lock)
            monkeypatch.setattr(sqlite_module, "_Writes", Collecting)
            monkeypatch.setattr(sqlite_module, "asyncio", asyncio)
            monkeypatch.setattr(sys, "unraisablehook", lambda _info: None)  # "Task was destroyed but it is pending"
            saving = threading.Thread(target=lambda: asyncio.run(provider.put(next_one)), daemon=True)
            saving.start()
            saving.join(DEADLINE)
        finally:
            gc.enable()
        assert asked_meanwhile == [1]  # the save left behind was closed inside that allocation, and asked
        assert not saving.is_alive()
        assert not db.writing  # and is no longer counted


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
        assert not db.writing  # a write that failed is over too


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
            return await uploads.write_through(self.context, put)

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
        wrote = await asyncio.wait_for(asyncio.gather(*tasks), DEADLINE)
        assert provider.log == [("sent", 1), ("stored", 1), ("sent", 2), ("stored", 2)]
        assert wrote == [True, True, False, False]  # the last two were stored by the second one's write

    async def test_a_save_another_write_stored_counts_as_coalesced(self, monkeypatch):
        """``tracing:saves`` counts writes: four saves asked for at once are two of them, and two coalesced."""
        monkeypatch.setenv(uploads.METRICS_ENV, "1")
        provider = Remote.fresh()
        working, finish = asyncio.Event(), asyncio.Event()

        async def work() -> str:
            working.set()
            await finish.wait()
            return "ok"

        agent = Agent(
            name="agent",
            model=TestModel(responses=[calls("work"), "done"]),
            tools=[Tool(name="work", handler=work)],
            tracing_provider=provider,
        )
        context = RunContext(tracing_provider=provider)
        running = asyncio.create_task(turn(agent, context))
        await asyncio.wait_for(working.wait(), DEADLINE)
        usage = context.root_span().usage
        before = {key: usage.get(key, 0) for key in ("tracing:saves", "tracing:saves_coalesced")}
        sent = sum(1 for kind, _ in provider.log if kind == "sent")
        first_lands = provider.hold(sent + 1)
        asked = 0

        async def save():
            nonlocal asked
            asked += 1
            await context._save_trace()

        tasks = [asyncio.create_task(save()) for _ in range(4)]
        await asyncio.wait_for(until(lambda: asked == 4), DEADLINE)
        first_lands.set()
        await asyncio.wait_for(asyncio.gather(*tasks), DEADLINE)
        assert sum(1 for kind, _ in provider.log if kind == "sent") == sent + 2
        assert usage.get("tracing:saves", 0) - before["tracing:saves"] == 2
        assert usage.get("tracing:saves_coalesced", 0) - before["tracing:saves_coalesced"] == 2
        finish.set()
        await asyncio.wait_for(running, DEADLINE)

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
        assert not context.__dict__["_save_turn"]  # no mark, no queue

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


class Slow(TracingProvider):
    """Stores the run after a moment, as a request would, so saves asked for together overlap."""

    stored: dict

    @classmethod
    def fresh(cls, **kwargs):
        return cls.configured(stored={}, **kwargs)

    @classmethod
    async def get(cls, _run_context):
        return None

    @classmethod
    async def _store(cls, run_context):
        body = json.dumps(run_context._trace.model_dump(), default=str)
        await asyncio.sleep(0.005)
        cls.stored[run_context.id] = body


def stored_t1(path, run_id, call_id):
    """The root's ``t1`` as the file holds it, read on a connection of its own."""
    import sqlite3

    conn = sqlite3.connect(str(path))
    try:
        row = conn.execute("SELECT span FROM spans WHERE run_id = ? AND call_id = ?", (run_id, call_id)).fetchone()
    finally:
        conn.close()
    return None if row is None else json.loads(row[0]).get("t1")


class TestAcrossEventLoopsCopiesAndConnections:
    """The order holds where a run's saves come from more than one event loop, a copy of its context, two
    connections to one file, or a provider with exporters; and a save from another loop never fails for it."""

    async def test_a_tools_nested_event_loops_store_every_save_of_its_run(self, monkeypatch):
        """A tool whose blocking handler runs two batches with ``asyncio.run()``, each saving the tool's own run
        context: two event loops saving one run. In each batch two saves overlap (as parallel nested calls ending
        together do), so the first batch's saves queue on its loop. Every save of the second batch is stored, none
        fails on a lock of the first loop, and the tool's own end is stored on its loop."""
        import contextvars

        from timbal.state import get_run_context

        main = asyncio.get_running_loop()
        entered = [0]
        arrival = contextvars.ContextVar("arrival", default=0)
        real = uploads.write_through

        async def counted(run_context, put):
            entered[0] += 1
            arrival.set(entered[0])
            return await real(run_context, put)

        monkeypatch.setattr(uploads, "write_through", counted)

        class Overlapping(Slow):
            """On a nested loop, the first save of each pair is stored only once the second has arrived."""

            @classmethod
            async def _store(cls, run_context):
                mine = arrival.get()
                if asyncio.get_running_loop() is not main and mine % 2 == 1:
                    await asyncio.wait_for(until(lambda: entered[0] > mine), DEADLINE)
                await super()._store(run_context)

        provider = Overlapping.fresh()

        def outer_handler() -> list:
            run = get_run_context()

            async def batch():
                return list(await asyncio.gather(run._save_trace(barrier=True), run._save_trace(barrier=True)))

            return [asyncio.run(batch()) for _ in range(2)]

        outer = Tool(name="outer", handler=outer_handler, offload_blocking=True, tracing_provider=provider)
        result = await asyncio.wait_for(outer().collect(), DEADLINE)
        assert result.status.code == "success"
        assert result.output == [[True, True], [True, True]]  # every save of both loops stored
        assert result.run_id in provider.stored

    async def test_a_copy_of_a_context_made_during_a_write_does_not_wait_for_it(self):
        """A run context is copied (``model_copy``) while its save is being written, as a fork would: the copy's
        own save does not wait for a write it never had."""
        provider = Remote.fresh()
        context = RunContext(tracing_provider=provider)
        context._trace["r"] = {"call_id": "r", "path": "a", "t0": 0, "parent_call_id": None}
        lands = provider.hold(1)
        writing = asyncio.create_task(uploads.write_through(context, provider.put))
        await asyncio.wait_for(until(lambda: provider.log), DEADLINE)
        forked = context.model_copy(update={"id": "forked"})
        lands.set()
        await asyncio.wait_for(writing, DEADLINE)
        assert await asyncio.wait_for(uploads.write_through(forked, provider.put), DEADLINE) is True
        assert "forked" in provider.stored

    def test_sqlite_numbers_a_snapshot_before_it_reads_the_run(self, tmp_path, monkeypatch):
        """Two event loops save one run to SQLite. The first is held while it reads the run (after the root,
        still running); the second saves the run's end. Whichever commits last, the file keeps the end."""
        path = tmp_path / "t.db"
        provider = SqliteTracingProvider.configured(_path=path)
        db = sqlite_module.database(path)
        reading, go = threading.Event(), threading.Event()

        class Slow:
            def __str__(self):
                if not reading.is_set():
                    reading.set()
                    assert go.wait(DEADLINE)
                return "output"

        class Watched:
            """The order lock, saying when a thread had to wait for it."""

            def __init__(self, lock):
                self.lock, self.waiting = lock, threading.Event()

            def __enter__(self):
                if not self.lock.acquire(blocking=False):
                    self.waiting.set()
                    self.lock.acquire()
                return self

            def __exit__(self, *_):
                self.lock.release()

        watched = Watched(db.writing_lock)
        monkeypatch.setattr(db, "writing_lock", watched)
        context = RunContext(tracing_provider=provider)
        context._trace["r"] = {"call_id": "r", "path": "a", "t0": 0, "parent_call_id": None, "output": Slow()}
        errors = []

        def save():
            try:
                asyncio.run(provider.put(context))
            except BaseException as e:  # noqa: BLE001 - reported below
                errors.append(e)

        first = threading.Thread(target=save, daemon=True)
        first.start()
        assert reading.wait(DEADLINE)
        context._trace["r"].t1 = 5
        context._trace["r"].output = "the end"
        second = threading.Thread(target=save, daemon=True)
        second.start()
        deadline = threading.Event()
        timer = threading.Timer(DEADLINE, deadline.set)
        timer.start()
        try:
            # The second save waits for the first one's read, or (without that) commits the end meanwhile.
            while not (watched.waiting.is_set() or stored_t1(path, context.id, "r") == 5 or deadline.is_set()):
                threading.Event().wait(0.001)
        finally:
            timer.cancel()
            go.set()
        first.join(DEADLINE)
        second.join(DEADLINE)
        assert not errors and not first.is_alive() and not second.is_alive()
        assert stored_t1(path, context.id, "r") == 5

    async def test_two_connections_to_one_file_share_the_order(self, tmp_path, monkeypatch):
        """A provider with ``synchronous="NORMAL"`` and one with ``"FULL"`` on one file get a connection each. A
        save through the first is held before it commits; the run's end is committed through the second; the
        held write then finds a newer one stored and leaves it."""
        path = tmp_path / "t.db"
        normal = SqliteTracingProvider.configured(_path=path)
        full = SqliteTracingProvider.configured(_path=path, synchronous="FULL")
        lock = HeldLock(sqlite_module.database(path), monkeypatch)
        lock.armed = True
        context = RunContext(tracing_provider=normal)
        context._trace["r"] = {"call_id": "r", "path": "a", "t0": 0, "parent_call_id": None}
        held = asyncio.create_task(normal.put(context))
        try:
            assert await asyncio.to_thread(lock.parked.wait, DEADLINE)
            context._trace["r"].t1 = 5
            await asyncio.wait_for(full.put(context), DEADLINE)
            assert stored_t1(path, context.id, "r") == 5
        finally:
            lock.go.set()
        await asyncio.wait_for(held, DEADLINE)
        assert stored_t1(path, context.id, "r") == 5

    async def test_exports_of_a_provider_that_orders_its_own_saves_are_ordered_too(self):
        """An in-memory provider with an exporter: its exports are awaited in ``put()``. A mid-run export is held
        while the run's end is saved; the exporter ends with the end."""
        from timbal.state.tracing.providers import InMemoryTracingProvider

        release, held = asyncio.Event(), asyncio.Event()
        sink = {}

        class Sink:
            async def export(self, run_context):
                t1 = run_context._trace["r"].t1  # what this export carries
                if not held.is_set():
                    held.set()
                    await release.wait()
                sink["t1"] = t1

        provider = InMemoryTracingProvider.configured(_exporters=[Sink()])
        context = RunContext(tracing_provider=provider)
        context._trace["r"] = {"call_id": "r", "path": "a", "t0": 0, "parent_call_id": None}
        first = asyncio.create_task(uploads.write_through(context, provider.put))
        await asyncio.wait_for(held.wait(), DEADLINE)
        context._trace["r"].t1 = 5
        second = asyncio.create_task(uploads.write_through(context, provider.put))
        await asyncio.wait_for(until(lambda: "_save_turn" in context.__dict__ or second.done()), DEADLINE)
        release.set()
        await asyncio.wait_for(asyncio.gather(first, second), DEADLINE)
        assert sink["t1"] == 5

    async def test_a_buffered_barrier_on_another_event_loop_is_stored(self):
        """Buffered uploads of a run are under way on this loop (an upload in flight holds the run's lock) when a
        barrier of the run comes from another event loop: it is stored, not left waiting on a lock of this loop."""
        provider = Remote.fresh(durability="buffered", flush_interval_s=0.0)
        context = RunContext(tracing_provider=provider)
        context._trace["r"] = {"call_id": "r", "path": "a", "t0": 0, "parent_call_id": None}
        lands = provider.hold(1)
        try:
            assert uploads.schedule(context)
            await asyncio.wait_for(until(lambda: provider.log), DEADLINE)  # the upload is in flight
            done = await asyncio.wait_for(
                asyncio.to_thread(lambda: asyncio.run(uploads.write_through(context, provider.put))), DEADLINE
            )
            assert done is True
            assert ("stored", 2) in provider.log
        finally:
            lands.set()
            for run in list(uploads._RUNS.values()):
                if run.task is not None:
                    run.task.cancel()
            uploads._reset_for_tests()


def remote_t1(provider, run_id):
    """The root's ``t1`` as ``Remote`` stored it last."""
    return next(span for span in json.loads(provider.stored[run_id]) if span["call_id"] == "r")["t1"]


async def give_way():
    """A few steps of this loop, for a write that must not start to show that it did (a bounded window: what is
    asserted afterwards is the order the writes were stored in, not that nothing happened in it)."""
    for _ in range(50):
        await asyncio.sleep(0)


class TestEachLoopKeepsItsOwnOrder:
    """Saves of one run from several event loops: each loop's saves stay in order, and no loop's save waits on, or
    releases, another loop's queue."""

    @staticmethod
    def context(provider):
        context = RunContext(tracing_provider=provider)
        context._trace["r"] = {"call_id": "r", "path": "a", "t0": 0, "parent_call_id": None}
        return context

    async def test_a_save_left_on_a_closed_loop_does_not_release_this_loops_queue(self):
        """A save of the run is left suspended on a loop that was closed. This loop's saves then queue (the second
        waits for the first); the abandoned save is collected meanwhile. The second still waits, and the run's
        newer state is what is stored last."""
        provider = Remote.fresh()
        context = self.context(provider)
        abandoned = {}

        def abandon():
            loop = asyncio.new_event_loop()
            try:

                async def main():
                    inside, never = asyncio.Event(), asyncio.Event()

                    async def put(_context):
                        inside.set()
                        await never.wait()

                    abandoned["task"] = asyncio.ensure_future(uploads.write_through(context, put))
                    await inside.wait()

                loop.run_until_complete(main())
            finally:
                loop.close()

        thread = threading.Thread(target=abandon, daemon=True)
        thread.start()
        await asyncio.to_thread(thread.join, DEADLINE)
        lands = provider.hold(1)
        first = asyncio.create_task(uploads.write_through(context, provider.put))
        await asyncio.wait_for(until(lambda: ("sent", 1) in provider.log), DEADLINE)
        context._trace["r"].t1 = 5
        second = asyncio.create_task(uploads.write_through(context, provider.put))
        await give_way()
        abandoned.clear()
        gc.collect()  # the abandoned save's cleanup runs here
        await give_way()
        lands.set()
        assert await asyncio.wait_for(asyncio.gather(first, second), DEADLINE) == [True, True]
        assert provider.log.index(("stored", 1)) < provider.log.index(("sent", 2))
        assert remote_t1(provider, context.id) == 5

    async def test_this_loops_saves_stay_in_order_while_another_loop_writes(self):
        """Another loop (a tool's nested ``asyncio.run()``) has a write of the run in flight. Two saves of the run on
        this loop still go one after the other, and the newer state is stored last."""
        provider = Remote.fresh()
        context = self.context(provider)
        inside, go = threading.Event(), threading.Event()

        async def nested_put(_context):
            inside.set()
            await asyncio.to_thread(go.wait, DEADLINE)

        thread = threading.Thread(target=lambda: asyncio.run(uploads.write_through(context, nested_put)), daemon=True)
        thread.start()
        try:
            assert await asyncio.to_thread(inside.wait, DEADLINE)
            lands = provider.hold(1)
            first = asyncio.create_task(uploads.write_through(context, provider.put))
            await asyncio.wait_for(until(lambda: ("sent", 1) in provider.log), DEADLINE)
            context._trace["r"].t1 = 5
            second = asyncio.create_task(uploads.write_through(context, provider.put))
            await give_way()
            lands.set()
            await asyncio.wait_for(asyncio.gather(first, second), DEADLINE)
        finally:
            go.set()
            await asyncio.to_thread(thread.join, DEADLINE)
        assert provider.log.index(("stored", 1)) < provider.log.index(("sent", 2))
        assert remote_t1(provider, context.id) == 5

    async def test_a_buffered_run_whose_loop_ended_is_taken_over(self):
        """A nested loop started the run's buffered uploads and ended. Back on this loop, an upload is in flight
        when the run's end is saved: the end waits for it, and is what is stored last."""
        provider = Remote.fresh(durability="buffered", flush_interval_s=0.0)
        context = self.context(provider)
        try:

            async def nested():
                assert uploads.schedule(context)
                await asyncio.wait_for(until(lambda: ("stored", 1) in provider.log), DEADLINE)

            thread = threading.Thread(target=lambda: asyncio.run(nested()), daemon=True)
            thread.start()
            await asyncio.to_thread(thread.join, DEADLINE)
            lands = provider.hold(2)
            assert uploads.schedule(context)
            await asyncio.wait_for(until(lambda: ("sent", 2) in provider.log), DEADLINE)
            context._trace["r"].t1 = 5
            end = asyncio.create_task(uploads.write_through(context, provider.put))
            await give_way()
            lands.set()
            await asyncio.wait_for(end, DEADLINE)
            await asyncio.wait_for(until(lambda: ("stored", 2) in provider.log), DEADLINE)
            assert provider.log.index(("stored", 2)) < provider.log.index(("sent", 3))
            assert remote_t1(provider, context.id) == 5
        finally:
            for run in list(uploads._RUNS.values()):
                if run.task is not None and not run.task.done():
                    run.task.cancel()
            uploads._reset_for_tests()

    def test_sqlite_with_a_redactor_numbers_a_snapshot_before_it_is_taken(self, tmp_path, monkeypatch):
        """Two event loops save one run through a SQLite provider with a trace redactor. The first is held before
        it numbers its snapshot; the second saves the run's end. The file keeps the end."""
        path = tmp_path / "t.db"
        provider = SqliteTracingProvider.configured(_path=path, _trace_redactor=lambda value: value)
        db = sqlite_module.database(path)
        real, parked, go, arming = db.writing_lock, threading.Event(), threading.Event(), [True]

        class Parking:
            """The order lock: the first thread to ask for it waits for ``go`` before it takes it."""

            def __enter__(self):
                mine, arming[0] = arming[0], False
                if mine:
                    parked.set()
                    assert go.wait(DEADLINE)
                real.acquire()
                return self

            def __exit__(self, *_):
                real.release()

        monkeypatch.setattr(db, "writing_lock", Parking())
        context = self.context(provider)
        errors = []

        def save():
            try:
                asyncio.run(provider.put(context))
            except BaseException as e:  # noqa: BLE001 - reported below
                errors.append(e)

        first = threading.Thread(target=save, daemon=True)
        first.start()
        try:
            assert parked.wait(DEADLINE)
            context._trace["r"].t1 = 5
            second = threading.Thread(target=save, daemon=True)
            second.start()
            second.join(DEADLINE)
            assert stored_t1(path, context.id, "r") == 5
        finally:
            go.set()
        first.join(DEADLINE)
        assert not errors and not first.is_alive()
        assert stored_t1(path, context.id, "r") == 5

    async def test_one_sqlite_file_under_two_names_keeps_one_order(self, tmp_path, monkeypatch):
        """Two providers on one file whose paths differ only in case (a case-insensitive file system): a save
        through the first is held before it commits, the end is committed through the second, then the held write
        finds a newer one stored and leaves it."""
        import os

        lower, upper = tmp_path / "t.db", tmp_path / "T.db"
        first, second = SqliteTracingProvider.configured(_path=lower), SqliteTracingProvider.configured(_path=upper)
        db = sqlite_module.database(lower)
        if not (upper.exists() and os.path.samefile(lower, upper)):
            pytest.skip("case-sensitive file system: two names are two files")
        lock = HeldLock(db, monkeypatch)
        lock.armed = True
        context = self.context(first)
        held = asyncio.create_task(first.put(context))
        try:
            assert await asyncio.to_thread(lock.parked.wait, DEADLINE)
            context._trace["r"].t1 = 5
            await asyncio.wait_for(second.put(context), DEADLINE)
        finally:
            lock.go.set()
        await asyncio.wait_for(held, DEADLINE)
        assert stored_t1(lower, context.id, "r") == 5


class TestCollectionsForksAndKeptLoops:
    """What runs a save's cleanup, or holds its locks, from outside: a collection, a fork, a loop kept but stopped."""

    @staticmethod
    def context(provider):
        context = RunContext(tracing_provider=provider)
        context._trace["r"] = {"call_id": "r", "path": "a", "t0": 0, "parent_call_id": None}
        return context

    async def test_a_collection_while_a_save_registers_keeps_this_loops_order(self, monkeypatch):
        """A save of the run is abandoned on a closed loop. This loop's first save sets off a collection as it
        registers (any allocation can), which finishes the abandoned save and its cleanup. This loop's second save
        still waits for the first, and the newer state is stored last."""
        provider = Remote.fresh()
        context = self.context(provider)
        abandoned, closed_out = {}, []

        def abandon():
            loop = asyncio.new_event_loop()
            try:

                async def main():
                    inside, never = asyncio.Event(), asyncio.Event()

                    async def put(_context):
                        inside.set()
                        try:
                            await never.wait()
                        finally:
                            closed_out.append(True)

                    abandoned["task"] = asyncio.ensure_future(uploads.write_through(context, put))
                    await inside.wait()

                loop.run_until_complete(main())
            finally:
                loop.close()

        thread = threading.Thread(target=abandon, daemon=True)
        thread.start()
        await asyncio.to_thread(thread.join, DEADLINE)
        armed = [True]

        class Collecting(uploads._Writing):
            __slots__ = ()

            def __init__(self):
                if armed[0]:
                    armed[0] = False
                    abandoned.clear()
                    gc.collect()

        gc.collect()
        gc.disable()
        monkeypatch.setattr(sys, "unraisablehook", lambda _info: None)  # "Task was destroyed but it is pending"
        try:
            monkeypatch.setattr(uploads, "_Writing", Collecting)
            lands = provider.hold(1)
            first = asyncio.create_task(uploads.write_through(context, provider.put))
            await asyncio.wait_for(until(lambda: ("sent", 1) in provider.log), DEADLINE)
            assert closed_out == [True]  # the abandoned save was finished by that collection
            context._trace["r"].t1 = 5
            second = asyncio.create_task(uploads.write_through(context, provider.put))
            await give_way()
            lands.set()
            assert await asyncio.wait_for(asyncio.gather(first, second), DEADLINE) == [True, True]
        finally:
            gc.enable()
        assert provider.log.index(("stored", 1)) < provider.log.index(("sent", 2))
        assert remote_t1(provider, context.id) == 5

    @pytest.mark.skipif(not hasattr(__import__("os"), "fork"), reason="needs os.fork")
    def test_a_child_forked_while_a_thread_holds_the_queue_lock_can_save(self):
        """Another thread holds the lock the queues are changed under when the process forks: the child's saves do
        not wait for a thread that is not there."""
        import os
        import select
        import warnings

        provider = Remote.fresh()
        holding, release = threading.Event(), threading.Event()

        def hold():
            with uploads._QUEUES_LOCK:
                holding.set()
                release.wait(DEADLINE)

        thread = threading.Thread(target=hold, daemon=True)
        thread.start()
        assert holding.wait(DEADLINE)
        read_end, write_end = os.pipe()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)  # forking a process that has threads
            pid = os.fork()
        if pid == 0:  # the child: one save of a run, then out
            report = "error"
            try:
                result = asyncio.run(uploads.write_through(self.context(provider), provider.put))
                report = f"ok {result}"
            finally:
                os.write(write_end, report.encode())
                os._exit(0)
        os.close(write_end)
        try:
            ready, _, _ = select.select([read_end], [], [], DEADLINE)
            report = os.read(read_end, 100).decode() if ready else "child hung"
        finally:
            release.set()
            thread.join(DEADLINE)
            os.close(read_end)
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass
            os.waitpid(pid, 0)
        assert report == "ok True"

    async def test_a_buffered_run_whose_loop_was_stopped_is_taken_over(self):
        """A helper ran nested work on a loop it keeps (``run_until_complete``, not closed) and scheduled a save of
        the run there. Back on this loop, the root's end and the background work's end are two barriers: they go
        one after the other, and the newer state is stored last."""
        provider = Remote.fresh(durability="buffered", flush_interval_s=0.0)
        context = self.context(provider)
        kept = {}
        try:

            def helper():
                loop = kept["loop"] = asyncio.new_event_loop()

                async def nested():
                    assert uploads.schedule(context)
                    await asyncio.wait_for(until(lambda: ("stored", 1) in provider.log), DEADLINE)
                    await asyncio.wait_for(until(lambda: not uploads.pending_runs()), DEADLINE)

                loop.run_until_complete(nested())

            thread = threading.Thread(target=helper, daemon=True)
            thread.start()
            await asyncio.to_thread(thread.join, DEADLINE)
            lands = provider.hold(2)
            first = asyncio.create_task(uploads.write_through(context, provider.put))
            await asyncio.wait_for(until(lambda: ("sent", 2) in provider.log), DEADLINE)
            context._trace["r"].t1 = 5
            second = asyncio.create_task(uploads.write_through(context, provider.put))
            await give_way()
            lands.set()
            await asyncio.wait_for(asyncio.gather(first, second), DEADLINE)
            assert provider.log.index(("stored", 2)) < provider.log.index(("sent", 3))
            assert remote_t1(provider, context.id) == 5
        finally:
            for run in list(uploads._RUNS.values()):
                if run.task is not None and not run.task.done():
                    run.task.cancel()
            uploads._reset_for_tests()
            if "loop" in kept:
                kept["loop"].close()

    @pytest.mark.parametrize("redactor", [None, lambda value: value], ids=["no redactor", "a redactor"])
    async def test_a_sqlite_subclass_store_is_called(self, tmp_path, redactor):
        """A ``SqliteTracingProvider`` subclass that wraps ``_store`` (as tests and applications do): ``put`` goes
        through it, with or without a trace redactor."""
        calls = []

        class Wrapped(SqliteTracingProvider):
            @classmethod
            async def _store(cls, run_context):
                calls.append(run_context.id)
                await super()._store(run_context)

        provider = Wrapped.configured(_path=tmp_path / "t.db", _trace_redactor=redactor)
        context = self.context(provider)
        context._trace["r"].t1 = 1
        await asyncio.wait_for(provider.put(context), DEADLINE)
        assert calls == [context.id]
        assert stored_t1(tmp_path / "t.db", context.id, "r") == 1


class TestExportsOfDroppedWrites:
    @pytest.mark.parametrize("redactor", [None, lambda value: value], ids=["no redactor", "a redactor"])
    async def test_a_sqlite_write_the_file_was_newer_than_is_not_exported(self, tmp_path, monkeypatch, redactor):
        """A SQLite save of a run is numbered and held before it commits; the run's end is saved meanwhile
        (another event loop's save, not queued behind it), committed and exported. The held write then finds the
        end stored and leaves it: it exports nothing either, so the exporter ends with the end."""
        path = tmp_path / "t.db"
        exported = []

        class Sink:
            async def export(self, run_context):
                exported.append(run_context._trace["r"].t1)

        provider = SqliteTracingProvider.configured(_path=path, _trace_redactor=redactor, _exporters=[Sink()])
        lock = HeldLock(sqlite_module.database(path), monkeypatch)
        lock.armed = True
        context = RunContext(tracing_provider=provider)
        context._trace["r"] = {"call_id": "r", "path": "a", "t0": 0, "parent_call_id": None}
        held = asyncio.create_task(provider.put(context))
        try:
            assert await asyncio.to_thread(lock.parked.wait, DEADLINE)
            context._trace["r"].t1 = 5
            await asyncio.wait_for(provider.put(context), DEADLINE)
        finally:
            lock.go.set()
        await asyncio.wait_for(held, DEADLINE)
        assert stored_t1(path, context.id, "r") == 5
        assert exported == [5]
