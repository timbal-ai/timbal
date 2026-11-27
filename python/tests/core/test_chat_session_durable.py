"""Durable chat sessions: the session layer stored ahead of every change, restored after a restart or a crash."""

import asyncio
import copy
from typing import Any

import pytest
from timbal import Agent, Tool
from timbal.core.chat_session import ChatSession
from timbal.core.test_model import TestModel
from timbal.state import RunContext, set_call_id, set_parent_call_id, set_run_context
from timbal.state.background import forget_idle_background_store
from timbal.state.chat_store import (
    InMemoryChatSessionStore,
    SessionLeased,
    SessionNotStored,
    SessionSnapshot,
    SnapshotConflict,
    SqliteChatSessionStore,
    process_owner,
    storable,
)
from timbal.state.tracing.providers import SqliteTracingProvider
from timbal.types.approval import ApprovalResolution, Cancel
from timbal.types.content import TextContent, ToolUseContent
from timbal.types.events import ApprovalEvent, OutputEvent
from timbal.types.message import Message

# A test that times out because nothing in its process ran, or because something blocked the event loop, says
# so, with the stacks (conftest.py `loop_watchdog`): on loaded CI runners a test process has stood still for
# 70-260 s, and the sessions it was waiting for looked stuck.
pytestmark = pytest.mark.usefixtures("loop_watchdog")


@pytest.fixture(autouse=True)
def isolated_context():
    set_run_context(None)
    set_call_id(None)
    set_parent_call_id(None)
    yield
    set_run_context(None)


class CrashableSqlite(SqliteTracingProvider):
    """Stops writing once ``crashed``: what a killed process stores after its death."""

    crashed = False

    @classmethod
    async def _store(cls, run_context):
        if not cls.crashed:
            await super()._store(run_context)


@pytest.fixture
def db(tmp_path):
    provider = CrashableSqlite.configured(_path=tmp_path / "timbal.db", crashed=False)
    return provider, SqliteChatSessionStore(tmp_path / "timbal.db")


def _stable_id(value):
    """A parametrize id that is the same in every process: pytest-xdist workers must collect identical ids,
    and ``repr`` is not (an ``object()``'s address, an ``ApprovalResolution``'s ``decided_at``)."""
    if isinstance(value, ApprovalResolution):
        return f"ApprovalResolution(approved={value.approved})"
    if type(value) is object:
        return "object()"
    return repr(value)


def tool_call(name, call_id):
    return Message(role="assistant", content=[ToolUseContent(id=call_id, name=name, input={})], stop_reason="tool_use")


def last_user_text(messages):
    for message in reversed(messages):
        if message.role == "user":
            return message.collect_text()
    return ""


def handler(messages):
    """``start`` calls ``work``, ``pay`` the gated ``charge``; a recovery notice is answered ``recovered``."""
    last = messages[-1]
    text = last_user_text(messages)
    if last.role == "tool":
        return "done"
    if "cut off by a server restart" in text:
        return "recovered"
    if text == "start":
        return tool_call("work", f"call-{len(messages)}")
    if text == "pay":
        return tool_call("charge", f"call-{len(messages)}")
    return f"echo: {text}"


def two_step(messages):
    """``lookup`` completes, then ``work`` (which the test cuts) runs."""
    if "cut off by a server restart" in last_user_text(messages):
        return "recovered"
    results = [m for m in messages if m.role == "tool"]
    if not results:
        return tool_call("lookup", "call-lookup")
    if len(results) == 1:
        return tool_call("work", "call-work")
    return "done"


class Gate:
    def __init__(self, *, released=False):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        if released:
            self.release.set()
        self.calls = 0

    async def work(self) -> str:
        self.calls += 1
        self.entered.set()
        await self.release.wait()
        return "worked"


def make_agent(provider, gate, charged=None):
    return Agent(
        name="agent",
        model=TestModel(handler=handler),
        tools=[
            Tool(name="work", handler=gate.work),
            Tool(
                name="charge",
                handler=lambda: (charged.append(1) if charged is not None else None) or "charged",
                requires_approval=True,
            ),
        ],
        tracing_provider=provider,
    )


async def records(session, kind, count, timeout=30.0, after=0):  # a hang guard (slow runners stretch timers)
    found = []

    async def collect():
        async for item in session.events(after):
            if isinstance(item, dict) and item.get("kind") == kind:
                found.append(item)
                if len(found) >= count:
                    return

    await asyncio.wait_for(collect(), timeout)
    return found


async def until(condition, timeout=30.0):
    """Wait for ``condition()``. The budget is for a slow machine: one where it never holds fails here."""

    async def poll():
        while not condition():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout)


async def ended_unsettled(session, timeout=30.0):
    """The turn's run has ended and the session still holds the turn: its trace is not stored yet."""

    async def ended():
        while session._turn is None or not session._turn.done():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(ended(), timeout)


def held_before_its_turn(session) -> asyncio.Event:
    """Keep the session's worker from taking a turn until the returned event is set: what a test does in
    between (store its message, break the store) then comes before the turn on any machine."""
    go, real_admit = asyncio.Event(), session._admit

    async def admit():
        await go.wait()
        await real_admit()

    session._admit = admit
    return go


async def worker_stopped(session):
    """Stop a session's worker while it waits (no turn running) and keep it stopped: what the session is sent
    from then on is stored and stays queued, as in a process that dies before its worker gets to it."""
    assert session._turn is None, "a turn is running"
    session._hold_worker = True
    worker = session._worker
    if worker is not None:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)


def root_outputs(session):
    return [
        event
        for event in session.log.peek(session.log.forgotten_through)[0]
        if isinstance(event, OutputEvent) and event.parent_call_id is None and event.path == "agent"
    ]


async def crash(session):
    """The process dies: nothing more is stored, the session's tasks just stop."""
    session._agent.tracing_provider.crashed = True
    await session.flush()
    session._lost = True  # stores nothing more
    tasks = [task for task in (session._worker, session._turn, session._lease_task) if task is not None]
    for task in tasks:
        task.cancel()
    # Its turn's teardown must be over before the test lets the provider write again.
    await asyncio.gather(*tasks, return_exceptions=True)
    # Process-local state dies with the process: the scheduler's hold on the run's task store too.
    session.closed = True
    session._scheduler.release(session._store)
    forget_idle_background_store(session._store)


class Charge:
    """An approval-gated tool that waits until released (a test crashes the process mid-charge)."""

    def __init__(self, *, released=False):
        self.gate = Gate(released=released)

    async def charge(self) -> str:
        await self.gate.work()
        return "charged"


def gated_agent(provider, charge, *, extra_tools=(), model_handler=handler):
    return Agent(
        name="agent",
        model=TestModel(handler=model_handler),
        tools=[
            Tool(name="work", handler=Gate(released=True).work),
            Tool(name="charge", handler=charge.charge, requires_approval=True),
            *extra_tools,
        ],
        tracing_provider=provider,
    )


async def parked(provider, store, charge):
    session = ChatSession(gated_agent(provider, charge), session_store=store)
    session.send("pay")
    await records(session, "turn_finished", 1)
    approval = next(e for e in session.log.peek()[0] if isinstance(e, ApprovalEvent))
    return session, approval


def after_restore(session):
    return session.log.forgotten_through


async def stored_answer(provider, run_id):
    from timbal.state import RunContext

    trace = await provider.get(RunContext(parent_id=run_id, tracing_provider=provider))
    return trace.get_path("agent")[0].output["content"][0]["text"]


async def one_turn(session, text):
    after = session.log.cursor_end
    session.send(text)
    await records(session, "turn_finished", 1, after=after)
    return session.head


class TestRootRunId:
    """``root_run_id`` is the thread's root run as the platform knows it, sent as the MCP session hint
    (``RunContext._session_id``): a caller's ``session_id`` (the platform's chat reserves it as the root and
    files the first run under it), else the run that started the thread. Never a run id taken from a caller."""

    async def test_a_generated_id_is_the_first_turns_run_id(self, db):
        provider, _ = db
        async with ChatSession(make_agent(provider, Gate(released=True))) as session:
            assert session.root_run_id is None  # no thread yet
            assert await one_turn(session, "hi") == session.id
            assert session.root_run_id == session.id == session._last_context._session_id
            second = await one_turn(session, "again")
            assert session._last_context._session_id == session.id
        assert second != session.id and session.root_run_id == session.id

    async def test_a_caller_session_id_is_the_hint_and_never_a_run_id(self, db):
        provider, _ = db
        async with ChatSession(make_agent(provider, Gate(released=True)), session_id="conv-42") as first:
            assert first.root_run_id == "conv-42"
            run_1 = await one_turn(first, "one")
            assert run_1 != "conv-42" and first._last_context.parent_id is None
            assert first._last_context._session_id == "conv-42"
            await one_turn(first, "again")
            assert first._last_context._session_id == first.root_run_id == "conv-42"
        # The same conversation key again, without from_store(): a new first run, never the old one's id.
        async with ChatSession(make_agent(provider, Gate(released=True)), session_id="conv-42") as second:
            run_2 = await one_turn(second, "two")
        assert run_2 not in (run_1, "conv-42") and second.root_run_id == "conv-42"
        assert await stored_answer(provider, run_1) == "echo: one"  # the first thread's root is untouched

    async def test_continuing_from_a_parent_leaves_the_root_unknown(self, db):
        provider, _ = db
        async with ChatSession(make_agent(provider, Gate(released=True))) as first:
            head = await one_turn(first, "one")
        for session_id in (None, "conv-9"):
            agent = make_agent(provider, Gate(released=True))
            async with ChatSession(agent, parent_id=head, session_id=session_id) as session:
                assert session.root_run_id is None
                await one_turn(session, "two")
                assert session.root_run_id is None and session._last_context._session_id is None

    async def test_restore_keeps_the_root_and_an_older_snapshot_claims_none(self, db):
        provider, store = db
        session = ChatSession(make_agent(provider, Gate(released=True)), session_id="conv-7", session_store=store)
        await one_turn(session, "one")
        await session.close(keep=True)
        restored = await ChatSession.from_store(make_agent(provider, Gate(released=True)), store, "conv-7")
        assert restored.root_run_id == "conv-7"
        await one_turn(restored, "two")
        assert restored._last_context._session_id == "conv-7"
        await restored.close(keep=True)

        snapshot = await store.load("conv-7")
        snapshot.root_run_id = None  # written before snapshots recorded the root
        await store.save(snapshot, expect_version=snapshot.version)
        older = await ChatSession.from_store(make_agent(provider, Gate(released=True)), store, "conv-7")
        assert older.root_run_id is None
        await one_turn(older, "three")
        assert older._last_context._session_id is None
        await older.close()

    async def test_a_generated_session_restored_before_its_first_turn(self, db):
        """Restored, a session never runs under its id: the run that starts the thread is the root."""
        provider, store = db
        session = ChatSession(make_agent(provider, Gate(released=True)), session_store=store)
        session.persist()
        await session.flush()
        await session.close(keep=True)
        restored = await ChatSession.from_store(make_agent(provider, Gate(released=True)), store, session.id)
        assert restored.root_run_id is None
        first = await one_turn(restored, "hi")
        assert first != session.id and restored.root_run_id == first == restored._last_context._session_id
        assert (await store.load(session.id)).root_run_id == first
        await restored.close()

    @pytest.mark.parametrize("steps", range(16))
    async def test_the_session_id_is_one_run_at_most(self, db, steps):
        """Swept over where an interrupt lands in the first turn. A turn cut before it started hands its message
        back, and the turn that answers it is another run: never a second run under the session's id (the
        session's records of the cut attempt would name the run that answers). The run that did start the thread
        is its root."""
        provider, _ = db
        session = ChatSession(make_agent(provider, Gate(released=True)))
        session.send("hi")
        for _ in range(steps):
            await asyncio.sleep(0)
        session.interrupt()
        await until(lambda: not session.busy and not session.pending_messages())
        log = [
            r for r in session.log.peek()[0] if isinstance(r, dict) and r.get("kind") in ("turn_started", "interrupt")
        ]
        await session.close()
        started = [r["run_id"] for r in log if r["kind"] == "turn_started"]
        assert len(set(started)) == len(started)
        for i, record in enumerate(log):
            before = [r["run_id"] for r in log[:i] if r["kind"] == "turn_started"]
            if record["kind"] == "interrupt" and record.get("run_id") and record["run_id"] not in before:
                cut = record["run_id"]  # cut before it started
                assert cut not in [r["run_id"] for r in log[i:] if r["kind"] == "turn_started"], f"{cut} ran again"
        assert session.root_run_id == (started[0] if started else None)

    async def test_an_older_snapshot_with_its_first_turn_in_flight_claims_no_root(self, db):
        """Written before the root was recorded (the platform's chat at the upgrade: a rolling deploy cuts the
        first turns still running), the dead first run recovery adopts is no proof of the thread's root: for the
        platform's chat that is its session id. No hint, as before the field, not the dead run's id."""
        provider, store = db
        gate = Gate()
        session = ChatSession(make_agent(provider, gate), session_id="conv-old", session_store=store)
        session.send("start")
        await asyncio.wait_for(gate.entered.wait(), 30)
        await crash(session)
        provider.crashed = False
        gate.release.set()
        snapshot = await store.load("conv-old")
        assert snapshot.in_flight is not None and snapshot.in_flight["parent_id"] is None
        snapshot.root_run_id = None  # neither field existed
        snapshot.in_flight.pop("starts_thread", None)
        await store.save(snapshot, expect_version=snapshot.version)
        restored = await ChatSession.from_store(make_agent(provider, Gate(released=True)), store, "conv-old")
        await records(restored, "turn_finished", 1, after=after_restore(restored))  # the recovery turn
        assert restored.root_run_id is None and restored._last_context._session_id is None
        await one_turn(restored, "after")
        assert restored._last_context._session_id is None
        await restored.close()

    async def test_a_first_turn_that_died_before_acting_does_not_name_the_thread(self, db):
        """Cut while its model thought (nothing dispatched), its message runs again under another id: that run
        starts the thread, and the hint follows it, not the id the dead run had."""
        provider, store = db
        model = ThinkingModel(slow={"first"})
        agent = Agent(name="agent", model=model, tools=[], tracing_provider=provider)
        session = ChatSession(agent, session_store=store)
        session.send("first")
        await asyncio.wait_for(model.thinking.wait(), 30)
        await session.flush()  # (the snapshot of its start may still be on its way)
        assert (await store.load(session.id)).root_run_id == session.id  # stored as it started
        await crash(session)
        provider.crashed = False
        model.slow.clear()
        restored = await ChatSession.from_store(agent, store, session.id)
        await records(restored, "turn_finished", 1, after=after_restore(restored))
        assert model.answered == ["first"]
        assert restored.head != session.id and restored._last_context.parent_id is None
        assert restored.root_run_id == restored.head == restored._last_context._session_id
        await restored.flush()
        assert (await store.load(session.id)).root_run_id == restored.head
        await restored.close()


# What a snapshot held before root_run_id (timbal 54473c37, the version the platform's chat pins).
FIELDS_BEFORE_ROOT_RUN_ID = (
    "session_id",
    "agent",
    "config",
    "head",
    "turns",
    "last_status",
    "parked",
    "last_prompt",
    "last_options",
    "in_flight",
    "pending",
    "wakes",
    "resume",
    "log_cursor",
    "closed",
    "version",
)


def test_a_snapshot_built_by_position_keeps_its_fields():
    """``root_run_id`` is the last field: a snapshot built by position before it existed means the same."""
    snapshot = SessionSnapshot("s", "agent", {}, "head", 3, {"code": "success"}, True)
    assert (snapshot.head, snapshot.turns, snapshot.last_status, snapshot.parked) == (
        "head",
        3,
        {"code": "success"},
        True,
    )
    assert snapshot.root_run_id is None


class TriggerRecordingSqlite(CrashableSqlite):
    """Records, for every save it is asked for, the run and the trigger its root carries then."""

    saves: list = []

    @classmethod
    async def _store(cls, run_context):
        root = run_context.root_span()
        if root is not None:
            cls.saves.append((run_context.id, root.metadata.get("trigger")))
        await super()._store(run_context)


class TestEveryStoredTurnSaysWhatStartedIt:
    @pytest.mark.parametrize("durability", ["sync", "buffered"])
    async def test_every_save_of_every_turn_carries_its_trigger(self, tmp_path, durability):
        """From a turn's first save to its last, recovery turns included: the trigger is on the root as it is
        made, from the turn's run context (a store-before-acting provider saves before a tool batch)."""
        provider = TriggerRecordingSqlite.configured(
            _path=tmp_path / "timbal.db", crashed=False, saves=[], durability=durability, store_before_acting=True
        )
        store = SqliteChatSessionStore(tmp_path / "timbal.db")
        gate = Gate()
        session = ChatSession(make_agent(provider, gate), session_id="conv-t", session_store=store)
        session.send("start")
        await asyncio.wait_for(gate.entered.wait(), 30)
        await crash(session)
        provider.crashed = False
        restored = await ChatSession.from_store(make_agent(provider, Gate(released=True)), store, "conv-t")
        await records(restored, "turn_finished", 1, after=after_restore(restored))  # the recovery turn
        await one_turn(restored, "after")
        started = {
            r["run_id"]: r["trigger"]
            for r in restored.log.peek(after_restore(restored))[0]
            if isinstance(r, dict) and r.get("kind") == "turn_started"
        }
        await restored.close()
        assert [t["kind"] for t in started.values()] == ["recovery", "user_message"]
        saves = [(run_id, trigger) for run_id, trigger in provider.saves if run_id in started]
        assert {run_id for run_id, _ in saves} == set(started)
        assert all(trigger == started[run_id] for run_id, trigger in saves), saves


class TestThePlatformChatsPath:
    """The platform's own chat (monolith ``services/chat``, pinned to timbal 54473c37) hosts a session exactly so
    (``SessionHost._bring_in``): ``from_store``, else ``ChatSession(agent, session_id=<the platform's id>,
    session_store=...)`` with no ``parent_id``, stored before its first turn; let go with ``close(keep=True)``
    and brought back with ``from_store``. Its store keeps the snapshot as a JSON object and hands it back to
    ``SessionSnapshot.from_json``; its trace provider is buffered and stores before acting.

    Nothing it relied on changes, whether the snapshot it restores was written before this field or after: the
    session's id is the platform's, the first turn runs under a run id of its own with no parent (the platform
    files it under the session's reserved root run), and the next turn continues it."""

    PLATFORM_ID = "06b04a1f3c2d7e5f8000a1b2c3d4e5f6"

    @staticmethod
    def _agent(provider):
        return Agent(
            name="chat",
            model=TestModel(handler=handler),
            system_prompt="You are a helpful assistant.",
            tracing_provider=provider,
        )

    @staticmethod
    async def _bring_in(agent, store, session_id):
        try:
            return await ChatSession.from_store(agent, store, session_id)
        except LookupError:
            pass
        session = ChatSession(agent, session_id=session_id, session_store=store)
        await session.restore()
        session.persist()
        await session.flush()
        return session

    @staticmethod
    def _as_written_before_root_run_id(store, session_id):
        """The stored snapshot as 54473c37 wrote it: its JSON object without ``root_run_id`` (nor the
        ``answer_format`` that came after it)."""
        import json

        stored, holder, until = store._rows[session_id]
        runtime = json.loads(stored.to_json())
        runtime.pop("version")
        assert set(runtime) - {"root_run_id", "answer_format"} == set(FIELDS_BEFORE_ROOT_RUN_ID) - {"version"}
        del runtime["root_run_id"], runtime["answer_format"]
        older = SessionSnapshot.from_json(json.dumps(runtime))
        older.version = stored.version
        store._rows[session_id] = (older, holder, until)

    @staticmethod
    async def _memory(provider, run_id):
        trace = await provider.get(RunContext(parent_id=run_id, tracing_provider=provider))
        return [
            block["text"]
            for message in trace.get_path("chat")[0].memory
            for block in message["content"]
            if block.get("type") == "text"
        ]

    @pytest.mark.parametrize("written", ["before root_run_id", "with root_run_id"])
    async def test_open_a_turn_let_go_restore_a_turn(self, tmp_path, written):
        provider = CrashableSqlite.configured(
            _path=tmp_path / "timbal.db", crashed=False, durability="buffered", store_before_acting=True
        )
        store = InMemoryChatSessionStore()
        session = await self._bring_in(self._agent(provider), store, self.PLATFORM_ID)
        assert session.id == self.PLATFORM_ID and session.head is None
        assert (await store.load(self.PLATFORM_ID)).head is None  # stored before its first turn
        first = await one_turn(session, "one")
        assert first != self.PLATFORM_ID and session._last_context.parent_id is None
        assert session._last_context._session_id == self.PLATFORM_ID  # the hint: the platform's thread
        await session.close(keep=True)
        if written == "before root_run_id":
            self._as_written_before_root_run_id(store, self.PLATFORM_ID)

        restored = await self._bring_in(self._agent(provider), store, self.PLATFORM_ID)
        assert restored.id == self.PLATFORM_ID and restored.head == first and restored.turns == 1
        second = await one_turn(restored, "two")
        assert second not in (first, self.PLATFORM_ID) and restored._last_context.parent_id == first
        assert await self._memory(provider, second) == ["one", "echo: one", "two", "echo: two"]
        # Before this field nothing was sent; restored from such a snapshot, still nothing (the root is unknown).
        hint = None if written == "before root_run_id" else self.PLATFORM_ID
        assert restored._last_context._session_id == hint
        await restored.close(keep=True)
        assert (await store.load(self.PLATFORM_ID)).head == second


class TestEdges:
    async def test_a_queued_message_waits_for_the_recovery_turn(self, db):
        provider, store = db
        gate = Gate()
        session = ChatSession(make_agent(provider, gate), session_store=store)
        session.send("start")
        await asyncio.wait_for(gate.entered.wait(), 30)
        session.send("also this", mode="followup")
        await crash(session)
        provider.crashed = False

        restored = await ChatSession.from_store(make_agent(provider, Gate(released=True)), store, session.id)
        started = await records(restored, "turn_started", 2, after=after_restore(restored))
        await records(restored, "turn_finished", 2, after=after_restore(restored))
        assert [s["trigger"]["kind"] for s in started] == ["recovery", "user_message"]
        assert [o.output.collect_text() for o in root_outputs(restored)] == ["recovered", "echo: also this"]
        await restored.close()

    async def test_answers_stored_before_the_crash_resume_the_parked_turn(self, db):
        provider, store = db
        session, approval = await parked(provider, store, Charge(released=True))
        await session.close(keep=True)
        snapshot = await store.load(session.id)
        snapshot.resume = {approval.approval_id: True}  # resume() was stored, the turn never started
        await store.save(snapshot, expect_version=snapshot.version)

        charge = Charge(released=True)
        restored = await ChatSession.from_store(gated_agent(provider, charge), store, session.id)
        (finished,) = await records(restored, "turn_finished", 1, after=after_restore(restored))
        assert finished["status"]["code"] == "success" and not restored.parked
        assert charge.gate.calls == 1
        await restored.close()

    async def test_a_crash_during_a_resumed_approval_turn_is_recovered_without_charging_again(self, db):
        provider, store = db
        first = Charge()
        session, approval = await parked(provider, store, first)
        session.resume({approval.approval_id: True})
        await asyncio.wait_for(first.gate.entered.wait(), 30)
        await crash(session)
        provider.crashed = False

        again = Charge(released=True)
        restored = await ChatSession.from_store(gated_agent(provider, again), store, session.id)
        started = await records(restored, "turn_started", 2, after=after_restore(restored))
        await records(restored, "turn_finished", 2, after=after_restore(restored))
        # The question is abandoned (a resume that cancels it), not answered again: no second charge, and
        # nothing the model called after it runs again either. The recovery turn names the approved call.
        assert [s["trigger"]["kind"] for s in started] == ["resume", "recovery"]
        text = Message.validate(root_outputs(restored)[1].input["prompt"]).collect_text()
        assert "Started but not confirmed" in text and "charge(" in text
        assert first.gate.calls == 1 and again.gate.calls == 0
        assert not restored.parked
        await restored.close()

    async def test_a_dead_resume_is_not_answered_again_without_approval_claims_either(self, tmp_path):
        from timbal.state.tracing.providers.base import TracingProvider

        class NoClaims(CrashableSqlite):
            claim_approval = TracingProvider.claim_approval

        provider = NoClaims.configured(_path=tmp_path / "timbal.db", crashed=False)
        store = SqliteChatSessionStore(tmp_path / "timbal.db")
        first = Charge()
        session, approval = await parked(provider, store, first)
        session.resume({approval.approval_id: True})
        await asyncio.wait_for(first.gate.entered.wait(), 30)
        await crash(session)
        provider.crashed = False

        again = Charge(released=True)
        restored = await ChatSession.from_store(gated_agent(provider, again), store, session.id)
        started = await records(restored, "turn_started", 2, after=after_restore(restored))
        await records(restored, "turn_finished", 2, after=after_restore(restored))
        assert [s["trigger"]["kind"] for s in started] == ["resume", "recovery"]
        # Not left parked for the user to approve a second time: that would charge again.
        assert not restored.parked and again.gate.calls == 0
        await restored.close()

    async def test_a_second_crash_during_the_recovery_turn(self, db):
        provider, store = db
        checking = Gate()

        def recovering(messages):
            text = last_user_text(messages)
            if "cut off by a server restart" in text:
                return "recovered" if messages[-1].role == "tool" else tool_call("check", f"check-{len(messages)}")
            return handler(messages)

        def agent(gate, check_gate):
            return Agent(
                name="agent",
                model=TestModel(handler=recovering),
                tools=[
                    Tool(name="work", handler=gate.work),
                    Tool(name="check", handler=check_gate.work, read_only=True),
                ],
                tracing_provider=provider,
            )

        gate = Gate()
        session = ChatSession(agent(gate, checking), session_store=store)
        session.send("start")
        await asyncio.wait_for(gate.entered.wait(), 30)
        await crash(session)
        provider.crashed = False

        second = await ChatSession.from_store(agent(Gate(), checking), store, session.id)
        await asyncio.wait_for(checking.entered.wait(), 30)  # the recovery turn is checking: crash again
        recovery_run = second.run_id
        await crash(second)
        provider.crashed = False

        third = await ChatSession.from_store(agent(Gate(), Gate(released=True)), store, session.id)
        (started,) = await records(third, "turn_started", 1, after=after_restore(third))
        await records(third, "turn_finished", 1, after=after_restore(third))
        assert started["trigger"]["dead_run_id"] == recovery_run
        text = Message.validate(root_outputs(third)[0].input["prompt"]).collect_text()
        assert "check(" in text
        assert root_outputs(third)[0].output.collect_text() == "recovered"
        await third.close()

    async def test_wakes_survive_with_their_tool_policies(self, db):
        provider, store = db
        gate = Gate()
        session = ChatSession(make_agent(provider, gate), session_store=store)
        session.send("start")
        await asyncio.wait_for(gate.entered.wait(), 30)
        session.wake("routine", key="r", tool_policy=["work"])
        session.wake("custom", key="c", tool_policy=lambda _tool: True)
        await crash(session)
        provider.crashed = False

        restored = await ChatSession.from_store(make_agent(provider, Gate(released=True)), store, session.id)
        wakes = [(w.kind, w.key, w.tool_policy) for w in restored._wakes]
        assert wakes == [
            ("recovery", "recovery", "read_only"),
            ("wake", "r", frozenset({"work"})),
            # A callable cannot be stored, and "read_only" could be wider than it was: no tools at all.
            ("wake", "c", frozenset()),
        ]
        await records(restored, "turn_finished", 3, after=after_restore(restored))
        await restored.close()

    async def test_interrupting_a_restored_parked_session_abandons_the_approval(self, db):
        provider, store = db
        charge = Charge(released=True)
        session, _ = await parked(provider, store, charge)
        await session.close(keep=True)

        restored = await ChatSession.from_store(gated_agent(provider, charge), store, session.id)
        restored.send("never mind", mode="interrupt")
        finished = await records(restored, "turn_finished", 2, after=after_restore(restored))
        assert finished[0]["status"]["code"] == "cancelled"
        assert root_outputs(restored)[-1].output.collect_text() == "echo: never mind"
        assert charge.gate.calls == 0
        await restored.close()

    async def test_an_owner_whose_lease_was_taken_cannot_write_before_the_new_owner_does(self, db):
        """Split brain: the lease lapsed (or renewing it failed) and another process took it, but has not
        written yet. The old owner's next save must fail, not the new owner's."""
        provider, store = db
        session = ChatSession(make_agent(provider, Gate(released=True)), session_store=store)
        session.send("hi")
        await records(session, "turn_finished", 1)
        await session.flush()
        session._lease_task.cancel()  # renewals stopped: nothing notices the takeover but the save
        await store.release_lease(session.id, process_owner())  # as good as lapsed
        await store.acquire_lease(session.id, "elsewhere:1:x")
        version = (await store.load(session.id)).version
        session.wake("ping", key="p")
        await until(lambda: session.closed)
        assert session.closed and session.handed_off
        assert (await store.load(session.id)).version == version  # the new owner's version still stands

    async def test_a_restored_parked_wake_still_does_not_take_steered_messages(self, db):
        provider, store = db

        def pays(messages):
            if messages[-1].role == "tool":
                return "paid"
            text = last_user_text(messages)
            return tool_call("charge", "call-pay") if "pay" in text else f"echo: {text}"

        session = ChatSession(gated_agent(provider, Charge(), model_handler=pays), session_store=store)
        session.wake("pay the invoice", key="pay", tool_policy=None)
        await records(session, "turn_finished", 1)
        approval = next(e for e in session.log.peek()[0] if isinstance(e, ApprovalEvent))
        session.send("hello", mode="steer")  # parked: waits
        await session.flush()
        await session.close(keep=True)
        assert (await store.load(session.id)).last_options["steer"] is False

        restored = await ChatSession.from_store(
            gated_agent(provider, Charge(released=True), model_handler=pays), store, session.id
        )
        restored.resume({approval.approval_id: True})
        started = await records(restored, "turn_started", 2, after=after_restore(restored))
        await records(restored, "turn_finished", 2, after=after_restore(restored))
        # The wake's turn finishes alone; the user's message gets a turn of its own, with every tool.
        assert [s["trigger"]["kind"] for s in started] == ["resume", "user_message"]
        assert [o.output.collect_text() for o in root_outputs(restored)] == ["paid", "echo: hello"]
        await restored.close()

    async def test_a_newer_snapshot_from_another_owner_stops_the_session(self, db):
        provider, store = db
        session = ChatSession(make_agent(provider, Gate(released=True)), session_store=store)
        session.send("hi")
        await records(session, "turn_finished", 1)
        await session.flush()
        foreign = await store.load(session.id)
        foreign.turns = 99
        await store.save(foreign, expect_version=foreign.version)
        session.wake("ping", key="p")
        await until(lambda: session.closed)
        assert session.closed
        assert (await store.load(session.id)).turns == 99

    async def test_a_failing_store_does_not_hang_and_the_next_change_retries(self, db, monkeypatch):
        provider, store = db
        real_save, failures = store.save, []

        async def flaky(snapshot, *, expect_version, owner=None):
            if not failures:
                failures.append(1)
                raise OSError("disk full")
            return await real_save(snapshot, expect_version=expect_version, owner=owner)

        monkeypatch.setattr(store, "save", flaky)
        async with ChatSession(make_agent(provider, Gate(released=True)), session_store=store) as session:
            session.send("hi")
            await asyncio.wait_for(session.flush(), 1)
            await records(session, "turn_finished", 1)
            await session.flush()
            snapshot = await store.load(session.id)
            assert failures and snapshot is not None and snapshot.turns == 1


class TestStores:
    @pytest.fixture(params=["memory", "sqlite"])
    def store(self, request, tmp_path):
        return InMemoryChatSessionStore() if request.param == "memory" else SqliteChatSessionStore(tmp_path / "s.db")

    async def test_save_is_a_compare_and_swap(self, store):
        snapshot = SessionSnapshot(session_id="s1", agent="agent", head="r1")
        version = await store.save(snapshot, expect_version=None)
        with pytest.raises(SnapshotConflict):
            await store.save(snapshot, expect_version=None)
        snapshot.head = "r2"
        assert await store.save(snapshot, expect_version=version) == version + 1
        with pytest.raises(SnapshotConflict):
            await store.save(snapshot, expect_version=version)
        loaded = await store.load("s1")
        assert loaded.head == "r2" and loaded.version == version + 1

    def test_a_snapshot_never_stores_a_value_as_its_text(self):
        """A value JSON cannot carry fails the write. Written as its ``str()`` it would read back as another
        value, with nothing to say so."""
        snapshot = SessionSnapshot(session_id="s", agent="agent", resume={"a1": ApprovalResolution(approved=True)})
        with pytest.raises(TypeError):
            snapshot.to_json()
        snapshot.resume = {"a1": ApprovalResolution(approved=True).model_dump(mode="json")}
        assert SessionSnapshot.from_json(snapshot.to_json()).resume == snapshot.resume
        # NaN and Infinity are not JSON either, though Python would write them: refused the same way.
        for not_json in (float("nan"), float("inf")):
            snapshot.resume = {"a1": {"approved": True, "metadata": {"ratio": not_json}}}
            with pytest.raises(ValueError):
                snapshot.to_json()

    async def test_a_save_with_an_owner_needs_the_lease(self, store):
        version = await store.save(SessionSnapshot(session_id="s1", agent="agent"), expect_version=None)
        version = await store.save(await store.load("s1"), expect_version=version, owner="a")  # no lease yet
        await store.acquire_lease("s1", "a")
        with pytest.raises(SnapshotConflict):
            await store.save(await store.load("s1"), expect_version=version, owner="b")
        version = await store.save(await store.load("s1"), expect_version=version, owner="a")
        store.lease_ttl_s = -1.0
        await store.acquire_lease("s1", "a")  # a's lease has lapsed, but nobody took it...
        with pytest.raises(SnapshotConflict):  # ...so it is still a's: b must take it before writing
            await store.save(await store.load("s1"), expect_version=version, owner="b")
        await store.acquire_lease("s1", "b")
        with pytest.raises(SnapshotConflict):
            await store.save(await store.load("s1"), expect_version=version, owner="a")
        assert await store.save(await store.load("s1"), expect_version=version, owner="b") == version + 1

    async def test_a_lease_holds_until_it_expires_or_is_released(self, store):
        await store.save(SessionSnapshot(session_id="s1", agent="agent"), expect_version=None)
        await store.acquire_lease("s1", "a")
        await store.acquire_lease("s1", "a")  # renewing
        with pytest.raises(SessionLeased):
            await store.acquire_lease("s1", "b")
        await store.release_lease("s1", "a")
        await store.acquire_lease("s1", "b")
        store.lease_ttl_s = -1.0  # b's next lease is already expired
        await store.acquire_lease("s1", "b")
        await store.acquire_lease("s1", "a")

    async def test_a_dead_local_holders_lease_is_free(self, store):
        """``kill -9``: the restarted process must not wait out the dead one's lease."""
        import socket
        import subprocess
        import sys

        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait()
        await store.save(SessionSnapshot(session_id="s1", agent="agent"), expect_version=None)
        await store.acquire_lease("s1", f"{socket.gethostname()}:{dead.pid}:old")
        await store.acquire_lease("s1", process_owner())
        with pytest.raises(SessionLeased):  # a live one's (this process, as another owner id) holds
            await store.acquire_lease("s1", f"{socket.gethostname()}:{__import__('os').getpid()}:other")

    def test_the_liveness_probe_never_signals_on_windows(self, monkeypatch):
        """On Windows os.kill(pid, 0) is TerminateProcess: probing a live holder would kill it."""
        from timbal.state import chat_store

        def no_kill(*_):
            raise AssertionError("os.kill on Windows terminates the process")

        monkeypatch.setattr(chat_store.os, "name", "nt")
        monkeypatch.setattr(chat_store.os, "kill", no_kill)
        monkeypatch.setattr(chat_store, "_windows_pid_alive", lambda pid: pid == 42)
        assert chat_store._pid_alive(42) is True and chat_store._pid_alive(7) is False

    async def test_listing_and_deleting(self, store):
        await store.save(SessionSnapshot(session_id="idle", agent="agent"), expect_version=None)
        await store.save(
            SessionSnapshot(session_id="busy", agent="agent", pending=[{"id": "e", "text": "hi"}]), expect_version=None
        )
        await store.save(SessionSnapshot(session_id="gone", agent="agent", closed=True), expect_version=None)
        assert sorted(await store.list_ids()) == ["busy", "idle"]
        assert await store.list_ids(with_work=True) == ["busy"]
        await store.delete("idle")
        assert await store.load("idle") is None


class TestFindingSessionsWithWork:
    """Startup asks the store for the sessions that have work waiting. Idle ones stay in the table until
    they are deleted, and there are many more of them: the answer must not be read out of every row."""

    MESSAGE = {"id": "m1", "text": "hi", "source": "user", "lane": "later", "mode": "followup", "posted_at": 0.0}

    async def _store_with_idle_rows(self, tmp_path, idle=40):
        store = SqliteChatSessionStore(tmp_path / "sessions.db")
        for n in range(idle):
            await store.save(SessionSnapshot(session_id=f"idle-{n:02d}", agent="agent"), expect_version=None)
        await store.save(
            SessionSnapshot(session_id="queued", agent="agent", pending=[self.MESSAGE]), expect_version=None
        )
        await store.save(
            SessionSnapshot(session_id="waking", agent="agent", wakes=[{"id": "w1", "text": "tick"}]),
            expect_version=None,
        )
        return store

    async def test_only_the_rows_with_work_are_looked_at(self, tmp_path, monkeypatch):
        store = await self._store_with_idle_rows(tmp_path)
        read = []
        real = SessionSnapshot.from_json.__func__
        monkeypatch.setattr(
            SessionSnapshot, "from_json", classmethod(lambda cls, raw: read.append(1) or real(cls, raw))
        )
        assert await store.list_ids(with_work=True) == ["queued", "waking"]
        assert read == []  # no snapshot was parsed to tell

        def plan(conn):
            from timbal.state.chat_store import _WITH_WORK_SQL

            return " ".join(row[-1] for row in conn.execute(f"EXPLAIN QUERY PLAN {_WITH_WORK_SQL}"))

        assert "idx_chat_sessions_work" in await store._run(plan)  # nor a row without work visited

    async def test_the_mark_follows_every_save(self, tmp_path):
        store = SqliteChatSessionStore(tmp_path / "sessions.db")
        version = await store.save(SessionSnapshot(session_id="s1", agent="agent"), expect_version=None)
        assert await store.list_ids(with_work=True) == []
        version = await store.save(
            SessionSnapshot(session_id="s1", agent="agent", pending=[self.MESSAGE]), expect_version=version
        )
        assert await store.list_ids(with_work=True) == ["s1"]
        version = await store.save(
            SessionSnapshot(session_id="s1", agent="agent", pending=[self.MESSAGE], closed=True),
            expect_version=version,
            owner="a",
        )
        assert await store.list_ids(with_work=True) == []  # closed: its work goes nowhere
        await store.save(
            SessionSnapshot(session_id="s1", agent="agent", resume={"a1": True}), expect_version=version, owner="a"
        )
        assert await store.list_ids(with_work=True) == ["s1"]
        assert await store.list_ids() == ["s1"]

    async def test_rows_written_before_the_mark_are_still_found(self, tmp_path):
        """A file from before the column: its rows carry no mark until they are written again."""
        import sqlite3

        path = tmp_path / "old.db"
        conn = sqlite3.connect(path)
        conn.execute(
            "CREATE TABLE chat_sessions (session_id TEXT PRIMARY KEY, agent TEXT NOT NULL, snapshot TEXT NOT NULL, "
            "version INTEGER NOT NULL, lease_owner TEXT, lease_until REAL, updated_at INTEGER NOT NULL)"
        )
        rows = [
            SessionSnapshot(session_id="old-idle", agent="agent", version=1),
            SessionSnapshot(session_id="old-queued", agent="agent", version=1, pending=[self.MESSAGE]),
        ]
        for row in rows:
            conn.execute(
                "INSERT INTO chat_sessions (session_id, agent, snapshot, version, updated_at) VALUES (?, ?, ?, 1, 0)",
                (row.session_id, row.agent, row.to_json()),
            )
        conn.execute(
            "INSERT INTO chat_sessions (session_id, agent, snapshot, version, updated_at) "
            "VALUES ('old-broken', 'agent', '{not json', 1, 0)"
        )
        conn.commit()
        conn.close()

        store = SqliteChatSessionStore(path)
        assert await store.list_ids(with_work=True) == ["old-queued"]  # the unreadable one is skipped, not fatal
        idle = await store.load("old-idle")
        idle.wakes = [{"id": "w1", "text": "tick"}]
        await store.save(idle, expect_version=1)
        assert await store.list_ids(with_work=True) == ["old-idle", "old-queued"]
        assert sorted(await store.list_ids()) == ["old-idle", "old-queued"]


class TestWriteAhead:
    async def test_a_message_is_stored_before_its_receipt_is_trusted(self, db):
        provider, store = db
        gate = Gate()
        async with ChatSession(make_agent(provider, gate), session_store=store) as session:
            session.send("start")
            await asyncio.wait_for(gate.entered.wait(), 30)
            queued = session.send("later", mode="followup")
            await session.flush()
            snapshot = await store.load(session.id)
            assert [event["id"] for event in snapshot.pending] == [queued.id]
            assert snapshot.in_flight["run_id"] == session.run_id
            gate.release.set()
            await records(session, "turn_finished", 2)
            await session.flush()
            snapshot = await store.load(session.id)
            assert snapshot.pending == [] and snapshot.in_flight is None
            assert snapshot.head == session.head and snapshot.turns == 2
        assert await store.load(session.id) is None  # closed for good

    async def test_without_a_store_nothing_is_stored(self, db):
        provider, _ = db
        async with ChatSession(make_agent(provider, Gate(released=True))) as session:
            session.send("hi")
            await records(session, "turn_finished", 1)
            assert session.state()["durable"] is False


class TestRestore:
    async def test_a_parked_session_comes_back_with_its_queue_and_can_be_answered(self, db):
        provider, store = db
        charged = []
        session = ChatSession(make_agent(provider, Gate(), charged), session_store=store)
        session.send("pay")
        await records(session, "turn_finished", 1)
        approval = next(e for e in session.log.peek()[0] if isinstance(e, ApprovalEvent))
        waiting = session.send("hello?", mode="followup")
        old_cursor = session.log.cursor_end
        await session.close(keep=True)

        restored = await ChatSession.from_store(make_agent(provider, Gate(), charged), store, session.id)
        state = restored.state()
        assert state["parked"] and [p["event_id"] for p in state["pending"]] == [waiting.id]
        assert state["pending_approvals"][0]["approval_id"] == approval.approval_id
        assert restored.log.cursor_end > old_cursor
        restored.resume({approval.approval_id: True})
        await records(restored, "turn_finished", 2, after=restored.log.forgotten_through)
        assert charged == [1]
        assert [o.output.collect_text() for o in root_outputs(restored)] == ["done", "echo: hello?"]
        await restored.close()

    async def test_a_turn_that_died_mid_tool_gets_a_read_only_recovery_turn(self, db):
        provider, store = db
        gate = Gate()
        session = ChatSession(make_agent(provider, gate), session_store=store)
        session.send("start")
        await asyncio.wait_for(gate.entered.wait(), 30)
        dead_run = session.run_id
        await crash(session)
        provider.crashed = False

        again = Gate(released=True)
        restored = await ChatSession.from_store(make_agent(provider, again), store, session.id)
        (started,) = await records(restored, "turn_started", 1, after=restored.log.forgotten_through)
        (finished,) = await records(restored, "turn_finished", 1, after=restored.log.forgotten_through)
        assert started["trigger"] == {"kind": "recovery", "dead_run_id": dead_run, "cut": "crash"}
        assert started["parent_id"] == dead_run
        assert root_outputs(restored)[0].output.collect_text() == "recovered"
        prompt = root_outputs(restored)[0].input["prompt"]
        text = Message.validate(prompt).collect_text()
        assert "Started but not confirmed" in text and "work(" in text
        assert "can only read" in text  # read-only: it cannot redo a write on its own
        assert again.calls == 0  # nothing re-run behind the model's back
        await restored.close()

    async def test_the_notice_tells_completed_calls_from_unconfirmed_ones(self, db):
        provider, store = db
        gate = Gate()

        def agent():
            return Agent(
                name="agent",
                model=TestModel(handler=two_step),
                tools=[Tool(name="lookup", handler=lambda: "2 unread"), Tool(name="work", handler=gate.work)],
                tracing_provider=provider,
            )

        session = ChatSession(agent(), session_store=store)
        session.send("go")
        await asyncio.wait_for(gate.entered.wait(), 30)
        await crash(session)
        provider.crashed = False

        restored = await ChatSession.from_store(agent(), store, session.id)
        notice = restored._wakes[0].text
        assert "Completed before the restart: lookup({}) → 2 unread." in notice
        assert "Started but not confirmed (each may or may not have run): work({})." in notice
        await restored.close()

    async def test_the_notice_pairs_dispatched_calls_with_their_own_spans(self, db):
        """A dispatch to a tool that does not exist leaves a ``call_tool`` span of its own (the error). It is
        that call's span, not the span of an earlier dispatch that ran as its tool: each request is paired
        by the tool call id its span recorded, so the earlier call is listed once, with its own outcome."""
        provider, store = db
        gate = Gate()
        steps = ["lookup", "send_rocket", "work"]

        def through_call_tool(messages):
            if "cut off by a server restart" in last_user_text(messages):
                return "recovered"
            name = steps[len([m for m in messages if m.role == "tool"])]
            return Message(
                role="assistant",
                content=[ToolUseContent(id=f"call-{name}", name="call_tool", input={"name": name, "arguments": {}})],
                stop_reason="tool_use",
            )

        def agent():
            return Agent(
                name="agent",
                model=TestModel(handler=through_call_tool),
                tools=[
                    Tool(name="lookup", handler=lambda: "2 unread", defer_loading=True),
                    Tool(name="work", handler=gate.work, defer_loading=True),
                ],
                tool_search="dispatch",
                tracing_provider=provider,
            )

        session = ChatSession(agent(), session_store=store)
        session.send("go")
        await asyncio.wait_for(gate.entered.wait(), 30)
        await crash(session)
        provider.crashed = False

        restored = await ChatSession.from_store(agent(), store, session.id)
        notice = restored._wakes[0].text
        completed, unconfirmed = notice.split("Started but not confirmed")
        asked = {name: f"call_tool({{'name': '{name}'" for name in steps}  # each request, as the notice labels it
        # Each call once, with its own outcome (before: the lookup got the failed dispatch's span, and its own
        # span was then listed again as a call nobody had asked for).
        assert notice.count("lookup") == 1 and f"{asked['lookup']}, 'arguments': {{}}}}) → 2 unread" in completed
        assert notice.count(asked["send_rocket"]) == 1 and "No tool named 'send_rocket'" in completed  # ran, failed
        assert notice.count("work") == 1 and asked["work"] in unconfirmed
        await restored.close()

    async def test_the_notice_names_a_dispatched_call_once(self, db):
        """``tool_search="dispatch"``: the model asks for ``call_tool(name=…)`` and the tool runs under its own
        span. The notice pairs the two, so one call is not listed twice (once completed, once unconfirmed)."""
        provider, store = db
        gate = Gate()

        def through_call_tool(messages):
            if "cut off by a server restart" in last_user_text(messages):
                return "recovered"
            results = [m for m in messages if m.role == "tool"]
            name = "lookup" if not results else "work"
            return Message(
                role="assistant",
                content=[ToolUseContent(id=f"call-{name}", name="call_tool", input={"name": name, "arguments": {}})],
                stop_reason="tool_use",
            )

        def agent():
            return Agent(
                name="agent",
                model=TestModel(handler=through_call_tool),
                tools=[
                    Tool(name="lookup", handler=lambda: "2 unread", defer_loading=True),
                    Tool(name="work", handler=gate.work, defer_loading=True),
                ],
                tool_search="dispatch",
                tracing_provider=provider,
            )

        session = ChatSession(agent(), session_store=store)
        session.send("go")
        await asyncio.wait_for(gate.entered.wait(), 30)
        dispatched = (await store.load(session.id)).in_flight["dispatched"]
        assert [call["name"] for call in dispatched] == ["call_tool", "call_tool"]  # as the model asked for them
        await crash(session)
        provider.crashed = False

        restored = await ChatSession.from_store(agent(), store, session.id)
        notice = restored._wakes[0].text
        assert notice.count("lookup") == 1 and "→ 2 unread" in notice
        assert notice.count("work") == 1
        completed, unconfirmed = notice.split("Started but not confirmed")
        assert "lookup" in completed and "work" in unconfirmed
        await restored.close()

    async def test_a_turn_that_asked_for_tools_and_left_no_trace_is_not_run_again(self, db):
        """A provider that saves a run at its end (or in the background) holds nothing of a turn cut mid-tool.
        The session stored that the model had asked for tools before any ran: the input is not queued again."""
        provider, store = db
        provider.crashed = True  # no trace of the dead turn ever lands
        gate = Gate()
        session = ChatSession(make_agent(provider, gate), session_store=store)
        session.send("start")
        await asyncio.wait_for(gate.entered.wait(), 30)
        dispatched = (await store.load(session.id)).in_flight["dispatched"]
        assert [call["name"] for call in dispatched] == ["work"]  # stored before the tool ran
        await crash(session)
        provider.crashed = False

        again = Gate(released=True)
        restored = await ChatSession.from_store(make_agent(provider, again), store, session.id)
        assert restored.state()["pending"] == []
        notice = restored._wakes[0].text
        assert 'It was answering: "start".' in notice and "not recorded before the restart" in notice
        assert "Started but not confirmed (each may or may not have run): work(" in notice
        (started,) = await records(restored, "turn_started", 1, after=restored.log.forgotten_through)
        assert started["trigger"]["kind"] == "recovery"
        (finished,) = await records(restored, "turn_finished", 1, after=restored.log.forgotten_through)
        assert finished["status"]["code"] == "success"
        assert root_outputs(restored)[0].output.collect_text() == "recovered"
        assert again.calls == 0  # the tool was not run a second time
        await restored.close()

    async def test_a_turn_that_died_before_asking_for_a_tool_is_queued_again(self, db):
        provider, store = db
        provider.crashed = True  # no trace of the dead turn ever lands
        thinking = asyncio.Event()

        async def before_the_model() -> None:
            thinking.set()
            await asyncio.sleep(30)  # the process dies here: started, the model not called yet

        agent = Agent(
            name="agent", model=TestModel(handler=handler), pre_hook=before_the_model, tracing_provider=provider
        )
        session = ChatSession(agent, session_store=store)
        sent = session.send("hello")
        await asyncio.wait_for(thinking.wait(), 3)
        await crash(session)
        provider.crashed = False
        assert "dispatched" not in (await store.load(session.id)).in_flight

        restored = await ChatSession.from_store(make_agent(provider, Gate(released=True)), store, session.id)
        assert [p["event_id"] for p in restored.state()["pending"]] == [sent.id]
        (finished,) = await records(restored, "turn_finished", 1, after=restored.log.forgotten_through)
        assert finished["status"]["code"] == "success"
        assert root_outputs(restored)[0].output.collect_text() == "echo: hello"
        await restored.close()

    async def test_a_turn_cut_by_shutdown_is_recovered_too(self, db):
        provider, store = db
        gate = Gate()
        session = ChatSession(make_agent(provider, gate), session_store=store)
        session.send("start")
        await asyncio.wait_for(gate.entered.wait(), 30)
        await session.close(keep=True)

        restored = await ChatSession.from_store(make_agent(provider, Gate(released=True)), store, session.id)
        (started,) = await records(restored, "turn_started", 1, after=restored.log.forgotten_through)
        assert started["trigger"]["kind"] == "recovery" and started["trigger"]["cut"] == "shutdown"
        await records(restored, "turn_finished", 1, after=restored.log.forgotten_through)
        await restored.close()

    async def test_a_turn_that_had_ended_when_the_session_closed_is_not_a_cut_one(self, db):
        """A shutdown that comes between a turn ending and the session recording it must not store the turn
        as cut: here it ended waiting for an approval, and a recovery turn would be chained onto it."""
        provider, store = db
        charge = Charge(released=True)
        session = ChatSession(gated_agent(provider, charge), session_store=store)
        real_pump, closing = session._pump, []

        async def pump(*args, **kwargs):
            output = await real_pump(*args, **kwargs)
            # Runs once the turn's task is over, before the worker has settled the turn.
            closing.append(asyncio.get_running_loop().create_task(session.close(keep=True)))
            return output

        session._pump = pump
        session.send("pay")
        await until(lambda: closing)
        await closing[0]
        stored = await store.load(session.id)
        assert stored.in_flight is None and stored.parked and stored.turns == 1

        restored = await ChatSession.from_store(gated_agent(provider, charge), store, session.id)
        assert restored.parked and restored._wakes == []  # the approval still waits; nothing to recover
        (approval,) = restored.state()["pending_approvals"]
        restored.resume({approval["approval_id"]: True})
        (finished,) = await records(restored, "turn_finished", 1, after=after_restore(restored))
        assert finished["status"]["code"] == "success" and charge.gate.calls == 1
        await restored.close()

    async def test_a_second_cancellation_of_the_worker_does_not_cut_the_turns_cleanup(self, db):
        """Stopping waits for the turn to be over, also when something cancels the worker again meanwhile (a
        second close, the loop shutting down): the turn's own cleanup is not interrupted, and what is read
        afterwards is the outcome of a task that has ended."""
        provider, store = db
        entered, cleaned = asyncio.Event(), []

        async def work() -> str:
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await asyncio.sleep(0.2)  # letting go of what it held takes a moment
                cleaned.append("done")
                raise
            return "worked"

        agent = Agent(
            name="agent",
            model=TestModel(handler=handler),
            tools=[Tool(name="work", handler=work)],
            tracing_provider=provider,
        )
        session = ChatSession(agent, session_store=store)
        session.send("start")
        await asyncio.wait_for(entered.wait(), 3)
        worker = session._worker
        closing = asyncio.get_running_loop().create_task(session.close(keep=True))
        await asyncio.sleep(0.05)
        worker.cancel()  # again, while the turn is still cleaning up
        await closing
        assert cleaned == ["done"] and not session.busy
        assert (await store.load(session.id)).in_flight["cut"] == "shutdown"  # cut by the close: recovered later

    async def test_a_run_that_ended_by_itself_is_adopted_even_if_marked_cut(self, db):
        provider, store = db
        charge = Charge(released=True)
        session, approval = await parked(provider, store, charge)
        await session.close(keep=True)
        snapshot = await store.load(session.id)
        snapshot.in_flight = {  # as a shutdown that cancelled the worker right after the run ended left it
            "run_id": snapshot.head,
            "parent_id": None,
            "trigger": {"kind": "user_message"},
            "events": [],
            "wake": None,
            "cut": "shutdown",
        }
        await store.save(snapshot, expect_version=snapshot.version)

        restored = await ChatSession.from_store(gated_agent(provider, charge), store, session.id)
        assert restored.parked and restored._wakes == []
        assert restored.state()["pending_approvals"][0]["approval_id"] == approval.approval_id
        await restored.close()

    @pytest.mark.parametrize("on_dead_turn", ["notify", "drop"])
    async def test_on_dead_turn_can_skip_the_recovery_turn(self, db, on_dead_turn):
        provider, store = db
        gate = Gate()
        session = ChatSession(make_agent(provider, gate), session_store=store, on_dead_turn=on_dead_turn)
        session.send("start")
        await asyncio.wait_for(gate.entered.wait(), 30)
        await crash(session)
        provider.crashed = False

        restored = await ChatSession.from_store(make_agent(provider, Gate(released=True)), store, session.id)
        await asyncio.sleep(0.1)
        assert restored.turns == 1 and not restored.busy  # no turn of its own
        notices = [e for e in restored.store.pending_events() if e.key == "recovery"]
        assert len(notices) == (1 if on_dead_turn == "notify" else 0)
        await restored.close()

    async def test_a_turn_that_finished_before_the_snapshot_is_adopted(self, db):
        provider, store = db
        session = ChatSession(make_agent(provider, Gate(released=True)), session_store=store)
        session.send("hi")
        await records(session, "turn_finished", 1)
        await session.flush()
        snapshot = await store.load(session.id)
        snapshot.in_flight = {"run_id": session.head, "parent_id": None, "trigger": {}, "events": []}
        await store.save(snapshot, expect_version=snapshot.version)
        session._lost = True
        await session.close()

        restored = await ChatSession.from_store(make_agent(provider, Gate()), store, session.id)
        await asyncio.sleep(0.05)
        assert restored.head == session.head and not restored.busy and restored.last_status["code"] == "success"
        await restored.close()

    async def test_another_live_process_keeps_its_session(self, db):
        provider, store = db
        session = ChatSession(make_agent(provider, Gate()), session_store=store)
        session.send("hi")
        await records(session, "turn_finished", 1)
        await session.flush()
        await store.release_lease(session.id, process_owner())
        await store.acquire_lease(session.id, "another-process")
        with pytest.raises(SessionLeased):
            await ChatSession.from_store(make_agent(provider, Gate()), store, session.id)
        session._lost = True
        await session.close()

    async def test_unknown_and_foreign_sessions(self, db):
        provider, store = db
        with pytest.raises(LookupError):
            await ChatSession.from_store(make_agent(provider, Gate()), store, "nope")
        session = ChatSession(make_agent(provider, Gate()), session_store=store)
        session.send("hi")
        await records(session, "turn_finished", 1)
        await session.close(keep=True)
        other = Agent(name="other", model=TestModel(responses=["x"]), tracing_provider=provider)
        with pytest.raises(ValueError, match="belongs to agent"):
            await ChatSession.from_store(other, store, session.id)

    async def test_an_old_cursor_gets_a_gap_then_the_new_events(self, db):
        provider, store = db
        session = ChatSession(make_agent(provider, Gate(released=True)), session_store=store)
        session.send("hi")
        await records(session, "turn_finished", 1)
        old = session.log.cursor_end
        await session.close(keep=True)
        restored = await ChatSession.from_store(make_agent(provider, Gate(released=True)), store, session.id)
        events, _, gapped = restored.log.read(old)
        assert gapped and events == []
        restored.send("again")
        await records(restored, "turn_finished", 1, after=restored.log.forgotten_through)
        assert root_outputs(restored)[0].output.collect_text() == "echo: again"
        await restored.close()


class FlakyReads(CrashableSqlite):
    """The next ``fail_reads`` reads of a run fail; the next ``slow_reads`` take ``read_delay`` seconds."""

    fail_reads = 0
    slow_reads = 0
    read_delay = 0.0

    @classmethod
    async def _flaky(cls):
        if cls.slow_reads > 0:
            cls.slow_reads -= 1
            await asyncio.sleep(cls.read_delay)
        if cls.fail_reads > 0:
            cls.fail_reads -= 1
            raise OSError("store unreachable")

    @classmethod
    async def get(cls, run_context):
        await cls._flaky()
        return await super().get(run_context)

    @classmethod
    async def get_strict(cls, run_context):
        await cls._flaky()
        return await super().get_strict(run_context)


@pytest.fixture
def flaky_db(tmp_path):
    provider = FlakyReads.configured(
        _path=tmp_path / "timbal.db", crashed=False, fail_reads=0, slow_reads=0, read_delay=0.0
    )
    return provider, SqliteChatSessionStore(tmp_path / "timbal.db")


class TestAStoppedTurnThatDied:
    """The user stops a turn (a message sent as ``interrupt``) and the process dies before the stop took effect
    in what was stored. The turn had run no tool call, so a restart would queue its message again: behind the
    stop, where it reads as the user's latest words, and the gated tool it asked for is asked for again (and,
    approved, runs). Live, the stop cut that turn and its request was never acted on again (CI, Windows: the
    `approval_abandoned_as_it_parks` sweep under a trace outage, its `dispatched` write never landing)."""

    async def _stopped_before_its_tool(self, db, charge):
        """``pay``'s model asks for the charge; the turn is held before it records that call (the write that
        would, never landed), and the stop is sent and stored (acknowledged), the cut not made yet."""
        provider, store = db
        session = ChatSession(gated_agent(provider, charge), session_store=store)
        about_to_dispatch = asyncio.Event()

        async def held(_in_flight, _agent_path, _calls):
            about_to_dispatch.set()
            await asyncio.Event().wait()

        session._note_dispatch = held
        session.send("pay")
        await asyncio.wait_for(about_to_dispatch.wait(), 30)
        stop = session.send("never mind", mode="interrupt", defer_cut=True)
        await session.flush()
        stored = await store.load(session.id)
        assert stored.in_flight is not None and not stored.in_flight.get("dispatched")
        assert [event["text"] for event in stored.pending] == ["never mind"]
        return provider, store, session, stop

    async def _restored(self, provider, store, session, charge):
        if not session.closed:
            await crash(session)
        provider.crashed = False
        restored = await ChatSession.from_store(gated_agent(provider, charge), store, session.id)
        await records(restored, "turn_finished", 1, after=after_restore(restored))
        await until(lambda: not restored.busy and not restored.pending_messages())
        return restored

    async def _the_stop_holds(self, restored, charge):
        assert not restored.parked, "the stopped request asked for the charge again"
        assert charge.gate.calls == 0
        outputs = root_outputs(restored)
        assert [output.output.collect_text() for output in outputs] == ["echo: never mind"]
        # Not lost: the next turn read it, named as what the user stopped.
        run = await restored._load_run(outputs[-1].run_id)
        texts = [str(message.get("content")) for message in run.root_span().memory]
        notice = next(text for text in texts if "stopped that turn" in text)
        assert '"pay"' in notice and "cut off by a server restart" in notice

    async def test_only_the_stop_message_was_stored(self, db):
        charge = Charge(released=True)
        provider, store, session, _ = await self._stopped_before_its_tool(db, charge)
        assert not (await store.load(session.id)).in_flight.get("stopped")
        restored = await self._restored(provider, store, session, charge)
        await self._the_stop_holds(restored, charge)
        await restored.close()

    async def test_the_stop_was_stored_with_the_turn(self, db):
        charge = Charge(released=True)
        provider, store, session, stop = await self._stopped_before_its_tool(db, charge)
        # The cut reached the turn (_interrupt_now marks it stopped and stores that), and the process died
        # before the turn was cancelled and settled.
        session._in_flight["stopped"] = True
        session._persist()
        await session.flush()
        assert (await store.load(session.id)).in_flight["stopped"] is True
        restored = await self._restored(provider, store, session, charge)
        await self._the_stop_holds(restored, charge)
        await restored.close()

    async def test_its_messages_listed_as_pending_too_are_not_queued_again(self, db):
        """A turn cut before it started hands its messages back, so a snapshot can list them both with the turn
        and as pending. Dropped as stopped, they are not restored from the pending list either."""
        charge = Charge(released=True)
        provider, store, session, _ = await self._stopped_before_its_tool(db, charge)
        await crash(session)
        stored = await store.load(session.id)
        stored.pending = [*stored.in_flight["events"], *stored.pending]
        await store.save(stored, expect_version=stored.version, owner=process_owner())
        restored = await self._restored(provider, store, session, charge)
        await self._the_stop_holds(restored, charge)
        await restored.close()

    async def test_a_resume_stopped_before_its_tool_is_not_answered_again(self, db):
        """The user approves, then stops the turn acting on it before it records the charge; only the stop message
        is stored. The answers go back with the question, which the stop abandons: approved or not, the charge
        does not run."""
        provider, store = db
        charge = Charge(released=True)
        session, approval = await parked(provider, store, charge)
        about_to_dispatch = asyncio.Event()

        async def held(_in_flight, _agent_path, _calls):
            about_to_dispatch.set()
            await asyncio.Event().wait()

        session._note_dispatch = held
        session.resume({approval.approval_id: True})
        await asyncio.wait_for(about_to_dispatch.wait(), 30)
        session.send("never mind", mode="interrupt", defer_cut=True)
        await session.flush()
        stored = await store.load(session.id)
        assert stored.in_flight["resume"] and not stored.in_flight.get("stopped")
        await crash(session)
        provider.crashed = False
        restored = await ChatSession.from_store(gated_agent(provider, charge), store, session.id)
        await records(restored, "turn_finished", 2, after=after_restore(restored))
        await until(lambda: not restored.busy and not restored.pending_messages())
        assert not restored.parked and charge.gate.calls == 0
        assert root_outputs(restored)[-1].output.collect_text() == "echo: never mind"
        await restored.close()

    async def test_a_wake_turn_stopped_before_its_tool_does_not_run_again(self, db):
        """A turn the application started (a wake) is stopped by the user's message; the process dies before the
        turn records its call. The wake is not run again after the restart (it would redo what was stopped):
        it is named to the next turn, as the user's stopped messages are."""
        provider, store = db

        def pays(messages):
            if messages[-1].role == "tool":
                return "paid"
            text = last_user_text(messages)
            return tool_call("charge", "call-pay") if "pay" in text else f"echo: {text}"

        charge = Charge(released=True)
        session = ChatSession(gated_agent(provider, charge, model_handler=pays), session_store=store)
        about_to_dispatch = asyncio.Event()

        async def held(_in_flight, _agent_path, _calls):
            about_to_dispatch.set()
            await asyncio.Event().wait()

        session._note_dispatch = held
        session.wake("pay the invoice", key="pay", tool_policy=None)
        await asyncio.wait_for(about_to_dispatch.wait(), 30)
        session.send("never mind", mode="interrupt", defer_cut=True)
        await session.flush()
        stored = await store.load(session.id)
        assert stored.in_flight["wake"] is not None and not stored.in_flight.get("dispatched")
        await crash(session)
        provider.crashed = False

        restored = await ChatSession.from_store(gated_agent(provider, charge, model_handler=pays), store, session.id)
        assert not restored._wakes, "the stopped wake was queued again"
        await records(restored, "turn_finished", 1, after=after_restore(restored))
        await until(lambda: not restored.busy and not restored.pending_messages())
        assert not restored.parked and charge.gate.calls == 0
        outputs = root_outputs(restored)
        assert [output.output.collect_text() for output in outputs] == ["echo: never mind"]
        run = await restored._load_run(outputs[-1].run_id)
        texts = [str(message.get("content")) for message in run.root_span().memory]
        notice = next(text for text in texts if "stopped that turn" in text)
        assert '"pay the invoice"' in notice
        await restored.close()


class TestAParkedRunThatIsGone:
    """The snapshot says the session waits for an answer, and the run that asked is not in the trace store
    (traces kept apart from the sessions and rotated, a platform run that is a 404). Nothing can answer it:
    the session is not parked, and goes on."""

    async def _restored_without_its_traces(self, db, tmp_path, *, then):
        provider, store = db
        session, approval = await parked(provider, store, Charge(released=True))
        # The process stores what it is sent and dies before its worker takes any of it. Left running, the
        # worker would take a queued answer at once and, on a slow machine, have its charge started before
        # the close: a different restart (a recovery turn), every time the test was slow.
        await worker_stopped(session)
        then(session, approval)
        await session.flush()
        await session.close(keep=True)
        rotated = CrashableSqlite.configured(_path=tmp_path / "rotated.db", crashed=False)
        charge = Charge(released=True)
        restored = await ChatSession.from_store(gated_agent(rotated, charge), store, session.id)
        return restored, charge, approval

    async def test_it_is_not_parked_and_takes_the_message_that_waited(self, db, tmp_path):
        restored, charge, approval = await self._restored_without_its_traces(
            db, tmp_path, then=lambda session, _approval: session.send("hello")
        )
        assert not restored.parked and restored.state()["parked"] is False
        (finished,) = await records(restored, "turn_finished", 1, after=after_restore(restored))
        assert finished["status"]["code"] == "success"
        assert root_outputs(restored)[-1].output.collect_text() == "echo: hello"
        assert not restored._worker.done() or restored._worker.exception() is None  # it did not die on the way
        with pytest.raises(RuntimeError, match="waiting for an answer"):
            restored.resume({approval.approval_id: True})
        assert charge.gate.calls == 0
        await restored.close()

    async def test_an_answer_that_was_queued_for_it_is_dropped(self, db, tmp_path):
        restored, charge, approval = await self._restored_without_its_traces(
            db, tmp_path, then=lambda session, approval: session.resume({approval.approval_id: True})
        )
        (dropped,) = await records(restored, "resume_dropped", 1, after=after_restore(restored))
        assert dropped["run_id"] is None and not restored.parked
        restored.send("hello")
        (finished,) = await records(restored, "turn_finished", 1, after=after_restore(restored))
        assert finished["status"]["code"] == "success" and charge.gate.calls == 0
        await restored.close()


class TestRestoreSafety:
    async def test_a_failed_write_is_reported_by_flush(self, db, monkeypatch):
        provider, store = db

        async def down(*_args, **_kwargs):
            raise OSError("disk full")

        agent = make_agent(provider, Gate(released=True))
        async with ChatSession(agent, session_store=store, collect_window=30) as session:
            monkeypatch.setattr(store, "save", down)
            event = session.send("later", mode="collect")  # held by the collect window: no turn takes it yet
            with pytest.raises(SessionNotStored, match="disk full"):
                await session.flush()
            assert session.withdraw(event.id) and event.state == "withdrawn"
            assert session.pending_messages() == []
            monkeypatch.undo()

    async def test_a_taken_message_is_stored_before_its_turn_starts(self, db):
        """Taken from the queue, a message is in no turn yet: a restart then must not lose it."""
        provider, store = db
        session = ChatSession(make_agent(provider, Gate(released=True)), session_store=store)
        hold = asyncio.Event()
        real_pump = session._pump

        async def slow_start(*args, **kwargs):
            await hold.wait()  # the turn's first event is late
            return await real_pump(*args, **kwargs)

        session._pump = slow_start
        first = session.send("first")
        await until(lambda: session._in_flight is not None)
        session.send("second", mode="followup")
        await session.flush()
        snapshot = await store.load(session.id)
        assert [e["id"] for e in snapshot.in_flight["events"]] == [first.id]
        await crash(session)
        hold.set()
        provider.crashed = False

        restored = await ChatSession.from_store(make_agent(provider, Gate(released=True)), store, session.id)
        await records(restored, "turn_finished", 2, after=after_restore(restored))
        assert [o.output.collect_text() for o in root_outputs(restored)] == ["echo: first", "echo: second"]
        await restored.close()

    async def test_an_unreadable_dead_turn_fails_the_restore_instead_of_running_it_again(self, flaky_db):
        provider, store = flaky_db
        gate = Gate()
        session = ChatSession(make_agent(provider, gate), session_store=store)
        session.send("start")
        await asyncio.wait_for(gate.entered.wait(), 30)
        await crash(session)
        provider.crashed = False
        stored = await store.load(session.id)

        provider.fail_reads = 1
        again = Gate(released=True)
        with pytest.raises(OSError, match="unreachable"):
            await ChatSession.from_store(make_agent(provider, again), store, session.id)
        assert (await store.load(session.id)).in_flight == stored.in_flight  # kept for the retry
        # The failed attempt let go of the lease: the retry gets it, and recovers instead of re-running.
        restored = await ChatSession.from_store(make_agent(provider, again), store, session.id)
        (started,) = await records(restored, "turn_started", 1, after=after_restore(restored))
        assert started["trigger"]["kind"] == "recovery"
        assert again.calls == 0  # the tool was not run a second time
        await restored.close()

    async def test_no_turn_starts_before_the_dead_turn_is_settled(self, flaky_db, monkeypatch):
        provider, store = flaky_db
        real_recover = ChatSession._recover

        async def work_arrives_meanwhile(self, *args):
            # A background task finishing, or an event, while the dead turn's trace is still being read.
            self._work_arrived()
            return await real_recover(self, *args)

        monkeypatch.setattr(ChatSession, "_recover", work_arrives_meanwhile)
        gate = Gate()
        session = ChatSession(make_agent(provider, gate), session_store=store)
        session.send("start")
        await asyncio.wait_for(gate.entered.wait(), 30)
        session.send("also this", mode="followup")
        await crash(session)
        provider.crashed = False

        # The dead turn's trace (the first read) is slow; anything read after it is not. The queued message must
        # still wait for the recovery turn.
        provider.slow_reads, provider.read_delay = 1, 0.3
        restored = await ChatSession.from_store(make_agent(provider, Gate(released=True)), store, session.id)
        started = await records(restored, "turn_started", 2, after=after_restore(restored))
        await records(restored, "turn_finished", 2, after=after_restore(restored))
        assert [s["trigger"]["kind"] for s in started] == ["recovery", "user_message"]
        await restored.close()

    async def test_a_parked_turn_under_a_callable_policy_keeps_its_restrictions(self, db):
        """The callable allowed charge only: read-only lookup must stay out after a restart."""
        provider, store = db

        def pays(messages):
            if messages[-1].role == "tool":
                return "paid"
            text = last_user_text(messages)
            return tool_call("charge", "call-pay") if "pay" in text else f"echo: {text}"

        def lookup() -> str:
            return "found"

        tools = (Tool(name="lookup", handler=lookup, read_only=True),)
        session = ChatSession(
            gated_agent(provider, Charge(), extra_tools=tools, model_handler=pays), session_store=store
        )
        session.wake("pay the invoice", key="pay", tool_policy=lambda tool: tool.name == "charge")
        await records(session, "turn_finished", 1)
        assert session.parked
        await session.close(keep=True)
        assert (await store.load(session.id)).last_options["tool_policy"] == "custom"

        restored = await ChatSession.from_store(
            gated_agent(provider, Charge(released=True), extra_tools=tools, model_handler=pays), store, session.id
        )
        policy = restored._last_options.tool_policy
        assert policy != "read_only" and policy.paths == {"agent.charge"}
        await restored.close()

    async def test_the_snapshot_is_read_under_the_lease_and_a_failed_restore_lets_it_go(self, db):
        provider, store = db
        session = ChatSession(make_agent(provider, Gate(released=True)), session_store=store)
        session.send("hi")
        await records(session, "turn_finished", 1)
        await session.close(keep=True)
        calls = []
        real_acquire, real_load = store.acquire_lease, store.load

        async def acquire(session_id, owner):
            calls.append("lease")
            return await real_acquire(session_id, owner)

        async def load(session_id):
            calls.append("load")
            return await real_load(session_id)

        store.acquire_lease, store.load = acquire, load
        other = Agent(name="other", model=TestModel(responses=["x"]), tracing_provider=provider)
        with pytest.raises(ValueError):
            await ChatSession.from_store(other, store, session.id)
        assert calls == ["lease", "load"]
        await store.acquire_lease(session.id, "elsewhere:1:x")  # the failed attempt holds nothing


class TestAcknowledgement:
    """What a caller may tell its client after flush() raised: stored, taken back, or neither."""

    async def test_stored_since_tells_a_stored_change_from_one_the_store_never_got(self, db, monkeypatch):
        provider, store = db

        async def down(*_args, **_kwargs):
            raise OSError("disk full")

        agent = make_agent(provider, Gate(released=True))
        async with ChatSession(agent, session_store=store, collect_window=30) as session:
            first = session.send("one", mode="collect")
            stored = session.mark()
            await session.flush()
            monkeypatch.setattr(store, "save", down)
            second = session.send("two", mode="collect")
            lost = session.mark()
            with pytest.raises(SessionNotStored):
                await session.flush()
            # The first message is in the store; a later change is what failed. Only the second is not.
            assert session.stored_since(stored) and not session.stored_since(lost)
            assert [e["id"] for e in (await store.load(session.id)).pending] == [first.id]
            assert session.withdraw(second.id)
            monkeypatch.undo()
            await session.flush()
            assert [e["id"] for e in (await store.load(session.id)).pending] == [first.id]

    @pytest.mark.parametrize("mode", ["interrupt", "now"])
    async def test_a_deferred_cut_leaves_the_turn_alone_until_asked(self, db, mode):
        provider, store = db
        gate = Gate()
        async with ChatSession(make_agent(provider, gate), session_store=store) as session:
            session.send("start")
            await asyncio.wait_for(gate.entered.wait(), 30)
            running, context = session.run_id, session._turn_context
            event = session.send("stop", mode=mode, defer_cut=True)
            await asyncio.sleep(0.05)
            assert session.busy and session.run_id == running and not context._cut.is_set()
            assert session.withdraw(event.id)  # the caller could not store it: nothing was cut
            assert session.cut_for(event, mode) is False
            assert session.busy and session.run_id == running and not context._cut.is_set()

            again = session.send("stop", mode=mode, defer_cut=True)
            assert session.cut_for(again, mode) is True  # stored: now it cuts
            if mode == "now":
                assert context._cut.is_set()
            gate.release.set()
            if mode == "interrupt":  # the turn is cancelled, the message gets the next one
                finished = await records(session, "turn_finished", 2)
                assert finished[0]["status"]["code"] == "cancelled"
            else:  # 'now' joins the running turn
                await records(session, "turn_finished", 1)
            assert root_outputs(session)[-1].output.collect_text() == "echo: stop"

    async def test_a_cut_is_not_applied_to_the_turn_that_took_the_message(self, db):
        provider, store = db
        gate = Gate()
        async with ChatSession(make_agent(provider, gate), session_store=store) as session:
            event = session.send("start", mode="interrupt", defer_cut=True)
            await asyncio.wait_for(gate.entered.wait(), 30)  # its own turn runs by the time it is stored
            assert session.cut_for(event, "interrupt") is False
            assert session.busy
            gate.release.set()
            (finished,) = await records(session, "turn_finished", 1)
            assert finished["status"]["code"] == "success"

    async def test_a_withdrawn_wake_puts_back_the_one_it_replaced(self, db):
        provider, store = db
        async with ChatSession(make_agent(provider, Gate(released=True)), session_store=store) as session:
            first = session.wake("first", key="k")
            second = session.wake("second", key="k")
            assert [wake.id for wake in session._wakes] == [second]
            assert session.withdraw(second)
            assert [wake.id for wake in session._wakes] == [first]
            # One the sender of which never heard back, replaced meanwhile: gone for good once withdrawn.
            third = session.wake("third", key="k")
            assert session.withdraw(first) and session.withdraw(third)
            assert session._wakes == []

    async def test_a_stored_interrupt_message_abandons_the_question_after_a_restart(self, db):
        """The message is stored before it cuts anything: a restart in between must still apply it."""
        provider, store = db
        charge = Charge(released=True)
        session, _ = await parked(provider, store, charge)
        session.send("never mind", mode="interrupt", defer_cut=True)  # stored; the process dies before the cut
        await session.flush()
        await session.close(keep=True)

        restored = await ChatSession.from_store(gated_agent(provider, charge), store, session.id)
        finished = await records(restored, "turn_finished", 2, after=after_restore(restored))
        assert finished[0]["status"]["code"] == "cancelled"
        assert root_outputs(restored)[-1].output.collect_text() == "echo: never mind"
        assert charge.gate.calls == 0
        await restored.close()

    async def test_work_arriving_during_a_restore_starts_the_worker_afterwards(self, db, monkeypatch):
        provider, store = db
        gate = Gate()
        session = ChatSession(make_agent(provider, gate), session_store=store)
        session.send("start")
        await asyncio.wait_for(gate.entered.wait(), 30)
        await crash(session)
        provider.crashed = False
        real_recover = ChatSession._recover

        async def a_notice_arrives(self, *args):
            result = await real_recover(self, *args)
            self._wakes.clear()  # nothing left but what arrived while the worker was held
            self._work_arrived()
            return result

        monkeypatch.setattr(ChatSession, "_recover", a_notice_arrives)
        restored = await ChatSession.from_store(make_agent(provider, Gate(released=True)), store, session.id)
        assert restored._worker is not None
        await restored.close()


class TestWriteAheadOfTurns:
    """A turn runs only once the store holds that it took its input, and its tools only once the store holds
    that the model asked for some."""

    async def test_a_turn_waits_for_the_store_to_hold_its_start(self, db):
        provider, store = db
        gate = Gate()
        session = ChatSession(make_agent(provider, gate), session_store=store)
        go = held_before_its_turn(session)
        sent = session.send("start")
        await session.flush()  # acknowledged: stored as pending
        hold, reached, real_save = asyncio.Event(), asyncio.Event(), store.save

        async def slow_save(snapshot, *, expect_version, owner=None):
            if snapshot.in_flight is not None:  # the write that says a turn took the message is slow
                reached.set()
                await hold.wait()
            return await real_save(snapshot, expect_version=expect_version, owner=owner)

        store.save = slow_save
        go.set()
        await asyncio.wait_for(reached.wait(), 30)  # the worker took the message
        await asyncio.sleep(0.1)  # and nothing goes on while that write is out
        stored = await store.load(session.id)
        assert [e["id"] for e in stored.pending] == [sent.id] and stored.in_flight is None
        assert gate.calls == 0 and session.turns == 0  # nothing ran: a restart now runs the message once
        hold.set()
        await asyncio.wait_for(gate.entered.wait(), 30)
        stored = await store.load(session.id)
        assert stored.pending == [] and stored.in_flight["run_id"] == session.run_id
        gate.release.set()
        await records(session, "turn_finished", 1)
        await session.close()

    async def test_a_turn_whose_start_cannot_be_stored_does_not_run_and_is_tried_again(self, db, monkeypatch):
        import timbal.core.chat_session as chat_session_module

        monkeypatch.setattr(chat_session_module, "_TURN_RETRY_DELAY_S", 0.01)
        provider, store = db
        gate = Gate(released=True)
        session = ChatSession(make_agent(provider, gate), session_store=store)
        go = held_before_its_turn(session)
        sent = session.send("start")
        await session.flush()
        real_save = store.save

        async def down(*_args, **_kwargs):
            raise OSError("disk full")

        store.save = down
        go.set()
        failed = await records(session, "turn_failed", 2)
        assert {f["reason"] for f in failed} == {"not_stored"}
        assert gate.calls == 0 and session.turns == 0
        assert [p["event_id"] for p in session.pending_messages()] == [sent.id]  # handed back
        store.save = real_save  # the store is back: no new message needed
        (finished,) = await records(session, "turn_finished", 1)
        assert finished["status"]["code"] == "success" and gate.calls == 1
        await session.close()

    async def test_answers_to_a_parked_turn_are_kept_when_its_resume_cannot_be_stored(self, db, monkeypatch):
        import timbal.core.chat_session as chat_session_module

        monkeypatch.setattr(chat_session_module, "_TURN_RETRY_DELAY_S", 0.01)
        provider, store = db
        charge = Charge(released=True)
        session, approval = await parked(provider, store, charge)
        await session.flush()
        real_save = store.save

        async def down(*_args, **_kwargs):
            raise OSError("disk full")

        store.save = down
        session.resume({approval.approval_id: True})
        await records(session, "turn_failed", 1)
        assert charge.gate.calls == 0 and session.parked
        store.save = real_save
        finished = await records(session, "turn_finished", 2)
        assert finished[-1]["status"]["code"] == "success" and charge.gate.calls == 1
        await session.close()

    async def test_no_tool_runs_before_the_store_holds_that_the_model_asked_for_one(self, db, monkeypatch):
        import timbal.core.chat_session as chat_session_module

        monkeypatch.setattr(chat_session_module, "_TURN_RETRY_DELAY_S", 0.01)
        provider, store = db
        gate = Gate(released=True)
        session = ChatSession(make_agent(provider, gate), session_store=store)
        real_save, down = store.save, [True]

        async def save(snapshot, *, expect_version, owner=None):
            if down and (snapshot.in_flight or {}).get("dispatched"):
                raise OSError("disk full")  # exactly the write that says tools were asked for
            return await real_save(snapshot, expect_version=expect_version, owner=owner)

        store.save = save
        session.send("start")
        await records(session, "turn_started", 1)
        await asyncio.sleep(0.2)
        assert gate.calls == 0 and session.busy  # the model asked for `work`; it is held, not dropped
        assert not (await store.load(session.id)).in_flight.get("dispatched")
        down.clear()
        (finished,) = await records(session, "turn_finished", 1)
        assert finished["status"]["code"] == "success" and gate.calls == 1
        await session.close()

    async def test_a_snapshot_does_not_change_after_it_was_taken(self, db):
        """A store may serialize a snapshot after its first await. The turn goes on meanwhile: what is written
        must be what the save was asked to hold, not a later state (the tool calls the model asked for since)."""
        provider, store = db
        session = ChatSession(make_agent(provider, Gate(released=True)), session_store=store)
        taken = {"run_id": "r1", "parent_id": None, "trigger": {"kind": "user_message"}, "events": [], "wake": None}
        session._in_flight = copy.deepcopy(taken)
        snapshot = session._snapshot()
        session._in_flight["dispatched"] = [{"id": "call-1", "name": "work", "input": "{}"}]
        session._in_flight["events"].append({"id": "later"})
        session._in_flight["trigger"]["kind"] = "resume"
        assert snapshot.in_flight == taken

    def test_the_notice_for_tool_calls_the_trace_does_not_show(self):
        from timbal.core.chat_session import _recovery_notice

        notice = _recovery_notice([], ["charge({})"], asked=["pay the invoice"], untraced=True, answered=True)
        assert 'It was answering: "pay the invoice".' in notice
        assert "Started but not confirmed (each may or may not have run): charge({})." in notice
        assert "That question is closed now" in notice
        assert "not recorded before the restart" in notice and "No tool call had run yet" not in notice
        assert "No tool call had run yet." in _recovery_notice([], [])


class TestStoredWithTheSession:
    async def test_an_event_posted_through_the_task_store_is_stored(self, db):
        provider, store = db
        async with ChatSession(make_agent(provider, Gate(released=True)), session_store=store) as session:
            session.persist()
            await session.flush()
            event = session.store.post_event("the user opened the billing page", source="client", announce="silent")
            await session.flush()
            assert [e["id"] for e in (await store.load(session.id)).pending] == [event.id]
            await session.close(keep=True)
            restored = await ChatSession.from_store(make_agent(provider, Gate(released=True)), store, session.id)
            assert [e.id for e in restored.store.pending_events()] == [event.id]
            await restored.close()

    async def test_turn_arguments_come_back_and_those_a_snapshot_cannot_hold_are_asked_for(self, db):
        provider, store = db
        agent = make_agent(provider, Gate(released=True))
        session = ChatSession(agent, session_store=store, max_tokens=37)
        session.send("hi")
        await records(session, "turn_finished", 1)
        await session.close(keep=True)
        restored = await ChatSession.from_store(agent, store, session.id)
        assert restored._kwargs == {"max_tokens": 37}
        restored.send("again")
        await records(restored, "turn_finished", 1, after=after_restore(restored))
        trace = await RunContext(tracing_provider=provider)._get_parent_trace(restored.head)
        llm = next(span for span in trace.values() if span.path == "agent.llm")
        assert llm.input["max_tokens"] == 37  # the restored turn ran with it
        await restored.close()

        def chooser() -> str:
            return "x"

        other = ChatSession(agent, session_store=store, max_tokens=37, chooser=chooser)
        other.persist()
        await other.flush()
        await other.close(keep=True)
        with pytest.raises(ValueError, match="chooser"):
            await ChatSession.from_store(agent, store, other.id)
        again = await ChatSession.from_store(agent, store, other.id, chooser=chooser)  # the lease was let go
        assert again._kwargs == {"max_tokens": 37, "chooser": chooser}
        await again.close()

    async def test_the_stored_wake_survives_replacements_that_were_never_stored(self, db):
        provider, store = db
        gate = Gate()
        async with ChatSession(make_agent(provider, gate), session_store=store) as session:
            session.send("start")  # busy: wakes wait
            await asyncio.wait_for(gate.entered.wait(), 30)
            first = session.wake("first", key="k")
            await session.flush()
            real_save = store.save

            async def down(*_args, **_kwargs):
                raise OSError("disk full")

            for order in ("second-then-third", "third-then-second"):
                store.save = down
                second = session.wake("second", key="k")
                third = session.wake("third", key="k")
                with pytest.raises(SessionNotStored):
                    await session.flush()
                for wake_id in (second, third) if order == "second-then-third" else (third, second):
                    assert session.withdraw(wake_id)
                assert [wake.id for wake in session._wakes] == [first], order
                store.save = real_save
                await session.flush()
                assert [wake["id"] for wake in (await store.load(session.id)).wakes] == [first], order
            gate.release.set()


def steering_agent(provider, first, second):
    """``start`` calls ``first``; a message steered in after it calls ``second``."""

    def model(messages):
        text = last_user_text(messages)
        if "cut off by a server restart" in text:
            return "recovered"
        results = sum(1 for message in messages if message.role == "tool")
        if messages[-1].role == "tool":
            return "done"
        if "start" in text and not results:
            return tool_call("first", "call-first")
        if "also" in text and results == 1:
            return tool_call("second", "call-second")
        return f"echo: {text}"

    return Agent(
        name="agent",
        model=TestModel(handler=model),
        tools=[Tool(name="first", handler=first.work), Tool(name="second", handler=second.work)],
        tracing_provider=provider,
    )


class TestSteeredMessages:
    """A message steered into a running turn is no longer pending: it stays with the turn in the snapshot."""

    async def _steered_then_crashed(self, provider, store, *, trace_until_steered):
        first, second = Gate(), Gate()
        session = ChatSession(steering_agent(provider, first, second), session_store=store)
        session.send("start also")  # holds the text of the message steered in after it
        await asyncio.wait_for(first.entered.wait(), 30)
        if not trace_until_steered:
            provider.crashed = True  # what is stored of the run from here on never lands
        also = session.send("also", mode="steer")
        first.release.set()
        await asyncio.wait_for(second.entered.wait(), 30)  # the turn took the steered message and went on
        await session.flush()
        stored = await store.load(session.id)
        assert stored.pending == []
        assert [(e["text"], e.get("steered", False)) for e in stored.in_flight["events"]] == [
            ("start also", False),
            ("also", True),
        ]
        await crash(session)
        provider.crashed = False
        return session, also

    async def test_one_the_stored_run_does_not_show_goes_back_in_the_queue(self, db):
        provider, store = db
        session, also = await self._steered_then_crashed(provider, store, trace_until_steered=False)
        agent = steering_agent(provider, Gate(released=True), Gate(released=True))
        restored = await ChatSession.from_store(agent, store, session.id)
        assert [p["event_id"] for p in restored.state()["pending"]] == [also.id]
        started = await records(restored, "turn_started", 2, after=after_restore(restored))
        await records(restored, "turn_finished", 2, after=after_restore(restored))
        assert [s["trigger"]["kind"] for s in started] == ["recovery", "user_message"]
        await restored.close()

    async def test_one_the_stored_run_shows_is_not_answered_twice(self, db):
        provider, store = db
        session, _ = await self._steered_then_crashed(provider, store, trace_until_steered=True)
        agent = steering_agent(provider, Gate(released=True), Gate(released=True))
        restored = await ChatSession.from_store(agent, store, session.id)
        assert restored.state()["pending"] == []
        await records(restored, "turn_finished", 1, after=after_restore(restored))
        await restored.close()

    async def test_the_recovery_of_a_run_with_no_trace_is_told_about_it(self, db):
        provider, store = db
        provider.crashed = True  # nothing of the run is ever stored
        session, _ = await self._steered_then_crashed(provider, store, trace_until_steered=True)
        agent = steering_agent(provider, Gate(released=True), Gate(released=True))
        restored = await ChatSession.from_store(agent, store, session.id)
        assert 'It was answering: "start also"; "also".' in restored._wakes[0].text
        assert restored.state()["pending"] == []
        await records(restored, "turn_finished", 1, after=after_restore(restored))
        await restored.close()

    async def test_one_the_stored_run_shows_is_not_restored_from_an_older_snapshot(self, db):
        """The snapshot that says a turn took it may never be written: the stored run says so too."""
        provider, store = db
        session, also = await self._steered_then_crashed(provider, store, trace_until_steered=True)
        stale = await store.load(session.id)
        taken = stale.in_flight["events"].pop()  # as stored before the turn took it: still pending
        stale.pending = [{key: value for key, value in taken.items() if key != "steered"}]
        await store.save(stale, expect_version=stale.version)
        agent = steering_agent(provider, Gate(released=True), Gate(released=True))
        restored = await ChatSession.from_store(agent, store, session.id)
        assert also.id not in [p["event_id"] for p in restored.state()["pending"]]
        await records(restored, "turn_finished", 1, after=after_restore(restored))
        await restored.close()

    async def test_an_app_event_steered_in_stays_with_the_turn_too(self, db):
        provider, store = db
        first, second = Gate(), Gate()
        session = ChatSession(steering_agent(provider, first, second), session_store=store)
        session.send("start")
        await asyncio.wait_for(first.entered.wait(), 30)
        event = session.store.post_event("also: the invoice was paid elsewhere", lane="next")
        first.release.set()
        await asyncio.wait_for(second.entered.wait(), 30)
        await session.flush()
        stored = await store.load(session.id)
        assert stored.pending == []
        assert [(e["id"], e.get("steered", False)) for e in stored.in_flight["events"]][1:] == [(event.id, True)]
        second.release.set()
        await records(session, "turn_finished", 1)
        await session.close()


class Till:
    """A tool in flight: it charges when released, unless its turn was stopped first."""

    def __init__(self):
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.charges = 0
        self.stopped = False

    async def work(self) -> str:
        self.entered.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.stopped = True
            raise
        self.charges += 1
        return "worked"


async def taken_over(store, session_id, timeout=30.0):
    """Another worker asks for the session until the store gives it to it (the lease there has run out)."""

    async def ask():
        while True:
            try:
                return await store.acquire_lease(session_id, "elsewhere:1:x")
            except SessionLeased:
                await asyncio.sleep(0.005)

    await asyncio.wait_for(ask(), timeout)


def renewals(store, renew):
    """This process's lease calls on ``store`` go to ``renew`` (given the real call); another worker's go on."""
    real = store.acquire_lease

    async def acquire(session_id, owner):
        if owner == process_owner():
            return await renew(lambda: real(session_id, owner))
        return await real(session_id, owner)

    store.acquire_lease = acquire


class TestLeaseRunsOut:
    """A lease this process could not renew runs out at a time the store and the session both know. The
    session has stopped by then, its turn with it: the worker the store gives the session to next must not
    find the old one still acting."""

    # Every test here first gets its session mid-tool, still the owner. A stalled event loop (a busy CI
    # runner) uses up the lease as surely as a store that does not answer, and the session then rightly gives
    # it up: a 0.9 s lease leaves about 0.4 s between a failed renewal and the end of the lease, and a
    # runner took more than that during the setup alone; this one leaves 1.4 s. It is the lease for the
    # whole test: the store stamps it when it gives it, and the session has planned its next renewal by the
    # one it holds, so a shorter one set under a live session is given up without a renewal being tried.
    KEPT_TTL = 3.0

    async def _mid_tool(self, db, ttl=KEPT_TTL):
        provider, store = db
        store.lease_ttl_s = ttl
        till = Till()
        session = ChatSession(make_agent(provider, till), session_store=store)
        session.send("start")
        await asyncio.wait_for(till.entered.wait(), 30)
        await session.flush()
        return session, store, till

    @pytest.mark.parametrize("store_is", ["hung", "down", "deaf"])
    async def test_the_session_has_stopped_when_another_worker_can_take_it(self, db, store_is):
        session, store, till = await self._mid_tool(db)
        version = (await store.load(session.id)).version
        back = asyncio.Event()

        async def renew(_real):
            if store_is == "down":
                raise OSError("store unreachable")
            while not back.is_set():
                try:
                    await back.wait()  # hung: no answer...
                except asyncio.CancelledError:
                    if store_is == "hung":
                        raise  # ...deaf: and it does not stop when it is told to, either

        renewals(store, renew)
        try:
            await taken_over(store, session.id)
            # Not a third of a lease later, when the renewal that was out gives up: by the time the store
            # hands the session to someone else, this copy knows it is not the owner.
            assert session.lost
            await asyncio.wait_for(session.wait_closed(), 30)
            assert session.closed and not session.busy and till.stopped
            till.release.set()  # the tool it had dispatched, let go under the new owner: it is not there to charge
            await asyncio.sleep(0.05)
            assert till.charges == 0
            assert (await store.load(session.id)).version == version  # and it wrote nothing more
        finally:
            back.set()

    async def test_a_store_that_answers_within_the_lease_keeps_the_session(self, db):
        """Slow is not gone: renewals that take a good part of a lease, and land, are renewals."""
        ttl = self.KEPT_TTL
        session, store, till = await self._mid_tool(db, ttl)
        landed = []

        async def renew(real):
            await asyncio.sleep(ttl / 5)
            result = await real()
            landed.append(1)
            return result

        renewals(store, renew)
        # Four slow renewals, sent a third of a lease apart: the last lands more than a lease after the first
        # was sent, when any lease confirmed before them has run out. A session that did not count them has
        # given up by then, and its renewals stopped.
        await until(lambda: len(landed) >= 4 or session.lost)
        assert not session.lost and session.busy
        with pytest.raises(SessionLeased):
            await store.acquire_lease(session.id, "elsewhere:1:x")  # still its own: renewed each time
        till.release.set()
        await records(session, "turn_finished", 1)
        assert till.charges == 1
        await session.close()

    async def test_one_failed_renewal_is_tried_again_before_the_lease_runs_out(self, db):
        session, store, till = await self._mid_tool(db, self.KEPT_TTL)
        failed, retried = [], []

        async def renew(real):
            if not failed:
                failed.append(1)
                raise OSError("store unreachable")
            result = await real()
            retried.append(1)
            return result

        renewals(store, renew)
        # The retry lands, or the session gave up first (and never tries it): wait for whichever comes.
        await until(lambda: retried or session.lost)
        assert failed and retried and not session.lost and session.busy
        with pytest.raises(SessionLeased):
            await store.acquire_lease(session.id, "elsewhere:1:x")  # the retry renewed it: still its own
        till.release.set()
        await records(session, "turn_finished", 1)
        await session.close()

    async def test_a_restored_session_counts_its_lease_from_when_it_took_it(self, db):
        """Not from when the restore was done: a slow one has used up part of the lease already, and the
        first renewal is due as much sooner."""
        provider, store = db
        session = ChatSession(make_agent(provider, Gate(released=True)), session_store=store)
        session.send("hi")
        await records(session, "turn_finished", 1)
        await session.close(keep=True)
        store.lease_ttl_s = ttl = 3.0
        asked: list[float] = []
        real_load, loop = store.load, asyncio.get_running_loop()

        async def slow_load(session_id):
            await asyncio.sleep(ttl * 0.4)  # more than the third of a lease between renewals
            return await real_load(session_id)

        async def renew(real):
            asked.append(loop.time())
            return await real()

        store.load = slow_load
        renewals(store, renew)
        restored = await ChatSession.from_store(make_agent(provider, Gate(released=True)), store, session.id)
        done = loop.time()
        try:
            await until(lambda: len(asked) >= 2)  # the one that took it, then the first renewal
            assert asked[1] - done < ttl / 6  # at once: not a third of a lease after the restore returned
            assert not restored.lost
        finally:
            await restored.close(keep=True)

    async def test_a_restore_that_outlasts_its_lease_serves_nothing(self, db):
        """Its lease ran out while it was still reading: another worker may have the session by now."""
        provider, store = db
        model_calls = []

        def counted(messages):
            model_calls.append(1)
            return handler(messages)

        agent = Agent(name="agent", model=TestModel(handler=counted), tools=[], tracing_provider=provider)
        session = ChatSession(agent, session_store=store)
        held_before_its_turn(session)
        session.send("hi")
        await session.flush()
        await crash(session)
        store.lease_ttl_s = ttl = 0.3
        real_load = store.load

        async def slow_load(session_id):
            await asyncio.sleep(ttl * 1.2)
            return await real_load(session_id)

        store.load = slow_load
        restored = await ChatSession.from_store(agent, store, session.id)
        await asyncio.wait_for(restored.wait_closed(), 30)
        assert restored.lost and restored.handed_off
        store.load = real_load
        assert [event["text"] for event in (await store.load(session.id)).pending] == ["hi"]  # still waiting
        assert model_calls == []

    async def test_a_lease_that_was_taken_is_noticed_at_the_next_renewal(self, db):
        """Refused, not unanswered: the store says who holds the session, and this copy stops without writing."""
        session, store, till = await self._mid_tool(db)
        refused, taken = [], asyncio.Event()

        async def renew(real):
            await taken.wait()  # (one landing between the two calls below would take the session back)
            try:
                return await real()
            except SessionLeased:
                refused.append(1)
                raise

        renewals(store, renew)
        store.lease_ttl_s = 60  # (the other worker's lease: it must not run out before this copy looks)
        await store.release_lease(session.id, process_owner())
        await store.acquire_lease(session.id, "elsewhere:1:x")
        store.lease_ttl_s = self.KEPT_TTL
        version = (await store.load(session.id)).version
        taken.set()
        await until(lambda: session.closed)
        # Refused once: it stopped there, and did not go on asking until the lease ran out.
        assert refused == [1] and session.lost and session.state()["durable"] is False
        await asyncio.wait_for(session.wait_closed(), 30)
        assert till.stopped and (await store.load(session.id)).version == version  # not written over


class ThinkingModel(TestModel):
    """A model that is still thinking: a call whose last user message is in ``slow`` waits at ``go``."""

    __test__ = False

    def __init__(self, slow=()):
        super().__init__(handler=self._answer)
        self.slow, self.thinking, self.go = set(slow), asyncio.Event(), asyncio.Event()
        self.answered: list[str] = []

    def _answer(self, messages):
        if messages[-1].role == "tool":
            return "done"
        text = last_user_text(messages)
        if text == "build":
            return tool_call("build", f"call-{len(messages)}")
        self.answered.append(text)
        return f"echo: {text}"

    async def stream(self, messages, **kwargs):
        if last_user_text(messages) in self.slow and messages[-1].role == "user":
            self.thinking.set()
            await self.go.wait()
        async for item in super().stream(messages, **kwargs):
            yield item


class TestRestoredWhereItWasClosed:
    """``close(keep=True)`` and ``from_store`` in one process: an idle session let go and asked for again, a
    lease lost and taken back. The run's task store may still be there (a background task going, an event
    no turn took), with the closed session's events in it. The restored session must get what one restored
    in another process gets, and the closed one must stay closed."""

    async def _cut_while_thinking(self, db, *, then):
        """A session whose turn ``first`` is cut before its model answered (nothing dispatched: its message
        is queued again), with a background task still running and whatever ``then`` queued behind it."""
        provider, store = db
        model, built = ThinkingModel(slow={"first"}), asyncio.Event()

        async def build() -> str:
            await built.wait()
            return "built"

        agent = Agent(
            name="agent",
            model=model,
            tools=[Tool(name="build", handler=build, background_mode="always", background_announce="silent")],
            tracing_provider=provider,
        )
        session = ChatSession(agent, session_store=store, busy_mode="followup")
        session.send("build")  # a background task: the run's task store outlives the session
        await records(session, "turn_finished", 1)
        session.send("first")
        await asyncio.wait_for(model.thinking.wait(), 30)
        then(session)
        await session.flush()
        await session.close(keep=True)
        model.slow.clear()
        return session, agent, store, model, built

    async def test_what_the_cut_turn_took_goes_back_ahead_of_what_was_queued_behind_it(self, db):
        session, agent, store, model, built = await self._cut_while_thinking(db, then=lambda s: s.send("second"))
        try:
            restored = await ChatSession.from_store(agent, store, session.id)
            assert restored._store is session._store  # the case: one process, the task store still there
            await records(restored, "turn_finished", 2, after=after_restore(restored))
            assert model.answered == ["first", "second"]  # in the order they were sent
            await restored.close()
        finally:
            built.set()

    async def test_a_newer_event_with_the_same_key_stands(self, db):
        """The cut turn took an event; a newer one with its key was posted behind it. Queued again, the
        older one must not replace the newer."""
        provider, store = db
        model = ThinkingModel(slow={"first"})
        agent = Agent(name="agent", model=model, tools=[], tracing_provider=provider)
        session = ChatSession(agent, session_store=store, busy_mode="followup")
        session.send("first")
        await asyncio.wait_for(model.thinking.wait(), 30)
        old = session.store.post_event("stock: 3 left", key="stock", lane="next")
        await until(lambda: old.state == "pending")
        await session.flush()
        # Taken by the turn the way a steered event is, and kept with it; then a newer one is posted.
        session.store.drain_events(session._in_flight["run_id"], where=lambda event: event is old)
        new = session.store.post_event("stock: sold out", key="stock", lane="next")
        await session.flush()
        assert [event["id"] for event in (await store.load(session.id)).in_flight["events"]][-1] == old.id
        await session.close(keep=True)
        model.slow.clear()

        restored = await ChatSession.from_store(agent, store, session.id)
        try:
            pending = [event for event in restored._store.pending_events() if event.key == "stock"]
            assert [event.text for event in pending] == ["stock: sold out"] and new.state == "pending"
        finally:
            await restored.close()

    async def test_the_restored_session_follows_the_events_and_the_closed_one_writes_nothing(self, db):
        session, agent, store, model, built = await self._cut_while_thinking(db, then=lambda s: s.send("second"))
        try:
            closed_changes, closed_log = session._changes, session.log.cursor_end
            restored = await ChatSession.from_store(agent, store, session.id)
            second = next(event for event in restored._store.pending_events() if event.text == "second")
            finished = await records(restored, "turn_finished", 2, after=after_restore(restored))
            settled = [
                item
                for item in restored.log.peek(after_restore(restored))[0]
                if isinstance(item, dict) and item.get("kind") == "message_settled"
            ]
            # The restored session's log says where the message went: its clients are watching that one.
            assert {"event_id": second.id, "state": "delivered", "run_id": finished[1]["run_id"]}.items() <= next(
                item for item in settled if item["event_id"] == second.id
            ).items()
            await restored.flush()
            # The closed session did nothing with it: no record, no write under this process's lease (which
            # the store could not tell from the live session's).
            assert session._changes == closed_changes and session.log.cursor_end == closed_log
            assert session._saver is None or session._saver.done()
            assert not restored.lost and (await store.load(session.id)).turns == restored.turns
            await restored.close()
        finally:
            built.set()

    async def test_an_event_the_snapshot_names_twice_is_queued_once(self, db):
        """With the turn that took it, and pending again: the turn was cut before it started and handed the
        message back. Whichever process restores it, the message is answered once."""
        provider, store = db
        model = ThinkingModel()
        agent = Agent(name="agent", model=model, tools=[], tracing_provider=provider)
        message = {"id": "m1", "text": "hello", "source": "user", "lane": "later", "mode": "followup", "posted_at": 1.0}
        snapshot = SessionSnapshot(
            session_id="s1",
            agent="agent",
            pending=[message],
            in_flight={"run_id": "a-run-that-never-started", "parent_id": None, "events": [message]},
        )
        await store.save(snapshot, expect_version=None)
        restored = await ChatSession.from_store(agent, store, "s1")
        (started,) = await records(restored, "turn_started", 1, after=after_restore(restored))
        assert started["trigger"]["event_ids"] == ["m1"]
        await records(restored, "turn_finished", 1, after=after_restore(restored))
        assert model.answered == ["hello"]
        await restored.close()

    async def test_a_steered_message_kept_from_before_is_stored_with_the_turn_that_takes_it(self, db):
        """Kept in the task store across the close: the restored session must still record that a turn took
        it (``in_flight.events``), or a crash during that turn loses the message."""
        provider, store = db
        gate = Gate()
        session = ChatSession(make_agent(provider, gate), session_store=store)
        session.send("hi")  # a run to continue from: the task store is the one registered for it
        await records(session, "turn_finished", 1)
        hold = held_before_its_turn(session)
        session.send("start", mode="followup")
        session.send("keep the old API", mode="steer")  # for the turn that will be running: after its tools
        await session.flush()
        await session.close(keep=True)
        hold.set()
        with_the_turn: list[list[tuple[str, bool]]] = []
        real_save = store.save

        async def save(snapshot, **kwargs):
            events = (snapshot.in_flight or {}).get("events") or []
            with_the_turn.append([(event["text"], bool(event.get("steered"))) for event in events])
            return await real_save(snapshot, **kwargs)

        store.save = save
        restored = await ChatSession.from_store(make_agent(provider, gate), store, session.id)
        assert restored._store is session._store
        await asyncio.wait_for(gate.entered.wait(), 30)
        gate.release.set()
        await records(restored, "turn_finished", 1, after=after_restore(restored))
        await restored.flush()
        assert [("start", False), ("keep the old API", True)] in with_the_turn
        await restored.close()


class TestRecordedToolCalls:
    """The agent tells the session which tool calls it is about to run; they are stored before any does."""

    async def test_a_resumed_question_is_not_answered_again_once_the_turn_went_on(self, db):
        """Answering the question again would run again the tool the model called after it."""
        from timbal.state import suspend

        provider, store = db

        def ask() -> str:
            return str(suspend({"question": "Ship it?"}))

        def model(messages):
            text = last_user_text(messages)
            if "cut off by a server restart" in text:
                return "recovered"
            results = sum(1 for message in messages if message.role == "tool")
            if not results:
                return tool_call("ask", "call-ask")
            return tool_call("ship", "call-ship") if results == 1 else "done"

        def agent(ship):
            return Agent(
                name="agent",
                model=TestModel(handler=model),
                tools=[Tool(name="ask", handler=ask), Tool(name="ship", handler=ship.work)],
                tracing_provider=provider,
            )

        first = Gate()
        session = ChatSession(agent(first), session_store=store)
        session.send("go")
        await records(session, "turn_finished", 1)
        (question,) = session.state()["pending_interactions"]
        provider.crashed = True  # nothing of the resumed turn is ever stored
        session.resume({question["interaction_id"]: "yes"})
        await asyncio.wait_for(first.entered.wait(), 30)  # it took the answer and went on to an ordinary tool
        await crash(session)
        provider.crashed = False

        again = Gate(released=True)
        restored = await ChatSession.from_store(agent(again), store, session.id)
        notice = restored._wakes[0].text
        assert "That question is closed now" in notice and "ship(" in notice
        started = await records(restored, "turn_started", 2, after=after_restore(restored))
        await records(restored, "turn_finished", 2, after=after_restore(restored))
        assert [s["trigger"]["kind"] for s in started] == ["resume", "recovery"]
        assert again.calls == 0 and not restored.parked  # the tool did not run a second time
        await restored.close()

    async def test_a_resume_that_ran_nothing_gets_its_answers_back(self, db):
        """Stored as taken, then the process died before any of the calls it was for had started."""
        provider, store = db
        charge = Charge(released=True)
        session, approval = await parked(provider, store, charge)
        hold = asyncio.Event()
        real_pump = session._pump

        async def never_starts(*args, **kwargs):
            await hold.wait()
            return await real_pump(*args, **kwargs)

        session._pump = never_starts
        session.resume({approval.approval_id: True})
        await until(lambda: session._in_flight is not None)
        await session.flush()
        assert (await store.load(session.id)).in_flight["resume"] and charge.gate.calls == 0
        await crash(session)
        provider.crashed = False

        restored = await ChatSession.from_store(gated_agent(provider, charge), store, session.id)
        (finished,) = await records(restored, "turn_finished", 1, after=after_restore(restored))
        assert finished["status"]["code"] == "success" and charge.gate.calls == 1  # approved once, charged once
        await restored.close()

    async def test_a_slash_command_is_recorded_whatever_its_tool_is_called(self, db):
        provider, store = db
        provider.crashed = True  # no trace of the turn ever lands

        def agent(gate):
            return Agent(
                name="agent",
                model=TestModel(handler=handler),
                tools=[Tool(name="llm", command="/doit", handler=gate.work)],  # named like the model's own span
                tracing_provider=provider,
            )

        gate = Gate()
        session = ChatSession(agent(gate), session_store=store)
        session.send("/doit")
        await asyncio.wait_for(gate.entered.wait(), 30)
        assert [call["name"] for call in (await store.load(session.id)).in_flight["dispatched"]] == ["llm"]
        await crash(session)
        provider.crashed = False

        again = Gate(released=True)
        restored = await ChatSession.from_store(agent(again), store, session.id)
        assert restored.state()["pending"] == []
        (started,) = await records(restored, "turn_started", 1, after=after_restore(restored))
        await records(restored, "turn_finished", 1, after=after_restore(restored))
        assert started["trigger"]["kind"] == "recovery" and again.calls == 0
        await restored.close()

    async def test_a_call_started_after_the_last_stored_step_is_reported(self, db):
        """The trace shows `lookup` done; `ship` started after the last save that landed."""
        provider, store = db

        def lookup() -> str:
            return "found"

        def model(messages):
            text = last_user_text(messages)
            if "cut off by a server restart" in text:
                return "recovered"
            results = sum(1 for message in messages if message.role == "tool")
            if not results:
                return tool_call("lookup", "call-lookup")
            if results == 1:
                provider.crashed = True  # from here on nothing of the run is stored
                return tool_call("ship", "call-ship")
            return "done"

        def agent(ship):
            return Agent(
                name="agent",
                model=TestModel(handler=model),
                tools=[Tool(name="lookup", handler=lookup), Tool(name="ship", handler=ship.work)],
                tracing_provider=provider,
            )

        first = Gate()
        session = ChatSession(agent(first), session_store=store)
        session.send("go")
        await asyncio.wait_for(first.entered.wait(), 30)
        await crash(session)
        provider.crashed = False

        restored = await ChatSession.from_store(agent(Gate(released=True)), store, session.id)
        notice = restored._wakes[0].text
        assert "Completed before the restart: lookup(" in notice
        assert "Started but not confirmed (each may or may not have run): ship(" in notice
        await records(restored, "turn_finished", 1, after=after_restore(restored))
        await restored.close()

    async def test_a_sub_agents_calls_are_left_to_the_call_that_runs_it(self, db):
        provider, store = db
        gate = Gate()
        child = Agent(
            name="child",
            model=TestModel(handler=lambda messages: "done" if messages[-1].role == "tool" else tool_call("dig", "c1")),
            tools=[Tool(name="dig", handler=gate.work)],
            tracing_provider=provider,
        )
        parent = Agent(
            name="agent",
            model=TestModel(
                handler=lambda messages: (
                    "done"
                    if messages[-1].role == "tool"
                    else Message(
                        role="assistant",
                        content=[ToolUseContent(id="p1", name="child", input={"prompt": "dig"})],
                        stop_reason="tool_use",
                    )
                )
            ),
            tools=[child],
            tracing_provider=provider,
        )
        async with ChatSession(parent, session_store=store) as session:
            session.send("go")
            await asyncio.wait_for(gate.entered.wait(), 30)
            dispatched = (await store.load(session.id)).in_flight["dispatched"]
            assert [call["name"] for call in dispatched] == ["child"]
            gate.release.set()
            await records(session, "turn_finished", 1)


class TestCreatedWithItsLease:
    @pytest.fixture(params=["memory", "sqlite"])
    def store(self, request, tmp_path):
        return InMemoryChatSessionStore() if request.param == "memory" else SqliteChatSessionStore(tmp_path / "s.db")

    async def test_a_new_row_is_its_owners_from_the_write_that_creates_it(self, store):
        await store.save(SessionSnapshot(session_id="s1", agent="agent"), expect_version=None, owner="a")
        with pytest.raises(SessionLeased):
            await store.acquire_lease("s1", "b")
        await store.acquire_lease("s1", "a")
        await store.save(SessionSnapshot(session_id="s2", agent="agent"), expect_version=None)  # no owner: free
        await store.acquire_lease("s2", "b")

    async def test_a_session_does_not_need_a_second_write_to_hold_its_lease(self, db):
        provider, store = db

        async def unreachable(*_args, **_kwargs):
            raise OSError("store unreachable")

        store.acquire_lease = unreachable  # the write that used to take the lease after the row was there
        async with ChatSession(make_agent(provider, Gate(released=True)), session_store=store) as session:
            session.persist()
            await session.flush()
            assert session._leased and session.state()["durable"]


async def kill(session):
    """The process dies where it stands: no last flush (unlike :func:`crash`), its tasks just stop."""
    session._lost = True  # stores nothing more
    tasks = [t for t in (session._worker, session._turn, session._lease_task, session._saver) if t is not None]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    session.closed = True
    session._scheduler.release(session._store)
    forget_idle_background_store(session._store)


class TestSnapshotBehindTheTrace:
    """The trace provider and the session store are written separately: either can be ahead at a crash."""

    async def test_a_steered_message_a_finished_run_shows_is_not_answered_again(self, db):
        """The run answered the steered message and its trace was stored; the snapshots after it never were,
        so the last one still lists the message as pending."""
        provider, store = db
        first = Gate()

        def model(messages):
            if messages[-1].role == "tool":
                return "done"
            text = last_user_text(messages)
            return tool_call("first", "call-first") if text == "start" else f"echo: {text}"

        def agent(gate):
            return Agent(
                name="agent",
                model=TestModel(handler=model),
                tools=[Tool(name="first", handler=gate.work)],
                tracing_provider=provider,
            )

        session = ChatSession(agent(first), session_store=store)
        session.send("start")
        await asyncio.wait_for(first.entered.wait(), 30)
        also = session.send("also", mode="steer")
        await session.flush()  # acknowledged: stored as pending, with the turn in flight
        real_save = store.save

        async def never_lands(*_args, **_kwargs):
            await asyncio.Event().wait()

        store.save = never_lands  # from here on no snapshot is written
        first.release.set()
        await records(session, "turn_finished", 1)  # it took the message, answered it, and its trace is stored
        assert root_outputs(session)[0].output.collect_text() == "echo: also"
        await kill(session)
        store.save = real_save
        stored = await store.load(session.id)
        assert [e["id"] for e in stored.pending] == [also.id] and stored.in_flight is not None

        restored = await ChatSession.from_store(agent(Gate(released=True)), store, session.id)
        assert restored.state()["pending"] == []
        await asyncio.sleep(0.3)
        assert restored.turns == 1 and not restored.busy  # no second turn for the same message
        await restored.close()


class TestConditionalDelete:
    @pytest.fixture(params=["memory", "sqlite"])
    def store(self, request, tmp_path):
        return InMemoryChatSessionStore() if request.param == "memory" else SqliteChatSessionStore(tmp_path / "s.db")

    async def test_delete_takes_the_row_only_at_the_version_and_owner_given(self, store):
        version = await store.save(SessionSnapshot(session_id="s1", agent="agent"), expect_version=None, owner="a")
        with pytest.raises(SnapshotConflict):
            await store.delete("s1", expect_version=version + 1, owner="a")  # someone wrote since
        with pytest.raises(SnapshotConflict):
            await store.delete("s1", expect_version=version, owner="b")  # someone else holds it
        assert await store.load("s1") is not None
        await store.delete("s1", expect_version=version, owner="a")
        assert await store.load("s1") is None
        await store.delete("s1", expect_version=version, owner="a")  # already gone: not an error
        await store.save(SessionSnapshot(session_id="s2", agent="agent"), expect_version=None)
        await store.delete("s2")  # unconditional, as before
        assert await store.load("s2") is None

    async def test_closing_does_not_delete_a_session_another_process_took(self, db):
        """This copy stalled past its lease; another worker took the session and wrote to it. A close here
        whose last write fails must not then delete the other worker's row."""
        provider, store = db
        session = ChatSession(make_agent(provider, Gate(released=True)), session_store=store)
        session.send("hi")
        await records(session, "turn_finished", 1)
        await session.flush()
        session._lease_task.cancel()
        await store.release_lease(session.id, process_owner())  # lapsed...
        await store.acquire_lease(session.id, "elsewhere:1:x")  # ...taken...
        theirs = await store.load(session.id)
        theirs.turns = 99
        await store.save(theirs, expect_version=theirs.version, owner="elsewhere:1:x")  # ...and written to
        real_save = store.save

        async def down(*_args, **_kwargs):
            raise OSError("disk full")

        store.save = down  # this copy's closing snapshot fails, so it never learns from a version conflict
        await session.close()
        store.save = real_save
        kept = await store.load(session.id)
        assert kept is not None and kept.turns == 99  # the other worker's row is still there
        assert session.lost  # and the caller is told (DELETE answers 409)


class TestFoundByTheCrashSweeps:
    """Bugs ``test_chat_session_crash_points.py`` found, each pinned here on its own."""

    async def test_a_message_whose_turn_a_shutdown_cut_during_the_model_call_is_answered_after_it(self, db):
        """An orderly stop stores the cut run's trace, and that trace need not hold the prompt. The turn ran no
        tool call, so its message goes back in the queue: a recovery turn on top of it would answer nobody."""
        provider, store = db
        thinking = asyncio.Event()

        async def before_the_model() -> None:
            thinking.set()
            await asyncio.sleep(30)  # the shutdown comes here: started, no tool call yet

        slow = Agent(
            name="agent", model=TestModel(handler=handler), pre_hook=before_the_model, tracing_provider=provider
        )
        session = ChatSession(slow, session_store=store)
        sent = session.send("hello")
        await asyncio.wait_for(thinking.wait(), 3)
        await session.flush()
        await session.close(keep=True)

        restored = await ChatSession.from_store(make_agent(provider, Gate(released=True)), store, session.id)
        assert [p["event_id"] for p in restored.state()["pending"]] == [sent.id]
        (started,) = await records(restored, "turn_started", 1, after=after_restore(restored))
        assert started["trigger"]["kind"] == "user_message"  # its own turn, not a recovery turn
        await records(restored, "turn_finished", 1, after=after_restore(restored))
        assert root_outputs(restored)[0].output.collect_text() == "echo: hello"
        await restored.close()

    async def test_a_turn_is_not_over_until_its_trace_is_stored(self, tmp_path, monkeypatch):
        """The next turn reads this one from the trace store. A turn whose last save failed is held open until
        a save lands: settled without it, the conversation would go on with no memory of it."""
        import timbal.core.chat_session as chat_session_module

        monkeypatch.setattr(chat_session_module, "_TURN_RETRY_DELAY_S", 0.01)

        class Unreachable(SqliteTracingProvider):
            down = False

            @classmethod
            async def _store(cls, run_context):
                if cls.down:
                    raise OSError("the trace store is unreachable")
                await super()._store(run_context)

        provider = Unreachable.configured(_path=tmp_path / "timbal.db", down=True)
        store = SqliteChatSessionStore(tmp_path / "timbal.db")
        session = ChatSession(make_agent(provider, Gate(released=True)), session_store=store)
        session.send("hi")
        await records(session, "turn_started", 1)
        await asyncio.sleep(0.3)
        log = [r for r in session.log.peek()[0] if isinstance(r, dict)]
        assert session.busy and not [r for r in log if r.get("kind") == "turn_finished"]  # held open
        provider.down = False
        (finished,) = await records(session, "turn_finished", 1)
        assert finished["status"]["code"] == "success"
        session.send("again")
        await records(session, "turn_finished", 2)
        memory = await session._load_run(session.head)
        texts = [m["content"][0].get("text") for m in memory.root_span().memory if m.get("role") == "user"]
        assert texts == ["hi", "again"]  # the first turn is in the history the second one read
        await session.close()

    @pytest.mark.parametrize("how", ["interrupt()", "an interrupt message", "a now message"])
    async def test_an_interrupt_does_not_give_up_the_wait_for_a_turns_trace(self, tmp_path, monkeypatch, how):
        """The session is busy while a finished turn waits for its trace, so an interrupt is likely to come
        just then. The turn has ended: there is nothing to cut, and going on without its trace would lose it."""
        import timbal.core.chat_session as chat_session_module

        monkeypatch.setattr(chat_session_module, "_TURN_RETRY_DELAY_S", 0.01)

        class Unreachable(SqliteTracingProvider):
            down = False

            @classmethod
            async def _store(cls, run_context):
                if cls.down:
                    raise OSError("the trace store is unreachable")
                await super()._store(run_context)

        provider = Unreachable.configured(_path=tmp_path / "timbal.db", down=True)
        store = SqliteChatSessionStore(tmp_path / "timbal.db")
        session = ChatSession(make_agent(provider, Gate(released=True)), session_store=store)
        session.send("hi")
        await records(session, "turn_started", 1)
        await ended_unsettled(session)  # the turn has ended; its last save keeps failing
        if how == "interrupt()":
            assert session.interrupt() is False  # nothing to cut, and it left no question
        else:
            session.send("stop", mode="interrupt" if how == "an interrupt message" else "now")
        await asyncio.sleep(0.2)
        log = [r for r in session.log.peek()[0] if isinstance(r, dict)]
        assert session.busy and not [r for r in log if r.get("kind") == "turn_finished"]  # still held open
        assert session.turns == 1  # and nothing else started
        provider.down = False
        expected = 1 if how == "interrupt()" else 2
        finished = await records(session, "turn_finished", expected)
        assert [f["status"]["code"] for f in finished] == ["success"] * expected
        memory = await session._load_run(session.head)
        texts = [m["content"][0].get("text") for m in memory.root_span().memory if m.get("role") == "user"]
        assert texts == (["hi"] if how == "interrupt()" else ["hi", "stop"])  # the first turn is in the history
        await session.close()

    @pytest.mark.parametrize("how", ["interrupt()", "an interrupt message"])
    async def test_an_interrupt_during_the_wait_for_a_parked_turns_trace_abandons_the_question(
        self, tmp_path, monkeypatch, how
    ):
        """The turn ended waiting for an approval and its trace is not stored yet. A stop then has nothing to
        cut, but it must not be dropped: once the turn is settled the question is abandoned, as a stop on a
        parked session does. Otherwise the session waits, and a later approval runs the gated tool."""
        import timbal.core.chat_session as chat_session_module

        monkeypatch.setattr(chat_session_module, "_TURN_RETRY_DELAY_S", 0.01)

        class Unreachable(SqliteTracingProvider):
            down = False

            @classmethod
            async def _store(cls, run_context):
                if cls.down:
                    raise OSError("the trace store is unreachable")
                await super()._store(run_context)

        provider = Unreachable.configured(_path=tmp_path / "timbal.db", down=True)
        store = SqliteChatSessionStore(tmp_path / "timbal.db")
        charge = Charge(released=True)
        session = ChatSession(gated_agent(provider, charge), session_store=store)
        session.send("pay")
        await records(session, "turn_started", 1)
        await ended_unsettled(session)  # the turn has ended on the approval; its last save keeps failing
        assert session.busy and not session.parked
        if how == "interrupt()":
            assert session.interrupt() is True
        else:
            session.send("never mind", mode="interrupt")
        provider.down = False
        expected = 2 if how == "interrupt()" else 3  # the parked turn, the one that cancels it (, the message)
        finished = await records(session, "turn_finished", expected)
        assert finished[0]["parked"] is True and finished[1]["status"]["code"] == "cancelled"
        assert not session.parked and session.state().get("pending_approvals") is None
        assert charge.gate.calls == 0  # the gated tool never ran, and nothing is left to approve
        if how != "interrupt()":
            assert root_outputs(session)[-1].output.collect_text() == "echo: never mind"
        await session.close()

    async def test_a_stop_taken_just_before_a_shutdown_still_abandons_the_question_after_it(self, db):
        """interrupt() said yes to a turn that had ended and was not settled yet; the session is stopped before
        it is. The stop is stored with the session, and the question is abandoned by whoever restores it."""
        provider, store = db
        charge = Charge(released=True)
        session = ChatSession(gated_agent(provider, charge), session_store=store)
        real_pump, taken, closing = session._pump, [], []

        async def pump(*args, **kwargs):
            output = await real_pump(*args, **kwargs)
            loop = asyncio.get_running_loop()
            # Both land once the turn's task is over and before the worker has settled the turn.
            loop.call_soon(lambda: taken.append(session.interrupt()))
            closing.append(loop.create_task(session.close(keep=True)))
            return output

        session._pump = pump
        session.send("pay")
        await until(lambda: closing)
        await closing[0]
        assert taken == [True]

        restored = await ChatSession.from_store(gated_agent(provider, charge), store, session.id)
        (finished,) = await records(restored, "turn_finished", 1, after=after_restore(restored))
        assert finished["status"]["code"] == "cancelled"
        assert not restored.parked and restored.state().get("pending_approvals") is None
        assert charge.gate.calls == 0
        await restored.close()

    @pytest.mark.parametrize("end", ["a crash", "a shutdown"])
    @pytest.mark.parametrize("trace", ["not stored", "stored"])
    async def test_a_stop_taken_during_that_wait_outlives_the_process(self, tmp_path, monkeypatch, trace, end):
        """The stop is stored as what it amounts to (a cancel for the question) when it is taken, not kept in
        mind until the turn is settled: the process may never get there."""
        import timbal.core.chat_session as chat_session_module

        monkeypatch.setattr(chat_session_module, "_TURN_RETRY_DELAY_S", 0.01)

        class Unreachable(SqliteTracingProvider):
            down = False

            @classmethod
            async def _store(cls, run_context):
                if cls.down:
                    raise OSError("the trace store is unreachable")
                await super()._store(run_context)

        provider = Unreachable.configured(_path=tmp_path / "timbal.db", down=True)
        store = SqliteChatSessionStore(tmp_path / "timbal.db")
        charge = Charge(released=True)
        session = ChatSession(gated_agent(provider, charge), session_store=store)
        real_trace_stored = session._trace_stored

        async def stored_and_no_further(context):
            await real_trace_stored(context)
            await asyncio.Event().wait()  # the process ends before the turn is settled

        session._trace_stored = stored_and_no_further
        session.send("pay")
        await records(session, "turn_started", 1)
        await ended_unsettled(session)  # the turn has ended on the approval; its last save keeps failing
        run_id = session.run_id
        assert session.interrupt() is True
        await session.flush()  # what the route waits for before it says so
        assert (await store.load(session.id)).resume  # the cancel, next to the turn in flight
        if trace == "stored":
            provider.down = False
            for _ in range(250):
                run = await session._load_run(run_id)
                if run is not None and run.root_span().t1 is not None:
                    break
                await asyncio.sleep(0.02)
            assert session.busy and not session.parked  # ended, stored, and not settled
        if end == "a crash":
            await kill(session)
        else:
            await session.close(keep=True)
        provider.down = False

        restored = await ChatSession.from_store(gated_agent(provider, charge), store, session.id)
        (finished,) = await records(restored, "turn_finished", 1, after=after_restore(restored))
        if trace == "stored":
            assert finished["status"]["code"] == "cancelled"  # the question, answered with the stored cancel
        await asyncio.sleep(0.1)
        assert not restored.parked and restored.state().get("pending_approvals") is None
        assert charge.gate.calls == 0  # nothing is left to approve, and the gated tool never ran
        await restored.close()


class HeldAtItsEnd(SqliteTracingProvider):
    """Parks one save of a run that has ended (its last one), so a test can cut the run right there. ``landed``:
    the write is made first, as when the cut reaches the run while the write is on its way back."""

    hold = False
    landed = False
    entered: asyncio.Event | None = None

    @classmethod
    async def _store(cls, run_context):
        root = run_context.root_span()
        if cls.hold and root is not None and root.t1 is not None:
            cls.hold = False
            if cls.landed:
                await super()._store(run_context)
            cls.entered.set()
            await asyncio.Event().wait()
        await super()._store(run_context)


class TestAStopThatReachesATurnAtItsEdges:
    """Found on a slow CI machine, then by stopping the session inside each of its writes (the sweeps): a cut
    that lands in a run's last save, after it ended on a question, or before a turn has done anything."""

    @staticmethod
    def held(tmp_path, *, landed=False):
        provider = HeldAtItsEnd.configured(
            _path=tmp_path / "timbal.db", hold=True, landed=landed, entered=asyncio.Event()
        )
        return provider, SqliteChatSessionStore(tmp_path / "timbal.db")

    @pytest.mark.parametrize("how", ["interrupt()", "an interrupt message"])
    async def test_in_the_last_save_of_a_run_that_ended_on_a_question(self, tmp_path, how):
        """The run has ended waiting for an approval and is storing that when the stop cancels it: no OUTPUT
        reaches the session. It is not a turn that never ran. It ended parked, and the stop is about its
        question; left as the head with nobody having settled it, the next turn would ask for the approval
        again, and granting it would run the tool."""
        provider, store = self.held(tmp_path)
        charge = Charge(released=True)
        session = ChatSession(gated_agent(provider, charge), session_store=store)
        session.send("pay")
        await asyncio.wait_for(provider.entered.wait(), 30)
        assert session.busy and not session.parked
        if how == "interrupt()":
            assert session.interrupt() is True
        else:
            session.send("never mind", mode="interrupt")
        expected = 2 if how == "interrupt()" else 3  # the parked turn, the one that cancels it (, the message)
        finished = await records(session, "turn_finished", expected)
        assert finished[0]["parked"] is True and finished[1]["status"]["code"] == "cancelled"
        assert [o.status.reason for o in root_outputs(session)][0] == "approval_required"  # its end is in the log
        assert not session.parked and session.state().get("pending_approvals") is None
        assert charge.gate.calls == 0
        if how != "interrupt()":
            assert root_outputs(session)[-1].output.collect_text() == "echo: never mind"
        stored = await session._load_run(finished[0]["run_id"])
        assert stored.root_span().t1 is not None  # the save that was cut was made again
        await session.close()

    async def test_and_the_process_dies_before_the_turn_is_settled(self, tmp_path):
        """The stop is stored with the turn when it is taken: the process that restores the session finds a
        run that ended on a question, and a turn its user had stopped."""
        provider, store = self.held(tmp_path, landed=True)
        charge = Charge(released=True)
        session = ChatSession(gated_agent(provider, charge), session_store=store)
        real_trace_stored = session._trace_stored

        async def no_further(context):
            await real_trace_stored(context)
            await asyncio.Event().wait()  # the process ends before the turn is settled

        session._trace_stored = no_further
        session.send("pay")
        await asyncio.wait_for(provider.entered.wait(), 30)
        assert session.interrupt() is True
        await session.flush()  # what the route waits for before it says so
        assert (await store.load(session.id)).in_flight["stopped"] is True
        await kill(session)

        restored = await ChatSession.from_store(gated_agent(provider, charge), store, session.id)
        (finished,) = await records(restored, "turn_finished", 1, after=after_restore(restored))
        assert finished["status"]["code"] == "cancelled"
        assert not restored.parked and restored.state().get("pending_approvals") is None
        assert charge.gate.calls == 0
        await restored.close()

    async def test_before_a_resume_has_used_its_answers(self, db):
        """The approval was given and the turn that acts on it is cut before it starts: the answers are not
        used, and the session would be parked on the same question again. The stop came last: abandoned."""
        provider, store = db
        charge = Charge(released=True)
        session, approval = await parked(provider, store, charge)
        await session.flush()
        real_save, held, entered = store.save, asyncio.Event(), asyncio.Event()

        async def save(snapshot, **kwargs):
            if snapshot.in_flight is not None and not entered.is_set():
                entered.set()
                await held.wait()  # the turn waits for this write before it starts
            return await real_save(snapshot, **kwargs)

        store.save = save
        session.resume({approval.approval_id: True})
        await asyncio.wait_for(entered.wait(), 5)
        assert session.interrupt() is True
        held.set()
        finished = await records(session, "turn_finished", 2)
        assert finished[1]["status"]["code"] == "cancelled"
        assert not session.parked and session.state().get("pending_approvals") is None
        assert charge.gate.calls == 0
        await session.close()

    async def test_before_the_turn_has_resolved_its_input(self, db, monkeypatch):
        """A turn stopped that early stores no memory, and stored no input either: the message it was to
        answer was gone from the conversation. It is stored as it was given, and the next turn reads it."""
        provider, store = db
        seen = []

        def model(messages):
            seen.append([m.collect_text() for m in messages if m.role == "user"])
            return handler(messages)

        session = ChatSession(gated_agent(provider, Charge(released=True), model_handler=model), session_store=store)
        session.send("one")
        await records(session, "turn_finished", 1)
        load_session, entered = RunContext.get_session, asyncio.Event()

        async def held_session(ctx, *args, **kwargs):
            if not entered.is_set():
                entered.set()
                await asyncio.Event().wait()
            return await load_session(ctx, *args, **kwargs)

        monkeypatch.setattr(RunContext, "get_session", held_session)
        session.send("two")
        await asyncio.wait_for(entered.wait(), 5)
        assert session.interrupt() is True
        finished = await records(session, "turn_finished", 2)
        assert finished[1]["status"] == {"code": "cancelled", "reason": "interrupted", "message": ""}
        session.send("three")
        await records(session, "turn_finished", 3)
        assert seen[-1] == ["one", "two", "three"]
        await session.close()

    async def test_before_the_turns_task_has_run_at_all(self, db):
        """The stop lands between the turn's task being created and its first step: none of its code runs, its
        own cleanup included. The message it had taken goes back in the queue, as for any turn that never
        started; it used to be gone."""
        provider, store = db
        session = ChatSession(make_agent(provider, Gate(released=True)), session_store=store)
        real_pump, taken = session._pump, []

        def pump(*args, **kwargs):
            turn = real_pump(*args, **kwargs)
            if not taken:
                # Queued ahead of the first step of the task this coroutine is about to be wrapped in.
                taken.append(None)
                asyncio.get_running_loop().call_soon(lambda: taken.__setitem__(0, session.interrupt()))
            return turn

        session._pump = pump
        session.send("hello")
        finished = await records(session, "turn_finished", 1)
        assert taken == [True]
        assert finished[0]["status"]["code"] == "success"
        assert root_outputs(session)[-1].output.collect_text() == "echo: hello"
        await session.close()


class TestAStopStoredWithATurnThatNeverRan:
    """A stop reaches a resume before any of its calls ran, and the process dies before the session settles
    the turn (live, settling turns the stop into a cancel answer). The snapshot holds the answers and the
    stop: whoever restores it must honour the stop, not the answers."""

    def _dies_before_settling(self, session):
        """The process ends where the worker would settle the cut turn: what the stop left in the snapshot
        (``in_flight.stopped``) is all a restore has to go on."""
        settling = asyncio.Event()

        def settle(*_args, **_kwargs):
            settling.set()
            raise asyncio.CancelledError  # the worker stops here, as a dying process does

        session._settle_turn = settle
        return settling

    async def _stopped_resume(self, provider, store, charge):
        session, approval = await parked(provider, store, charge)
        reached = asyncio.Event()

        async def pump(_context, _kwargs, _followup, _trigger, _in_flight, mark):
            # The store holds that the turn took the answers; the turn stops here, before the agent runs,
            # so none of the batch's calls was dispatched.
            await session._write_ahead(mark)
            reached.set()
            await asyncio.Event().wait()

        session._pump = pump
        session.resume({approval.approval_id: True})
        await asyncio.wait_for(reached.wait(), 30)
        return session, approval

    async def test_a_stop_of_a_resume_that_dispatched_nothing_outlives_the_process(self, db):
        provider, store = db
        charge = Charge(released=True)
        session, approval = await self._stopped_resume(provider, store, charge)
        settling = self._dies_before_settling(session)

        assert session.interrupt() is True  # the user stops it before anything of it ran
        await asyncio.wait_for(settling.wait(), 30)
        await session.flush()  # the stop is stored (what the route waits for before it says so)
        in_flight = (await store.load(session.id)).in_flight
        assert in_flight["stopped"] is True and in_flight["resume"] and not in_flight.get("dispatched")
        await kill(session)

        restored = await ChatSession.from_store(gated_agent(provider, charge), store, session.id)
        (finished,) = await records(restored, "turn_finished", 1, after=after_restore(restored))
        assert finished["status"]["code"] == "cancelled"
        assert charge.gate.calls == 0  # the answers went with the turn the user stopped
        assert not restored.parked and restored.state().get("pending_approvals") is None
        await restored.close()

    async def test_without_the_stop_the_same_turn_runs_again(self, db):
        """The control: a crash alone replays the answers — none of the calls they were for had started."""
        provider, store = db
        charge = Charge(released=True)
        session, _ = await self._stopped_resume(provider, store, charge)
        self._dies_before_settling(session)
        await session.flush()
        await kill(session)

        restored = await ChatSession.from_store(gated_agent(provider, charge), store, session.id)
        await records(restored, "turn_finished", 1, after=after_restore(restored))
        assert charge.gate.calls == 1  # the user approved it and nothing said otherwise
        await restored.close()


async def stored_before_the_turns_policy_was(store, session_id):
    """Rewrite the stored session as the format before ``in_flight.tool_policy`` wrote it: the turn's policy
    only in its wake (``in_flight.wake.tool_policy``) and in the saved options (``last_options``)."""
    snapshot = await store.load(session_id)
    assert "tool_policy" in snapshot.in_flight
    del snapshot.in_flight["tool_policy"]
    await store.save(snapshot, expect_version=snapshot.version)


async def recovery_turn(restored, timeout=30.0):
    """Wait for the restored session's recovery turn to end: its ``turn_started`` record."""

    async def follow():
        run_id = None
        async for item in restored.events(after_restore(restored)):
            if not isinstance(item, dict):
                continue
            if item.get("kind") == "turn_started" and item["trigger"]["kind"] == "recovery":
                run_id, started = item["run_id"], item
            elif item.get("kind") == "turn_finished" and item["run_id"] == run_id:
                return started

    return await asyncio.wait_for(follow(), timeout)


def prompt_of(session, run_id):
    (output,) = [output for output in root_outputs(session) if output.run_id == run_id]
    return Message.validate(output.input["prompt"]).collect_text()


OLDER = pytest.mark.parametrize("older", [False, True], ids=["policy stored with the turn", "older snapshot"])


class TestARecoveryTurnStaysWithinTheDeadTurnsPolicy:
    """A turn that ran under a narrow tool policy dies. The recovery turn that follows it is read-only, and
    read-only alone is wider than what the dead turn was allowed: it stays within both. Also for a session
    stored before the turn's policy was kept with it (``older``): the wake and the saved options still say."""

    def _agent(self, provider, gate, ran, plan):
        """``plan``: the tool each recovery turn calls, in order (shared across restores)."""

        def model(messages):
            if messages[-1].role == "tool":
                return "done"
            if "cut off by a server restart" in last_user_text(messages):
                return tool_call(plan.pop(0), f"call-recovery-{len(messages)}")
            return tool_call("lookup", "call-lookup")

        async def lookup() -> str:
            ran["lookup"] += 1
            return await gate.work()

        async def other_lookup() -> str:
            ran["other_lookup"] += 1
            return "other"

        async def confirm() -> str:
            ran["confirm"] += 1
            return await gate.work()

        return Agent(
            name="agent",
            model=TestModel(handler=model),
            tools=[
                Tool(name="lookup", handler=lookup, read_only=True),
                Tool(name="other_lookup", handler=other_lookup, read_only=True),
                Tool(name="confirm", handler=confirm, read_only=True, requires_approval=True),
            ],
            tracing_provider=provider,
        )

    async def _died_under(self, db, policy, calls_on_recovery, *, older=False):
        """A wake under ``policy`` dies with its tool running; the recovery turn calls ``calls_on_recovery``.
        What ran, and the recovery turn's prompt."""
        provider, store = db
        ran = {"lookup": 0, "other_lookup": 0, "confirm": 0}
        plan = [calls_on_recovery]
        session = ChatSession(self._agent(provider, Gate(), ran, plan), session_store=store)
        session.wake("check the balance", tool_policy=policy, silent_ok=True)
        await until(lambda: ran["lookup"] == 1)  # the wake's own tool is running: its call is stored
        await crash(session)
        provider.crashed = False
        if older:
            await stored_before_the_turns_policy_was(store, session.id)
        agent = self._agent(provider, Gate(released=True), ran, plan)
        restored = await ChatSession.from_store(agent, store, session.id)
        started = await recovery_turn(restored)
        prompt = prompt_of(restored, started["run_id"])
        await restored.close()
        return ran, prompt

    @OLDER
    async def test_it_cannot_call_a_read_only_tool_the_dead_turn_was_not_allowed(self, db, older):
        ran, _ = await self._died_under(db, ["lookup"], "other_lookup", older=older)
        assert ran["other_lookup"] == 0  # read-only, but outside the wake's allowlist

    @OLDER
    async def test_it_can_still_call_the_ones_it_was(self, db, older):
        ran, _ = await self._died_under(db, ["lookup", "other_lookup"], "other_lookup", older=older)
        assert ran["other_lookup"] == 1  # allowed there and read-only: the recovery turn may use it

    @OLDER
    async def test_a_callable_policy_does_not_come_back_wider(self, db, older):
        ran, _ = await self._died_under(db, lambda tool: tool.name == "lookup", "other_lookup", older=older)
        assert ran["other_lookup"] == 0  # a callable is not stored: what it let through is, at most

    @OLDER
    async def test_an_unrestricted_wake_recovers_read_only(self, db, older):
        ran, prompt = await self._died_under(db, None, "other_lookup", older=older)
        assert ran["other_lookup"] == 1 and "can only read" in prompt

    @OLDER
    async def test_it_is_told_it_can_only_read(self, db, older):
        """Narrowed or not, the recovery turn is read-only, and says so: a model that is not told looks for
        a way around the tools it does not see."""
        _, prompt = await self._died_under(db, ["lookup"], "lookup", older=older)
        assert "can only read" in prompt

    @OLDER
    async def test_a_dead_resume_of_a_parked_wake_keeps_the_wakes_policy(self, db, older):
        """The turn in flight is the *resume* of a wake that had parked: it has no wake of its own
        (``in_flight.wake`` is None), only the options the session kept for it."""
        provider, store = db
        ran = {"lookup": 0, "other_lookup": 0, "confirm": 0}
        plan = ["other_lookup"]

        def asks_first(agent):
            agent.model = TestModel(
                handler=lambda messages: (
                    "done"
                    if messages[-1].role == "tool"
                    else tool_call(plan.pop(0), "call-recovery")
                    if "cut off by a server restart" in last_user_text(messages)
                    else tool_call("confirm", "call-confirm")
                )
            )
            return agent

        session = ChatSession(asks_first(self._agent(provider, Gate(), ran, plan)), session_store=store)
        session.wake("check the balance", tool_policy=["confirm", "lookup"], silent_ok=True)
        await records(session, "turn_finished", 1)
        approval = next(e for e in session.log.peek()[0] if isinstance(e, ApprovalEvent))
        assert session.parked
        session.resume({approval.approval_id: True})
        await until(lambda: ran["confirm"] == 1)  # the approved call is running
        in_flight = (await store.load(session.id)).in_flight
        assert in_flight["wake"] is None and in_flight["resume"]
        await crash(session)
        provider.crashed = False
        if older:
            await stored_before_the_turns_policy_was(store, session.id)

        agent = asks_first(self._agent(provider, Gate(released=True), ran, plan))
        restored = await ChatSession.from_store(agent, store, session.id)
        await recovery_turn(restored)
        assert ran["other_lookup"] == 0 and ran["confirm"] == 1  # outside the wake's allowlist; nothing re-run
        await restored.close()

    async def test_a_second_crash_does_not_widen_it(self, db):
        """The recovery turn dies too. The one after it reports on a turn that was read-only *within* the
        first one's policy, and stays there."""
        provider, store = db
        ran = {"lookup": 0, "other_lookup": 0, "confirm": 0}
        plan = ["lookup", "other_lookup"]
        session = ChatSession(self._agent(provider, Gate(), ran, plan), session_store=store)
        session.wake("check the balance", tool_policy=["lookup"], silent_ok=True)
        await until(lambda: ran["lookup"] == 1)
        await crash(session)
        provider.crashed = False

        second = await ChatSession.from_store(self._agent(provider, Gate(), ran, plan), store, session.id)
        await until(lambda: ran["lookup"] == 2)  # the recovery turn's own (allowed) call is running
        await crash(second)
        provider.crashed = False

        third = await ChatSession.from_store(self._agent(provider, Gate(released=True), ran, plan), store, session.id)
        await recovery_turn(third)
        assert plan == [] and ran["other_lookup"] == 0
        await third.close()


class TestOlderSnapshots:
    async def test_a_turn_stored_before_its_policy_was_recovers_read_only(self, db):
        """`in_flight.tool_policy` is new: a snapshot written without it recovers as it did before."""
        provider, store = db
        gate = Gate()
        session = ChatSession(make_agent(provider, gate), session_store=store)
        session.send("start")
        await asyncio.wait_for(gate.entered.wait(), 30)
        await crash(session)
        provider.crashed = False

        snapshot = await store.load(session.id)
        assert snapshot.in_flight["tool_policy"] is None  # a user turn: no policy of its own
        await stored_before_the_turns_policy_was(store, session.id)

        restored = await ChatSession.from_store(make_agent(provider, Gate(released=True)), store, session.id)
        (started,) = await records(restored, "turn_started", 1, after=after_restore(restored))
        await records(restored, "turn_finished", 1, after=after_restore(restored))
        # The recovery turn ran (read-only, as before the field existed) and the wake is spent.
        assert started["trigger"]["kind"] == "recovery" and restored._wakes == []
        await restored.close()


async def worker_gone(session):
    """The process is about to die between storing what it was told and acting on it: from here on its
    worker takes nothing, so what a test sends next is stored and never run by this process."""
    session._hold_worker = True
    worker, session._worker = session._worker, None
    if worker is not None:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)


def priced_agent(provider, charged):
    """``charge(amount)`` behind an approval; the model always asks for 5."""

    def model(messages):
        if messages[-1].role == "tool":
            return "done"
        call = ToolUseContent(id=f"call-{len(messages)}", name="charge", input={"amount": 5})
        return Message(role="assistant", content=[call], stop_reason="tool_use")

    def charge(amount: int) -> str:
        charged.append(amount)
        return f"charged {amount}"

    return Agent(
        name="agent",
        model=TestModel(handler=model),
        tools=[Tool(name="charge", handler=charge, requires_approval=True)],
        tracing_provider=provider,
    )


async def parked_on_a_price(provider, store, charged):
    session = ChatSession(priced_agent(provider, charged), session_store=store)
    session.send("pay")
    await records(session, "turn_finished", 1)
    approval = next(e for e in session.log.peek()[0] if isinstance(e, ApprovalEvent))
    return session, approval


# Every form the approval gate takes, and what the tool then does with the model's charge(amount=5).
ANSWERS = [
    pytest.param(lambda: True, [5], "success", id="True"),
    pytest.param(lambda: {"approved": True, "override_input": {"amount": 9}}, [9], "success", id="a JSON object"),
    pytest.param(lambda: ApprovalResolution(approved=True), [5], "success", id="ApprovalResolution"),
    pytest.param(
        lambda: ApprovalResolution(approved=True, override_input={"amount": 7}, approver_id="ana", comment="ok"),
        [7],
        "success",
        id="ApprovalResolution with an input override",
    ),
    pytest.param(lambda: ApprovalResolution(approved=False, reason="too much"), [], "success", id="a typed decline"),
    pytest.param(lambda: Cancel(reason="closed the dialog"), [], "cancelled", id="Cancel"),
    pytest.param(lambda: {"type": "timbal.cancel", "reason": "closed"}, [], "cancelled", id="a JSON cancel"),
]


class TestAnswersAreStoredAsTheGateReadsThem:
    """``resume()`` takes what ``resume=`` takes: booleans, JSON objects, ``ApprovalResolution``, ``Cancel``.
    A durable session stores the answers twice (queued, then with the turn that took them); whichever copy
    a restart finds must mean to the approval gate what the caller said."""

    @pytest.mark.parametrize("answer, charged_after, status", ANSWERS)
    async def test_an_answer_stored_and_not_yet_taken(self, db, answer, charged_after, status):
        provider, store = db
        charged: list[int] = []
        session, approval = await parked_on_a_price(provider, store, charged)
        await worker_gone(session)
        session.resume({approval.approval_id: answer()})
        await session.flush()  # acknowledged: the store holds it
        assert (await store.load(session.id)).resume is not None
        await kill(session)

        restored = await ChatSession.from_store(priced_agent(provider, charged), store, session.id)
        (finished,) = await records(restored, "turn_finished", 1, after=after_restore(restored))
        assert finished["status"]["code"] == status and charged == charged_after
        assert not restored.parked
        await restored.close()

    @pytest.mark.parametrize("answer, charged_after, status", ANSWERS)
    async def test_an_answer_a_turn_took_and_died_before_using(self, db, answer, charged_after, status):
        """The other copy: the turn took the answers, the store holds that, and none of its calls had started.
        The answers go back in the queue from ``in_flight.resume``."""
        provider, store = db
        charged: list[int] = []
        session, approval = await parked_on_a_price(provider, store, charged)
        reached = asyncio.Event()

        async def pump(_context, _kwargs, _followup, _trigger, _in_flight, mark):
            await session._write_ahead(mark)
            reached.set()
            await asyncio.Event().wait()  # the process dies before the agent runs

        session._pump = pump
        session.resume({approval.approval_id: answer()})
        await asyncio.wait_for(reached.wait(), 30)
        await session.flush()
        stored = await store.load(session.id)
        assert stored.resume is None and stored.in_flight["resume"] is not None
        await kill(session)

        restored = await ChatSession.from_store(priced_agent(provider, charged), store, session.id)
        (finished,) = await records(restored, "turn_finished", 1, after=after_restore(restored))
        assert finished["status"]["code"] == status and charged == charged_after
        await restored.close()

    async def test_a_typed_approval_is_stored_as_the_json_object_the_gate_accepts(self, db):
        provider, store = db
        session, approval = await parked_on_a_price(provider, store, [])
        await worker_gone(session)
        answer = ApprovalResolution(
            approved=True, scope="session", override_input={"amount": 7}, approver_id="ana", comment="fine"
        )
        session.resume({approval.approval_id: answer})
        await session.flush()
        stored = (await store.load(session.id)).resume[approval.approval_id]
        # Not its repr: a JSON object, every field kept, and the same decision when the gate reads it back.
        assert isinstance(stored, dict) and ApprovalResolution.model_validate(stored) == answer
        assert stored["decided_at"] == answer.decided_at  # the decision's own time, not the restore's
        await kill(session)

    async def test_dont_ask_again_survives_the_restart(self, db):
        provider, store = db
        charged: list[int] = []
        session, approval = await parked_on_a_price(provider, store, charged)
        await worker_gone(session)
        session.resume({approval.approval_id: ApprovalResolution(approved=True, scope="session")})
        await session.flush()
        await kill(session)

        restored = await ChatSession.from_store(priced_agent(provider, charged), store, session.id)
        await records(restored, "turn_finished", 1, after=after_restore(restored))
        restored.send("pay again")
        (_, second) = await records(restored, "turn_finished", 2, after=after_restore(restored))
        # scope="session" was kept: the second charge is not asked about.
        assert second["status"]["code"] == "success" and not restored.parked and charged == [5, 5]
        await restored.close()

    @pytest.mark.parametrize(
        "answer",
        [
            object(),
            {"approved": True, "metadata": {"seen": {1, 2}}},
            {"approved": True, "metadata": {"at": object()}},
            {"approved": True, "metadata": {"pair": (1, 2)}},
            {"approved": True, "metadata": {1: "one"}},
            {"approved": True, "metadata": {"ratio": float("nan")}},
            {"approved": True, "override_input": {"amount": float("inf")}},
            ApprovalResolution(approved=True, metadata={"by": object()}),
            ApprovalResolution(approved=True, metadata={"ratio": float("nan")}),
            ApprovalResolution(approved=True, override_input={"amount": float("inf")}),
            ApprovalResolution(approved=True, metadata={"seen": {1, 2}}),
        ],
        ids=[
            "an object",
            "a set inside",
            "an object inside",
            "a tuple inside",
            "a number key",
            "NaN",
            "Infinity",
            "a typed approval holding an object",
            "a typed approval holding NaN",
            "a typed approval overriding an input with Infinity",
            "a typed approval holding a set",
        ],
    )
    async def test_an_answer_the_store_cannot_hold_is_refused_not_stored_as_text(self, db, answer):
        provider, store = db
        charged: list[int] = []
        session, approval = await parked_on_a_price(provider, store, charged)
        with pytest.raises(TypeError, match="cannot be stored"):
            session.resume({approval.approval_id: answer})
        # Refused whole: nothing queued, the question still open, and a storable answer goes through.
        assert session._resume is None and session.parked
        await session.flush()
        assert (await store.load(session.id)).resume is None
        session.resume({approval.approval_id: True})
        await records(session, "turn_finished", 2)
        assert charged == [5]
        await session.close()

    async def test_a_session_without_a_store_takes_typed_answers_as_before(self, db):
        provider, _ = db
        charged: list[int] = []
        session, approval = await parked_on_a_price(provider, None, charged)
        session.resume({approval.approval_id: ApprovalResolution(approved=True, override_input={"amount": 3})})
        await records(session, "turn_finished", 2)
        assert charged == [3]
        await session.close()

    async def test_the_same_answer_sent_again_after_a_restore_is_not_a_conflict(self, db):
        provider, store = db
        charged: list[int] = []
        session, approval = await parked_on_a_price(provider, store, charged)
        await worker_gone(session)
        answer = ApprovalResolution(approved=True)
        session.resume({approval.approval_id: answer})
        await session.flush()
        await kill(session)

        restored = await ChatSession.from_store(priced_agent(provider, charged), store, session.id)
        restored.resume({approval.approval_id: answer})  # a caller that did not hear back, trying again
        with pytest.raises(ValueError, match="Already answered differently"):
            restored.resume({approval.approval_id: ApprovalResolution(approved=False)})
        await records(restored, "turn_finished", 1, after=after_restore(restored))
        assert charged == [5]
        await restored.close()

    async def test_an_answer_can_be_taken_back_after_a_restore(self, db):
        """What the route does when it cannot confirm an answer was stored: the answer it withdraws is the
        one it sent, also once the queued copy has been through the store."""
        provider, store = db
        charged: list[int] = []
        session, approval = await parked_on_a_price(provider, store, charged)
        await worker_gone(session)
        answer = ApprovalResolution(approved=True)
        session.resume({approval.approval_id: answer})
        await session.flush()
        await kill(session)

        restored = await ChatSession.from_store(priced_agent(provider, charged), store, session.id)
        assert restored.withdraw_answers({approval.approval_id: ApprovalResolution(approved=False)}) is False
        assert restored.withdraw_answers({"another-id": answer}) is False
        assert restored.withdraw_answers({approval.approval_id: answer}) is True
        await restored.flush()
        assert (await store.load(session.id)).resume is None and restored.parked and charged == []
        await restored.close()

    @pytest.mark.parametrize(
        "answer, approves",
        [
            (True, True),
            ({"approved": True}, True),
            (ApprovalResolution(approved=True), True),
            (False, False),
            ({"approved": False}, False),
            (ApprovalResolution(approved=False), False),
            (Cancel(reason="no"), False),
            ({"type": "timbal.cancel"}, False),
            ({}, False),
        ],
        ids=_stable_id,
    )
    async def test_the_recovery_notice_names_a_call_only_when_the_answer_approves_it(self, db, answer, approves):
        """``in_flight.approved``: the gated calls a dead resume had been told to run. A typed approval is
        one; a cancel sent as JSON is not."""
        provider, store = db
        session, approval = await parked_on_a_price(provider, store, [])
        calls = session._approved_calls({approval.approval_id: answer})
        assert [call.split("(")[0] for call in calls] == (["charge"] if approves else [])
        await session.close()


class TestASnapshotThatCannotBeWritten:
    async def test_it_is_reported_and_retried_never_stored_as_something_else(self, db):
        """The backstop behind the checks at the door: state JSON cannot carry fails its save. The caller
        waiting on it is told (``SessionNotStored``), the store keeps the last good state, and the next save
        goes through once the state can be written."""
        provider, store = db
        session = ChatSession(make_agent(provider, Gate(released=True)), session_store=store)
        session.send("hello")
        await records(session, "turn_finished", 1)
        await session.flush()
        before = await store.load(session.id)

        good, session.last_status = session.last_status, {"code": object()}
        session.persist()
        with pytest.raises(SessionNotStored):
            await session.flush()
        assert (await store.load(session.id)).to_json() == before.to_json()

        session.last_status = good
        await session.flush()  # the write that failed, once more
        assert (await store.load(session.id)).version > before.version
        await session.close()

    @pytest.mark.parametrize(
        "value",
        [float("nan"), float("inf"), float("-inf"), object(), {1, 2}, b"raw", (1, 2), "ok", 1.5, None, {"a": [1]}],
        ids=_stable_id,
    )
    def test_the_check_and_the_write_use_one_encoder(self, value):
        """``storable`` is what the session asks before it keeps a value; ``to_json`` is what writes it. They
        must agree on every value, or one the check let through fails every save after it."""
        snapshot = SessionSnapshot(session_id="s", agent="agent", config={"kwargs": {"x": value}})
        try:
            snapshot.to_json()
            written = True
        except (TypeError, ValueError):
            written = False
        assert storable(value) is written

    async def test_awkward_settings_are_stored_as_what_they_mean(self, db):
        """Values a caller may pass where a bool or a number is meant (a numpy scalar, a Fraction): the
        session works with them live, so it must store itself with them too."""
        from fractions import Fraction

        provider, store = db
        gate = Gate()
        agent = make_agent(provider, gate)
        session = ChatSession(agent, session_store=store, allow_silence=Fraction(1), collect_window=Fraction(1, 2))
        session.send("start")  # a turn runs, so the wake below stays queued
        await asyncio.wait_for(gate.entered.wait(), 30)
        session.wake("heartbeat", silent_ok=Fraction(1))
        await session.flush()
        snapshot = await store.load(session.id)
        assert snapshot.config["allow_silence"] is True and snapshot.config["collect_window"] == 0.5
        assert [wake["silent_ok"] for wake in snapshot.wakes] == [True]
        await session.close()

    async def test_a_wake_key_that_is_not_text_is_refused(self, db):
        provider, store = db
        session = ChatSession(make_agent(provider, Gate(released=True)), session_store=store)
        for key in (5, ["a"], {"k": 1}):
            with pytest.raises(TypeError, match="key"):
                session.wake("heartbeat", key=key)
        assert session._wakes == []
        await session.close()

    async def test_a_turn_argument_json_cannot_carry_is_asked_for_again(self, db):
        """NaN is one more value a snapshot cannot hold: named as unstored, like a callable, so the session
        still saves and ``from_store`` asks for the argument again."""
        provider, store = db
        agent = make_agent(provider, Gate(released=True))
        session = ChatSession(agent, session_store=store, max_tokens=64, temperature=float("nan"))
        session.persist()
        await session.flush()
        config = (await store.load(session.id)).config
        assert config["kwargs"] == {"max_tokens": 64} and config["kwargs_unstored"] == ["temperature"]
        await session.close(keep=True)
        with pytest.raises(ValueError, match="temperature"):
            await ChatSession.from_store(agent, store, session.id)

    @pytest.mark.parametrize(
        "prompt",
        [
            Message(role="user", content=[TextContent(text="pay")], metadata={"seen": {1, 2}}),
            Message(role="user", content=[TextContent(text="pay")], metadata={"score": float("nan")}),
            Message(role="user", content=[TextContent(text="pay")], metadata={"score": float("inf")}),
            {"role": "user", "content": [{"type": "text", "text": "pay"}], "metadata": {"at": object()}},
            {"role": "user", "content": [{"type": "text", "text": "pay"}], "metadata": {"score": float("-inf")}},
        ],
        ids=["a message", "a message with NaN", "a message with Infinity", "a stored prompt", "one with Infinity"],
    )
    async def test_a_prompt_json_cannot_carry_whole_is_kept_as_its_text(self, db, prompt):
        """The last prompt is re-sent with a resume and dropped by the agent: metadata JSON cannot carry (a
        session continuing a caller's own run) must not stop the session from storing itself."""
        provider, store = db
        session = ChatSession(make_agent(provider, Gate(released=True)), session_store=store)
        session.send("hello")
        await records(session, "turn_finished", 1)
        session._last_prompt = prompt
        session.persist()
        await session.flush()
        stored = (await store.load(session.id)).last_prompt
        assert Message.validate(stored).collect_text() == "pay"  # stored at all: the flush went through
        await session.close()


def hang(store):
    """Make ``store`` hang (not fail) until the returned event is set: its writes and its lease calls wait."""
    back = asyncio.Event()

    def stalled(real):
        async def call(*args, **kwargs):
            await back.wait()
            return await real(*args, **kwargs)

        return call

    for name in ("save", "delete", "acquire_lease", "release_lease"):
        setattr(store, name, stalled(getattr(store, name)))
    return back


class TestAStoreThatHangs:
    """A store that neither takes a write nor refuses it (a stalled network mount). Nothing here may wait
    for it without a limit when the caller set one, and what was told to a sender must hold whenever the
    write finally lands."""

    async def _idle_session(self, db):
        provider, store = db
        session = ChatSession(make_agent(provider, Gate(released=True)), session_store=store)
        session.send("hello")
        await records(session, "turn_finished", 1)
        await session.flush()
        return session, store

    @pytest.mark.parametrize("keep", [True, False])
    async def test_a_close_given_a_limit_does_not_wait_past_it(self, db, keep):
        session, store = await self._idle_session(db)
        before = await store.load(session.id)
        back = hang(store)
        await asyncio.wait_for(session.close(keep=keep, store_timeout=0.2), 30)
        assert session.closed and session.close_timed_out
        # Kept: the row stands as it was, for a restore. Not kept: the caller is told the row may still be open.
        assert (session.close_error is None) is keep
        assert session.handed_off is keep
        await until(lambda: session._saver is None or session._saver.done())  # nothing of it writes later
        back.set()
        assert (await store.load(session.id)).version == before.version

    async def test_a_close_without_a_limit_waits_as_before(self, db):
        session, store = await self._idle_session(db)
        back = hang(store)
        closing = asyncio.ensure_future(session.close(keep=True))
        await asyncio.sleep(0.2)
        assert not closing.done() and not session.close_timed_out
        back.set()
        await asyncio.wait_for(closing, 30)
        assert (await store.load(session.id)).closed is False

    async def test_a_change_is_known_to_be_in_the_write_that_is_out_or_not(self, db):
        """What a route that stopped waiting needs to know: a change in the write that is out may land (keep
        it, do not ask for a retry); one made after that write started cannot have (safe to take back)."""
        session, store = await self._idle_session(db)
        held = held_before_its_turn(session)  # no turn takes the messages: this is about the queue
        assert session.may_be_stored(session.mark()) and session.stored_since(session.mark())
        back = hang(store)
        first = session.send("one", mode="followup")
        in_the_write = session.mark()
        await asyncio.sleep(0.05)  # the save loop has taken its snapshot and is waiting on the store
        second = session.send("two", mode="followup")
        after_it = session.mark()
        assert not session.stored_since(in_the_write) and session.may_be_stored(in_the_write)
        assert not session.stored_since(after_it) and not session.may_be_stored(after_it)
        assert session.withdraw(second.id) is True

        back.set()
        await session.flush()
        stored = await store.load(session.id)
        assert [event["id"] for event in stored.pending] == [first.id]  # "two" never reached the store
        assert session.stored_since(in_the_write) and session.may_be_stored(after_it)
        held.set()
        await records(session, "turn_finished", 2)
        await session.close()

    async def test_a_lease_the_store_does_not_renew_is_given_up(self, db):
        """A renewal that gets no answer is one that did not happen. After a whole lease without one, another
        process may hold the session: this copy stops, as it does when renewals fail."""
        provider, store = db
        # Long enough to still be held when the turn below is over, on a stalled runner too (see KEPT_TTL).
        store.lease_ttl_s = TestLeaseRunsOut.KEPT_TTL
        session = ChatSession(make_agent(provider, Gate(released=True)), session_store=store)
        session.send("hello")
        await records(session, "turn_finished", 1)
        await session.flush()
        hang(store)
        await asyncio.wait_for(session.wait_closed(), 30)
        assert session.lost and session.handed_off

    def test_opening_one_file_does_not_hold_up_another(self, tmp_path, monkeypatch):
        """The registry of open databases has one lock for the whole process. A file that cannot be opened
        now (its mount has stalled) must not hold it: every other file's saves go through it too."""
        import sqlite3
        import threading

        from timbal.state.tracing.providers import sqlite as sqlite_module

        opening, release, opened = threading.Event(), threading.Event(), threading.Event()
        real = sqlite3.connect

        def connect(path, *args, **kwargs):
            if "stalled" in str(path):
                opening.set()
                release.wait(30)
            return real(path, *args, **kwargs)

        monkeypatch.setattr(sqlite_module.sqlite3, "connect", connect)
        stalled = threading.Thread(target=sqlite_module.database, args=(tmp_path / "stalled.db",), daemon=True)
        stalled.start()
        assert opening.wait(30)
        healthy = threading.Thread(
            target=lambda: (sqlite_module.database(tmp_path / "healthy.db"), opened.set()), daemon=True
        )
        healthy.start()
        try:
            assert opened.wait(10), "a file that could not be opened held up another one"
        finally:
            release.set()
            stalled.join(30)
            healthy.join(30)

    async def test_a_hung_call_keeps_one_thread_of_its_own(self, tmp_path):
        """Callers that stop waiting for a hung store must not leave a thread behind each: on the loop's shared
        pool they add up until nothing else in the process gets one. The store's calls run on one thread
        per file, a daemon, and a call nobody waits for any more is not run when its turn comes."""
        import threading

        store = SqliteChatSessionStore(tmp_path / "one-thread-of-its-own.db")
        await store.save(SessionSnapshot(session_id="s1", agent="agent"), expect_version=None)
        stuck, release = threading.Event(), threading.Event()
        ran: list[Any] = []

        def hangs(_conn):
            stuck.set()
            release.wait(30)
            ran.append("first")

        first = asyncio.ensure_future(store._run(hangs))
        await until(stuck.is_set)
        for attempt in range(48):  # more than the shared pool has threads
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(store._run(lambda _conn, attempt=attempt: ran.append(attempt)), 0.01)
        kept = asyncio.ensure_future(store._run(lambda _conn: ran.append("kept")))
        try:
            # The shared pool is free: with 48 abandoned calls stuck on it, this would never get a thread.
            assert await asyncio.wait_for(asyncio.to_thread(lambda: "free"), 10) == "free"
            mine = [t for t in threading.enumerate() if t.name == "timbal-chat-store:one-thread-of-its-own.db"]
            assert len(mine) == 1 and mine[0].daemon  # one, and it does not hold the process open at exit
        finally:
            release.set()
        await asyncio.wait_for(asyncio.gather(first, kept), 30)
        assert ran == ["first", "kept"]  # in order, and none of the abandoned ones

    async def test_the_stores_thread_ends_when_idle_and_comes_back(self, tmp_path, monkeypatch):
        import threading

        from timbal.state import chat_store

        monkeypatch.setattr(chat_store._StoreThread, "IDLE_S", 0.05)
        store = SqliteChatSessionStore(tmp_path / "idle-thread.db")
        await store.save(SessionSnapshot(session_id="s1", agent="agent"), expect_version=None)

        def alive():
            return [t for t in threading.enumerate() if t.name == "timbal-chat-store:idle-thread.db"]

        await until(lambda: not alive())
        assert (await store.load("s1")).session_id == "s1"  # started again by the next call
        assert await store.list_ids() == ["s1"]

    @pytest.mark.parametrize("ended_by", ["an error", "a cancellation"])
    async def test_a_restore_that_fails_does_not_wait_past_its_lease_to_let_it_go(self, db, ended_by):
        """It lets go of the lease it took, through the store. One that does not answer is waited for only
        while the lease lasts: after that there is nothing left to let go of."""
        provider, store = db
        session, _ = await self._idle_session(db)
        await session.close(keep=True)
        store.lease_ttl_s = 0.4
        real_load, real_release = store.load, store.release_lease
        back, reading = asyncio.Event(), asyncio.Event()

        async def load(session_id):
            reading.set()
            if ended_by == "an error":
                raise OSError("unreadable")
            await back.wait()
            return await real_load(session_id)

        async def release(*args):
            await back.wait()
            return await real_release(*args)

        store.load, store.release_lease = load, release
        agent = make_agent(provider, Gate(released=True))
        restore = asyncio.ensure_future(ChatSession.from_store(agent, store, session.id))
        try:
            await asyncio.wait_for(reading.wait(), 30)
            if ended_by == "a cancellation":
                restore.cancel()
            # Not a second cancellation, not the store coming back: the lease running out ends the wait.
            done, _ = await asyncio.wait({restore}, timeout=30)
            assert done
            if ended_by == "an error":
                assert isinstance(restore.exception(), OSError)
            else:
                assert restore.cancelled()
        finally:
            back.set()
            restore.cancel()
        store.load, store.release_lease = real_load, real_release
        restored = await ChatSession.from_store(agent, store, session.id)  # nothing was written, nothing is held
        await restored.close()

    async def test_a_session_that_lost_ownership_closes_without_waiting_for_the_store(self, db):
        """Another process serves it: this copy writes nothing more, so it has nothing to wait for. A server
        waits for this close before it restores the session (``wait_closed``)."""
        session, store = await self._idle_session(db)
        hang(store)
        session.send("one", mode="followup")  # a write is out, and will not come back
        await asyncio.sleep(0.05)
        session._lose("another process took the session")
        await asyncio.wait_for(session.wait_closed(), 30)
        assert session.closed and session.lost and session.handed_off
        saver = session._saver
        if saver is not None and not saver.done():
            saver.cancel()
            await asyncio.gather(saver, return_exceptions=True)


class TestStoredAnswers:
    def test_an_answer_shaped_like_a_stored_cancel_comes_back_as_itself(self):
        """A snapshot holds a Cancel as ``{"$cancel": reason}``. A question may ask for any object, one with
        that key too: after a restart its answer must be the object, not a Cancel that would end the run."""
        import json

        from timbal.core.chat_session import ANSWER_FORMAT, _dump_resume, _load_resume, _same_answer

        answers = {
            "ask-1": {"$cancel": "a field of the survey"},
            "ask-2": {"$answer": 1},
            "ask-3": Cancel(reason="stop"),
            "ask-4": {"x": 1},
            "approval-1": True,
        }
        back = _load_resume(json.loads(json.dumps(_dump_resume(answers))), ANSWER_FORMAT)
        assert back["ask-1"] == {"$cancel": "a field of the survey"} and back["ask-2"] == {"$answer": 1}
        assert isinstance(back["ask-3"], Cancel) and back["ask-3"].reason == "stop"
        assert back["ask-4"] == {"x": 1} and back["approval-1"] is True
        assert all(_same_answer(answers[key], back[key]) for key in answers)

    def test_an_older_snapshots_answers_are_read_as_it_wrote_them(self):
        """A snapshot written before answers were wrapped holds a question's answer as it was given, ``$answer``
        key and all: it is read back as that object (the snapshot says which rules wrote it)."""
        from timbal.core.chat_session import _load_resume
        from timbal.state.chat_store import SessionSnapshot

        written_before = (
            '{"session_id": "s", "agent": "a", "config": {}, "head": null, "turns": 1, "last_status": null, '
            '"parked": true, "last_prompt": null, "last_options": null, "in_flight": {"resume": '
            '{"ask-1": {"$answer": "yes"}, "ask-2": {"$cancel": "stop"}}}, "pending": [], "wakes": [], '
            '"resume": {"ask-1": {"$answer": "yes"}, "ask-2": {"$cancel": "stop"}}, "log_cursor": 0, '
            '"closed": false, "version": 3}'
        )
        snapshot = SessionSnapshot.from_json(written_before)
        for stored, written_by in (
            (snapshot.resume, snapshot.answer_format),
            (snapshot.in_flight["resume"], snapshot.in_flight.get("answer_format")),
        ):
            back = _load_resume(stored, written_by)
            assert back["ask-1"] == {"$answer": "yes"}
            assert isinstance(back["ask-2"], Cancel) and back["ask-2"].reason == "stop"
