"""Durable chat sessions: the session layer stored ahead of every change, restored after a restart or a crash."""

import asyncio

import pytest
from timbal import Agent, Tool
from timbal.core.chat_session import ChatSession
from timbal.core.test_model import TestModel
from timbal.state import set_call_id, set_parent_call_id, set_run_context
from timbal.state.background import forget_idle_background_store
from timbal.state.chat_store import (
    PROCESS_OWNER,
    InMemoryChatSessionStore,
    SessionLeased,
    SessionSnapshot,
    SnapshotConflict,
    SqliteChatSessionStore,
)
from timbal.state.tracing.providers import SqliteTracingProvider
from timbal.types.content import ToolUseContent
from timbal.types.events import ApprovalEvent, OutputEvent
from timbal.types.message import Message


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


async def records(session, kind, count, timeout=5.0, after=0):
    found = []

    async def collect():
        async for item in session.events(after):
            if isinstance(item, dict) and item.get("kind") == kind:
                found.append(item)
                if len(found) >= count:
                    return

    await asyncio.wait_for(collect(), timeout)
    return found


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


class TestEdges:
    async def test_a_queued_message_waits_for_the_recovery_turn(self, db):
        provider, store = db
        gate = Gate()
        session = ChatSession(make_agent(provider, gate), session_store=store)
        session.send("start")
        await asyncio.wait_for(gate.entered.wait(), 3)
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
        await asyncio.wait_for(first.gate.entered.wait(), 3)
        await crash(session)
        provider.crashed = False

        again = Charge(released=True)
        restored = await ChatSession.from_store(gated_agent(provider, again), store, session.id)
        started = await records(restored, "turn_started", 2, after=after_restore(restored))
        await records(restored, "turn_finished", 2, after=after_restore(restored))
        # The stored answers go again, but the approval stays claimed by the dead run: no second charge.
        assert [s["trigger"]["kind"] for s in started] == ["resume", "recovery"]
        text = Message.validate(root_outputs(restored)[1].input["prompt"]).collect_text()
        assert "Started but not confirmed" in text and "charge(" in text
        assert first.gate.calls == 1 and again.gate.calls == 0
        assert not restored.parked
        await restored.close()

    async def test_without_approval_claims_a_dead_resume_parks_again(self, tmp_path):
        from timbal.state.tracing.providers.base import TracingProvider

        class NoClaims(CrashableSqlite):
            claim_approval = TracingProvider.claim_approval

        provider = NoClaims.configured(_path=tmp_path / "timbal.db", crashed=False)
        store = SqliteChatSessionStore(tmp_path / "timbal.db")
        first = Charge()
        session, approval = await parked(provider, store, first)
        session.resume({approval.approval_id: True})
        await asyncio.wait_for(first.gate.entered.wait(), 3)
        await crash(session)
        provider.crashed = False

        again = Charge(released=True)
        restored = await ChatSession.from_store(gated_agent(provider, again), store, session.id)
        await asyncio.sleep(0.1)
        assert restored.parked and restored._resume is None and again.gate.calls == 0
        assert restored.state()["pending_approvals"][0]["approval_id"] == approval.approval_id
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
        await asyncio.wait_for(gate.entered.wait(), 3)
        await crash(session)
        provider.crashed = False

        second = await ChatSession.from_store(agent(Gate(), checking), store, session.id)
        await asyncio.wait_for(checking.entered.wait(), 3)  # the recovery turn is checking: crash again
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
        await asyncio.wait_for(gate.entered.wait(), 3)
        session.wake("routine", key="r", tool_policy=["work"])
        session.wake("custom", key="c", tool_policy=lambda _tool: True)
        await crash(session)
        provider.crashed = False

        restored = await ChatSession.from_store(make_agent(provider, Gate(released=True)), store, session.id)
        wakes = [(w.kind, w.key, w.tool_policy) for w in restored._wakes]
        assert wakes == [
            ("recovery", "recovery", "read_only"),
            ("wake", "r", frozenset({"work"})),
            ("wake", "c", "read_only"),  # a callable cannot be stored: the safe policy comes back
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

    async def test_losing_the_lease_stops_the_session_without_writing(self, db):
        provider, store = db
        store.lease_ttl_s = 0.15
        session = ChatSession(make_agent(provider, Gate(released=True)), session_store=store)
        session.send("hi")
        await records(session, "turn_finished", 1)
        await session.flush()
        session._lease_task.cancel()  # this process stalls past its lease...
        await asyncio.sleep(0.2)
        await store.acquire_lease(session.id, "elsewhere:1:x")  # ...and another host takes the session
        version = (await store.load(session.id)).version
        session._lease_task = asyncio.get_running_loop().create_task(session._renew_lease())
        for _ in range(50):
            if session.closed:
                break
            await asyncio.sleep(0.02)
        assert session.closed and session.state()["durable"] is False
        assert (await store.load(session.id)).version == version  # not written over

    async def test_an_owner_whose_lease_was_taken_cannot_write_before_the_new_owner_does(self, db):
        """Split brain: the lease lapsed (or renewing it failed) and another process took it, but has not
        written yet. The old owner's next save must fail, not the new owner's."""
        provider, store = db
        session = ChatSession(make_agent(provider, Gate(released=True)), session_store=store)
        session.send("hi")
        await records(session, "turn_finished", 1)
        await session.flush()
        session._lease_task.cancel()  # renewals stopped: nothing notices the takeover but the save
        await store.release_lease(session.id, PROCESS_OWNER)  # as good as lapsed
        await store.acquire_lease(session.id, "elsewhere:1:x")
        version = (await store.load(session.id)).version
        session.wake("ping", key="p")
        for _ in range(50):
            if session.closed:
                break
            await asyncio.sleep(0.02)
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
        for _ in range(50):
            if session.closed:
                break
            await asyncio.sleep(0.02)
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
        await store.acquire_lease("s1", PROCESS_OWNER)
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


class TestWriteAhead:
    async def test_a_message_is_stored_before_its_receipt_is_trusted(self, db):
        provider, store = db
        gate = Gate()
        async with ChatSession(make_agent(provider, gate), session_store=store) as session:
            session.send("start")
            await asyncio.wait_for(gate.entered.wait(), 3)
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
        await asyncio.wait_for(gate.entered.wait(), 3)
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
        await asyncio.wait_for(gate.entered.wait(), 3)
        await crash(session)
        provider.crashed = False

        restored = await ChatSession.from_store(agent(), store, session.id)
        notice = restored._wakes[0].text
        assert "Completed before the restart: lookup({}) → 2 unread." in notice
        assert "Started but not confirmed (each may or may not have run): work({})." in notice
        await restored.close()

    async def test_a_turn_that_died_before_saving_anything_is_queued_again(self, db):
        provider, store = db
        provider.crashed = True  # no trace of the dead turn ever lands
        gate = Gate()
        session = ChatSession(make_agent(provider, gate), session_store=store)
        sent = session.send("start")
        await asyncio.wait_for(gate.entered.wait(), 3)
        await crash(session)
        provider.crashed = False

        restored = await ChatSession.from_store(make_agent(provider, Gate(released=True)), store, session.id)
        assert [p["event_id"] for p in restored.state()["pending"]] == [sent.id]
        (finished,) = await records(restored, "turn_finished", 1, after=restored.log.forgotten_through)
        assert finished["status"]["code"] == "success"
        assert root_outputs(restored)[0].output.collect_text() == "done"
        await restored.close()

    async def test_a_turn_cut_by_shutdown_is_recovered_too(self, db):
        provider, store = db
        gate = Gate()
        session = ChatSession(make_agent(provider, gate), session_store=store)
        session.send("start")
        await asyncio.wait_for(gate.entered.wait(), 3)
        await session.close(keep=True)

        restored = await ChatSession.from_store(make_agent(provider, Gate(released=True)), store, session.id)
        (started,) = await records(restored, "turn_started", 1, after=restored.log.forgotten_through)
        assert started["trigger"]["kind"] == "recovery" and started["trigger"]["cut"] == "shutdown"
        await records(restored, "turn_finished", 1, after=restored.log.forgotten_through)
        await restored.close()

    @pytest.mark.parametrize("on_dead_turn", ["notify", "drop"])
    async def test_on_dead_turn_can_skip_the_recovery_turn(self, db, on_dead_turn):
        provider, store = db
        gate = Gate()
        session = ChatSession(make_agent(provider, gate), session_store=store, on_dead_turn=on_dead_turn)
        session.send("start")
        await asyncio.wait_for(gate.entered.wait(), 3)
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
        await store.release_lease(session.id, PROCESS_OWNER)
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
