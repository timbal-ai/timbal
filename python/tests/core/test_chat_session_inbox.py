"""A durable ``ChatSession`` taking what other processes queued for it in its store's inbox (``_drain_inbox``): a
message, a wake or a message from another agent that reached a worker not serving the session.

It applies each item as the route that queued it would have, stores the session, and the item is acked: in that
same save with a store that ``acks_with_save`` (memory, SQLite), after it otherwise, when the snapshot's
``inbox_seen`` keeps an item pulled again after a crash between the two from being applied twice."""

import asyncio
import sqlite3
import time

import pytest
from timbal import Agent, Tool
from timbal.core import chat_session as chat_session_module
from timbal.core.chat_session import ChatSession
from timbal.core.test_model import TestModel
from timbal.state import set_call_id, set_parent_call_id, set_run_context
from timbal.state.background import forget_idle_background_store
from timbal.state.chat_store import (
    MAX_INBOX_ITEMS,
    InMemoryChatSessionStore,
    SessionSnapshot,
    SqliteChatSessionStore,
    process_owner,
)
from timbal.state.tracing.providers import SqliteTracingProvider
from timbal.types.content import ToolUseContent
from timbal.types.message import Message

pytestmark = pytest.mark.usefixtures("loop_watchdog")


@pytest.fixture(autouse=True)
def isolated_context(monkeypatch):
    monkeypatch.setattr(chat_session_module, "_INBOX_RETRY_S", 0.01)
    set_run_context(None)
    set_call_id(None)
    set_parent_call_id(None)
    yield
    set_run_context(None)


class Gate:
    def __init__(self, *, released=False):
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        if released:
            self.release.set()

    async def work(self) -> str:
        self.entered.set()
        await self.release.wait()
        return "worked"


def handler(messages):
    text = next((m.collect_text() for m in reversed(messages) if m.role == "user"), "")
    if messages[-1].role == "tool":
        return "done"
    if text == "start":
        return Message(
            role="assistant",
            content=[ToolUseContent(id=f"w{len(messages)}", name="work", input={})],
            stop_reason="tool_use",
        )
    return f"echo: {text.splitlines()[-1][:60]}"


def make_agent(provider, gate=None):
    gate = gate or Gate(released=True)
    return Agent(
        name="agent",
        model=TestModel(handler=handler),
        tools=[Tool(name="work", handler=gate.work)],
        tracing_provider=provider,
    )


class AcksApart(SqliteChatSessionStore):
    """A store that acks with a call of its own after the save, as the platform's does."""

    acks_with_save = False


class OlderSqlite(SqliteChatSessionStore):
    """A subclass written before the inbox: its save takes no ``inbox_acked``."""

    async def save(self, snapshot, *, expect_version, owner=None):
        return await super().save(snapshot, expect_version=expect_version, owner=owner)


@pytest.fixture(params=["sqlite", "acks apart", "memory"])
def db(request, tmp_path):
    provider = SqliteTracingProvider.configured(_path=tmp_path / "timbal.db")
    if request.param == "memory":
        return provider, InMemoryChatSessionStore()
    store_class = AcksApart if request.param == "acks apart" else SqliteChatSessionStore
    store = store_class(tmp_path / "timbal.db")
    store.inbox_poll_s = 0.01
    return provider, store


async def records(session, kind, count=1, timeout=30.0):  # a hang guard (slow runners stretch timers)
    found = []

    async def collect():
        async for item in session.events(0):
            if isinstance(item, dict) and item.get("kind") == kind:
                found.append(item)
                if len(found) >= count:
                    return

    await asyncio.wait_for(collect(), timeout)
    return found


async def until(condition, timeout=30.0):
    async def poll():
        while not condition():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(poll(), timeout)


async def acked(store, session_id, timeout=30.0):
    """Until nothing waits in the session's inbox (with a store that acks apart, the ack follows the save)."""

    async def poll():
        while await store.pull_inbox(session_id, process_owner()):
            await asyncio.sleep(0.005)

    await asyncio.wait_for(poll(), timeout)


async def stored_session(provider, store, gate=None) -> ChatSession:
    """A session stored and leased here: its inbox is being drained."""
    session = ChatSession(make_agent(provider, gate), session_store=store)
    session.persist()
    await session.flush()
    await until(lambda: session._inbox_task is not None)
    return session


async def idle(session):
    await until(lambda: not session.busy and not session.pending_messages() and not session._wakes)


async def history(provider, session):
    from timbal.state import RunContext

    trace = await provider.get(RunContext(parent_id=session.head, tracing_provider=provider))
    return [
        block["text"]
        for message in trace.get_path("agent")[0].memory
        for block in message["content"]
        if block.get("type") == "text"
    ]


async def kill(session):
    """The process dies where it stands: nothing more is stored, the session's tasks just stop."""
    session._lost = True
    tasks = [t for t in (session._worker, session._turn, session._lease_task, session._saver) if t is not None]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    session.closed = True
    session._scheduler.release(session._store)
    forget_idle_background_store(session._store)


class TestApplying:
    async def test_a_queued_message_is_answered(self, db):
        provider, store = db
        session = await stored_session(provider, store)
        inbox_id = await store.post_inbox(session.id, "message", {"text": "hi", "mode": None})
        (applied,) = await records(session, "inbox_item")
        assert applied["inbox_id"] == inbox_id and applied["item_kind"] == "message"
        await until(lambda: session.turns == 1)
        await idle(session)
        assert await history(provider, session) == ["hi", "echo: hi"]
        await acked(store, session.id)
        await session.close()

    async def test_a_queued_wake_runs_as_a_wake(self, db):
        provider, store = db
        session = await stored_session(provider, store)
        await store.post_inbox(
            session.id, "wake", {"text": "check in", "key": "k", "tool_policy": None, "silent_ok": False}
        )
        (applied,) = await records(session, "inbox_item")
        assert applied["wake_id"]
        (started,) = await records(session, "turn_started")
        assert started["trigger"]["kind"] == "wake" and started["trigger"]["wake_id"] == applied["wake_id"]
        await idle(session)
        await session.close()

    async def test_a_queued_agent_message_is_taken_as_one(self, db):
        provider, store = db
        session = await stored_session(provider, store)
        body = {"from": "bob", "text": "the build is green", "message_id": "am_7", "reply_to": None}
        await store.post_inbox(session.id, "agent_message", {**body, "priority": "normal", "hops": 1})
        (applied,) = await records(session, "inbox_item")
        assert applied["duplicate"] is False
        (taken,) = await records(session, "agent_message")
        assert (taken["sender"], taken["message_id"], taken["hops"]) == ("bob", "am_7", 1)
        assert taken["event_id"] == applied["event_id"]
        await idle(session)
        await session.close()

    @pytest.mark.parametrize(
        ("kind", "body"),
        [
            ("wake", {"text": "   "}),
            ("message", {"text": "hi", "mode": "sideways"}),
            ("message", {"text": 7}),
            ("agent_message", {"from": "no spaces allowed", "text": "x"}),
        ],
    )
    async def test_an_item_the_session_refuses_is_logged_and_never_comes_back(self, db, kind, body):
        provider, store = db
        session = await stored_session(provider, store)
        inbox_id = await store.post_inbox(session.id, kind, body)
        (refused,) = await records(session, "inbox_refused")
        assert refused["inbox_id"] == inbox_id and refused["error"]
        await acked(store, session.id)
        assert session.turns == 0 and not session.pending_messages()
        await session.close()

    async def test_a_body_the_store_cannot_read_is_refused_too(self, tmp_path):
        provider = SqliteTracingProvider.configured(_path=tmp_path / "timbal.db")
        store = SqliteChatSessionStore(tmp_path / "timbal.db")
        store.inbox_poll_s = 0.01
        session = await stored_session(provider, store)

        def written_by_something_else():
            conn = sqlite3.connect(str(tmp_path / "timbal.db"), timeout=10)
            conn.execute(
                "INSERT INTO chat_inbox (id, session_id, kind, body, posted_at) VALUES ('in_bad', ?, 'message', "
                "'{not json', ?)",
                (session.id, time.time()),
            )
            conn.commit()
            conn.close()

        await asyncio.to_thread(written_by_something_else)
        (refused,) = await records(session, "inbox_refused")
        assert refused["inbox_id"] == "in_bad"
        await acked(store, session.id)
        await session.close()

    async def test_items_come_in_the_order_they_were_posted(self, db):
        provider, store = db
        session = await stored_session(provider, store)
        session.busy_mode = "followup"
        ids = [await store.post_inbox(session.id, "message", {"text": f"m{n}", "mode": None}) for n in range(5)]
        applied = await records(session, "inbox_item", 5)
        assert [r["inbox_id"] for r in applied] == ids
        await until(lambda: session.turns == 5)
        await session.close()


class TestAckedWithItsEffect:
    async def test_an_item_is_applied_once_while_the_store_is_away(self, db, monkeypatch):
        """Pulled again and again while its save fails (the ack waits for it), it is applied once, and acked by the
        save that finally stores it."""
        provider, store = db
        session = await stored_session(provider, store)
        go = asyncio.Event()
        real_admit = session._admit

        async def admit():  # no turn takes the message meanwhile: it stays pending, in the snapshot
            await go.wait()
            await real_admit()

        session._admit = admit
        real_save, failed = store.save, []

        async def away(_snapshot, **_kwargs):
            failed.append(1)
            raise OSError("the session store is unreachable")

        monkeypatch.setattr(store, "save", away)
        await store.post_inbox(session.id, "message", {"text": "hi", "mode": None})
        await until(lambda: len(failed) >= 4)  # the drain stored, pulled again, stored again...
        assert len(session.pending_messages()) == 1  # ...and applied it once
        assert len(await store.pull_inbox(session.id, process_owner())) == 1  # not acked: nothing was stored
        monkeypatch.setattr(store, "save", real_save)
        await acked(store, session.id)  # (with a store that acks apart, once the save that stores it landed)
        assert [e["text"] for e in (await store.load(session.id)).pending] == ["hi"]
        go.set()
        await until(lambda: session.turns == 1)
        await idle(session)
        assert await history(provider, session) == ["hi", "echo: hi"]
        await session.close()

    async def test_a_store_that_acks_in_its_saves_is_never_asked_to_ack_apart(self, tmp_path, monkeypatch):
        provider = SqliteTracingProvider.configured(_path=tmp_path / "timbal.db")
        store = SqliteChatSessionStore(tmp_path / "timbal.db")
        acked_apart = []
        monkeypatch.setattr(store, "ack_inbox", lambda *args: acked_apart.append(args))
        session = await stored_session(provider, store)
        await store.post_inbox(session.id, "message", {"text": "hi", "mode": None})
        await records(session, "inbox_item")
        await until(lambda: session.turns == 1)
        await idle(session)
        assert acked_apart == [] and await store.pull_inbox(session.id, process_owner()) == []
        assert (await store.load(session.id)).inbox_seen == []  # nothing for the snapshot to remember
        await session.close()

    async def test_a_store_whose_save_takes_no_acks_is_acked_apart(self, tmp_path):
        """A subclass written before ``inbox_acked`` existed is never handed it (its save would raise)."""
        provider = SqliteTracingProvider.configured(_path=tmp_path / "timbal.db")
        store = OlderSqlite(tmp_path / "timbal.db")
        store.inbox_poll_s = 0.01
        session = await stored_session(provider, store)
        assert not session._acks_with_save
        inbox_id = await store.post_inbox(session.id, "message", {"text": "hi", "mode": None})
        await records(session, "inbox_item")
        await until(lambda: session.turns == 1)
        await idle(session)
        await acked(store, session.id)
        assert inbox_id in (await store.load(session.id)).inbox_seen
        await session.close()


class TestACrashBetweenTheSaveAndTheAck:
    async def test_the_restored_session_does_not_apply_the_item_again(self, tmp_path, monkeypatch):
        """With a store that acks apart, the process dies after the save that stored the message and before its
        ack: the next process pulls the item again, and the snapshot's ``inbox_seen`` says it was applied."""
        provider = SqliteTracingProvider.configured(_path=tmp_path / "timbal.db")
        store = AcksApart(tmp_path / "timbal.db")
        store.inbox_poll_s = 0.01
        session = await stored_session(provider, store)
        go = asyncio.Event()
        real_admit = session._admit

        async def admit():  # the message stays pending in the snapshot until the process dies
            await go.wait()
            await real_admit()

        session._admit = admit
        real_ack, acking = store.ack_inbox, asyncio.Event()

        async def dies_first(*_args):
            acking.set()
            await asyncio.Event().wait()  # the process dies before this lands

        monkeypatch.setattr(store, "ack_inbox", dies_first)
        inbox_id = await store.post_inbox(session.id, "message", {"text": "hi", "mode": None})
        await asyncio.wait_for(acking.wait(), 30)
        assert inbox_id in (await store.load(session.id)).inbox_seen  # stored, with what it applied
        await kill(session)
        monkeypatch.setattr(store, "ack_inbox", real_ack)

        restored = await ChatSession.from_store(make_agent(provider), store, session.id)
        await until(lambda: restored.turns == 1)
        await idle(restored)
        await until(lambda: not restored._dirty)
        assert await history(provider, restored) == ["hi", "echo: hi"]  # once
        await acked(store, restored.id)  # by the new process
        assert not [e for e in restored.log.peek(restored.log.forgotten_through)[0] if isinstance(e, dict)
                    and e.get("kind") == "inbox_item"]  # fmt: skip
        await restored.close()

    async def test_a_session_restored_by_a_store_that_acks_in_its_saves_acks_what_an_older_one_applied(self, tmp_path):
        """The snapshot was written with acks apart; the store now acks in its saves: the first save acks them."""
        provider = SqliteTracingProvider.configured(_path=tmp_path / "timbal.db")
        apart = AcksApart(tmp_path / "timbal.db")
        session = await stored_session(provider, apart)
        await session.close(keep=True)
        store = SqliteChatSessionStore(tmp_path / "timbal.db")
        await store.acquire_lease(session.id, process_owner())
        inbox_id = await store.post_inbox(session.id, "message", {"text": "hi", "mode": None})
        snapshot = await store.load(session.id)
        snapshot.inbox_seen = [inbox_id]  # applied by the older process, which died before acking it
        await store.save(snapshot, expect_version=snapshot.version, owner=process_owner())

        restored = await ChatSession.from_store(make_agent(provider), store, session.id)
        await until(lambda: not restored._inbox_unacked and not restored._dirty)
        assert await store.pull_inbox(restored.id, process_owner()) == []
        assert restored.turns == 0 and not restored.pending_messages()  # not applied again
        assert (await store.load(restored.id)).inbox_seen == []
        await restored.close()

    def test_the_window_is_at_least_what_the_store_lets_wait(self, tmp_path):
        """An item pulled again after a crash is one still waiting, and at most ``max_inbox_items`` wait: the
        session keeps at least that many ids, whatever the store's own cap."""
        provider = SqliteTracingProvider.configured(_path=tmp_path / "timbal.db")
        agent = make_agent(provider)
        assert ChatSession(agent, session_store=AcksApart(tmp_path / "a.db"))._inbox_seen.maxlen == MAX_INBOX_ITEMS
        bigger = AcksApart(tmp_path / "b.db")
        bigger.max_inbox_items = MAX_INBOX_ITEMS * 4
        assert ChatSession(agent, session_store=bigger)._inbox_seen.maxlen == MAX_INBOX_ITEMS * 4


class TestCutsWaitForTheStore:
    async def test_a_queued_interrupt_cuts_the_turn_only_once_it_is_stored(self, tmp_path, monkeypatch):
        provider = SqliteTracingProvider.configured(_path=tmp_path / "timbal.db")
        store = SqliteChatSessionStore(tmp_path / "timbal.db")
        store.inbox_poll_s = 0.01
        gate = Gate()
        session = await stored_session(provider, store, gate)
        session.send("start")
        await asyncio.wait_for(gate.entered.wait(), 30)
        turn = session._turn
        real_save, failed = store.save, []

        async def away(_snapshot, **_kwargs):
            failed.append(1)
            raise OSError("the session store is unreachable")

        monkeypatch.setattr(store, "save", away)
        await store.post_inbox(session.id, "message", {"text": "stop", "mode": "interrupt"})
        await records(session, "inbox_item")
        count = len(failed)
        await until(lambda: len(failed) >= count + 3)  # tried, and failed, more than once since
        assert session._turn is turn and not turn.done()  # nothing is cut for a message not stored
        monkeypatch.setattr(store, "save", real_save)
        (finished, *_) = await records(session, "turn_finished")
        assert finished["status"]["code"] == "cancelled"
        await idle(session)
        await session.close()


class TestTheDrainStopsWithTheLease:
    async def test_closing_leaves_nothing_running(self, db):
        provider, store = db
        before = asyncio.all_tasks()
        session = await stored_session(provider, store)
        drain = session._inbox_task
        await session.close()
        await asyncio.wait_for(asyncio.wait({drain}), 10)
        assert drain.done()
        left = [t for t in asyncio.all_tasks() - before if not t.done() and t is not asyncio.current_task()]
        await asyncio.gather(*left, return_exceptions=True)

    async def test_a_session_whose_lease_was_taken_stops_draining_and_applies_nothing(self, tmp_path):
        provider = SqliteTracingProvider.configured(_path=tmp_path / "timbal.db")
        store = SqliteChatSessionStore(tmp_path / "timbal.db")
        store.inbox_poll_s = 0.01
        session = await stored_session(provider, store)

        def taken(conn):
            conn.execute(
                "UPDATE chat_sessions SET lease_owner = 'elsewhere:1:x', lease_until = ? WHERE session_id = ?",
                (time.time() + 3600, session.id),
            )
            conn.commit()

        await store._run(taken)
        await store.post_inbox(session.id, "message", {"text": "for the new owner", "mode": None})
        await asyncio.wait_for(session.wait_closed(), 30)
        assert session.lost and not session.pending_messages()
        assert [i["body"]["text"] for i in await store.pull_inbox(session.id, "elsewhere:1:x")] == ["for the new owner"]

    async def test_a_session_without_a_store_has_no_drain(self, tmp_path):
        provider = SqliteTracingProvider.configured(_path=tmp_path / "timbal.db")
        session = ChatSession(make_agent(provider))
        session.send("hi")
        await until(lambda: session.turns == 1)
        assert session._inbox_task is None
        await session.close()


async def test_a_snapshot_from_before_the_inbox_restores_with_nothing_seen(tmp_path):
    provider = SqliteTracingProvider.configured(_path=tmp_path / "timbal.db")
    store = AcksApart(tmp_path / "timbal.db")
    session = await stored_session(provider, store)
    await session.close(keep=True)

    def as_written_before(conn):
        import json

        raw = conn.execute("SELECT snapshot FROM chat_sessions WHERE session_id = ?", (session.id,)).fetchone()[0]
        data = json.loads(raw)
        del data["inbox_seen"]
        conn.execute("UPDATE chat_sessions SET snapshot = ? WHERE session_id = ?", (json.dumps(data), session.id))
        conn.commit()

    await store._run(as_written_before)
    assert SessionSnapshot.from_json('{"session_id": "x", "agent": "a"}').inbox_seen == []
    restored = await ChatSession.from_store(make_agent(provider), store, session.id)
    assert list(restored._inbox_seen) == []
    await restored.close()


class TestStoresThatDoNotTakeTheAck:
    async def test_a_save_that_takes_only_kwargs_is_acked_apart(self, tmp_path):
        """It may not hand ``inbox_acked`` on: trusted, the ack it drops left the item to be applied on every pull."""

        class KwargsSqlite(SqliteChatSessionStore):
            async def save(self, snapshot, **kwargs):
                return await super().save(snapshot, expect_version=kwargs["expect_version"], owner=kwargs.get("owner"))

        provider = SqliteTracingProvider.configured(_path=tmp_path / "timbal.db")
        store = KwargsSqlite(tmp_path / "timbal.db")
        store.inbox_poll_s = 0.01
        session = await stored_session(provider, store)
        assert not session._acks_with_save
        await store.post_inbox(session.id, "message", {"text": "hi", "mode": None})
        await records(session, "inbox_item")
        await until(lambda: session.turns == 1)
        await acked(store, session.id)
        await idle(session)
        assert await history(provider, session) == ["hi", "echo: hi"]
        await session.close()

    async def test_an_item_pulled_again_after_its_ack_is_not_applied_again(self, tmp_path, monkeypatch):
        """A store that takes ``inbox_acked`` and does not ack (or a pull that read the inbox before the save that
        acked it committed): the item comes back, and the session knows it applied it."""

        class DropsAcks(SqliteChatSessionStore):
            async def save(self, snapshot, *, expect_version, owner=None, inbox_acked=()):  # noqa: ARG002
                return await super().save(snapshot, expect_version=expect_version, owner=owner)

        provider = SqliteTracingProvider.configured(_path=tmp_path / "timbal.db")
        store = DropsAcks(tmp_path / "timbal.db")
        store.inbox_poll_s = 0.01
        pulls = []
        real_pull = store.pull_inbox

        async def counted(*args, **kwargs):
            items = await real_pull(*args, **kwargs)
            pulls.append(len(items))
            return items

        monkeypatch.setattr(store, "pull_inbox", counted)
        session = await stored_session(provider, store)
        await store.post_inbox(session.id, "message", {"text": "hi", "mode": None})
        await until(lambda: pulls.count(1) >= 4)  # handed out again and again
        await until(lambda: session.turns == 1)
        await idle(session)
        applied = [e for e in session.log.peek(session.log.forgotten_through)[0] if isinstance(e, dict)
                   and e.get("kind") == "inbox_item"]  # fmt: skip
        assert len(applied) == 1
        assert await history(provider, session) == ["hi", "echo: hi"]
        await session.close()


class TestAnItemThatMustWait:
    async def test_an_agent_message_for_a_loop_that_is_not_running_waits_for_it(self, tmp_path):
        """The session is open and the loop that serves it is not running (``receive_agent_message`` says
        "closed"): the item is neither refused nor the drain stopped (it stopped for good, and the inbox filled)."""
        provider = SqliteTracingProvider.configured(_path=tmp_path / "timbal.db")
        store = SqliteChatSessionStore(tmp_path / "timbal.db")
        store.inbox_poll_s = 0.01
        session = await stored_session(provider, store)
        served_on, stopped = session._loop, asyncio.new_event_loop()
        session._loop = stopped
        try:
            pulls = []
            real_pull = store.pull_inbox

            async def counted(*args, **kwargs):
                items = await real_pull(*args, **kwargs)
                pulls.append(len(items))
                return items

            store.pull_inbox = counted
            await store.post_inbox(session.id, "agent_message", {"from": "bob", "text": "ping", "message_id": "am_1"})
            await until(lambda: pulls.count(1) >= 3)  # tried, and kept for later, more than once
            assert not session._inbox_task.done() and not session.pending_messages()
            session._loop = served_on
            (applied,) = await records(session, "inbox_item")
            assert applied["item_kind"] == "agent_message"
        finally:
            session._loop = served_on
            stopped.close()
        await idle(session)
        await session.close()


class TestACutAfterAWriteThatFailed:
    async def test_a_queued_interrupt_cuts_once_stored_even_when_the_next_write_fails(self, tmp_path, monkeypatch):
        """The save that stores the message lands and a later one fails: the message is stored, so it cuts (it
        waited for every later write to land, unlike the route)."""
        provider = SqliteTracingProvider.configured(_path=tmp_path / "timbal.db")
        store = SqliteChatSessionStore(tmp_path / "timbal.db")
        store.inbox_poll_s = 0.01
        gate = Gate()
        session = await stored_session(provider, store, gate)
        session.send("start")
        await asyncio.wait_for(gate.entered.wait(), 30)
        turn, real_save, landed = session._turn, store.save, []

        async def then_away(snapshot, **kwargs):
            if landed:
                raise OSError("the session store is unreachable")
            version = await real_save(snapshot, **kwargs)
            if any(e.get("text") == "stop" for e in snapshot.pending):
                landed.append(1)
                session.wake("a change after it")  # stored next, and that write fails
            return version

        monkeypatch.setattr(store, "save", then_away)
        await store.post_inbox(session.id, "message", {"text": "stop", "mode": "interrupt"})
        await asyncio.wait_for(asyncio.wait({turn}), 30)
        assert turn.done() and landed
        monkeypatch.setattr(store, "save", real_save)
        await session.close()
