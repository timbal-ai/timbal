"""The session inbox of the memory and SQLite stores (``ChatSessionStore.post_inbox`` / ``pull_inbox`` / ``ack_inbox``):
work another process takes for a session it does not serve, for the lease owner to apply.

The rules follow the platform's session store (monolith ``services/session_runtime``): a post repeated with its
idempotency key is the same item, also for a day after its ack; at most ``max_inbox_items`` wait; a pull or an ack
by a process without the lease is refused; a closed or deleted session takes nothing and drops what waited. On
top of it, these two stores ack an item in the save that stores what it did (one write)."""

import asyncio
import sqlite3
import time

import pytest
from timbal.state import chat_store
from timbal.state.chat_store import (
    INBOX_KEY_KEPT_S,
    MAX_INBOX_ITEMS,
    MAX_INBOX_KEY_CHARS,
    InboxFull,
    InMemoryChatSessionStore,
    SessionLeased,
    SessionSnapshot,
    SnapshotConflict,
    SqliteChatSessionStore,
)

OWNER = "here:1:a"
OTHER = "elsewhere:2:b"


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    if request.param == "memory":
        return InMemoryChatSessionStore()
    return SqliteChatSessionStore(tmp_path / "sessions.db")


async def stored(store, session_id="s1", *, owner=OWNER) -> int:
    """A stored session leased to ``owner``: its version."""
    return await store.save(SessionSnapshot(session_id=session_id, agent="agent"), expect_version=None, owner=owner)


def message(text):
    return {"text": text, "mode": None}


async def waiting(store, session_id="s1"):
    return [item["body"]["text"] for item in await store.pull_inbox(session_id, OWNER)]


class TestPosting:
    async def test_items_come_out_oldest_first_with_their_kind_and_body(self, store):
        await stored(store)
        first = await store.post_inbox("s1", "message", message("one"))
        second = await store.post_inbox("s1", "wake", {"text": "two", "key": None})
        third = await store.post_inbox("s1", "agent_message", {"from": "bob", "text": "three"})
        items = await store.pull_inbox("s1", OWNER)
        assert [(i["id"], i["kind"], i["body"]["text"]) for i in items] == [
            (first, "message", "one"),
            (second, "wake", "two"),
            (third, "agent_message", "three"),
        ]
        assert all(isinstance(i["posted_at"], float) for i in items)

    async def test_what_a_pull_hands_out_is_its_own_copy(self, store):
        await stored(store)
        body = message("one")
        await store.post_inbox("s1", "message", body)
        body["text"] = "changed by the sender"
        (item,) = await store.pull_inbox("s1", OWNER)
        item["body"]["text"] = "changed by the owner"
        assert await waiting(store) == ["one"]

    @pytest.mark.parametrize(
        ("kind", "body", "key", "error"),
        [
            ("note", {"text": "x"}, None, ValueError),
            ("message", "text", None, TypeError),
            ("message", {"x": float("nan")}, None, TypeError),
            ("message", {"x": object()}, None, TypeError),
            ("message", {"text": "x"}, "", ValueError),
            ("message", {"text": "x"}, "k" * (MAX_INBOX_KEY_CHARS + 1), ValueError),
        ],
    )
    async def test_what_the_store_cannot_take_is_refused_before_anything_is_queued(self, store, kind, body, key, error):
        await stored(store)
        with pytest.raises(error):
            await store.post_inbox("s1", kind, body, idempotency_key=key)
        assert await store.pull_inbox("s1", OWNER) == []

    async def test_a_key_of_the_longest_length_is_taken(self, store):
        await stored(store)
        await store.post_inbox("s1", "message", message("x"), idempotency_key="k" * MAX_INBOX_KEY_CHARS)
        assert await waiting(store) == ["x"]

    async def test_a_session_never_stored_takes_nothing(self, store):
        with pytest.raises(LookupError):
            await store.post_inbox("nobody", "message", message("x"))

    async def test_a_closed_session_takes_nothing_and_drops_what_waited(self, store):
        version = await stored(store)
        await store.post_inbox("s1", "message", message("waited"))
        closed = SessionSnapshot(session_id="s1", agent="agent", closed=True)
        await store.save(closed, expect_version=version, owner=OWNER)
        with pytest.raises(LookupError):
            await store.post_inbox("s1", "message", message("late"))
        assert await store.pull_inbox("s1", OWNER) == []

    async def test_a_deleted_session_drops_what_waited(self, store):
        await stored(store)
        await store.post_inbox("s1", "message", message("waited"), idempotency_key="k")
        await store.delete("s1")
        await stored(store)  # the same id again, a new session
        assert await store.pull_inbox("s1", OWNER) == []
        await store.post_inbox("s1", "message", message("new"), idempotency_key="k")  # the key is free again
        assert await waiting(store) == ["new"]

    async def test_a_delete_another_owner_refuses_drops_nothing(self, store):
        await stored(store)
        await store.post_inbox("s1", "message", message("waited"))
        await store.release_lease("s1", OWNER)
        await store.acquire_lease("s1", OTHER)
        with pytest.raises(SnapshotConflict):
            await store.delete("s1", owner=OWNER)
        assert [i["body"]["text"] for i in await store.pull_inbox("s1", OTHER)] == ["waited"]


class TestIdempotencyKeys:
    async def test_a_repeat_while_the_item_waits_is_the_same_item(self, store):
        await stored(store)
        first = await store.post_inbox("s1", "message", message("hi"), idempotency_key="k1")
        again = await store.post_inbox("s1", "message", message("hi"), idempotency_key="k1")
        assert again == first
        assert await waiting(store) == ["hi"]

    async def test_a_repeat_after_the_ack_is_the_same_item_and_queues_nothing(self, store):
        version = await stored(store)
        first = await store.post_inbox("s1", "message", message("hi"), idempotency_key="k1")
        await store.save(SessionSnapshot(session_id="s1", agent="agent"), expect_version=version, owner=OWNER,
                         inbox_acked=[first])  # fmt: skip
        assert await store.pull_inbox("s1", OWNER) == []
        assert await store.post_inbox("s1", "message", message("hi"), idempotency_key="k1") == first
        assert await store.pull_inbox("s1", OWNER) == []

    async def test_a_key_is_kept_a_day_after_the_ack_then_a_post_with_it_is_new(self, store, monkeypatch):
        await stored(store)
        first = await store.post_inbox("s1", "message", message("hi"), idempotency_key="k1")
        await store.ack_inbox("s1", OWNER, [first])
        now = time.time()
        monkeypatch.setattr(time, "time", lambda: now + INBOX_KEY_KEPT_S - 60)
        assert await store.post_inbox("s1", "message", message("hi"), idempotency_key="k1") == first
        monkeypatch.setattr(time, "time", lambda: now + INBOX_KEY_KEPT_S + 60)
        await store.acquire_lease("s1", OWNER)  # (the lease ran out with the clock)
        later = await store.post_inbox("s1", "message", message("hi again"), idempotency_key="k1")
        assert later != first
        assert await waiting(store) == ["hi again"]

    async def test_keys_are_per_session(self, store):
        await stored(store, "s1")
        await stored(store, "s2")
        a = await store.post_inbox("s1", "message", message("a"), idempotency_key="k")
        b = await store.post_inbox("s2", "message", message("b"), idempotency_key="k")
        assert a != b
        assert await waiting(store, "s1") == ["a"] and await waiting(store, "s2") == ["b"]

    async def test_posts_without_a_key_are_each_an_item(self, store):
        await stored(store)
        await store.post_inbox("s1", "message", message("hi"))
        await store.post_inbox("s1", "message", message("hi"))
        assert await waiting(store) == ["hi", "hi"]


class TestTheCap:
    async def test_a_post_past_the_cap_is_refused(self, store):
        store.max_inbox_items = 2
        await stored(store)
        await store.post_inbox("s1", "message", message("1"))
        await store.post_inbox("s1", "message", message("2"))
        with pytest.raises(InboxFull):
            await store.post_inbox("s1", "message", message("3"))
        assert await waiting(store) == ["1", "2"]

    async def test_the_cap_is_the_platforms(self):
        assert MAX_INBOX_ITEMS == 256
        assert InMemoryChatSessionStore().max_inbox_items == MAX_INBOX_ITEMS
        assert SqliteChatSessionStore.max_inbox_items == MAX_INBOX_ITEMS

    async def test_a_repeat_of_a_waiting_key_at_the_cap_is_still_the_same_item(self, store):
        store.max_inbox_items = 1
        await stored(store)
        first = await store.post_inbox("s1", "message", message("1"), idempotency_key="k")
        assert await store.post_inbox("s1", "message", message("1"), idempotency_key="k") == first

    async def test_acked_items_and_kept_keys_do_not_count(self, store):
        store.max_inbox_items = 1
        await stored(store)
        first = await store.post_inbox("s1", "message", message("1"), idempotency_key="k")
        await store.ack_inbox("s1", OWNER, [first])
        await store.post_inbox("s1", "message", message("2"))
        assert await waiting(store) == ["2"]

    async def test_the_cap_is_per_session(self, store):
        store.max_inbox_items = 1
        await stored(store, "s1")
        await stored(store, "s2")
        await store.post_inbox("s1", "message", message("a"))
        await store.post_inbox("s2", "message", message("b"))


class TestOnlyTheLeaseHolder:
    async def test_a_pull_by_another_owner_is_refused(self, store):
        await stored(store)
        await store.post_inbox("s1", "message", message("x"))
        with pytest.raises(SessionLeased):
            await store.pull_inbox("s1", OTHER)

    async def test_a_pull_on_a_lease_that_ran_out_is_refused(self, store, monkeypatch):
        await stored(store)
        later = time.time() + store.lease_ttl_s + 1
        monkeypatch.setattr(time, "time", lambda: later)
        with pytest.raises(SessionLeased):
            await store.pull_inbox("s1", OWNER)

    async def test_a_pull_after_the_lease_was_let_go_is_refused(self, store):
        await stored(store)
        await store.release_lease("s1", OWNER)
        with pytest.raises(SessionLeased):
            await store.pull_inbox("s1", OWNER)

    async def test_a_pull_of_no_stored_session_is_a_lookup_error(self, store):
        with pytest.raises(LookupError):
            await store.pull_inbox("nobody", OWNER)

    async def test_an_ack_by_another_owner_is_refused_and_removes_nothing(self, store):
        await stored(store)
        item = await store.post_inbox("s1", "message", message("x"))
        await store.release_lease("s1", OWNER)
        await store.acquire_lease("s1", OTHER)
        with pytest.raises(SessionLeased):
            await store.ack_inbox("s1", OWNER, [item])
        assert [i["id"] for i in await store.pull_inbox("s1", OTHER)] == [item]

    async def test_an_ack_of_no_stored_session_is_a_lookup_error(self, store):
        with pytest.raises(LookupError):
            await store.ack_inbox("nobody", OWNER, ["in_x"])


class TestAckedInTheSave:
    async def test_the_save_acks_what_it_names(self, store):
        version = await stored(store)
        first = await store.post_inbox("s1", "message", message("1"))
        await store.post_inbox("s1", "message", message("2"))
        await store.save(
            SessionSnapshot(session_id="s1", agent="agent"), expect_version=version, owner=OWNER, inbox_acked=[first]
        )
        assert await waiting(store) == ["2"]

    async def test_a_refused_save_acks_nothing(self, store):
        version = await stored(store)
        item = await store.post_inbox("s1", "message", message("1"))
        with pytest.raises(SnapshotConflict):
            await store.save(
                SessionSnapshot(session_id="s1", agent="agent"),
                expect_version=version + 5,
                owner=OWNER,
                inbox_acked=[item],
            )
        assert await waiting(store) == ["1"]

    async def test_an_ack_names_its_owner(self, store):
        version = await stored(store)
        item = await store.post_inbox("s1", "message", message("1"))
        with pytest.raises(ValueError):
            await store.save(SessionSnapshot(session_id="s1", agent="agent"), expect_version=version,
                             inbox_acked=[item])  # fmt: skip
        assert (await store.load("s1")).version == version
        assert await waiting(store) == ["1"]

    async def test_many_ids_at_once(self, store):
        version = await stored(store)
        ids = [await store.post_inbox("s1", "message", message(str(n))) for n in range(MAX_INBOX_ITEMS)]
        await store.save(
            SessionSnapshot(session_id="s1", agent="agent"),
            expect_version=version,
            owner=OWNER,
            inbox_acked=[f"in_unknown_{n}" for n in range(600)] + ids,  # more than SQLite takes in one statement
        )
        await store.acquire_lease("s1", OWNER)  # (256 posts on a slow disk outlast the lease)
        assert await store.pull_inbox("s1", OWNER) == []

    async def test_sqlite_stores_the_snapshot_and_the_ack_together_or_neither(self, tmp_path, monkeypatch):
        """A failure in the ack rolls the snapshot back: the item is not acked without the effect stored, nor the
        effect stored without the ack (it would be applied twice)."""
        store = SqliteChatSessionStore(tmp_path / "sessions.db")
        version = await stored(store)
        item = await store.post_inbox("s1", "message", message("1"))

        def broken(*_args):
            raise sqlite3.OperationalError("disk I/O error")

        monkeypatch.setattr(chat_store, "_ack_rows", broken)
        with pytest.raises(sqlite3.OperationalError):
            await store.save(
                SessionSnapshot(session_id="s1", agent="agent", turns=7),
                expect_version=version,
                owner=OWNER,
                inbox_acked=[item],
            )
        monkeypatch.undo()
        snapshot = await store.load("s1")
        assert (snapshot.version, snapshot.turns) == (version, 0)
        assert await waiting(store) == ["1"]
        # And the connection is not left inside that transaction: the next save is a save of its own.
        await store.save(SessionSnapshot(session_id="s1", agent="agent", turns=1), expect_version=version, owner=OWNER)
        assert (await store.load("s1")).turns == 1


class TestFindingSessionsWithItems:
    async def test_a_session_with_items_waiting_has_work(self, store):
        version = await stored(store)
        assert await store.list_ids(with_work=True) == []
        item = await store.post_inbox("s1", "message", message("x"))
        assert await store.list_ids(with_work=True) == ["s1"]
        await store.save(
            SessionSnapshot(session_id="s1", agent="agent"), expect_version=version, owner=OWNER, inbox_acked=[item]
        )
        assert await store.list_ids(with_work=True) == []

    async def test_a_kept_key_is_no_work(self, store):
        await stored(store)
        item = await store.post_inbox("s1", "message", message("x"), idempotency_key="k")
        await store.ack_inbox("s1", OWNER, [item])
        assert await store.list_ids(with_work=True) == []

    async def test_sqlite_lists_them_oldest_first_with_the_sessions_that_have_work(self, tmp_path):
        store = SqliteChatSessionStore(tmp_path / "sessions.db")
        await stored(store, "a")
        await store.save(SessionSnapshot(session_id="b", agent="agent", wakes=[{"id": "w"}]), expect_version=None)
        await stored(store, "c")
        await store.post_inbox("c", "message", message("x"))
        await store.post_inbox("a", "message", message("y"))
        assert await store.list_ids(with_work=True) == ["a", "b", "c"]


class TestWaitingForAnItem:
    async def test_a_pull_waits_and_a_post_ends_the_wait(self, store):
        await stored(store)

        async def post_soon():
            await asyncio.sleep(0.05)
            await store.post_inbox("s1", "message", message("x"))

        poster = asyncio.ensure_future(post_soon())
        started = time.monotonic()
        items = await store.pull_inbox("s1", OWNER, wait_s=20)
        await poster
        assert [i["body"]["text"] for i in items] == ["x"]
        assert time.monotonic() - started < 10  # woken, not timed out (a hang guard, not a timing check)

    async def test_a_pull_with_nothing_posted_comes_back_empty_when_its_wait_is_over(self, store):
        await stored(store)
        assert await store.pull_inbox("s1", OWNER, wait_s=0.05) == []

    async def test_a_waiting_pull_ends_when_its_caller_stops_waiting(self, store):
        await stored(store)
        pull = asyncio.ensure_future(store.pull_inbox("s1", OWNER, wait_s=20))
        await asyncio.sleep(0.05)
        pull.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pull

    async def test_sqlite_sees_a_post_another_process_made(self, tmp_path):
        """Another process's post reaches the file only: the watch's poll finds it."""
        path = tmp_path / "sessions.db"
        store = SqliteChatSessionStore(path)
        store.inbox_poll_s = 0.02
        await stored(store)

        def other_process():
            conn = sqlite3.connect(str(path), timeout=10)
            conn.execute(
                "INSERT INTO chat_inbox (id, session_id, kind, body, posted_at) VALUES (?, ?, ?, ?, ?)",
                ("in_other", "s1", "message", '{"text": "from elsewhere", "mode": null}', time.time()),
            )
            conn.commit()
            conn.close()

        async def post_soon():
            await asyncio.sleep(0.05)
            await asyncio.to_thread(other_process)

        poster = asyncio.ensure_future(post_soon())
        items = await store.pull_inbox("s1", OWNER, wait_s=20)
        await poster
        assert [i["id"] for i in items] == ["in_other"]

    async def test_sqlite_pulls_that_wait_do_not_each_ask_the_file(self, tmp_path, monkeypatch):
        """One query per poll for all of them, off the store's own thread: a query per waiting session and per poll
        would queue on that one thread ahead of every save."""
        store = SqliteChatSessionStore(tmp_path / "sessions.db")
        store.inbox_poll_s = 0.01
        for n in range(20):
            await stored(store, f"s{n}")
        calls = []
        real_run = store._run

        async def counted(fn):
            calls.append(fn)
            return await real_run(fn)

        monkeypatch.setattr(store, "_run", counted)
        polls = []
        real_ready = store._sessions_with_items
        monkeypatch.setattr(store, "_sessions_with_items", lambda ids: polls.append(len(ids)) or real_ready(ids))
        await asyncio.gather(*(store.pull_inbox(f"s{n}", OWNER, wait_s=0.5) for n in range(20)))
        assert len(calls) == 40  # each pull: one look before its wait, one after it
        assert polls and max(polls) == 20  # the watch asked once per poll, for every session at once


class TestATransactionLeftOpen:
    """A statement that failed before its commit (``database is locked``) left a transaction open on the shared
    connection: the inbox read the file as it was then, and a post joined that transaction."""

    async def test_a_pull_sees_what_another_process_posted_since(self, tmp_path):
        path = tmp_path / "sessions.db"
        store = SqliteChatSessionStore(path)
        await stored(store)
        db = store._db()
        with db.lock:
            db.conn.execute("BEGIN")
            db.conn.execute("SELECT count(*) FROM chat_inbox").fetchone()  # a read, left open

        def other_process():
            conn = sqlite3.connect(str(path), timeout=10)
            conn.execute(
                "INSERT INTO chat_inbox (id, session_id, kind, body, posted_at) VALUES ('in_other', 's1', 'message', "
                '\'{"text": "x"}\', ?)',
                (time.time(),),
            )
            conn.commit()
            conn.close()

        await asyncio.to_thread(other_process)
        assert [i["id"] for i in await store.pull_inbox("s1", OWNER)] == ["in_other"]

    async def test_a_post_does_not_commit_what_was_left_open(self, tmp_path):
        store = SqliteChatSessionStore(tmp_path / "sessions.db")
        await stored(store)
        db = store._db()
        with db.lock:
            db.conn.execute("BEGIN")
            db.conn.execute("UPDATE chat_sessions SET agent = 'half-written' WHERE session_id = 's1'")
        await store.post_inbox("s1", "message", message("x"))

        def column(conn):
            return conn.execute("SELECT agent FROM chat_sessions WHERE session_id = 's1'").fetchone()[0]

        assert await store._run(column) == "agent"  # the statement left open was not committed with the post
        assert await waiting(store) == ["x"]


class TestTheAgentAndTheKeysAWorkerAsksAbout:
    async def test_a_post_for_another_agents_session_is_refused(self, store):
        await stored(store)  # agent "agent"
        with pytest.raises(LookupError):
            await store.post_inbox("s1", "message", message("x"), agent="another_agent")
        await store.post_inbox("s1", "message", message("y"), agent="agent")
        assert await waiting(store) == ["y"]

    async def test_a_key_is_found_while_its_item_waits_and_a_day_after_its_ack(self, store, monkeypatch):
        await stored(store)
        assert await store.find_inbox_key("s1", "k") is None
        item = await store.post_inbox("s1", "message", message("x"), idempotency_key="k")
        assert await store.find_inbox_key("s1", "k") == (item, False)
        await store.ack_inbox("s1", OWNER, [item])
        assert await store.find_inbox_key("s1", "k") == (item, True)
        later = time.time() + INBOX_KEY_KEPT_S + 60
        monkeypatch.setattr(time, "time", lambda: later)
        assert await store.find_inbox_key("s1", "k") is None

    async def test_an_item_acked_as_refused_takes_its_key_with_it(self, store):
        """A retry of a refused item is screened again: its key does not name the item any more."""
        await stored(store)
        item = await store.post_inbox("s1", "message", message("x"), idempotency_key="k")
        await store.ack_inbox("s1", OWNER, [item], forget=True)
        assert await store.find_inbox_key("s1", "k") is None
        again = await store.post_inbox("s1", "message", message("x"), idempotency_key="k")
        assert again != item and await waiting(store) == ["x"]

    async def test_how_many_items_wait_needs_no_lease(self, store):
        await stored(store)
        assert await store.inbox_waiting("s1") == 0
        first = await store.post_inbox("s1", "message", message("1"), idempotency_key="k")
        await store.post_inbox("s1", "message", message("2"))
        await store.release_lease("s1", OWNER)
        assert await store.inbox_waiting("s1") == 2
        await store.acquire_lease("s1", OWNER)
        await store.ack_inbox("s1", OWNER, [first])
        assert await store.inbox_waiting("s1") == 1  # the key kept after the ack does not count
        assert await store.inbox_waiting("nobody") == 0
