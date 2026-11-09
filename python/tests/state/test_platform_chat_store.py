"""PlatformChatSessionStore against a fake platform implementing rfcs/session-runtime.md (monolith#567),
and the session inbox through every store that has one."""

import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from timbal import Agent, Tool
from timbal.core.chat_session import ChatSession
from timbal.core.test_model import TestModel
from timbal.errors import PlatformError
from timbal.state import set_call_id, set_parent_call_id, set_run_context
from timbal.state.chat_store import (
    PROCESS_OWNER,
    InMemoryChatSessionStore,
    SessionLeased,
    SessionSnapshot,
    SnapshotConflict,
    SqliteChatSessionStore,
)
from timbal.state.config import PlatformAuth, PlatformAuthType, PlatformConfig, PlatformSubject
from timbal.state.platform_chat_store import PlatformChatSessionStore
from timbal.state.tracing.providers import InMemoryTracingProvider
from timbal.types.content import ToolUseContent
from timbal.types.events import OutputEvent
from timbal.types.message import Message

CONFIG = PlatformConfig(
    host="api.runtime.test",
    auth=PlatformAuth(type=PlatformAuthType.BEARER, token="t2_test"),
    subject=PlatformSubject(org_id="1", app_id="2"),
)


@pytest.fixture(autouse=True)
def isolated_context():
    set_run_context(None)
    set_call_id(None)
    set_parent_call_id(None)
    yield
    set_run_context(None)


def _error(status, message="error"):
    return PlatformError(message, status_code=status)


class FakeRuntime:
    """The platform side of rfcs/session-runtime.md, in memory."""

    def __init__(self):
        self.rows: dict[str, dict] = {}
        self.inbox: dict[str, list[dict]] = {}
        self.changed: dict[str, asyncio.Event] = {}
        self.requests: list[tuple[str, str]] = []
        self.commit_then_fail = 0  # PUTs that land and then answer 503 (a timeout after commit)
        self.seq = 0

    @staticmethod
    def _ok(payload=None, status=200):
        return SimpleNamespace(status_code=status, json=lambda: payload or {}, headers={})

    async def request(self, method, path, headers=None, params=None, json=None, content=None, **_):
        assert path.startswith("orgs/1/apps/2/session_runtime"), path
        self.requests.append((method, path))
        parts = path.split("/")[5:]
        headers, params = headers or {}, params or {}
        if not parts:
            with_work = params.get("with_work") == "true"
            ids = [
                sid
                for sid, row in self.rows.items()
                if not row["closed"] and (not with_work or row["has_work"] or self.inbox.get(sid))
            ]
            return self._ok({"session_ids": ids})
        sid, rest = parts[0], parts[1:]
        row = self.rows.get(sid)
        if not rest:
            if method == "GET":
                if row is None:
                    raise _error(404)
                return self._ok(
                    {k: row[k] for k in ("runtime", "runtime_version", "write_id", "lease_owner", "lease_until")}
                )
            if method == "PUT":
                body = globals()["json"].loads(content)
                if "If-None-Match" in headers:
                    if row is not None:
                        raise _error(412)
                    row = self.rows[sid] = {"runtime_version": 0, "lease_owner": None, "lease_until": 0.0}
                else:
                    expected = int(headers["If-Match"].strip('"'))
                    if row is None or row["runtime_version"] != expected:
                        raise _error(412)
                row.update(
                    runtime=body["runtime"],
                    has_work=body["has_work"],
                    closed=body["closed"],
                    write_id=body["write_id"],
                    runtime_version=row["runtime_version"] + 1,
                )
                if self.commit_then_fail:
                    self.commit_then_fail -= 1
                    raise _error(503)
                return self._ok({"runtime_version": row["runtime_version"]})
            if method == "DELETE":
                self.rows.pop(sid, None)
                self.inbox.pop(sid, None)
                return self._ok(status=204)
        if rest == ["lease"]:
            if row is None:
                raise _error(404)
            if method == "POST":
                owner, ttl = json["owner"], json["ttl_s"]
                if row["lease_owner"] not in (None, owner) and row["lease_until"] > time.time():
                    raise _error(409)
                row.update(lease_owner=owner, lease_until=time.time() + ttl)
                return self._ok({"lease_owner": owner, "lease_until": row["lease_until"]})
            if method == "DELETE":
                if row["lease_owner"] == params["owner"]:
                    row.update(lease_owner=None, lease_until=0.0)
                return self._ok(status=204)
        if rest == ["inbox"]:
            if row is None:
                raise _error(404)
            if method == "POST":
                self.seq += 1
                item = {"id": f"pin_{self.seq}", "kind": json["kind"], "body": json["body"], "posted_at": time.time()}
                self.inbox.setdefault(sid, []).append(item)
                if sid in self.changed:
                    self.changed[sid].set()
                return self._ok({"id": item["id"]}, status=201)
            if method == "GET":
                self._fence(row, params["owner"])
                if not self.inbox.get(sid) and params.get("wait_ms"):
                    changed = self.changed.setdefault(sid, asyncio.Event())
                    changed.clear()
                    try:
                        await asyncio.wait_for(changed.wait(), params["wait_ms"] / 1000)
                    except TimeoutError:
                        pass
                return self._ok({"items": list(self.inbox.get(sid, []))})
        if rest == ["inbox", "ack"]:
            done = set(json["ids"])
            self.inbox[sid] = [item for item in self.inbox.get(sid, []) if item["id"] not in done]
            return self._ok(status=204)
        raise AssertionError(f"unexpected {method} {path}")

    @staticmethod
    def _fence(row, owner):
        if row["lease_owner"] != owner or row["lease_until"] < time.time():
            raise _error(409)


@pytest.fixture
def platform():
    server = FakeRuntime()
    with patch("timbal.platform.utils._request", server.request):
        yield server, PlatformChatSessionStore(platform_config=CONFIG)


class TestContract:
    async def test_save_is_a_compare_and_swap_on_the_version(self, platform):
        server, store = platform
        snap = SessionSnapshot(session_id="s1", agent="agent", head="r1", pending=[{"id": "e", "text": "hi"}])
        assert await store.save(snap, expect_version=None) == 1
        with pytest.raises(SnapshotConflict):
            await store.save(snap, expect_version=None)
        assert await store.save(snap, expect_version=1) == 2
        with pytest.raises(SnapshotConflict):
            await store.save(snap, expect_version=1)
        loaded = await store.load("s1")
        assert loaded.version == 2 and loaded.head == "r1" and loaded.pending == [{"id": "e", "text": "hi"}]
        assert server.rows["s1"]["has_work"] is True and "version" not in server.rows["s1"]["runtime"]
        assert await store.list_ids(with_work=True) == ["s1"]
        await store.delete("s1")
        assert await store.load("s1") is None and await store.list_ids() == []

    async def test_a_save_that_landed_but_timed_out_is_not_a_takeover(self, platform):
        server, store = platform
        snap = SessionSnapshot(session_id="s1", agent="agent")
        await store.save(snap, expect_version=None)
        server.commit_then_fail = 1
        with pytest.raises(PlatformError):
            await store.save(snap, expect_version=1)  # landed as version 2; the caller does not know
        snap.head = "r2"
        assert await store.save(snap, expect_version=1) == 3  # our own write: on top of it, not a conflict
        server.commit_then_fail = 1
        with pytest.raises(PlatformError):
            await store.save(snap, expect_version=3)  # landed as 4
        server.rows["s1"].update(write_id="another-writer", runtime_version=5)  # ...and someone else wrote after
        with pytest.raises(SnapshotConflict):
            await store.save(snap, expect_version=3)

    async def test_leases_exclude_other_owners_until_they_expire(self, platform):
        server, store = platform
        await store.acquire_lease("nope", PROCESS_OWNER)  # nothing stored yet: no error, no lease
        await store.save(SessionSnapshot(session_id="s1", agent="agent"), expect_version=None)
        await store.acquire_lease("s1", "elsewhere:1:x")
        with pytest.raises(SessionLeased):
            await store.acquire_lease("s1", PROCESS_OWNER)
        server.rows["s1"]["lease_until"] = time.time() - 1
        await store.acquire_lease("s1", PROCESS_OWNER)
        await store.release_lease("s1", "elsewhere:1:x")  # not the holder: nothing
        assert server.rows["s1"]["lease_owner"] == PROCESS_OWNER
        await store.release_lease("s1", PROCESS_OWNER)
        assert server.rows["s1"]["lease_owner"] is None

    async def test_the_inbox_is_for_the_lease_holder_and_long_polls(self, platform):
        server, store = platform
        with pytest.raises(LookupError):
            await store.post_inbox("nope", "message", {"text": "hi"})
        await store.save(SessionSnapshot(session_id="s1", agent="agent"), expect_version=None)
        await store.acquire_lease("s1", PROCESS_OWNER)
        with pytest.raises(SessionLeased):
            await store.pull_inbox("s1", "elsewhere:1:x")
        waiting = asyncio.ensure_future(store.pull_inbox("s1", PROCESS_OWNER, wait_s=5))
        await asyncio.sleep(0.05)
        item_id = await store.post_inbox("s1", "wake", {"text": "check in"})
        items = await asyncio.wait_for(waiting, 1)
        assert [(i["id"], i["kind"], i["body"]) for i in items] == [(item_id, "wake", {"text": "check in"})]
        assert await store.list_ids(with_work=True) == ["s1"]
        await store.ack_inbox("s1", PROCESS_OWNER, [item_id])
        assert await store.pull_inbox("s1", PROCESS_OWNER) == []


def echo(messages):
    last = messages[-1]
    if last.role == "tool":
        return "done"
    text = next((m.collect_text() for m in reversed(messages) if m.role == "user"), "")
    if "start" in text:
        return Message(
            role="assistant", content=[ToolUseContent(id="w", name="work", input={})], stop_reason="tool_use"
        )
    return f"echo: {text[-60:]}"


def make_agent(gate=None):
    async def work() -> str:
        if gate is not None:
            await gate.wait()
        return "worked"

    return Agent(
        name="agent",
        model=TestModel(handler=echo),
        tools=[Tool(name="work", handler=work)],
        tracing_provider=InMemoryTracingProvider,
    )


async def records(session, kind, count, timeout=5.0):
    async def wait():
        while (
            sum(
                1
                for r in session.log.peek(session.log.forgotten_through)[0]
                if isinstance(r, dict) and r.get("kind") == kind
            )
            < count
        ):
            await asyncio.sleep(0.02)

    await asyncio.wait_for(wait(), timeout)
    return [
        r for r in session.log.peek(session.log.forgotten_through)[0] if isinstance(r, dict) and r.get("kind") == kind
    ]


def outputs(session):
    return [
        e.output.collect_text()
        for e in session.log.peek(session.log.forgotten_through)[0]
        if isinstance(e, OutputEvent) and e.path == "agent" and e.parent_call_id is None and e.output is not None
    ]


@pytest.fixture(params=["memory", "sqlite", "platform"])
def store(request, tmp_path):
    if request.param == "memory":
        yield InMemoryChatSessionStore()
    elif request.param == "sqlite":
        sqlite = SqliteChatSessionStore(tmp_path / "chat.db")
        sqlite.inbox_poll_s = 0.02
        yield sqlite
    else:
        server = FakeRuntime()
        with patch("timbal.platform.utils._request", server.request):
            yield PlatformChatSessionStore(platform_config=CONFIG)


class TestInbox:
    async def test_work_queued_by_another_process_runs_in_the_owners_session(self, store):
        session = ChatSession(make_agent(), session_store=store)
        session.send("hello")
        await records(session, "turn_finished", 1)
        # Another process: queues instead of serving the session it does not hold.
        await store.post_inbox(session.id, "message", {"text": "from worker B", "mode": "followup"})
        await store.post_inbox(
            session.id,
            "wake",
            {"text": "routine check", "tool_policy": None, "trigger": {"kind": "routine", "routine_id": "r1"}},
        )
        await store.post_inbox(
            session.id, "agent_message", {"from": "planner", "text": "build is green", "message_id": "am_1"}
        )
        await store.post_inbox(session.id, "wake", {"text": "bad"} | {"tool_policy": 3})  # rejected, not retried
        # The message and the wake get turns; the agent message rides along with the first of them.
        finished = await records(session, "turn_finished", 3)
        applied = await records(session, "inbox_item", 3)
        assert [r["item_kind"] for r in applied] == ["message", "wake", "agent_message"]
        assert len(await records(session, "inbox_rejected", 1)) == 1
        started = await records(session, "turn_started", 3)
        assert [s["trigger"]["kind"] for s in started] == ["user_message", "user_message", "routine"]
        assert session._agent_message("am_1").state == "delivered"
        assert all(f["status"]["code"] == "success" for f in finished)
        for _ in range(100):
            if not await store.pull_inbox(session.id, PROCESS_OWNER):
                break
            await asyncio.sleep(0.02)
        assert await store.pull_inbox(session.id, PROCESS_OWNER) == []  # all acked
        await session.close()

    async def test_items_applied_but_not_acked_are_not_applied_again_after_a_restore(self, store):
        session = ChatSession(make_agent(), session_store=store)
        session.send("hello")
        await records(session, "turn_finished", 1)
        item = await store.post_inbox(session.id, "message", {"text": "once", "mode": "followup"})
        await records(session, "inbox_item", 1)
        await records(session, "turn_finished", 2)
        await session.close(keep=True)
        # As if the ack had been lost: the item is queued again for the next owner.
        if isinstance(store, InMemoryChatSessionStore):
            store._inbox[session.id] = [{"id": item, "kind": "message", "body": {"text": "once"}, "posted_at": 0.0}]
        else:
            pytest.skip("re-queues the item through the in-memory store's internals; the others via inbox_seen below")
        restored = await ChatSession.from_store(make_agent(), store, session.id)
        try:
            await asyncio.sleep(0.3)
            assert item in restored._inbox_seen
            assert not await records(restored, "inbox_item", 0)
            assert await store.pull_inbox(session.id, PROCESS_OWNER) == []  # acked by the new owner
        finally:
            await restored.close()

    async def test_the_seen_ids_survive_in_the_snapshot(self, store):
        session = ChatSession(make_agent(), session_store=store)
        session.send("hello")
        await records(session, "turn_finished", 1)
        item = await store.post_inbox(session.id, "message", {"text": "once", "mode": "followup"})
        await records(session, "inbox_item", 1)
        await session.flush()
        assert item in (await store.load(session.id)).inbox_seen
        await session.close()


class TestServerQueuesForTheOwner:
    def test_a_request_for_a_session_another_worker_holds_is_queued_in_its_inbox(self, tmp_path, monkeypatch):
        from pathlib import Path

        from fastapi.testclient import TestClient
        from timbal.server.http import create_app

        fixtures = Path(__file__).parents[1] / "server" / "fixtures"
        db = tmp_path / "chat.db"
        monkeypatch.setenv("TIMBAL_RUNNABLE", f"{fixtures / 'chat_agent_fixture.py'}::chat_agent")
        monkeypatch.setenv("TIMBAL_CHAT_STORE", f"sqlite:///{db}")
        store = SqliteChatSessionStore(db)

        async def held_elsewhere():
            await store.save(SessionSnapshot(session_id="held", agent="chat_agent", head="r0"), expect_version=None)
            await store.acquire_lease("held", "elsewhere:1:x")

        asyncio.run(held_elsewhere())
        with TestClient(create_app()) as client:
            sent = client.post("/chat/sessions/held/messages", json={"text": "hi", "mode": "followup"})
            assert sent.status_code == 202, sent.text
            assert sent.json()["queued"] == "inbox"
            woke = client.post("/chat/sessions/held/wake", json={"text": "routine", "tool_policy": None})
            assert woke.status_code == 202 and woke.json()["queued"] == "inbox"
            agent = client.post("/chat/sessions/held/agent_messages", json={"from": "planner", "text": "x"})
            assert agent.status_code == 202 and agent.json()["queued"] == "inbox"
            assert client.post("/chat/sessions/held/messages", json={"text": "hi", "mode": "loud"}).status_code == 400
            assert client.post("/chat/sessions/nope/messages", json={"text": "hi"}).status_code == 404

        async def queued():
            store.lease_ttl_s = 30
            conn_items = []

            def read(conn):
                return conn.execute("SELECT item FROM chat_session_inbox WHERE session_id = 'held'").fetchall()

            for (raw,) in await store._run(read):
                conn_items.append(json.loads(raw))
            return conn_items

        items = asyncio.run(queued())
        assert [(i["kind"], i["body"].get("text")) for i in items] == [
            ("message", "hi"),
            ("wake", "routine"),
            ("agent_message", "x"),
        ]
        assert items[1]["body"]["tool_policy"] is None and items[2]["body"]["priority"] == "normal"
