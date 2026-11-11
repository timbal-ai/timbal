"""Durable chat sessions over HTTP, in one process: lazy restore, one restore per id, 409, delete, the store choice."""

import asyncio
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from timbal.core.chat_session import ChatSession
from timbal.server.chat import session_store_for
from timbal.server.http import create_app
from timbal.state.chat_store import SessionSnapshot, SqliteChatSessionStore, process_owner

AGENT = textwrap.dedent(
    """
    import asyncio, os
    from pathlib import Path
    from timbal import Agent, Tool
    from timbal.core.test_model import TestModel
    from timbal.state.tracing.providers import SqliteTracingProvider
    from timbal.types.content import ToolUseContent
    from timbal.types.message import Message

    DIR = Path(os.environ["E2E_DIR"])

    async def wait() -> str:
        while (DIR / "hang").exists():  # a tool that runs for as long as the test wants a turn busy
            await asyncio.sleep(0.02)
        return "waited"

    def handler(messages):
        text = next((m.collect_text() for m in reversed(messages) if m.role == "user"), "")
        if messages[-1].role == "tool":
            return "done"
        if text == "wait":
            return Message(role="assistant", content=[ToolUseContent(id="w1", name="wait", input={})],
                           stop_reason="tool_use")
        return f"echo: {text}"

    agent = Agent(
        name="agent",
        model=TestModel(handler=handler),
        tools=[Tool(name="wait", handler=wait)],
        tracing_provider=SqliteTracingProvider.configured(_path=DIR / "timbal.db"),
    )
    """
)


@pytest.fixture
def client(tmp_path, monkeypatch):
    (tmp_path / "agent.py").write_text(AGENT)
    monkeypatch.setenv("E2E_DIR", str(tmp_path))
    monkeypatch.setenv("TIMBAL_RUNNABLE", f"{tmp_path / 'agent.py'}::agent")
    monkeypatch.delenv("TIMBAL_CHAT_STORE", raising=False)
    with TestClient(create_app()) as client:
        yield client


def wait_turns(client, session_id, turns, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = client.get(f"/chat/sessions/{session_id}").json()
        if state.get("turns") == turns and not state["busy"]:
            return state
        time.sleep(0.02)
    raise AssertionError(f"session {session_id} did not reach {turns} turns")


def forget(client, session_id):
    """What a restart or idle reaping leaves: the session gone from this process, kept in the store."""
    session = client.app.state.chat_sessions.pop(session_id)
    client.portal.call(lambda: session.close(keep=True))


def test_meta_says_durable(client):
    assert client.get("/chat/meta").json()["durable"] is True


def test_a_session_this_process_does_not_hold_is_restored_on_first_use(client):
    session_id = client.post("/chat/sessions", json={}).json()["session_id"]
    client.post(f"/chat/sessions/{session_id}/messages", json={"text": "hi"})
    wait_turns(client, session_id, 1)
    forget(client, session_id)

    state = client.get(f"/chat/sessions/{session_id}").json()
    assert state["durable"] and state["turns"] == 1
    client.post(f"/chat/sessions/{session_id}/messages", json={"text": "again"})
    assert wait_turns(client, session_id, 2)["last_status"]["code"] == "success"


def test_idle_reaping_keeps_the_session_for_later(client, monkeypatch):
    first = client.post("/chat/sessions", json={}).json()["session_id"]
    client.post(f"/chat/sessions/{first}/messages", json={"text": "hi"})
    wait_turns(client, first, 1)
    monkeypatch.setenv("TIMBAL_CHAT_SESSION_TTL_S", "0.01")
    time.sleep(0.05)
    client.post("/chat/sessions", json={})  # creating a session reaps idle ones
    assert first not in client.app.state.chat_sessions
    monkeypatch.setenv("TIMBAL_CHAT_SESSION_TTL_S", "3600")
    assert client.get(f"/chat/sessions/{first}").json()["turns"] == 1


def test_requests_racing_for_one_session_restore_it_once(client, monkeypatch):
    session_id = client.post("/chat/sessions", json={}).json()["session_id"]
    forget(client, session_id)
    real, calls = ChatSession.from_store.__func__, []

    async def slow_restore(cls, *args, **kwargs):
        calls.append(1)
        await asyncio.sleep(0.2)  # every request arrives while the first restore is still going
        return await real(cls, *args, **kwargs)

    monkeypatch.setattr(ChatSession, "from_store", classmethod(slow_restore))
    with ThreadPoolExecutor(6) as pool:
        codes = list(pool.map(lambda _: client.get(f"/chat/sessions/{session_id}").status_code, range(6)))
    assert codes == [200] * 6 and len(calls) == 1


def test_a_session_another_process_serves_is_a_409(client):
    session_id = client.post("/chat/sessions", json={}).json()["session_id"]
    forget(client, session_id)
    store = client.app.state.chat_store
    client.portal.call(store.acquire_lease, session_id, "elsewhere:1:x")
    response = client.get(f"/chat/sessions/{session_id}")
    assert response.status_code == 409 and "another process" in response.json()["detail"]


def test_a_session_this_process_lost_answers_409_then_comes_back(client):
    """Losing the lease closes the session here: requests must not keep finding the closed object (404)."""
    session_id = client.post("/chat/sessions", json={}).json()["session_id"]
    store = client.app.state.chat_store
    session = client.app.state.chat_sessions[session_id]
    client.portal.call(store.release_lease, session_id, process_owner())  # lapsed
    client.portal.call(store.acquire_lease, session_id, "elsewhere:1:x")

    async def lose():
        session._lose("taken")

    client.portal.call(lose)
    response = client.get(f"/chat/sessions/{session_id}")
    assert response.status_code == 409, response.text
    assert client.post(f"/chat/sessions/{session_id}/messages", json={"text": "hi"}).status_code == 409
    client.portal.call(store.release_lease, session_id, "elsewhere:1:x")  # the other process let it go
    assert client.get(f"/chat/sessions/{session_id}").status_code == 200
    assert client.app.state.chat_sessions[session_id] is not session


def test_a_session_still_closing_is_restored_only_once_its_close_is_done(client, monkeypatch):
    """Until close() ends, the old copy may run its turn and holds the session's task store: a restore started
    before that would bind a second worker to it."""
    session_id = client.post("/chat/sessions", json={}).json()["session_id"]
    session = client.app.state.chat_sessions[session_id]
    order = []
    real_settle = session._settle_store

    async def slow_settle():
        await asyncio.sleep(0.3)
        await real_settle()
        order.append("closed")

    session._settle_store = slow_settle
    real_restore = ChatSession.from_store.__func__

    async def restore(cls, *args, **kwargs):
        order.append("restore")
        return await real_restore(cls, *args, **kwargs)

    monkeypatch.setattr(ChatSession, "from_store", classmethod(restore))

    async def close_later():
        await session.close(keep=True)

    client.portal.start_task_soon(close_later)
    client.portal.call(asyncio.sleep, 0.05)  # close() has started: handed off, still settling
    assert session.handed_off and session_id in client.app.state.chat_sessions
    assert client.get(f"/chat/sessions/{session_id}").status_code == 200
    assert order == ["closed", "restore"]
    assert client.app.state.chat_sessions[session_id] is not session


def test_a_message_that_could_not_be_stored_is_a_503_and_not_kept(client, monkeypatch):
    # A long collect window: no turn takes the message before the route answers, so the route can take it back.
    session_id = client.post("/chat/sessions", json={"collect_window": 30}).json()["session_id"]

    monkeypatch.setattr(client.app.state.chat_store, "save", _down)
    response = client.post(f"/chat/sessions/{session_id}/messages", json={"text": "hi", "mode": "collect"})
    assert response.status_code == 503 and response.headers["retry-after"] == "1"
    assert "Nothing was accepted" in response.json()["detail"]
    monkeypatch.undo()
    state = client.get(f"/chat/sessions/{session_id}").json()
    assert state["pending"] == []
    assert client.app.state.chat_sessions[session_id]._store.pending_events() == []  # nothing left to run
    assert client.get(f"/chat/sessions/{session_id}").json()["turns"] == 0


async def _down(*_args, **_kwargs):
    raise OSError("disk full")


@pytest.mark.parametrize("mode", ["interrupt", "now"])
def test_a_503_for_a_cutting_message_has_not_cut_the_turn(client, monkeypatch, tmp_path, mode):
    """'Nothing was accepted' must be true of the running turn too: no cancel, no cut, before the store has it."""
    (tmp_path / "hang").touch()
    session_id = client.post("/chat/sessions", json={}).json()["session_id"]
    client.post(f"/chat/sessions/{session_id}/messages", json={"text": "wait"})
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not client.get(f"/chat/sessions/{session_id}").json()["busy"]:
        time.sleep(0.02)
    running = client.get(f"/chat/sessions/{session_id}").json()["run_id"]
    assert running

    monkeypatch.setattr(client.app.state.chat_store, "save", _down)
    response = client.post(f"/chat/sessions/{session_id}/messages", json={"text": "stop", "mode": mode})
    assert response.status_code == 503 and response.headers["retry-after"] == "1", response.text
    monkeypatch.undo()
    state = client.get(f"/chat/sessions/{session_id}").json()
    assert state["busy"] and state["run_id"] == running and state["pending"] == []
    (tmp_path / "hang").unlink()
    assert wait_turns(client, session_id, 1)["last_status"]["code"] == "success"  # it ran to its end, uncut


def test_a_message_the_store_already_holds_is_acknowledged_when_a_later_write_fails(client, monkeypatch):
    """The 503 is for a change the store does not hold. Taking back one it does would leave it there for a
    restart to run, next to the client's retry."""
    session_id = client.post("/chat/sessions", json={"collect_window": 3}).json()["session_id"]
    session, store = client.app.state.chat_sessions[session_id], client.app.state.chat_store
    real_save, calls = store.save, []

    async def save(snapshot, *, expect_version, owner=None):
        if calls:
            raise OSError("disk full")
        calls.append(1)
        version = await real_save(snapshot, expect_version=expect_version, owner=owner)
        session._persist()  # a later change: its write, and the retry of it, fail
        return version

    monkeypatch.setattr(store, "save", save)
    response = client.post(f"/chat/sessions/{session_id}/messages", json={"text": "hi", "mode": "collect"})
    assert response.status_code == 202, response.text
    event_id = response.json()["message"]["event_id"]
    assert [e["id"] for e in client.portal.call(store.load, session_id).pending] == [event_id]
    assert [p["event_id"] for p in session.pending_messages()] == [event_id]  # not taken back
    monkeypatch.undo()
    assert wait_turns(client, session_id, 1, timeout=20)["last_status"]["code"] == "success"


def test_a_delete_the_store_did_not_take_is_a_503_and_a_retry_finishes_it(client, monkeypatch):
    session_id = client.post("/chat/sessions", json={}).json()["session_id"]
    store = client.app.state.chat_store
    monkeypatch.setattr(store, "save", _down)
    monkeypatch.setattr(store, "delete", _down)
    response = client.delete(f"/chat/sessions/{session_id}")
    assert response.status_code == 503 and response.headers["retry-after"] == "1"
    monkeypatch.undo()
    assert client.portal.call(store.load, session_id) is not None  # still there, open: it would come back
    assert client.delete(f"/chat/sessions/{session_id}").json() == {"closed": True}
    assert client.portal.call(store.load, session_id) is None


def test_a_session_this_process_serves_is_never_restored_a_second_time(client, monkeypatch):
    """The startup retry and a request can both ask: this process renews its own lease, so nothing else stops
    a second copy."""
    from timbal.server.chat import _restore_once

    session_id = client.post("/chat/sessions", json={}).json()["session_id"]
    session = client.app.state.chat_sessions[session_id]

    async def never(_cls, *_args, **_kwargs):
        raise AssertionError("restored a session this process already serves")

    monkeypatch.setattr(ChatSession, "from_store", classmethod(never))
    app = client.app
    again = client.portal.call(_restore_once, app, app.state.runnable, app.state.chat_store, session_id)
    assert again is session


OLD_OWNER = "old-host:1:x"  # another host's: alive for as long as its lease has not run out


def _seed_leased_session(tmp_path, monkeypatch):
    """A stored session with a message waiting, leased by a predecessor on another host for a minute: startup
    skips it however slow the runner is, and retries every 0.1 s. The test ends the old lease when it wants."""
    (tmp_path / "agent.py").write_text(AGENT)
    monkeypatch.setenv("E2E_DIR", str(tmp_path))
    monkeypatch.setenv("TIMBAL_RUNNABLE", f"{tmp_path / 'agent.py'}::agent")
    monkeypatch.delenv("TIMBAL_CHAT_STORE", raising=False)
    monkeypatch.setattr(SqliteChatSessionStore, "lease_ttl_s", 0.3)  # the server's: a retry every 0.1 s
    store = SqliteChatSessionStore(tmp_path / "timbal.db")
    store.lease_ttl_s = 60.0  # this instance's: how long the predecessor's lease lasts
    message = {
        "id": "m1",
        "text": "hi",
        "source": "user",
        "lane": "later",
        "mode": "followup",
        "posted_at": time.time(),
    }

    async def seed():
        await store.save(SessionSnapshot(session_id="s1", agent="agent", pending=[message]), expect_version=None)
        await store.acquire_lease("s1", OLD_OWNER)

    asyncio.run(seed())
    return store


def _lease_owner(store, session_id):
    def query(conn):
        return conn.execute("SELECT lease_owner FROM chat_sessions WHERE session_id = ?", (session_id,)).fetchone()

    return asyncio.run(store._run(query))[0]


def test_shutdown_waits_for_a_restore_in_flight_and_closes_its_session(tmp_path, monkeypatch):
    """A restore the cancelled startup retry was waiting for still finishes: its session must be closed with
    the others (kept, its lease let go), not left running through the shutdown."""
    store = _seed_leased_session(tmp_path, monkeypatch)
    real_restore, restored = ChatSession.from_store.__func__, []

    async def slow_restore(cls, *args, **kwargs):
        session = await real_restore(cls, *args, **kwargs)
        restored.append(session)
        await asyncio.sleep(0.5)  # shutdown comes while the restore is still going
        return session

    monkeypatch.setattr(ChatSession, "from_store", classmethod(slow_restore))
    with TestClient(create_app()) as client:
        assert not restored  # skipped at startup
        client.portal.call(client.app.state.chat_store.release_lease, "s1", OLD_OWNER)  # the old lease ends
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not restored:
            time.sleep(0.02)
        assert restored
    (session,) = restored
    assert session.closed and session.handed_off
    assert _lease_owner(store, "s1") is None  # let go for the next process


def test_startup_retries_a_session_whose_old_lease_had_not_run_out(tmp_path, monkeypatch):
    """A replacement starting before its predecessor's lease expires restores the session once it does."""
    _seed_leased_session(tmp_path, monkeypatch)
    with TestClient(create_app()) as client:
        sessions = client.app.state.chat_sessions
        assert "s1" not in sessions  # skipped at startup: the lease was still live
        client.portal.call(client.app.state.chat_store.release_lease, "s1", OLD_OWNER)  # ...and now it ends
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not (sessions.get("s1") and sessions["s1"].turns == 1):
            time.sleep(0.05)
        assert sessions.get("s1") is not None and sessions["s1"].turns == 1


def test_a_session_being_deleted_is_not_restored_meanwhile(client):
    """Until its row is gone the session stays in the map, closed: a request then is a 404, not a restore of
    the row that is still there."""
    session_id = client.post("/chat/sessions", json={}).json()["session_id"]
    session = client.app.state.chat_sessions[session_id]

    async def a_worker_slow_to_stop():
        async def worker():
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                await asyncio.sleep(1.0)  # its cleanup: close() waits here, the row still open in the store

        if session._worker is not None:
            session._worker.cancel()
        session._worker = asyncio.get_running_loop().create_task(worker())

    client.portal.call(a_worker_slow_to_stop)
    with ThreadPoolExecutor(1) as pool:
        deleting = pool.submit(client.delete, f"/chat/sessions/{session_id}")
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not session.closed:
            time.sleep(0.01)
        assert session.closed
        assert client.get(f"/chat/sessions/{session_id}").status_code == 404
        assert deleting.result(10).json() == {"closed": True}
    assert client.get(f"/chat/sessions/{session_id}").status_code == 404
    assert session_id not in client.app.state.chat_sessions
    assert client.portal.call(client.app.state.chat_store.load, session_id) is None


def test_deleting_a_session_another_process_took_is_a_409(client):
    session_id = client.post("/chat/sessions", json={}).json()["session_id"]
    store = client.app.state.chat_store
    client.portal.call(store.release_lease, session_id, process_owner())  # lapsed, before this copy noticed...
    client.portal.call(store.acquire_lease, session_id, "elsewhere:1:x")  # ...and another worker has it
    response = client.delete(f"/chat/sessions/{session_id}")
    assert response.status_code == 409 and "another process" in response.json()["detail"]
    stored = client.portal.call(store.load, session_id)
    assert stored is not None and not stored.closed  # it goes on there


def test_a_session_is_created_with_its_lease_in_one_write(client, monkeypatch):
    """No second write to take the lease: one that fails cannot leave a row nobody holds or knows of."""
    store = client.app.state.chat_store

    async def unreachable(*_args, **_kwargs):
        raise OSError("store unreachable")

    monkeypatch.setattr(store, "acquire_lease", unreachable)
    response = client.post("/chat/sessions", json={})
    assert response.status_code == 201, response.text
    session_id = response.json()["session_id"]
    monkeypatch.undo()
    assert session_id in [state["session_id"] for state in client.get("/chat/sessions").json()]

    def owner(conn):
        return conn.execute("SELECT lease_owner FROM chat_sessions WHERE session_id = ?", (session_id,)).fetchone()

    assert client.portal.call(store._run, owner)[0] == process_owner()


def test_delete_removes_it_for_good(client):
    session_id = client.post("/chat/sessions", json={}).json()["session_id"]
    assert client.delete(f"/chat/sessions/{session_id}").json() == {"closed": True}
    assert client.portal.call(client.app.state.chat_store.load, session_id) is None
    assert client.get(f"/chat/sessions/{session_id}").status_code == 404


def test_the_store_choice(tmp_path, monkeypatch):
    from timbal import Agent
    from timbal.core.test_model import TestModel
    from timbal.state.tracing.providers import InMemoryTracingProvider, SqliteTracingProvider

    sqlite_agent = Agent(
        name="a", model=TestModel(), tracing_provider=SqliteTracingProvider.configured(_path=tmp_path / "t.db")
    )
    memory_agent = Agent(name="a", model=TestModel(), tracing_provider=InMemoryTracingProvider)
    monkeypatch.delenv("TIMBAL_CHAT_STORE", raising=False)
    assert isinstance(session_store_for(sqlite_agent), SqliteChatSessionStore)
    assert session_store_for(memory_agent) is None
    monkeypatch.setenv("TIMBAL_CHAT_STORE", "memory")
    assert session_store_for(sqlite_agent) is None
    monkeypatch.setenv("TIMBAL_CHAT_STORE", f"sqlite:///{tmp_path}/s.db")
    assert session_store_for(memory_agent).path == tmp_path / "s.db"  # four slashes: absolute
    monkeypatch.setenv("TIMBAL_CHAT_STORE", "postgres://x")
    with pytest.raises(RuntimeError, match="TIMBAL_CHAT_STORE"):
        session_store_for(memory_agent)
