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
from timbal.state.chat_store import SqliteChatSessionStore

AGENT = textwrap.dedent(
    """
    import os
    from pathlib import Path
    from timbal import Agent
    from timbal.core.test_model import TestModel
    from timbal.state.tracing.providers import SqliteTracingProvider

    def handler(messages):
        text = next((m.collect_text() for m in reversed(messages) if m.role == "user"), "")
        return f"echo: {text}"

    agent = Agent(
        name="agent",
        model=TestModel(handler=handler),
        tracing_provider=SqliteTracingProvider.configured(_path=Path(os.environ["E2E_DIR"]) / "timbal.db"),
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
    real, calls = ChatSession.restore.__func__, []

    async def slow_restore(cls, *args, **kwargs):
        calls.append(1)
        await asyncio.sleep(0.2)  # every request arrives while the first restore is still going
        return await real(cls, *args, **kwargs)

    monkeypatch.setattr(ChatSession, "restore", classmethod(slow_restore))
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
