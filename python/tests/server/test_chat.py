"""Chat sessions over HTTP: /chat/sessions, the SSE log, and the playground page."""

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from timbal.server.http import create_app

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("TIMBAL_RUNNABLE", f"{FIXTURES / 'chat_agent_fixture.py'}::chat_agent")
    with TestClient(create_app()) as client:
        yield client


def open_session(client, **body):
    response = client.post("/chat/sessions", json=body)
    assert response.status_code == 201, response.text
    return response.json()["session_id"]


def read_log(client, session_id, *, until_kind, count, after=0, timeout=5.0):
    """``(seq, item)`` pairs from the long-polled log, up to the ``count``-th SESSION record of ``until_kind``."""
    items, seen, cursor, deadline = [], 0, after, time.monotonic() + timeout
    while time.monotonic() < deadline:
        page = client.get(f"/chat/sessions/{session_id}/log", params={"after": cursor, "wait_ms": 500}).json()
        for entry in page["events"]:
            item = entry["data"]
            items.append((entry["seq"], item))
            if item.get("type") == "SESSION" and item.get("kind") == until_kind:
                seen += 1
                if seen >= count:
                    return items
        cursor = page["next_cursor"]
        assert not page["closed"], "the session closed early"
    raise AssertionError("the log did not reach the expected record in time")


def agent_outputs(items):
    return [item for _, item in items if item.get("type") == "OUTPUT" and item.get("parent_call_id") is None]


def text(output):
    return "".join(c.get("text", "") for c in (output.get("output") or {}).get("content", []))


def test_page_and_meta(client):
    page = client.get("/chat/")
    assert page.status_code == 200 and "Chat playground" in page.text
    meta = client.get("/chat/meta").json()
    assert meta["agent"] == "chat_agent" and meta["is_agent"] is True
    assert meta["modes"] == ["steer", "followup", "collect", "interrupt"]


def test_steer_joins_the_running_turn(client):
    session_id = open_session(client)
    client.post(f"/chat/sessions/{session_id}/messages", json={"text": "start"})
    time.sleep(0.15)
    sent = client.post(f"/chat/sessions/{session_id}/messages", json={"text": "also this", "mode": "steer"})
    assert sent.status_code == 202 and sent.json()["busy"] is True
    items = read_log(client, session_id, until_kind="turn_finished", count=1)
    (output,) = agent_outputs(items)
    assert text(output) == "echo: also this"
    settled = [item for _, item in items if item.get("kind") == "message_settled"]
    assert {item["state"] for item in settled} == {"delivered"}
    assert {item["run_id"] for item in settled} == {output["run_id"]}
    state = client.get(f"/chat/sessions/{session_id}").json()
    assert state["turns"] == 1 and state["busy"] is False and state["pending"] == []


def test_followup_gets_its_own_turn_and_the_log_resumes_from_a_cursor(client):
    session_id = open_session(client)
    client.post(f"/chat/sessions/{session_id}/messages", json={"text": "start"})
    time.sleep(0.15)
    client.post(f"/chat/sessions/{session_id}/messages", json={"text": "later", "mode": "followup"})
    items = read_log(client, session_id, until_kind="turn_finished", count=2)
    assert [text(o) for o in agent_outputs(items)] == ["done", "echo: later"]
    seqs = [seq for seq, _ in items]
    assert seqs == list(range(1, len(items) + 1))
    tail = read_log(client, session_id, until_kind="turn_finished", count=1, after=seqs[-3])
    assert tail[0][0] == seqs[-3] + 1


def test_sse_replays_the_log_with_cursors(client):
    session_id = open_session(client)
    client.post(f"/chat/sessions/{session_id}/messages", json={"text": "hi"})
    polled = read_log(client, session_id, until_kind="turn_finished", count=1)
    # The test client buffers a streaming body until it ends: close the session so the stream does.
    client.portal.call(client.app.state.chat_sessions[session_id].close)
    with client.stream("GET", f"/chat/sessions/{session_id}/events", params={"after": 1}) as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        lines = list(response.iter_lines())
    ids = [int(line[4:]) for line in lines if line.startswith("id: ")]
    data = [json.loads(line[6:]) for line in lines if line.startswith("data: ")]
    assert ids == list(range(2, len(ids) + 2)) and len(data) == len(ids)
    assert len(ids) == len(polled)  # everything after the first, plus the "closed" record
    assert data[-1]["kind"] == "closed"
    last = client.get(f"/chat/sessions/{session_id}/log", params={"after": ids[-1] - 1}).json()
    assert last["closed"] is True and last["events"][-1]["data"]["kind"] == "closed"


def test_interrupt_endpoint_cuts_the_turn(client):
    session_id = open_session(client)
    client.post(f"/chat/sessions/{session_id}/messages", json={"text": "start"})
    time.sleep(0.15)
    assert client.post(f"/chat/sessions/{session_id}/interrupt").json() == {"interrupted": True}
    items = read_log(client, session_id, until_kind="turn_finished", count=1)
    finished = [item for _, item in items if item.get("kind") == "turn_finished"]
    assert finished[0]["status"]["reason"] == "interrupted"
    assert client.post(f"/chat/sessions/{session_id}/interrupt").json() == {"interrupted": False}


def test_wake_starts_a_read_only_turn(client):
    session_id = open_session(client)
    woke = client.post(f"/chat/sessions/{session_id}/wake", json={"text": "anything new?", "key": "heartbeat"})
    assert woke.status_code == 202 and woke.json()["wake_id"]
    items = read_log(client, session_id, until_kind="turn_finished", count=1)
    (started,) = [item for _, item in items if item.get("kind") == "turn_started"]
    assert started["trigger"]["kind"] == "wake" and started["trigger"]["tool_policy"] == "read_only"
    (wake,) = [item for _, item in items if item.get("kind") == "wake"]
    assert wake["key"] == "heartbeat"


def test_wake_validation(client):
    session_id = open_session(client)
    url = f"/chat/sessions/{session_id}/wake"
    assert client.post(url, json={}).status_code == 400
    assert client.post(url, json={"text": "  "}).status_code == 400
    assert client.post(url, json={"text": "x", "tool_policy": "everything"}).status_code == 400
    assert client.post(url, json={"text": "x", "tool_policy": ["work"]}).status_code == 202
    assert client.post(url, json={"text": "x", "tool_policy": None}).status_code == 202


def test_errors(client):
    assert client.get("/chat/sessions/nope").status_code == 404
    assert client.post("/chat/sessions/nope/messages", json={"text": "x"}).status_code == 404
    session_id = open_session(client)
    assert client.post(f"/chat/sessions/{session_id}/messages", json={"mode": "steer"}).status_code == 400
    bad_mode = client.post(f"/chat/sessions/{session_id}/messages", json={"text": "x", "mode": "later"})
    assert bad_mode.status_code == 400
    assert client.post(f"/chat/sessions/{session_id}/resume", json={"resume": {"a": True}}).status_code == 409
    assert client.post("/chat/sessions", json={"busy_mode": "later"}).status_code == 400
    assert client.delete(f"/chat/sessions/{session_id}").json() == {"closed": True}
    assert client.get(f"/chat/sessions/{session_id}").status_code == 404


def test_a_conflicting_answer_is_a_409(client, monkeypatch):
    from timbal.core.chat_session import ChatSession

    def conflicting(_self, _values):
        raise ValueError("Already answered differently: a.")

    session_id = open_session(client)
    monkeypatch.setattr(ChatSession, "resume", conflicting)
    response = client.post(f"/chat/sessions/{session_id}/resume", json={"resume": {"a": False}})
    assert response.status_code == 409 and "answered differently" in response.json()["detail"]


def test_an_unreadable_parent_is_a_503(client, monkeypatch):
    from timbal.core.chat_session import ChatSession

    async def unreadable(_self):
        raise ConnectionError("trace store down")

    monkeypatch.setattr(ChatSession, "restore", unreadable)
    response = client.post("/chat/sessions", json={"parent_id": "some-run"})
    assert response.status_code == 503 and "trace store down" in response.json()["detail"]
    assert client.get("/chat/sessions").json() == []


def test_a_runnable_that_is_not_an_agent(monkeypatch):
    monkeypatch.setenv("TIMBAL_RUNNABLE", f"{FIXTURES / 'tool_fixture.py'}::tool_fixture")
    with TestClient(create_app()) as client:
        assert client.get("/chat/meta").json()["is_agent"] is False
        assert client.post("/chat/sessions", json={}).status_code == 400
