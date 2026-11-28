"""The Timbal server queueing a request for a session another live process serves (``POST .../messages``,
``.../wake``, ``.../agent_messages``) in the store's inbox: ``202 {"queued": "inbox", "inbox_id"}`` instead of the
``409`` a store without an inbox still answers."""

import asyncio
import textwrap
import time

import pytest
from fastapi.testclient import TestClient
from timbal.server.http import create_app
from timbal.state.chat_store import process_owner

AGENT = textwrap.dedent(
    """
    import os
    from pathlib import Path
    from timbal import Agent
    from timbal.core.test_model import TestModel
    from timbal.state.tracing.providers import SqliteTracingProvider

    DIR = Path(os.environ["E2E_DIR"])

    def handler(messages):
        text = next((m.collect_text() for m in reversed(messages) if m.role == "user"), "")
        return f"echo: {text.splitlines()[-1][:60]}"

    agent = Agent(
        name="agent",
        model=TestModel(handler=handler),
        tracing_provider=SqliteTracingProvider.configured(_path=DIR / "timbal.db"),
    )
    """
)

ELSEWHERE = "elsewhere:1:x"  # another host's process: only its lease's expiry would free the session


@pytest.fixture
def client(tmp_path, monkeypatch):
    (tmp_path / "agent.py").write_text(AGENT)
    monkeypatch.setenv("E2E_DIR", str(tmp_path))
    monkeypatch.setenv("TIMBAL_RUNNABLE", f"{tmp_path / 'agent.py'}::agent")
    monkeypatch.delenv("TIMBAL_CHAT_STORE", raising=False)
    with TestClient(create_app()) as client:
        yield client


def served_elsewhere(client) -> str:
    """A stored session that another live process serves now: this one let it go, the other took it."""
    session_id = client.post("/chat/sessions", json={"busy_mode": "followup"}).json()["session_id"]
    session = client.app.state.chat_sessions.pop(session_id)
    client.portal.call(lambda: session.close(keep=True))
    client.portal.call(client.app.state.chat_store.acquire_lease, session_id, ELSEWHERE)
    return session_id


def waiting(client, session_id):
    return client.portal.call(client.app.state.chat_store.pull_inbox, session_id, ELSEWHERE)


def post(client, session_id, route, body, **headers):
    return client.post(f"/chat/sessions/{session_id}/{route}", json=body, headers=headers)


class TestQueued:
    def test_a_message_is_queued_for_the_process_that_serves_the_session(self, client):
        session_id = served_elsewhere(client)
        assert client.get(f"/chat/sessions/{session_id}").status_code == 409  # reads still say where it is
        response = post(client, session_id, "messages", {"text": "hi", "mode": "steer"})
        assert response.status_code == 202, response.text
        answer = response.json()
        assert answer == {"queued": "inbox", "inbox_id": answer["inbox_id"], "session_id": session_id}
        (item,) = waiting(client, session_id)
        assert (item["id"], item["kind"], item["body"]) == (
            answer["inbox_id"],
            "message",
            {"text": "hi", "mode": "steer"},
        )

    def test_a_wake_is_queued(self, client):
        session_id = served_elsewhere(client)
        body = {"text": "check in", "key": "daily", "tool_policy": ["lookup"], "trigger": {"kind": "routine"}}
        response = post(client, session_id, "wake", body)
        assert response.status_code == 202 and response.json()["queued"] == "inbox", response.text
        (item,) = waiting(client, session_id)
        assert item["kind"] == "wake"
        assert item["body"] == {**body, "silent_ok": True}

    def test_an_agent_message_is_queued(self, client):
        session_id = served_elsewhere(client)
        body = {"from": "bob", "text": "done", "message_id": "am_1", "priority": "urgent", "hops": 2}
        response = post(client, session_id, "agent_messages", body)
        assert response.status_code == 202 and response.json()["queued"] == "inbox", response.text
        (item,) = waiting(client, session_id)
        assert item["kind"] == "agent_message"
        assert item["body"] == {**body, "reply_to": None}

    def test_what_was_queued_is_applied_once_the_session_is_served_here(self, client):
        session_id = served_elsewhere(client)
        post(client, session_id, "messages", {"text": "one"})
        post(client, session_id, "wake", {"text": "check in"})
        post(client, session_id, "agent_messages", {"from": "bob", "text": "two", "message_id": "am_1"})
        client.portal.call(client.app.state.chat_store.release_lease, session_id, ELSEWHERE)  # it let go
        assert client.get(f"/chat/sessions/{session_id}").status_code == 200  # restored here
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            state = client.get(f"/chat/sessions/{session_id}").json()
            log = client.get(f"/chat/sessions/{session_id}/log", params={"after": 0, "limit": 2000}).json()
            applied = [
                e["data"]
                for e in log["events"]
                if isinstance(e["data"], dict) and e["data"].get("kind") == "inbox_item"
            ]
            # (the agent's message rides with the user's turn, as any session event does; the wake has its own)
            if len(applied) == 3 and state["turns"] == 2 and not state["busy"] and not state["pending"]:
                break
            time.sleep(0.02)
        else:
            raise AssertionError(f"the queued work did not run: {state}")
        assert sorted(a["item_kind"] for a in applied) == ["agent_message", "message", "wake"]
        assert client.portal.call(client.app.state.chat_store.pull_inbox, session_id, process_owner()) == []


class TestIdempotency:
    def test_a_message_repeated_with_its_key_is_the_same_item(self, client):
        session_id = served_elsewhere(client)
        first = post(client, session_id, "messages", {"text": "hi"}, **{"Idempotency-Key": "send-1"})
        again = post(client, session_id, "messages", {"text": "hi"}, **{"Idempotency-Key": "send-1"})
        other = post(client, session_id, "messages", {"text": "hi"}, **{"Idempotency-Key": "send-2"})
        assert first.json()["inbox_id"] == again.json()["inbox_id"] != other.json()["inbox_id"]
        assert len(waiting(client, session_id)) == 2

    def test_a_wake_repeated_with_its_key_is_the_same_item(self, client):
        session_id = served_elsewhere(client)
        first = post(client, session_id, "wake", {"text": "w"}, **{"Idempotency-Key": "fire-9"})
        again = post(client, session_id, "wake", {"text": "w"}, **{"Idempotency-Key": "fire-9"})
        assert first.json()["inbox_id"] == again.json()["inbox_id"]
        assert len(waiting(client, session_id)) == 1

    def test_an_agent_message_is_keyed_by_its_sender_and_id(self, client):
        session_id = served_elsewhere(client)
        body = {"from": "bob", "text": "done", "message_id": "am_1"}
        first = post(client, session_id, "agent_messages", body)
        again = post(client, session_id, "agent_messages", body)
        from_carol = post(client, session_id, "agent_messages", {**body, "from": "carol"})
        unkeyed = [post(client, session_id, "agent_messages", {"from": "bob", "text": "x"}) for _ in range(2)]
        assert first.json()["inbox_id"] == again.json()["inbox_id"] != from_carol.json()["inbox_id"]
        assert unkeyed[0].json()["inbox_id"] != unkeyed[1].json()["inbox_id"]
        assert len(waiting(client, session_id)) == 4

    @pytest.mark.parametrize("key", ["", "k" * 256])
    def test_a_key_the_store_cannot_take_is_a_400(self, client, key):
        session_id = served_elsewhere(client)
        response = post(client, session_id, "messages", {"text": "hi"}, **{"Idempotency-Key": key})
        assert response.status_code == 400, response.text
        assert waiting(client, session_id) == []


class TestRefusedBeforeQueuing:
    """What the serving process would refuse is refused here: it could tell nobody."""

    @pytest.mark.parametrize(
        ("route", "body", "status"),
        [
            ("messages", {"text": 5}, 400),
            ("messages", {"text": "hi", "mode": "sideways"}, 400),
            ("messages", {"text": "NaN", "mode": float("nan")}, 400),
            ("wake", {"text": "   "}, 400),
            ("wake", {"text": "w", "trigger": {"kind": "user_message"}}, 400),
            ("wake", {"text": "w", "trigger": {"wake_id": "mine"}}, 400),
            ("wake", {"text": "w", "tool_policy": "everything"}, 400),
            ("agent_messages", {"from": "bob", "text": "x", "priority": "asap"}, 422),
            ("agent_messages", {"from": "no spaces", "text": "x"}, 422),
            ("agent_messages", {"from": "bob", "text": "   "}, 422),
            ("agent_messages", {"from": "bob", "text": "x", "message_id": "not an id"}, 422),
            ("agent_messages", {"from": "bob", "text": "x" * 4001}, 422),
            ("agent_messages", {"from": "bob", "text": "x", "hops": "2"}, 400),
        ],
    )
    def test_a_bad_request_queues_nothing(self, client, route, body, status):
        session_id = served_elsewhere(client)
        if isinstance(body.get("mode"), float):
            response = client.post(
                f"/chat/sessions/{session_id}/{route}",
                content='{"text": "hi", "mode": NaN}',
                headers={"Content-Type": "application/json"},
            )
        else:
            response = post(client, session_id, route, body)
        assert response.status_code == status, response.text
        if status == 422:
            assert response.json()["reason"] == "invalid"
        assert waiting(client, session_id) == []

    def test_an_unknown_session_is_a_404(self, client):
        response = post(client, "no-such-session", "messages", {"text": "hi"})
        assert response.status_code == 404, response.text


class TestTheCap:
    def test_a_full_inbox_is_a_429(self, client):
        session_id = served_elsewhere(client)
        client.app.state.chat_store.max_inbox_items = 2
        assert post(client, session_id, "messages", {"text": "1"}).status_code == 202
        assert post(client, session_id, "wake", {"text": "2"}).status_code == 202
        response = post(client, session_id, "messages", {"text": "3"})
        assert response.status_code == 429, response.text
        assert response.headers.get("Retry-After") and "2 inbox items" in response.json()["detail"]
        assert len(waiting(client, session_id)) == 2


class TestAStoreThatDoesNotAnswer:
    @pytest.mark.parametrize("keyed", [False, True])
    def test_a_post_that_may_have_landed_says_whether_a_retry_is_safe(self, client, monkeypatch, keyed):
        session_id = served_elsewhere(client)
        monkeypatch.setenv("TIMBAL_CHAT_STORE_TIMEOUT_S", "0.1")
        store = client.app.state.chat_store

        async def hangs(*_args, **_kwargs):
            await asyncio.Event().wait()

        monkeypatch.setattr(store, "post_inbox", hangs)
        headers = {"Idempotency-Key": "send-1"} if keyed else {}
        response = post(client, session_id, "messages", {"text": "hi"}, **headers)
        assert response.status_code == 503, response.text
        assert ("Retry-After" in response.headers) is keyed

    def test_a_store_that_fails_says_so(self, client, monkeypatch):
        session_id = served_elsewhere(client)
        store = client.app.state.chat_store

        async def fails(*_args, **_kwargs):
            raise OSError("disk I/O error")

        monkeypatch.setattr(store, "post_inbox", fails)
        response = post(client, session_id, "messages", {"text": "hi"})
        assert response.status_code == 503 and "disk I/O error" in response.json()["detail"]


class TestAStoreWithoutAnInbox:
    @pytest.mark.parametrize(
        ("route", "body"),
        [
            ("messages", {"text": "hi"}),
            ("wake", {"text": "w"}),
            ("agent_messages", {"from": "bob", "text": "x"}),
        ],
    )
    def test_it_still_answers_409(self, client, monkeypatch, route, body):
        session_id = served_elsewhere(client)
        monkeypatch.setattr(client.app.state.chat_store, "supports_inbox", False)
        response = post(client, session_id, route, body)
        assert response.status_code == 409, response.text
        assert "served by another process" in response.json()["detail"]
        assert waiting(client, session_id) == []
