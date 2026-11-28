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
            ("messages", {"text": "   "}, 400),
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
    def test_a_post_that_may_have_landed_is_a_503_without_retry_after(self, client, monkeypatch, keyed):
        """``Retry-After`` on a ``503`` from these routes means nothing was accepted (``RemotePeer`` reads it as
        "not sent", and a model then sends again under a new id). A post that may have landed never says so,
        keyed or not; the detail says whether a retry with the key is safe."""
        session_id = served_elsewhere(client)
        monkeypatch.setenv("TIMBAL_CHAT_STORE_TIMEOUT_S", "0.1")
        store = client.app.state.chat_store

        async def hangs(*_args, **_kwargs):
            await asyncio.Event().wait()

        monkeypatch.setattr(store, "post_inbox", hangs)
        headers = {"Idempotency-Key": "send-1"} if keyed else {}
        response = post(client, session_id, "messages", {"text": "hi"}, **headers)
        assert response.status_code == 503, response.text
        assert "Retry-After" not in response.headers
        assert ("same Idempotency-Key" in response.json()["detail"]) is keyed

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


def turns_and_applied(client, session_id):
    state = client.get(f"/chat/sessions/{session_id}").json()
    log = client.get(f"/chat/sessions/{session_id}/log", params={"after": 0, "limit": 2000}).json()
    records = [e["data"] for e in log["events"] if isinstance(e["data"], dict) and e["data"].get("type") == "SESSION"]
    return state, records


def settled(client, session_id, done, timeout=30.0):
    deadline = time.monotonic() + timeout  # a hang guard
    while time.monotonic() < deadline:
        state, records = turns_and_applied(client, session_id)
        if done(state, records) and not state["busy"] and not state["pending"]:
            return state, records
        time.sleep(0.02)
    raise AssertionError(f"never settled: {state}")


class TestAKeyedRequestToTheServingWorker:
    """With a key, the serving worker queues the request in the inbox too: a retry is the same item whichever
    worker it reaches (it was a second message when the retry reached the serving worker directly)."""

    def test_a_retry_that_reaches_the_serving_worker_is_the_same_message(self, client):
        session_id = served_elsewhere(client)
        first = post(client, session_id, "messages", {"text": "pay me"}, **{"Idempotency-Key": "send-1"})
        assert first.json()["queued"] == "inbox"
        client.portal.call(client.app.state.chat_store.release_lease, session_id, ELSEWHERE)
        assert client.get(f"/chat/sessions/{session_id}").status_code == 200  # served here from now on
        again = post(client, session_id, "messages", {"text": "pay me"}, **{"Idempotency-Key": "send-1"})
        assert again.status_code == 202 and again.json()["inbox_id"] == first.json()["inbox_id"], again.text
        state, records = settled(client, session_id, lambda st, _rs: st["turns"] >= 1)
        assert [r["text"] for r in records if r["kind"] == "message"] == ["pay me"]
        assert state["turns"] == 1

    def test_a_keyed_message_to_the_serving_worker_is_applied(self, client):
        session_id = client.post("/chat/sessions", json={"busy_mode": "followup"}).json()["session_id"]
        response = post(client, session_id, "messages", {"text": "hi"}, **{"Idempotency-Key": "k"})
        assert response.status_code == 202 and response.json()["queued"] == "inbox", response.text
        again = post(client, session_id, "messages", {"text": "hi"}, **{"Idempotency-Key": "k"})
        assert again.json()["inbox_id"] == response.json()["inbox_id"]
        state, records = settled(client, session_id, lambda st, _rs: st["turns"] >= 1)
        assert [r["text"] for r in records if r["kind"] == "message"] == ["hi"] and state["turns"] == 1

    def test_without_a_key_the_serving_worker_answers_as_before(self, client):
        session_id = client.post("/chat/sessions", json={}).json()["session_id"]
        response = post(client, session_id, "messages", {"text": "hi"})
        assert response.status_code == 202 and "message" in response.json() and "queued" not in response.json()

    def test_a_store_without_an_inbox_ignores_the_header(self, client, monkeypatch):
        """As before the inbox: not looked at (one a store could not take is no 400)."""
        monkeypatch.setattr(client.app.state.chat_store, "supports_inbox", False)
        session_id = client.post("/chat/sessions", json={}).json()["session_id"]
        response = post(client, session_id, "messages", {"text": "hi"}, **{"Idempotency-Key": "k" * 300})
        assert response.status_code == 202 and "message" in response.json(), response.text

    def test_a_store_without_an_inbox_refuses_a_held_session_before_reading_the_body(self, client, monkeypatch):
        session_id = served_elsewhere(client)
        monkeypatch.setattr(client.app.state.chat_store, "supports_inbox", False)
        response = client.post(
            f"/chat/sessions/{session_id}/messages", content="not json", headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 409, response.text


class TestOrder:
    def test_a_queued_message_comes_before_a_later_direct_one(self, client):
        """Queued by another process (its sender had its 202), then a direct one: the serving worker's drain
        sees the other's post only at its watch's next look, and the direct message went first."""
        store = client.app.state.chat_store
        store.inbox_poll_s = 60.0  # (before the session starts its drain: the watch looks this rarely)
        session_id = client.post("/chat/sessions", json={"busy_mode": "followup"}).json()["session_id"]

        def other_process():
            import sqlite3

            conn = sqlite3.connect(str(store.path), timeout=10)
            conn.execute(
                "INSERT INTO chat_inbox (id, session_id, kind, body, posted_at) VALUES (?, ?, ?, ?, ?)",
                ("in_first", session_id, "message", '{"text": "first", "mode": null}', time.time()),
            )
            conn.commit()
            conn.close()

        other_process()
        assert post(client, session_id, "messages", {"text": "second"}).status_code == 202
        _, records = settled(client, session_id, lambda st, _rs: st["turns"] >= 2, timeout=10)
        assert [r["text"] for r in records if r["kind"] == "message"] == ["first", "second"]


def test_startup_restores_a_session_whose_only_work_is_its_inbox(tmp_path, monkeypatch):
    """Skipped at startup (its old process's lease still looked live), retried until it is free: a session with
    nothing pending but items in its inbox has work too (it was dropped at the first retry, its items left)."""
    import timbal.server.chat as chat_module
    from timbal.state.chat_store import SessionSnapshot, SqliteChatSessionStore

    (tmp_path / "agent.py").write_text(AGENT)
    monkeypatch.setenv("E2E_DIR", str(tmp_path))
    monkeypatch.setenv("TIMBAL_RUNNABLE", f"{tmp_path / 'agent.py'}::agent")
    monkeypatch.delenv("TIMBAL_CHAT_STORE", raising=False)
    monkeypatch.setattr(chat_module, "_RESTORE_RETRY_S", 0.1)
    store = SqliteChatSessionStore(tmp_path / "timbal.db")
    store.lease_ttl_s = 60.0

    async def seed():
        await store.save(SessionSnapshot(session_id="s1", agent="agent"), expect_version=None, owner=ELSEWHERE)
        await store.post_inbox("s1", "message", {"text": "queued while it moved", "mode": None})

    asyncio.run(seed())
    with TestClient(create_app()) as client:
        sessions = client.app.state.chat_sessions
        assert "s1" not in sessions  # skipped: the lease was still live
        client.portal.call(client.app.state.chat_store.release_lease, "s1", ELSEWHERE)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not (sessions.get("s1") and sessions["s1"].turns == 1):
            time.sleep(0.05)
        assert sessions.get("s1") is not None and sessions["s1"].turns == 1


class TestReviewOn341:
    """Findings of the review on timbal#341, each reproduced before its fix."""

    def test_a_session_of_another_agent_is_not_queued_for(self, client):
        """Two agents on one store: the lease is met before the snapshot's agent is read, and the request was
        queued for a session that is not this agent's (its own worker would then run it)."""
        from timbal.state.chat_store import SessionSnapshot

        store = client.app.state.chat_store
        snapshot = SessionSnapshot(session_id="not-mine", agent="another_agent")
        client.portal.call(lambda: store.save(snapshot, expect_version=None, owner=ELSEWHERE))
        for route, body in (
            ("messages", {"text": "hi"}),
            ("wake", {"text": "check"}),
            ("agent_messages", {"from": "bob", "text": "x", "message_id": "am_1"}),
        ):
            response = post(client, "not-mine", route, body)
            assert response.status_code == 404, (route, response.text)
        assert waiting(client, "not-mine") == []

    def test_work_queued_for_a_session_its_owner_then_lets_go_is_applied(self, client, monkeypatch):
        """The owner lets the session go (idle reaping) before it pulls what was queued: the worker that took the
        request restores the session once it is free, and applies it (it waited for the next startup)."""
        import timbal.server.chat as chat_module

        monkeypatch.setattr(chat_module, "_RESTORE_RETRY_S", 0.05)
        session_id = served_elsewhere(client)
        assert post(client, session_id, "messages", {"text": "while it moved"}).json()["queued"] == "inbox"
        client.portal.call(client.app.state.chat_store.release_lease, session_id, ELSEWHERE)  # let go, undrained
        deadline = time.monotonic() + 10
        sessions = client.app.state.chat_sessions
        while time.monotonic() < deadline and not (sessions.get(session_id) and sessions[session_id].turns == 1):
            time.sleep(0.02)
        assert sessions.get(session_id) is not None and sessions[session_id].turns == 1

    def _restored_anew(self, client, session_id):
        """This process lets the session go and serves it again from the store: what it held only in memory
        (the last messages from agents, by sender and id) is gone."""
        session = client.app.state.chat_sessions.pop(session_id)
        client.portal.call(lambda: session.close(keep=True))
        assert client.get(f"/chat/sessions/{session_id}").status_code == 200

    def test_an_agent_message_queued_then_retried_at_its_new_owner_is_a_duplicate(self, client):
        session_id = served_elsewhere(client)
        body = {"from": "bob", "text": "the build is green", "message_id": "am_9"}
        assert post(client, session_id, "agent_messages", body).json()["queued"] == "inbox"
        client.portal.call(client.app.state.chat_store.release_lease, session_id, ELSEWHERE)
        assert client.get(f"/chat/sessions/{session_id}").status_code == 200  # restored here, the item applied
        settled(client, session_id, lambda st, _rs: st["turns"] == 1)
        self._restored_anew(client, session_id)
        retried = post(client, session_id, "agent_messages", body)
        assert retried.status_code == 202 and retried.json()["duplicate"] is True, retried.text
        state, records = settled(client, session_id, lambda _st, _rs: True)
        assert state["turns"] == 1 and len([r for r in records if r["kind"] == "agent_message"]) == 0

    def test_an_agent_message_taken_directly_then_retried_through_another_worker_is_not_queued(self, client):
        session_id = client.post("/chat/sessions", json={}).json()["session_id"]
        body = {"from": "bob", "text": "the build is green", "message_id": "am_9"}
        first = post(client, session_id, "agent_messages", body)
        assert first.status_code == 202 and first.json()["duplicate"] is False, first.text
        settled(client, session_id, lambda st, _rs: st["turns"] == 1)
        session = client.app.state.chat_sessions.pop(session_id)
        client.portal.call(lambda: session.close(keep=True))
        client.portal.call(client.app.state.chat_store.acquire_lease, session_id, ELSEWHERE)  # another worker has it
        retried = post(client, session_id, "agent_messages", body)
        assert retried.status_code == 202, retried.text
        assert waiting(client, session_id) == []  # the same message: nothing more queued for it

    def test_a_direct_request_waits_no_longer_than_the_store_bound_for_what_is_queued(self, client, monkeypatch):
        """The read of what is queued comes before the request is taken: a store that does not answer is a 503
        that took nothing (with Retry-After), within the routes' bound, not a request that waits with it."""
        # A bound above what a slow disk adds to every call (TIMBAL_TEST_SLOW_IO), far below how long the read hangs.
        monkeypatch.setenv("TIMBAL_CHAT_STORE_TIMEOUT_S", "1")
        session_id = client.post("/chat/sessions", json={}).json()["session_id"]
        store = client.app.state.chat_store

        async def hangs(*_args, **_kwargs):
            await asyncio.sleep(20)
            return []

        monkeypatch.setattr(store, "pull_inbox", hangs)
        started = time.monotonic()
        response = post(client, session_id, "messages", {"text": "hi"})
        assert time.monotonic() - started < 15  # a hang guard: it answered, it did not wait out the read
        assert response.status_code == 503 and response.headers.get("Retry-After"), response.text
        assert client.app.state.chat_sessions[session_id].pending_messages() == []

    def test_a_direct_request_is_not_taken_ahead_of_queued_work_it_could_not_read(self, client, monkeypatch):
        session_id = client.post("/chat/sessions", json={}).json()["session_id"]
        store = client.app.state.chat_store

        async def fails(*_args, **_kwargs):
            raise OSError("disk I/O error")

        monkeypatch.setattr(store, "pull_inbox", fails)
        response = post(client, session_id, "wake", {"text": "check in"})
        assert response.status_code == 503 and response.headers.get("Retry-After"), response.text
        assert client.app.state.chat_sessions[session_id]._wakes == []

    def test_senders_and_ids_with_colons_are_different_messages(self, client):
        session_id = served_elsewhere(client)
        first = post(client, session_id, "agent_messages", {"from": "support:eu", "text": "a", "message_id": "7"})
        second = post(client, session_id, "agent_messages", {"from": "support", "text": "b", "message_id": "eu:7"})
        assert first.json()["inbox_id"] != second.json()["inbox_id"]
        assert sorted(i["body"]["text"] for i in waiting(client, session_id)) == ["a", "b"]
