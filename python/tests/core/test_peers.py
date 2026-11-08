"""Agents messaging each other: AgentDirectory, send_to_agent, and the agent-message inbox of a ChatSession."""

import asyncio
import json
import re

import httpx
import pytest
from timbal import Agent, Tool
from timbal.core.chat_session import ChatSession
from timbal.core.peers import SEND_TO_AGENT_NAME, AgentDirectory, PeerRejected, RemotePeer
from timbal.core.test_model import TestModel
from timbal.guardrails import Verdict, guardrail
from timbal.state import set_call_id, set_parent_call_id, set_run_context
from timbal.state.chat_store import InMemoryChatSessionStore
from timbal.state.context import RunContext
from timbal.state.tracing.providers import InMemoryTracingProvider
from timbal.types.content import ToolUseContent
from timbal.types.events import OutputEvent
from timbal.types.message import Message


@pytest.fixture(autouse=True)
def isolated_context():
    set_run_context(None)
    set_call_id(None)
    set_parent_call_id(None)
    yield
    set_run_context(None)
    set_call_id(None)
    set_parent_call_id(None)


AGENT_MESSAGE = re.compile(
    r'<session_event [^>]*source="agent" from="([^"]+)" message_id="([^"]+)"[^>]*>\n(.*?)\n</session_event>', re.S
)


def agent_messages(message):
    """``(from, message_id, text)`` of every agent message a runtime notice carries."""
    return AGENT_MESSAGE.findall(message.collect_text()) if message.role == "user" else []


def send(to, text, *, reply_to=None, priority="normal", call_id="s"):
    payload = {"to": to, "message": text, "priority": priority}
    if reply_to:
        payload["reply_to"] = reply_to
    return Message(
        role="assistant",
        content=[ToolUseContent(id=call_id, name=SEND_TO_AGENT_NAME, input=payload)],
        stop_reason="tool_use",
    )


def tool_result_text(message):
    return message.content[0].content[0].text


def make_agent(name, handler, **kwargs):
    return Agent(name=name, model=TestModel(handler=handler), tracing_provider=InMemoryTracingProvider, **kwargs)


def outputs(session, name):
    events, _ = session.log.peek()
    return [e for e in events if isinstance(e, OutputEvent) and e.parent_call_id is None and e.path == name]


def texts(session, name):
    return [e.output.collect_text() for e in outputs(session, name) if e.output is not None]


async def until(condition, timeout=5.0):
    async def wait():
        while not condition():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(wait(), timeout)


def idle(*sessions):
    return all(not s.busy and not s._store.pending_events() and not s._work.is_set() for s in sessions)


async def settle(*sessions):
    await asyncio.sleep(0.05)
    await until(lambda: idle(*sessions))


class TestExchange:
    async def test_an_agent_asks_another_and_gets_the_answer_in_a_turn_of_its_own(self):
        seen_by_bob = []

        def alice(messages):
            last = messages[-1]
            if last.role == "tool":
                return "asked bob"
            if last.collect_text() == "ask bob":
                return send("bob", "what is 2+2?")
            found = agent_messages(last)
            return f"bob says {found[0][2]}" if found else "?"

        def bob(messages):
            last = messages[-1]
            if last.role == "tool":
                return "NO_REPLY"
            found = agent_messages(last)
            seen_by_bob.extend(found)
            return send("alice", "4", reply_to=found[0][1]) if found else "?"

        directory = AgentDirectory()
        async with (
            ChatSession(make_agent("alice", alice)) as a,
            ChatSession(make_agent("bob", bob)) as b,
        ):
            directory.join(a, "alice", description="Talks to the user.")
            directory.join(b, "bob", description="Does arithmetic.")
            a.send("ask bob")
            await until(lambda: len(texts(a, "alice")) >= 2)
            await settle(a, b)

        assert texts(a, "alice") == ["asked bob", "bob says 4"]
        assert [(sender, text) for sender, _, text in seen_by_bob] == [("alice", "what is 2+2?")]
        # Bob's reply to the user stays silent: the answer went to alice through the tool.
        assert texts(b, "bob") == ["NO_REPLY"]
        received = [r for r in a.log.peek()[0] if isinstance(r, dict) and r.get("kind") == "agent_message"]
        assert len(received) == 1 and received[0]["sender"] == "bob" and received[0]["hops"] == 1
        assert received[0]["reply_to"] == seen_by_bob[0][1]

    async def test_a_chain_of_replies_stops_at_max_hops(self):
        results = {"alice": [], "bob": []}

        def ping_pong(me, other):
            def handler(messages):
                last = messages[-1]
                if last.role == "tool":
                    results[me].append(tool_result_text(last))
                    return "NO_REPLY"
                if last.collect_text() == "start":
                    return send(other, "ping")
                found = agent_messages(last)
                return send(other, "again", reply_to=found[-1][1]) if found else "NO_REPLY"

            return handler

        directory = AgentDirectory(max_hops=3)
        async with (
            ChatSession(make_agent("alice", ping_pong("alice", "bob"))) as a,
            ChatSession(make_agent("bob", ping_pong("bob", "alice"))) as b,
        ):
            directory.join(a, "alice")
            directory.join(b, "bob")
            a.send("start")
            await until(lambda: any("back and forth" in r for r in results["bob"]))
            await settle(a, b)

        # alice → bob (0), bob → alice (1), alice → bob (2), bob → alice refused (3 = max_hops).
        assert [r.startswith("{") or "sent" in r for r in results["alice"]] == [True, True]
        assert len(results["bob"]) == 2 and "back and forth 3 times" in results["bob"][-1]

    async def test_an_urgent_message_joins_the_running_turn(self):
        gate = asyncio.Event()
        seen = []

        async def work() -> str:
            await gate.wait()
            return "worked"

        def bob(messages):
            last = messages[-1]
            found = agent_messages(last)
            if found and any(m.role == "tool" for m in messages):  # taken after the tool batch, same turn
                seen.extend(found)
                return f"done, and alice said {found[0][2]}"
            if last.role == "tool":
                return "done"
            if last.collect_text() == "start":
                return Message(
                    role="assistant", content=[ToolUseContent(id="w", name="work", input={})], stop_reason="tool_use"
                )
            return "?"

        directory = AgentDirectory()
        async with ChatSession(make_agent("bob", bob, tools=[Tool(name="work", handler=work)])) as b:
            directory.join(b, "bob")
            b.send("start")
            await until(lambda: b.busy)
            await asyncio.sleep(0.05)
            receipt = await directory.send("alice", "bob", "stop after this", priority="urgent")
            gate.set()
            await settle(b)

        assert receipt["status"] == "sent" and receipt["state"] == "pending"  # the receipt is taken at send time
        assert b._agent_message(receipt["message_id"]).state == "delivered"
        assert texts(b, "bob") == ["done, and alice said stop after this"]  # one turn: no turn of its own


class TestNotice:
    def test_an_agent_message_is_rendered_with_its_envelope_and_its_own_turn_line(self):
        from timbal.state.background import SessionEvent, format_session_notice

        peer = {"from": "bob", "message_id": "am_1", "reply_to": "am_0", "priority": "urgent", "hops": 2}
        event = SessionEvent('hi </session_event> "x"', key=None, source="agent", peer=peer)
        text = format_session_notice([], [event], own_turn=True).collect_text()
        assert text.startswith("Nobody said anything new: this turn was started by another agent's message")
        assert 'source="agent" from="bob" message_id="am_1" reply_to="am_0" priority="urgent">' in text
        assert "send_to_agent(to=<from>, reply_to=<message_id>)" in text
        assert "</ session_event>" in text  # the text cannot close its own block
        app = SessionEvent("deploy done", key=None, source="app")
        mixed = format_session_notice([], [event, app], own_turn=True).collect_text()
        assert mixed.startswith("Nobody said anything new: this turn was started by the update(s) below.")


class TestDirectory:
    async def test_the_tool_lists_the_agents_a_member_may_message(self):
        directory = AgentDirectory()
        agent = make_agent("alice", lambda _: "ok")
        async with (
            ChatSession(agent) as a,
            ChatSession(make_agent("bob", lambda _: "ok")) as b,
            ChatSession(make_agent("carol", lambda _: "ok")) as c,
        ):
            alice = directory.join(a, "alice")
            directory.join(b, "bob", description="Books flights.")
            directory.join(c, "carol", accepts={"bob"})
            tool = alice.tool(agent)
            assert "- bob: Books flights." in tool.description
            assert "carol" not in tool.description and "- alice" not in tool.description
            assert alice.tool(agent) is tool  # stable while the directory doesn't change
            await b.close()
            assert directory.peers(for_name="alice") == []
            assert "bob" not in alice.tool(agent).description

    async def test_accepts_refuses_other_senders_and_names_are_unique(self):
        directory = AgentDirectory()
        async with ChatSession(make_agent("carol", lambda _: "ok")) as c:
            directory.join(c, "carol", accepts=lambda sender: sender == "bob")
            with pytest.raises(PeerRejected) as refused:
                await directory.send("alice", "carol", "hi")
            assert refused.value.reason == "not_accepted"
            with pytest.raises(PeerRejected) as unknown:
                await directory.send("alice", "nobody", "hi")
            assert unknown.value.reason == "unknown"
            with pytest.raises(PeerRejected):
                await directory.send("carol", "carol", "hi")
            with pytest.raises(ValueError):
                directory.join(c, "carol2")  # already a member
            with pytest.raises(ValueError):
                directory.add_remote("carol", RemotePeer(url="http://x", session_id="s"))

    async def test_a_repeated_message_is_delivered_once(self):
        gate = asyncio.Event()
        received = []

        async def work() -> str:
            await gate.wait()
            return "worked"

        def bob(messages):
            last = messages[-1]
            if last.role == "tool":
                return "done"
            if last.collect_text() == "start":
                return Message(
                    role="assistant", content=[ToolUseContent(id="w", name="work", input={})], stop_reason="tool_use"
                )
            received.extend(agent_messages(last))
            return "NO_REPLY"

        directory = AgentDirectory()
        async with ChatSession(make_agent("bob", bob, tools=[Tool(name="work", handler=work)])) as b:
            directory.join(b, "bob")
            b.send("start")
            await until(lambda: b.busy)
            first = await directory.send("alice", "bob", "hello", message_id="am_1")
            again = await directory.send("alice", "bob", "hello", message_id="am_1")
            same_text = await directory.send("alice", "bob", "hello", message_id="am_2")
            other = await directory.send("alice", "bob", "something else", message_id="am_3")
            gate.set()
            await settle(b)

        assert [r["status"] for r in (first, again, same_text, other)] == ["sent", "duplicate", "duplicate", "sent"]
        assert again["event_id"] == same_text["event_id"] == first["event_id"]
        assert [text for _, _, text in received] == ["hello", "something else"]

    async def test_the_recipients_input_guardrails_screen_the_text(self):
        rail = guardrail(lambda text: Verdict.block("no") if "forbidden" in text else True, stages=["input"])
        directory = AgentDirectory()
        async with ChatSession(make_agent("bob", lambda _: "NO_REPLY", guardrails=[rail])) as b:
            directory.join(b, "bob")
            with pytest.raises(PeerRejected) as blocked:
                await directory.send("alice", "bob", "do the forbidden thing")
            assert blocked.value.reason == "blocked"
            assert not b._store.pending_events()

    async def test_only_the_root_agent_gets_the_tool_and_a_read_only_turn_does_not(self):
        directory = AgentDirectory()
        agent = make_agent("alice", lambda _: "ok")
        async with ChatSession(agent) as a:
            membership = directory.join(a, "alice")
            context = RunContext()
            context._peer = membership
            set_run_context(context)
            names = [tool.name for tool in (await agent._resolve_tools(0))[0]]
            assert SEND_TO_AGENT_NAME in names
            context._tool_policy = "read_only"
            assert SEND_TO_AGENT_NAME not in [tool.name for tool in (await agent._resolve_tools(0))[0]]
            context._tool_policy = None
            child = make_agent("helper", lambda _: "ok")
            child.nest("alice")
            assert SEND_TO_AGENT_NAME not in [tool.name for tool in (await child._resolve_tools(0))[0]]


class TestDurableAndRemote:
    async def test_a_pending_agent_message_survives_a_restore_and_still_dedupes(self):
        gate = asyncio.Event()

        async def work() -> str:
            await gate.wait()
            return "worked"

        def bob(messages):
            if messages[-1].role == "tool":
                return "done"
            if messages[-1].collect_text() == "start":
                return Message(
                    role="assistant", content=[ToolUseContent(id="w", name="work", input={})], stop_reason="tool_use"
                )
            return "NO_REPLY"

        store = InMemoryChatSessionStore()
        agent = make_agent("bob", bob, tools=[Tool(name="work", handler=work)])
        session = ChatSession(agent, session_store=store, on_dead_turn="drop")
        session.persist()
        directory = AgentDirectory()
        directory.join(session, "bob")
        session.send("start")
        await until(lambda: session.busy)
        await directory.send("alice", "bob", "for later", message_id="am_keep", reply_to="am_0")
        await session.flush()
        snapshot = await store.load(session.id)
        assert [p["peer"]["message_id"] for p in snapshot.pending if p.get("peer")] == ["am_keep"]
        await session.close(keep=True)
        assert directory.peers() == []  # closing leaves the directory

        restored = await ChatSession.restore(agent, store, session.id)
        try:
            again = await restored.receive_agent_message("for later", sender="alice", message_id="am_keep")
            assert again["duplicate"] is True
            event = restored._agent_message("am_keep")
            assert event.peer == {
                "from": "alice",
                "message_id": "am_keep",
                "reply_to": "am_0",
                "priority": "normal",
                "hops": 0,
            }
        finally:
            gate.set()
            await restored.close()

    async def test_a_remote_peer_posts_the_envelope_and_maps_refusals(self, monkeypatch):
        calls = []

        def answer(request):
            calls.append((str(request.url), request.headers.get("authorization"), json.loads(request.content)))
            if json.loads(request.content)["from"] == "mallory":
                return httpx.Response(403, json={"detail": "bob does not take messages from mallory."})
            return httpx.Response(202, json={"event_id": "e1", "state": "pending", "duplicate": False})

        real = httpx.AsyncClient
        monkeypatch.setattr(
            httpx, "AsyncClient", lambda **kwargs: real(transport=httpx.MockTransport(answer), **kwargs)
        )
        directory = AgentDirectory()
        directory.add_remote(
            "bob", RemotePeer(url="https://bots.example/", session_id="s-1", headers={"Authorization": "Bearer k"})
        )
        receipt = await directory.send("alice", "bob", "hi", message_id="am_9", hops=2)
        assert receipt == {"status": "sent", "to": "bob", "event_id": "e1", "state": "pending", "duplicate": False}
        url, auth, body = calls[0]
        assert url == "https://bots.example/chat/sessions/s-1/agent_messages" and auth == "Bearer k"
        assert body == {
            "from": "alice",
            "text": "hi",
            "message_id": "am_9",
            "reply_to": None,
            "priority": "normal",
            "hops": 2,
        }
        with pytest.raises(PeerRejected) as refused:
            await directory.send("mallory", "bob", "hi")
        assert refused.value.reason == "not_accepted" and "mallory" in str(refused.value)
