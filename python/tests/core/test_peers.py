"""Agents messaging each other: AgentDirectory, send_to_agent, and the agent-message inbox of a ChatSession."""

import asyncio
import json
import re

import httpx
import pytest
from pydantic import PrivateAttr
from timbal import Agent, Tool
from timbal.core.chat_session import ChatSession
from timbal.core.peers import SEND_TO_AGENT_NAME, AgentDirectory, PeerRejected, RemotePeer
from timbal.core.test_model import TestModel
from timbal.core.tool_set import ToolSet
from timbal.guardrails import Verdict, guardrail
from timbal.state import set_call_id, set_parent_call_id, set_run_context
from timbal.state.chat_store import InMemoryChatSessionStore
from timbal.state.context import RunContext
from timbal.state.tracing.providers import InMemoryTracingProvider
from timbal.types.content import ToolResultContent, ToolUseContent
from timbal.types.events import ApprovalEvent, OutputEvent
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
        assert b._agent_message("alice", receipt["message_id"]).state == "delivered"
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
        helper = make_agent("helper", lambda _: "ok")
        alice = make_agent(
            "alice", lambda messages: "done" if messages[-1].role == "tool" else call("helper", prompt="hi"), tools=[helper]
        )
        async with ChatSession(alice) as a:
            membership = directory.join(a, "alice")
            context = fresh(peer=membership)
            await run(alice)
            assert SEND_TO_AGENT_NAME in offered(context, "alice.llm")[0]
            assert SEND_TO_AGENT_NAME not in offered(context, "alice.helper.llm")[0]
            context = fresh(peer=membership)
            context._tool_policy = "read_only"
            await run(alice)
            assert SEND_TO_AGENT_NAME not in offered(context, "alice.llm")[0]


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

        restored = await ChatSession.from_store(agent, store, session.id)
        try:
            again = await restored.receive_agent_message("for later", sender="alice", message_id="am_keep")
            assert again["duplicate"] is True
            event = restored._agent_message("alice", "am_keep")
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


# -- Against today's main ---------------------------------------------------------------------------------------


class Catalog(ToolSet):
    """An application's tools that may change between runs (an MCP server's, say)."""

    _offered: list = PrivateAttr(default_factory=list)

    @property
    def offered(self) -> list:
        return self._offered

    async def resolve(self):
        return list(self._offered)


def call(tool, call_id=None, **input):
    return Message(
        role="assistant",
        content=[ToolUseContent(id=call_id or f"c-{tool}", name=tool, input=input)],
        stop_reason="tool_use",
    )


def fresh(parent_id=None, peer=None):
    context = RunContext(parent_id=parent_id, tracing_provider=InMemoryTracingProvider)
    context._peer = peer
    set_run_context(context)
    set_call_id(None)
    set_parent_call_id(None)
    return context


async def run(agent, **kwargs):
    events = [event async for event in agent(prompt="go", **kwargs)]
    out = next(e for e in reversed(events) if isinstance(e, OutputEvent) and e.path == agent.name)
    return events, out


def results(context):
    root = next(s for s in context._trace.values() if s.parent_call_id is None)
    return [
        "".join(getattr(item, "text", "") for item in c.content) if isinstance(c.content, list) else str(c.content)
        for m in root.memory
        for c in m.content
        if isinstance(c, ToolResultContent)
    ]


def offered(context, path=None):
    """The tool names each model call of the run was offered (of the agent at ``path`` only), in order."""
    spans = sorted(
        (s for s in context._trace.values() if s.path.endswith(".llm") and path in (None, s.path)), key=lambda s: s.t0
    )
    return [[t["name"] if isinstance(t, dict) else t.name for t in (s.input.get("tools") or [])] for s in spans]


def ask_for(name):
    def ask(_text, ctx):
        return Verdict.escalate("a person checks it") if ctx.tool_name == name else True

    return guardrail(ask, stages=["tool_args"], name="ask")


def received(session):
    """``(from, text)`` of the agent messages a session holds (pending or taken), oldest first."""
    return [(event.peer["from"], event.text) for event in session._agent_messages.values()]


class TestTheSendToolAmongTheAgentsTools:
    """``send_to_agent`` is a tool the agent adds itself: it is named, bound and screened like ``send_message``
    and ``mark_salient`` (L8, L9)."""

    async def test_a_tool_set_with_the_name_keeps_it_and_the_send_takes_another(self):
        app_ran = []
        catalog = Catalog()
        catalog.offered[:] = [
            Tool(name="send_to_agent", handler=lambda to, message: app_ran.append(message) or "app")
        ]
        model = TestModel(responses=[call("message_agent", to="bob", message="hi bob"), "done"])
        alice = Agent(name="alice", model=model, tools=[catalog], tracing_provider=InMemoryTracingProvider)
        directory = AgentDirectory()
        async with ChatSession(make_agent("bob", lambda _: "NO_REPLY")) as b:
            directory.join(b, "bob")
            membership = directory.join(ChatSession(alice), "alice")
            context = fresh(peer=membership)
            await run(alice)
            await settle(b)
        assert app_ran == [] and received(b) == [("alice", "hi bob")]
        assert sorted(offered(context)[0]) == ["message_agent", "send_to_agent"]

    async def test_join_refuses_an_agent_that_has_a_send_to_agent_of_its_own(self):
        agent = make_agent("alice", lambda _: "ok", tools=[Tool(name="send_to_agent", handler=lambda to: to)])
        with pytest.raises(ValueError, match="send_to_agent"):
            AgentDirectory().join(ChatSession(agent), "alice")

    async def test_an_applications_waiting_call_is_not_taken_by_the_send(self):
        """An application's ``send_to_agent`` waited for approval; the agent was redeployed without it, in a
        directory. The send, free to take the name, would have messaged an agent with that approval."""
        posted = []
        model = TestModel(responses=[call("send_to_agent", to="bob", message="Internal: refund"), "done", "x"])
        app_tool = Tool(
            name="send_to_agent",
            handler=lambda to, message: posted.append(message) or "posted",
            requires_approval=True,
        )
        first = Agent(name="alice", model=model, tools=[app_tool], tracing_provider=InMemoryTracingProvider)
        fresh()
        events, parked = await run(first)
        assert parked.status.reason == "approval_required"
        approval = next(e for e in events if isinstance(e, ApprovalEvent))
        second = Agent(name="alice", model=model, tracing_provider=InMemoryTracingProvider)
        directory = AgentDirectory()
        async with ChatSession(make_agent("bob", lambda _: "NO_REPLY")) as b:
            directory.join(b, "bob")
            context = fresh(parked.run_id, peer=directory.join(ChatSession(second), "alice"))
            await run(second, resume={approval.approval_id: True})
            await settle(b)
        assert posted == [] and received(b) == []
        assert any("this call was made for" in r for r in results(context))

    async def test_a_send_waits_for_the_agents_tool_args_rails_and_stays_a_send(self):
        """Wired to the agent's rails like its own tools: an escalation is an approval, and the send that waited
        is still the send at the resume, though an application's tool took the name meanwhile."""
        app_ran = []
        catalog = Catalog()
        model = TestModel(responses=[call("send_to_agent", to="bob", message="hello"), "done", "x"])
        alice = Agent(
            name="alice",
            model=model,
            tools=[catalog],
            guardrails=[ask_for("send_to_agent")],
            tracing_provider=InMemoryTracingProvider,
        )
        directory = AgentDirectory()
        async with ChatSession(make_agent("bob", lambda _: "NO_REPLY")) as b:
            directory.join(b, "bob")
            membership = directory.join(ChatSession(alice), "alice")
            fresh(peer=membership)
            events, parked = await run(alice)
            assert parked.status.reason == "approval_required" and received(b) == []
            approval = next(e for e in events if isinstance(e, ApprovalEvent))
            catalog.offered[:] = [Tool(name="send_to_agent", handler=lambda to, message: app_ran.append(message))]
            fresh(parked.run_id, peer=membership)
            await run(alice, resume={approval.approval_id: True})
            await settle(b)
        assert app_ran == [] and received(b) == [("alice", "hello")]

    async def test_a_send_that_waited_under_another_name_keeps_it(self):
        """Named ``message_agent`` when the turn paused (a tool set had ``send_to_agent``); the set dropped it
        meanwhile. The resumed turn keeps the name, and the call runs on the send."""
        catalog = Catalog()
        catalog.offered[:] = [Tool(name="send_to_agent", handler=lambda to, message: "app")]
        model = TestModel(responses=[call("message_agent", to="bob", message="hello"), "done", "x"])
        alice = Agent(
            name="alice",
            model=model,
            tools=[catalog],
            guardrails=[ask_for("message_agent")],
            tracing_provider=InMemoryTracingProvider,
        )
        directory = AgentDirectory()
        async with ChatSession(make_agent("bob", lambda _: "NO_REPLY")) as b:
            directory.join(b, "bob")
            membership = directory.join(ChatSession(alice), "alice")
            fresh(peer=membership)
            events, parked = await run(alice)
            assert parked.status.reason == "approval_required"
            approval = next(e for e in events if isinstance(e, ApprovalEvent))
            catalog.offered.clear()
            context = fresh(parked.run_id, peer=membership)
            await run(alice, resume={approval.approval_id: True})
            await settle(b)
        assert received(b) == [("alice", "hello")]
        assert offered(context)[0] == ["message_agent"]

    async def test_a_member_that_left_is_not_offered_the_send_and_cannot_use_it(self):
        """Left mid-turn (another session may take the name): its sends would be stamped with a name that is no
        longer its own."""
        directory = AgentDirectory()
        model = TestModel(responses=[call("leave"), call("send_to_agent", to="bob", message="as alice"), "done"])
        alice = Agent(
            name="alice",
            model=model,
            tools=[Tool(name="leave", handler=lambda: membership.leave() or "left")],
            tracing_provider=InMemoryTracingProvider,
        )
        async with ChatSession(make_agent("bob", lambda _: "NO_REPLY")) as b:
            directory.join(b, "bob")
            membership = directory.join(ChatSession(alice), "alice")
            context = fresh(peer=membership)
            await run(alice)
            await settle(b)
        assert received(b) == []
        first, second, *_ = offered(context)
        assert "send_to_agent" in first and "send_to_agent" not in second

    async def test_a_callable_policy_records_the_send_it_allowed(self):
        """A parked wake restores a callable policy from what each agent recorded it allowed."""
        directory = AgentDirectory()
        alice = make_agent("alice", lambda _: "ok", tools=[Tool(name="work", handler=lambda: "w")])
        async with ChatSession(make_agent("bob", lambda _: "NO_REPLY")) as b:
            directory.join(b, "bob")
            context = fresh(peer=directory.join(ChatSession(alice), "alice"))
            context._tool_policy = lambda tool: True
            await run(alice)
        root = next(s for s in context._trace.values() if s.parent_call_id is None)
        assert root.metadata["tool_policy_allowed"] == ["alice.send_to_agent", "alice.work"]


class TestTheInbox:
    async def test_message_ids_are_per_sender(self):
        directory = AgentDirectory()
        async with ChatSession(make_agent("bob", lambda _: "NO_REPLY")) as b:
            directory.join(b, "bob")
            first = await directory.send("alice", "bob", "hi", message_id="am_1")
            second = await directory.send("carol", "bob", "hello", message_id="am_1")
            await settle(b)
        assert first["duplicate"] is False and second["duplicate"] is False
        assert received(b) == [("alice", "hi"), ("carol", "hello")]

    async def test_a_copy_that_arrives_while_the_first_is_screened_is_a_duplicate(self):
        gate, both = asyncio.Event(), asyncio.Event()
        entered = []

        async def slow(text):
            entered.append(text)
            if len(entered) == 2:
                both.set()
            await gate.wait()
            return True

        directory = AgentDirectory()
        async with ChatSession(make_agent("bob", lambda _: "NO_REPLY", guardrails=[guardrail(slow, stages=["input"])])) as b:
            directory.join(b, "bob")
            sends = [asyncio.create_task(directory.send("alice", "bob", "hi", message_id="am_1")) for _ in range(2)]
            await asyncio.wait_for(both.wait(), 5)
            gate.set()
            receipts = await asyncio.wait_for(asyncio.gather(*sends), 5)
            await settle(b)
        assert sorted(r["duplicate"] for r in receipts) == [False, True]
        assert received(b) == [("alice", "hi")]

    async def test_a_repeat_of_a_redacted_message_is_a_duplicate(self):
        gate = asyncio.Event()

        async def work() -> str:
            await gate.wait()
            return "worked"

        def bob(messages):
            if messages[-1].role == "tool":
                return "done"
            if messages[-1].collect_text() == "start":
                return call("work")
            return "NO_REPLY"

        redact = guardrail(lambda text: text.replace("4111", "[card]"), stages=["input"], action="redact")
        directory = AgentDirectory()
        agent = make_agent("bob", bob, tools=[Tool(name="work", handler=work)], guardrails=[redact])
        async with ChatSession(agent) as b:
            directory.join(b, "bob")
            b.send("start")
            await until(lambda: b.busy)
            first = await directory.send("alice", "bob", "card 4111")
            again = await directory.send("alice", "bob", "card 4111")
            gate.set()
            await settle(b)
        assert (first["duplicate"], again["duplicate"]) == (False, True)
        assert received(b) == [("alice", "card [card]")]

    async def test_a_message_a_rail_empties_is_blocked(self):
        empty = guardrail(lambda text: "", stages=["input"], action="redact")
        directory = AgentDirectory()
        async with ChatSession(make_agent("bob", lambda _: "NO_REPLY", guardrails=[empty])) as b:
            directory.join(b, "bob")
            with pytest.raises(PeerRejected) as refused:
                await directory.send("alice", "bob", "secret")
            assert not b._store.pending_events()
        assert refused.value.reason == "blocked"

    async def test_a_session_closed_while_the_message_was_screened_refuses_it(self):
        gate, entered = asyncio.Event(), asyncio.Event()

        async def slow(text):
            entered.set()
            await gate.wait()
            return True

        directory = AgentDirectory()
        b = ChatSession(make_agent("bob", lambda _: "NO_REPLY", guardrails=[guardrail(slow, stages=["input"])]))
        directory.join(b, "bob")
        send = asyncio.create_task(directory.send("alice", "bob", "hi"))
        await asyncio.wait_for(entered.wait(), 5)
        await b.close()
        gate.set()
        with pytest.raises(PeerRejected) as refused:
            await asyncio.wait_for(send, 5)
        assert refused.value.reason == "closed" and not b._store.pending_events()

    async def test_an_urgent_message_joins_a_turn_and_never_asks_to_cut_one(self):
        """Urgent is the steering lane. A channel's ``interrupt`` policy would let another agent's model cut a
        reply in flight (a call shares its session with this chat and speaks its events)."""
        directory = AgentDirectory()
        async with ChatSession(make_agent("bob", lambda _: "NO_REPLY")) as b:
            directory.join(b, "bob")
            b._hold_worker = True  # nothing takes it while it is looked at
            receipt = await directory.send("alice", "bob", "stop", priority="urgent")
            (event,) = [e for e in b._store.pending_events() if e.peer and e.peer["message_id"] == receipt["message_id"]]
            assert event.lane == "next" and event.announce is None
            assert b._scheduler.pending_turn() == "when_idle"
            b._hold_worker = False

    def test_an_app_event_is_stored_as_before(self):
        from timbal.state.background import SessionEvent

        assert "peer" not in SessionEvent("deploy done", key=None, source="app").to_dict()

    async def test_the_sender_must_be_a_name(self):
        """``from`` reaches the receiving model's notice and the session's records; over HTTP it is whatever the
        caller sends."""
        async with ChatSession(make_agent("bob", lambda _: "NO_REPLY")) as b:
            for sender in ("the user", "x" * 65, 'a"b'):
                with pytest.raises(PeerRejected) as refused:
                    await b.receive_agent_message("hi", sender=sender)
                assert refused.value.reason == "invalid"
            for message_id in ("has space", "x" * 129, "\u00e9t\u00e9"):
                with pytest.raises(PeerRejected):
                    await b.receive_agent_message("hi", sender="alice", message_id=message_id)
            assert not b._store.pending_events()


class TestTheDirectoryAcrossThreads:
    async def test_two_sessions_cannot_both_join_under_one_name(self):
        """Sessions served on their own loops join and leave from their own threads. A second join runs between
        the first one's check of the name and its write (a dict that starts it as the first one writes, and
        waits a bounded time for it): without a lock both took the name, and the first overwrote the second."""
        import threading

        directory = AgentDirectory()
        first, second = ChatSession(make_agent("a", lambda _: "ok")), ChatSession(make_agent("b", lambda _: "ok"))
        outcome, threads = {}, []

        def join_second():
            try:
                outcome["second"] = directory.join(second, "bob")
            except ValueError as e:
                outcome["second"] = e

        class Racy(dict):
            def _race(self):
                if not threads:
                    threads.append(threading.Thread(target=join_second))
                    threads[0].start()
                    threads[0].join(0.5)  # done here without a lock; held up by one until the first join ends

            def __setitem__(self, key, value):
                self._race()
                dict.__setitem__(self, key, value)

            def setdefault(self, key, default=None):
                self._race()
                return dict.setdefault(self, key, default)

        directory._members = Racy()
        joined = directory.join(first, "bob")
        threads[0].join(5)
        assert isinstance(outcome["second"], ValueError)
        assert dict.get(directory._members, "bob") is joined and second._peer is None


class TestRemotePeer:
    def _mock(self, monkeypatch, answer):
        real = httpx.AsyncClient
        calls = []

        def handler(request):
            calls.append(request)
            return answer(request)

        monkeypatch.setattr(
            httpx, "AsyncClient", lambda **kwargs: real(transport=httpx.MockTransport(handler), **kwargs)
        )
        return calls

    def test_its_headers_are_not_in_its_repr(self):
        peer = RemotePeer(url="https://bots.example", session_id="s", headers={"Authorization": "Bearer t1_secret"})
        assert "t1_secret" not in repr(peer)

    async def test_a_blocked_message_says_so(self, monkeypatch):
        self._mock(
            monkeypatch,
            lambda request: httpx.Response(403, json={"detail": "Blocked by 'pii'.", "reason": "blocked"}),
        )
        with pytest.raises(PeerRejected) as refused:
            await RemotePeer(url="https://bots.example", session_id="s").deliver({"from": "a", "text": "x"})
        assert refused.value.reason == "blocked"

    @pytest.mark.parametrize(
        "response",
        [
            httpx.Response(307, headers={"location": "https://elsewhere.example/"}),
            httpx.Response(202, text="not json"),
            httpx.Response(202, json=["a list"]),
        ],
    )
    async def test_an_answer_that_is_no_receipt_is_unreachable(self, monkeypatch, response):
        self._mock(monkeypatch, lambda request: response)
        with pytest.raises(PeerRejected) as refused:
            await RemotePeer(url="https://bots.example", session_id="s").deliver({"from": "a", "text": "x"})
        assert refused.value.reason == "unreachable"

    async def test_the_session_id_is_one_path_segment(self, monkeypatch):
        calls = self._mock(monkeypatch, lambda request: httpx.Response(202, json={"event_id": "e"}))
        await RemotePeer(url="https://bots.example/", session_id="../../run").deliver({"from": "a", "text": "x"})
        assert calls[0].url.raw_path == b"/chat/sessions/..%2F..%2Frun/agent_messages"

    async def test_only_its_own_headers_go_out(self, monkeypatch):
        """Not the run's platform credentials: the host is one the application configured for this peer."""
        from timbal.state.config import PlatformConfig

        calls = self._mock(monkeypatch, lambda request: httpx.Response(202, json={"event_id": "e"}))
        context = fresh()
        context.platform_config = PlatformConfig.model_validate(
            {"host": "api.timbal.ai", "auth": {"type": "bearer", "token": "t2_platformtoken"}}
        )
        await RemotePeer(url="https://bots.example", session_id="s", headers={"X-Key": "k"}).deliver(
            {"from": "a", "text": "x"}
        )
        sent = dict(calls[0].headers)
        assert sent["x-key"] == "k" and "authorization" not in sent
        assert "t2_platformtoken" not in str(sent)
