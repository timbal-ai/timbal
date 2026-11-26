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
    assert meta["modes"] == ["steer", "now", "followup", "collect", "interrupt"]


def test_the_playground_builds_ask_cards_from_the_session_state(client):
    """A restored session's log has no APPROVAL / INTERACTION event to replay: the cards must come from
    ``pending_approvals`` / ``pending_interactions`` of the state, every time it is fetched."""
    page = client.get("/chat/").text
    refresh = page[page.index("async function refreshState()") : page.index("function resetView()")]
    assert "pendingCards(state)" in refresh
    cards = page[page.index("function pendingCards(state)") : page.index("function resetView()")]
    assert "state.pending_approvals" in cards and 'askCard("approval"' in cards
    assert "state.pending_interactions" in cards and 'askCard("interaction"' in cards


PLAYGROUND_HARNESS = r"""
// Runs the playground's own script against a few stand-ins for the browser, feeds it a session's events
// (argv[3], a JSON list) the way its EventSource would, and prints what each tool card says.
const fs = require("fs"), vm = require("vm");
const html = fs.readFileSync(process.argv[2], "utf8");
const script = html.slice(html.indexOf("<script>") + "<script>".length, html.lastIndexOf("</script>"));

class Node {
  constructor(tag) {
    Object.assign(this, { tag, children: [], className: "", textContent: "", dataset: {}, style: {}, parent: null,
      hidden: false, disabled: false, value: "", checked: false, title: "", scrollTop: 0, scrollHeight: 0,
      clientHeight: 0, id: "" });
    const names = () => this.className.split(" ").filter(Boolean);
    this.classList = {
      add: (...added) => { this.className = [...new Set([...names(), ...added])].join(" "); },
      remove: (...gone) => { this.className = names().filter((name) => !gone.includes(name)).join(" "); },
      toggle: (name, on) => (on ? this.classList.add(name) : this.classList.remove(name)),
      contains: (name) => names().includes(name),
    };
  }
  appendChild(node) { node.parent = this; this.children.push(node); return node; }
  append(...nodes) { nodes.forEach((node) => this.appendChild(node)); }
  remove() { if (this.parent) this.parent.children = this.parent.children.filter((child) => child !== this); }
  addEventListener() {}
  querySelectorAll() { return []; }
  closest() { return null; }
  focus() {}
  set innerHTML(_value) { this.children = []; }
  get innerHTML() { return ""; }
}

const byId = {};
const sources = [];
const answer = (data) => ({ ok: true, statusText: "", json: async () => data });
const context = {
  document: {
    getElementById: (id) => (byId[id] = byId[id] || new Node("div")),
    createElement: (tag) => new Node(tag),
    querySelectorAll: () => [],
  },
  location: { href: "http://playground.test/chat/", hash: "" },
  history: { replaceState() {} },
  navigator: {},
  URL, JSON, Promise, Object, String, Math, Set, Error, console, setTimeout, clearTimeout,
  fetch: async (url, options) => {
    if (String(url).endsWith("/meta")) return answer({ agent: "agent", is_agent: true });
    if (options && options.method === "POST") return answer({ session_id: "abc123" });
    return answer({});
  },
  EventSource: class { constructor() { sources.push(this); } addEventListener() {} close() {} },
};
vm.runInNewContext(script, context);

setTimeout(() => {
  const source = sources[sources.length - 1];
  JSON.parse(process.argv[3]).forEach((event, n) => source.onmessage({ data: JSON.stringify(event), lastEventId: String(n + 1) }));
  const cards = byId.timeline.children.filter((node) => node.className.split(" ").includes("tool"));
  // A card's head is: icon, name, state.
  console.log(JSON.stringify(cards.map((card) => card.children[0].children.slice(1).map((part) => part.textContent))));
}, 50);
"""


def _playground_tool_cards(client, tmp_path, events):
    """What the playground's tool cards show after ``events``: [name, state] each, in the order drawn."""
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("needs node to run the playground's script")
    (tmp_path / "chat.html").write_text(client.get("/chat/").text, encoding="utf-8")
    (tmp_path / "harness.js").write_text(PLAYGROUND_HARNESS, encoding="utf-8")
    out = subprocess.run(
        [node, str(tmp_path / "harness.js"), str(tmp_path / "chat.html"), json.dumps(events)],
        capture_output=True,
        text=True,
        encoding="utf-8",  # node writes UTF-8 whatever the platform's own encoding is ("done · 5 ms")
        timeout=60,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def _run_events(run_id="r1"):
    """Event makers for one run: the root's START, a model call's START / tool_use delta / OUTPUT, a tool's
    START / OUTPUT."""
    base = {"run_id": run_id, "parent_run_id": None}

    def child(kind, call_id, path, **fields):
        return {**base, "type": kind, "call_id": call_id, "parent_call_id": "root", "path": path, **fields}

    return {
        "turn": {"type": "SESSION", "kind": "turn_started", "run_id": run_id, "trigger": {"kind": "user_message"}},
        "root": {**base, "type": "START", "call_id": "root", "parent_call_id": None, "path": "agent"},
        "llm_start": lambda call: child("START", call, "agent.llm"),
        "llm_calls": lambda call, use, name: child(
            "DELTA", call, "agent.llm", item={"type": "tool_use", "id": use, "name": name, "input": "{}"}
        ),
        "llm_output": lambda call, use, name: child(
            "OUTPUT",
            call,
            "agent.llm",
            status={"code": "success"},
            output={"role": "assistant", "content": [{"type": "tool_use", "id": use, "name": name, "input": {}}]},
        ),
        "tool_start": lambda call, name: child("START", call, f"agent.{name}", input={}),
        "tool_output": lambda call, name: child(
            "OUTPUT", call, f"agent.{name}", status={"code": "success"}, output="found", t0=0, t1=5
        ),
    }


def test_the_playground_does_not_give_a_cut_calls_card_to_the_next_call(client, tmp_path):
    """A model call streams a tool call and is cut by a 'now' message before the tool runs: its card never
    gets a START. The step after the cut calls a tool of the same name, and that call's START and result
    must land on its own card, not on the one left over."""
    e = _run_events()
    cards = _playground_tool_cards(
        client,
        tmp_path,
        [
            e["turn"],
            e["root"],
            e["llm_start"]("llm1"),
            e["llm_calls"]("llm1", "use1", "search"),  # ...and the answer is cut here: no OUTPUT, no tool
            e["llm_start"]("llm2"),
            e["llm_calls"]("llm2", "use2", "search"),
            e["llm_output"]("llm2", "use2", "search"),
            e["tool_start"]("tool1", "search"),
            e["tool_output"]("tool1", "search"),
        ],
    )
    assert cards == [["search", "not run"], ["search", "done · 5 ms"]]


def test_the_playground_still_pairs_a_batch_of_same_named_calls_in_order(client, tmp_path):
    e = _run_events()
    cards = _playground_tool_cards(
        client,
        tmp_path,
        [
            e["turn"],
            e["root"],
            e["llm_start"]("llm1"),
            e["llm_calls"]("llm1", "use1", "search"),
            e["llm_calls"]("llm1", "use2", "search"),
            e["tool_start"]("tool1", "search"),
            e["tool_start"]("tool2", "search"),
            e["tool_output"]("tool1", "search"),
            e["llm_start"]("llm2"),  # the second one is still running when the model is asked again
        ],
    )
    assert cards == [["search", "done · 5 ms"], ["search", "running…"]]


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


def test_an_agent_message_starts_a_turn_and_a_repeat_is_a_duplicate(client):
    session_id = open_session(client)
    client.post(f"/chat/sessions/{session_id}/messages", json={"text": "hi"})
    read_log(client, session_id, until_kind="turn_finished", count=1)
    url = f"/chat/sessions/{session_id}/agent_messages"
    body = {"from": "planner", "text": "the build is green", "message_id": "am_1", "hops": 1}
    sent = client.post(url, json=body)
    assert sent.status_code == 202, sent.text
    assert sent.json()["duplicate"] is False and sent.json()["message_id"] == "am_1"
    assert client.post(url, json=body).json()["duplicate"] is True
    items = read_log(client, session_id, until_kind="turn_finished", count=2)
    (received,) = [item for _, item in items if item.get("kind") == "agent_message"]
    assert received["sender"] == "planner" and received["hops"] == 1
    started = [item for _, item in items if item.get("kind") == "turn_started"]
    assert started[-1]["trigger"]["kind"] == "session_event"
    assert client.post(url, json={"from": "planner"}).status_code == 400
    assert client.post(url, json={"from": "planner", "text": "x", "priority": "asap"}).status_code == 422
    assert client.post("/chat/sessions/nope/agent_messages", json=body).status_code == 404


def test_an_agent_message_the_answers_could_not_carry_is_a_400_and_not_queued(client):
    session_id = open_session(client)
    url = f"/chat/sessions/{session_id}/agent_messages"
    for body in (
        {"from": "planner", "text": "hi \ud800"},
        {"from": "planner", "text": "hi", "message_id": "am_\ud800"},
        {"from": "planner \ud800", "text": "hi"},
    ):
        sent = client.post(url, content=json.dumps(body), headers={"content-type": "application/json"})
        assert sent.status_code == 400, sent.text
    log = client.get(f"/chat/sessions/{session_id}/log", params={"after": 0, "wait_ms": 0}).json()["events"]
    assert not [entry for entry in log if entry["data"].get("kind") == "agent_message"]


def test_an_agent_message_refused_by_the_session_says_why(client, monkeypatch):
    from timbal.core.chat_session import ChatSession
    from timbal.core.peers import PeerRejected

    session_id = open_session(client)
    url = f"/chat/sessions/{session_id}/agent_messages"
    refused = client.post(url, json={"from": "the planner", "text": "hi"})
    assert refused.status_code == 422 and refused.json()["reason"] == "invalid"

    async def blocked(_self, _text, _sender, _peer):
        raise PeerRejected("blocked", "Blocked by the recipient's 'pii' guardrail.")

    monkeypatch.setattr(ChatSession, "_screen_agent_message", blocked)
    refused = client.post(url, json={"from": "planner", "text": "my card is 4111"})
    assert refused.status_code == 403
    assert refused.json() == {"detail": "Blocked by the recipient's 'pii' guardrail.", "reason": "blocked"}


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


def test_a_wake_a_response_cannot_carry_is_a_400(client):
    """The wake's fields reach the session's log, its state and its run's output: NaN, Infinity or a lone
    surrogate (the body parser takes all three) would leave answers about the session that cannot be encoded."""
    session_id = open_session(client)
    url = f"/chat/sessions/{session_id}/wake"
    for body in (
        '{"text": "x", "trigger": {"kind": "routine", "ratio": NaN}}',
        '{"text": "x", "trigger": {"kind": "routine", "note": "\\ud800"}}',
        '{"text": "x", "key": "k\\udc80"}',
        '{"text": "x\\ud800"}',
        '{"text": "x", "tool_policy": ["\\ud800"]}',
    ):
        refused = client.post(url, content=body, headers={"content-type": "application/json"})
        assert refused.status_code == 400, (body, refused.text)
    state = client.get(f"/chat/sessions/{session_id}")
    assert state.status_code == 200 and state.json()["wakes"] == []


def test_a_message_or_session_a_response_cannot_carry_is_a_400(client):
    """Before: the message was queued and every answer about the session was a 500 from then on (a client's
    retry sent it twice); a session opened on such a parent made the session list a 500 for everyone."""
    session_id = open_session(client)
    sent = client.post(
        f"/chat/sessions/{session_id}/messages",
        content='{"text": "hi \\ud800"}',
        headers={"content-type": "application/json"},
    )
    assert sent.status_code == 400, sent.text
    assert client.get(f"/chat/sessions/{session_id}").status_code == 200
    assert client.get(f"/chat/sessions/{session_id}/log", params={"after": 0}).status_code == 200
    opened = client.post(
        "/chat/sessions", content='{"parent_id": "p\\ud800"}', headers={"content-type": "application/json"}
    )
    assert opened.status_code == 400, opened.text
    assert client.get("/chat/sessions").status_code == 200
    # A key no route reads is taken as before, whatever it holds.
    unread = client.post(
        f"/chat/sessions/{session_id}/messages",
        content='{"text": "hi", "client_ts": NaN}',
        headers={"content-type": "application/json"},
    )
    assert unread.status_code == 202, unread.text


def test_a_wakes_silent_ok_is_taken_as_before(client):
    """The route makes it a bool and nothing echoes it: whatever it holds, the wake is taken, as on main."""
    session_id = open_session(client)
    for body in ('{"text": "x", "silent_ok": NaN}', '{"text": "x", "silent_ok": "\\ud800"}'):
        woke = client.post(
            f"/chat/sessions/{session_id}/wake", content=body, headers={"content-type": "application/json"}
        )
        assert woke.status_code == 202, woke.text


@pytest.mark.parametrize(
    ("values", "carried"),
    [
        ({"ask-1": float("nan")}, True),  # an interaction's answer: the tool gets it, nothing echoes it raw
        ({"ask-1": {"score": float("inf")}}, True),
        ({"approval-1": True, "stale": {"x": float("nan")}}, True),
        ({"approval-1": {"approved": True, "comment": float("nan")}}, True),
        ({"approval-1": {"approved": True, "metadata": {"ratio": float("nan")}}}, False),  # stamped as given
        ({"approval-1": {"approved": True, "override_input": {"amount": float("inf")}}}, False),
        ({"approval-1": {"approved": True, "comment": "ok \ud800"}}, False),  # no UTF-8 answer carries it
        ({"ask-1": "a\udc80"}, False),
        # A question's answer is whatever the tool asked for, an "approved" field included: it is told from an
        # approval's by the id it answers, not by its shape.
        ({"ask-1": {"approved": True, "metadata": {"score": float("nan")}}}, True),
        ({"approval-1": {"metadata": {"ratio": float("nan")}}}, False),  # an approval's, whatever it holds
    ],
    ids=[
        "nan answer",
        "infinity answer",
        "stale id",
        "nan comment",
        "nan metadata",
        "infinity override",
        "surrogate comment",
        "surrogate",
        "a question's answer with an approved field",
        "an approval's answer without one",
    ],
)
def test_which_answers_a_session_can_carry(values, carried):
    from timbal.server.chat import _answers_carried

    assert _answers_carried(values, {"approval-1"}) is carried


def test_answers_to_an_unknown_question_are_checked_as_approvals():
    """While the session does not know what its turn waits for (it is still reading the run it continues),
    every answer may be an approval's."""
    from timbal.server.chat import _answers_carried

    assert _answers_carried({"ask-1": {"metadata": {"score": float("nan")}}}, None) is False
    assert _answers_carried({"ask-1": {"score": float("nan")}}, None) is True


SURVEY_AGENT = '''
from timbal import Agent, Tool
from timbal.core.test_model import TestModel
from timbal.state import suspend
from timbal.state.tracing.providers import InMemoryTracingProvider
from timbal.types.content import ToolUseContent
from timbal.types.message import Message


def survey() -> str:
    """Ask the user to rate the answer."""
    return f"rated {suspend({'question': 'How was it?'})!r}"


def refund(order: str) -> str:
    """Refund an order."""
    return f"refunded {order}"


def handler(messages):
    last = messages[-1]
    if last.role == "tool":
        return "thanks"
    name = "refund" if last.collect_text() == "refund" else "survey"
    return Message(
        role="assistant",
        content=[ToolUseContent(id=f"c-{len(messages)}", name=name, input={"order": "A-1"} if name == "refund" else {})],
        stop_reason="tool_use",
    )


agent = Agent(
    name="agent",
    model=TestModel(handler=handler),
    tools=[survey, Tool(name="refund", handler=refund, requires_approval=True)],
    tracing_provider=InMemoryTracingProvider,
)
'''


def test_a_questions_answer_is_not_checked_as_an_approval(tmp_path, monkeypatch):
    """The reported case: a survey (``suspend()``) asks for an object of its own, one with an ``approved``
    field. Its answer, NaN in its ``metadata`` included, reaches the tool (202); an approval's answer with the
    same body is refused (400), since its metadata is stamped on the span as given."""
    (tmp_path / "agent.py").write_text(SURVEY_AGENT)
    monkeypatch.setenv("TIMBAL_RUNNABLE", f"{tmp_path / 'agent.py'}::agent")
    monkeypatch.delenv("TIMBAL_CHAT_STORE", raising=False)
    answer = '{"approved": true, "metadata": {"score": NaN}}'
    with TestClient(create_app()) as client:
        for prompt, expected in (("rate", 202), ("refund", 400)):
            session_id = open_session(client)
            sent = client.post(f"/chat/sessions/{session_id}/messages", json={"text": prompt})
            assert sent.status_code == 202
            read_log(client, session_id, until_kind="turn_finished", count=1)
            state = client.get(f"/chat/sessions/{session_id}").json()
            assert state["parked"], state
            (pending,) = state.get("pending_interactions") or state.get("pending_approvals")
            pending_id = pending.get("interaction_id") or pending.get("approval_id")
            resumed = client.post(
                f"/chat/sessions/{session_id}/resume",
                content=f'{{"resume": {{"{pending_id}": {answer}}}}}',
                headers={"content-type": "application/json"},
            )
            assert resumed.status_code == expected, resumed.text
            if expected == 202:
                items = read_log(client, session_id, until_kind="turn_finished", count=2)
                finished = [item for _, item in items if item.get("kind") == "turn_finished"]
                assert finished[-1]["status"]["code"] == "success" and not finished[-1]["parked"]
            else:
                assert client.get(f"/chat/sessions/{session_id}").json()["parked"]


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


REFUSED_TRACES_AGENT = '''
from timbal import Agent
from timbal.core.test_model import TestModel
from timbal.state.tracing.providers import InMemoryTracingProvider


class Refused(InMemoryTracingProvider):
    """A trace store that turns every save down, for good (a platform answering 403)."""

    @classmethod
    async def _store(cls, _run_context):
        raise PermissionError("403: the platform refused the trace")


agent = Agent(
    name="agent",
    model=TestModel(handler=lambda messages: f"echo: {messages[-1].collect_text()}"),
    tools=[],
    tracing_provider=Refused,
)
'''


def test_a_session_without_a_store_goes_on_when_its_traces_are_refused(tmp_path, monkeypatch):
    """The chat route with no session store (the default over platform traces): a trace the provider refuses
    is logged, and the turn is over. Held open for it, the session would stay busy and every later message
    queued, with no ``turn_finished`` ever sent."""
    (tmp_path / "agent.py").write_text(REFUSED_TRACES_AGENT)
    monkeypatch.setenv("TIMBAL_RUNNABLE", f"{tmp_path / 'agent.py'}::agent")
    monkeypatch.delenv("TIMBAL_CHAT_STORE", raising=False)
    with TestClient(create_app()) as client:
        assert client.get("/chat/meta").json()["durable"] is False
        session_id = open_session(client)
        for message in ("one", "two"):
            # followup: a turn of its own each (steered, the second may join the first, and one turn ends)
            sent = client.post(f"/chat/sessions/{session_id}/messages", json={"text": message, "mode": "followup"})
            assert sent.status_code == 202
        items = read_log(client, session_id, until_kind="turn_finished", count=2, timeout=20)
        finished = [item for _, item in items if item.get("kind") == "turn_finished"]
        assert [turn["status"]["code"] for turn in finished] == ["success", "success"]
        state = client.get(f"/chat/sessions/{session_id}").json()
        assert not state["busy"] and not state["pending"]
