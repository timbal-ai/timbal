"""Deferred tool schemas: client-side search_tools, native Anthropic tool search, persistence."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from timbal import Agent, Tool
from timbal.core.agent import SEARCH_TOOLS_NAME, rank_tools
from timbal.core.llm.messages import ANTHROPIC_TOOL_SEARCH_NAME, ANTHROPIC_TOOL_SEARCH_TYPE, prepare_messages_request
from timbal.core.test_model import TestModel
from timbal.state import set_call_id, set_parent_call_id, set_run_context
from timbal.state.context import RunContext
from timbal.state.tracing.providers import InMemoryTracingProvider
from timbal.types.content import ToolUseContent
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


def call(name, call_id="c1", **input):
    return Message(
        role="assistant", content=[ToolUseContent(id=call_id, name=name, input=input)], stop_reason="tool_use"
    )


def context(parent_id=None):
    ctx = RunContext(parent_id=parent_id, tracing_provider=InMemoryTracingProvider)
    set_run_context(ctx)
    return ctx


async def trace_of(output):
    return await RunContext(tracing_provider=InMemoryTracingProvider)._get_parent_trace(output.run_id)


async def offered_per_call(output):
    """Tool names the model was offered on each LLM call of a run, in order."""
    trace = await trace_of(output)
    llm_spans = sorted((s for s in trace.values() if s.path.endswith(".llm")), key=lambda s: s.t0)
    return [[t["name"] if isinstance(t, dict) else t.name for t in (s.input.get("tools") or [])] for s in llm_spans]


def catalog(calls=None):
    calls = calls if calls is not None else []

    def get_weather(city: str) -> str:
        calls.append(("get_weather", city))
        return f"sunny in {city}"

    def create_invoice(customer: str, amount: float = 0) -> str:  # noqa: ARG001
        calls.append(("create_invoice", customer))
        return "invoice created"

    def list_repos(org: str) -> str:  # noqa: ARG001
        return "timbal"

    def lookup(q: str) -> str:
        return q

    return [
        Tool(name="lookup", handler=lookup, read_only=True),
        Tool(name="get_weather", description="Current weather for a city.", handler=get_weather, defer_loading=True),
        Tool(
            name="create_invoice",
            description="Bill a customer.",
            handler=create_invoice,
            defer_loading=True,
        ),
        Tool(name="list_repos", description="List GitHub repositories.", handler=list_repos, defer_loading=True),
    ]


def agent(model, tools=None, **kwargs):
    return Agent(
        name="agent",
        model=model,
        tools=tools or catalog(),
        tracing_provider=InMemoryTracingProvider,
        max_tokens=1024,
        **kwargs,
    )


class TestRanking:
    def test_name_beats_description(self):
        tools = catalog()[1:]
        assert [t.name for t in rank_tools(tools, "weather")] == ["get_weather"]
        assert [t.name for t in rank_tools(tools, "bill customer")][0] == "create_invoice"

    def test_param_names_count(self):
        assert [t.name for t in rank_tools(catalog()[1:], "org")] == ["list_repos"]

    def test_exact_name_first(self):
        assert rank_tools(catalog()[1:], "list_repos")[0].name == "list_repos"

    def test_a_word_every_tool_shares_does_not_count(self):
        def noop() -> str:
            return ""

        tools = [
            Tool(name=f"timbal__{n}", description=f"Timbal: {d}.", handler=noop)
            for n, d in [("whoami", "the current user"), ("list_runs", "list runs"), ("get_span", "one span")]
        ]
        assert [t.name for t in rank_tools(tools, "timbal whoami")] == ["timbal__whoami"]
        # Only shared words: keep them rather than match nothing.
        assert len(rank_tools(tools, "timbal")) == 3

    def test_no_match_and_empty_query(self):
        assert rank_tools(catalog(), "zzz") == []
        assert rank_tools(catalog(), "  ") == []


class TestClientSearch:
    async def test_deferred_hidden_behind_search_tools(self):
        context()
        a = agent(TestModel(responses=["ok"]))
        tools, _ = await a._resolve_tools(0)
        offered, dispatch, native, unloaded = await a._plan_tool_search(tools, a.model)
        assert [t.name for t in offered] == ["lookup", SEARCH_TOOLS_NAME]
        assert native is None
        assert unloaded == {"get_weather", "create_invoice", "list_repos"}
        assert {t.name for t in dispatch} == {
            "lookup",
            "get_weather",
            "create_invoice",
            "list_repos",
            SEARCH_TOOLS_NAME,
        }
        search = offered[-1]
        assert search.read_only
        for name in unloaded:
            assert name in search.description

    async def test_search_loads_then_calls(self):
        calls = []
        model = TestModel(
            responses=[
                call(SEARCH_TOOLS_NAME, query="weather"),
                call("get_weather", "c2", city="Paris"),
                "It is sunny.",
            ]
        )
        out = await agent(model, catalog(calls))(prompt="weather in Paris?").collect()
        assert out.status.code == "success", out.error
        assert calls == [("get_weather", "Paris")]
        offered = await offered_per_call(out)
        assert offered[0] == ["lookup", SEARCH_TOOLS_NAME]
        # Loaded tools are appended after the others: the earlier prefix is unchanged.
        assert offered[1] == ["lookup", "get_weather", SEARCH_TOOLS_NAME]
        trace = await trace_of(out)
        search_out = next(s for s in trace.values() if s.path.endswith(SEARCH_TOOLS_NAME)).output
        assert json.loads(search_out)["loaded"][0]["name"] == "get_weather"

    async def test_loaded_tools_persist_across_turns(self):
        model = TestModel(responses=[call(SEARCH_TOOLS_NAME, query="invoice"), "done", "second turn"])
        a = agent(model)
        first = await a(prompt="bill acme").collect()
        context(parent_id=first.run_id)
        second = await a(prompt="again").collect()
        assert second.status.code == "success", second.error
        assert (await offered_per_call(second))[0] == ["lookup", "create_invoice", SEARCH_TOOLS_NAME]

    async def test_search_tools_gone_once_everything_loaded(self):
        model = TestModel(responses=[call(SEARCH_TOOLS_NAME, query="weather invoice repos github", limit=10), "done"])
        out = await agent(model)(prompt="load all").collect()
        offered = (await offered_per_call(out))[1]
        assert offered[0] == "lookup" and set(offered[1:]) == {"get_weather", "create_invoice", "list_repos"}

    async def test_weak_matches_not_loaded(self):
        def noop() -> str:
            return ""

        tools = [
            Tool(name=name, description=desc, handler=noop, defer_loading=True)
            for name, desc in [
                ("zendesk_create_ticket", "Create a support ticket."),
                ("jira_create_issue", "Create an issue."),
                ("zendesk_list_ticket", "List support tickets."),
            ]
        ]
        model = TestModel(responses=[call(SEARCH_TOOLS_NAME, query="zendesk create ticket"), "ok"])
        out = await agent(model, tools)(prompt="x").collect()
        offered = (await offered_per_call(out))[1]
        assert "jira_create_issue" not in offered
        assert {"zendesk_create_ticket", "zendesk_list_ticket"} <= set(offered)

    async def test_no_match_lists_hidden_tools(self):
        model = TestModel(responses=[call(SEARCH_TOOLS_NAME, query="spaceship"), "nothing"])
        out = await agent(model)(prompt="x").collect()
        trace = await trace_of(out)
        result = next(s for s in trace.values() if s.path.endswith(SEARCH_TOOLS_NAME)).output
        assert "No tool matched" in result and "list_repos" in result
        assert (await offered_per_call(out))[1] == ["lookup", SEARCH_TOOLS_NAME]

    async def test_repeated_misses_tell_the_model_to_stop(self):
        model = TestModel(
            responses=[call(SEARCH_TOOLS_NAME, query="spaceship"), call(SEARCH_TOOLS_NAME, "c2", query="rocket"), "ok"]
        )
        out = await agent(model)(prompt="x").collect()
        trace = await trace_of(out)
        results = [s.output for s in sorted(trace.values(), key=lambda s: s.t0) if s.path.endswith(SEARCH_TOOLS_NAME)]
        assert "Try other words" in results[0]
        assert "Stop searching" in results[1] and "2 searches" in results[1]
        assert out.metadata["tool_search"]["searches"] == [
            {"query": "spaceship", "loaded": []},
            {"query": "rocket", "loaded": []},
        ]

    async def test_restricted_turn_says_hidden_tools_cannot_be_found(self):
        tools = catalog()
        tools[1].read_only = True
        a = agent(TestModel(responses=["ok"]), tools)
        ctx = context()
        ctx._tool_policy = "read_only"
        resolved, _ = await a._resolve_tools(0)
        offered, *_ = await a._plan_tool_search(resolved, a.model)
        assert "can't be found" in offered[-1].description
        ctx._tool_policy = None
        offered, *_ = await a._plan_tool_search((await a._resolve_tools(0))[0], a.model)
        assert "can't be found" not in offered[-1].description

    async def test_direct_call_of_unloaded_tool_runs_and_loads(self):
        calls = []
        model = TestModel(responses=[call("get_weather", city="Rome"), "sunny"])
        out = await agent(model, catalog(calls))(prompt="weather").collect()
        assert calls == [("get_weather", "Rome")]
        assert (await offered_per_call(out))[1] == ["lookup", "get_weather", SEARCH_TOOLS_NAME]

    async def test_tool_policy_applies_before_deferral(self):
        tools = catalog()
        tools[1].read_only = True  # get_weather
        a = agent(TestModel(responses=["ok"]), tools)
        ctx = context()
        ctx._tool_policy = "read_only"
        resolved, _ = await a._resolve_tools(0)
        offered, dispatch, _, unloaded = await a._plan_tool_search(resolved, a.model)
        assert unloaded == {"get_weather"}
        assert "create_invoice" not in {t.name for t in dispatch}
        assert "create_invoice" not in offered[-1].description

    async def test_search_tool_renamed_when_an_agent_tool_has_its_name(self):
        def their_search(q: str) -> str:
            return q

        calls = []
        tools = [
            *catalog(calls),
            Tool(name=SEARCH_TOOLS_NAME, description="The server's own search.", handler=their_search),
        ]
        model = TestModel(responses=[call("find_tools", query="weather"), call("get_weather", "c2", city="Bern"), "ok"])
        out = await agent(model, tools)(prompt="x").collect()
        assert out.status.code == "success", out.error
        offered = await offered_per_call(out)
        assert offered[0] == ["lookup", SEARCH_TOOLS_NAME, "find_tools"]
        assert offered[1][-1] == "find_tools" and "get_weather" in offered[1]
        assert calls == [("get_weather", "Bern")]

    async def test_off_sends_everything(self):
        context()
        a = agent(TestModel(responses=["ok"]), tool_search="off")
        tools, _ = await a._resolve_tools(0)
        offered, _, native, unloaded = await a._plan_tool_search(tools, a.model)
        assert [t.name for t in offered] == ["lookup", "get_weather", "create_invoice", "list_repos"]
        assert native is None and not unloaded

    async def test_nothing_deferred_no_search_tool(self):
        context()
        a = agent(TestModel(responses=["ok"]), [catalog()[0]])
        tools, _ = await a._resolve_tools(0)
        offered, _, native, _ = await a._plan_tool_search(tools, a.model)
        assert [t.name for t in offered] == ["lookup"] and native is None


class TestNativeAnthropic:
    @pytest.mark.parametrize(
        "model, native",
        [
            ("anthropic/claude-haiku-4-5", True),
            ("anthropic/claude-sonnet-4-6", True),
            ("anthropic/claude-opus-4-5-20251101", True),
            ("anthropic/claude-opus-5", True),
            ("anthropic/claude-sonnet-4-20250514", False),
            ("anthropic/claude-opus-4-1", False),
            ("anthropic/claude-3-7-sonnet-latest", False),
            ("openai/gpt-5.5", True),
            ("openai/gpt-5.4", True),
            ("openai/gpt-6-luna", True),
            ("openai/gpt-5.3", False),
            ("openai/gpt-4o-mini", False),
            ("groq/openai/gpt-oss-120b", False),
        ],
    )
    async def test_model_gating(self, model, native):
        context()
        a = agent(model)
        tools, _ = await a._resolve_tools(0)
        offered, _, deferred, _ = await a._plan_tool_search(tools, model)
        if native:
            assert deferred == ["get_weather", "create_invoice", "list_repos"]
            assert len(offered) == 4  # all sent; Anthropic hides the deferred ones
        else:
            assert deferred is None
            assert offered[-1].name == SEARCH_TOOLS_NAME

    async def test_client_mode_overrides_native(self):
        context()
        a = agent("anthropic/claude-haiku-4-5", tool_search="client")
        tools, _ = await a._resolve_tools(0)
        _, _, deferred, _ = await a._plan_tool_search(tools, a.model)
        assert deferred is None

    async def test_request_marks_deferred_and_adds_search_tool(self):
        sent = await anthropic_request_tools(catalog(), deferred_tools=["get_weather", "list_repos"])
        by_name = {t["name"]: t for t in sent}
        assert by_name["get_weather"]["defer_loading"] is True
        assert by_name["list_repos"]["defer_loading"] is True
        assert "defer_loading" not in by_name["lookup"]
        assert "defer_loading" not in by_name["create_invoice"]
        assert sent[-1] == {"type": ANTHROPIC_TOOL_SEARCH_TYPE, "name": ANTHROPIC_TOOL_SEARCH_NAME}

    async def test_no_deferred_no_search_tool(self):
        sent = await anthropic_request_tools(catalog())
        assert all("defer_loading" not in t and "type" not in t for t in sent)

    async def test_deferred_names_absent_from_tools_ignored(self):
        sent = await anthropic_request_tools(catalog()[:1], deferred_tools=["get_weather"])
        assert [t.get("name") for t in sent] == ["lookup"]


async def anthropic_request_tools(tools, **kwargs):
    captured = {}

    async def nothing():
        return
        yield

    class Messages:
        async def create(self, **request):
            captured.update(request)
            return nothing()

    create_stream, _ = prepare_messages_request(
        client=SimpleNamespace(messages=Messages(), beta=SimpleNamespace(messages=Messages())),
        model_name="claude-haiku-4-5",
        request_headers={},
        system_prompt=None,
        messages=[Message.validate({"role": "user", "content": "hi"})],
        tools=tools,
        max_tokens=100,
        temperature=None,
        output_model=None,
        provider_params={},
        **kwargs,
    )
    async for _ in create_stream():
        pass
    return captured["tools"]


class TestMCPDeferLoading:
    def test_server_flag_marks_tools(self):
        from mcp.types import Tool as MCPToolDef
        from timbal.core.mcp import MCPServer

        server = MCPServer(name="gh", transport="http", url="http://localhost:1/mcp", defer_loading=True)
        tool = server._make_tool(MCPToolDef(name="list_issues", inputSchema={"type": "object", "properties": {}}))
        assert tool.defer_loading is True
        plain = MCPServer(name="gh", transport="http", url="http://localhost:1/mcp")
        assert plain._make_tool(MCPToolDef(name="x", inputSchema={"type": "object"})).defer_loading is False


# -- Native search through the agent loop (scripted Anthropic stream) ------------


class FakeAnthropic:
    """Streams scripted assistant messages as real SDK events; records every request."""

    def __init__(self, scripts):
        self.scripts = list(scripts)
        self.requests = []
        self.messages = self

    async def create(self, **request):
        self.requests.append(request)
        blocks, stop_reason = self.scripts.pop(0)
        return self._stream(blocks, stop_reason)

    async def _stream(self, blocks, stop_reason):
        from anthropic.types import (
            InputJSONDelta,
            RawContentBlockDeltaEvent,
            RawContentBlockStartEvent,
            RawContentBlockStopEvent,
            RawMessageDeltaEvent,
            RawMessageStartEvent,
            RawMessageStopEvent,
            ServerToolUseBlock,
            TextBlock,
            TextDelta,
            ToolSearchToolResultBlock,
            ToolUseBlock,
        )

        yield RawMessageStartEvent(
            type="message_start",
            message={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-haiku-5-5",
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 0},
            },
        )
        for index, block in enumerate(blocks):
            kind = block["type"]
            if kind == "text":
                start = TextBlock(type="text", text="")
                delta = TextDelta(type="text_delta", text=block["text"])
            elif kind in ("tool_use", "server_tool_use"):
                cls = ToolUseBlock if kind == "tool_use" else ServerToolUseBlock
                start = cls(type=kind, id=block["id"], name=block["name"], input={})
                delta = InputJSONDelta(type="input_json_delta", partial_json=json.dumps(block["input"]))
            else:
                start, delta = ToolSearchToolResultBlock(**block), None
            yield RawContentBlockStartEvent(type="content_block_start", index=index, content_block=start)
            if delta is not None:
                yield RawContentBlockDeltaEvent(type="content_block_delta", index=index, delta=delta)
            yield RawContentBlockStopEvent(type="content_block_stop", index=index)
        yield RawMessageDeltaEvent(
            type="message_delta",
            delta={"stop_reason": stop_reason, "stop_sequence": None},
            usage={"output_tokens": 5},
        )
        yield RawMessageStopEvent(type="message_stop")


SEARCH_RESULT = {
    "type": "tool_search_tool_result",
    "tool_use_id": "srvtoolu_1",
    "content": {
        "type": "tool_search_tool_search_result",
        "tool_references": [{"type": "tool_reference", "tool_name": "get_weather"}],
    },
}


class TestNativeLoop:
    async def test_search_round_trips_through_the_loop(self, monkeypatch):
        from unittest.mock import patch

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        calls = []
        fake = FakeAnthropic(
            [
                (
                    [
                        {
                            "type": "server_tool_use",
                            "id": "srvtoolu_1",
                            "name": ANTHROPIC_TOOL_SEARCH_NAME,
                            "input": {"query": "weather"},
                        },
                        SEARCH_RESULT,
                        {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Oslo"}},
                    ],
                    "tool_use",
                ),
                ([{"type": "text", "text": "Sunny in Oslo."}], "end_turn"),
            ]
        )
        with patch("timbal.core.llm.clients._get_client", return_value=fake):
            out = await agent("anthropic/claude-haiku-5-5", catalog(calls))(prompt="weather in Oslo?").collect()
        assert out.status.code == "success", out.error
        assert out.output.collect_text() == "Sunny in Oslo."
        assert calls == [("get_weather", "Oslo")]

        first, second = fake.requests
        by_name = {t["name"]: t for t in first["tools"]}
        assert by_name["get_weather"].get("defer_loading") is True
        assert "defer_loading" not in by_name["lookup"]
        assert by_name[ANTHROPIC_TOOL_SEARCH_NAME]["type"] == ANTHROPIC_TOOL_SEARCH_TYPE
        assert SEARCH_TOOLS_NAME not in by_name  # native: no client-side search tool
        # The search blocks go back unchanged, and only the client tool gets a tool_result.
        assistant = next(m for m in second["messages"] if m["role"] == "assistant")
        types = [block["type"] for block in assistant["content"]]
        assert types == ["server_tool_use", "tool_search_tool_result", "tool_use"]
        assert assistant["content"][1] == SEARCH_RESULT
        results = [b for m in second["messages"] if m["role"] == "user" for b in m["content"] if isinstance(b, dict)]
        assert [b["tool_use_id"] for b in results if b.get("type") == "tool_result"] == ["toolu_1"]
        assert out.metadata["tool_search"] == {
            "mode": "native",
            "loaded": ["get_weather"],
            "searches": [{"query": None, "loaded": ["get_weather"]}],
        }

    async def test_searched_turn_replays_on_the_next_turn(self, monkeypatch):
        from unittest.mock import patch

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        fake = FakeAnthropic(
            [
                (
                    [
                        {
                            "type": "server_tool_use",
                            "id": "srvtoolu_1",
                            "name": ANTHROPIC_TOOL_SEARCH_NAME,
                            "input": {"query": "weather"},
                        },
                        SEARCH_RESULT,
                        {"type": "text", "text": "Found it."},
                    ],
                    "end_turn",
                ),
                ([{"type": "text", "text": "Again."}], "end_turn"),
            ]
        )
        a = agent("anthropic/claude-haiku-5-5")
        with patch("timbal.core.llm.clients._get_client", return_value=fake):
            first = await a(prompt="find weather").collect()
            context(parent_id=first.run_id)
            second = await a(prompt="again").collect()
        assert second.status.code == "success", second.error
        replayed = fake.requests[1]["messages"]
        assert any(
            isinstance(b, dict) and b.get("type") == "tool_search_tool_result"
            for m in replayed
            if m["role"] == "assistant"
            for b in m["content"]
        )


# -- Observability ----------------------------------------------------------------


class TestRecord:
    async def test_client_search_and_direct_call_recorded(self):
        model = TestModel(
            responses=[
                call(SEARCH_TOOLS_NAME, query="weather"),
                call("list_repos", "c2", org="timbal"),
                "done",
            ]
        )
        out = await agent(model)(prompt="x").collect()
        record = out.metadata["tool_search"]
        assert record["mode"] == "client"
        assert record["loaded"] == ["get_weather", "list_repos"]
        assert record["searches"] == [
            {"query": "weather", "loaded": ["get_weather"]},
            {"query": None, "loaded": ["list_repos"]},
        ]

    async def test_no_record_without_loads(self):
        out = await agent(TestModel(responses=["hi"]))(prompt="x").collect()
        assert "tool_search" not in out.metadata


# -- Parallel calls, sub-agents, approval, persistence -----------------------------


class TestEdges:
    async def test_search_and_direct_call_in_one_batch(self):
        calls = []
        model = TestModel(
            responses=[
                Message(
                    role="assistant",
                    content=[
                        ToolUseContent(id="c1", name=SEARCH_TOOLS_NAME, input={"query": "invoice"}),
                        ToolUseContent(id="c2", name="get_weather", input={"city": "Rome"}),
                    ],
                    stop_reason="tool_use",
                ),
                "done",
            ]
        )
        out = await agent(model, catalog(calls))(prompt="x").collect()
        assert calls == [("get_weather", "Rome")]
        offered = (await offered_per_call(out))[1]
        assert offered[0] == "lookup" and set(offered[1:3]) == {"create_invoice", "get_weather"}
        assert offered[-1] == SEARCH_TOOLS_NAME

    async def test_sub_agent_loads_stay_its_own(self):
        child = Agent(
            name="child",
            model=TestModel(responses=[call(SEARCH_TOOLS_NAME, query="weather"), "child done"]),
            tools=catalog()[1:],
            tracing_provider=InMemoryTracingProvider,
        )
        parent_model = TestModel(responses=[call("child", prompt="go"), "parent done"])
        parent = Agent(
            name="parent",
            model=parent_model,
            tools=[child, *catalog()[1:]],
            tracing_provider=InMemoryTracingProvider,
        )
        out = await parent(prompt="x").collect()
        assert out.status.code == "success", out.error
        trace = await trace_of(out)
        session = next(s for s in trace.values() if s.parent_call_id is None).session
        assert session["__loaded_tools"] == {"parent.child": ["get_weather"]}
        parent_calls = [
            [t.name for t in s.input["tools"]]
            for s in sorted(trace.values(), key=lambda s: s.t0)
            if s.path == "parent.llm"
        ]
        assert parent_calls[-1] == ["child", SEARCH_TOOLS_NAME]

    async def test_approval_on_a_deferred_tool(self):
        from timbal.types.events import ApprovalEvent

        charged = []
        tools = [
            Tool(
                name="charge_card",
                description="Charge a card.",
                handler=lambda: charged.append(1) or "charged",
                requires_approval=True,
                defer_loading=True,
            )
        ]
        model = TestModel(responses=[call(SEARCH_TOOLS_NAME, query="charge card"), call("charge_card", "c2"), "Paid."])
        a = agent(model, tools)
        events = [event async for event in a(prompt="pay")]
        approval = next(e for e in events if isinstance(e, ApprovalEvent))
        parked = events[-1]
        assert parked.status.reason == "approval_required" and charged == []
        context(parent_id=parked.run_id)
        resumed = await a(prompt="pay", resume={approval.approval_id: True}).collect()
        assert resumed.status.code == "success", resumed.error
        assert charged == [1]

    async def test_loaded_tools_survive_a_restart(self, tmp_path):
        from timbal.state.tracing.providers import JsonlTracingProvider

        path = tmp_path / "traces.jsonl"

        def fresh(model):
            return Agent(
                name="agent",
                model=model,
                tools=catalog(),
                tracing_provider=JsonlTracingProvider.configured(_path=path),
            )

        first = await fresh(TestModel(responses=[call(SEARCH_TOOLS_NAME, query="repos"), "ok"]))(prompt="x").collect()
        set_run_context(
            RunContext(parent_id=first.run_id, tracing_provider=JsonlTracingProvider.configured(_path=path))
        )
        second = await fresh(TestModel(responses=["second"]))(prompt="y").collect()
        assert second.status.code == "success", second.error
        provider = JsonlTracingProvider.configured(_path=path)
        trace = await RunContext(id=second.run_id, tracing_provider=provider)._get_parent_trace(second.run_id)
        llm = next(s for s in trace.values() if s.path == "agent.llm")
        names = [t["name"] if isinstance(t, dict) else t.name for t in llm.input["tools"]]
        assert names == ["lookup", "list_repos", SEARCH_TOOLS_NAME]


# -- ChatSession: turns, steering mid-search, read-only wakes ------------------------


async def session_records(session, kind, count, timeout=3.0):
    found = []

    async def collect():
        async for item in session.events():
            if isinstance(item, dict) and item.get("kind") == kind:
                found.append(item)
                if len(found) >= count:
                    return

    await asyncio.wait_for(collect(), timeout)
    return found


def session_llm_tools(session):
    """Tool names offered on each LLM call the session's turns made, in order."""
    from timbal.types.events import StartEvent

    events, _ = session.log.peek()
    return [
        [t["name"] if isinstance(t, dict) else t.name for t in (e.input or {}).get("tools") or []]
        for e in events
        if isinstance(e, StartEvent) and e.path.endswith(".llm")
    ] or None


class TestChatSession:
    async def test_loads_persist_across_session_turns(self):
        from timbal.core.chat_session import ChatSession

        offered = []

        def handler(messages):
            last = messages[-1]
            if last.role == "tool":
                return "done"
            if last.collect_text() == "find weather":
                return call(SEARCH_TOOLS_NAME, f"c{len(messages)}", query="weather")
            return "plain"

        a = agent(TestModel(handler=handler))
        original = a._plan_tool_search

        async def spy(tools, model):
            planned = await original(tools, model)
            offered.append([t.name for t in planned[0]])
            return planned

        a._plan_tool_search = spy
        async with ChatSession(a) as session:
            session.send("find weather")
            (first,) = await session_records(session, "turn_finished", 1)
            session.send("hello")
            await session_records(session, "turn_finished", 2)
        assert first["tool_search"]["loaded"] == ["get_weather"]
        assert offered[0] == ["lookup", SEARCH_TOOLS_NAME]
        assert offered[-1] == ["lookup", "get_weather", SEARCH_TOOLS_NAME]

    async def test_steer_during_a_search_batch_sees_the_loaded_tool(self):
        from timbal.core.chat_session import ChatSession

        entered, release = asyncio.Event(), asyncio.Event()
        seen = []

        async def work() -> str:
            entered.set()
            await release.wait()
            return "worked"

        def handler(messages):
            seen.append(messages)
            if messages[-1].role == "user" and messages[-1].collect_text() == "start":
                return Message(
                    role="assistant",
                    content=[
                        ToolUseContent(id="s1", name=SEARCH_TOOLS_NAME, input={"query": "weather"}),
                        ToolUseContent(id="w1", name="work", input={}),
                    ],
                    stop_reason="tool_use",
                )
            return "done"

        a = agent(TestModel(handler=handler), [*catalog(), Tool(name="work", handler=work)], steering=True)
        offered = []
        original = a._plan_tool_search

        async def spy(tools, model):
            planned = await original(tools, model)
            offered.append([t.name for t in planned[0]])
            return planned

        a._plan_tool_search = spy
        async with ChatSession(a) as session:
            session.send("start")
            await asyncio.wait_for(entered.wait(), 2)
            session.send("use it for Paris", mode="steer")
            release.set()
            await session_records(session, "turn_finished", 1)
        assert "get_weather" in offered[1]
        assert any(m.collect_text() == "use it for Paris" for m in seen[1] if m.role == "user")

    async def test_read_only_wake_only_finds_read_only_tools(self):
        from timbal.core.chat_session import ChatSession

        sent = []

        def read_inbox() -> str:
            return "1 unread"

        def send_email(to: str) -> str:
            sent.append(to)
            return "sent"

        tools = [
            Tool(
                name="read_inbox",
                description="Read the email inbox.",
                handler=read_inbox,
                read_only=True,
                defer_loading=True,
            ),
            Tool(name="send_email", description="Send an email.", handler=send_email, defer_loading=True),
        ]
        searches = []

        def handler(messages):
            last = messages[-1]
            if last.role == "tool":
                searches.append(last.content[0].content[0].text if last.content else "")
                return "NO_REPLY"
            return call(SEARCH_TOOLS_NAME, f"c{len(messages)}", query="email inbox send")

        a = agent(TestModel(handler=handler), tools)
        offered = []
        original = a._plan_tool_search

        async def spy(tools, model):
            planned = await original(tools, model)
            offered.append(planned)
            return planned

        a._plan_tool_search = spy
        async with ChatSession(a) as session:
            session.wake("Check email.")
            (finished,) = await session_records(session, "turn_finished", 1)
        assert finished["silent"] is True
        first_offered, first_dispatch, _, unloaded = offered[0]
        assert unloaded == {"read_inbox"}
        assert "send_email" not in first_offered[-1].description
        assert "send_email" not in {t.name for t in first_dispatch}
        assert finished["tool_search"]["loaded"] == ["read_inbox"]
        assert sent == []


# -- Native OpenAI (hosted tool_search, Responses API) -----------------------------


OPENAI_SEARCH_CALL = {
    "id": "tsc_1", "status": "completed", "type": "tool_search_call",
    "arguments": {"paths": ["get_weather"]}, "execution": "server",
}
OPENAI_SEARCH_OUTPUT = {
    "id": "tso_1", "status": "completed", "type": "tool_search_output", "execution": "server",
    "tools": [{"type": "function", "name": "get_weather", "defer_loading": True, "parameters": {"type": "object"}}],
}


class TestNativeOpenAI:
    async def test_chat_completions_api_falls_back_to_client(self, monkeypatch):
        from timbal.core.llm import registry

        monkeypatch.setattr(registry, "TIMBAL_OPENAI_API", "chat_completions")
        context()
        a = agent("openai/gpt-6-luna")
        tools, _ = await a._resolve_tools(0)
        offered, _, deferred, _ = await a._plan_tool_search(tools, a.model)
        assert deferred is None and offered[-1].name == SEARCH_TOOLS_NAME

    async def test_request_marks_deferred_and_adds_tool_search(self):
        from timbal.core.llm.responses import prepare_responses_request

        captured = {}

        async def nothing():
            return
            yield

        class Responses:
            async def create(self, **request):
                captured.update(request)
                return nothing()

        create_stream, _ = prepare_responses_request(
            provider="openai",
            client=SimpleNamespace(responses=Responses()),
            model_name="gpt-6-luna",
            request_headers={},
            system_prompt=None,
            messages=[Message.validate({"role": "user", "content": "hi"})],
            tools=catalog(),
            max_tokens=100,
            temperature=None,
            output_model=None,
            provider_params={},
            deferred_tools=["get_weather"],
        )
        async for _ in create_stream():
            pass
        by_name = {t.get("name", t["type"]): t for t in captured["tools"]}
        assert by_name["get_weather"]["defer_loading"] is True
        assert "defer_loading" not in by_name["lookup"] and "defer_loading" not in by_name["create_invoice"]
        assert captured["tools"][-1] == {"type": "tool_search"}

    def test_replayed_as_top_level_items_and_skipped_elsewhere(self):
        from timbal.types.content import CustomContent, TextContent

        message = Message(
            role="assistant",
            content=[
                CustomContent(value=OPENAI_SEARCH_CALL),
                CustomContent(value=OPENAI_SEARCH_OUTPUT),
                ToolUseContent(id="call_1", name="get_weather", input={"city": "Oslo"}),
                TextContent(text="Checking."),
            ],
        )
        items = message.to_openai_responses_input()
        assert [item.get("type") for item in items[:3]] == ["tool_search_call", "tool_search_output", "function_call"]
        assert items[0] == OPENAI_SEARCH_CALL
        anthropic = message.to_anthropic_input()
        assert all(block.get("type") not in CustomContent.OPENAI_ITEMS for block in anthropic["content"])
        chat = message.to_openai_chat_completions_input()
        assert "tool_search" not in json.dumps(chat)

    def test_native_finds_read_functions_and_namespaces(self):
        from timbal.core.agent import _native_search_finds
        from timbal.types.content import CustomContent

        namespaced = {
            "type": "tool_search_output",
            "tools": [
                {"type": "namespace", "name": "crm", "tools": [{"type": "function", "name": "list_orders"}]},
                {"type": "function", "name": "get_weather"},
            ],
        }
        message = Message(role="assistant", content=[CustomContent(value=namespaced)])
        assert _native_search_finds(message) == ["list_orders", "get_weather"]
