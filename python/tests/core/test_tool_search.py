"""Deferred tool schemas: client-side search_tools, native Anthropic tool search, persistence."""

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
            ("openai/gpt-5.5", False),
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
