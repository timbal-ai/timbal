"""tool_search="dispatch": a fixed tool list (cache-friendly), schemas in search results, call_tool runs them."""

import json

import pytest
from timbal import Tool
from timbal.core.agent import CALL_TOOL_NAME, SEARCH_TOOLS_NAME
from timbal.core.test_model import TestModel
from timbal.types.content import ToolUseContent
from timbal.types.message import Message

from .test_tool_search import agent, call, catalog, context, isolated_context, trace_of  # noqa: F401


def dispatch(tool: str, call_id: str = "c1", *, arguments) -> Message:
    return Message(
        role="assistant",
        content=[ToolUseContent(id=call_id, name=CALL_TOOL_NAME, input={"name": tool, "arguments": arguments})],
        stop_reason="tool_use",
    )


async def tool_lists(output) -> list[str]:
    """The serialized tools of each LLM call of a run, in order: what the prompt cache sees."""
    trace = await trace_of(output)
    llm_spans = sorted((s for s in trace.values() if s.path.endswith(".llm")), key=lambda s: s.t0)
    return [
        json.dumps([t if isinstance(t, dict) else t.anthropic_schema for t in s.input.get("tools") or []])
        for s in llm_spans
    ]


def names(serialized: str) -> list[str]:
    return [t["name"] for t in json.loads(serialized)]


class TestDispatch:
    async def test_search_then_call_tool_runs_it_and_the_tools_never_change(self):
        calls = []
        model = TestModel(
            responses=[
                call(SEARCH_TOOLS_NAME, query="weather"),
                dispatch("get_weather", "c2", arguments={"city": "Paris"}),
                "Sunny.",
            ]
        )
        a = agent(model, catalog(calls), tool_search="dispatch")
        out = await a(prompt="weather in Paris?").collect()
        assert out.status.code == "success", out.error
        assert calls == [("get_weather", "Paris")]
        lists = await tool_lists(out)
        assert len(lists) == 3 and len(set(lists)) == 1
        assert names(lists[0]) == ["lookup", SEARCH_TOOLS_NAME, CALL_TOOL_NAME]

        trace = await trace_of(out)
        search = next(s for s in trace.values() if s.path == f"agent.{SEARCH_TOOLS_NAME}")
        found = json.loads(search.output)["found"]
        assert found[0]["name"] == "get_weather" and "city" in found[0]["input_schema"]["properties"]
        # The call runs as the tool itself: its own span, its result answering call_tool's id.
        weather = next(s for s in trace.values() if s.path == "agent.get_weather")
        assert weather.input == {"city": "Paris"}
        agent_span = next(s for s in trace.values() if s.path == "agent")
        assert agent_span.metadata["tool_search"]["mode"] == "dispatch"
        assert agent_span.metadata["tool_search"]["loaded"] == ["get_weather"]
        memory = agent_span.memory
        use = next(b for m in memory for b in m.content if b.type == "tool_use" and b.id == "c2")
        result = next(b for m in memory for b in m.content if b.type == "tool_result" and b.id == "c2")
        assert use.name == CALL_TOOL_NAME and "sunny in Paris" in str(result.content)

    async def test_the_next_turn_offers_the_same_tools(self):
        model = TestModel(
            responses=[
                call(SEARCH_TOOLS_NAME, query="weather"),
                dispatch("get_weather", "c2", arguments={"city": "Paris"}),
                "Sunny.",
            ]
        )
        a = agent(model, tool_search="dispatch")
        first = await a(prompt="weather?").collect()
        context(parent_id=first.run_id)
        a.model = TestModel(responses=["Still sunny."])
        second = await a(prompt="and now?").collect()
        assert second.status.code == "success", second.error
        assert (await tool_lists(second))[0] == (await tool_lists(first))[0]

    async def test_arguments_as_a_json_string(self):
        calls = []
        model = TestModel(responses=[dispatch("get_weather", arguments=json.dumps({"city": "Rome"})), "ok"])
        out = await agent(model, catalog(calls), tool_search="dispatch")(prompt="x").collect()
        assert out.status.code == "success", out.error
        assert calls == [("get_weather", "Rome")]

    async def test_an_unknown_name_is_an_error_result_and_the_turn_goes_on(self):
        model = TestModel(responses=[dispatch("send_rocket", arguments={}), "Can't do that."])
        out = await agent(model, tool_search="dispatch")(prompt="x").collect()
        assert out.status.code == "success", out.error
        trace = await trace_of(out)
        span = next(s for s in trace.values() if s.path == f"agent.{CALL_TOOL_NAME}")
        assert span.status.code == "error" and SEARCH_TOOLS_NAME in span.error["message"]

    async def test_bad_arguments_fail_like_a_direct_call(self):
        model = TestModel(responses=[dispatch("get_weather", arguments={"town": "Rome"}), "Oops."])
        out = await agent(model, tool_search="dispatch")(prompt="x").collect()
        assert out.status.code == "success", out.error
        trace = await trace_of(out)
        span = next(s for s in trace.values() if s.path == "agent.get_weather")
        assert span.status.code == "error"

    async def test_a_found_tool_called_by_its_own_name_runs_too(self):
        calls = []
        model = TestModel(responses=[call("get_weather", city="Oslo"), "ok"])
        out = await agent(model, catalog(calls), tool_search="dispatch")(prompt="x").collect()
        assert out.status.code == "success", out.error
        assert calls == [("get_weather", "Oslo")]

    async def test_approval_through_call_tool_survives_the_resume(self):
        from timbal.types.events import ApprovalEvent

        charged = []
        tools = [
            Tool(
                name="charge_card",
                description="Charge a card.",
                handler=lambda amount: charged.append(amount) or "charged",
                requires_approval=True,
                defer_loading=True,
            )
        ]
        model = TestModel(
            responses=[
                call(SEARCH_TOOLS_NAME, query="charge card"),
                dispatch("charge_card", "c2", arguments={"amount": 5}),
                "Paid.",
            ]
        )
        a = agent(model, tools, tool_search="dispatch")
        events = [event async for event in a(prompt="pay")]
        approval = next(e for e in events if isinstance(e, ApprovalEvent))
        assert approval.runnable_path == "agent.charge_card" and approval.tool_call_id == "c2"
        assert events[-1].status.reason == "approval_required" and charged == []
        context(parent_id=events[-1].run_id)
        resumed = await a(prompt="pay", resume={approval.approval_id: True}).collect()
        assert resumed.status.code == "success", resumed.error
        assert charged == [5]

    async def test_a_tool_named_call_tool_keeps_its_name(self):
        def call_tool(x: str) -> str:
            return x

        tools = [*catalog(), Tool(name="call_tool", handler=call_tool)]
        model = TestModel(responses=["ok"])
        out = await agent(model, tools, tool_search="dispatch")(prompt="x").collect()
        assert names((await tool_lists(out))[0]) == ["lookup", "call_tool", SEARCH_TOOLS_NAME, "call_found_tool"]

    @pytest.mark.parametrize("mode", ["client", "auto"])
    async def test_other_modes_offer_no_dispatcher(self, mode):
        model = TestModel(responses=["ok"])
        out = await agent(model, tool_search=mode)(prompt="x").collect()
        assert CALL_TOOL_NAME not in names((await tool_lists(out))[0])
