"""tool_search="dispatch": a fixed tool list (cache-friendly), schemas in search results, call_tool runs them."""

import json

import pytest
from timbal import Tool
from timbal.core.agent import (
    _CALL_TOOL_FALLBACKS,
    _SEARCH_TOOLS_FALLBACKS,
    CALL_TOOL_NAME,
    SEARCH_TOOLS_NAME,
    _unused_name,
    is_call_tool_name,
)
from timbal.core.test_model import TestModel
from timbal.core.tool_set import ToolSet
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

    @pytest.mark.parametrize("arguments", [[], [1], "[]", "not json", 0, "", False, 3])
    async def test_arguments_that_are_not_an_object_are_an_error_not_an_empty_object(self, arguments):
        # A tool whose every parameter has a default: an empty object runs it. A list, a number or a
        # string that is not an object is a malformed call, and must not be run as one with no arguments.
        ran = []

        def search_orders(limit: int = 100) -> str:
            ran.append(limit)
            return f"{limit} orders"

        tools = [Tool(name="search_orders", handler=search_orders, defer_loading=True)]
        model = TestModel(responses=[dispatch("search_orders", arguments=arguments), "Oops."])
        out = await agent(model, tools, tool_search="dispatch")(prompt="x").collect()
        assert out.status.code == "success", out.error
        assert ran == []
        trace = await trace_of(out)
        span = next(s for s in trace.values() if s.path == f"agent.{CALL_TOOL_NAME}")
        assert span.status.code == "error"

    @pytest.mark.parametrize(
        "call_input",
        [
            {"name": "search_orders"},
            {"name": "search_orders", "arguments": None},
            {"name": "search_orders", "arguments": "null"},  # null, from a model that sends arguments as JSON text
            {"name": "search_orders", "arguments": "{}"},
        ],
    )
    async def test_missing_or_null_arguments_are_an_empty_object(self, call_input):
        ran = []

        def search_orders(limit: int = 100) -> str:
            ran.append(limit)
            return f"{limit} orders"

        tools = [Tool(name="search_orders", handler=search_orders, defer_loading=True)]
        asked = Message(
            role="assistant",
            content=[ToolUseContent(id="c1", name=CALL_TOOL_NAME, input=call_input)],
            stop_reason="tool_use",
        )
        out = await agent(TestModel(responses=[asked, "ok"]), tools, tool_search="dispatch")(prompt="x").collect()
        assert out.status.code == "success", out.error
        assert ran == [100]

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

    async def test_a_catalog_that_takes_every_dispatcher_name_still_gets_one(self):
        ran = []

        def lookup(q: str = "x") -> str:
            ran.append(q)
            return "found"

        mine = [
            Tool(name=name, handler=lambda: "mine") for name in ("call_tool", "call_found_tool", "call_deferred_tool")
        ]
        tools = [*mine, Tool(name="lookup", handler=lookup, defer_loading=True)]
        out = await agent(TestModel(responses=["hola"]), tools, tool_search="dispatch")(prompt="hola").collect()
        assert out.status.code == "success", out.error
        offered = names((await tool_lists(out))[0])
        [generated] = [name for name in offered if name not in {tool.name for tool in tools} | {SEARCH_TOOLS_NAME}]
        assert generated not in {tool.name for tool in mine}
        # The generated name dispatches like the usual one.
        asked = Message(
            role="assistant",
            content=[ToolUseContent(id="c1", name=generated, input={"name": "lookup", "arguments": {"q": "y"}})],
            stop_reason="tool_use",
        )
        context()  # a run of its own, not a turn after the greeting
        out = await agent(TestModel(responses=[asked, "ok"]), tools, tool_search="dispatch")(prompt="x").collect()
        assert out.status.code == "success", out.error
        assert ran == ["y"]
        # Under the tool's own name in the trace, as with the usual dispatcher name.
        assert any(span.path == "agent.lookup" for span in (await trace_of(out)).values())

    @pytest.mark.parametrize("mode", ["dispatch", "client"])
    async def test_a_catalog_that_takes_every_search_name_still_gets_one(self, mode):
        mine = [
            Tool(name=name, handler=lambda: "mine")
            for name in ("search_tools", "find_tools", "load_tools", "search_deferred_tools")
        ]
        tools = [*mine, Tool(name="lookup", handler=lambda: "found", defer_loading=True)]
        out = await agent(TestModel(responses=["hola"]), tools, tool_search=mode)(prompt="hola").collect()
        assert out.status.code == "success", out.error
        offered = names((await tool_lists(out))[0])
        assert len(offered) == len(set(offered)) and len(offered) == len(mine) + (2 if mode == "dispatch" else 1)

    @pytest.mark.parametrize("mode", ["dispatch", "client"])
    async def test_the_added_tools_keep_their_names_under_a_tool_policy(self, mode):
        """A policy hides tools from a run, not from the conversation: their names stay taken. A dispatcher
        named after a hidden tool would make one name two tools in one history, and a call of it kept
        for a later run (a resume) would reach the agent's own tool."""
        mine = [Tool(name="call_tool", handler=lambda: "mine"), Tool(name="search_tools", handler=lambda: "mine")]
        found = Tool(name="lookup", handler=lambda: "found", defer_loading=True, read_only=True)
        a = agent(TestModel(responses=["ok"]), [*mine, found], tool_search=mode)
        ctx = context()

        async def offered():
            tools, _, every_name, _ = await a._resolve_step_tools(0)
            return [tool.name for tool in (await a._plan_tool_search(tools, a.model, every_name))[0]]

        usual = await offered()
        assert usual[:2] == ["call_tool", "search_tools"] and len(set(usual)) == len(usual)
        ctx._tool_policy = "read_only"
        assert await offered() == usual[2:]  # the agent's own two are hidden, the added ones named as before

    @pytest.mark.parametrize("mode", ["dispatch", "client"])
    async def test_a_tool_policy_does_not_resolve_the_catalog_a_second_time(self, mode):
        """Keeping the hidden names taken must not cost a resolution: a ToolSet is asked once per model
        call, and what the model was offered in a step is what that step can run."""
        ran = []

        class Advancing(ToolSet):
            """A catalog that moves on every time it is asked: lookup_1, then lookup_2, ..."""

            asked: int = 0

            async def resolve(self):
                self.asked += 1
                number = self.asked

                def lookup() -> str:
                    ran.append(number)
                    return f"found {number}"

                return [Tool(name=f"lookup_{number}", handler=lookup, defer_loading=True, read_only=True)]

        catalog_ = Advancing()
        write = Tool(name="write", handler=lambda: "written")
        direct = (
            mode == "client"
        )  # a deferred tool called by its own name runs in both; dispatch goes through call_tool
        first = call("lookup_1") if direct else dispatch("lookup_1", arguments={})
        second = call("lookup_2", "c2") if direct else dispatch("lookup_2", "c2", arguments={})
        model = TestModel(responses=[first, second, "done"])
        a = agent(model, [write, catalog_], tool_search=mode)
        ctx = context()
        ctx._tool_policy = "read_only"
        out = await a(prompt="go").collect()
        assert out.status.code == "success", out.error
        assert model.call_count == 3 and catalog_.asked == 3
        assert ran == [1, 2]

    def test_a_generated_name_is_the_lowest_free_number(self):
        every = {CALL_TOOL_NAME, *_CALL_TOOL_FALLBACKS}
        assert _unused_name(CALL_TOOL_NAME, _CALL_TOOL_FALLBACKS, set()) == CALL_TOOL_NAME
        assert _unused_name(CALL_TOOL_NAME, _CALL_TOOL_FALLBACKS, {CALL_TOOL_NAME}) == _CALL_TOOL_FALLBACKS[0]
        assert _unused_name(CALL_TOOL_NAME, _CALL_TOOL_FALLBACKS, every) == "call_tool_2"
        assert (
            _unused_name(CALL_TOOL_NAME, _CALL_TOOL_FALLBACKS, every | {"call_tool_2", "call_tool_3"}) == "call_tool_4"
        )
        taken = {SEARCH_TOOLS_NAME, *_SEARCH_TOOLS_FALLBACKS}
        assert _unused_name(SEARCH_TOOLS_NAME, _SEARCH_TOOLS_FALLBACKS, taken) == "search_tools_2"

    def test_every_name_the_dispatcher_can_get_is_recognized(self):
        """What reads a stored call by name alone (a recovery notice) knows every name the picker can give."""
        taken: set[str] = set()
        for _ in range(8):
            name = _unused_name(CALL_TOOL_NAME, _CALL_TOOL_FALLBACKS, taken)
            assert is_call_tool_name(name), name
            taken.add(name)
        # Only those: not a number the picker never gives, in a spelling it never uses.
        others = ("lookup", "call_tool_now", "call_tool_", "call_tools", "search_tools", None, 3)
        never_picked = (
            "call_tool_0",
            "call_tool_1",
            "call_tool_02",
            "call_tool_\u00b2",
            "call_tool_\u0663",
            "call_tool_2 ",
        )
        for other in (*others, *never_picked):
            assert not is_call_tool_name(other), other

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
