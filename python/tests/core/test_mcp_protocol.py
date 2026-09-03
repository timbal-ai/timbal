"""MCP client protocol features: elicitation -> suspend(), annotations -> approval gate,
tools/list_changed staleness, server logging, and sampling.

All tests drive a real FastMCP server over stdio so the server -> client request
path (elicitation, sampling) and notification path (logging, list_changed) are the
real SDK code, not mocks.
"""

import asyncio
import sys
from contextlib import asynccontextmanager

import anyio
import pytest
import structlog
from timbal import Agent
from timbal.core.mcp import ELICITATION_KIND, MCPServer, MCPTool, _coerce_elicit_resume, _is_destructive
from timbal.core.test_model import TestModel
from timbal.state import RunContext, set_run_context
from timbal.types.approval import Cancel
from timbal.types.content import ToolUseContent
from timbal.types.events import ApprovalEvent, InteractionEvent, OutputEvent
from timbal.types.message import Message

try:
    from mcp import types as mcp_types
except ImportError:  # pragma: no cover
    mcp_types = None

pytestmark = pytest.mark.skipif(mcp_types is None, reason="mcp package not installed")


SERVER_SCRIPT = '''
from mcp.server.fastmcp import Context, FastMCP
from mcp.types import SamplingMessage, TextContent, ToolAnnotations
from pydantic import BaseModel

mcp = FastMCP("proto-server")


class Confirm(BaseModel):
    confirm: bool
    note: str = ""


@mcp.tool(annotations=ToolAnnotations(destructiveHint=True))
async def swap_kb(kb_id: int, ctx: Context) -> str:
    """Swap the linked KB (destructive)."""
    result = await ctx.elicit(message=f"Swap KB to {kb_id}?", schema=Confirm)
    if result.action == "accept":
        return f"swapped:{kb_id}:{result.data.confirm}:{result.data.note}"
    if result.action == "decline":
        return "declined"
    raise ValueError("confirmation cancelled")


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def read_kb() -> str:
    """Read the KB (read-only)."""
    return "kb"


@mcp.tool()
def plain() -> str:
    """No annotations at all."""
    return "plain"


@mcp.tool(title="Delete Thing", annotations=ToolAnnotations(destructiveHint=True))
def delete_thing() -> str:
    """Delete something."""
    return "deleted"


@mcp.tool()
async def log_hello(ctx: Context) -> str:
    """Emit a server-side log line."""
    await ctx.info("hello from server")
    return "logged"


@mcp.tool()
async def add_tool(ctx: Context) -> str:
    """Register a new tool and notify tools/list_changed."""

    def extra() -> str:
        """Extra tool."""
        return "extra"

    mcp.add_tool(extra)
    await ctx.session.send_tool_list_changed()
    return "added"


@mcp.tool()
async def ask_llm(question: str, ctx: Context) -> str:
    """Sample from the client's LLM."""
    result = await ctx.session.create_message(
        messages=[SamplingMessage(role="user", content=TextContent(type="text", text=question))],
        max_tokens=50,
    )
    return result.content.text


mcp.run()
'''


@pytest.fixture
def server_script(tmp_path):
    script = tmp_path / "server.py"
    script.write_text(SERVER_SCRIPT)
    return str(script)


@pytest.fixture
def make_server(server_script):
    servers: list[MCPServer] = []

    def _make(**kwargs) -> MCPServer:
        server = MCPServer(transport="stdio", command=sys.executable, args=[server_script], **kwargs)
        servers.append(server)
        return server

    yield _make


def _tool_call(*specs: tuple[str, str, dict]) -> Message:
    return Message(
        role="assistant",
        content=[ToolUseContent(id=cid, name=name, input=inp) for cid, name, inp in specs],
        stop_reason="tool_use",
    )


# --- pure helpers ----------------------------------------------------------------


class TestIsDestructive:
    def _tool(self, **annotations):
        return mcp_types.Tool(
            name="t",
            inputSchema={"type": "object", "properties": {}},
            annotations=mcp_types.ToolAnnotations(**annotations) if annotations else None,
        )

    def test_unannotated_is_not_destructive(self):
        assert _is_destructive(self._tool()) is False

    def test_read_only_wins(self):
        assert _is_destructive(self._tool(readOnlyHint=True, destructiveHint=True)) is False

    def test_destructive_hint_defaults_true_when_annotated(self):
        assert _is_destructive(self._tool(openWorldHint=True)) is True
        assert _is_destructive(self._tool(destructiveHint=False)) is False
        assert _is_destructive(self._tool(destructiveHint=True)) is True


class TestCoerceElicitResume:
    SCHEMA = {
        "type": "object",
        "properties": {"confirm": {"type": "boolean"}, "size": {"type": "string", "enum": ["s", "m"]}},
        "required": ["confirm"],
    }

    def test_none_and_false_decline(self):
        assert _coerce_elicit_resume(None, self.SCHEMA)[0].action == "decline"
        assert _coerce_elicit_resume(False, self.SCHEMA)[0].action == "decline"

    def test_true_accepts_with_empty_content_when_nothing_required(self):
        result, error = _coerce_elicit_resume(True, {"type": "object", "properties": {}})
        assert error is None
        assert result.action == "accept"
        assert result.content == {}

    def test_true_with_required_fields_is_an_error(self):
        result, error = _coerce_elicit_resume(True, self.SCHEMA)
        assert result is None
        assert "confirm" in error

    def test_plain_dict_is_content(self):
        result, error = _coerce_elicit_resume({"confirm": True, "size": "m"}, self.SCHEMA)
        assert error is None
        assert result.action == "accept"
        assert result.content == {"confirm": True, "size": "m"}

    def test_explicit_action_envelope(self):
        assert _coerce_elicit_resume({"action": "decline"}, self.SCHEMA)[0].action == "decline"
        assert _coerce_elicit_resume({"action": "cancel"}, self.SCHEMA)[0].action == "cancel"
        result, _ = _coerce_elicit_resume({"action": "accept", "content": {"confirm": False}}, self.SCHEMA)
        assert result.action == "accept"
        assert result.content == {"confirm": False}

    def test_validation_errors(self):
        assert "type boolean" in _coerce_elicit_resume({"confirm": "yes"}, self.SCHEMA)[1]
        assert "one of" in _coerce_elicit_resume({"confirm": True, "size": "xl"}, self.SCHEMA)[1]
        assert "unsupported" in _coerce_elicit_resume("yes", self.SCHEMA)[1]

    def test_url_mode_accept_sends_no_content(self):
        result, error = _coerce_elicit_resume(True, None)
        assert error is None
        assert result.action == "accept"
        assert result.content is None


# --- annotations -> approval gate ------------------------------------------------


class TestApprovalFromAnnotations:
    async def test_destructive_policy_gates_only_destructive_tools(self, make_server):
        server = make_server(approval="destructive")
        try:
            tools = {t.name: t for t in await server.resolve()}
            assert tools["swap_kb"].requires_approval is True
            assert tools["delete_thing"].requires_approval is True
            assert tools["read_kb"].requires_approval is False
            assert tools["plain"].requires_approval is False

            delete = tools["delete_thing"]
            assert isinstance(delete, MCPTool)
            assert delete.title == "Delete Thing"
            assert delete.tool_annotations == {"destructiveHint": True}
            assert delete.approval_kind == "mcp_tool"
            assert delete.approval_ui["annotations"] == {"destructiveHint": True}
            assert delete.approval_ui["tool"] == "delete_thing"
        finally:
            await server.close()

    async def test_all_and_callable_policies(self, make_server):
        everything = make_server(approval="all")
        only_plain = make_server(approval=lambda t: t.name == "plain")
        try:
            assert all(t.requires_approval for t in await everything.resolve())
            gated = {t.name for t in await only_plain.resolve() if t.requires_approval}
            assert gated == {"plain"}
        finally:
            await everything.close()
            await only_plain.close()

    async def test_default_policy_gates_nothing(self, make_server):
        server = make_server()
        try:
            assert not any(t.requires_approval for t in await server.resolve())
        finally:
            await server.close()

    async def test_agent_pauses_for_approval_then_runs_tool(self, make_server):
        server = make_server(approval="destructive")
        model = TestModel(responses=[_tool_call(("c1", "delete_thing", {})), "done"])
        agent = Agent(name="a", model=model, tools=[server], max_iter=3)
        try:
            approvals: list[ApprovalEvent] = []
            final = None
            async for event in agent(prompt="delete it"):
                if isinstance(event, ApprovalEvent):
                    approvals.append(event)
                if isinstance(event, OutputEvent) and event.path == "a":
                    final = event

            assert len(approvals) == 1
            assert approvals[0].kind == "mcp_tool"
            assert approvals[0].runnable_name == "delete_thing"
            assert approvals[0].input_schema["type"] == "object"
            assert final.status.reason == "approval_required"

            events = [
                e
                async for e in agent(
                    prompt="delete it", parent_id=final.run_id, resume={approvals[0].approval_id: True}
                )
            ]
            tool_out = [e for e in events if isinstance(e, OutputEvent) and e.path == "a.delete_thing"]
            assert tool_out[0].status.code == "success"
            assert tool_out[0].output == "deleted"
            assert events[-1].status.code == "success"
        finally:
            await server.close()


# --- elicitation -> suspend() ----------------------------------------------------


class TestElicitation:
    async def test_direct_tool_suspends_and_resumes(self, make_server):
        server = make_server()
        try:
            tools = {t.name: t for t in await server.resolve()}
            swap = tools["swap_kb"]

            events = [e async for e in swap(kb_id=5)]
            interactions = [e for e in events if isinstance(e, InteractionEvent)]
            first = events[-1]
            assert isinstance(first, OutputEvent)
            assert first.status.code == "cancelled"
            assert first.status.reason == "input_required"
            assert len(interactions) == 1
            ev = interactions[0]
            assert ev.kind == ELICITATION_KIND
            assert ev.payload["message"] == "Swap KB to 5?"
            assert ev.payload["tool"] == "swap_kb"
            assert ev.payload["mode"] == "form"
            assert "confirm" in ev.payload["requested_schema"]["properties"]
            assert ev.response_schema == ev.payload["requested_schema"]

            resumed = await swap(
                kb_id=5, parent_id=first.run_id, resume={ev.interaction_id: {"confirm": True, "note": "ok"}}
            ).collect()
            assert resumed.status.code == "success"
            assert resumed.output == "swapped:5:True:ok"
        finally:
            await server.close()

    async def test_suspension_id_is_stable_across_reexecution(self, make_server):
        server = make_server()
        try:
            swap = {t.name: t for t in await server.resolve()}["swap_kb"]
            a = await swap(kb_id=7).collect()
            b = await swap(kb_id=7).collect()
            assert a.output["suspension_id"] == b.output["suspension_id"]
            c = await swap(kb_id=8).collect()
            # Different input -> different server message -> different id.
            assert c.output["suspension_id"] != a.output["suspension_id"]
        finally:
            await server.close()

    async def _run_agent_until_pause(self, agent, prompt):
        pending: list[InteractionEvent] = []
        final = None
        async for event in agent(prompt=prompt):
            if isinstance(event, InteractionEvent):
                pending.append(event)
            if isinstance(event, OutputEvent) and event.path == agent.name:
                final = event
        return pending, final

    async def test_agent_pauses_then_accepts(self, make_server):
        server = make_server()
        model = TestModel(responses=[_tool_call(("c1", "swap_kb", {"kb_id": 5})), "done"])
        agent = Agent(name="a", model=model, tools=[server], max_iter=3)
        try:
            pending, final = await self._run_agent_until_pause(agent, "swap")
            assert final.status.reason == "input_required"
            assert len(pending) == 1
            assert pending[0].kind == ELICITATION_KIND
            assert pending[0].tool_call_id == "c1"

            events = [
                e
                async for e in agent(
                    prompt="swap", parent_id=final.run_id, resume={pending[0].interaction_id: {"confirm": True}}
                )
            ]
            tool_out = [e for e in events if isinstance(e, OutputEvent) and e.path == "a.swap_kb"]
            assert tool_out[0].status.code == "success"
            assert tool_out[0].output == "swapped:5:True:"
            assert events[-1].status.code == "success"
            assert events[-1].output.collect_text() == "done"
        finally:
            await server.close()

    async def test_agent_decline_reaches_server(self, make_server):
        server = make_server()
        model = TestModel(responses=[_tool_call(("c1", "swap_kb", {"kb_id": 5})), "done"])
        agent = Agent(name="a", model=model, tools=[server], max_iter=3)
        try:
            pending, final = await self._run_agent_until_pause(agent, "swap")
            events = [
                e async for e in agent(prompt="swap", parent_id=final.run_id, resume={pending[0].interaction_id: False})
            ]
            tool_out = [e for e in events if isinstance(e, OutputEvent) and e.path == "a.swap_kb"]
            assert tool_out[0].output == "declined"
        finally:
            await server.close()

    async def test_agent_cancel_stops_run(self, make_server):
        server = make_server()
        model = TestModel(responses=[_tool_call(("c1", "swap_kb", {"kb_id": 5})), "done"])
        agent = Agent(name="a", model=model, tools=[server], max_iter=3)
        try:
            pending, final = await self._run_agent_until_pause(agent, "swap")
            result = await agent(
                prompt="swap", parent_id=final.run_id, resume={pending[0].interaction_id: Cancel(reason="no")}
            ).collect()
            assert result.status.code == "cancelled"
            assert result.status.reason == "cancelled"
        finally:
            await server.close()

    async def test_invalid_resume_value_errors_the_tool(self, make_server):
        server = make_server()
        model = TestModel(responses=[_tool_call(("c1", "swap_kb", {"kb_id": 5})), "done"])
        agent = Agent(name="a", model=model, tools=[server], max_iter=3)
        try:
            pending, final = await self._run_agent_until_pause(agent, "swap")
            events = [
                e
                async for e in agent(
                    prompt="swap", parent_id=final.run_id, resume={pending[0].interaction_id: {"nope": 1}}
                )
            ]
            tool_out = [e for e in events if isinstance(e, OutputEvent) and e.path == "a.swap_kb"]
            assert tool_out[0].status.code == "error"
            assert "missing required field(s): confirm" in tool_out[0].error["message"]
        finally:
            await server.close()

    async def test_parallel_elicitations_cannot_be_attributed(self, make_server):
        """Two elicit-capable calls in flight on one session -> both get a retry hint, nothing suspends."""
        server = make_server()
        model = TestModel(
            responses=[_tool_call(("c1", "swap_kb", {"kb_id": 1}), ("c2", "swap_kb", {"kb_id": 2})), "done"]
        )
        agent = Agent(name="a", model=model, tools=[server], max_iter=3)
        try:
            events = [e async for e in agent(prompt="swap both")]
            assert not any(isinstance(e, InteractionEvent) for e in events)
            tool_out = [e for e in events if isinstance(e, OutputEvent) and e.path == "a.swap_kb"]
            assert len(tool_out) == 2
            for out in tool_out:
                assert out.status.code == "error"
                assert "Call this tool on its own" in out.error["message"]
            assert events[-1].status.code == "success"
        finally:
            await server.close()

    async def test_elicitation_disabled_hides_capability(self, make_server):
        server = make_server(elicitation=False)
        try:
            swap = {t.name: t for t in await server.resolve()}["swap_kb"]
            result = await swap(kb_id=5).collect()
            # The server cannot elicit; the tool fails instead of silently proceeding.
            assert result.status.code == "error"
        finally:
            await server.close()


# --- tools/list_changed ----------------------------------------------------------


class TestToolListChanged:
    async def test_stale_list_refreshes_only_on_next_run(self, make_server):
        server = make_server()
        try:
            run_a = RunContext()
            set_run_context(run_a)
            tools_a = await server.resolve()
            names_a = {t.name for t in tools_a}
            assert "extra" not in names_a
            assert [t.name for t in tools_a] == sorted(names_a)

            add_tool = {t.name: t for t in tools_a}["add_tool"]
            result = await add_tool().collect()
            assert result.output == "added"
            await asyncio.sleep(0)
            assert server._tools_stale is True

            # Same run: the tool set must not change under the LLM's feet.
            set_run_context(run_a)
            assert await server.resolve() is tools_a

            # Next run: refreshed.
            set_run_context(RunContext())
            tools_b = await server.resolve()
            assert tools_b is not tools_a
            assert "extra" in {t.name for t in tools_b}
            assert server._tools_stale is False
        finally:
            set_run_context(None)
            await server.close()

    async def test_stale_list_refreshes_immediately_outside_a_run(self, make_server):
        server = make_server()
        try:
            set_run_context(None)
            tools_a = await server.resolve()
            await {t.name: t for t in tools_a}["add_tool"]().collect()
            await asyncio.sleep(0)
            set_run_context(None)
            tools_b = await server.resolve()
            assert "extra" in {t.name for t in tools_b}
        finally:
            set_run_context(None)
            await server.close()


# --- connection lifecycle --------------------------------------------------------


class TestConnectionLifecycle:
    async def test_two_servers_close_in_open_order(self, make_server):
        """Each session lives in its own task, so close order no longer matters."""
        a = make_server(name="a")
        b = make_server(name="b")
        assert await a.resolve()
        assert await b.resolve()
        await a.close()
        await b.close()
        assert a._session is None and a._session_task is None
        assert b._session is None and b._session_task is None

    async def test_close_from_another_task(self, make_server):
        server = make_server()
        assert await server.resolve()
        await asyncio.create_task(server.close())
        assert server._session is None

    async def test_reopen_after_close(self, make_server):
        server = make_server()
        try:
            first = await server.resolve()
            await server.close()
            second = await server.resolve()
            assert second is not first
            assert {t.name for t in second} == {t.name for t in first}
            greet = {t.name: t for t in second}["plain"]
            assert (await greet().collect()).output == "plain"
        finally:
            await server.close()

    async def test_connect_failure_propagates_and_leaves_no_session(self):
        server = MCPServer(transport="stdio", command=sys.executable, args=["-c", "import sys; sys.exit(3)"])
        with pytest.raises((RuntimeError, OSError, ExceptionGroup)):
            await server.resolve()
        assert server._session is None
        assert server._session_task is None

    async def test_dead_transport_resets_session_and_reconnects(self):
        """If the owner task dies (transport error), the next call reopens instead of failing forever."""
        server = MCPServer(transport="http", url="https://example.com/mcp")
        die = asyncio.Event()
        opens = 0

        class FakeSession:
            async def list_tools(self):
                return mcp_types.ListToolsResult(tools=[])

        @asynccontextmanager
        async def flaky_connect():
            nonlocal opens
            opens += 1
            first = opens == 1
            async with anyio.create_task_group() as tg:
                if first:

                    async def killer():
                        await die.wait()
                        raise RuntimeError("transport died")

                    tg.start_soon(killer)
                yield FakeSession()

        server._connect_http = flaky_connect  # type: ignore[method-assign]

        s1 = await server._connect()
        task1 = server._session_task
        die.set()
        await asyncio.wait({task1})
        await asyncio.sleep(0)  # let the done-callback run
        assert server._session is None

        s2 = await server._connect()
        assert s2 is not s1
        assert opens == 2
        await server.close()


# --- logging & sampling ----------------------------------------------------------


class TestLoggingAndSampling:
    async def test_server_logs_land_in_structlog(self, make_server):
        server = make_server(name="srv")
        try:
            log_hello = {t.name: t for t in await server.resolve()}["srv__log_hello"]
            with structlog.testing.capture_logs() as logs:
                result = await log_hello().collect()
                await asyncio.sleep(0)
            assert result.output == "logged"
            entries = [entry for entry in logs if entry.get("event") == "MCP server log"]
            assert entries, logs
            assert entries[0]["data"] == "hello from server"
            assert entries[0]["server"] == "srv"
            assert entries[0]["log_level"] == "info"
        finally:
            await server.close()

    async def test_sampling_routes_to_configured_model(self, make_server):
        server = make_server(sampling_model=TestModel(responses=["forty-two"]))
        try:
            ask = {t.name: t for t in await server.resolve()}["ask_llm"]
            result = await ask(question="6*7?").collect()
            assert result.status.code == "success"
            assert result.output == "forty-two"
        finally:
            await server.close()

    async def test_sampling_unsupported_without_model(self, make_server):
        server = make_server()
        try:
            ask = {t.name: t for t in await server.resolve()}["ask_llm"]
            result = await ask(question="6*7?").collect()
            assert result.status.code == "error"
        finally:
            await server.close()
