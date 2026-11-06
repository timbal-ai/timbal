"""Call context and ``report_call_outcome`` on the agent side of a phone call (issue #200)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from timbal import Agent
from timbal.core.test_model import TestModel
from timbal.state import set_run_context
from timbal.types.content import ToolUseContent
from timbal.types.events import OutputEvent
from timbal.types.message import Message
from timbal.voice import call_context
from timbal.voice.call import CallControl, current_call_control, run_context_from_wire
from timbal.voice.session import _current_voice_session

from ..platform.voice_platform import FakeVoicePlatform

SCHEMA = {
    "type": "object",
    "properties": {
        "outcome": {"type": "string", "enum": ["confirmed", "declined", "callback_requested"]},
        "note": {"type": "string"},
    },
    "required": ["outcome"],
}


@pytest.fixture
def platform(monkeypatch):
    fake = FakeVoicePlatform()
    monkeypatch.setattr("timbal.voice.call._outcome_client", fake.outcome_client())
    return fake


@pytest.fixture(autouse=True)
def _clear_context():
    yield
    set_run_context(None)  # type: ignore[arg-type]


def _report(tool_id: str, **outcome) -> Message:
    return Message(
        role="assistant",
        content=[ToolUseContent(id=tool_id, name="report_call_outcome", input=outcome)],
        stop_reason="tool_use",
    )


def _wire(call_id: str, token: str, schema=SCHEMA) -> dict:
    return {
        "id": "run-1",
        "call_context": {
            "call_id": call_id,
            "direction": "outbound",
            "from": "+34930000000",
            "to": "+34600000001",
            "user_number": "+34600000001",
            "agent_number": "+34930000000",
            "customer_name": "Ada",
            "appointment": {"at": "2026-11-02T10:00"},
        },
        "call_control": {
            "url": f"https://api.timbal.ai/call-control/{call_id}",
            "token": token,
            "outcome_schema": schema,
        },
    }


class TestWireContext:
    def test_call_control_is_kept_off_the_context_fields(self):
        ctx = run_context_from_wire(_wire("call_1", "secret-token"))
        assert "call_control" not in ctx.model_dump()
        assert "secret-token" not in json.dumps(ctx.model_dump(), default=str)
        assert isinstance(ctx._call_control, CallControl)
        assert "secret-token" not in repr(ctx._call_control)

    def test_run_context_from_wire_with_run_context_and_non_dict(self):
        from pydantic import ValidationError

        ctx = run_context_from_wire(_wire("call_1", "secret-token"))
        same = run_context_from_wire(ctx)
        assert same is ctx
        assert same._call_control is ctx._call_control

        with pytest.raises(ValidationError):
            run_context_from_wire("not-a-mapping")

    def test_call_context_from_a_stream_turn(self):
        set_run_context(run_context_from_wire(_wire("call_1", "t")))
        ctx = call_context()
        assert ctx["user_number"] == "+34600000001"
        assert ctx["appointment"] == {"at": "2026-11-02T10:00"}

    def test_call_context_from_a_voice_session(self):
        token = _current_voice_session.set(SimpleNamespace(call_context={"user_number": "+34611"}, call_control=None))
        try:
            assert call_context() == {"user_number": "+34611"}
        finally:
            _current_voice_session.reset(token)

    def test_outside_a_call(self):
        assert call_context() == {}
        assert current_call_control() is None


class TestReportCallOutcome:
    @pytest.mark.asyncio
    async def test_agent_reports_corrects_and_never_traces_the_token(self, platform):
        call_id, token = platform.seed(SCHEMA)
        seen_prompts: list[str] = []

        def system_prompt() -> str:
            ctx = call_context()
            prompt = f"You call {ctx['customer_name']} at {ctx['user_number']} about {ctx['appointment']['at']}."
            seen_prompts.append(prompt)
            return prompt

        agent = Agent(
            name="caller",
            model=TestModel(
                responses=[
                    _report("t1", outcome="maybe"),
                    _report("t2", outcome="confirmed", note="Viene el lunes"),
                    "Perfecto, hasta el lunes.",
                ]
            ),
            system_prompt=system_prompt,
        )
        ctx = run_context_from_wire(_wire(call_id, token))
        set_run_context(ctx)
        events = [e async for e in agent(prompt="Sí, puedo ir")]
        final = events[-1]
        assert final.status.code == "success", final.error

        assert seen_prompts and seen_prompts[0] == "You call Ada at +34600000001 about 2026-11-02T10:00."
        reports = [e for e in events if isinstance(e, OutputEvent) and e.path.endswith("report_call_outcome")]
        assert "rejected" in reports[0].output and "/outcome" in reports[0].output
        assert reports[1].output == "Outcome recorded."
        assert platform.calls[call_id]["outcome"] == {"outcome": "confirmed", "note": "Viene el lunes"}

        dumped = json.dumps(ctx._trace.model_dump(), default=str)
        assert token not in dumped
        assert "report_call_outcome" in dumped

    def test_no_tool_without_an_outcome_schema(self):
        set_run_context(run_context_from_wire(_wire("call_1", "t", schema=None)))
        control = current_call_control()
        assert control is not None
        assert control.report_tool() is None

    @pytest.mark.asyncio
    async def test_after_final_and_with_a_bad_token(self, platform):
        call_id, token = platform.seed(SCHEMA)
        tool = CallControl(
            url=f"https://api.timbal.ai/call-control/{call_id}", token=token, outcome_schema=SCHEMA
        ).report_tool()
        platform.finish(call_id)
        out = await tool(outcome="confirmed").collect()
        assert "already final" in out.output

        other_id, _ = platform.seed(SCHEMA)
        forged = CallControl(url=f"https://api.timbal.ai/call-control/{other_id}", token="wrong", outcome_schema=SCHEMA)
        out = await forged.report_tool()(outcome="confirmed").collect()
        assert "not available" in out.output
        assert platform.calls[other_id]["outcome"] is None

    def test_the_tool_exposes_the_outcome_schema(self):
        control = CallControl(url="https://x/calls/1", token="t", outcome_schema=SCHEMA)
        tool = control.report_tool()
        assert tool is not control.report_tool(), "fresh Tool instance per call so nest() never mutates shared state"
        assert tool.anthropic_schema["input_schema"]["properties"]["outcome"]["enum"] == [
            "confirmed",
            "declined",
            "callback_requested",
        ]
        assert "BEFORE saying goodbye" in tool.description

    @pytest.mark.asyncio
    async def test_helper_and_nested_agents_do_not_receive_report_tool(self):
        main_agent = Agent(name="caller", model=TestModel(responses=["ok"]))
        filler_agent = Agent(name="voice_filler", model=TestModel(responses=["un segundo"]))
        sub_agent = Agent(name="helper", model=TestModel(responses=["ok"]))
        sub_agent.nest("caller")

        control = CallControl(url="https://x/calls/1", token="t", outcome_schema=SCHEMA)
        token = _current_voice_session.set(
            SimpleNamespace(agent=main_agent, call_context={"user_number": "+34611"}, call_control=control)
        )
        try:
            main_tools, _ = await main_agent._resolve_tools(0)
            filler_tools, _ = await filler_agent._resolve_tools(0)
            sub_tools, _ = await sub_agent._resolve_tools(0)
            assert [t.name for t in main_tools] == ["report_call_outcome"]
            assert filler_tools == []
            assert sub_tools == []
        finally:
            _current_voice_session.reset(token)


class TestTelephonyStartFrame:
    def test_inbound_roles(self):
        from timbal.server.telephony import _call_context_from_start

        ctx = _call_context_from_start({"from": "+34611", "to": "+34930", "call_id": "CA1"}, {"lead": "42"})
        assert ctx["user_number"] == "+34611" and ctx["agent_number"] == "+34930"
        assert ctx["lead"] == "42"

    def test_outbound_roles(self):
        from timbal.server.telephony import _call_context_from_start

        ctx = _call_context_from_start({"from": "+34930", "to": "+34611"}, {"direction": "outbound"})
        assert ctx["user_number"] == "+34611" and ctx["agent_number"] == "+34930"

    def test_outbound_direction_is_preserved_in_call_context(self):
        from timbal.server.telephony import _call_context_from_start

        ctx = _call_context_from_start(
            {"from": "+34930", "to": "+34611", "call_id": "CA2"},
            {"direction": "outbound", "greeting": "Hola"},
        )
        assert ctx["direction"] == "outbound"
        assert "greeting" not in ctx
