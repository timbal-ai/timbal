"""Contract tests for outbound voice calls (issue #200). A fake platform; no real calls are placed."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from typing import Literal

import pytest
from pydantic import BaseModel
from timbal import Agent, Tool, Workflow
from timbal.core.test_model import TestModel
from timbal.platform.voice_calls import (
    IdempotencyKeyReused,
    VoiceCallError,
    VoiceCallsUnavailable,
    cancel_voice_call,
    normalize_context,
    outcome_schema_for,
    start_voice_call,
    wait_for_voice_call,
)
from timbal.state import get_run_context
from timbal.state.config import PlatformAuth, PlatformAuthType, PlatformConfig, PlatformSubject
from timbal.state.tracing.providers.jsonl import JsonlTracingProvider
from timbal.tools import StartVoiceCall, WaitForVoiceCall
from timbal.types.content import ToolUseContent
from timbal.types.events import InteractionEvent, OutputEvent
from timbal.types.message import Message

from .voice_platform import FakeVoicePlatform


class AppointmentOutcome(BaseModel):
    outcome: Literal["confirmed", "declined", "callback_requested"]
    note: str | None = None


@pytest.fixture
def platform(monkeypatch):
    import timbal.state.config_loader as loader

    fake = FakeVoicePlatform()
    cfg = PlatformConfig(
        host="api.timbal.ai",
        auth=PlatformAuth(type=PlatformAuthType.BEARER, token="t"),
        subject=PlatformSubject(org_id="org_1", project_id="77"),
    )
    monkeypatch.setattr(loader, "_cached_default_config", cfg)
    monkeypatch.setattr(loader, "_default_config_resolved", True)
    monkeypatch.setattr("timbal.platform.utils.request_client", fake.request_client())
    monkeypatch.setattr("timbal.voice.call._outcome_client", fake.outcome_client())
    return fake


@pytest.fixture
def jsonl(tmp_path):
    return JsonlTracingProvider.configured(_path=tmp_path / "traces.jsonl")


def _final(events) -> OutputEvent:
    return next(e for e in reversed(events) if isinstance(e, OutputEvent) and e.parent_call_id is None)


def _interactions(events) -> list[InteractionEvent]:
    return [e for e in events if isinstance(e, InteractionEvent)]


async def _run(runnable, **kwargs):
    events = [e async for e in runnable(**kwargs)]
    return _final(events), _interactions(events)


def _reminder_workflow(prov, side_effects: list[str], *, key: str | None = "appt:1:reminder") -> Workflow:
    async def call_customer() -> dict:
        call = await start_voice_call(
            "receptionist",
            "+34600000001",
            context={"customer_name": "Ada", "appointment": {"at": "2026-11-02T10:00"}},
            outcome=AppointmentOutcome,
            idempotency_key=key,
        )
        result = await wait_for_voice_call(call)
        return result.model_dump(mode="json")

    def apply_outcome(result: dict) -> str:
        side_effects.append(result["outcome"]["outcome"] if result["outcome"] else result["outcome_status"])
        return side_effects[-1]

    return (
        Workflow(name="reminder", tracing_provider=prov)
        .step(Tool(name="call_customer", handler=call_customer))
        .step(apply_outcome, result=lambda: get_run_context().step_span("call_customer").output)
    )


class TestStart:
    @pytest.mark.asyncio
    async def test_request_carries_context_schema_origin_and_key(self, platform):
        call = await start_voice_call(
            "receptionist",
            "+34600000001",
            context={"customer_name": "Ada", "missing": None},
            outcome=AppointmentOutcome,
            greeting="Hola Ada",
            idempotency_key="k1",
            rev="main",
        )
        sent = platform.posts[0]
        assert sent["key"] == "k1"
        assert sent["rev"] == "main"
        assert sent["body"]["context"] == {"customer_name": "Ada"}
        assert sent["body"]["outcome_schema"]["type"] == "object"
        assert sent["body"]["greeting"] == "Hola Ada"
        assert "run_id" in sent["body"]["origin"]
        assert call.id == "call_1" and call.org_id == "org_1" and call.project_id == "77"

    @pytest.mark.asyncio
    async def test_duplicate_triggers_with_a_business_key_dial_once(self, platform):
        first = await start_voice_call("receptionist", "+34600000001", idempotency_key="appt:1:reminder")
        second = await start_voice_call("receptionist", "+34600000001", idempotency_key="appt:1:reminder")
        assert platform.dials == 1
        assert first.id == second.id
        assert second.replayed is True

    @pytest.mark.asyncio
    async def test_ambiguous_start_retries_with_the_same_key_and_gets_the_original(self, platform):
        platform.timeout_after_accept = 1
        call = await start_voice_call("receptionist", "+34600000001", idempotency_key="k-timeout")
        assert platform.dials == 1
        assert len(platform.posts) == 2
        assert {p["key"] for p in platform.posts} == {"k-timeout"}
        assert call.replayed is True

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("platform")
    async def test_key_reused_for_a_different_dial(self):
        await start_voice_call("receptionist", "+34600000001", idempotency_key="k")
        with pytest.raises(IdempotencyKeyReused):
            await start_voice_call("receptionist", "+34600000002", idempotency_key="k")

    @pytest.mark.asyncio
    async def test_disabled_platform_is_a_clear_error(self, platform):
        platform.enabled = False
        with pytest.raises(VoiceCallsUnavailable, match="not available"):
            await start_voice_call("receptionist", "+34600000001", idempotency_key="k")

    @pytest.mark.asyncio
    async def test_platform_without_call_results_fails_loudly(self, platform):
        platform.legacy = True
        with pytest.raises(VoiceCallsUnavailable, match="outcome_status"):
            await start_voice_call("receptionist", "+34600000001", idempotency_key="k")

    def test_context_rules(self):
        with pytest.raises(ValueError, match="reserved"):
            normalize_context({"to": "+34"})
        with pytest.raises(ValueError, match="identifier-like"):
            normalize_context({"bad key": 1})
        with pytest.raises(ValueError, match="at most 64"):
            normalize_context({f"k{i}": i for i in range(65)})
        with pytest.raises(ValueError, match="bytes"):
            normalize_context({"blob": "x" * 17000})
        assert normalize_context({"a": None, "b": {"c": 1}}) == {"b": {"c": 1}}

    def test_outcome_schema_rules(self):
        assert outcome_schema_for(AppointmentOutcome)["type"] == "object"
        with pytest.raises(ValueError, match="object"):
            outcome_schema_for({"type": "string"})
        with pytest.raises(ValueError, match="reference itself"):
            outcome_schema_for({"type": "object", "properties": {"a": {"$ref": "https://x/y.json"}}})

    @pytest.mark.asyncio
    async def test_default_key_is_distinct_per_call_site_and_attempt(self, platform):
        async def two_calls() -> list[str]:
            a = await start_voice_call("receptionist", "+34600000001")
            b = await start_voice_call("receptionist", "+34600000001")
            c = await start_voice_call("receptionist", "+34600000001", attempt=1)
            return [a.idempotency_key, b.idempotency_key, c.idempotency_key]

        out = await Tool(name="two_calls", handler=two_calls)().collect()
        assert out.status.code == "success", out.error
        assert len(set(out.output)) == 3
        assert platform.dials == 3


class TestWaitAndResume:
    @pytest.mark.asyncio
    async def test_suspends_then_resumes_with_a_validated_typed_result(self, platform, jsonl):
        effects: list[str] = []
        wf = _reminder_workflow(jsonl, effects)
        out1, interactions = await _run(wf)
        assert out1.status.reason == "input_required"
        assert interactions[0].kind == "voice_call"
        assert interactions[0].payload == {"call_id": "call_1"}
        assert effects == []

        view = platform.finish("call_1", outcome={"outcome": "confirmed"})
        out2, _ = await _run(wf, parent_id=out1.run_id, resume={interactions[0].interaction_id: view})
        assert out2.status.code == "success", out2.error
        assert effects == ["confirmed"]
        assert platform.dials == 1
        assert len(platform.posts) == 1, "the resumed run replays the call from the parent trace"

    @pytest.mark.asyncio
    async def test_result_before_wait_does_not_suspend(self, platform):
        async def start_then_wait() -> str:
            call = await start_voice_call("receptionist", "+34600000001", idempotency_key="early")
            platform.finish(call.id, outcome={"outcome": "declined"})
            result = await wait_for_voice_call(call, outcome=AppointmentOutcome)
            return result.outcome.outcome

        out = await Tool(name="start_then_wait", handler=start_then_wait)().collect()
        assert out.status.code == "success", out.error
        assert out.output == "declined"

    @pytest.mark.asyncio
    async def test_chain_of_two_suspends_dials_each_call_once(self, platform, jsonl):
        async def two_calls() -> list[str]:
            first = await start_voice_call("receptionist", "+34600000001")
            a = await wait_for_voice_call(first)
            second = await start_voice_call("receptionist", "+34600000002")
            b = await wait_for_voice_call(second)
            return [a.status, b.status]

        tool = Tool(name="two_calls", handler=two_calls, tracing_provider=jsonl)
        out1, i1 = await _run(tool)
        assert out1.status.reason == "input_required"
        view1 = platform.finish("call_1")
        out2, i2 = await _run(tool, parent_id=out1.run_id, resume={i1[0].interaction_id: view1})
        assert out2.status.reason == "input_required"
        assert i2[0].payload == {"call_id": "call_2"}
        assert platform.gets == ["call_1", "call_2"]
        view2 = platform.finish("call_2", status="no_answer", end_reason="user_unavailable")
        out3, _ = await _run(tool, parent_id=out2.run_id, resume={i2[0].interaction_id: view2})
        assert out3.status.code == "success", out3.error
        assert out3.output == ["completed", "no_answer"]
        assert platform.dials == 2
        assert platform.gets == ["call_1", "call_2"], "call_1 result replayed from parent trace without another GET"
        first_key = platform.posts[0]["key"]
        assert [p["key"] for p in platform.posts].count(first_key) == 1, "exactly one POST for the first call"

    @pytest.mark.asyncio
    async def test_reusing_key_across_resume_for_different_dial_is_rejected(self, platform, jsonl):
        async def bad_reuse() -> str:
            first = await start_voice_call("receptionist", "+34600000001", idempotency_key="shared")
            await wait_for_voice_call(first)
            await start_voice_call("receptionist", "+34600000099", idempotency_key="shared")
            return "never"

        tool = Tool(name="bad_reuse", handler=bad_reuse, tracing_provider=jsonl)
        out1, i1 = await _run(tool)
        assert out1.status.reason == "input_required"
        view1 = platform.finish("call_1")
        out2, _ = await _run(tool, parent_id=out1.run_id, resume={i1[0].interaction_id: view1})
        assert out2.status.code == "error"
        assert "already used for a different dial" in str(out2.error)

    @pytest.mark.asyncio
    async def test_duplicate_completion_runs_downstream_once(self, platform, jsonl):
        effects: list[str] = []
        wf = _reminder_workflow(jsonl, effects)
        out1, interactions = await _run(wf)
        view = platform.finish("call_1", outcome={"outcome": "confirmed"})
        resume = {interactions[0].interaction_id: view}
        first, _ = await _run(wf, parent_id=out1.run_id, resume=resume)
        duplicate, _ = await _run(wf, parent_id=out1.run_id, resume=resume)
        assert first.status.code == "success", first.error
        assert duplicate.status.code == "cancelled"
        assert effects == ["confirmed"]

    @pytest.mark.asyncio
    async def test_result_for_another_call_is_rejected(self, platform, jsonl):
        effects: list[str] = []
        wf = _reminder_workflow(jsonl, effects)
        out1, interactions = await _run(wf)
        await start_voice_call("receptionist", "+34600000009", idempotency_key="other")
        other = platform.finish("call_2", outcome={"outcome": "confirmed"})
        out2, _ = await _run(wf, parent_id=out1.run_id, resume={interactions[0].interaction_id: other})
        assert out2.status.code == "error"
        assert "while waiting for 'call_1'" in str(out2.error)
        assert effects == []

    @pytest.mark.asyncio
    async def test_result_that_is_not_final_is_rejected(self, platform, jsonl):
        effects: list[str] = []
        wf = _reminder_workflow(jsonl, effects)
        out1, interactions = await _run(wf)
        out2, _ = await _run(
            wf, parent_id=out1.run_id, resume={interactions[0].interaction_id: platform.view("call_1")}
        )
        assert out2.status.code == "error"
        assert "before its result was final" in str(out2.error)

    @pytest.mark.asyncio
    async def test_invalid_typed_outcome_is_a_bug_not_unavailable(self, platform, jsonl):
        effects: list[str] = []
        wf = _reminder_workflow(jsonl, effects)
        out1, interactions = await _run(wf)
        view = platform.finish("call_1", outcome={"outcome": "maybe"})
        out2, _ = await _run(wf, parent_id=out1.run_id, resume={interactions[0].interaction_id: view})
        assert out2.status.code == "error"
        assert "does not match AppointmentOutcome" in str(out2.error)

        null_reported = {**view, "outcome": None, "outcome_status": "reported"}
        out3, _ = await _run(wf, parent_id=out1.run_id, resume={interactions[0].interaction_id: null_reported})
        assert out3.status.code == "error"
        assert "does not match AppointmentOutcome" in str(out3.error)

    @pytest.mark.asyncio
    async def test_parallel_waits_correlate_and_resuspend_when_superseded(self, platform, jsonl):
        async def call_a() -> str:
            result = await wait_for_voice_call(
                await start_voice_call("receptionist", "+34600000001", idempotency_key="a")
            )
            return f"a:{result.status}"

        async def call_b() -> str:
            result = await wait_for_voice_call(
                await start_voice_call("receptionist", "+34600000002", idempotency_key="b")
            )
            return f"b:{result.status}"

        wf = (
            Workflow(name="parallel", tracing_provider=jsonl)
            .step(Tool(name="call_a", handler=call_a))
            .step(Tool(name="call_b", handler=call_b))
        )
        out1, i1 = await _run(wf)
        by_call = {i.payload["call_id"]: i for i in i1}
        assert set(by_call) == {"call_1", "call_2"}

        view_a = platform.finish("call_1")
        out2, i2 = await _run(wf, parent_id=out1.run_id, resume={by_call["call_1"].interaction_id: view_a})
        assert out2.status.reason == "input_required"
        assert [i.payload for i in i2] == [{"call_id": "call_2"}]

        view_b = platform.finish("call_2", status="busy", end_reason="user_rejected")
        out3, _ = await _run(wf, parent_id=out2.run_id, resume={i2[0].interaction_id: view_b})
        assert out3.status.code == "success", out3.error
        assert platform.dials == 2

    @pytest.mark.asyncio
    async def test_cancellation_is_final_and_unavailable(self, platform):
        async def start_and_cancel() -> dict:
            call = await start_voice_call("receptionist", "+34600000001", idempotency_key="c")
            await cancel_voice_call(call)
            return (await wait_for_voice_call(call)).model_dump(mode="json")

        out = await Tool(name="start_and_cancel", handler=start_and_cancel)().collect()
        assert out.status.code == "success", out.error
        assert out.output["status"] == "cancelled"
        assert out.output["outcome_status"] == "unavailable"
        assert platform.dials == 1

    @pytest.mark.asyncio
    async def test_no_answer_and_finalization_timeout_end_unavailable(self, platform):
        no_answer = await start_voice_call("receptionist", "+34600000001", idempotency_key="na")
        platform.finish(no_answer.id, status="no_answer", end_reason="user_unavailable")
        silent = await start_voice_call("receptionist", "+34600000002", idempotency_key="to")
        platform.finish(silent.id, status="completed")

        async def read(call) -> dict:
            return (await wait_for_voice_call(call)).model_dump(mode="json")

        r1 = await Tool(name="read", handler=read)(call=no_answer.model_dump()).collect()
        r2 = await Tool(name="read", handler=read)(call=silent.model_dump()).collect()
        assert (r1.output["status"], r1.output["outcome_status"]) == ("no_answer", "unavailable")
        assert (r2.output["status"], r2.output["outcome_status"]) == ("completed", "unavailable")

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("platform")
    async def test_another_tenants_call_cannot_be_read(self):
        call = await start_voice_call("receptionist", "+34600000001", idempotency_key="t")
        foreign = call.model_dump()
        foreign["org_id"] = "org_2"
        with pytest.raises(VoiceCallError):
            await wait_for_voice_call(foreign)


class TestToolsAndAgents:
    @pytest.mark.asyncio
    async def test_workflow_steps_with_the_tools(self, platform, jsonl):
        wf = (
            Workflow(name="steps", tracing_provider=jsonl)
            .step(
                StartVoiceCall(workforce="receptionist", outcome=AppointmentOutcome),
                to="+34600000001",
                context={"customer_name": "Ada"},
            )
            .step(
                WaitForVoiceCall(outcome=AppointmentOutcome),
                call=lambda: get_run_context().step_span("start_voice_call").output,
            )
        )
        out1, interactions = await _run(wf)
        assert out1.status.reason == "input_required"
        assert platform.posts[0]["body"]["outcome_schema"]["properties"]["outcome"]["enum"] == [
            "confirmed",
            "declined",
            "callback_requested",
        ]
        view = platform.finish("call_1", outcome={"outcome": "confirmed"})
        out2, _ = await _run(wf, parent_id=out1.run_id, resume={interactions[0].interaction_id: view})
        assert out2.status.code == "success", out2.error
        assert out2.output["outcome"]["outcome"] == "confirmed"
        assert platform.dials == 1

    def test_preconfigured_tool_fields_are_excluded_from_llm_schema(self):
        configured = StartVoiceCall(workforce="receptionist", outcome=AppointmentOutcome)
        props = configured.anthropic_schema["input_schema"]["properties"]
        assert "workforce" not in props
        assert "outcome_schema" not in props
        assert "to" in props

        open_tool = StartVoiceCall()
        open_props = open_tool.anthropic_schema["input_schema"]["properties"]
        assert "workforce" in open_props
        assert "outcome_schema" in open_props

    @pytest.mark.asyncio
    async def test_agent_tool_starts_a_call(self, platform):
        tool_call = Message(
            role="assistant",
            content=[
                ToolUseContent(
                    id="toolu_1",
                    name="start_voice_call",
                    input={"to": "+34600000001", "context": {"reason": "reminder"}},
                )
            ],
            stop_reason="tool_use",
        )
        agent = Agent(
            name="ops",
            model=TestModel(responses=[tool_call, "Calling now."]),
            tools=[StartVoiceCall(workforce="receptionist")],
        )
        out = await agent(prompt="call Ada").collect()
        assert out.status.code == "success", out.error
        assert platform.dials == 1
        assert platform.posts[0]["body"]["context"] == {"reason": "reminder"}


_SETUP = textwrap.dedent(
    """
    import asyncio, json, sys
    from pathlib import Path
    from timbal import Tool, Workflow
    from timbal.platform.voice_calls import start_voice_call, wait_for_voice_call
    from timbal.state import get_run_context
    from timbal.state.tracing.providers.jsonl import JsonlTracingProvider

    prov = JsonlTracingProvider.configured(_path=Path({path!r}))
    side_effect = sys.argv[1]

    async def call_customer() -> dict:
        call = await start_voice_call("receptionist", "+34600000001", idempotency_key="appt:1", org_id="org_1", project_id="77")
        return (await wait_for_voice_call(call)).model_dump(mode="json")

    def apply_outcome(result: dict) -> str:
        with open(side_effect, "a") as fh:
            fh.write(result["outcome"]["outcome"] + "\\n")
        return result["outcome"]["outcome"]

    wf = (
        Workflow(name="reminder", tracing_provider=prov)
        .step(Tool(name="call_customer", handler=call_customer))
        .step(
            apply_outcome,
            depends_on=["call_customer"],
            result=lambda: get_run_context().step_span("call_customer").output,
        )
    )

    async def main():
        out = await wf(parent_id=sys.argv[2], resume={{sys.argv[3]: json.loads(sys.argv[4])}}).collect()
        sys.stdout.write("<<<" + out.status.code + "::" + str(out.output) + ">>>")

    asyncio.run(main())
    """
)


class TestFreshProcessResume:
    @pytest.mark.asyncio
    async def test_a_fresh_process_resumes_without_dialing(self, platform, tmp_path):
        path = tmp_path / "traces.jsonl"
        prov = JsonlTracingProvider.configured(_path=path)
        effects: list[str] = []
        wf = _reminder_workflow(prov, effects, key="appt:1")
        out1, interactions = await _run(wf)
        view = platform.finish("call_1", outcome={"outcome": "callback_requested"})

        # The child process has no platform credentials and no fake: any dial or read would fail.
        side_effect = tmp_path / "effects.txt"
        script = _SETUP.format(path=path.as_posix())
        env = {k: v for k, v in os.environ.items() if not k.startswith("TIMBAL_")}
        env.update({"HOME": str(tmp_path), "USERPROFILE": str(tmp_path), "TIMBAL_LOG_LEVEL": "CRITICAL"})
        done = subprocess.run(  # noqa: S603
            [
                sys.executable,
                "-c",
                script,
                str(side_effect),
                out1.run_id,
                interactions[0].interaction_id,
                json.dumps(view),
            ],
            capture_output=True,
            text=True,
            timeout=90,
            env=env,
        )
        assert done.returncode == 0, done.stderr
        assert "<<<success::callback_requested>>>" in done.stdout, done.stdout + done.stderr
        assert side_effect.read_text().splitlines() == ["callback_requested"]
        assert platform.dials == 1


class TestDocumentedExample:
    @pytest.mark.asyncio
    async def test_reminder_page_runs_event_call_wait_branch(self, platform, jsonl, tmp_path):
        import importlib.util
        import re
        from pathlib import Path

        page = Path(__file__).resolve().parents[3] / "docs/voice/outbound-calls.mdx"
        code = re.search(r"```python reminder\.py\n(.*?)```", page.read_text(encoding="utf-8"), re.S).group(1)
        # A real file, so step dependencies can be read from the lambdas' source.
        (tmp_path / "reminder.py").write_text(code, encoding="utf-8")
        spec = importlib.util.spec_from_file_location("docs_reminder", tmp_path / "reminder.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        wf = module.reminder
        wf.tracing_provider = jsonl

        out1, interactions = await _run(wf, appointment_id="a1")
        assert out1.status.reason == "input_required"
        assert interactions[0].payload == {"call_id": "call_1"}
        assert platform.posts[0]["body"]["context"]["customer_name"] == "Ada"

        view = platform.finish("call_1", outcome={"outcome": "confirmed"})
        out2, _ = await _run(
            wf, appointment_id="a1", parent_id=out1.run_id, resume={interactions[0].interaction_id: view}
        )
        assert out2.status.code == "success", out2.error
        assert out2.output == "appointment confirmed"
        assert platform.dials == 1
