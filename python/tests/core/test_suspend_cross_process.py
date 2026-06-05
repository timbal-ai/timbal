"""Serialization / cross-process proof for the suspend()/resume= substrate.

The approval gate has its own cross-process suite (``test_approval_cross_process``).
This file pins the *same* durability contract for ``suspend()``-based
interactions (``ask_user``, ``confirm``, custom tools):

    1. process A runs, calls ``suspend()``, persists the trace to durable
       storage (here: :class:`JsonlTracingProvider`), and exits with status
       ``cancelled``/``input_required``.
    2. some external system (UI, queue) holds the ``interaction_id``.
    3. process B starts fresh, reads the trace from disk, and calls the
       runnable again with ``parent_id=...`` and ``resume={interaction_id: ...}``.

If the ``InteractionEvent``, the paused ``OutputEvent``, and the per-span
``metadata['suspension']`` do not survive a real ``model_dump()`` →
``json.dumps()`` → ``json.loads()`` → ``Trace(spans)`` round-trip, this whole
feature is an in-memory toy. The autouse ``clear_in_memory_tracing_storage``
fixture wipes ``InMemoryTracingProvider._storage`` between tests, so any silent
reliance on shared in-memory state fails here.

JSONL is chosen on purpose: unlike the in-memory (live objects) and SQLite
(opaque blob) providers, every JSONL round-trip exercises the full
serialize/deserialize pipeline in human-readable form.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from pydantic import TypeAdapter
from timbal import Agent, Tool, Workflow
from timbal.core.test_model import TestModel
from timbal.state import RunContext, get_run_context, suspend
from timbal.state.tracing.providers.jsonl import JsonlTracingProvider
from timbal.state.tracing.trace import Trace
from timbal.types.content import ToolUseContent
from timbal.types.events import Event, InteractionEvent, OutputEvent
from timbal.types.message import Message

_EVENT_ADAPTER = TypeAdapter(Event)


def _interaction_events(events) -> list[InteractionEvent]:
    return [e for e in events if isinstance(e, InteractionEvent)]


def _final_output(events) -> OutputEvent:
    return next(e for e in reversed(events) if isinstance(e, OutputEvent))


@pytest.fixture
def provider(tmp_path):
    path = tmp_path / "traces.jsonl"
    yield JsonlTracingProvider.configured(_path=path), path


def _load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Pure event round-trip — does the wire payload survive serialization at all?
# ---------------------------------------------------------------------------


class TestEventSerialization:
    def test_interaction_event_round_trips_through_event_union(self):
        ev = InteractionEvent(
            run_id="06a22e62b85d7b558000db9caf2790b2",
            parent_run_id=None,
            path="assistant.ask_user",
            call_id="c1",
            parent_call_id="c0",
            t0=1780672043531,
            interaction_id="bfd231bf5760d201b31382e168c339d4",
            kind="ask_user",
            runnable_path="assistant.ask_user",
            runnable_name="ask_user",
            runnable_type="Tool",
            payload={"question": "Which database?", "options": ["postgres", "mysql"]},
        )

        # model_dump_json -> json.loads -> discriminated-union validate.
        raw = json.loads(ev.model_dump_json())
        restored = _EVENT_ADAPTER.validate_python(raw)

        assert isinstance(restored, InteractionEvent)
        assert restored.type == "INTERACTION"
        assert restored.interaction_id == ev.interaction_id
        assert restored.kind == "ask_user"
        assert restored.payload == {"question": "Which database?", "options": ["postgres", "mysql"]}
        assert restored == ev

    def test_paused_output_event_round_trips(self):
        out = OutputEvent(
            run_id="r1",
            parent_run_id=None,
            path="assistant",
            call_id="c1",
            parent_call_id=None,
            input={"prompt": "hi"},
            status={"code": "cancelled", "reason": "input_required", "message": "Input required to resume."},
            output={
                "suspension_id": "abc123",
                "status": "input_required",
                "kind": "ask_user",
                "payload": {"question": "name?"},
            },
            error=None,
            t0=1,
            t1=2,
            usage={"suspends:required": 1},
            metadata={"type": "Agent"},
        )

        restored = _EVENT_ADAPTER.validate_python(json.loads(out.model_dump_json()))
        assert isinstance(restored, OutputEvent)
        assert restored.status.code == "cancelled"
        assert restored.status.reason == "input_required"
        assert restored.output["suspension_id"] == "abc123"
        assert restored.output["payload"] == {"question": "name?"}


# ---------------------------------------------------------------------------
# Tool suspend -> persisted -> resumed from disk
# ---------------------------------------------------------------------------


def ask_color() -> str:
    """Suspend asking for a color."""
    return suspend({"question": "favorite color?", "options": ["red", "blue"]}, kind="ask_user")


class TestToolSuspendDurable:
    @pytest.mark.asyncio
    async def test_suspension_metadata_survives_in_jsonl(self, provider):
        prov, path = provider
        tool = Tool(handler=ask_color, tracing_provider=prov)

        events = [e async for e in tool()]
        out = _final_output(events)
        interaction = _interaction_events(events)[0]

        assert out.status.reason == "input_required"
        assert path.exists()

        # The suspension metadata must survive the round-trip onto disk, keyed
        # the same way RunContext.pending_interactions() reads it.
        record = next(r for r in _load_records(path) if r["run_id"] == out.run_id)
        suspension_spans = [
            s for s in record["spans"]
            if (s.get("metadata") or {}).get("suspension", {}).get("id")
        ]
        assert len(suspension_spans) == 1
        meta = suspension_spans[0]["metadata"]["suspension"]
        assert meta["id"] == interaction.interaction_id
        assert meta["kind"] == "ask_user"
        assert meta["payload"] == {"question": "favorite color?", "options": ["red", "blue"]}

    @pytest.mark.asyncio
    async def test_pending_interactions_recoverable_from_storage_only(self, provider):
        """A worker reconstructs the pending interaction purely from the
        deserialized trace — no original RunContext in memory."""
        prov, path = provider
        tool = Tool(handler=ask_color, tracing_provider=prov)

        out = await tool().collect()
        assert out.status.reason == "input_required"

        # Rebuild a Trace straight from the JSON on disk and run the public
        # enumeration helper against it.
        record = next(r for r in _load_records(path) if r["run_id"] == out.run_id)
        ctx = RunContext(id="worker-run")
        ctx._trace = Trace(record["spans"])

        pending = ctx.pending_interactions()
        assert len(pending) == 1
        assert pending[0]["interaction_id"] == out.output["suspension_id"]
        assert pending[0]["kind"] == "ask_user"
        assert pending[0]["payload"] == {"question": "favorite color?", "options": ["red", "blue"]}

    @pytest.mark.asyncio
    async def test_resume_from_durable_store(self, provider):
        prov, path = provider
        calls: list[str] = []

        def ask_then_record() -> str:
            answer = suspend({"question": "color?"}, kind="ask_user")
            calls.append(answer)  # only runs once suspend() returns a value
            return answer

        tool = Tool(handler=ask_then_record, tracing_provider=prov)

        out1 = await tool().collect()
        assert out1.status.reason == "input_required"
        assert calls == []
        interaction_id = out1.output["suspension_id"]

        out2 = await tool(parent_id=out1.run_id, resume={interaction_id: "green"}).collect()
        assert out2.status.code == "success", out2.error
        assert out2.output == "green"
        assert calls == ["green"], "post-suspend code must run exactly once after resume"


# ---------------------------------------------------------------------------
# Agent loop suspend -> resume from disk
# ---------------------------------------------------------------------------


def ask_user(question: str) -> str:
    """Ask the user a question."""
    return suspend({"question": question}, kind="ask_user")


class TestAgentSuspendDurable:
    @pytest.mark.asyncio
    async def test_agent_suspends_persists_then_resumes(self, provider):
        prov, path = provider
        tool_call = Message(
            role="assistant",
            content=[ToolUseContent(id="t1", name="ask_user", input={"question": "name?"})],
            stop_reason="tool_use",
        )
        agent = Agent(
            name="assistant",
            model=TestModel(responses=[tool_call, "Nice to meet you, Ada!"]),
            tools=[ask_user],
            tracing_provider=prov,
        )

        events1 = [e async for e in agent(prompt="hi")]
        out1 = _final_output(events1)
        interaction = _interaction_events(events1)[0]
        assert out1.status.reason == "input_required"

        # Proof it's on disk before resuming.
        assert any(r["run_id"] == out1.run_id for r in _load_records(path))

        out2 = await agent(
            prompt="hi",
            parent_id=out1.run_id,
            resume={interaction.interaction_id: "Ada"},
        ).collect()
        assert out2.status.code == "success", out2.error
        assert out2.output.collect_text() == "Nice to meet you, Ada!"


# ---------------------------------------------------------------------------
# Workflow suspend -> resume from disk (incl. parallel)
# ---------------------------------------------------------------------------


def needs_input() -> str:
    """A workflow step that suspends for input."""
    return suspend({"question": "proceed value?"}, kind="ask_user")


def echo(value: str) -> str:
    return f"got: {value}"


class TestWorkflowSuspendDurable:
    @pytest.mark.asyncio
    async def test_workflow_step_suspends_persists_then_resumes(self, provider):
        prov, path = provider
        wf = (
            Workflow(name="wf", tracing_provider=prov)
            .step(needs_input)
            .step(echo, value=lambda: get_run_context().step_span("needs_input").output)
        )

        events1 = [e async for e in wf()]
        out1 = _final_output(events1)
        interaction = _interaction_events(events1)[0]
        assert out1.status.reason == "input_required"
        assert any(r["run_id"] == out1.run_id for r in _load_records(path))

        out2 = await wf(parent_id=out1.run_id, resume={interaction.interaction_id: "hello"}).collect()
        assert out2.status.code == "success", out2.error
        assert out2.output == "got: hello"

    @pytest.mark.asyncio
    async def test_two_parallel_suspends_resume_via_storage(self, provider):
        prov, path = provider
        calls: list[str] = []

        def ask_a() -> str:
            v = suspend({"q": "a?"}, kind="ask_user")
            calls.append(f"a={v}")
            return v

        def ask_b() -> str:
            v = suspend({"q": "b?"}, kind="ask_user")
            calls.append(f"b={v}")
            return v

        a = Tool(name="ask_a", handler=ask_a)
        b = Tool(name="ask_b", handler=ask_b)
        wf = Workflow(name="parallel_wf", tracing_provider=prov).step(a).step(b)

        events1 = [e async for e in wf()]
        interactions = {e.runnable_path: e for e in _interaction_events(events1)}
        out1 = _final_output(events1)

        assert len(interactions) == 2
        assert out1.status.reason == "input_required"
        assert calls == []

        resume = {
            interactions["parallel_wf.ask_a"].interaction_id: "AA",
            interactions["parallel_wf.ask_b"].interaction_id: "BB",
        }
        out2 = await wf(parent_id=out1.run_id, resume=resume).collect()
        assert out2.status.code == "success", out2.error
        assert sorted(calls) == ["a=AA", "b=BB"], "both suspended steps must resume from storage"


# ---------------------------------------------------------------------------
# True process restart — second interpreter resumes via the JSONL file.
# ---------------------------------------------------------------------------


_SETUP = textwrap.dedent(
    """
    from pathlib import Path
    from timbal.state.tracing.providers.jsonl import JsonlTracingProvider
    TRACE_PATH = Path(r"{path}")
    provider = JsonlTracingProvider.configured(_path=TRACE_PATH)
    """
)

_SUSPEND_BODY = textwrap.dedent(
    """
    import asyncio, json, sys
    from pathlib import Path
    from timbal import Tool
    from timbal.state import suspend
    from timbal.types.events import InteractionEvent

    SIDE_EFFECT_FILE = Path(sys.argv[1])

    def ask_then_write() -> str:
        answer = suspend({"question": "db?"}, kind="ask_user")
        SIDE_EFFECT_FILE.write_text(answer)
        return answer

    tool = Tool(name="ask_then_write", handler=ask_then_write, tracing_provider=provider)

    async def main():
        interaction_id = None
        run_id = None
        async for event in tool():
            if isinstance(event, InteractionEvent):
                interaction_id = event.interaction_id
                run_id = event.run_id
        sys.stdout.write(
            "<<<RESULT>>>"
            + json.dumps({"interaction_id": interaction_id, "run_id": run_id})
            + "<<<END>>>"
        )

    asyncio.run(main())
    """
).strip()

_RESUME_BODY = textwrap.dedent(
    """
    import asyncio, sys
    from pathlib import Path
    from timbal import Tool
    from timbal.state import suspend

    SIDE_EFFECT_FILE = Path(sys.argv[1])
    RUN_ID = sys.argv[2]
    INTERACTION_ID = sys.argv[3]

    def ask_then_write() -> str:
        answer = suspend({"question": "db?"}, kind="ask_user")
        SIDE_EFFECT_FILE.write_text(answer)
        return answer

    tool = Tool(name="ask_then_write", handler=ask_then_write, tracing_provider=provider)

    async def main():
        out = await tool(parent_id=RUN_ID, resume={INTERACTION_ID: "postgres"}).collect()
        sys.stdout.write("<<<RESULT>>>" + out.status.code + "::" + str(out.output) + "<<<END>>>")

    asyncio.run(main())
    """
).strip()


def _extract_result(stdout: str) -> str:
    start = stdout.find("<<<RESULT>>>")
    end = stdout.find("<<<END>>>")
    assert start != -1 and end != -1, f"sentinel not found in stdout:\n{stdout}"
    return stdout[start + len("<<<RESULT>>>"):end]


class TestRealSubprocessResume:
    def test_suspend_in_one_process_resume_in_another(self, tmp_path):
        path = tmp_path / "traces.jsonl"
        side_effect = tmp_path / "side_effect.txt"
        setup = _SETUP.format(path=path.as_posix()).strip()
        suspend_script = setup + "\n" + _SUSPEND_BODY
        resume_script = setup + "\n" + _RESUME_BODY

        first = subprocess.run(  # noqa: S603
            [sys.executable, "-c", suspend_script, str(side_effect)],
            capture_output=True, text=True, timeout=60,
        )
        assert first.returncode == 0, f"suspend failed:\nstdout={first.stdout}\nstderr={first.stderr}"
        payload = json.loads(_extract_result(first.stdout))
        assert payload["interaction_id"], first.stderr
        assert payload["run_id"], first.stderr
        assert path.exists(), first.stderr
        assert not side_effect.exists(), "post-suspend code must NOT run before resume"

        second = subprocess.run(  # noqa: S603
            [sys.executable, "-c", resume_script, str(side_effect), payload["run_id"], payload["interaction_id"]],
            capture_output=True, text=True, timeout=60,
        )
        assert second.returncode == 0, f"resume failed:\nstdout={second.stdout}\nstderr={second.stderr}"
        result = _extract_result(second.stdout)
        assert result == "success::postgres", second.stderr
        assert side_effect.read_text() == "postgres", "handler must resume across processes via the JSONL trace"
