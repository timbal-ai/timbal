"""Cross-process / cross-restart proof for the human-in-the-loop approval gate.

Real HITL is:

    1. process A runs the agent, persists trace to durable storage,
       hits a gate, exits.
    2. some external system (queue, DB, UI) holds the ``approval_id``.
    3. process B starts fresh, asks the trace store for pending
       approvals, then calls the runnable again with
       ``approval_decisions=...`` and ``parent_id=...``.

If that flow doesn't work, the whole feature is an in-memory toy.

The tests in this file pin the contract by using on-disk providers
(:class:`JsonlTracingProvider` and :class:`SqliteTracingProvider`) as
the only path between turns. The autouse fixture in ``conftest`` clears
``InMemoryTracingProvider._storage`` between tests, so any silent
reliance on shared in-memory state would fail here.

Every test is parametrised across both providers via the ``backend``
fixture. The final test boots a real subprocess to literally prove the
resume loop works across Python processes — not just across
``RunContext`` instances within one interpreter.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
from timbal import Agent, Tool, Workflow
from timbal.core.test_model import TestModel
from timbal.state.tracing.providers.jsonl import JsonlTracingProvider
from timbal.state.tracing.providers.sqlite import SqliteTracingProvider
from timbal.types.approval import ApprovalResolution
from timbal.types.content import ToolUseContent
from timbal.types.events import ApprovalEvent, OutputEvent
from timbal.types.message import Message


def _approval_events(events) -> list[ApprovalEvent]:
    return [e for e in events if isinstance(e, ApprovalEvent)]


def _final_output(events) -> OutputEvent:
    return next(e for e in reversed(events) if isinstance(e, OutputEvent))


# ---------------------------------------------------------------------------
# Backend-agnostic durable-store fixture
# ---------------------------------------------------------------------------


class _Backend:
    """Provider + raw-record reader, abstracted over backend storage."""

    def __init__(self, name: str, path: Path, provider, load_records, script_setup: str):
        self.name = name
        self.path = path
        self.provider = provider
        self._load_records = load_records
        self.script_setup = script_setup  # python snippet for subprocess scripts

    def load_records(self) -> list[dict]:
        """Return [{run_id, parent_id, spans}, ...] regardless of backend."""
        return self._load_records()


def _make_jsonl_backend(tmp_path: Path) -> _Backend:
    path = tmp_path / "traces.jsonl"
    provider = JsonlTracingProvider.configured(_path=path)

    def load() -> list[dict]:
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    script_setup = textwrap.dedent(
        f"""
        from pathlib import Path
        from timbal.state.tracing.providers.jsonl import JsonlTracingProvider
        TRACE_PATH = Path(r"{path}")
        provider = JsonlTracingProvider.configured(_path=TRACE_PATH)
        """
    ).strip()
    return _Backend("jsonl", path, provider, load, script_setup)


def _make_sqlite_backend(tmp_path: Path) -> _Backend:
    path = tmp_path / "traces.db"
    provider = SqliteTracingProvider.configured(_path=path)

    def load() -> list[dict]:
        if not path.exists():
            return []
        conn = sqlite3.connect(str(path))
        try:
            rows = conn.execute(
                "SELECT run_id, parent_id, spans FROM runs"
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        finally:
            conn.close()
        return [
            {"run_id": r[0], "parent_id": r[1], "spans": json.loads(r[2])}
            for r in rows
        ]

    script_setup = textwrap.dedent(
        f"""
        from pathlib import Path
        from timbal.state.tracing.providers.sqlite import SqliteTracingProvider
        TRACE_PATH = Path(r"{path}")
        provider = SqliteTracingProvider.configured(_path=TRACE_PATH)
        """
    ).strip()
    return _Backend("sqlite", path, provider, load, script_setup)


@pytest.fixture(params=["jsonl", "sqlite"])
def backend(request, tmp_path) -> _Backend:
    return {
        "jsonl": _make_jsonl_backend,
        "sqlite": _make_sqlite_backend,
    }[request.param](tmp_path)


def _agent_with_tool(provider, *, responses, tool_name, handler, requires_approval=True):
    tool = Tool(name=tool_name, handler=handler, requires_approval=requires_approval)
    return Agent(
        name="cross_process_agent",
        model=TestModel(responses=responses),
        tools=[tool],
        tracing_provider=provider,
    )


# ---------------------------------------------------------------------------
# Single tool gate
# ---------------------------------------------------------------------------


class TestSingleToolGate:
    @pytest.mark.asyncio
    async def test_gate_persists_then_resumes_from_durable_store(self, backend):
        """Turn 1 hits the gate, turn 2 resumes via parent_id + decision.

        The provider's ``get()`` only reads from disk/SQLite, so the second
        run cannot rely on any in-memory state from turn 1.
        """
        calls: list[int] = []

        def wire(amount: int) -> str:
            calls.append(amount)
            return f"wired {amount}"

        agent = _agent_with_tool(
            backend.provider,
            responses=[
                Message(
                    role="assistant",
                    content=[ToolUseContent(id="t1", name="wire", input={"amount": 500})],
                    stop_reason="tool_use",
                ),
                "done",
            ],
            tool_name="wire",
            handler=wire,
        )

        events1 = [e async for e in agent(prompt="wire 500")]
        approvals = _approval_events(events1)
        out1 = _final_output(events1)

        assert out1.status.code == "cancelled"
        assert out1.status.reason == "approval_required"
        assert len(approvals) == 1
        assert calls == []

        # Sanity: trace really hit durable storage before we resume.
        assert backend.path.exists()
        records = backend.load_records()
        assert any(r["run_id"] == out1.run_id for r in records)

        out2 = await agent(
            prompt="wire 500",
            parent_id=out1.run_id,
            approval_decisions={approvals[0].approval_id: True},
        ).collect()

        assert out2.status.code == "success", out2.error
        assert calls == [500], "tool must execute exactly once after resume"

    @pytest.mark.asyncio
    async def test_denial_persists(self, backend):
        calls: list[int] = []

        def charge(amount: int) -> str:
            calls.append(amount)
            return "charged"

        agent = _agent_with_tool(
            backend.provider,
            responses=[
                Message(
                    role="assistant",
                    content=[ToolUseContent(id="t1", name="charge", input={"amount": 999})],
                    stop_reason="tool_use",
                ),
                "I will not charge $999.",
            ],
            tool_name="charge",
            handler=charge,
        )

        events1 = [e async for e in agent(prompt="charge 999")]
        approval = _approval_events(events1)[0]
        out1 = _final_output(events1)
        assert out1.status.reason == "approval_required"

        out2 = await agent(
            prompt="charge 999",
            parent_id=out1.run_id,
            approval_decisions={
                approval.approval_id: {"approved": False, "reason": "policy"},
            },
        ).collect()

        assert out2.status.code == "success", out2.error
        assert "will not charge" in out2.output.collect_text().lower()
        assert calls == [], "denied tool must never execute"

    @pytest.mark.asyncio
    async def test_pending_approval_recoverable_from_storage_only(self, backend):
        """A 'queue worker' should be able to discover the pending approval
        without ever importing the original RunContext — just by reading the
        durable store and pulling ``metadata['approval']``."""

        def deploy(env: str) -> str:
            return f"deployed {env}"

        tool = Tool(
            name="deploy",
            handler=deploy,
            requires_approval=True,
            approval_prompt="Promote to prod?",
            approval_description="Production deploy",
        )
        wf = Workflow(name="deploy_wf", tracing_provider=backend.provider).step(tool)

        events1 = [e async for e in wf(env="prod")]
        out1 = _final_output(events1)
        assert out1.status.reason == "approval_required"

        records = backend.load_records()
        run_record = next(r for r in records if r["run_id"] == out1.run_id)
        approval_spans = [
            s for s in run_record["spans"]
            if (s.get("metadata") or {}).get("approval", {}).get("id")
        ]
        assert len(approval_spans) == 1
        approval_meta = approval_spans[0]["metadata"]["approval"]
        assert approval_meta["prompt"] == "Promote to prod?"
        assert approval_meta["description"] == "Production deploy"

        ev_id = _approval_events(events1)[0].approval_id
        assert approval_meta["id"] == ev_id


# ---------------------------------------------------------------------------
# Expired-resolution
# ---------------------------------------------------------------------------


class TestExpiredResolution:
    @pytest.mark.asyncio
    async def test_expired_decision_emits_fresh_event(self, backend):
        calls: list[int] = []

        def transfer(amount: int) -> str:
            calls.append(amount)
            return "ok"

        agent = _agent_with_tool(
            backend.provider,
            responses=[
                Message(
                    role="assistant",
                    content=[ToolUseContent(id="t1", name="transfer", input={"amount": 50})],
                    stop_reason="tool_use",
                ),
                Message(
                    role="assistant",
                    content=[ToolUseContent(id="t2", name="transfer", input={"amount": 50})],
                    stop_reason="tool_use",
                ),
                "done",
            ],
            tool_name="transfer",
            handler=transfer,
        )

        events1 = [e async for e in agent(prompt="transfer 50")]
        approval = _approval_events(events1)[0]
        out1 = _final_output(events1)
        assert out1.status.reason == "approval_required"

        already_expired = ApprovalResolution(
            approved=True,
            expires_at=int(time.time() * 1000) - 1,
        )
        events2 = [
            e async for e in agent(
                prompt="transfer 50",
                parent_id=out1.run_id,
                approval_decisions={approval.approval_id: already_expired},
            )
        ]
        approvals2 = _approval_events(events2)
        out2 = _final_output(events2)

        assert out2.status.reason == "approval_required", "expired decision must NOT execute"
        assert calls == []
        assert len(approvals2) >= 1
        assert approvals2[0].approval_id == approval.approval_id

        records = backend.load_records()
        run2 = next(r for r in records if r["run_id"] == out2.run_id)
        approval_meta = next(
            (s["metadata"] or {}).get("approval", {})
            for s in run2["spans"]
            if (s.get("metadata") or {}).get("approval", {}).get("id")
        )
        assert approval_meta.get("expired") is True


# ---------------------------------------------------------------------------
# Parallel gates
# ---------------------------------------------------------------------------


class TestParallelGatesResume:
    @pytest.mark.asyncio
    async def test_workflow_two_parallel_steps_resume_via_storage(self, backend):
        calls: list[str] = []

        def step_a(x: int) -> str:
            calls.append(f"a={x}")
            return "a"

        def step_b(y: int) -> str:
            calls.append(f"b={y}")
            return "b"

        a = Tool(name="step_a", handler=step_a, requires_approval=True)
        b = Tool(name="step_b", handler=step_b, requires_approval=True)
        wf = Workflow(name="parallel_wf", tracing_provider=backend.provider).step(a).step(b)

        events1 = [e async for e in wf(x=1, y=2)]
        approvals = _approval_events(events1)
        out1 = _final_output(events1)

        assert len(approvals) == 2
        assert out1.status.reason == "approval_required"
        assert calls == []

        decisions = {a.approval_id: True for a in approvals}
        out2 = await wf(
            x=1, y=2,
            parent_id=out1.run_id,
            approval_decisions=decisions,
        ).collect()

        assert out2.status.code == "success", out2.error
        assert sorted(calls) == ["a=1", "b=2"], "both gated steps must execute on resume"

    @pytest.mark.asyncio
    async def test_workflow_partial_resume_persists_remaining_gate(self, backend):
        calls: list[str] = []

        def step_a(x: int) -> str:
            calls.append(f"a={x}")
            return "a"

        def step_b(y: int) -> str:
            calls.append(f"b={y}")
            return "b"

        a = Tool(name="step_a", handler=step_a, requires_approval=True)
        b = Tool(name="step_b", handler=step_b, requires_approval=True)
        wf = Workflow(name="parallel_wf", tracing_provider=backend.provider).step(a).step(b)

        events1 = [e async for e in wf(x=1, y=2)]
        approvals = {e.runnable_path: e for e in _approval_events(events1)}
        out1 = _final_output(events1)

        events2 = [
            e async for e in wf(
                x=1, y=2,
                parent_id=out1.run_id,
                approval_decisions={approvals["parallel_wf.step_a"].approval_id: True},
            )
        ]
        out2 = _final_output(events2)
        approvals2 = _approval_events(events2)

        assert calls == ["a=1"], "only the approved step must run"
        assert out2.status.reason == "approval_required"

        records = backend.load_records()
        run2 = next(r for r in records if r["run_id"] == out2.run_id)
        pending = [
            (s.get("metadata") or {}).get("approval", {}).get("id")
            for s in run2["spans"]
            if (
                isinstance(s.get("status"), dict)
                and s["status"].get("code") == "cancelled"
                and s["status"].get("reason") == "approval_required"
            )
        ]
        assert approvals["parallel_wf.step_b"].approval_id in pending
        assert {a.approval_id for a in approvals2} == {
            approvals["parallel_wf.step_b"].approval_id
        }


# ---------------------------------------------------------------------------
# True process restart — second Python interpreter resumes through storage.
# ---------------------------------------------------------------------------


_GATE_BODY = textwrap.dedent(
    """
    import asyncio, json, sys
    from pathlib import Path
    from timbal import Agent, Tool
    from timbal.core.test_model import TestModel
    from timbal.types.content import ToolUseContent
    from timbal.types.events import ApprovalEvent
    from timbal.types.message import Message

    SIDE_EFFECT_FILE = Path(sys.argv[1])

    def wire(amount: int) -> str:
        SIDE_EFFECT_FILE.write_text(f"wired {amount}")
        return f"wired {amount}"

    agent = Agent(
        name="cp_agent",
        model=TestModel(responses=[
            Message(
                role="assistant",
                content=[ToolUseContent(id="t1", name="wire", input={"amount": 42})],
                stop_reason="tool_use",
            ),
            "done",
        ]),
        tools=[Tool(name="wire", handler=wire, requires_approval=True)],
        tracing_provider=provider,
    )

    async def main():
        approval_id = None
        run_id = None
        async for event in agent(prompt="wire 42"):
            if isinstance(event, ApprovalEvent):
                approval_id = event.approval_id
                run_id = event.run_id
        sys.stdout.write(
            "<<<RESULT>>>"
            + json.dumps({"approval_id": approval_id, "run_id": run_id})
            + "<<<END>>>"
        )

    asyncio.run(main())
    """
).strip()


_RESUME_BODY = textwrap.dedent(
    """
    import asyncio, sys
    from pathlib import Path
    from timbal import Agent, Tool
    from timbal.core.test_model import TestModel
    from timbal.types.content import ToolUseContent
    from timbal.types.message import Message

    SIDE_EFFECT_FILE = Path(sys.argv[1])
    RUN_ID = sys.argv[2]
    APPROVAL_ID = sys.argv[3]

    def wire(amount: int) -> str:
        SIDE_EFFECT_FILE.write_text(f"wired {amount}")
        return f"wired {amount}"

    agent = Agent(
        name="cp_agent",
        model=TestModel(responses=[
            Message(
                role="assistant",
                content=[ToolUseContent(id="t1", name="wire", input={"amount": 42})],
                stop_reason="tool_use",
            ),
            "done",
        ]),
        tools=[Tool(name="wire", handler=wire, requires_approval=True)],
        tracing_provider=provider,
    )

    async def main():
        out = await agent(
            prompt="wire 42",
            parent_id=RUN_ID,
            approval_decisions={APPROVAL_ID: True},
        ).collect()
        sys.stdout.write("<<<RESULT>>>" + out.status.code + "<<<END>>>")

    asyncio.run(main())
    """
).strip()


def _extract_result(stdout: str) -> str:
    start = stdout.find("<<<RESULT>>>")
    end = stdout.find("<<<END>>>")
    assert start != -1 and end != -1, f"sentinel not found in stdout:\n{stdout}"
    return stdout[start + len("<<<RESULT>>>"):end]


class TestRealSubprocessResume:
    """Hard proof: gate happens in process A, resume happens in process B.

    Parametrised across both durable-store backends.
    """

    def test_gate_in_one_process_resume_in_another(self, backend, tmp_path):
        side_effect = tmp_path / "side_effect.txt"
        gate_script = backend.script_setup + "\n" + _GATE_BODY
        resume_script = backend.script_setup + "\n" + _RESUME_BODY

        gate = subprocess.run(  # noqa: S603
            [sys.executable, "-c", gate_script, str(side_effect)],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        payload = json.loads(_extract_result(gate.stdout))
        assert payload["approval_id"], gate.stderr
        assert payload["run_id"], gate.stderr
        assert backend.path.exists(), gate.stderr
        assert not side_effect.exists(), "tool must NOT have run before approval"

        resume = subprocess.run(  # noqa: S603
            [
                sys.executable, "-c", resume_script,
                str(side_effect),
                payload["run_id"], payload["approval_id"],
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert _extract_result(resume.stdout) == "success", resume.stderr
        assert side_effect.read_text() == "wired 42", "tool must run after cross-process resume"
