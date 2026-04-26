# ruff: noqa: ARG001, ARG005 — test handlers declare params for schema, do not consume them
"""Double-resume protection for approval gates.

The dangerous race is:

    1. process A hits approval_required and persists the parent trace
    2. two queue workers both receive the same approval_id
    3. both call the runnable with parent_id=<gate_run_id> and approved=True

Without a durable claim on ``(parent_id, approval_id)``, both workers execute
the gated handler. For irreversible tools (charge card, delete data, deploy)
that defeats the point of HITL.

These tests pin the provider contract for durable tracing providers:

- first resume claims the approval and executes
- later/concurrent duplicate resumes stop before the handler
- claim works through agent-mediated pending tool execution, not just direct tools
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import pytest
from timbal import Agent, Tool
from timbal.core.test_model import TestModel
from timbal.state.tracing.providers.jsonl import JsonlTracingProvider
from timbal.state.tracing.providers.sqlite import SqliteTracingProvider
from timbal.types.content import ToolUseContent
from timbal.types.events import ApprovalEvent, OutputEvent
from timbal.types.message import Message


class _Backend:
    def __init__(self, name: str, path: Path, provider):
        self.name = name
        self.path = path
        self.provider = provider


@pytest.fixture(params=["jsonl", "sqlite"])
def backend(request, tmp_path) -> _Backend:
    if request.param == "jsonl":
        path = tmp_path / "traces.jsonl"
        return _Backend("jsonl", path, JsonlTracingProvider.configured(_path=path))
    path = tmp_path / "traces.db"
    return _Backend("sqlite", path, SqliteTracingProvider.configured(_path=path))


def _approval_event(events) -> ApprovalEvent:
    return next(e for e in events if isinstance(e, ApprovalEvent))


def _final_output(events) -> OutputEvent:
    return next(e for e in reversed(events) if isinstance(e, OutputEvent))


def _claim_count(backend: _Backend) -> int:
    if backend.name == "jsonl":
        claims_path = backend.path.with_suffix(backend.path.suffix + ".approval_claims.json")
        if not claims_path.exists():
            return 0
        return len(json.loads(claims_path.read_text() or "{}"))

    conn = sqlite3.connect(backend.path)
    try:
        return conn.execute("SELECT COUNT(*) FROM approval_claims").fetchone()[0]
    finally:
        conn.close()


class TestDirectToolDoubleResume:
    @pytest.mark.asyncio
    async def test_second_resume_does_not_execute_handler(self, backend):
        calls: list[int] = []

        def charge(amount: int) -> str:
            calls.append(amount)
            return f"charged {amount}"

        tool = Tool(
            name="charge",
            handler=charge,
            requires_approval=True,
            tracing_provider=backend.provider,
        )

        events = [e async for e in tool(amount=100)]
        approval = _approval_event(events)
        parent_id = _final_output(events).run_id

        first = await tool(
            amount=100,
            parent_id=parent_id,
            approval_decisions={approval.approval_id: True},
        ).collect()
        second = await tool(
            amount=100,
            parent_id=parent_id,
            approval_decisions={approval.approval_id: True},
        ).collect()

        assert first.status.code == "success"
        assert second.status.code == "cancelled"
        assert second.status.reason == "approval_already_claimed"
        assert calls == [100]
        assert _claim_count(backend) == 1

    @pytest.mark.asyncio
    async def test_concurrent_resumes_only_execute_once(self, backend):
        calls: list[int] = []

        async def charge(amount: int) -> str:
            calls.append(amount)
            await asyncio.sleep(0.05)
            return f"charged {amount}"

        tool = Tool(
            name="charge",
            handler=charge,
            requires_approval=True,
            tracing_provider=backend.provider,
        )

        events = [e async for e in tool(amount=200)]
        approval = _approval_event(events)
        parent_id = _final_output(events).run_id

        async def resume_once():
            return await tool(
                amount=200,
                parent_id=parent_id,
                approval_decisions={approval.approval_id: True},
            ).collect()

        out_a, out_b = await asyncio.gather(resume_once(), resume_once())
        reasons = sorted((out_a.status.reason, out_b.status.reason), key=lambda x: x or "")
        codes = sorted((out_a.status.code, out_b.status.code))

        assert calls == [200]
        assert codes == ["cancelled", "success"]
        assert reasons == [None, "approval_already_claimed"]
        assert _claim_count(backend) == 1


class TestAgentToolDoubleResume:
    @pytest.mark.asyncio
    async def test_duplicate_agent_resume_does_not_execute_tool_twice(self, backend):
        calls: list[int] = []

        def charge(amount: int) -> str:
            calls.append(amount)
            return f"charged {amount}"

        tool = Tool(name="charge", handler=charge, requires_approval=True)
        agent = Agent(
            name="agent",
            model=TestModel(
                responses=[
                    Message(
                        role="assistant",
                        content=[ToolUseContent(id="t1", name="charge", input={"amount": 300})],
                        stop_reason="tool_use",
                    ),
                    "done",
                ]
            ),
            tools=[tool],
            tracing_provider=backend.provider,
        )

        events = [e async for e in agent(prompt="charge 300")]
        approval = _approval_event(events)
        parent_id = _final_output(events).run_id

        first = await agent(
            prompt="charge 300",
            parent_id=parent_id,
            approval_decisions={approval.approval_id: True},
        ).collect()
        second = await agent(
            prompt="charge 300",
            parent_id=parent_id,
            approval_decisions={approval.approval_id: True},
        ).collect()

        assert first.status.code == "success"
        assert second.status.code == "success"
        assert calls == [300]
        assert _claim_count(backend) == 1
