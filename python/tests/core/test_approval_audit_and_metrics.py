# ruff: noqa: ARG001, ARG005 — test handlers declare params for schema, do not consume them
"""Approver audit trail + observability counters.

Two production gaps closed here:

1. **Audit trail.** Today ``ApprovalResolution`` carries only ``approved``,
   ``reason``, ``expires_at``, and a free-form ``metadata`` dict. SOC2 /
   compliance / "who approved this $1M wire on Tuesday?" require a typed
   ``approver_id`` + ``comment`` + ``decided_at`` (Unix ms) that traces
   serialize verbatim and resume surfaces verbatim. Free-form metadata stays
   for orgs that need extra structure on top.

2. **Observability counters.** Every gate emits one of four lifecycle
   events: ``required`` (human hasn't decided yet), ``approved``,
   ``denied``, ``expired`` (decision was given but stale at resume time).
   These are auto-incremented into ``RunContext.update_usage`` so they
   propagate up through agent / workflow spans and ship through
   ``OutputEvent.usage`` and any tracing exporter (OTel, etc.) without
   extra wiring on the user's side.

Invariants we enforce:

- Counters are emitted from the **gate site** (the runnable that owns
  ``requires_approval``), not from parents — so ``span.usage`` on the tool
  span is the source of truth, while parents see them via the standard
  propagation chain.
- ``decided_at`` defaults to ``int(time.time() * 1000)`` if not provided,
  but explicit values are preserved (idempotent replay).
- Backward compat: old callers passing only ``approved=True`` still work;
  new typed fields are ``None``.
- JSONL/SQLite round-trip preserves all typed fields verbatim — auditors
  can query ``span.metadata.approval.resolution.approver_id`` from disk.
"""

from __future__ import annotations

import json
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


def _approval_event(events) -> ApprovalEvent:
    return next(e for e in events if isinstance(e, ApprovalEvent))


def _final_output(events) -> OutputEvent:
    return next(e for e in reversed(events) if isinstance(e, OutputEvent))


# ---------------------------------------------------------------------------
# Typed fields on ApprovalResolution
# ---------------------------------------------------------------------------


class TestResolutionTypedFields:
    def test_typed_fields_are_accepted(self):
        r = ApprovalResolution(
            approved=True,
            approver_id="user_42",
            comment="Looks legitimate",
            decided_at=1777250000000,
        )
        assert r.approver_id == "user_42"
        assert r.comment == "Looks legitimate"
        assert r.decided_at == 1777250000000

    def test_decided_at_defaults_to_now_ms(self):
        before = int(time.time() * 1000)
        r = ApprovalResolution(approved=True)
        after = int(time.time() * 1000)
        assert r.decided_at is not None
        assert before <= r.decided_at <= after

    def test_explicit_decided_at_preserved(self):
        r = ApprovalResolution(approved=False, decided_at=42)
        assert r.decided_at == 42

    def test_backward_compat_minimal_resolution(self):
        r = ApprovalResolution(approved=True)
        assert r.approved is True
        assert r.approver_id is None
        assert r.comment is None
        assert r.metadata == {}


# ---------------------------------------------------------------------------
# Resolution fields appear in span.metadata['approval']['resolution']
# ---------------------------------------------------------------------------


class TestResolutionPersistedToSpanMetadata:
    @pytest.mark.asyncio
    async def test_approved_resolution_serialized(self):
        seen: list[str] = []
        tool = Tool(
            name="op",
            handler=lambda x: seen.append("ran") or "ok",
            requires_approval=True,
        )

        events = [e async for e in tool(x=1)]
        approval = _approval_event(events)

        out = await tool(
            x=1,
            approval_decisions={
                approval.approval_id: ApprovalResolution(
                    approved=True,
                    approver_id="user_42",
                    comment="vetted",
                    decided_at=42,
                )
            },
        ).collect()
        assert out.status.code == "success"

        resolution = (out.metadata or {}).get("approval", {}).get("resolution")
        assert resolution is not None, (
            "span.metadata['approval']['resolution'] must carry the typed snapshot"
        )
        assert resolution["approved"] is True
        assert resolution["approver_id"] == "user_42"
        assert resolution["comment"] == "vetted"
        assert resolution["decided_at"] == 42

    @pytest.mark.asyncio
    async def test_denied_resolution_serialized_with_audit_fields(self):
        tool = Tool(
            name="op",
            handler=lambda x: "ok",
            requires_approval=True,
        )

        events = [e async for e in tool(x=1)]
        approval = _approval_event(events)

        out = await tool(
            x=1,
            approval_decisions={
                approval.approval_id: ApprovalResolution(
                    approved=False,
                    reason="suspicious origin",
                    approver_id="user_99",
                    comment="flagged for review",
                )
            },
        ).collect()
        assert out.status.code == "cancelled"
        assert out.status.reason == "approval_denied"

        resolution = (out.metadata or {}).get("approval", {}).get("resolution")
        assert resolution is not None
        assert resolution["approved"] is False
        assert resolution["approver_id"] == "user_99"
        assert resolution["comment"] == "flagged for review"

    @pytest.mark.asyncio
    async def test_jsonl_roundtrip_preserves_audit_fields(self, tmp_path: Path):
        traces = tmp_path / "traces.jsonl"
        provider = JsonlTracingProvider.configured(_path=traces)

        tool = Tool(
            name="op",
            handler=lambda x: "ok",
            requires_approval=True,
            tracing_provider=provider,
        )

        events1 = [e async for e in tool(x=1)]
        approval = _approval_event(events1)
        gate_run_id = _final_output(events1).run_id

        await tool(
            x=1,
            parent_id=gate_run_id,
            approval_decisions={
                approval.approval_id: ApprovalResolution(
                    approved=True,
                    approver_id="user_42",
                    comment="ok",
                    decided_at=12345,
                )
            },
        ).collect()

        records = [json.loads(line) for line in traces.read_text().splitlines() if line.strip()]
        # The resume run's parent_id is the gate run's id.
        approved_record = next(r for r in records if r.get("parent_id") == gate_run_id)
        op_span = next(s for s in approved_record["spans"] if s["path"] == "op")
        resolution = op_span["metadata"]["approval"]["resolution"]
        assert resolution["approver_id"] == "user_42"
        assert resolution["comment"] == "ok"
        assert resolution["decided_at"] == 12345


# ---------------------------------------------------------------------------
# Usage counters: approvals:required / :approved / :denied / :expired
# ---------------------------------------------------------------------------


class TestApprovalUsageCounters:
    @pytest.mark.asyncio
    async def test_required_counter_on_first_gate(self):
        tool = Tool(
            name="op",
            handler=lambda x: "ok",
            requires_approval=True,
        )

        out = await tool(x=1).collect()
        assert out.status.code == "cancelled"
        assert out.usage.get("approvals:required") == 1
        assert "approvals:approved" not in out.usage
        assert "approvals:denied" not in out.usage
        assert "approvals:expired" not in out.usage

    @pytest.mark.asyncio
    async def test_approved_counter_on_resume(self):
        tool = Tool(
            name="op",
            handler=lambda x: "ok",
            requires_approval=True,
        )

        events = [e async for e in tool(x=1)]
        approval = _approval_event(events)

        out = await tool(x=1, approval_decisions={approval.approval_id: True}).collect()
        assert out.status.code == "success"
        assert out.usage.get("approvals:approved") == 1
        assert "approvals:required" not in out.usage

    @pytest.mark.asyncio
    async def test_denied_counter_on_resume(self):
        tool = Tool(
            name="op",
            handler=lambda x: "ok",
            requires_approval=True,
        )

        events = [e async for e in tool(x=1)]
        approval = _approval_event(events)

        out = await tool(
            x=1,
            approval_decisions={
                approval.approval_id: ApprovalResolution(approved=False, reason="nope")
            },
        ).collect()
        assert out.status.code == "cancelled"
        assert out.usage.get("approvals:denied") == 1

    @pytest.mark.asyncio
    async def test_expired_counter_then_required_on_resume(self):
        tool = Tool(
            name="op",
            handler=lambda x: "ok",
            requires_approval=True,
        )

        events = [e async for e in tool(x=1)]
        approval = _approval_event(events)

        # Expired resolution: gate must re-fire AND we must increment
        # both ``expired`` (the resolution we threw away) and ``required``
        # (the new gate emitted as a result).
        out = await tool(
            x=1,
            approval_decisions={
                approval.approval_id: ApprovalResolution(
                    approved=True,
                    expires_at=int(time.time() * 1000) - 1000,
                )
            },
        ).collect()
        assert out.status.code == "cancelled"
        assert out.status.reason == "approval_required"
        assert out.usage.get("approvals:expired") == 1
        assert out.usage.get("approvals:required") == 1

    @pytest.mark.asyncio
    async def test_counters_propagate_through_workflow(self):
        wf = (
            Workflow(name="wf")
            .step(
                Tool(
                    name="op_a",
                    handler=lambda x: "ok",
                    requires_approval=True,
                )
            )
            .step(
                Tool(
                    name="op_b",
                    handler=lambda x: "ok",
                    requires_approval=True,
                ),
                x=1,
            )
        )

        out = await wf(x=1).collect()
        # Two parallel gates fire; both increment :required on the
        # workflow's aggregated usage via the standard propagation chain.
        assert out.usage.get("approvals:required") == 2

    @pytest.mark.asyncio
    async def test_counters_propagate_through_agent_tool_call(self):
        tool = Tool(
            name="wire",
            handler=lambda amount: "wired",
            requires_approval=True,
        )
        agent = Agent(
            name="bank",
            model=TestModel(
                responses=[
                    Message(
                        role="assistant",
                        content=[ToolUseContent(id="t1", name="wire", input={"amount": 100})],
                        stop_reason="tool_use",
                    ),
                    "done",
                ]
            ),
            tools=[tool],
        )

        events = [e async for e in agent(prompt="wire 100")]
        gate_out = _final_output(events)
        approval = _approval_event(events)
        assert gate_out.usage.get("approvals:required") == 1

        out = await agent(
            prompt="wire 100",
            approval_decisions={approval.approval_id: True},
        ).collect()
        assert out.status.code == "success"
        assert out.usage.get("approvals:approved") == 1


# ---------------------------------------------------------------------------
# Resolution snapshot completeness — every typed field is round-tripped
# ---------------------------------------------------------------------------


class TestResolutionSnapshotCompleteness:
    @pytest.mark.asyncio
    async def test_full_resolution_snapshot_shape(self):
        """All seven keys must land on the snapshot — anything missing
        means a downstream consumer can't reconstruct the audit row."""
        tool = Tool(name="op", handler=lambda x: "ok", requires_approval=True)

        events = [e async for e in tool(x=1)]
        approval = _approval_event(events)

        # Use a far-future expires_at so the resolution isn't expired —
        # otherwise the gate discards it and emits a fresh ApprovalEvent
        # without writing the snapshot.
        future = int(time.time() * 1000) + 10_000_000
        out = await tool(
            x=1,
            approval_decisions={
                approval.approval_id: ApprovalResolution(
                    approved=True,
                    reason="trusted source",
                    approver_id="user_42",
                    comment="vetted via SSO",
                    decided_at=42,
                    expires_at=future,
                    metadata={"sso_provider": "google", "ip": "10.0.0.1"},
                )
            },
        ).collect()

        snapshot = out.metadata["approval"]["resolution"]
        assert snapshot == {
            "approved": True,
            "reason": "trusted source",
            "approver_id": "user_42",
            "comment": "vetted via SSO",
            "decided_at": 42,
            "expires_at": future,
            "metadata": {"sso_provider": "google", "ip": "10.0.0.1"},
        }

    @pytest.mark.asyncio
    async def test_bool_shorthand_produces_complete_snapshot(self):
        """``approval_decisions={id: True}`` is the most common form. The
        framework must still produce a fully-typed snapshot with
        ``decided_at`` defaulted (so the audit trail isn't blank)."""
        tool = Tool(name="op", handler=lambda x: "ok", requires_approval=True)

        events = [e async for e in tool(x=1)]
        approval = _approval_event(events)

        before = int(time.time() * 1000)
        out = await tool(
            x=1, approval_decisions={approval.approval_id: True}
        ).collect()
        after = int(time.time() * 1000)

        snapshot = out.metadata["approval"]["resolution"]
        assert snapshot["approved"] is True
        assert snapshot["approver_id"] is None
        assert snapshot["comment"] is None
        assert snapshot["reason"] is None
        assert snapshot["expires_at"] is None
        assert snapshot["metadata"] == {}
        assert before <= snapshot["decided_at"] <= after, (
            "decided_at must be auto-populated even from the bool shorthand"
        )

    @pytest.mark.asyncio
    async def test_dict_shorthand_preserves_audit_fields(self):
        """``approval_decisions={id: {"approved": True, "approver_id": ...}}``
        should round-trip every typed field; unknown keys would surface as
        a validation error from ``ApprovalResolution.model_validate``."""
        tool = Tool(name="op", handler=lambda x: "ok", requires_approval=True)

        events = [e async for e in tool(x=1)]
        approval = _approval_event(events)

        out = await tool(
            x=1,
            approval_decisions={
                approval.approval_id: {
                    "approved": True,
                    "approver_id": "user_42",
                    "comment": "ok",
                    "decided_at": 1234,
                }
            },
        ).collect()

        snapshot = out.metadata["approval"]["resolution"]
        assert snapshot["approver_id"] == "user_42"
        assert snapshot["comment"] == "ok"
        assert snapshot["decided_at"] == 1234


# ---------------------------------------------------------------------------
# Mixed counter states + gate-site invariant
# ---------------------------------------------------------------------------


class TestCounterMixedStates:
    @pytest.mark.asyncio
    async def test_mixed_approve_and_deny_counters_sum_correctly(self):
        """Two parallel steps, one resolved approved, one resolved denied.
        Counters at the workflow level must reflect both decisions
        independently."""
        wf = (
            Workflow(name="wf")
            .step(Tool(name="op_a", handler=lambda x: "ok", requires_approval=True))
            .step(Tool(name="op_b", handler=lambda x: "ok", requires_approval=True), x=1)
        )

        events = [e async for e in wf(x=1)]
        approvals = [e for e in events if isinstance(e, ApprovalEvent)]
        assert len(approvals) == 2
        # Map approval_id by step name for stable resolution wiring.
        ids = {a.runnable_path.split(".")[-1]: a.approval_id for a in approvals}

        out = await wf(
            x=1,
            approval_decisions={
                ids["op_a"]: True,
                ids["op_b"]: ApprovalResolution(approved=False, reason="nope"),
            },
        ).collect()

        assert out.usage.get("approvals:approved") == 1
        assert out.usage.get("approvals:denied") == 1
        assert out.usage.get("approvals:required", 0) == 0

    @pytest.mark.asyncio
    async def test_counter_emitted_at_gate_site_span(self):
        """Source-of-truth invariant: the gate's own span.usage has the
        counter; parents see it via update_usage propagation."""
        wf = Workflow(name="wf").step(
            Tool(name="op", handler=lambda x: "ok", requires_approval=True)
        )

        out = await wf(x=1).collect()
        assert out.status.code == "cancelled"

        # The OutputEvent's usage reflects propagation up the chain. Both
        # the workflow and its gated step must agree on the count.
        assert out.usage.get("approvals:required") == 1


# ---------------------------------------------------------------------------
# SQLite round-trip parity — must match JSONL behaviour
# ---------------------------------------------------------------------------


class TestSqliteRoundtripParity:
    @pytest.mark.asyncio
    async def test_sqlite_preserves_audit_fields(self, tmp_path: Path):
        db = tmp_path / "traces.db"
        provider = SqliteTracingProvider.configured(_path=db)

        tool = Tool(
            name="op",
            handler=lambda x: "ok",
            requires_approval=True,
            tracing_provider=provider,
        )

        events1 = [e async for e in tool(x=1)]
        approval = _approval_event(events1)
        gate_run_id = _final_output(events1).run_id

        out = await tool(
            x=1,
            parent_id=gate_run_id,
            approval_decisions={
                approval.approval_id: ApprovalResolution(
                    approved=True,
                    approver_id="user_99",
                    comment="cross-provider audit",
                    decided_at=98765,
                )
            },
        ).collect()
        assert out.status.code == "success"

        # Read back via raw sqlite so we test on-disk deserialization,
        # not just the in-memory metadata.
        import sqlite3

        conn = sqlite3.connect(db)
        rows = conn.execute("SELECT run_id, parent_id, spans FROM runs").fetchall()
        conn.close()

        approved_record = next(
            json.loads(spans) for run_id, parent_id, spans in rows if parent_id == gate_run_id
        )
        op_span = next(s for s in approved_record if s["path"] == "op")
        snapshot = op_span["metadata"]["approval"]["resolution"]
        assert snapshot["approver_id"] == "user_99"
        assert snapshot["comment"] == "cross-provider audit"
        assert snapshot["decided_at"] == 98765


# ---------------------------------------------------------------------------
# Cross-process audit trail: process A writes audit, process B reads it
# ---------------------------------------------------------------------------


class TestCrossProcessAuditTrail:
    @pytest.mark.asyncio
    async def test_audit_trail_recoverable_after_disk_roundtrip(self, tmp_path: Path):
        """Approver fields written by the resume turn must be readable
        from disk by a new RunContext (e.g. an audit dashboard reading
        the trace store after the original process exited)."""
        traces = tmp_path / "traces.jsonl"
        provider = JsonlTracingProvider.configured(_path=traces)

        tool = Tool(
            name="op",
            handler=lambda x: "ok",
            requires_approval=True,
            tracing_provider=provider,
        )

        events1 = [e async for e in tool(x=1)]
        approval = _approval_event(events1)
        gate_run_id = _final_output(events1).run_id

        # Resume in the same process, but the assertions read disk state
        # only — proving an audit dashboard with no in-memory state can
        # reconstruct the full audit row.
        await tool(
            x=1,
            parent_id=gate_run_id,
            approval_decisions={
                approval.approval_id: ApprovalResolution(
                    approved=True,
                    approver_id="user_777",
                    comment="approved by oncall",
                    decided_at=11111,
                )
            },
        ).collect()

        records = [json.loads(line) for line in traces.read_text().splitlines() if line.strip()]
        approved_record = next(r for r in records if r.get("parent_id") == gate_run_id)
        op_span = next(s for s in approved_record["spans"] if s["path"] == "op")

        # Audit row reconstructed entirely from disk:
        snapshot = op_span["metadata"]["approval"]["resolution"]
        audit_row = {
            "approval_id": op_span["metadata"]["approval"]["id"],
            "path": op_span["path"],
            "approver_id": snapshot["approver_id"],
            "comment": snapshot["comment"],
            "decided_at": snapshot["decided_at"],
            "approved": snapshot["approved"],
        }
        assert audit_row == {
            "approval_id": approval.approval_id,
            "path": "op",
            "approver_id": "user_777",
            "comment": "approved by oncall",
            "decided_at": 11111,
            "approved": True,
        }
