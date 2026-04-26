# ruff: noqa: ARG001, ARG005 — test handlers declare params for schema, do not consume them
"""Approval input redaction.

A tool gated on a sensitive input (password, api_key, PII, etc.) currently
leaks the raw input through three public surfaces:

    1. :class:`ApprovalEvent.input` — sent to UI / queue worker.
    2. :attr:`Span.input` (and ``span._input_dump``) — written to disk by
       every tracing provider, forwarded to OTel/Langfuse exporters.
    3. ``span.metadata['approval']['input']`` — the human-facing snapshot
       a UI uses to render "Approve this action?".

Redaction must:

- Apply only when the runnable goes through the approval gate. Non-gated
  runs are untouched (handler still gets the full input verbatim).
- Apply to **all** public surfaces above. Any one leak defeats the point.
- Survive a JSONL/SQLite round-trip — the persisted trace must contain
  only redacted values.
- **Not** affect the ``approval_id``: it stays derived from the unredacted
  validated input so that resume calls (which carry the full input)
  match the original gate's id.
- **Not** affect the handler invocation on resume — the handler always
  receives the full validated input.

Two ways to opt in:

- ``approval_redactor=callable`` — full control; receives validated input,
  returns a redacted dict.
- ``approval_redact_keys=["api_key", ...]`` — ergonomic shortcut; listed
  keys are masked with ``"***"``.

When both are set, the callable wins.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from timbal import Agent, Tool, Workflow
from timbal.core.test_model import TestModel
from timbal.state.tracing.providers.jsonl import JsonlTracingProvider
from timbal.types.content import ToolUseContent
from timbal.types.events import ApprovalEvent, OutputEvent
from timbal.types.message import Message


def _approval_event(events) -> ApprovalEvent:
    return next(e for e in events if isinstance(e, ApprovalEvent))


def _final_output(events) -> OutputEvent:
    return next(e for e in reversed(events) if isinstance(e, OutputEvent))


# ---------------------------------------------------------------------------
# Tool-level redaction
# ---------------------------------------------------------------------------


class TestToolRedactor:
    @pytest.mark.asyncio
    async def test_redactor_masks_event_and_span_input(self):
        seen: list[dict] = []

        def rotate(api_key: str, env: str) -> str:
            seen.append({"api_key": api_key, "env": env})
            return "rotated"

        tool = Tool(
            name="rotate",
            handler=rotate,
            requires_approval=True,
            approval_redactor=lambda inp: {**inp, "api_key": "***"},
        )

        events = [e async for e in tool(api_key="SECRET_KEY", env="prod")]
        approval = _approval_event(events)
        gated = _final_output(events)

        # 1. Event surface
        assert approval.input == {"api_key": "***", "env": "prod"}, (
            f"ApprovalEvent.input must be redacted, got {approval.input}"
        )

        # 2. Span surface: cancelled gate persists redacted input
        assert "SECRET_KEY" not in json.dumps(gated.input, default=str), (
            "OutputEvent.input must not leak the secret"
        )

        # 3. Resume passes full input, handler runs with full input
        approved = await tool(
            api_key="SECRET_KEY",
            env="prod",
            approval_decisions={approval.approval_id: True},
        ).collect()
        assert approved.status.code == "success"
        assert seen == [{"api_key": "SECRET_KEY", "env": "prod"}], (
            "handler must receive the full unredacted input"
        )

    @pytest.mark.asyncio
    async def test_redact_keys_shortcut_masks_listed_keys(self):
        def op(amount: int, api_key: str, password: str) -> str:
            return "ok"

        tool = Tool(
            name="op",
            handler=op,
            requires_approval=True,
            approval_redact_keys=["api_key", "password"],
        )

        events = [e async for e in tool(amount=42, api_key="K", password="P")]
        approval = _approval_event(events)

        assert approval.input == {"amount": 42, "api_key": "***", "password": "***"}

    @pytest.mark.asyncio
    async def test_redactor_does_not_change_approval_id(self):
        """approval_id must stay derived from the unredacted input so the
        resume call (which carries the full input) matches."""
        tool_unredacted = Tool(
            name="t",
            handler=lambda secret: "ok",
            requires_approval=True,
        )
        tool_redacted = Tool(
            name="t",
            handler=lambda secret: "ok",
            requires_approval=True,
            approval_redactor=lambda inp: {"secret": "***"},
        )

        events_a = [e async for e in tool_unredacted(secret="foo")]
        events_b = [e async for e in tool_redacted(secret="foo")]

        id_a = _approval_event(events_a).approval_id
        id_b = _approval_event(events_b).approval_id
        assert id_a == id_b, "redaction must not change approval_id"

    @pytest.mark.asyncio
    async def test_redactor_exception_falls_back_safely(self):
        """A buggy redactor must not crash the gate — fall back to a
        placeholder so the secret still doesn't leak."""

        def boom(_inp):
            raise RuntimeError("redactor blew up")

        tool = Tool(
            name="op",
            handler=lambda api_key: "ok",
            requires_approval=True,
            approval_redactor=boom,
        )

        events = [e async for e in tool(api_key="LEAK_ME")]
        approval = _approval_event(events)

        assert "LEAK_ME" not in json.dumps(approval.input, default=str), (
            "buggy redactor must NOT cause secret to leak"
        )

    @pytest.mark.asyncio
    async def test_redactor_returning_non_dict_falls_back_safely(self):
        tool = Tool(
            name="op",
            handler=lambda api_key: "ok",
            requires_approval=True,
            approval_redactor=lambda _inp: "not a dict",  # type: ignore[arg-type]
        )

        events = [e async for e in tool(api_key="LEAK_ME")]
        approval = _approval_event(events)

        assert "LEAK_ME" not in json.dumps(approval.input, default=str)

    @pytest.mark.asyncio
    async def test_redactor_takes_precedence_over_redact_keys(self):
        tool = Tool(
            name="op",
            handler=lambda a, b: "ok",
            requires_approval=True,
            approval_redactor=lambda inp: {"only": "callable_won"},
            approval_redact_keys=["a", "b"],
        )

        events = [e async for e in tool(a="x", b="y")]
        approval = _approval_event(events)
        assert approval.input == {"only": "callable_won"}

    @pytest.mark.asyncio
    async def test_no_redaction_without_requires_approval(self):
        """Redaction is a property of the gate. If the runnable doesn't
        gate, span.input keeps the full input (the redactor is inert)."""
        tool = Tool(
            name="op",
            handler=lambda x: "ok",
            requires_approval=False,
            approval_redactor=lambda inp: {"x": "***"},
        )

        out = await tool(x="visible").collect()
        assert out.status.code == "success"
        # No approval event was emitted; nothing to assert on event input.
        # Span input on an ungated run is left alone.
        assert "visible" in json.dumps(out.input, default=str)


# ---------------------------------------------------------------------------
# Span / metadata / pending_approvals surface
# ---------------------------------------------------------------------------


class TestSpanAndMetadataRedaction:
    @pytest.mark.asyncio
    async def test_span_metadata_approval_carries_redacted_input(self):
        """``span.metadata['approval']['input']`` is the snapshot a UI
        renders — it must NOT contain the secret."""
        tool = Tool(
            name="op",
            handler=lambda token: "ok",
            requires_approval=True,
            approval_redact_keys=["token"],
        )
        wf = Workflow(name="wf").step(tool)

        out = await wf(token="LEAK_ME").collect()

        # OutputEvent.metadata.pending_approvals is what `.collect()` exposes.
        pending = (out.metadata or {}).get("pending_approvals", [])
        assert pending, "expected at least one pending approval"
        for entry in pending:
            assert "LEAK_ME" not in json.dumps(entry, default=str), (
                f"pending_approvals leaked secret: {entry}"
            )
            assert entry["input"] == {"token": "***"}

    @pytest.mark.asyncio
    async def test_jsonl_persisted_span_input_is_redacted(self, tmp_path: Path):
        """Calling the gated tool directly (no workflow wrapper) so the
        only span that sees the secret is the tool's, and redaction must
        scrub that specific span before disk write.

        Wrapping in a workflow would persist the workflow's own span.input
        unredacted (the caller passed the secret to the workflow directly);
        that's a separate concern from gate-surface redaction."""
        traces = tmp_path / "traces.jsonl"
        provider = JsonlTracingProvider.configured(_path=traces)

        tool = Tool(
            name="op",
            handler=lambda api_key, env: "ok",
            requires_approval=True,
            approval_redact_keys=["api_key"],
            tracing_provider=provider,
        )

        await tool(api_key="LEAK_ME", env="prod").collect()

        raw = traces.read_text()
        assert raw, "expected JSONL to be written"
        assert "LEAK_ME" not in raw, (
            "secret leaked into the JSONL trace; redaction must apply to "
            "span.input (and span._input_dump) not just the in-memory event"
        )
        record = json.loads(raw.splitlines()[-1])
        op_span = next(s for s in record["spans"] if s["path"] == "op")
        assert op_span["input"] == {"api_key": "***", "env": "prod"}


# ---------------------------------------------------------------------------
# Agent-mediated tool calls
# ---------------------------------------------------------------------------


class TestAgentToolRedaction:
    @pytest.mark.asyncio
    async def test_agent_tool_call_redacts_event_and_span_but_handler_gets_full(self):
        seen: list[dict] = []

        def wire(amount: int, account: str, api_key: str) -> str:
            seen.append({"amount": amount, "account": account, "api_key": api_key})
            return "wired"

        tool = Tool(
            name="wire",
            handler=wire,
            requires_approval=True,
            approval_redact_keys=["api_key"],
        )
        agent = Agent(
            name="bank_agent",
            model=TestModel(
                responses=[
                    Message(
                        role="assistant",
                        content=[
                            ToolUseContent(
                                id="t1",
                                name="wire",
                                input={"amount": 500, "account": "A1", "api_key": "K"},
                            ),
                        ],
                        stop_reason="tool_use",
                    ),
                    "done",
                ]
            ),
            tools=[tool],
        )

        events = [e async for e in agent(prompt="please wire 500")]
        approval = _approval_event(events)
        assert approval.input == {"amount": 500, "account": "A1", "api_key": "***"}

        approved = await agent(
            prompt="please wire 500",
            approval_decisions={approval.approval_id: True},
        ).collect()
        assert approved.status.code == "success"
        assert seen == [{"amount": 500, "account": "A1", "api_key": "K"}]


# ---------------------------------------------------------------------------
# Cross-process: secret must not survive disk round-trip
# ---------------------------------------------------------------------------


class TestCrossProcessRedaction:
    @pytest.mark.asyncio
    async def test_secret_never_lands_on_disk_then_resumes_via_jsonl(self, tmp_path: Path):
        traces = tmp_path / "traces.jsonl"
        provider = JsonlTracingProvider.configured(_path=traces)

        seen: list[dict] = []

        def rotate(api_key: str, env: str) -> str:
            seen.append({"api_key": api_key, "env": env})
            return f"rotated {env}"

        tool = Tool(
            name="rotate",
            handler=rotate,
            requires_approval=True,
            approval_redact_keys=["api_key"],
        )
        agent = Agent(
            name="ops_agent",
            model=TestModel(
                responses=[
                    Message(
                        role="assistant",
                        content=[
                            ToolUseContent(
                                id="t1",
                                name="rotate",
                                input={"api_key": "TOPSECRET", "env": "prod"},
                            ),
                        ],
                        stop_reason="tool_use",
                    ),
                    "done",
                ]
            ),
            tools=[tool],
            tracing_provider=provider,
        )

        events1 = [e async for e in agent(prompt="rotate prod")]
        approval = _approval_event(events1)
        out1 = _final_output(events1)

        # Disk must not contain the secret in the gated tool's span.input.
        raw = traces.read_text()
        assert raw
        op_record = json.loads(
            next(line for line in raw.splitlines()
                 if json.loads(line)["run_id"] == out1.run_id)
        )
        op_span = next(s for s in op_record["spans"] if s["path"] == "ops_agent.rotate")
        assert op_span["input"] == {"api_key": "***", "env": "prod"}, (
            f"persisted span.input leaked secret: {op_span['input']}"
        )
        assert "TOPSECRET" not in json.dumps(op_span, default=str)

        # Resume from a separate parent_id; tool MUST run with full secret.
        out2 = await agent(
            prompt="rotate prod",
            parent_id=out1.run_id,
            approval_decisions={approval.approval_id: True},
        ).collect()
        assert out2.status.code == "success", out2.error
        assert seen == [{"api_key": "TOPSECRET", "env": "prod"}], (
            "handler must run with full unredacted input on cross-process resume"
        )
