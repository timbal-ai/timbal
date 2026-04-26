"""Regression tests for the human-in-the-loop approval gate.

Each test in this file targets a *specific* known bug or API gap in the
current HITL implementation. They are written to **fail** on the current
code and pass once the corresponding fix is applied. Keep them isolated:
one bug per test, with assertion messages that explain the failure.

Order roughly matches severity — critical correctness bugs first, then
UX/observability gaps.
"""

from __future__ import annotations

import asyncio

import pytest
from timbal import Agent, Tool, Workflow
from timbal.core.test_model import TestModel
from timbal.state.tracing.providers.in_memory import InMemoryTracingProvider
from timbal.types.content import ToolUseContent
from timbal.types.events import ApprovalEvent, OutputEvent
from timbal.types.message import Message

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _approval_event(events) -> ApprovalEvent:
    return next(e for e in events if isinstance(e, ApprovalEvent))


def _approval_events(events) -> list[ApprovalEvent]:
    return [e for e in events if isinstance(e, ApprovalEvent)]


def _final_output(events) -> OutputEvent:
    return next(e for e in reversed(events) if isinstance(e, OutputEvent))


def _stored_span(run_id: str):
    """Return the root span for ``run_id`` from the in-memory provider."""
    trace = InMemoryTracingProvider._storage[run_id]
    # Root span has parent_call_id None.
    for span in trace.values():
        if span.parent_call_id is None:
            return span
    raise AssertionError(f"No root span for run_id={run_id!r}")


def _span_at_path(run_id: str, path: str):
    trace = InMemoryTracingProvider._storage[run_id]
    matches = [s for s in trace.values() if s.path == path]
    assert matches, f"No span at path={path!r} for run_id={run_id!r}"
    assert len(matches) == 1, f"Multiple spans at path={path!r}: {matches}"
    return matches[0]


# ---------------------------------------------------------------------------
# 1. Critical: breaking the stream after ApprovalEvent corrupts stored status
# ---------------------------------------------------------------------------


class TestStreamBreakPreservesApprovalStatus:
    """Real consumers `break` after seeing ApprovalEvent. The stored span
    must still report status=cancelled/approval_required so consumers and
    OTel dashboards can identify pending approvals via the trace alone."""

    @pytest.mark.asyncio
    async def test_tool_break_on_approval_event(self):
        def dangerous(x: int) -> int:
            return x

        tool = Tool(name="dangerous", handler=dangerous, requires_approval=True)

        run_id: str | None = None
        async for event in tool(x=1):
            if event.type == "START":
                run_id = event.run_id
            if isinstance(event, ApprovalEvent):
                break

        assert run_id is not None
        # Give the generator-finalisation path time to run.
        await asyncio.sleep(0.05)

        span = _stored_span(run_id)
        assert span.status.code == "cancelled", (
            f"Expected stored span code 'cancelled', got {span.status.code!r}"
        )
        assert span.status.reason == "approval_required", (
            f"Expected stored span reason 'approval_required' after stream break, "
            f"got {span.status.reason!r}. Status is set *after* the yield, so "
            f"GeneratorExit clobbers it with 'interrupted'."
        )
        # The metadata side already records approval info correctly — the bug
        # is purely in span.status.
        assert span.metadata["approval"]["required"] is True

    @pytest.mark.asyncio
    async def test_agent_break_on_approval_event(self):
        """Same bug surfaces at agent level when a tool gates."""
        tool = Tool(
            name="risky",
            handler=lambda x: f"did {x}",
            requires_approval=True,
        )
        agent = Agent(
            name="agent",
            model=TestModel(
                responses=[
                    Message(
                        role="assistant",
                        content=[ToolUseContent(id="c1", name="risky", input={"x": 1})],
                        stop_reason="tool_use",
                    ),
                ],
            ),
            tools=[tool],
        )

        run_id: str | None = None
        async for event in agent(prompt="run"):
            if event.type == "START" and event.path == "agent":
                run_id = event.run_id
            if isinstance(event, ApprovalEvent):
                break

        assert run_id is not None
        await asyncio.sleep(0.05)

        span = _stored_span(run_id)
        assert span.status.reason == "approval_required", (
            f"Agent root span lost approval_required status on stream break: "
            f"{span.status!r}"
        )

    @pytest.mark.asyncio
    async def test_workflow_break_propagates_approval_status_to_step_span(self):
        """Workflow with one gating step: break on the step's ApprovalEvent and
        verify the *step* span (not just root) records approval_required."""
        wf = Workflow(name="wf").step(
            Tool(name="step_a", handler=lambda x: x, requires_approval=True),
            x=1,
        )

        run_id: str | None = None
        async for event in wf():
            if event.type == "START" and event.path == "wf":
                run_id = event.run_id
            if isinstance(event, ApprovalEvent):
                break

        assert run_id is not None
        await asyncio.sleep(0.05)

        step_span = _span_at_path(run_id, "wf.step_a")
        assert step_span.status.reason == "approval_required", (
            f"Step span at wf.step_a lost approval_required after break: "
            f"{step_span.status!r}"
        )

    @pytest.mark.asyncio
    async def test_break_does_not_clear_approval_metadata(self):
        """Even on break, span.metadata['approval'] must record the id+prompt
        so a server can reconstruct the approval from the trace alone."""
        tool = Tool(
            name="t",
            handler=lambda x: x,
            requires_approval=True,
            approval_prompt="ok?",
        )
        approval_id_seen = None
        run_id = None
        async for event in tool(x=42):
            if event.type == "START":
                run_id = event.run_id
            if isinstance(event, ApprovalEvent):
                approval_id_seen = event.approval_id
                break

        assert approval_id_seen is not None
        await asyncio.sleep(0.05)

        span = _stored_span(run_id)
        assert span.metadata["approval"]["id"] == approval_id_seen
        assert span.metadata["approval"]["prompt"] == "ok?"
        assert span.status.code == "cancelled"
        assert span.status.reason == "approval_required"


# ---------------------------------------------------------------------------
# 2. Critical: multi-turn agent loses memory + tool never executes on resume
# ---------------------------------------------------------------------------


class TestMultiTurnApprovalResume:
    """Resuming an agent with parent_id + approval_decisions must run the
    gated tool *with* the prior conversation history intact. Currently the
    early-return in resolve_memory drops all memory and the LLM may never
    re-emit the same tool call."""

    @pytest.mark.asyncio
    async def test_resume_executes_gated_tool_after_multi_turn_history(self):
        calls: list[int] = []

        def risky(x: int) -> str:
            calls.append(x)
            return f"did {x}"

        tool = Tool(name="risky", handler=risky, requires_approval=True)

        agent = Agent(
            name="agent",
            model=TestModel(
                responses=[
                    # Turn 1: plain reply (sets up history).
                    "hello back",
                    # Turn 2: emit tool_use that will gate.
                    Message(
                        role="assistant",
                        content=[
                            ToolUseContent(id="c1", name="risky", input={"x": 7})
                        ],
                        stop_reason="tool_use",
                    ),
                    # Turn 3 resume: any non-tool reply once the tool has run.
                    "ok done",
                ],
            ),
            tools=[tool],
        )

        out1 = await agent(prompt="hi").collect()
        assert out1.status.code == "success"

        events2 = [e async for e in agent(prompt="run risky", parent_id=out1.run_id)]
        approval = _approval_event(events2)
        out2 = _final_output(events2)
        assert out2.status.reason == "approval_required"

        out3 = await agent(
            prompt="run risky",
            parent_id=out2.run_id,
            approval_decisions={approval.approval_id: True},
        ).collect()

        assert calls == [7], (
            f"Gated tool must run on resume. Got calls={calls}. The agent "
            f"loses memory on resume because resolve_memory short-circuits "
            f"when prev reason=='approval_required', so the LLM never "
            f"re-emits the tool_use."
        )

        # And history must still be present in the resumed memory so the LLM
        # has the original conversation context.
        span3 = _stored_span(out3.run_id)
        roles = [m.role for m in span3.memory]
        assert roles[:2] == ["user", "assistant"], (
            f"Expected resumed memory to contain prior turn(s), got roles={roles}."
        )

    @pytest.mark.asyncio
    async def test_resume_preserves_first_turn_user_prompt_in_memory(self):
        """Stronger assertion: the actual *user prompt* from turn 1 must
        survive into turn 3's memory after approval-resume.

        Using the user prompt as the canary makes this test robust against
        TestModel's response-cycling — only memory loading can put it back.
        """
        def risky(x: int) -> str:
            return f"did {x}"

        tool = Tool(name="risky", handler=risky, requires_approval=True)
        agent = Agent(
            name="agent",
            model=TestModel(
                responses=[
                    "ack",
                    Message(
                        role="assistant",
                        content=[ToolUseContent(id="c1", name="risky", input={"x": 1})],
                        stop_reason="tool_use",
                    ),
                    "all done",
                ],
            ),
            tools=[tool],
        )

        UNIQUE_PROMPT = "ALPHA-UNIQUE-USER-PROMPT-12345"
        out1 = await agent(prompt=UNIQUE_PROMPT).collect()
        events2 = [e async for e in agent(prompt="now risky", parent_id=out1.run_id)]
        approval = _approval_event(events2)
        out2 = _final_output(events2)

        out3 = await agent(
            prompt="now risky",
            parent_id=out2.run_id,
            approval_decisions={approval.approval_id: True},
        ).collect()

        span3 = _stored_span(out3.run_id)
        joined = " ".join(m.collect_text() for m in span3.memory)
        assert UNIQUE_PROMPT in joined, (
            f"Turn 1 user prompt missing from resumed memory — memory was "
            f"reset to just the turn-3 prompt. Memory contents: {joined!r}"
        )

    @pytest.mark.asyncio
    async def test_resume_passes_full_history_to_llm(self):
        """The strongest possible assertion of this bug: inspect the LLM
        span on the resumed turn and verify the messages it received contain
        prior conversation. Today the LLM input on resume is just the new
        prompt — no history.
        """
        def risky(x: int) -> str:
            return f"did {x}"

        tool = Tool(name="risky", handler=risky, requires_approval=True)
        agent = Agent(
            name="agent",
            model=TestModel(
                responses=[
                    "ack",
                    Message(
                        role="assistant",
                        content=[ToolUseContent(id="c1", name="risky", input={"x": 1})],
                        stop_reason="tool_use",
                    ),
                    "done",
                ],
            ),
            tools=[tool],
        )

        UNIQUE = "BETA-PROMPT-CANARY-XYZ"
        out1 = await agent(prompt=UNIQUE).collect()
        events2 = [e async for e in agent(prompt="run risky", parent_id=out1.run_id)]
        approval = _approval_event(events2)
        out2 = _final_output(events2)

        out3 = await agent(
            prompt="run risky",
            parent_id=out2.run_id,
            approval_decisions={approval.approval_id: True},
        ).collect()

        # Find the *first* LLM call in turn 3's trace.
        trace = InMemoryTracingProvider._storage[out3.run_id]
        llm_spans = sorted(
            (s for s in trace.values() if s.path == "agent.llm"),
            key=lambda s: s.t0,
        )
        assert llm_spans, "Expected at least one LLM call on resume."
        first_llm = llm_spans[0]
        msgs = first_llm.input["messages"]
        flat = repr(msgs)
        assert UNIQUE in flat, (
            f"Resumed LLM call must receive prior history; got messages "
            f"that don't contain the turn-1 user prompt. Messages: {msgs}"
        )


# ---------------------------------------------------------------------------
# 3. pre_hook fires on every gated attempt
# ---------------------------------------------------------------------------


class TestPreHookSemanticsUnderApproval:
    """pre_hook is supposed to be a stable place for cross-cutting concerns
    (auth, rate limiting, billing). Today it runs on the *gated* attempt as
    well as the resumed attempt, so external side effects double-fire for
    every approval cycle."""

    @pytest.mark.asyncio
    async def test_pre_hook_runs_only_once_per_logical_call(self):
        hook_calls: list[int] = []

        def pre() -> None:
            hook_calls.append(1)

        def handler(x: int) -> int:
            return x

        tool = Tool(
            name="t",
            handler=handler,
            requires_approval=True,
            pre_hook=pre,
        )

        events = [e async for e in tool(x=1)]
        approval = _approval_event(events)
        gated_count = len(hook_calls)

        await tool(x=1, approval_decisions={approval.approval_id: True}).collect()

        assert gated_count == 0, (
            f"pre_hook should not run on the gated attempt (no real work "
            f"happens), got {gated_count} call(s)."
        )
        assert len(hook_calls) == 1, (
            f"pre_hook should run exactly once across the gate→resume cycle; "
            f"got {len(hook_calls)} total calls."
        )

    @pytest.mark.asyncio
    async def test_pre_hook_does_not_run_when_decision_denies(self):
        """If the policy denies, pre_hook should not have side-effected. The
        gated path must short-circuit before pre_hook."""
        hook_calls: list[int] = []

        def pre() -> None:
            hook_calls.append(1)

        tool = Tool(
            name="t",
            handler=lambda x: x,
            requires_approval=True,
            pre_hook=pre,
        )

        events = [e async for e in tool(x=1)]
        approval = _approval_event(events)

        await tool(x=1, approval_decisions={approval.approval_id: False}).collect()

        assert hook_calls == [], (
            f"pre_hook ran on a denied gate cycle — that's an external side "
            f"effect for an action the user explicitly rejected. Got "
            f"{len(hook_calls)} call(s)."
        )

    @pytest.mark.asyncio
    async def test_pre_hook_runs_once_for_agent_gated_tool(self):
        """pre_hook on a tool inside an agent should not double-fire when the
        tool gates and the agent re-runs to resume it."""
        hook_calls: list[int] = []

        def pre() -> None:
            hook_calls.append(1)

        tool = Tool(
            name="risky",
            handler=lambda x: f"did {x}",
            requires_approval=True,
            pre_hook=pre,
        )
        agent = Agent(
            name="agent",
            model=TestModel(
                responses=[
                    Message(
                        role="assistant",
                        content=[ToolUseContent(id="c1", name="risky", input={"x": 1})],
                        stop_reason="tool_use",
                    ),
                    "done",
                ],
            ),
            tools=[tool],
        )

        events = [e async for e in agent(prompt="go")]
        approval = _approval_event(events)

        await agent(
            prompt="go",
            approval_decisions={approval.approval_id: True},
        ).collect()

        assert len(hook_calls) == 1, (
            f"Tool pre_hook fired {len(hook_calls)} times across the "
            f"agent→tool gate→resume cycle. Should be 1."
        )


# ---------------------------------------------------------------------------
# 4. .collect() only surfaces ONE pending approval in concurrent scenarios
# ---------------------------------------------------------------------------


class TestCollectExposesAllPendingApprovals:
    """A workflow with multiple parallel gating steps emits multiple
    ApprovalEvents on the stream, but ``.collect()`` consumers see only
    one approval_id in the OutputEvent.output dict. They have no way to
    discover the second pending approval without re-running the workflow."""

    @pytest.mark.asyncio
    async def test_workflow_collect_lists_all_pending_approvals(self):
        wf = Workflow(name="wf").step(
            Tool(name="step_a", handler=lambda: "a", requires_approval=True)
        ).step(
            Tool(name="step_b", handler=lambda: "b", requires_approval=True)
        )

        result = await wf().collect()
        assert result.status.reason == "approval_required"

        pending = None
        # Accept either output-level or metadata-level surfacing — whichever
        # the maintainers prefer. Both currently absent.
        if isinstance(result.output, dict) and "pending_approvals" in result.output:
            pending = result.output["pending_approvals"]
        elif "pending_approvals" in (result.metadata or {}):
            pending = result.metadata["pending_approvals"]

        assert pending is not None, (
            "OutputEvent must expose all pending approvals when "
            "status.reason == 'approval_required'. Today only the first "
            "approval shows up in output, hidden inside a flat dict."
        )
        assert len(pending) == 2, (
            f"Expected 2 pending approvals (step_a, step_b), got {len(pending)}: {pending}"
        )
        ids = {entry["approval_id"] if isinstance(entry, dict) else entry.approval_id for entry in pending}
        assert len(ids) == 2, f"Expected 2 distinct approval_ids, got {ids}"

    @pytest.mark.asyncio
    async def test_agent_collect_lists_all_concurrent_tool_approvals(self):
        """Agent multiplexes 2 tool calls in one LLM turn, both gate. The
        OutputEvent from .collect() should expose both approval_ids; today
        only the first is reachable from the result."""
        tool_a = Tool(name="tool_a", handler=lambda: "a", requires_approval=True)
        tool_b = Tool(name="tool_b", handler=lambda: "b", requires_approval=True)
        agent = Agent(
            name="agent",
            model=TestModel(
                responses=[
                    Message(
                        role="assistant",
                        content=[
                            ToolUseContent(id="c1", name="tool_a", input={}),
                            ToolUseContent(id="c2", name="tool_b", input={}),
                        ],
                        stop_reason="tool_use",
                    ),
                ],
            ),
            tools=[tool_a, tool_b],
        )

        result = await agent(prompt="parallel").collect()
        assert result.status.reason == "approval_required"

        pending = None
        if isinstance(result.output, dict) and "pending_approvals" in result.output:
            pending = result.output["pending_approvals"]
        elif "pending_approvals" in (result.metadata or {}):
            pending = result.metadata["pending_approvals"]

        assert pending is not None and len(pending) == 2, (
            f"Agent .collect() must expose every concurrent pending approval; "
            f"got pending={pending}"
        )


# ---------------------------------------------------------------------------
# 5. ApprovalEvent has no timestamp — SLA tracking impossible from the event
# ---------------------------------------------------------------------------


class TestApprovalEventMetadata:
    @pytest.mark.asyncio
    async def test_approval_event_has_request_timestamp(self):
        tool = Tool(name="t", handler=lambda x: x, requires_approval=True)

        events = [e async for e in tool(x=1)]
        approval = _approval_event(events)

        timestamp = getattr(approval, "t0", None) or getattr(approval, "requested_at", None)
        assert timestamp is not None, (
            "ApprovalEvent should carry a request timestamp so consumers "
            "can drive SLA timers without joining against the trace store."
        )
        assert isinstance(timestamp, int) and timestamp > 0

    @pytest.mark.asyncio
    async def test_concurrent_approval_event_timestamps_are_ordered(self):
        """Each ApprovalEvent should carry its *own* timestamp so a UI can
        sort/group them. Without it they're indistinguishable in time."""
        wf = (
            Workflow(name="wf")
            .step(Tool(name="step_a", handler=lambda: "a", requires_approval=True))
            .step(Tool(name="step_b", handler=lambda: "b", requires_approval=True))
        )

        events = [e async for e in wf()]
        approvals = _approval_events(events)
        assert len(approvals) == 2

        timestamps = [
            getattr(a, "t0", None) or getattr(a, "requested_at", None) for a in approvals
        ]
        assert all(t is not None for t in timestamps), (
            f"All ApprovalEvents should carry timestamps; got {timestamps}"
        )


# ---------------------------------------------------------------------------
# 6. Unrecognised approval_decisions are silently dropped
# ---------------------------------------------------------------------------


class TestUnknownApprovalDecisionsAreSurfaced:
    """If a caller passes an approval_id that doesn't match any gate in the
    run (typo, stale id replayed against a new trace, mismatched input
    after a retry), the run still cancels with a fresh ApprovalEvent and
    the bad id is silently swallowed. Make this debuggable."""

    @pytest.mark.asyncio
    async def test_unknown_approval_id_logs_warning(self, capfd):
        """A decision dict with an id that doesn't match any gate should
        emit a warning so users can detect typos / stale ids.

        The codebase emits via structlog (stderr); capfd grabs both streams.
        """
        tool = Tool(name="t", handler=lambda x: x, requires_approval=True)

        events = [e async for e in tool(x=1, approval_decisions={"definitely-not-a-real-id": True})]

        captured = capfd.readouterr()
        log_text = captured.out + captured.err

        approval = _approval_event(events)
        assert approval.approval_id != "definitely-not-a-real-id"

        assert (
            "definitely-not-a-real-id" in log_text
            or "Unrecognized approval_decisions" in log_text
        ), (
            "An approval_decisions entry that matched no gate should produce "
            "a WARNING. Otherwise typos/stale ids fail silently. "
            f"Captured output:\n{log_text!r}"
        )

    @pytest.mark.asyncio
    async def test_unknown_id_in_mixed_decisions_warns_only_for_unused(self, capfd):
        """One valid + one invalid decision → only the unused id should warn."""
        tool = Tool(name="t", handler=lambda x: x, requires_approval=True)

        events = [e async for e in tool(x=1)]
        approval = _approval_event(events)
        capfd.readouterr()  # drain the gating call's logs so we only assert the resume call

        result = await tool(
            x=1,
            approval_decisions={
                approval.approval_id: True,
                "stale-other-id": True,
            },
        ).collect()

        captured = capfd.readouterr()
        log_text = captured.out + captured.err

        assert result.status.code == "success"
        assert "stale-other-id" in log_text, (
            "Unused decision ('stale-other-id') should produce a WARNING "
            "even when other decisions in the same call were valid. "
            f"Captured output:\n{log_text!r}"
        )
        # And conversely the matched id must not be reported as unused.
        unused_lines = [line for line in log_text.splitlines() if "unused_approval_ids" in line]
        assert all(approval.approval_id not in line for line in unused_lines), (
            f"Matched approval_id ({approval.approval_id}) was wrongly "
            f"flagged as unused. Captured output:\n{log_text!r}"
        )
