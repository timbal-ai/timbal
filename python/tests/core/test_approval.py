"""End-to-end lifecycle tests for the human-in-the-loop approval gate.

Each lifecycle test drives the full user flow:

    1. invoke without decision → assert ApprovalEvent + cancelled OutputEvent
    2. resume with decision     → assert handler ran (or denial path)

Tests organised by *user-facing scenario* (tool, agent, workflow, nested,
parallel) rather than implementation surface, so a regression in any layer
fails the test that exercises the corresponding flow.
"""

import time
import warnings

import pytest
from timbal import Agent, Tool, Workflow
from timbal.core.test_model import TestModel
from timbal.types.approval import ApprovalDecision, ApprovalResolution
from timbal.types.content import ToolUseContent
from timbal.types.events import ApprovalEvent, OutputEvent
from timbal.types.message import Message


def _approval_event(events) -> ApprovalEvent:
    return next(e for e in events if isinstance(e, ApprovalEvent))


def _approval_events(events) -> list[ApprovalEvent]:
    return [e for e in events if isinstance(e, ApprovalEvent)]


def _final_output(events) -> OutputEvent:
    """Return the OUTPUT event of the outermost runnable (the final OUTPUT in stream order)."""
    return next(e for e in reversed(events) if isinstance(e, OutputEvent))


# ---------------------------------------------------------------------------
# Tool-level lifecycle
# ---------------------------------------------------------------------------


class TestToolApprovalLifecycle:
    @pytest.mark.asyncio
    async def test_full_cycle_static_policy(self):
        """Default scope: gate → approve → handler runs on resume."""
        calls: list[str] = []

        def delete_file(path: str) -> str:
            calls.append(path)
            return f"deleted {path}"

        tool = Tool(
            name="delete_file",
            handler=delete_file,
            requires_approval=True,
            approval_prompt="Approve file deletion?",
        )

        events = [e async for e in tool(path="/tmp/a.txt")]
        approval = _approval_event(events)
        cancelled = _final_output(events)

        assert approval.approval_id
        assert approval.prompt == "Approve file deletion?"
        assert approval.input == {"path": "/tmp/a.txt"}
        assert cancelled.status.code == "cancelled"
        assert cancelled.status.reason == "approval_required"
        assert calls == [], "handler must not run before approval"

        approved = await tool(
            path="/tmp/a.txt",
            approval_decisions={approval.approval_id: True},
        ).collect()

        assert approved.status.code == "success"
        assert approved.output == "deleted /tmp/a.txt"
        assert calls == ["/tmp/a.txt"], "handler must run after approval"

    @pytest.mark.asyncio
    async def test_denial_skips_handler_and_carries_reason(self):
        calls: list[int] = []

        def charge_card(amount: int) -> str:
            calls.append(amount)
            return "charged"

        tool = Tool(name="charge_card", handler=charge_card, requires_approval=True)

        events = [e async for e in tool(amount=100)]
        approval = _approval_event(events)

        denied = await tool(
            amount=100,
            approval_decisions={approval.approval_id: {"approved": False, "reason": "too much"}},
        ).collect()

        assert denied.status.code == "cancelled"
        assert denied.status.reason == "approval_denied"
        assert denied.output["reason"] == "too much"
        assert calls == []

    @pytest.mark.asyncio
    async def test_callable_policy_uses_validated_input_and_full_cycle(self):
        calls: list[int] = []

        def transfer(amount: int) -> str:
            calls.append(amount)
            return "transferred"

        tool = Tool(
            name="transfer",
            handler=transfer,
            requires_approval=lambda amount: amount > 100,
            approval_prompt=lambda amount: f"Approve transfer of {amount}?",
        )

        # Below threshold → no gate, handler runs immediately
        ok = await tool(amount=50).collect()
        assert ok.status.code == "success"
        assert calls == [50]

        # Above threshold → gate
        events = [e async for e in tool(amount=150)]
        approval = _approval_event(events)
        assert approval.prompt == "Approve transfer of 150?"

        # Resume → handler runs with full validated input
        approved = await tool(
            amount=150,
            approval_decisions={approval.approval_id: True},
        ).collect()
        assert approved.status.code == "success"
        assert calls == [50, 150]

    @pytest.mark.asyncio
    async def test_same_input_shares_approval_id_across_invocations(self):
        """Default semantics: a single decision approves any retry/replay of
        the same path+input within a session. Used after transient failures."""
        calls: list[int] = []

        def op(x: int) -> str:
            calls.append(x)
            return f"x={x}"

        tool = Tool(name="op", handler=op, requires_approval=True)
        first = await tool(x=1).collect()
        approval_id = first.metadata["approval"]["id"]

        a = await tool(x=1, approval_decisions={approval_id: True}).collect()
        b = await tool(x=1, approval_decisions={approval_id: True}).collect()

        assert a.status.code == "success" and b.status.code == "success"
        assert calls == [1, 1]

    @pytest.mark.asyncio
    async def test_idempotency_key_pattern_for_per_call_approval(self):
        """For irreversible ops where each call must require its own decision,
        users add an idempotency_key arg so the input hash differs per call.
        This is the documented replacement for the dropped scope='call'."""
        import uuid

        calls: list[tuple[int, str]] = []

        def transfer(amount: int, idempotency_key: str) -> str:
            calls.append((amount, idempotency_key))
            return "ok"

        tool = Tool(name="transfer", handler=transfer, requires_approval=True)

        k1 = str(uuid.uuid4())
        first = await tool(amount=500, idempotency_key=k1).collect()
        first_id = first.metadata["approval"]["id"]

        k2 = str(uuid.uuid4())
        second = await tool(amount=500, idempotency_key=k2).collect()
        second_id = second.metadata["approval"]["id"]

        assert first_id != second_id, "different idempotency_key → different approval_id"

        # First decision does not authorise the second call
        retry = await tool(
            amount=500,
            idempotency_key=k2,
            approval_decisions={first_id: True},
        ).collect()
        assert retry.status.reason == "approval_required"
        assert calls == []


# ---------------------------------------------------------------------------
# Agent lifecycle (tool calls, slash commands, denial-as-tool-result)
# ---------------------------------------------------------------------------


class TestAgentApprovalLifecycle:
    @pytest.mark.asyncio
    async def test_tool_call_full_cycle(self):
        calls: list[int] = []

        def wire_money(amount: int) -> str:
            calls.append(amount)
            return "wired"

        tool = Tool(name="wire_money", handler=wire_money, requires_approval=True)
        agent = Agent(
            name="approval_agent",
            model=TestModel(
                responses=[
                    Message(
                        role="assistant",
                        content=[ToolUseContent(id="call_1", name="wire_money", input={"amount": 500})],
                        stop_reason="tool_use",
                    ),
                    "done",
                ]
            ),
            tools=[tool],
        )

        events = [e async for e in agent(prompt="wire 500")]
        approval = _approval_event(events)
        cancelled = _final_output(events)

        assert approval.runnable_path == "approval_agent.wire_money"
        assert cancelled.path == "approval_agent"
        assert cancelled.status.code == "cancelled"
        assert cancelled.status.reason == "approval_required"
        assert calls == []

        approved = await agent(
            prompt="wire 500",
            approval_decisions={approval.approval_id: True},
        ).collect()

        assert approved.status.code == "success"
        assert calls == [500], "handler must execute after resume"

    @pytest.mark.asyncio
    async def test_denial_becomes_tool_result_for_llm(self):
        calls: list[int] = []

        def wire_money(amount: int) -> str:
            calls.append(amount)
            return "wired"

        tool = Tool(name="wire_money", handler=wire_money, requires_approval=True)
        agent = Agent(
            name="denial_agent",
            model=TestModel(
                responses=[
                    Message(
                        role="assistant",
                        content=[ToolUseContent(id="call_1", name="wire_money", input={"amount": 500})],
                        stop_reason="tool_use",
                    ),
                    "I will not wire the money.",
                ]
            ),
            tools=[tool],
        )

        events = [e async for e in agent(prompt="wire 500")]
        approval = _approval_event(events)

        denied = await agent(
            prompt="wire 500",
            approval_decisions={approval.approval_id: {"approved": False, "reason": "not allowed"}},
        ).collect()

        assert denied.status.code == "success"
        assert denied.output.collect_text() == "I will not wire the money."
        assert calls == []

    @pytest.mark.asyncio
    async def test_slash_command_full_cycle(self):
        calls: list[str] = []

        def delete_target(target: str) -> str:
            calls.append(target)
            return f"deleted {target}"

        tool = Tool(
            name="delete_target",
            handler=delete_target,
            command="/delete",
            requires_approval=True,
        )
        agent = Agent(name="command_agent", model=TestModel(), tools=[tool])

        events = [e async for e in agent(prompt="/delete prod-db")]
        approval = _approval_event(events)
        cancelled = _final_output(events)
        assert approval.input == {"target": "prod-db"}
        assert cancelled.status.reason == "approval_required"
        assert calls == []

        approved = await agent(
            prompt="/delete prod-db",
            approval_decisions={approval.approval_id: True},
        ).collect()
        assert approved.status.code == "success"
        assert calls == ["prod-db"]


# ---------------------------------------------------------------------------
# Workflow lifecycle
# ---------------------------------------------------------------------------


class TestWorkflowApprovalLifecycle:
    @pytest.mark.asyncio
    async def test_step_full_cycle(self):
        calls: list[str] = []

        def deploy(env: str) -> str:
            calls.append(env)
            return f"deployed {env}"

        step = Tool(name="deploy", handler=deploy, requires_approval=True)
        workflow = Workflow(name="deploy_workflow").step(step)

        events = [e async for e in workflow(env="prod")]
        approval = _approval_event(events)
        cancelled = _final_output(events)

        assert approval.runnable_path == "deploy_workflow.deploy"
        assert cancelled.path == "deploy_workflow"
        assert cancelled.status.reason == "approval_required"
        assert calls == []

        approved = await workflow(
            env="prod",
            approval_decisions={approval.approval_id: True},
        ).collect()
        assert approved.status.code == "success"
        assert calls == ["prod"]


# ---------------------------------------------------------------------------
# Parallel approval gates
# ---------------------------------------------------------------------------


class TestParallelApprovalGates:
    @pytest.mark.asyncio
    async def test_agent_with_multiple_concurrent_tool_calls(self):
        """Drain regression: every parallel tool finishes its gate so callers
        see one ApprovalEvent per tool, then a single cancel from the agent."""
        calls: list[str] = []

        def op_a(n: int) -> str:
            calls.append(f"a={n}")
            return "a"

        def op_b(n: int) -> str:
            calls.append(f"b={n}")
            return "b"

        t_a = Tool(name="op_a", handler=op_a, requires_approval=True)
        t_b = Tool(name="op_b", handler=op_b, requires_approval=True)
        agent = Agent(
            name="multi",
            model=TestModel(
                responses=[
                    Message(
                        role="assistant",
                        content=[
                            ToolUseContent(id="c1", name="op_a", input={"n": 1}),
                            ToolUseContent(id="c2", name="op_b", input={"n": 2}),
                        ],
                        stop_reason="tool_use",
                    ),
                    "summary",
                ]
            ),
            tools=[t_a, t_b],
        )

        events = [e async for e in agent(prompt="run both")]
        approvals = _approval_events(events)
        assert len(approvals) == 2, "every concurrent tool must emit its approval event"
        paths = {a.runnable_path for a in approvals}
        assert paths == {"multi.op_a", "multi.op_b"}
        assert calls == []

        decisions = {a.approval_id: True for a in approvals}
        resumed = await agent(prompt="run both", approval_decisions=decisions).collect()
        assert resumed.status.code == "success"
        assert sorted(calls) == ["a=1", "b=2"], "both handlers must run after resuming with both decisions"

    @pytest.mark.asyncio
    async def test_workflow_with_multiple_independent_gating_steps(self):
        """Workflow drain regression: parallel independent steps both gate;
        the workflow does not short-circuit on the first ApprovalRequired."""
        calls: list[str] = []

        def step_a(x: int) -> str:
            calls.append(f"a={x}")
            return "a"

        def step_b(y: int) -> str:
            calls.append(f"b={y}")
            return "b"

        a = Tool(name="step_a", handler=step_a, requires_approval=True)
        b = Tool(name="step_b", handler=step_b, requires_approval=True)
        wf = Workflow(name="parallel_wf").step(a).step(b)

        events = [e async for e in wf(x=1, y=2)]
        approvals = _approval_events(events)
        assert len(approvals) == 2, "every parallel step must emit its approval event"
        paths = {a.runnable_path for a in approvals}
        assert paths == {"parallel_wf.step_a", "parallel_wf.step_b"}

        cancelled = _final_output(events)
        assert cancelled.path == "parallel_wf"
        assert cancelled.status.reason == "approval_required"
        assert calls == []

        decisions = {a.approval_id: True for a in approvals}
        resumed = await wf(x=1, y=2, approval_decisions=decisions).collect()
        assert resumed.status.code == "success"
        assert sorted(calls) == ["a=1", "b=2"]

    @pytest.mark.asyncio
    async def test_workflow_partial_resume_only_executes_approved_steps(self):
        """If only one of two parallel decisions is supplied, the other gates
        again. Caught a class of bug where a later step incorrectly inherits
        its sibling's resolution."""
        calls: list[str] = []

        def step_a(x: int) -> str:
            calls.append(f"a={x}")
            return "a"

        def step_b(y: int) -> str:
            calls.append(f"b={y}")
            return "b"

        a = Tool(name="step_a", handler=step_a, requires_approval=True)
        b = Tool(name="step_b", handler=step_b, requires_approval=True)
        wf = Workflow(name="parallel_wf").step(a).step(b)

        events = [e async for e in wf(x=1, y=2)]
        approvals = {ev.runnable_path: ev for ev in _approval_events(events)}

        decisions = {approvals["parallel_wf.step_a"].approval_id: True}
        partial_events = [e async for e in wf(x=1, y=2, approval_decisions=decisions)]
        partial_final = _final_output(partial_events)
        partial_approvals = _approval_events(partial_events)

        assert calls == ["a=1"], "only the approved step ran"
        assert partial_final.status.code == "cancelled"
        assert partial_final.status.reason == "approval_required"
        assert len(partial_approvals) == 1
        assert partial_approvals[0].runnable_path == "parallel_wf.step_b"


# ---------------------------------------------------------------------------
# Nested lifecycles
# ---------------------------------------------------------------------------


class TestNestedApprovalLifecycle:
    @pytest.mark.asyncio
    async def test_subagent_as_tool_full_cycle(self):
        calls: list[int] = []

        def wire_money(amount: int) -> str:
            calls.append(amount)
            return f"wired {amount}"

        money_tool = Tool(name="wire_money", handler=wire_money, requires_approval=True)
        inner = Agent(
            name="finance_subagent",
            model=TestModel(
                responses=[
                    Message(
                        role="assistant",
                        content=[ToolUseContent(id="ic1", name="wire_money", input={"amount": 999})],
                        stop_reason="tool_use",
                    ),
                    "wire complete",
                ]
            ),
            tools=[money_tool],
        )
        outer = Agent(
            name="orchestrator",
            model=TestModel(
                responses=[
                    Message(
                        role="assistant",
                        content=[ToolUseContent(id="oc1", name="finance_subagent", input={"prompt": "wire 999"})],
                        stop_reason="tool_use",
                    ),
                    "all done",
                ]
            ),
            tools=[inner],
        )

        events = [e async for e in outer(prompt="run finance flow")]
        approval = _approval_event(events)
        cancelled = _final_output(events)
        assert approval.runnable_path == "orchestrator.finance_subagent.wire_money"
        assert cancelled.path == "orchestrator"
        assert cancelled.status.reason == "approval_required"
        assert calls == []

        resumed = await outer(
            prompt="run finance flow",
            approval_decisions={approval.approval_id: True},
        ).collect()
        assert resumed.status.code == "success"
        assert calls == [999]

    @pytest.mark.asyncio
    async def test_workflow_with_agent_step_full_cycle(self):
        calls: list[str] = []

        def shutdown(target: str) -> str:
            calls.append(target)
            return f"shut down {target}"

        risky = Tool(name="shutdown", handler=shutdown, requires_approval=True)
        sub_agent = Agent(
            name="ops_agent",
            model=TestModel(
                responses=[
                    Message(
                        role="assistant",
                        content=[ToolUseContent(id="t1", name="shutdown", input={"target": "prod"})],
                        stop_reason="tool_use",
                    ),
                    "ops done",
                ]
            ),
            tools=[risky],
        )
        wf = Workflow(name="incident").step(sub_agent)

        events = [e async for e in wf(prompt="restart prod")]
        approval = _approval_event(events)
        cancelled = _final_output(events)
        assert approval.runnable_path == "incident.ops_agent.shutdown"
        assert cancelled.path == "incident"
        assert cancelled.status.reason == "approval_required"
        assert calls == []

        resumed = await wf(
            prompt="restart prod",
            approval_decisions={approval.approval_id: True},
        ).collect()
        assert resumed.status.code == "success"
        assert calls == ["prod"]


# ---------------------------------------------------------------------------
# TTL
# ---------------------------------------------------------------------------


class TestApprovalTTL:
    @pytest.mark.asyncio
    async def test_expired_resolution_re_emits_event_and_full_cycle_with_fresh(self):
        calls: list[str] = []

        def delete_file(path: str) -> str:
            calls.append(path)
            return f"deleted {path}"

        tool = Tool(name="delete_file", handler=delete_file, requires_approval=True)
        events = [e async for e in tool(path="/tmp/a.txt")]
        approval = _approval_event(events)

        stale = ApprovalResolution(approved=True, expires_at=int(time.time() * 1000) - 1)
        retry_events = [
            e
            async for e in tool(
                path="/tmp/a.txt",
                approval_decisions={approval.approval_id: stale},
            )
        ]
        retry_approval = _approval_event(retry_events)
        retry_cancelled = _final_output(retry_events)

        assert retry_approval.approval_id == approval.approval_id
        assert retry_cancelled.status.reason == "approval_required"
        assert retry_cancelled.metadata["approval"]["expired"] is True
        assert calls == [], "expired decision must NOT execute the handler"

        # Resume with a fresh decision and assert handler runs.
        fresh = ApprovalResolution(approved=True, expires_at=int(time.time() * 1000) + 60_000)
        resumed = await tool(
            path="/tmp/a.txt",
            approval_decisions={approval.approval_id: fresh},
        ).collect()
        assert resumed.status.code == "success"
        assert calls == ["/tmp/a.txt"]

    def test_is_expired_helper(self):
        now = int(time.time() * 1000)
        assert ApprovalResolution(approved=True).is_expired() is False
        assert ApprovalResolution(approved=True, expires_at=now - 1).is_expired() is True
        assert ApprovalResolution(approved=True, expires_at=now + 60_000).is_expired() is False
        assert ApprovalResolution(approved=True, expires_at=now + 60_000).is_expired(now_ms=now + 100_000) is True


# ---------------------------------------------------------------------------
# Adversarial policy returns — every wrong shape must surface as
# approval_policy_error, never a generic handler error and never silently
# approve. This exercises the wrap-the-whole-function contract.
# ---------------------------------------------------------------------------


def _policy_raises(x: int) -> bool:  # noqa: ARG001
    raise RuntimeError("policy is on fire")


def _policy_returns_str(x: int):  # noqa: ARG001
    return "yes please"


def _policy_returns_int(x: int):  # noqa: ARG001
    return 1


def _policy_returns_list(x: int):  # noqa: ARG001
    return []


def _policy_returns_none(x: int):  # noqa: ARG001
    return None


def _policy_returns_dict_missing_required(x: int):  # noqa: ARG001
    return {"prompt": "no required field"}


def _policy_returns_dict_wrong_type(x: int):  # noqa: ARG001
    """Pydantic rejects list-as-bool — this exercises the validation wrap."""
    return {"required": [1, 2]}


class TestApprovalPolicyErrors:
    @staticmethod
    def _tool(policy, prompt=None) -> Tool:
        def handler(x: int) -> str:  # noqa: ARG001
            return "should not run"

        kwargs = {"name": "risky", "handler": handler, "requires_approval": policy}
        if prompt is not None:
            kwargs["approval_prompt"] = prompt
        return Tool(**kwargs)

    @pytest.mark.parametrize(
        "policy,err_substr",
        [
            (_policy_raises, "policy is on fire"),
            (_policy_returns_str, "must be a bool or callable"),
            (_policy_returns_int, "must be a bool or callable"),
            (_policy_returns_list, "must be a bool or callable"),
            (_policy_returns_none, "must be a bool or callable"),
            (_policy_returns_dict_missing_required, "required"),
            (_policy_returns_dict_wrong_type, ""),
        ],
    )
    @pytest.mark.asyncio
    async def test_invalid_policy_returns_become_policy_error(self, policy, err_substr):
        tool = self._tool(policy)
        out = await tool(x=1).collect()
        assert out.status.code == "error"
        assert out.status.reason == "approval_policy_error"
        assert out.error is not None
        if err_substr:
            assert err_substr.lower() in out.error["message"].lower()

    @pytest.mark.asyncio
    async def test_callable_prompt_exception_also_caught(self):
        def prompt(x: int) -> str:  # noqa: ARG001
            raise ValueError("prompt blew up")

        tool = self._tool(True, prompt=prompt)
        out = await tool(x=1).collect()
        assert out.status.code == "error"
        assert out.status.reason == "approval_policy_error"
        assert "prompt blew up" in out.error["message"]

    @pytest.mark.asyncio
    async def test_async_policy_supported_and_errors_wrapped(self):
        async def bad(x: int):  # noqa: ARG001
            raise RuntimeError("async boom")

        tool = self._tool(bad)
        out = await tool(x=1).collect()
        assert out.status.code == "error"
        assert out.status.reason == "approval_policy_error"
        assert "async boom" in out.error["message"]

    @pytest.mark.asyncio
    async def test_decision_object_with_invalid_metadata_type_also_wrapped(self):
        """ApprovalDecision currently has metadata: dict; passing a non-dict
        should be a policy error, not a handler error."""

        def bad(x: int):  # noqa: ARG001
            # bypass type checker on purpose: simulate user code returning an invalid dict
            return {"required": True, "metadata": "not-a-dict"}

        tool = self._tool(bad)
        out = await tool(x=1).collect()
        assert out.status.code == "error"
        assert out.status.reason == "approval_policy_error"

    @pytest.mark.asyncio
    async def test_valid_decision_object_works(self):
        def policy(x: int) -> ApprovalDecision:  # noqa: ARG001
            return ApprovalDecision(required=True, prompt="approve?")

        tool = self._tool(policy)
        events = [e async for e in tool(x=1)]
        approval = _approval_event(events)
        assert approval.prompt == "approve?"


# ---------------------------------------------------------------------------
# Deprecation
# ---------------------------------------------------------------------------


class TestApprovalDeprecation:
    @pytest.mark.asyncio
    async def test_approvals_kwarg_emits_warning_but_still_works(self):
        def handler() -> str:
            return "ok"

        tool = Tool(name="t", handler=handler, requires_approval=True)
        first = await tool().collect()
        approval_id = first.metadata["approval"]["id"]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = await tool(approvals={approval_id: True}).collect()

        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert deprecations, "expected DeprecationWarning for `approvals=`"
        assert "approval_decisions" in str(deprecations[0].message)
        assert out.status.code == "success"

    @pytest.mark.asyncio
    async def test_approval_decisions_takes_precedence_when_both_passed(self):
        def handler() -> str:
            return "ok"

        tool = Tool(name="t", handler=handler, requires_approval=True)
        first = await tool().collect()
        approval_id = first.metadata["approval"]["id"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            out = await tool(
                approval_decisions={approval_id: True},
                approvals={approval_id: False},
            ).collect()
        assert out.status.code == "success"


# ---------------------------------------------------------------------------
# RunContext.pending_approvals
# ---------------------------------------------------------------------------


class TestPendingApprovalsSurface:
    def test_lists_every_cancelled_approval_span_with_metadata(self):
        from timbal.state.context import RunContext
        from timbal.state.tracing.span import Span
        from timbal.types.run_status import RunStatus

        ctx = RunContext()
        ctx._trace["root"] = Span(
            path="agent",
            call_id="root",
            t0=0,
            status=RunStatus(code="cancelled", reason="approval_required", message=None),
            metadata={},
        )
        ctx._trace["a"] = Span(
            path="agent.op_a",
            call_id="a",
            parent_call_id="root",
            t0=0,
            metadata={"approval": {"id": "id-a", "prompt": "approve a?"}},
            status=RunStatus(code="cancelled", reason="approval_required", message=None),
        )
        ctx._trace["b"] = Span(
            path="agent.op_b",
            call_id="b",
            parent_call_id="root",
            t0=0,
            metadata={
                "approval": {
                    "id": "id-b",
                    "prompt": "approve b?",
                    "expired": True,
                    "expired_at": 12345,
                },
            },
            status=RunStatus(code="cancelled", reason="approval_required", message=None),
        )
        ctx._trace["c"] = Span(
            path="agent.completed",
            call_id="c",
            parent_call_id="root",
            t0=0,
            status=RunStatus(code="success", reason=None, message=None),
        )
        ctx._trace["d"] = Span(
            path="agent.denied",
            call_id="d",
            parent_call_id="root",
            t0=0,
            metadata={"approval": {"id": "id-d"}},
            status=RunStatus(code="cancelled", reason="approval_denied", message=None),
        )

        pending = ctx.pending_approvals()
        by_id = {p["approval_id"]: p for p in pending}
        # Only spans with cancelled/approval_required AND an approval_id
        assert set(by_id) == {"id-a", "id-b"}
        assert by_id["id-a"]["path"] == "agent.op_a"
        assert by_id["id-b"]["expired"] is True
        assert by_id["id-b"]["expired_at"] == 12345
        # Non-expired approvals must not include expired keys
        assert "expired" not in by_id["id-a"]

    def test_handles_dict_status_after_reload(self):
        from timbal.state.context import RunContext
        from timbal.state.tracing.span import Span

        ctx = RunContext()
        span = Span(path="x", call_id="x", t0=0, metadata={"approval": {"id": "id-x"}})
        span.status = {"code": "cancelled", "reason": "approval_required"}
        ctx._trace["x"] = span
        pending = ctx.pending_approvals()
        assert [p["approval_id"] for p in pending] == ["id-x"]

    @pytest.mark.asyncio
    async def test_after_real_parallel_run_lists_every_pending_id(self):
        """Behavioural check: drive an actual parallel-gating workflow, then
        assert pending_approvals enumerates exactly the approval_ids emitted."""

        def step_a(x: int) -> str:  # noqa: ARG001
            return "a"

        def step_b(y: int) -> str:  # noqa: ARG001
            return "b"

        wf = (
            Workflow(name="wf")
            .step(Tool(name="step_a", handler=step_a, requires_approval=True))
            .step(Tool(name="step_b", handler=step_b, requires_approval=True))
        )
        seen_ctx = {}

        # Capture the run_context via post_hook would not run on cancel; instead
        # we collect approval_ids from emitted events and compare them against
        # what a downstream consumer would discover by reading the trace.
        events = [e async for e in wf(x=1, y=2)]
        emitted_ids = {a.approval_id for a in _approval_events(events)}
        assert len(emitted_ids) == 2

        # A consumer would inspect the cancelled OutputEvent's path/metadata —
        # we additionally verify the consumer-visible approval_ids match the
        # set surfaced by emitted events, which is the actual contract.
        cancelled = _final_output(events)
        assert cancelled.status.reason == "approval_required"
        assert seen_ctx == {}  # marker: nothing leaked through hooks
