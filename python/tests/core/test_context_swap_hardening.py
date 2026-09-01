"""Regression guards for the T11 incident: orphan ``parent_call_id`` on root spans.

Producer shape (Benito ``email-handler``): a workflow Tool handler calls
``set_run_context(RunContext(...))`` (platform bootstrap — fresh context, empty
trace) while the task's ambient call ids still point at the Tool's span in the
*old* context, then does a nested ``.collect()`` on a top-level runnable.
Before the fix, the top-level invoke only cleared ambient ids when the ambient
context had traces (fork branch), so the first span minted in the swapped-in
context inherited an orphan ``parent_call_id`` — and platform trace validation
rejected the whole trace ("Parent call ID should be empty for root span").

Two layers under test:
1. Top-level invoke (no "." in path) clears ambient call ids even when the
   ambient context's trace is empty, keeping the caller-provided context.
2. Belt at span minting: an ambient parent id that doesn't resolve within the
   current context is repaired to the context's current root — ``None`` on an
   empty trace (the span becomes the root), the existing root otherwise (the
   trace stays closed and single-rooted). Leaving the orphan in place is not
   an option: ``RunContext.update_usage`` asserts trace closure on every
   propagation, so the run would die later with an opaque AssertionError.
"""

import time

from timbal import Agent, Workflow
from timbal.core import Tool
from timbal.core.test_model import TestModel
from timbal.state import (
    get_run_context,
    set_call_id,
    set_parent_call_id,
    set_run_context,
)
from timbal.state.context import RunContext
from timbal.state.tracing.providers import InMemoryTracingProvider
from timbal.state.tracing.span import Span


def _assert_trace_closure(ctx: RunContext) -> None:
    """Every span's parent resolves in-trace and there is exactly one root.

    This is the invariant the platform's trace validation enforces.
    """
    roots = [s for s in ctx._trace.values() if s.parent_call_id is None]
    assert len(roots) == 1, f"expected exactly one root span, got {len(roots)}"
    assert ctx._trace._root_call_id == roots[0].call_id
    for span in ctx._trace.values():
        if span.parent_call_id is not None:
            assert span.parent_call_id in ctx._trace, (
                f"span {span.path} has orphan parent_call_id {span.parent_call_id}"
            )


class TestBootstrapContextSwap:
    async def test_workflow_bootstrap_swap_mints_root_span(self):
        """Benito-shaped repro: workflow → Tool entry → set_run_context(fresh)
        → nested .collect(). The inner run's first span must be a root."""
        captured: dict = {}

        def capture() -> None:
            captured["inner_ctx"] = get_run_context()

        inner = Agent(
            name="inner",
            model=TestModel(responses=["done"]),
            pre_hook=capture,
            tracing_provider=InMemoryTracingProvider,
        )

        async def entry() -> str:
            swapped = RunContext(tracing_provider=InMemoryTracingProvider)
            set_run_context(swapped)
            captured["swapped_ctx"] = swapped
            result = await inner(prompt="hi").collect()
            captured["inner_output_event"] = result
            return result.output.collect_text()

        workflow = Workflow(name="wf").step(Tool(name="entry", handler=entry))
        result = await workflow().collect()
        assert result.status.code == "success", result.error

        # The caller-provided (empty-trace) context is kept, not forked away.
        inner_ctx = captured["inner_ctx"]
        assert inner_ctx is captured["swapped_ctx"]

        # The stale ambient ids from the outer context did not leak in.
        root = inner_ctx.root_span()
        assert root is not None
        assert root.path == "inner"
        assert root.parent_call_id is None
        assert captured["inner_output_event"].parent_call_id is None
        _assert_trace_closure(inner_ctx)

    async def test_direct_tool_bootstrap_swap_mints_root_span(self):
        """Same mechanism without the workflow wrapper: a Tool handler swaps
        in a fresh context and nested-collects a top-level agent."""
        captured: dict = {}

        def capture() -> None:
            captured["inner_ctx"] = get_run_context()

        inner = Agent(
            name="inner",
            model=TestModel(responses=["done"]),
            pre_hook=capture,
            tracing_provider=InMemoryTracingProvider,
        )

        async def bootstrap() -> str:
            set_run_context(RunContext(tracing_provider=InMemoryTracingProvider))
            result = await inner(prompt="hi").collect()
            return result.output.collect_text()

        tool = Tool(name="bootstrap", handler=bootstrap)
        result = await tool().collect()
        assert result.status.code == "success", result.error

        root = captured["inner_ctx"].root_span()
        assert root is not None
        assert root.parent_call_id is None
        _assert_trace_closure(captured["inner_ctx"])

    async def test_healthy_nested_collect_still_forks(self):
        """Guard against regression of the existing fork branch: a nested
        .collect() from a handler WITHOUT a context swap (ambient trace
        non-empty) still forks a fresh context with a clean root."""
        captured: dict = {}

        def capture() -> None:
            captured["inner_ctx"] = get_run_context()

        inner = Agent(
            name="inner",
            model=TestModel(responses=["done"]),
            pre_hook=capture,
            tracing_provider=InMemoryTracingProvider,
        )

        async def nested_call() -> str:
            captured["outer_ctx"] = get_run_context()
            result = await inner(prompt="hi").collect()
            return result.output.collect_text()

        tool = Tool(name="nested_call", handler=nested_call)
        result = await tool().collect()
        assert result.status.code == "success", result.error

        # Fresh context (concurrent-sibling fork), not the outer one.
        assert captured["inner_ctx"] is not captured["outer_ctx"]
        assert captured["inner_ctx"].id != captured["outer_ctx"].id
        root = captured["inner_ctx"].root_span()
        assert root is not None
        assert root.parent_call_id is None
        _assert_trace_closure(captured["inner_ctx"])
        _assert_trace_closure(captured["outer_ctx"])

    async def test_preset_empty_context_without_stale_ids_is_kept(self):
        """The legit pattern — set_run_context(RunContext(...)) at the top of
        a script to inject config — keeps working: context kept, root clean."""
        ctx = RunContext(tracing_provider=InMemoryTracingProvider)
        set_run_context(ctx)

        agent = Agent(
            name="solo",
            model=TestModel(responses=["done"]),
            tracing_provider=InMemoryTracingProvider,
        )
        result = await agent(prompt="hi").collect()
        assert result.status.code == "success", result.error

        assert get_run_context() is ctx
        root = ctx.root_span()
        assert root is not None
        assert root.path == "solo"
        assert root.parent_call_id is None
        _assert_trace_closure(ctx)


class TestOrphanParentBelt:
    async def test_belt_clears_orphan_parent_on_empty_trace(self):
        """A dotted-path runnable (bypasses the top-level guard) invoked under
        a swapped context with stale ambient ids: the orphan parent id is
        cleared so the first span of the context is a root."""
        tool = Tool(name="step", handler=lambda: "x")
        tool.nest("wf")  # path becomes "wf.step" — no top-level clearing

        ctx = RunContext(tracing_provider=InMemoryTracingProvider)
        set_run_context(ctx)
        set_parent_call_id("stale_parent_from_old_context")
        set_call_id("stale_call_from_old_context")

        result = await tool().collect()
        assert result.status.code == "success", result.error

        assert len(ctx._trace) == 1
        span = next(iter(ctx._trace.values()))
        assert span.path == "wf.step"
        assert span.parent_call_id is None
        assert ctx._trace._root_call_id == span.call_id

    async def test_belt_reparents_orphan_to_existing_root(self):
        """With an existing root in the trace, an unresolvable parent id is
        reparented to that root: the run completes, usage propagation (which
        asserts trace closure) works, and the trace stays single-rooted."""
        tool = Tool(name="step", handler=lambda: "x")
        tool.nest("wf")

        ctx = RunContext(tracing_provider=InMemoryTracingProvider)
        t0 = int(time.time() * 1000)
        ctx._trace["existing_root"] = Span(
            path="wf", call_id="existing_root", parent_call_id=None, t0=t0
        )
        set_run_context(ctx)
        set_parent_call_id(None)
        set_call_id("stale_call_from_old_context")

        result = await tool().collect()
        assert result.status.code == "success", result.error

        spans = [s for s in ctx._trace.values() if s.path == "wf.step"]
        assert len(spans) == 1
        assert spans[0].parent_call_id == "existing_root"
        assert ctx._trace._root_call_id == "existing_root"
        _assert_trace_closure(ctx)
        # Usage propagated through the repaired chain up to the root.
        assert ctx._trace["existing_root"].usage.get("step:requests", 0) >= 1
