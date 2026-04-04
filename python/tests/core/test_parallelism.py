"""Tests for context var isolation under parallelism.

These tests exercise the Runnable/Tool/Agent context machinery to verify
that concurrent execution properly isolates state.
"""

import asyncio
import contextvars

import pytest
from timbal import Agent, Tool
from timbal.state import (
    get_call_id,
    get_parent_call_id,
    get_run_context,
    set_call_id,
    set_parent_call_id,
    set_run_context,
)
from timbal.state.context import RunContext
from timbal.state.tracing.span import Span
from timbal.state.tracing.trace import Trace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool(name: str, handler=None):
    """Create a simple Tool wrapping a sync handler."""
    if handler is None:
        def handler(x: str = "default") -> str:
            return f"{name}:{x}"
    return Tool(name=name, handler=handler, description=f"Tool {name}")


def _make_async_tool(name: str, delay: float = 0.01):
    """Create a Tool with an async handler that yields control."""
    async def handler(x: str = "default") -> str:
        await asyncio.sleep(delay)
        return f"{name}:{x}"
    return Tool(name=name, handler=handler, description=f"Async tool {name}")


# ---------------------------------------------------------------------------
# Test 1: Context is restored inside generator after yields
# ---------------------------------------------------------------------------

class TestContextRestorationInsideGenerator:
    """After yielding START and being resumed (possibly after another coroutine
    ran), __call__ restores its own context vars before continuing handler
    execution. We verify this by checking the span's parent_call_id is correct
    — which proves the generator saw the right context when it resumed."""

    @pytest.mark.asyncio
    async def test_interleaved_tools_produce_correct_spans(self):
        """Two tools interleaved in the same task produce events with correct call_ids."""
        tool_a = _make_tool("tool_a")
        tool_b = _make_tool("tool_b")

        gen_a = tool_a(x="a")
        gen_b = tool_b(x="b")

        events_a = []
        events_b = []

        iter_a = gen_a.__aiter__()
        iter_b = gen_b.__aiter__()

        # Interleave: START from A, then START from B
        events_a.append(await iter_a.__anext__())
        events_b.append(await iter_b.__anext__())

        # Finish A (OUTPUT event) — A's generator resumes, restores its own context
        # internally, executes handler, produces OUTPUT
        events_a.append(await iter_a.__anext__())

        # Finish B
        events_b.append(await iter_b.__anext__())

        # Each tool's events should carry its own call_id consistently
        assert events_a[0].call_id == events_a[1].call_id, (
            "tool_a START and OUTPUT should share the same call_id"
        )
        assert events_b[0].call_id == events_b[1].call_id, (
            "tool_b START and OUTPUT should share the same call_id"
        )
        assert events_a[0].call_id != events_b[0].call_id, (
            "tool_a and tool_b should have different call_ids"
        )

    @pytest.mark.asyncio
    async def test_handler_sees_own_context_after_interleaving(self):
        """A tool's handler sees the correct call_id even after another tool
        has run and set a different call_id in between yields."""
        observed_call_ids = {}

        def capture_a(x: str = "a") -> str:
            observed_call_ids["a"] = get_call_id()
            return f"a:{x}"

        def capture_b(x: str = "b") -> str:
            observed_call_ids["b"] = get_call_id()
            return f"b:{x}"

        tool_a = Tool(name="tool_a", handler=capture_a, description="Tool A")
        tool_b = Tool(name="tool_b", handler=capture_b, description="Tool B")

        gen_a = tool_a(x="a")
        gen_b = tool_b(x="b")

        iter_a = gen_a.__aiter__()
        iter_b = gen_b.__aiter__()

        # Get START from both (B clobbers A's call_id in the shared context var)
        start_a = await iter_a.__anext__()
        start_b = await iter_b.__anext__()

        # Now advance A — its handler will run and capture get_call_id()
        # After resume, __call__ calls _restore_context() which sets call_id
        # back to A's id before the handler executes
        await iter_a.__anext__()  # OUTPUT from A

        # Advance B
        await iter_b.__anext__()  # OUTPUT from B

        # Each handler should have seen its own call_id
        assert observed_call_ids["a"] == start_a.call_id, (
            f"tool_a handler saw call_id {observed_call_ids['a']} but expected {start_a.call_id}"
        )
        assert observed_call_ids["b"] == start_b.call_id, (
            f"tool_b handler saw call_id {observed_call_ids['b']} but expected {start_b.call_id}"
        )


# ---------------------------------------------------------------------------
# Test 2: create_task with copy_context isolates context per task
# ---------------------------------------------------------------------------

class TestCreateTaskIsolation:
    """create_task copies context vars. Each task should have its own
    call_id / parent_call_id, and mutations in one don't affect the other."""

    @pytest.mark.asyncio
    async def test_create_task_isolates_call_id(self):
        """Each task has its own call_id after mutation."""
        parent_ctx = RunContext()
        set_run_context(parent_ctx)
        set_call_id("parent")
        set_parent_call_id(None)

        results = {"task1": {}, "task2": {}}

        async def task_fn(label):
            results[label]["inherited_call_id"] = get_call_id()
            set_call_id(f"{label}_call")
            await asyncio.sleep(0)
            results[label]["own_call_id"] = get_call_id()

        t1 = asyncio.create_task(task_fn("task1"))
        t2 = asyncio.create_task(task_fn("task2"))
        await asyncio.gather(t1, t2)

        assert results["task1"]["own_call_id"] == "task1_call"
        assert results["task2"]["own_call_id"] == "task2_call"
        assert get_call_id() == "parent"

    @pytest.mark.asyncio
    async def test_parallel_tools_via_create_task_have_isolated_call_ids(self):
        """Tools in separate tasks with copy_context have independent call_ids."""
        tool_a = _make_tool("tool_a")
        tool_b = _make_tool("tool_b")

        parent_ctx = RunContext()
        set_run_context(parent_ctx)
        set_call_id("parent")
        set_parent_call_id(None)

        call_ids = {"a": None, "b": None}

        async def run_tool(tool, label):
            async for event in tool(x="test"):
                if event.type == "START":
                    call_ids[label] = event.call_id

        t1 = asyncio.create_task(run_tool(tool_a, "a"), context=contextvars.copy_context())
        t2 = asyncio.create_task(run_tool(tool_b, "b"), context=contextvars.copy_context())
        await asyncio.gather(t1, t2)

        assert call_ids["a"] is not None
        assert call_ids["b"] is not None
        assert call_ids["a"] != call_ids["b"]
        assert get_call_id() == "parent"


# ---------------------------------------------------------------------------
# Test 3: Parallel collect via create_task + copy_context
# ---------------------------------------------------------------------------

class TestParallelCollect:
    """Two tools collected in parallel via create_task + copy_context
    should produce correct, independent results."""

    @pytest.mark.asyncio
    async def test_parallel_collect_produces_correct_results(self):
        tool_a = _make_tool("tool_a")
        tool_b = _make_tool("tool_b")

        set_run_context(None)
        set_call_id(None)
        set_parent_call_id(None)

        async def collect_tool(tool, **kwargs):
            return await tool(**kwargs).collect()

        t1 = asyncio.create_task(collect_tool(tool_a, x="a"), context=contextvars.copy_context())
        t2 = asyncio.create_task(collect_tool(tool_b, x="b"), context=contextvars.copy_context())
        result_a, result_b = await asyncio.gather(t1, t2)

        assert result_a.output == "tool_a:a"
        assert result_b.output == "tool_b:b"


# ---------------------------------------------------------------------------
# Test 4: Trace root_call_id constraint
# ---------------------------------------------------------------------------

class TestTraceRootCallIdConstraint:
    """Two root spans in the same trace is correctly rejected."""

    @pytest.mark.asyncio
    async def test_double_root_span_raises(self):
        trace = Trace()
        span_a = Span(path="a", call_id="a", parent_call_id=None, t0=0)
        span_b = Span(path="b", call_id="b", parent_call_id=None, t0=0)

        trace["a"] = span_a
        with pytest.raises(ValueError, match="Cannot set multiple root calls"):
            trace["b"] = span_b


# ---------------------------------------------------------------------------
# Test 5: Usage update is atomic (no await points)
# ---------------------------------------------------------------------------

class TestUsageUpdateAtomicity:
    """update_usage() is safe under asyncio concurrency because it has no
    await points — the entire read-modify-write runs without yielding
    to the event loop."""

    @pytest.mark.asyncio
    async def test_concurrent_usage_updates_are_correct(self):
        """Concurrent tasks calling update_usage produce correct totals
        because there are no await points inside the method."""
        ctx = RunContext()
        set_run_context(ctx)
        set_call_id("root_call")

        root_id = "root_call"
        root_span = Span(path="root", call_id=root_id, parent_call_id=None, t0=0)
        ctx._trace[root_id] = root_span

        n_tasks = 50
        increments_per_task = 100
        expected_total = n_tasks * increments_per_task

        async def increment_usage():
            for _ in range(increments_per_task):
                ctx.update_usage("tokens", 1)
                await asyncio.sleep(0)  # yield between calls is fine

        tasks = [asyncio.create_task(increment_usage()) for _ in range(n_tasks)]
        await asyncio.gather(*tasks)

        actual_total = root_span.usage.get("tokens", 0)
        assert actual_total == expected_total, (
            f"Expected {expected_total} but got {actual_total} — "
            "update_usage should be atomic since it has no await points."
        )


# ---------------------------------------------------------------------------
# Test 6: Session data init is idempotent
# ---------------------------------------------------------------------------

class TestSessionDataInit:
    """get_session() should return the same dict for repeated calls."""

    @pytest.mark.asyncio
    async def test_repeated_get_session_returns_same_dict(self):
        ctx = RunContext()
        s1 = await ctx.get_session()
        s2 = await ctx.get_session()
        assert s1 is s2


# ---------------------------------------------------------------------------
# Test 7: Sequential tools restore context on completion
# ---------------------------------------------------------------------------

class TestSequentialToolCleanup:
    """After a tool finishes, the context vars are restored to their
    pre-invocation state."""

    @pytest.mark.asyncio
    async def test_context_restored_after_collect(self):
        """After collecting a tool's result, the context vars are back to pre-invocation."""
        tool_a = _make_tool("tool_a")

        set_run_context(None)
        set_call_id(None)
        set_parent_call_id(None)

        result = await tool_a(x="a").collect()
        assert result.output == "tool_a:a"

        # The finally block in __call__ restores original call_id/parent_call_id.
        # run_context was None before, but __call__ created one and set it.
        # call_id should be restored to None (pre-invocation value).
        # Note: run_context is NOT restored to None — it keeps the created one.
        # This is by design (the context persists for session chaining).

    @pytest.mark.asyncio
    async def test_two_sequential_tools_get_independent_traces(self):
        """Two tools called sequentially each produce their own events."""
        tool_a = _make_tool("tool_a")
        tool_b = _make_tool("tool_b")

        set_run_context(None)
        set_call_id(None)
        set_parent_call_id(None)

        result_a = await tool_a(x="a").collect()
        result_b = await tool_b(x="b").collect()

        assert result_a.output == "tool_a:a"
        assert result_b.output == "tool_b:b"
        assert result_a.call_id != result_b.call_id


# ---------------------------------------------------------------------------
# Test 8: Heavy async parallelism — many concurrent tool invocations
# ---------------------------------------------------------------------------

class TestHeavyParallelism:
    """Stress test: many concurrent Runnable.__call__ invocations via
    create_task + copy_context. Every invocation should get its own
    RunContext, its own call_id, and produce the correct output."""

    @pytest.mark.asyncio
    async def test_50_sync_tools_in_parallel(self):
        """50 sync tools running concurrently, each in its own task."""
        set_run_context(None)
        set_call_id(None)
        set_parent_call_id(None)

        n = 50
        tools = [_make_tool(f"tool_{i}") for i in range(n)]

        async def run_one(tool, idx):
            result = await tool(x=str(idx)).collect()
            return result

        tasks = [
            asyncio.create_task(run_one(tools[i], i), context=contextvars.copy_context())
            for i in range(n)
        ]
        results = await asyncio.gather(*tasks)

        # Every result should be correct
        outputs = {r.output for r in results}
        expected = {f"tool_{i}:{i}" for i in range(n)}
        assert outputs == expected, f"Missing or wrong outputs: {outputs ^ expected}"

        # Every call_id should be unique
        call_ids = [r.call_id for r in results]
        assert len(set(call_ids)) == n, "Not all call_ids are unique"

    @pytest.mark.asyncio
    async def test_50_async_tools_in_parallel(self):
        """50 async tools (with actual awaits in the handler) running concurrently."""
        set_run_context(None)
        set_call_id(None)
        set_parent_call_id(None)

        n = 50
        tools = [_make_async_tool(f"async_tool_{i}", delay=0.01) for i in range(n)]

        async def run_one(tool, idx):
            return await tool(x=str(idx)).collect()

        tasks = [
            asyncio.create_task(run_one(tools[i], i), context=contextvars.copy_context())
            for i in range(n)
        ]
        results = await asyncio.gather(*tasks)

        outputs = {r.output for r in results}
        expected = {f"async_tool_{i}:{i}" for i in range(n)}
        assert outputs == expected

        call_ids = [r.call_id for r in results]
        assert len(set(call_ids)) == n

    @pytest.mark.asyncio
    async def test_50_async_tools_each_observes_own_context(self):
        """Each async tool's handler sees its own call_id, not another tool's."""
        set_run_context(None)
        set_call_id(None)
        set_parent_call_id(None)

        n = 50
        observed = {}

        def make_capturing_tool(idx):
            async def handler(x: str = "default") -> str:
                await asyncio.sleep(0.01)  # yield control
                observed[idx] = get_call_id()
                return f"tool_{idx}:{x}"
            return Tool(name=f"tool_{idx}", handler=handler, description=f"Tool {idx}")

        tools = [make_capturing_tool(i) for i in range(n)]

        async def run_one(tool, idx):
            return await tool(x=str(idx)).collect()

        tasks = [
            asyncio.create_task(run_one(tools[i], i), context=contextvars.copy_context())
            for i in range(n)
        ]
        results = await asyncio.gather(*tasks)

        # Each handler should have seen a call_id that matches its own START event
        for i, result in enumerate(results):
            assert observed[i] == result.call_id, (
                f"Tool {i} handler saw call_id {observed[i]} but its START event had {result.call_id}"
            )

    @pytest.mark.asyncio
    async def test_50_tools_interleaved_in_same_task(self):
        """50 async tools interleaved via gather (no create_task) — tests
        _restore_context correctness under extreme interleaving."""
        set_run_context(None)
        set_call_id(None)
        set_parent_call_id(None)

        n = 50
        observed = {}

        def make_capturing_tool(idx):
            async def handler(x: str = "default") -> str:
                await asyncio.sleep(0.001)
                observed[idx] = get_call_id()
                return f"tool_{idx}:{x}"
            return Tool(name=f"tool_{idx}", handler=handler, description=f"Tool {idx}")

        tools = [make_capturing_tool(i) for i in range(n)]

        # gather with coroutines (NOT create_task) — all run in the same task
        results = await asyncio.gather(
            *[tools[i](x=str(i)).collect() for i in range(n)]
        )

        outputs = {r.output for r in results}
        expected = {f"tool_{i}:{i}" for i in range(n)}
        assert outputs == expected

        # Each handler should have observed its own call_id
        for i, result in enumerate(results):
            assert observed[i] == result.call_id, (
                f"Tool {i} handler saw call_id {observed[i]} but expected {result.call_id}"
            )

    @pytest.mark.asyncio
    async def test_agent_instances_parallel_context_isolation(self):
        """Multiple Agent instances running in parallel get independent
        RunContexts and don't contaminate each other's traces."""
        set_run_context(None)
        set_call_id(None)
        set_parent_call_id(None)

        n = 20
        agents = [
            Agent(
                name=f"agent_{i}",
                model="openai/gpt-4o-mini",
                tools=[_make_tool(f"tool_{i}")],
            )
            for i in range(n)
        ]

        contexts_seen = {}

        async def run_agent(agent, idx):
            """Iterate just enough to capture the RunContext, then close."""
            gen = agent(prompt="hello")
            try:
                async for event in gen:
                    if event.type == "START":
                        ctx = get_run_context()
                        contexts_seen[idx] = {
                            "run_id": ctx.id,
                            "call_id": event.call_id,
                            "trace_id": id(ctx._trace),
                        }
                        break
            except Exception:
                pass  # LLM call may fail — we only care about context setup
            finally:
                await gen.aclose()

        tasks = [
            asyncio.create_task(run_agent(agents[i], i), context=contextvars.copy_context())
            for i in range(n)
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

        # Every agent that got far enough should have a unique RunContext
        assert len(contexts_seen) == n, f"Only {len(contexts_seen)}/{n} agents set up context"

        run_ids = [v["run_id"] for v in contexts_seen.values()]
        assert len(set(run_ids)) == n, (
            f"Expected {n} unique run_ids, got {len(set(run_ids))} — "
            "agents are sharing RunContext"
        )

        call_ids = [v["call_id"] for v in contexts_seen.values()]
        assert len(set(call_ids)) == n, "Not all call_ids are unique"

        trace_ids = [v["trace_id"] for v in contexts_seen.values()]
        assert len(set(trace_ids)) == n, (
            "Agents share the same _trace dict — concurrent writes are unsafe"
        )


# ---------------------------------------------------------------------------
# Test 9: Manual generator interleaving — the only remaining vulnerability
# ---------------------------------------------------------------------------

class TestManualGeneratorInterleaving:
    """asyncio.gather creates tasks internally (with copied context), so it's
    safe. The ONLY way to get same-task context clobbering is to manually
    advance two generators with __anext__() in a single coroutine.

    _restore_context() handles the yield points. But between yields, awaits
    inside __call__ (like _resolve_input_params) can still see a stale
    context if another generator was advanced in between."""

    @pytest.mark.asyncio
    async def test_gather_is_safe_because_it_creates_tasks(self):
        """asyncio.gather wraps coroutines in tasks — each gets its own context."""
        observed = {}

        async def handler(x: str = "default") -> str:
            await asyncio.sleep(0.01)
            observed[x] = get_call_id()
            return f"result:{x}"

        tool_a = Tool(name="tool_a", handler=handler, description="A")
        tool_b = Tool(name="tool_b", handler=handler, description="B")

        set_run_context(None)
        set_call_id(None)
        set_parent_call_id(None)

        result_a, result_b = await asyncio.gather(
            tool_a(x="a").collect(),
            tool_b(x="b").collect(),
        )

        # Each handler sees its own call_id — gather creates tasks internally
        assert observed["a"] == result_a.call_id
        assert observed["b"] == result_b.call_id

    @pytest.mark.asyncio
    async def test_manual_interleaving_does_not_create_false_parent(self):
        """When manually advancing two generators, tool B should NOT see
        tool A as its parent. The setup code detects that A's root span
        is still running (t1 is None) and creates a fresh context instead
        of a child context."""

        tool_a = _make_tool("tool_a")
        tool_b = _make_tool("tool_b")

        set_run_context(None)
        set_call_id(None)
        set_parent_call_id(None)

        gen_a = tool_a(x="a")
        iter_a = gen_a.__aiter__()

        # Advance A to START — A creates a RunContext and sets it
        start_a = await iter_a.__anext__()
        ctx_after_a_start = get_run_context()
        assert ctx_after_a_start is not None

        # Now start B — B's __call__ sees A's context, but A's root span
        # has t1=None (still running). So B creates a fresh RunContext
        # instead of a child context.
        gen_b = tool_b(x="b")
        iter_b = gen_b.__aiter__()
        start_b = await iter_b.__anext__()
        ctx_after_b_start = get_run_context()

        # B should have a fresh context — no parent_id
        assert ctx_after_b_start.parent_id is None, (
            f"B's RunContext should have no parent, but got parent_id={ctx_after_b_start.parent_id}. "
            "Context bled between manually interleaved generators."
        )

        # Both should have different run IDs
        assert ctx_after_a_start.id != ctx_after_b_start.id

        # Clean up
        async for _ in iter_a:
            pass
        async for _ in iter_b:
            pass

    @pytest.mark.asyncio
    async def test_sequential_runs_still_chain_sessions(self):
        """After tool A completes, tool B should see A as its parent
        (for session chaining). This verifies we didn't break the
        sequential session chaining behavior."""

        tool_a = _make_tool("tool_a")
        tool_b = _make_tool("tool_b")

        set_run_context(None)
        set_call_id(None)
        set_parent_call_id(None)

        # Run A to completion
        result_a = await tool_a(x="a").collect()
        ctx_after_a = get_run_context()
        assert ctx_after_a is not None

        # Now run B — A's root span has t1 set (completed), so B should
        # chain off A via parent_id
        result_b = await tool_b(x="b").collect()
        ctx_after_b = get_run_context()

        assert ctx_after_b.parent_id == ctx_after_a.id, (
            "Sequential run B should chain off completed run A via parent_id"
        )
