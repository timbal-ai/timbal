"""Tests for production-time tool result offloading."""

import json
import re

import pytest
from timbal.core.agent import Agent
from timbal.core.test_model import TestModel
from timbal.core.tool import Tool
from timbal.core.tool_result_offload import (
    OFFLOAD_MARKER,
    LocalOffloadStore,
    Spill,
    ToolResultLimit,
    Truncate,
    _shape_sketch,
    _truncate_text,
    apply_tool_result_limit,
    create_read_tool_result,
)
from timbal.state import set_run_context
from timbal.state.context import RunContext
from timbal.state.tracing.providers import InMemoryTracingProvider
from timbal.types.content import TextContent, ToolResultContent, ToolUseContent
from timbal.types.message import Message

_HANDLE_RE = re.compile(r'read_tool_result\(handle="([^"]+)"\)')


def _result(text: str, uid: str = "c1") -> ToolResultContent:
    return ToolResultContent(id=uid, content=[TextContent(text=text)])


# ---------------------------------------------------------------------------
# Truncation + sketch primitives
# ---------------------------------------------------------------------------


class TestTruncateText:
    def test_below_budget_unchanged(self) -> None:
        assert _truncate_text("short", "t", Truncate(max_chars=100)) == "short"

    def test_head(self) -> None:
        out = _truncate_text("a" * 50 + "b" * 50, "t", Truncate(strategy="head", max_chars=50))
        assert out.startswith("a" * 50)
        assert "truncated 50 of 100 chars" in out
        assert "b" not in out.replace("chars from 't' tool result", "")

    def test_tail(self) -> None:
        out = _truncate_text("a" * 50 + "b" * 50, "t", Truncate(strategy="tail", max_chars=50))
        assert out.endswith("b" * 50)
        assert "truncated 50 of 100 chars" in out

    def test_head_tail(self) -> None:
        out = _truncate_text("a" * 50 + "x" * 100 + "b" * 50, "t", Truncate(strategy="head_tail", max_chars=100))
        assert out.startswith("a" * 50)
        assert out.endswith("b" * 50)
        assert "truncated 100 of 200 chars" in out


class TestShapeSketch:
    def test_dict(self) -> None:
        sketch = _shape_sketch(json.dumps({"results": [1, 2, 3], "total": 3, "next": None}))
        assert sketch == '{"results": list[3], "total": int, "next": null}'

    def test_list(self) -> None:
        assert _shape_sketch(json.dumps([{"a": 1}, {"a": 2}])) == "list[2] of object"

    def test_non_json(self) -> None:
        assert _shape_sketch("plain old text") is None


# ---------------------------------------------------------------------------
# LocalOffloadStore
# ---------------------------------------------------------------------------


class TestLocalOffloadStore:
    @pytest.mark.asyncio
    async def test_write_read_roundtrip(self, tmp_path) -> None:
        store = LocalOffloadStore(root=tmp_path)
        handle = await store.write("run1/call1", b"payload")
        assert handle == "run1/call1"
        assert await store.read(handle) == b"payload"

    @pytest.mark.asyncio
    async def test_collision_gets_distinct_handle(self, tmp_path) -> None:
        store = LocalOffloadStore(root=tmp_path)
        h1 = await store.write("run1/call1", b"first")
        h2 = await store.write("run1/call1", b"second")
        assert h1 != h2
        assert await store.read(h1) == b"first"
        assert await store.read(h2) == b"second"

    @pytest.mark.asyncio
    async def test_unsafe_key_sanitized(self, tmp_path) -> None:
        store = LocalOffloadStore(root=tmp_path)
        handle = await store.write("run 1/call:1", b"data")
        assert handle == "run_1/call_1"

    @pytest.mark.asyncio
    async def test_dot_segments_rejected(self, tmp_path) -> None:
        store = LocalOffloadStore(root=tmp_path)
        with pytest.raises(ValueError):
            await store.read("../secrets")
        with pytest.raises(ValueError):
            await store.write("run/../x", b"data")

    @pytest.mark.asyncio
    async def test_absolute_handle_rejected(self, tmp_path) -> None:
        store = LocalOffloadStore(root=tmp_path)
        with pytest.raises(ValueError):
            await store.read("/etc/passwd")

    @pytest.mark.asyncio
    async def test_symlink_escape_rejected(self, tmp_path) -> None:
        root = tmp_path / "root"
        outside = tmp_path / "outside.txt"
        outside.write_text("secret")
        store = LocalOffloadStore(root=root)
        await store.write("run/anchor", b"x")  # creates root
        (root / "link").symlink_to(outside)
        with pytest.raises((ValueError, FileNotFoundError)):
            await store.read("link")

    @pytest.mark.asyncio
    async def test_unknown_handle_raises(self, tmp_path) -> None:
        store = LocalOffloadStore(root=tmp_path)
        with pytest.raises(FileNotFoundError):
            await store.read("run/missing")

    @pytest.mark.asyncio
    async def test_prune_deletes_only_expired_files(self, tmp_path) -> None:
        import os
        import time
        from datetime import timedelta

        store = LocalOffloadStore(root=tmp_path, cleanup_after=timedelta(hours=1))
        old_handle = await store.write("run/old", b"old")
        # Backdate the old file beyond the TTL.
        old_path = tmp_path / old_handle
        expired = time.time() - 2 * 3600
        os.utime(old_path, (expired, expired))

        fresh_handle = await store.write("run/fresh", b"fresh")  # triggers a prune
        # The prune runs on a daemon thread; wait for the expired file to disappear.
        for _ in range(100):
            if not old_path.exists():
                break
            time.sleep(0.01)
        assert not old_path.exists(), "expired file must be pruned"
        assert await store.read(fresh_handle) == b"fresh"


# ---------------------------------------------------------------------------
# apply_tool_result_limit
# ---------------------------------------------------------------------------


class TestApplyToolResultLimit:
    @pytest.mark.asyncio
    async def test_below_threshold_untouched(self, tmp_path) -> None:
        result = _result("small")
        record = await apply_tool_result_limit(
            result,
            limit=ToolResultLimit(threshold=1_000),
            tool_name="t",
            store=LocalOffloadStore(root=tmp_path),
            run_id="run1",
        )
        assert record is None
        assert result.content[0].text == "small"
        assert result.offload_handle is None

    @pytest.mark.asyncio
    async def test_spill(self, tmp_path) -> None:
        store = LocalOffloadStore(root=tmp_path)
        payload = "line\n" * 10_000
        result = _result(payload)
        record = await apply_tool_result_limit(
            result,
            limit=ToolResultLimit(threshold=1_000, action=Spill(preview_chars=100)),
            tool_name="search",
            store=store,
            run_id="run1",
        )
        assert record is not None and record["action"] == "spill"
        assert result.offload_handle == record["handle"]
        placeholder = result.content[0].text
        assert placeholder.startswith(OFFLOAD_MARKER)
        assert "'search'" in placeholder
        assert f'read_tool_result(handle="{record["handle"]}")' in placeholder
        assert "Preview (first 100" in placeholder
        # Lossless: the store holds the full payload.
        assert (await store.read(record["handle"])).decode() == payload

    @pytest.mark.asyncio
    async def test_spill_includes_shape_sketch_for_json(self, tmp_path) -> None:
        store = LocalOffloadStore(root=tmp_path)
        payload = json.dumps({"items": list(range(5_000)), "total": 5_000})
        result = _result(payload)
        await apply_tool_result_limit(
            result,
            limit=ToolResultLimit(threshold=1_000),
            tool_name="api",
            store=store,
            run_id="run1",
        )
        assert 'Shape: {"items": list[5000], "total": int}' in result.content[0].text

    @pytest.mark.asyncio
    async def test_truncate_action(self) -> None:
        result = _result("z" * 5_000)
        record = await apply_tool_result_limit(
            result,
            limit=ToolResultLimit(threshold=1_000, action=Truncate(strategy="head", max_chars=200)),
            tool_name="logs",
            store=None,
            run_id="run1",
        )
        assert record is not None and record["action"] == "truncate"
        assert result.offload_handle is None
        assert result.content[0].text.startswith("z" * 200)
        assert "truncated" in result.content[0].text

    @pytest.mark.asyncio
    async def test_spill_without_store_falls_back_to_truncate(self) -> None:
        result = _result("z" * 5_000)
        record = await apply_tool_result_limit(
            result,
            limit=ToolResultLimit(threshold=1_000, action=Spill(fallback=Truncate(max_chars=100))),
            tool_name="t",
            store=None,
            run_id="run1",
        )
        assert record is not None and record["action"] == "truncate_fallback"
        assert "truncated" in result.content[0].text

    @pytest.mark.asyncio
    async def test_spill_store_failure_falls_back(self) -> None:
        class BrokenStore:
            async def write(self, _key: str, _data: bytes) -> str:
                raise OSError("disk full")

            async def read(self, handle: str) -> bytes:
                raise FileNotFoundError(handle)

        result = _result("z" * 5_000)
        record = await apply_tool_result_limit(
            result,
            limit=ToolResultLimit(threshold=1_000),
            tool_name="t",
            store=BrokenStore(),
            run_id="run1",
        )
        assert record is not None and record["action"] == "truncate_fallback"

    @pytest.mark.asyncio
    async def test_file_content_preserved_on_spill(self, tmp_path) -> None:
        """Non-text content (files) must survive the spill untouched."""
        from timbal.types.content import FileContent
        from timbal.types.file import File

        image = tmp_path / "img.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
        file_content = FileContent(file=File.validate(str(image)))
        result = ToolResultContent(id="c1", content=[TextContent(text="z" * 5_000), file_content])

        record = await apply_tool_result_limit(
            result,
            limit=ToolResultLimit(threshold=1_000),
            tool_name="t",
            store=LocalOffloadStore(root=tmp_path),
            run_id="run1",
        )
        assert record is not None and record["action"] == "spill"
        assert result.content[0].text.startswith(OFFLOAD_MARKER)
        assert file_content in result.content  # the file rides along untouched

    @pytest.mark.asyncio
    async def test_multiple_text_items_measured_and_spilled_together(self, tmp_path) -> None:
        store = LocalOffloadStore(root=tmp_path)
        result = ToolResultContent(
            id="c1",
            content=[TextContent(text="a" * 3_000), TextContent(text="b" * 3_000)],
        )
        record = await apply_tool_result_limit(
            result,
            limit=ToolResultLimit(threshold=5_000),  # neither item alone crosses it
            tool_name="t",
            store=store,
            run_id="run1",
        )
        assert record is not None and record["action"] == "spill"
        assert len([c for c in result.content if isinstance(c, TextContent)]) == 1
        stored = (await store.read(record["handle"])).decode()
        assert "a" * 3_000 in stored and "b" * 3_000 in stored

    @pytest.mark.asyncio
    async def test_spill_no_store_no_fallback_passes_through(self) -> None:
        payload = "z" * 5_000
        result = _result(payload)
        record = await apply_tool_result_limit(
            result,
            limit=ToolResultLimit(threshold=1_000, action=Spill(fallback=None)),
            tool_name="t",
            store=None,
            run_id="run1",
        )
        assert record is None
        assert result.content[0].text == payload


# ---------------------------------------------------------------------------
# read_tool_result
# ---------------------------------------------------------------------------


class TestReadToolResult:
    @pytest.mark.asyncio
    async def test_paging(self, tmp_path) -> None:
        store = LocalOffloadStore(root=tmp_path)
        handle = await store.write("run/c1", "\n".join(f"line-{i}" for i in range(1, 1_001)).encode())
        tool = create_read_tool_result(store)

        out = (await tool(handle=handle, offset=0, limit=3).collect()).output
        assert "[lines 1-3 of 1000" in out
        assert "1: line-1" in out and "3: line-3" in out and "line-4" not in out

        out = (await tool(handle=handle, offset=500, limit=2).collect()).output
        assert "501: line-501" in out and "502: line-502" in out

    @pytest.mark.asyncio
    async def test_pattern_is_literal(self, tmp_path) -> None:
        store = LocalOffloadStore(root=tmp_path)
        handle = await store.write("run/c1", b"alpha\nbeta a.c one\ngamma\nabc two\n")
        tool = create_read_tool_result(store)
        # "a.c" must match the literal substring, not the regex (which would also hit "abc").
        out = (await tool(handle=handle, pattern="a.c").collect()).output
        assert "2: beta a.c one" in out
        assert "abc two" not in out
        assert "of 1 matching lines" in out

    @pytest.mark.asyncio
    async def test_limit_clamped(self, tmp_path) -> None:
        store = LocalOffloadStore(root=tmp_path)
        handle = await store.write("run/c1", ("x\n" * 2_000).encode())
        tool = create_read_tool_result(store)
        out = (await tool(handle=handle, limit=100_000).collect()).output
        assert "[lines 1-500 of 2000" in out

    @pytest.mark.asyncio
    async def test_unknown_handle_errors(self, tmp_path) -> None:
        store = LocalOffloadStore(root=tmp_path)
        tool = create_read_tool_result(store)
        result = await tool(handle="run/nope").collect()
        assert result.status.code == "error"

    @pytest.mark.asyncio
    async def test_output_clipped_at_char_cap(self, tmp_path) -> None:
        """500 long lines would exceed the char cap — output must clip, not balloon."""
        store = LocalOffloadStore(root=tmp_path)
        handle = await store.write("run/c1", ("\n".join("y" * 1_000 for _ in range(500))).encode())
        tool = create_read_tool_result(store)
        out = (await tool(handle=handle, limit=500).collect()).output
        assert len(out) <= 51_000
        assert "output clipped" in out

    def test_own_results_exempt(self, tmp_path) -> None:
        tool = create_read_tool_result(LocalOffloadStore(root=tmp_path))
        assert tool.result_limit is None


# ---------------------------------------------------------------------------
# Agent integration
# ---------------------------------------------------------------------------


def _big_payload() -> str:
    return "\n".join(f"row-{i}: {'x' * 90}" for i in range(1, 1_001))  # ~97k chars


class TestAgentOffload:
    @pytest.mark.asyncio
    async def test_spill_and_read_back_end_to_end(self, tmp_path) -> None:
        """Big tool result → placeholder in memory → model pages it back via read_tool_result."""
        plan = {"n": 0}

        def model_handler(messages):
            plan["n"] += 1
            if plan["n"] == 1:
                return Message(
                    role="assistant",
                    content=[ToolUseContent(id="t1", name="search", input={})],
                    stop_reason="tool_use",
                )
            if plan["n"] == 2:
                # The model reads the handle out of the offload placeholder.
                placeholder = messages[-1].content[0].content[0].text
                handle = _HANDLE_RE.search(placeholder).group(1)
                return Message(
                    role="assistant",
                    content=[
                        ToolUseContent(
                            id="t2", name="read_tool_result", input={"handle": handle, "offset": 499, "limit": 1}
                        )
                    ],
                    stop_reason="tool_use",
                )
            return "done"

        agent = Agent(
            name="offload_agent",
            model=TestModel(handler=model_handler),
            tools=[Tool(name="search", handler=lambda: _big_payload())],
            tool_result_limit=ToolResultLimit(threshold=10_000, store=LocalOffloadStore(root=tmp_path)),
        )

        ctx = RunContext(tracing_provider=InMemoryTracingProvider)
        set_run_context(ctx)
        result = await agent(prompt="find rows").collect()
        assert result.status.code == "success", result.error

        agent_span = ctx._trace.get_path(agent._path)[0]

        # The oversized result was replaced by a placeholder, in memory and in the trace.
        search_results = [
            c
            for m in agent_span.memory
            if m.role == "tool"
            for c in m.content
            if isinstance(c, ToolResultContent) and c.id == "t1"
        ]
        assert len(search_results) == 1
        assert search_results[0].offload_handle is not None
        assert search_results[0].content[0].text.startswith(OFFLOAD_MARKER)
        assert len(search_results[0].content[0].text) < 5_000

        # The read-back returned the requested page.
        read_results = [
            c
            for m in agent_span.memory
            if m.role == "tool"
            for c in m.content
            if isinstance(c, ToolResultContent) and c.id == "t2"
        ]
        assert len(read_results) == 1
        assert "500: row-500" in read_results[0].content[0].text

        # Offload metadata recorded on the span.
        records = agent_span.metadata.get("offload")
        assert records and records[0]["tool"] == "search" and records[0]["action"] == "spill"

        InMemoryTracingProvider._storage.clear()

    @pytest.mark.asyncio
    async def test_small_results_untouched(self, tmp_path) -> None:
        def model_handler(_messages):
            if _messages[-1].role == "user":
                return Message(
                    role="assistant",
                    content=[ToolUseContent(id="t1", name="fetch", input={})],
                    stop_reason="tool_use",
                )
            return "done"

        agent = Agent(
            name="small_agent",
            model=TestModel(handler=model_handler),
            tools=[Tool(name="fetch", handler=lambda: "tiny result")],
            tool_result_limit=ToolResultLimit(threshold=10_000, store=LocalOffloadStore(root=tmp_path)),
        )
        ctx = RunContext(tracing_provider=InMemoryTracingProvider)
        set_run_context(ctx)
        result = await agent(prompt="go").collect()
        assert result.status.code == "success", result.error

        agent_span = ctx._trace.get_path(agent._path)[0]
        tool_msgs = [m for m in agent_span.memory if m.role == "tool"]
        assert tool_msgs[0].content[0].content[0].text == "tiny result"
        assert "offload" not in agent_span.metadata
        InMemoryTracingProvider._storage.clear()

    @pytest.mark.asyncio
    async def test_pinned_tool_exempt(self, tmp_path) -> None:
        def model_handler(_messages):
            if _messages[-1].role == "user":
                return Message(
                    role="assistant",
                    content=[ToolUseContent(id="t1", name="load_docs", input={})],
                    stop_reason="tool_use",
                )
            return "done"

        payload = _big_payload()
        agent = Agent(
            name="pinned_agent",
            model=TestModel(handler=model_handler),
            tools=[Tool(name="load_docs", handler=lambda: payload, pin_result=True)],
            tool_result_limit=ToolResultLimit(threshold=10_000, store=LocalOffloadStore(root=tmp_path)),
        )
        ctx = RunContext(tracing_provider=InMemoryTracingProvider)
        set_run_context(ctx)
        result = await agent(prompt="go").collect()
        assert result.status.code == "success", result.error

        agent_span = ctx._trace.get_path(agent._path)[0]
        tool_result = agent_span.memory[2].content[0]
        assert tool_result.pinned is True
        assert tool_result.content[0].text == payload  # full text preserved
        assert "offload" not in agent_span.metadata
        InMemoryTracingProvider._storage.clear()

    @pytest.mark.asyncio
    async def test_error_results_exempt(self, tmp_path) -> None:
        def boom() -> str:
            raise ValueError("boom: " + "e" * 50_000)

        def model_handler(_messages):
            if _messages[-1].role == "user":
                return Message(
                    role="assistant",
                    content=[ToolUseContent(id="t1", name="boom", input={})],
                    stop_reason="tool_use",
                )
            return "done"

        agent = Agent(
            name="error_agent",
            model=TestModel(handler=model_handler),
            tools=[Tool(name="boom", handler=boom)],
            tool_result_limit=ToolResultLimit(threshold=1_000, store=LocalOffloadStore(root=tmp_path)),
        )
        ctx = RunContext(tracing_provider=InMemoryTracingProvider)
        set_run_context(ctx)
        result = await agent(prompt="go").collect()
        assert result.status.code == "success", result.error

        agent_span = ctx._trace.get_path(agent._path)[0]
        assert "offload" not in agent_span.metadata, "error results must never be reduced"
        InMemoryTracingProvider._storage.clear()

    @pytest.mark.asyncio
    async def test_per_tool_override_and_exemption(self, tmp_path) -> None:
        """Tool-level result_limit overrides the agent default; None exempts entirely."""
        plan = {"n": 0}

        def model_handler(_messages):
            plan["n"] += 1
            if plan["n"] == 1:
                return Message(
                    role="assistant",
                    content=[
                        ToolUseContent(id="t1", name="logs", input={}),
                        ToolUseContent(id="t2", name="raw", input={}),
                    ],
                    stop_reason="tool_use",
                )
            return "done"

        payload = _big_payload()
        agent = Agent(
            name="override_agent",
            model=TestModel(handler=model_handler),
            tools=[
                Tool(
                    name="logs",
                    handler=lambda: payload,
                    result_limit=ToolResultLimit(threshold=1_000, action=Truncate(strategy="tail", max_chars=300)),
                ),
                Tool(name="raw", handler=lambda: payload, result_limit=None),
            ],
            tool_result_limit=ToolResultLimit(threshold=10_000, store=LocalOffloadStore(root=tmp_path)),
        )
        ctx = RunContext(tracing_provider=InMemoryTracingProvider)
        set_run_context(ctx)
        result = await agent(prompt="go").collect()
        assert result.status.code == "success", result.error

        agent_span = ctx._trace.get_path(agent._path)[0]
        by_id = {
            c.id: c
            for m in agent_span.memory
            if m.role == "tool"
            for c in m.content
            if isinstance(c, ToolResultContent)
        }
        assert "truncated" in by_id["t1"].content[0].text  # per-tool truncate won over agent spill
        assert by_id["t1"].content[0].text.endswith(payload[-300:])
        assert by_id["t2"].content[0].text == payload  # exempt tool passes through
        InMemoryTracingProvider._storage.clear()

    @pytest.mark.asyncio
    async def test_int_shorthand(self, tmp_path, monkeypatch) -> None:
        """Agent(tool_result_limit=int) becomes a spill config with a default local store."""
        monkeypatch.setenv("HOME", str(tmp_path))

        def model_handler(_messages):
            if _messages[-1].role == "user":
                return Message(
                    role="assistant",
                    content=[ToolUseContent(id="t1", name="fetch", input={})],
                    stop_reason="tool_use",
                )
            return "done"

        agent = Agent(
            name="shorthand_agent",
            model=TestModel(handler=model_handler),
            tools=[Tool(name="fetch", handler=lambda: "y" * 50_000)],
            tool_result_limit=10_000,
        )
        assert isinstance(agent.tool_result_limit, ToolResultLimit)
        assert agent.tool_result_limit.threshold == 10_000
        assert agent._offload_store is not None

        ctx = RunContext(tracing_provider=InMemoryTracingProvider)
        set_run_context(ctx)
        result = await agent(prompt="go").collect()
        assert result.status.code == "success", result.error

        agent_span = ctx._trace.get_path(agent._path)[0]
        records = agent_span.metadata.get("offload")
        assert records and records[0]["action"] == "spill"
        assert (tmp_path / ".timbal" / "offload").is_dir()
        InMemoryTracingProvider._storage.clear()

    @pytest.mark.asyncio
    async def test_no_config_no_read_tool(self) -> None:
        agent = Agent(
            name="plain_agent",
            model=TestModel(responses=["done"]),
            tools=[Tool(name="fetch", handler=lambda: "x")],
        )
        assert agent._offload_store is None
        assert agent._read_tool_result is None
        tools, _ = await agent._resolve_tools(0)
        assert "read_tool_result" not in {t.name for t in tools}

    @pytest.mark.asyncio
    async def test_read_tool_registered_when_configured(self, tmp_path) -> None:
        agent = Agent(
            name="cfg_agent",
            model=TestModel(responses=["done"]),
            tools=[Tool(name="fetch", handler=lambda: "x")],
            tool_result_limit=ToolResultLimit(store=LocalOffloadStore(root=tmp_path)),
        )
        tools, _ = await agent._resolve_tools(0)
        assert "read_tool_result" in {t.name for t in tools}

    @pytest.mark.asyncio
    async def test_offloaded_placeholders_survive_midloop_compaction(self, tmp_path, monkeypatch) -> None:
        """Offload and compaction compose: when drop-mode compact_tool_results fires
        mid-loop, offloaded placeholders (and their paired tool_use) are kept so their
        handles stay dereferenceable — while nothing else about compaction changes."""
        from timbal.core.memory_compaction import compact_tool_results

        # Tiny window so even the ~1KB placeholders push utilization past the ratio.
        monkeypatch.setattr("timbal.core.agent.get_context_window", lambda _model: 500)

        plan = {"n": 0}

        def model_handler(_messages):
            plan["n"] += 1
            if plan["n"] <= 3:
                return Message(
                    role="assistant",
                    content=[ToolUseContent(id=f"f{plan['n']}", name="fetch", input={"n": plan["n"]})],
                    stop_reason="tool_use",
                )
            return "done"

        agent = Agent(
            name="offload_compact_agent",
            model=TestModel(handler=model_handler),
            tools=[Tool(name="fetch", handler=lambda n: f"payload-{n}: " + "x" * 20_000)],
            tool_result_limit=ToolResultLimit(threshold=10_000, store=LocalOffloadStore(root=tmp_path)),
            memory_compaction=compact_tool_results(),  # drop mode: most aggressive
            memory_compaction_ratio=0.75,
        )

        ctx = RunContext(tracing_provider=InMemoryTracingProvider)
        set_run_context(ctx)
        result = await agent(prompt="fetch everything").collect()
        assert result.status.code == "success", result.error

        agent_span = ctx._trace.get_path(agent._path)[0]
        assert agent_span.metadata.get("compaction", {}).get("triggered") is True
        assert len(agent_span.metadata.get("offload", [])) == 3

        # All three placeholders survived drop-mode compaction, handles intact.
        kept = [
            c
            for m in agent_span.memory
            if m.role == "tool"
            for c in m.content
            if isinstance(c, ToolResultContent)
        ]
        assert len(kept) == 3
        assert all(c.offload_handle for c in kept)
        InMemoryTracingProvider._storage.clear()

    @pytest.mark.asyncio
    async def test_offload_handle_survives_dump_and_reload(self, tmp_path) -> None:
        """offload_handle must round-trip trace serialization (compaction depends on it)."""
        from timbal.utils import dump

        store = LocalOffloadStore(root=tmp_path)
        result = _result("z" * 5_000)
        await apply_tool_result_limit(
            result,
            limit=ToolResultLimit(threshold=1_000),
            tool_name="t",
            store=store,
            run_id="run1",
        )
        msg = Message(role="tool", content=[result])
        reloaded = Message.validate(await dump(msg))
        assert reloaded.content[0].offload_handle == result.offload_handle

    @pytest.mark.asyncio
    async def test_offload_applies_on_approval_resume(self, tmp_path) -> None:
        """The resume path re-executes gated tool_uses through the same
        _process_tool_event — an oversized result produced *after* approval must be
        offloaded exactly like on a fresh turn, with only durable storage between turns."""
        import json as _json

        from timbal.state.tracing.providers.jsonl import JsonlTracingProvider
        from timbal.types.events import ApprovalEvent, OutputEvent

        trace_path = tmp_path / "traces.jsonl"
        provider = JsonlTracingProvider.configured(_path=trace_path)
        store = LocalOffloadStore(root=tmp_path / "offload")

        calls: list[int] = []

        def dump_data() -> str:
            calls.append(1)
            return _big_payload()

        agent = Agent(
            name="resume_offload_agent",
            model=TestModel(
                responses=[
                    Message(
                        role="assistant",
                        content=[ToolUseContent(id="t1", name="dump_data", input={})],
                        stop_reason="tool_use",
                    ),
                    "done",
                ]
            ),
            tools=[Tool(name="dump_data", handler=dump_data, requires_approval=True)],
            tool_result_limit=ToolResultLimit(threshold=10_000, store=store),
            tracing_provider=provider,
        )

        # Turn 1: gate fires, tool never runs, nothing offloaded.
        events1 = [e async for e in agent(prompt="dump it")]
        out1 = next(e for e in reversed(events1) if isinstance(e, OutputEvent))
        approval = next(e for e in events1 if isinstance(e, ApprovalEvent))
        assert out1.status.reason == "approval_required"
        assert calls == []

        # Turn 2: resume with approval — the gated tool executes now.
        out2 = await agent(prompt="dump it", parent_id=out1.run_id, resume={approval.approval_id: True}).collect()
        assert out2.status.code == "success", out2.error
        assert calls == [1]

        # The resumed run's trace holds the offloaded placeholder, not the payload.
        records = [_json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
        record = next(r for r in records if r["run_id"] == out2.run_id)
        agent_span = next(s for s in record["spans"] if s["path"] == "resume_offload_agent")
        tool_results = [
            c
            for m in agent_span["memory"]
            if m["role"] == "tool"
            for c in m["content"]
            if c.get("type") == "tool_result"
        ]
        assert len(tool_results) == 1
        assert tool_results[0].get("offload_handle")
        placeholder_text = tool_results[0]["content"][0]["text"]
        assert placeholder_text.startswith(OFFLOAD_MARKER)
        assert len(placeholder_text) < 5_000
        # And the payload is really in the store.
        assert (await store.read(tool_results[0]["offload_handle"])).decode() == _big_payload()

    @pytest.mark.asyncio
    async def test_offload_applies_on_command_path(self, tmp_path) -> None:
        """Command-triggered tools bypass the LLM but share _process_tool_event — the
        persisted tool_result must be the offloaded placeholder."""
        agent = Agent(
            name="command_offload_agent",
            model=TestModel(responses=["never called"]),
            tools=[Tool(name="dump", handler=lambda: _big_payload(), command="/dump")],
            tool_result_limit=ToolResultLimit(threshold=10_000, store=LocalOffloadStore(root=tmp_path)),
        )

        ctx = RunContext(tracing_provider=InMemoryTracingProvider)
        set_run_context(ctx)
        result = await agent(prompt="/dump").collect()
        assert result.status.code == "success", result.error

        agent_span = ctx._trace.get_path(agent._path)[0]
        records = agent_span.metadata.get("offload")
        assert records and records[0]["tool"] == "dump" and records[0]["action"] == "spill"

        # The dump (what persists and seeds the next turn's memory) carries the placeholder.
        dumped_tool_results = [
            c
            for m in agent_span._memory_dump
            if m.get("role") == "tool"
            for c in m.get("content", [])
            if c.get("type") == "tool_result"
        ]
        assert len(dumped_tool_results) == 1
        assert dumped_tool_results[0].get("offload_handle")
        assert dumped_tool_results[0]["content"][0]["text"].startswith(OFFLOAD_MARKER)
        InMemoryTracingProvider._storage.clear()
