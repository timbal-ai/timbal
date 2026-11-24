"""PlatformTracingProvider span deltas and negotiated gzip, against a fake platform server.

The fake implements the server side of rfcs/platform-trace-storage.md (A + C): request
decompression, `Accept-Encoding` / `Accept-Patch` advertisement, span upserts ordered by
`snapshot`, assembly at `ended`, and GET of a never-ended run from its span rows.
"""

import asyncio
import gzip
import json
import os
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from structlog.testing import capture_logs
from timbal import Agent, Tool
from timbal.core.test_model import TestModel
from timbal.errors import PlatformError
from timbal.state import get_run_context, set_call_id, set_parent_call_id, set_run_context
from timbal.state.config import PlatformAuth, PlatformAuthType, PlatformConfig, PlatformSubject
from timbal.state.context import RunContext
from timbal.state.tracing import uploads
from timbal.state.tracing.providers import InMemoryTracingProvider
from timbal.state.tracing.providers import platform as platform_module
from timbal.state.tracing.providers.platform import DELTA_MEDIA_TYPE, PlatformTracingProvider, RunPath
from timbal.types.content import ToolUseContent
from timbal.types.message import Message

CONFIG = PlatformConfig(
    host="api.deltas.test",
    auth=PlatformAuth(type=PlatformAuthType.BEARER, token="t2_test"),
    subject=PlatformSubject(org_id="1", app_id="2"),
)


@pytest.fixture(autouse=True)
def awaited_saves(monkeypatch):
    """These tests check what each save puts on the wire; the platform's default buffers saves mid-run."""
    monkeypatch.setenv(uploads.DURABILITY_ENV, "sync")


@pytest.fixture(autouse=True)
def clean():
    PlatformTracingProvider._caps.clear()
    PlatformTracingProvider._delta_runs.clear()
    PlatformTracingProvider._recent.clear()
    PlatformTracingProvider._flights.clear()
    PlatformTracingProvider._unsaved.clear()
    # Snapshots are numbered as in a new process, by the clock alone: a test here may leave the counter ahead
    # of the clock (a platform that holds a newer snapshot), and no test may count on an earlier one's saves.
    platform_module._last_snapshot = 0
    set_run_context(None)
    set_call_id(None)
    set_parent_call_id(None)
    yield
    PlatformTracingProvider._caps.clear()
    PlatformTracingProvider._delta_runs.clear()
    PlatformTracingProvider._recent.clear()
    PlatformTracingProvider._unsaved.clear()
    platform_module._last_snapshot = 0
    for task in PlatformTracingProvider._retries.values():
        task.cancel()
    PlatformTracingProvider._retries.clear()


class FakePlatform:
    def __init__(self, *, deltas=True, gzip_ok=True, advertise=True, limit=None):
        self.deltas, self.gzip_ok, self.advertise, self.limit = deltas, gzip_ok, advertise, limit
        self.runs: dict[str, dict] = {}
        self.patches: list[dict] = []  # {"kind", "bytes", "body"}
        self.fail_next = 0

    def _headers(self):
        if not self.advertise:
            return {}
        headers = {}
        if self.gzip_ok:
            headers["Accept-Encoding"] = "gzip"
        if self.deltas:
            headers["Accept-Patch"] = f"application/json, {DELTA_MEDIA_TYPE}"
        return headers

    def _ok(self, payload=None, stored=None, applied=True):
        headers = self._headers()
        if stored is not None and self.deltas and self.advertise:
            # A server that takes deltas implements 0.1: it echoes the snapshot it holds.
            headers = {**headers, "X-Timbal-Snapshot": str(stored), "X-Timbal-Snapshot-Applied": str(applied).lower()}
        return SimpleNamespace(raise_for_status=lambda: None, headers=headers, json=lambda: payload or {})

    async def request(self, method, path, headers=None, content=None, **_):
        run_id = path.rsplit("/", 1)[-1]
        if method == "GET":
            run = self.runs.get(run_id)
            if run is None:
                return self._ok({})
            if run.get("trace") is not None:
                return self._ok({"trace": run["trace"], "parent_id": run["parent_id"]})
            rows = [span for _seq, span in sorted(run["spans"].values(), key=lambda row: row[0])]
            return self._ok({"trace": [run["root"], *rows], "parent_id": run["parent_id"]})
        headers = headers or {}
        if self.fail_next:
            self.fail_next -= 1
            raise PlatformError("upstream", status_code=503)
        raw = content
        if headers.get("Content-Encoding") == "gzip":
            if not self.gzip_ok:
                raise PlatformError("bad json", status_code=400)
            raw = gzip.decompress(content)
        if self.limit is not None and len(raw) > self.limit:  # the limit applies to the decompressed body
            raise PlatformError("payload too large", status_code=413)
        body = json.loads(raw)
        kind = "delta" if headers.get("Content-Type") == DELTA_MEDIA_TYPE else "full"
        if kind == "delta" and not self.deltas:
            raise PlatformError("unsupported media type", status_code=415)
        self.patches.append(
            {"kind": kind, "bytes": len(content), "body": body, "gzip": headers.get("Content-Encoding") == "gzip"}
        )
        run = self.runs.setdefault(run_id, {"spans": {}, "trace": None, "root": None, "parent_id": None, "snapshot": 0})
        # 0.1: an older snapshot is acked and dropped; an equal one is a retry and applies again.
        sent = int(headers["X-Timbal-Snapshot"]) if "X-Timbal-Snapshot" in headers else None
        if sent is not None and sent < run["snapshot"]:
            return self._ok(stored=run["snapshot"], applied=False)
        if sent is not None:
            run["snapshot"] = sent
        run["parent_id"] = body.get("parent_id")
        if kind == "full":  # authoritative: replaces the span rows
            run.update(trace=body["trace"], spans={}, root=None)
            return self._ok(stored=sent)
        for entry in body["spans"]:
            assert set(entry) == {"seq", "span"}, entry
            if entry["seq"] <= 0 or not entry["span"].get("parent_call_id"):
                raise PlatformError("malformed delta", status_code=400)
        root = body["root"]
        if body["ended"]:
            for entry in body["spans"]:
                run["spans"][entry["span"]["call_id"]] = (entry["seq"], entry["span"])
            rows = [span for _seq, span in sorted(run["spans"].values(), key=lambda row: row[0])]
            run.update(trace=[root, *rows], spans={}, root=None)  # assembled: the rows are gone
        elif run["trace"] is not None and any(s["parent_call_id"] is None and s.get("t1") for s in run["trace"]):
            # After the end (background children): merged into the stored trace by call_id.
            by_id = {span["call_id"]: span for span in run["trace"]}
            for entry in body["spans"]:
                by_id[entry["span"]["call_id"]] = entry["span"]
            if root is not None:
                by_id[root["call_id"]] = {**by_id.get(root["call_id"], {}), **root}
            run["trace"] = list(by_id.values())
        else:
            if run["trace"] is not None:  # a whole save mid-run, then deltas: its spans become the rows
                stored_root = next(s for s in run["trace"] if s["parent_call_id"] is None)
                run["spans"] = {s["call_id"]: (i + 1, s) for i, s in enumerate(run["trace"]) if s is not stored_root}
                run.update(trace=None, root=stored_root)
            for entry in body["spans"]:
                run["spans"][entry["span"]["call_id"]] = (entry["seq"], entry["span"])
            if root is not None:  # a header merges into the stored root, never wiping memory, input or session
                run["root"] = {**(run["root"] or {}), **root}
        return self._ok(stored=sent)


async def until_joined(key):
    """The run's flight has a round queued: a save joined it behind the one on the wire."""
    while (flight := PlatformTracingProvider._flights.get(key)) is None or flight.next is None:
        await asyncio.sleep(0)


def tool_loop(steps, probe=None):
    def handler(messages):
        done = sum(1 for m in messages if m.role == "tool")
        if done >= steps:
            return "done"
        return Message(
            role="assistant", content=[ToolUseContent(id=f"c{done}", name="step", input={})], stop_reason="tool_use"
        )

    async def step() -> str:
        if probe is not None:
            await probe()
        return "out " * 100

    provider = PlatformTracingProvider.configured(platform_config=CONFIG)
    agent = Agent(
        name="agent",
        model=TestModel(handler=handler),
        tools=[Tool(name="step", handler=step)],
        tracing_provider=provider,
        max_iter=steps + 5,
    )
    return agent, provider


async def run(agent, provider):
    ctx = RunContext(tracing_provider=provider)
    ctx.platform_config = CONFIG
    set_run_context(ctx)
    out = await agent(prompt="go").collect()
    assert out.status.code == "success", out.error
    return ctx, out


async def read_back(provider, run_id):
    PlatformTracingProvider._recent.clear()  # from the platform, not this process's cache
    reader = RunContext(parent_id=run_id, tracing_provider=provider)
    reader.platform_config = CONFIG
    return await provider.get(reader)


def same_trace(stored, local, server):
    """What the platform holds is what the run produced (raw JSON), and get() returns every span."""
    assert {s.call_id for s in stored.as_records()} == set(local._trace.keys())
    raw = server.runs[local.id]["trace"]
    want = json.loads(json.dumps(local._trace.model_dump(), default=str))
    got = {s["call_id"]: s for s in raw}
    for span in want:
        diff = {k for k in set(got[span["call_id"]]) | set(span) if got[span["call_id"]].get(k) != span.get(k)}
        # Usage counters about saving itself are added after the last save lands.
        diff.discard("usage") if span["parent_call_id"] is None else None
        assert not diff, (span["path"], diff)


class TestNegotiation:
    async def test_without_advertisement_saves_are_whole_and_plain(self):
        server = FakePlatform(advertise=False)
        agent, provider = tool_loop(3)
        with patch("timbal.platform.utils._request", server.request):
            ctx, _ = await run(agent, provider)
            stored = await read_back(provider, ctx.id)
        assert {p["kind"] for p in server.patches} == {"full"}
        assert not any(p["gzip"] for p in server.patches)
        same_trace(stored, ctx, server)

    async def test_switches_to_gzipped_deltas_once_advertised(self):
        server = FakePlatform()
        agent, provider = tool_loop(5)
        with patch("timbal.platform.utils._request", server.request):
            ctx, _ = await run(agent, provider)
            stored = await read_back(provider, ctx.id)
        kinds = [p["kind"] for p in server.patches]
        assert kinds[0] == "full" and set(kinds[1:]) == {"delta"}
        assert all(p["gzip"] for p in server.patches[1:])
        same_trace(stored, ctx, server)

    async def test_a_host_that_rejects_deltas_gets_whole_traces(self):
        server = FakePlatform(deltas=False)
        agent, provider = tool_loop(3)
        server.deltas = True  # advertises at first...
        with patch("timbal.platform.utils._request", server.request):
            ctx = RunContext(tracing_provider=provider)
            ctx.platform_config = CONFIG
            set_run_context(ctx)
            server.deltas = False  # ...then rejects
            PlatformTracingProvider._host_caps(CONFIG.host).delta = True
            out = await agent(prompt="go").collect()
            stored = await read_back(provider, ctx.id)
        assert out.status.code == "success"
        assert PlatformTracingProvider._host_caps(CONFIG.host).delta_rejected
        assert server.patches and {p["kind"] for p in server.patches} == {"full"}
        same_trace(stored, ctx, server)

    async def test_a_host_that_cannot_decompress_gets_plain_bodies(self):
        server = FakePlatform(gzip_ok=False)
        agent, provider = tool_loop(2)
        PlatformTracingProvider._host_caps(CONFIG.host).gzip = True
        with patch("timbal.platform.utils._request", server.request):
            ctx, _ = await run(agent, provider)
            stored = await read_back(provider, ctx.id)
        assert PlatformTracingProvider._host_caps(CONFIG.host).gzip_rejected
        assert server.patches and not any(p["gzip"] for p in server.patches)
        same_trace(stored, ctx, server)

    async def test_forced_gzip_on_todays_server_falls_back(self):
        # Today's API has no decompression layer: its JSON extractor answers a gzip body with 400.
        server = FakePlatform(advertise=False, gzip_ok=False)
        agent, _ = tool_loop(2)
        provider = PlatformTracingProvider.configured(platform_config=CONFIG, compress=True)
        agent.tracing_provider = provider
        with patch("timbal.platform.utils._request", server.request):
            ctx, out = await run(agent, provider)
            stored = await read_back(provider, ctx.id)
        assert "tracing:save_failed" not in out.usage
        assert PlatformTracingProvider._host_caps(CONFIG.host).gzip_rejected
        same_trace(stored, ctx, server)

    async def test_a_rejection_outlives_later_advertisements(self):
        server = FakePlatform(deltas=False)  # advertises gzip, takes it, rejects deltas
        PlatformTracingProvider._host_caps(CONFIG.host).delta = True
        agent, provider = tool_loop(3)
        with patch("timbal.platform.utils._request", server.request):
            await run(agent, provider)
            server.advertise, server.deltas = True, False
            server._headers = lambda: {"Accept-Patch": DELTA_MEDIA_TYPE}
            server.patches.clear()
            await run(agent, provider)
        assert server.patches and {p["kind"] for p in server.patches} == {"full"}

    async def test_a_bad_save_is_not_taken_for_a_rejected_format(self):
        PlatformTracingProvider._host_caps(CONFIG.host).delta = True
        PlatformTracingProvider._host_caps(CONFIG.host).gzip = True
        calls = []

        async def always_400(*_args, **kwargs):
            calls.append(kwargs.get("headers", {}))
            raise PlatformError("bad request", status_code=400)

        agent, provider = tool_loop(1)
        ctx = RunContext(tracing_provider=provider)
        ctx.platform_config = CONFIG
        with patch("timbal.platform.utils._request", always_400), pytest.raises(PlatformError):
            await provider._store(ctx)
        caps = PlatformTracingProvider._host_caps(CONFIG.host)
        assert not caps.delta_rejected and not caps.gzip_rejected
        assert [h.get("Content-Encoding") for h in calls] == ["gzip", None]  # retried plain once, then raised


class TestDeltas:
    async def test_each_finished_span_goes_once_and_the_root_at_the_end(self):
        server = FakePlatform()
        PlatformTracingProvider._host_caps(CONFIG.host).delta = True
        agent, provider = tool_loop(6)
        with patch("timbal.platform.utils._request", server.request):
            ctx, _ = await run(agent, provider)
        deltas = [p["body"] for p in server.patches]
        sent = [s["span"]["call_id"] for body in deltas for s in body["spans"]]
        assert len(sent) == len(set(sent)) == len(ctx._trace) - 1
        start_order = {call_id: i for i, call_id in enumerate(ctx._trace.keys())}
        assert next(iter(ctx._trace.values())).parent_call_id is None  # the root is index 0
        assert all(s["seq"] == start_order[s["span"]["call_id"]] for body in deltas for s in body["spans"])
        assert [b["ended"] for b in deltas].count(True) == 1 and deltas[-1]["ended"]
        # The running root goes whole until it is stored with its memory, then as a header.
        whole = [i for i, body in enumerate(deltas[:-1]) if "memory" in body["root"]]
        assert whole and whole == list(range(len(whole))) and deltas[whole[-1]]["root"]["memory"]
        assert all(set(body["root"]) == {"call_id", "path", "parent_call_id", "t0"} for body in deltas[len(whole) : -1])
        assert deltas[-1]["root"]["memory"]

    async def test_a_failed_save_is_resent_by_the_next(self):
        server = FakePlatform()
        PlatformTracingProvider._host_caps(CONFIG.host).delta = True
        fail_once = {"done": False}

        async def probe():
            if not fail_once["done"]:
                fail_once["done"] = True
                server.fail_next = 1

        agent, provider = tool_loop(4, probe)
        with patch("timbal.platform.utils._request", server.request):
            ctx, out = await run(agent, provider)
            stored = await read_back(provider, ctx.id)
        assert out.usage.get("tracing:save_failed") == 1
        same_trace(stored, ctx, server)

    async def test_a_run_that_died_reads_back_its_finished_spans(self):
        server = FakePlatform()
        PlatformTracingProvider._host_caps(CONFIG.host).delta = True
        seen = []

        async def probe():
            ctx = get_run_context()
            stored = await read_back(PlatformTracingProvider.configured(platform_config=CONFIG), ctx.id)
            seen.append(sorted(s.path for s in stored.as_records()) if stored else None)

        agent, provider = tool_loop(3, probe)
        with patch("timbal.platform.utils._request", server.request):
            await run(agent, provider)
        assert seen[-1].count("agent.step") == 2 and "agent.llm" in seen[-1]  # earlier calls, as a crash leaves them

    async def test_a_run_that_died_keeps_its_prompt_and_history(self):
        """Mid-run reads (as a crash leaves the run) find the root's input and memory, also after header deltas."""
        server = FakePlatform()
        PlatformTracingProvider._host_caps(CONFIG.host).delta = True
        roots = []

        async def probe():
            ctx = get_run_context()
            stored = await read_back(PlatformTracingProvider.configured(platform_config=CONFIG), ctx.id)
            if stored is not None:
                roots.append(next(s for s in stored.as_records() if s.parent_call_id is None))

        agent, provider = tool_loop(4, probe)
        with patch("timbal.platform.utils._request", server.request):
            await run(agent, provider)
        headers_only = [p for p in server.patches if not p["body"]["ended"] and "memory" not in p["body"]["root"]]
        assert headers_only  # later saves did go out as headers
        last = roots[-1]
        assert last.input["prompt"] == "go"
        assert [m["role"] if isinstance(m, dict) else m.role for m in last.memory][:1] == ["user"]

    async def test_the_root_goes_whole_again_until_it_is_stored(self):
        server = FakePlatform()
        PlatformTracingProvider._host_caps(CONFIG.host).delta = True
        original = server.request
        dropped = {"n": 0}

        async def request(method, path, headers=None, content=None, **kwargs):
            body = json.loads(
                gzip.decompress(content) if (headers or {}).get("Content-Encoding") == "gzip" else content
            )
            if method == "PATCH" and "memory" in (body.get("root") or {}) and not body["ended"] and not dropped["n"]:
                dropped["n"] = 1
                raise PlatformError("upstream", status_code=503)
            return await original(method, path, headers=headers, content=content, **kwargs)

        agent, provider = tool_loop(4)
        with patch("timbal.platform.utils._request", request):
            await run(agent, provider)
        mid = [p["body"] for p in server.patches if not p["body"]["ended"]]
        assert dropped["n"] == 1 and "memory" in mid[0]["root"]  # the failed whole root went again

    async def test_a_span_changed_after_the_end_is_resent_alone(self):
        server = FakePlatform()
        PlatformTracingProvider._host_caps(CONFIG.host).delta = True
        agent, provider = tool_loop(2)
        with patch("timbal.platform.utils._request", server.request):
            ctx, _ = await run(agent, provider)
            step = next(s for s in ctx._trace.as_records() if s.path == "agent.step")
            step._output_dump = "late real output"
            await provider._store(ctx)
            stored = await read_back(provider, ctx.id)
        last = server.patches[-1]["body"]
        assert [s["span"]["call_id"] for s in last["spans"]] == [step.call_id]
        assert next(s for s in stored.as_records() if s.call_id == step.call_id).output == "late real output"

    @pytest.mark.parametrize("clock", ["clock", "coarse_clock"])
    async def test_snapshots_keep_rising_in_a_new_process(self, clock, monkeypatch):
        from timbal.state.tracing.providers import platform as platform_module

        if clock == "coarse_clock":
            # time_ns() in 15.6 ms ticks, as on Windows before Python 3.13: the new process's first number
            # used to be that tick, equal to what the first process last sent (the platform drops it).
            real, tick = platform_module.time_ns, 15_625_000
            monkeypatch.setattr(platform_module, "time_ns", lambda: real() // tick * tick)
        server = FakePlatform()
        PlatformTracingProvider._host_caps(CONFIG.host).delta = True
        agent, provider = tool_loop(2)
        with patch("timbal.platform.utils._request", server.request):
            ctx, _ = await run(agent, provider)
            first = max(p["body"]["snapshot"] for p in server.patches)
            # Another process continues the run (an approval resumed elsewhere): no counter carried over, and
            # its own clock calibration (in the same 15.6 ms tick as the last save, on Windows before 3.13).
            platform_module._last_snapshot = 0
            platform_module._epoch_offset_ns = None
            PlatformTracingProvider._delta_runs.clear()
            step = next(s for s in ctx._trace.as_records() if s.path == "agent.step")
            step._output_dump = "resumed elsewhere"
            await provider._store(ctx)
            stored = await read_back(provider, ctx.id)
        assert server.patches[-1]["body"]["snapshot"] > first
        assert next(s for s in stored.as_records() if s.call_id == step.call_id).output == "resumed elsewhere"

    async def test_bytes(self):
        full_server, delta_server = FakePlatform(advertise=False), FakePlatform()
        for server, deltas in ((full_server, False), (delta_server, True)):
            PlatformTracingProvider._caps.clear()
            PlatformTracingProvider._delta_runs.clear()
            PlatformTracingProvider._host_caps(CONFIG.host).delta = deltas
            PlatformTracingProvider._host_caps(CONFIG.host).gzip = deltas
            agent, provider = tool_loop(20)
            with patch("timbal.platform.utils._request", server.request):
                await run(agent, provider)
        full = sum(p["bytes"] for p in full_server.patches)
        delta = sum(p["bytes"] for p in delta_server.patches)
        assert delta * 20 < full, (full, delta)


class TestSingleFlight:
    """rfcs/platform-trace-storage.md 0.1–0.4 on the SDK side."""

    @staticmethod
    def held(server):
        """The first PATCH waits for ``release``; the rest go through."""
        first_sent, release = asyncio.Event(), asyncio.Event()
        original = server.request

        async def request(method, path, **kwargs):
            if method == "PATCH" and not first_sent.is_set():
                first_sent.set()
                await release.wait()
            return await original(method, path, **kwargs)

        return request, first_sent, release

    async def test_saves_arriving_meanwhile_go_out_once_from_the_latest_state(self):
        server = FakePlatform(advertise=False)
        request, first_sent, release = self.held(server)
        _, provider = tool_loop(1)
        ctx = RunContext(tracing_provider=provider)
        ctx.platform_config = CONFIG
        ctx._trace["r"] = {"call_id": "r", "path": "a", "t0": 0, "parent_call_id": None}
        with patch("timbal.platform.utils._request", request):
            first = asyncio.create_task(provider._store(ctx))
            await first_sent.wait()
            later = []
            for i in range(5):
                ctx._trace[f"c{i}"] = {"call_id": f"c{i}", "path": "a.x", "t0": i, "parent_call_id": "r"}
                later.append(asyncio.create_task(provider._store(ctx)))
            await asyncio.sleep(0.01)
            release.set()
            await asyncio.gather(first, *later)
        assert len(server.patches) == 2
        assert len(server.patches[1]["body"]["trace"]) == 6  # every save's state, in one round
        assert not PlatformTracingProvider._flights

    async def test_every_patch_carries_its_snapshot_and_they_rise(self):
        seen = []

        async def request(*_args, headers=None, **_):
            seen.append(int(headers["X-Timbal-Snapshot"]))
            return SimpleNamespace(raise_for_status=lambda: None, headers={}, json=lambda: {})

        agent, provider = tool_loop(3)
        with patch("timbal.platform.utils._request", request):
            await run(agent, provider)
        assert len(seen) > 2 and seen == sorted(set(seen))

    async def test_continues_above_the_snapshot_the_platform_holds(self):
        # Another process' clock runs ahead. Taken from the clock snapshots are numbered by, not from the last
        # number given: there is none when this is the first test of its process to save.
        ahead = platform_module._epoch_ns() + 10**15
        sent = []

        async def request(*_args, headers=None, **_):
            sent.append(int(headers["X-Timbal-Snapshot"]))
            return SimpleNamespace(
                raise_for_status=lambda: None, headers={"X-Timbal-Snapshot": str(ahead)}, json=lambda: {}
            )

        _, provider = tool_loop(1)
        ctx = RunContext(tracing_provider=provider)
        ctx.platform_config = CONFIG
        with patch("timbal.platform.utils._request", request):
            await provider._store(ctx)
            await provider._store(ctx)
        assert sent[0] < ahead < sent[1]

    async def test_the_end_of_a_run_waits_for_the_save_on_the_wire_and_still_goes(self):
        server = FakePlatform()
        PlatformTracingProvider._host_caps(CONFIG.host).delta = True
        request, first_sent, release = self.held(server)
        agent, provider = tool_loop(3)
        with patch("timbal.platform.utils._request", request):
            ctx = RunContext(tracing_provider=provider)
            ctx.platform_config = CONFIG
            set_run_context(ctx)
            running = asyncio.create_task(agent(prompt="go").collect())
            await first_sent.wait()
            await asyncio.sleep(0.05)
            release.set()
            out = await running
            stored = await read_back(provider, ctx.id)
        assert out.status.code == "success"
        assert server.patches[-1]["body"]["ended"]
        same_trace(stored, ctx, server)

    async def test_a_read_waits_for_a_round_queued_behind_the_wire(self):
        server = FakePlatform(advertise=False)
        request, first_sent, release = self.held(server)
        _, provider = tool_loop(1)
        ctx = RunContext(tracing_provider=provider)
        ctx.platform_config = CONFIG
        ctx._trace["c0"] = {"call_id": "c0", "path": "x", "t0": 0, "parent_call_id": None}
        reader = RunContext(parent_id=ctx.id, tracing_provider=provider)
        reader.platform_config = CONFIG
        with patch("timbal.platform.utils._request", request):
            first = asyncio.create_task(provider._store(ctx))
            await first_sent.wait()
            ctx._trace["c0"].metadata["final"] = True
            final = asyncio.create_task(provider._store(ctx))  # queued behind the first
            read = asyncio.create_task(provider.get(reader))
            await asyncio.sleep(0.01)
            release.set()
            trace = await read
            await asyncio.gather(first, final)
        assert trace["c0"].metadata == {"final": True}

    async def test_resume_reads_ask_for_the_projection_once_advertised(self):
        calls = []

        async def request(*_args, params=None, **_):
            calls.append(params)
            return SimpleNamespace(
                raise_for_status=lambda: None,
                headers={"X-Timbal-Select": "resume"},
                json=lambda: {"trace": [{"call_id": "r", "path": "a", "t0": 0, "parent_call_id": None}]},
            )

        _, provider = tool_loop(1)
        with patch("timbal.platform.utils._request", request):
            for _ in range(2):
                await read_back(provider, "some-run")
        assert calls == [None, {"select": "resume"}]

    @pytest.mark.parametrize(
        ("body", "expected"),
        [
            ({"parent_id": "4270383", "parent_idempotency_key": "run-a"}, "run-a"),  # newer hosts: the run id
            ({"parent_id": "4270383"}, "4270383"),  # older hosts: the row id, which a GET also resolves
            ({"parent_id": None, "parent_idempotency_key": None}, None),  # a thread root
        ],
    )
    async def test_the_parent_is_read_as_a_run_id_when_the_host_gives_one(self, body, expected):
        async def request(*_args, **_):
            trace = [{"call_id": "r", "path": "a", "t0": 0, "parent_call_id": None}]
            return SimpleNamespace(raise_for_status=lambda: None, headers={}, json=lambda: {"trace": trace, **body})

        _, provider = tool_loop(1)
        with patch("timbal.platform.utils._request", request):
            stored = await read_back(provider, "some-run")
        assert stored.parent_id == expected

    async def test_every_agent_span_is_in_the_resume_projection(self):
        """The platform keeps the root and every span with a ``memory`` key: a nested agent cut before it
        assembled its memory must still be there, or the next turn loses the conversation."""
        from timbal import Workflow

        gate = asyncio.Event()

        async def slow_prompt() -> str:
            await gate.wait()
            return "sys"

        inner = Agent(name="inner", model=TestModel(responses=["hi"]), system_prompt=slow_prompt)
        flow = Workflow(name="flow").step(inner, prompt="go")
        server = FakePlatform(advertise=False)
        provider = PlatformTracingProvider.configured(platform_config=CONFIG)
        with patch("timbal.platform.utils._request", server.request):
            ctx = RunContext(tracing_provider=provider)
            ctx.platform_config = CONFIG
            set_run_context(ctx)
            running = asyncio.create_task(flow(prompt="go").collect())
            for _ in range(50):
                await asyncio.sleep(0.01)
                if any(s.path == "flow.inner" for s in ctx._trace.as_records()):
                    break
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)  # the run settles as cancelled
            await provider._store(ctx)
        stored = server.runs[ctx.id]["trace"]
        projection = [s for s in stored if s["parent_call_id"] is None or "memory" in s]
        assert [s["path"] for s in projection] == ["flow", "flow.inner"]
        assert projection[1]["memory"] is None
        assert all("memory" not in s for s in stored if s["path"] not in ("flow", "flow.inner"))

    @staticmethod
    def stalled_run():
        """A context whose first PATCH hangs until ``release``; every PATCH is recorded."""
        sent, first_sent, release = [], asyncio.Event(), asyncio.Event()

        async def request(*_args, content=None, **_):
            sent.append(json.loads(content))
            if len(sent) == 1:
                first_sent.set()
                await release.wait()
            return SimpleNamespace(raise_for_status=lambda: None, headers={}, json=lambda: {})

        _, provider = tool_loop(1)
        ctx = RunContext(tracing_provider=provider)
        ctx.platform_config = CONFIG
        ctx._trace["r"] = {"call_id": "r", "path": "a", "t0": 0, "parent_call_id": None}
        return ctx, provider, request, sent, first_sent, release

    async def test_mid_run_saves_wait_out_a_stalled_patch_and_the_end_does_not(self, monkeypatch):
        monkeypatch.setattr(PlatformTracingProvider, "_stall_s", 0.05)
        ctx, provider, request, sent, first_sent, release = self.stalled_run()
        with patch("timbal.platform.utils._request", request):
            first = asyncio.create_task(provider._store(ctx))
            await first_sent.wait()
            mid = asyncio.create_task(provider._store(ctx))
            await asyncio.sleep(0.15)
            assert len(sent) == 1  # mid-run: superseded anyway, waits however long it takes
            ctx._trace["r"].t1 = 1
            end = asyncio.create_task(provider._store(ctx))
            await asyncio.wait_for(end, 1)  # the end goes past the stalled one
            assert mid.done()  # and carried the mid-run state with it
            assert len(sent) == 2 and sent[1]["trace"][0]["t1"] == 1
            release.set()
            await first

    @pytest.mark.parametrize("stalled_fails", [False, True])
    async def test_a_stalled_patch_ending_after_the_end_landed_is_settled_quietly(self, monkeypatch, stalled_fails):
        monkeypatch.setattr(PlatformTracingProvider, "_stall_s", 0.05)
        sent, first_sent, release = [], asyncio.Event(), asyncio.Event()

        async def request(*_args, content=None, **_):
            sent.append(json.loads(content))
            if len(sent) == 1:
                first_sent.set()
                await release.wait()
                if stalled_fails:
                    raise PlatformError("upstream", status_code=503)
            return SimpleNamespace(raise_for_status=lambda: None, headers={}, json=lambda: {})

        _, provider = tool_loop(1)
        ctx = RunContext(tracing_provider=provider)
        ctx.platform_config = CONFIG
        ctx._trace["r"] = {"call_id": "r", "path": "a", "t0": 0, "parent_call_id": None}
        loop = asyncio.get_running_loop()
        errors = []
        loop.set_exception_handler(lambda _loop, context: errors.append(context))
        try:
            with patch("timbal.platform.utils._request", request):
                first = asyncio.create_task(provider._store(ctx))
                await first_sent.wait()
                ctx._trace["r"].t1 = 1
                await asyncio.wait_for(provider._store(ctx), 1)  # the end goes past the stalled PATCH and lands
                release.set()
                await asyncio.wait_for(first, 1)  # settled by the end's save: it carried this state and newer
                await asyncio.sleep(0.05)
        finally:
            loop.set_exception_handler(None)
        assert errors == []  # the stalled PATCH finishing late raises nothing in its done callback
        assert len(sent) == 2 and not PlatformTracingProvider._retries  # and nothing is resent

    async def test_never_more_than_two_patches_on_the_wire(self, monkeypatch):
        monkeypatch.setattr(PlatformTracingProvider, "_stall_s", 0.0)
        on_wire, most, release = [0], [0], asyncio.Event()

        async def request(*_args, **_):
            on_wire[0] += 1
            most[0] = max(most[0], on_wire[0])
            await release.wait()
            on_wire[0] -= 1
            return SimpleNamespace(raise_for_status=lambda: None, headers={}, json=lambda: {})

        _, provider = tool_loop(1)
        ctx = RunContext(tracing_provider=provider)
        ctx.platform_config = CONFIG
        ctx._trace["r"] = {"call_id": "r", "path": "a", "t0": 0, "t1": 1, "parent_call_id": None}
        with patch("timbal.platform.utils._request", request):
            saves = []
            for _ in range(6):  # each arrives once the one before is on the wire
                saves.append(asyncio.create_task(provider._store(ctx)))
                await asyncio.sleep(0.01)
            assert most[0] == 2
            release.set()
            await asyncio.gather(*saves)

    @pytest.mark.parametrize("writer", ["another_process", "this_process"])
    async def test_a_save_dropped_as_stale_is_resent_only_if_the_newer_one_was_not_ours(self, writer):
        _, provider = tool_loop(1)
        ctx = RunContext(tracing_provider=provider)
        ctx.platform_config = CONFIG
        _, _, key = PlatformTracingProvider._target(ctx)
        sent = []

        async def request(*_args, headers=None, **_):
            mine = int(headers["X-Timbal-Snapshot"])
            sent.append(mine)
            if len(sent) > 1:
                return SimpleNamespace(raise_for_status=lambda: None, headers={}, json=lambda: {})
            stored = mine + 10**12  # the platform already holds something newer
            if writer == "this_process":
                PlatformTracingProvider._own(key).append(stored)  # a round of ours that overtook this one
            headers_out = {"X-Timbal-Snapshot": str(stored), "X-Timbal-Snapshot-Applied": "false"}
            return SimpleNamespace(raise_for_status=lambda: None, headers=headers_out, json=lambda: {})

        with patch("timbal.platform.utils._request", request):
            await provider._store(ctx)
        if writer == "another_process":
            assert len(sent) == 2 and sent[1] > sent[0] + 10**12  # numbered above what it lost to
        else:
            assert len(sent) == 1

    async def test_413_is_not_a_format_rejection(self):
        PlatformTracingProvider._host_caps(CONFIG.host).delta = True
        PlatformTracingProvider._host_caps(CONFIG.host).gzip = True
        calls = []

        async def too_large(*_args, **kwargs):
            calls.append(kwargs["headers"])
            raise PlatformError("payload too large", status_code=413)

        _, provider = tool_loop(1)
        ctx = RunContext(tracing_provider=provider)
        ctx.platform_config = CONFIG
        with patch("timbal.platform.utils._request", too_large), pytest.raises(PlatformError):
            await provider._store(ctx)
        caps = PlatformTracingProvider._host_caps(CONFIG.host)
        assert not caps.delta_rejected and not caps.gzip_rejected
        # Sent once more, whole and cut down; a second 413 is final (never resent as is).
        assert len(calls) == 2 and calls[1]["Content-Type"] == "application/json"

    async def test_a_run_over_the_limit_still_stores_its_end_cut_down(self, monkeypatch):
        monkeypatch.setattr(platform_module, "_SHRINK_TARGET_BYTES", 400_000)
        server = FakePlatform(limit=500_000)
        PlatformTracingProvider._host_caps(CONFIG.host).delta = True

        def huge() -> str:
            return "x" * 2_000_000

        def handler(messages):
            if any(m.role == "tool" for m in messages):
                return "done"
            return Message(
                role="assistant", content=[ToolUseContent(id="h1", name="huge", input={})], stop_reason="tool_use"
            )

        provider = PlatformTracingProvider.configured(platform_config=CONFIG)
        agent = Agent(
            name="agent",
            model=TestModel(handler=handler),
            tools=[Tool(name="huge", handler=huge)],
            tracing_provider=provider,
            tool_result_limit=None,
        )
        with patch("timbal.platform.utils._request", server.request):
            ctx, out = await run(agent, provider)
        assert "tracing:save_failed" not in out.usage
        stored = server.runs[ctx.id]["trace"]
        root = next(s for s in stored if s["parent_call_id"] is None)
        assert root["t1"] is not None and root["metadata"]["trace_truncated"]["strings_cut"] >= 1
        tool = next(s for s in stored if s["path"] == "agent.huge")
        assert len(tool["output"]) < 100_000 and "characters cut" in tool["output"]
        assert {s["call_id"] for s in stored} == set(ctx._trace.keys())  # every span, just shorter
        local = next(s for s in ctx._trace.values() if s.path == "agent.huge")
        assert len(local.output) == 2_000_000  # the run's own trace is untouched
        assert all(p["kind"] == "full" for p in server.patches[-2:])  # whole from the refusal on


def masking(secret, mark):
    """A trace redactor that replaces ``secret`` with ``mark`` in every string it is given."""

    def redact(value):
        if isinstance(value, str):
            return value.replace(secret, mark)
        if isinstance(value, dict):
            return {k: redact(v) for k, v in value.items()}
        if isinstance(value, list):
            return [redact(v) for v in value]
        return value

    return redact


def whole_root(server, run_id):
    run = server.runs.get(run_id)
    return run["trace"][0] if run and run.get("trace") else None


class TestOrderAcrossPaths:
    """Whatever path sends a save of a run (its own round, a resend after "send it again", a flush on another event
    loop, a background retry), the host never goes back to an older state than one it took; a newer save that
    failed stays owed until a PATCH carrying it lands; once the newest save landed, nothing is left to send. With
    a trace redactor throughout: that is where a copy of the run, made when it was saved, could be sent late."""

    @staticmethod
    def context(provider, **root):
        ctx = RunContext(tracing_provider=provider)
        ctx.platform_config = CONFIG
        ctx._trace["r"] = {"call_id": "r", "path": "a", "t0": 0, "parent_call_id": None, **root}
        return ctx

    @staticmethod
    def recording(server, run_id, ends):
        """``server.request``, recording the stored root's t1 after every PATCH the host takes."""

        async def request(method, path, headers=None, **kwargs):
            response = await server.request(method, path, headers=headers, **kwargs)
            if method == "PATCH" and (root := whole_root(server, run_id)) is not None:
                ends.append(root.get("t1"))
            return response

        return request

    @staticmethod
    def never_back(ends):
        """The host held the run's end, and kept it from then on."""
        return 5 in ends and all(end == 5 for end in ends[ends.index(5) :])

    @staticmethod
    def redacting():
        return PlatformTracingProvider.configured(platform_config=CONFIG, _trace_redactor=lambda value: value)

    @staticmethod
    def answering(answer, patches):
        """What the host answers the stalled PATCH: "send it again", one way or another."""
        if answer == "too_large":
            raise PlatformError("payload too large", status_code=413)
        if answer == "format_rejected":
            raise PlatformError("unsupported media type", status_code=415)
        if answer == "stale":  # another process holds something newer than this PATCH
            stored = {"X-Timbal-Snapshot": str(patches[0] + 10**12), "X-Timbal-Snapshot-Applied": "false"}
            return SimpleNamespace(raise_for_status=lambda: None, headers=stored, json=lambda: {})
        return SimpleNamespace(raise_for_status=lambda: None, headers={}, json=lambda: {})  # no echo of a delta

    ANSWERS = pytest.mark.parametrize("answer", ["too_large", "format_rejected", "stale", "delta_unconfirmed"])

    @pytest.mark.parametrize("end", ["landed", "stored_but_unanswered", "not_yet_at_the_host"])
    @ANSWERS
    async def test_a_stalled_patch_sent_again_never_takes_the_run_back(self, monkeypatch, answer, end):
        """A mid-run PATCH stalls; the run's end goes past it (and has landed, or is stored with its answer on the
        way, or has not reached the host yet); then the stalled one is answered "send it again"."""
        monkeypatch.setattr(PlatformTracingProvider, "_stall_s", 0.0)
        server, ends, patches = FakePlatform(), [], []
        first_sent, answer_it, answered = asyncio.Event(), asyncio.Event(), asyncio.Event()
        end_held, release_end = asyncio.Event(), asyncio.Event()
        if answer in ("format_rejected", "delta_unconfirmed"):
            PlatformTracingProvider._host_caps(CONFIG.host).delta = True  # the first PATCH goes out as a delta
        provider = self.redacting()
        ctx = self.context(provider)
        record = self.recording(server, ctx.id, ends)

        async def request(method, path, headers=None, **kwargs):
            if method != "PATCH":
                return await server.request(method, path, headers=headers, **kwargs)
            patches.append(int(headers["X-Timbal-Snapshot"]))
            if len(patches) == 1:
                first_sent.set()
                await answer_it.wait()
                answered.set()  # whatever the save does next with this answer, it does in this step
                return self.answering(answer, patches)
            PlatformTracingProvider._host_caps(CONFIG.host).delta = False  # what follows goes whole
            if len(patches) == 2 and end == "not_yet_at_the_host":
                end_held.set()
                await release_end.wait()
            response = await record(method, path, headers=headers, **kwargs)
            if len(patches) == 2 and end == "stored_but_unanswered":
                end_held.set()
                await release_end.wait()
            return response

        with patch("timbal.platform.utils._request", request):
            mid_run = asyncio.create_task(provider.put(ctx))
            await asyncio.wait_for(first_sent.wait(), 5)
            ctx._trace["r"].t1 = 5
            ending = asyncio.create_task(provider.put(ctx))  # goes past the PATCH on the wire
            if end == "landed":
                await asyncio.wait_for(ending, 5)
            else:
                await asyncio.wait_for(end_held.wait(), 5)
            answer_it.set()
            await asyncio.wait_for(answered.wait(), 5)
            release_end.set()
            await asyncio.wait_for(asyncio.gather(mid_run, ending), 5)
        assert self.never_back(ends)
        assert not PlatformTracingProvider._unsaved  # the end landed: nothing owed

    @ANSWERS
    async def test_a_stalled_patch_sent_again_after_the_end_failed_stores_the_end(self, monkeypatch, answer):
        """The end went past the stalled PATCH, was stored by the host, and failed on the way back (a timeout).
        What the stalled save sends again is the run as it is now, with its end: the host keeps it, and the run is
        no longer owed once that lands."""
        import httpx

        monkeypatch.setattr(PlatformTracingProvider, "_stall_s", 0.0)
        monkeypatch.setattr(PlatformTracingProvider, "_retry_delays_s", (300.0,))  # the resend itself is not this test
        server, ends, patches = FakePlatform(), [], []
        first_sent, answer_it = asyncio.Event(), asyncio.Event()
        if answer in ("format_rejected", "delta_unconfirmed"):
            PlatformTracingProvider._host_caps(CONFIG.host).delta = True
        provider = self.redacting()
        ctx = self.context(provider)
        record = self.recording(server, ctx.id, ends)

        async def request(method, path, headers=None, **kwargs):
            if method != "PATCH":
                return await server.request(method, path, headers=headers, **kwargs)
            patches.append(int(headers["X-Timbal-Snapshot"]))
            if len(patches) == 1:
                first_sent.set()
                await answer_it.wait()
                return self.answering(answer, patches)
            PlatformTracingProvider._host_caps(CONFIG.host).delta = False
            response = await record(method, path, headers=headers, **kwargs)
            if len(patches) == 2:
                raise httpx.ReadTimeout("the answer never came")  # the end: stored by the host
            return response

        with patch("timbal.platform.utils._request", request):
            mid_run = asyncio.create_task(ctx._save_trace(barrier=True))
            await asyncio.wait_for(first_sent.wait(), 5)
            ctx._trace["r"].t1 = 5
            assert await asyncio.wait_for(ctx._save_trace(barrier=True), 5) is False  # the end: stored, unanswered
            answer_it.set()
            assert await asyncio.wait_for(mid_run, 5) is True  # sent again: the run with its end, and it landed
        assert self.never_back(ends)
        assert not PlatformTracingProvider._unsaved
        assert not ctx._trace_unsaved

    async def test_when_every_save_of_the_end_fails_the_run_stays_owed_and_nothing_older_is_sent(self, monkeypatch):
        """The end fails, and so does everything the stalled save sends again: the run stays owed (resent in the
        background, reported by a flush), and no PATCH after the end was asked carries the run without it."""
        monkeypatch.setattr(PlatformTracingProvider, "_stall_s", 0.0)
        monkeypatch.setattr(PlatformTracingProvider, "_retry_delays_s", (300.0,))
        first_sent, answer_it, end_sent = asyncio.Event(), asyncio.Event(), asyncio.Event()
        bodies = []  # the root's t1 in each PATCH after the first

        async def request(method, content=None, **_):
            assert method == "PATCH"
            if not first_sent.is_set():
                first_sent.set()
                await answer_it.wait()
                raise PlatformError("payload too large", status_code=413)
            bodies.append(json.loads(content)["trace"][0].get("t1"))
            end_sent.set()
            raise PlatformError("upstream", status_code=503)

        provider = self.redacting()
        ctx = self.context(provider)
        _, _, key = PlatformTracingProvider._target(ctx)
        with patch("timbal.platform.utils._request", request):
            mid_run = asyncio.create_task(ctx._save_trace(barrier=True))
            await asyncio.wait_for(first_sent.wait(), 5)
            ctx._trace["r"].t1 = 5
            ending = asyncio.create_task(ctx._save_trace(barrier=True))
            await asyncio.wait_for(end_sent.wait(), 5)
            answer_it.set()
            assert await asyncio.wait_for(asyncio.gather(mid_run, ending), 5) == [False, False]
        assert bodies and all(t1 == 5 for t1 in bodies)
        assert PlatformTracingProvider._unsaved[key].context is ctx  # still owed
        assert key in PlatformTracingProvider._retries  # and resent in the background

    async def test_a_save_landing_on_another_loop_meanwhile_is_never_undone(self, monkeypatch):
        """A shutdown flush on another event loop stores the ended run while an older PATCH is still out here; that
        one is then answered "too large"."""
        import threading

        import httpx

        monkeypatch.setattr(PlatformTracingProvider, "_stall_s", 0.0)
        monkeypatch.setattr(PlatformTracingProvider, "_retry_delays_s", (300.0,))
        server, ends, patches = FakePlatform(), [], []
        first_sent, answer_it = asyncio.Event(), asyncio.Event()
        provider = self.redacting()
        ctx = self.context(provider)
        record = self.recording(server, ctx.id, ends)

        async def request(method, path, headers=None, **kwargs):
            if method != "PATCH":
                return await server.request(method, path, headers=headers, **kwargs)
            patches.append(int(headers["X-Timbal-Snapshot"]))
            if len(patches) == 1:
                first_sent.set()
                await answer_it.wait()
                raise PlatformError("payload too large", status_code=413)
            response = await record(method, path, headers=headers, **kwargs)
            if len(patches) == 2:
                raise httpx.ReadTimeout("the answer never came")  # the end, on this loop
            return response

        flushed = []
        with patch("timbal.platform.utils._request", request):
            mid_run = asyncio.create_task(ctx._save_trace(barrier=True))
            await asyncio.wait_for(first_sent.wait(), 5)
            ctx._trace["r"].t1 = 5
            assert await asyncio.wait_for(ctx._save_trace(barrier=True), 5) is False
            other_loop = threading.Thread(target=lambda: flushed.append(asyncio.run(provider.flush(5))))
            other_loop.start()
            await asyncio.wait_for(asyncio.to_thread(other_loop.join, 10), 15)
            assert flushed == [[]]  # the flush sent the ended run, and it landed
            answer_it.set()
            await asyncio.wait_for(mid_run, 5)
        assert self.never_back(ends)
        assert not PlatformTracingProvider._unsaved

    async def test_another_loops_save_at_the_moment_a_resend_is_decided_is_never_undone(self, monkeypatch):
        """Another event loop sends the ended run in the very step this loop's stalled PATCH is answered "too
        large", with no await between: what is sent again here is numbered above it, and must hold no less."""
        import threading

        monkeypatch.setattr(PlatformTracingProvider, "_stall_s", 3600.0)  # the end waits behind the stalled PATCH
        server, ends, patches = FakePlatform(), [], []
        first_sent, answer_it = asyncio.Event(), asyncio.Event()
        provider = self.redacting()
        ctx = self.context(provider)
        _, _, key = PlatformTracingProvider._target(ctx)
        record = self.recording(server, ctx.id, ends)
        flushed = []

        async def request(method, path, headers=None, **kwargs):
            if method != "PATCH":
                return await server.request(method, path, headers=headers, **kwargs)
            patches.append(int(headers["X-Timbal-Snapshot"]))
            if len(patches) == 1:
                first_sent.set()
                await answer_it.wait()
                other_loop = threading.Thread(target=lambda: flushed.append(asyncio.run(provider.flush(5))))
                other_loop.start()
                other_loop.join(10)  # blocking: nothing else runs on this loop meanwhile
                raise PlatformError("payload too large", status_code=413)
            return await record(method, path, headers=headers, **kwargs)

        with patch("timbal.platform.utils._request", request):
            mid_run = asyncio.create_task(ctx._save_trace(barrier=True))
            await asyncio.wait_for(first_sent.wait(), 5)
            ctx._trace["r"].t1 = 5
            ending = asyncio.create_task(ctx._save_trace(barrier=True))
            await asyncio.wait_for(until_joined(key), 5)
            answer_it.set()
            assert await asyncio.wait_for(asyncio.gather(mid_run, ending), 10) == [True, True]
        assert flushed == [[]]
        assert self.never_back(ends)
        assert not PlatformTracingProvider._unsaved

    async def test_a_flush_held_after_it_picked_a_run_never_sends_older_state(self, monkeypatch):
        """A flush on another loop picks the run and is held before it sends; this loop then stores the end."""
        import threading

        server, ends = FakePlatform(), []
        provider = self.redacting()
        ctx = self.context(provider)
        record = self.recording(server, ctx.id, ends)
        main, paused, go = threading.get_ident(), threading.Event(), threading.Event()
        first_sent, hold_first = asyncio.Event(), asyncio.Event()
        real_save = PlatformTracingProvider.__dict__["_save"].__func__
        sent = []

        async def save(cls, *args, **kwargs):
            if threading.get_ident() != main and not go.is_set():
                paused.set()
                go.wait(10)
            return await real_save(cls, *args, **kwargs)

        async def request(method, path, headers=None, **kwargs):
            if method == "PATCH":
                sent.append(None)
                if len(sent) == 1:
                    first_sent.set()
                    await hold_first.wait()
            return await record(method, path, headers=headers, **kwargs)

        monkeypatch.setattr(PlatformTracingProvider, "_save", classmethod(save))
        flushed = []
        with patch("timbal.platform.utils._request", request):
            mid_run = asyncio.create_task(ctx._save_trace(barrier=True))
            await asyncio.wait_for(first_sent.wait(), 5)
            other_loop = threading.Thread(target=lambda: flushed.append(asyncio.run(provider.flush(5))))
            other_loop.start()
            assert await asyncio.to_thread(paused.wait, 10)
            ctx._trace["r"].t1 = 5
            assert await asyncio.wait_for(ctx._save_trace(barrier=True), 5) is True  # the end, stored
            go.set()
            await asyncio.wait_for(asyncio.to_thread(other_loop.join, 10), 15)
            hold_first.set()
            await asyncio.wait_for(mid_run, 5)
        assert self.never_back(ends)
        assert flushed == [[]]

    async def test_a_save_held_between_put_and_sending_never_stores_older_state(self, monkeypatch):
        """This loop's save is held up after put(); another loop saves the run, ended, and that lands."""
        import threading

        server, ends = FakePlatform(), []
        provider = self.redacting()
        ctx = self.context(provider)
        main, held = threading.get_ident(), []
        real_target = PlatformTracingProvider.__dict__["_target"].__func__

        def target(cls, run_context):
            if threading.get_ident() == main and not held:
                held.append(None)
                ctx._trace["r"].t1 = 5
                other_loop = threading.Thread(target=lambda: asyncio.run(provider.put(ctx)))
                other_loop.start()
                other_loop.join(10)
                assert ends == [5]
            return real_target(cls, run_context)

        monkeypatch.setattr(PlatformTracingProvider, "_target", classmethod(target))
        with patch("timbal.platform.utils._request", self.recording(server, ctx.id, ends)):
            await asyncio.wait_for(provider.put(ctx), 10)
        assert held and self.never_back(ends)

    async def test_an_older_save_landing_late_does_not_settle_a_newer_one_that_failed(self, monkeypatch):
        """Another loop's save stores the run as it was before its end, and its answer is held; meanwhile this
        loop's save of the end fails. The run stays owed, and the flush reports it."""
        import threading

        monkeypatch.setattr(PlatformTracingProvider, "_retry_delays_s", (300.0,))
        server, ends, patches = FakePlatform(), [], []
        provider = self.redacting()
        ctx = self.context(provider)
        record = self.recording(server, ctx.id, ends)
        flush_stored, flush_answers = threading.Event(), threading.Event()

        async def request(method, path, headers=None, **kwargs):
            if method == "PATCH" and not patches:
                patches.append(None)
                raise PlatformError("upstream", status_code=503)  # the mid-run save: the run is owed
            if method == "PATCH" and len(patches) == 1:  # the flush, on the other loop
                patches.append(None)
                response = await record(method, path, headers=headers, **kwargs)
                flush_stored.set()
                flush_answers.wait(10)
                return response
            if method == "PATCH":  # the end, on this loop
                patches.append(None)
                raise PlatformError("upstream", status_code=503)
            return await record(method, path, headers=headers, **kwargs)

        flushed = []
        with patch("timbal.platform.utils._request", request):
            assert await asyncio.wait_for(ctx._save_trace(barrier=True), 5) is False
            other_loop = threading.Thread(target=lambda: flushed.append(asyncio.run(provider.flush(5))))
            other_loop.start()
            assert await asyncio.to_thread(flush_stored.wait, 10)
            assert ends == [None]  # the host has the run as it was before its end
            ctx._trace["r"].t1 = 5
            assert await asyncio.wait_for(ctx._save_trace(barrier=True), 5) is False  # the end: refused
            flush_answers.set()
            await asyncio.wait_for(asyncio.to_thread(other_loop.join, 10), 15)
        assert flushed == [[ctx.id]]  # the end is still owed, and reported
        assert [write.id for write in PlatformTracingProvider._unsaved.values()] == [ctx.id]

    async def test_once_the_newest_save_landed_a_flush_sends_nothing(self, monkeypatch):
        """A flush on another loop is held after it picked the run; this loop stores the end meanwhile. That flush
        and the ones after it have nothing left to send."""
        import threading

        server, ends, sent = FakePlatform(), [], []
        provider = self.redacting()
        ctx = self.context(provider)
        record = self.recording(server, ctx.id, ends)
        main, paused, go = threading.get_ident(), threading.Event(), threading.Event()
        real_send = PlatformTracingProvider.__dict__["_send"].__func__

        async def send(cls, *args, **kwargs):
            if threading.get_ident() != main and not go.is_set():
                paused.set()
                go.wait(10)
            return await real_send(cls, *args, **kwargs)

        async def request(method, path, headers=None, **kwargs):
            if method == "PATCH":
                sent.append(None)
                if len(sent) == 1:
                    raise PlatformError("upstream", status_code=503)  # the mid-run save: the run is owed
            return await record(method, path, headers=headers, **kwargs)

        monkeypatch.setattr(PlatformTracingProvider, "_send", classmethod(send))
        flushed = []

        def flush():
            flushed.append(asyncio.run(provider.flush(5)))

        with patch("timbal.platform.utils._request", request):
            assert await asyncio.wait_for(ctx._save_trace(barrier=True), 5) is False
            other_loop = threading.Thread(target=flush)
            other_loop.start()
            assert await asyncio.to_thread(paused.wait, 10)
            ctx._trace["r"].t1 = 5
            assert await asyncio.wait_for(ctx._save_trace(barrier=True), 5) is True
            go.set()
            await asyncio.wait_for(asyncio.to_thread(other_loop.join, 10), 15)
            after_end = len(sent)
            for _ in range(2):
                again = threading.Thread(target=flush)
                again.start()
                await asyncio.wait_for(asyncio.to_thread(again.join, 10), 15)
        assert flushed == [[], [], []]
        assert len(sent) == after_end  # nothing owed: no PATCH from any of the flushes
        assert self.never_back(ends)

    async def test_a_run_whose_every_save_failed_stays_owed_while_another_loop_sends_it(self):
        """This loop's stalled PATCH is answered "too large" while a flush on another loop is sending the run; both
        then fail. The run stays owed and the flush reports it."""
        import threading

        main = threading.get_ident()
        first_sent, answer_it = asyncio.Event(), asyncio.Event()
        flush_sent, flush_fails = threading.Event(), threading.Event()
        flushed = []
        provider = self.redacting()

        async def request(method, **_):
            assert method == "PATCH"
            if threading.get_ident() != main:
                flush_sent.set()
                flush_fails.wait(10)
                raise PlatformError("upstream", status_code=503)
            if not first_sent.is_set():
                first_sent.set()
                await answer_it.wait()
                threading.Thread(target=lambda: flushed.append(asyncio.run(provider.flush(5)))).start()
                assert flush_sent.wait(10)  # the other loop's PATCH is out
                raise PlatformError("payload too large", status_code=413)
            raise PlatformError("upstream", status_code=503)

        ctx = self.context(provider)
        _, _, key = PlatformTracingProvider._target(ctx)
        with patch("timbal.platform.utils._request", request):
            mid_run = asyncio.create_task(ctx._save_trace(barrier=True))
            await asyncio.wait_for(first_sent.wait(), 5)
            answer_it.set()
            assert await asyncio.wait_for(mid_run, 10) is False
            assert key in PlatformTracingProvider._unsaved  # still owed while the other loop's save is out
            flush_fails.set()

            async def flush_done():
                while not flushed:
                    await asyncio.sleep(0)

            await asyncio.wait_for(flush_done(), 10)
        assert flushed == [[ctx.id]]
        assert key in PlatformTracingProvider._unsaved

    async def test_a_fresh_context_with_the_same_run_id_replaces_what_is_stored(self, monkeypatch):
        """A finished run id used again (the HTTP job store allows it): the fresh context's save, sent after the old
        one's, stores the fresh context's trace."""
        monkeypatch.setattr(PlatformTracingProvider, "_stall_s", 3600.0)
        server = FakePlatform()
        provider = self.redacting()
        old = self.context(provider, t1=1, output="previous")
        fresh = RunContext(id=old.id, tracing_provider=provider)
        fresh.platform_config = CONFIG
        fresh._trace["r"] = {
            "call_id": "r",
            "path": "a",
            "t0": 0,
            "t1": 1,
            "parent_call_id": None,
            "output": "replacement",
        }
        on_wire, release, patches = asyncio.Event(), asyncio.Event(), []

        async def request(method, path, headers=None, **kwargs):
            if method == "PATCH":
                patches.append(None)
                if len(patches) == 2:  # the old context's second save: held on the wire
                    on_wire.set()
                    await release.wait()
            return await server.request(method, path, headers=headers, **kwargs)

        with patch("timbal.platform.utils._request", request):
            await asyncio.wait_for(provider.put(old), 5)
            second = asyncio.create_task(provider.put(old))
            await asyncio.wait_for(on_wire.wait(), 5)
            replacing = asyncio.create_task(provider.put(fresh))  # waits behind the PATCH on the wire
            await asyncio.wait_for(until_joined(PlatformTracingProvider._target(fresh)[2]), 5)
            release.set()
            await asyncio.wait_for(asyncio.gather(second, replacing), 5)
        assert whole_root(server, old.id)["output"] == "replacement"

    @pytest.mark.parametrize("fresh_lands", [True, False], ids=["the fresh run lands", "the fresh run fails"])
    async def test_an_older_context_of_a_reused_run_id_is_never_stored_over_the_fresh_one(
        self, monkeypatch, fresh_lands
    ):
        """A flush on another loop is about to send the old context (its end failed, so it was owed) when the id is
        used again by a fresh context. The old context is not stored over the fresh one, and does not settle the
        fresh one when it is the fresh one that is owed."""
        import threading

        monkeypatch.setattr(PlatformTracingProvider, "_retry_delays_s", (300.0,))
        server = FakePlatform(deltas=False)  # whole traces only: each PATCH says which context it sends
        provider = self.redacting()
        old = self.context(provider, t1=1, output="previous")
        fresh = RunContext(id=old.id, tracing_provider=provider)
        fresh.platform_config = CONFIG
        fresh._trace["r"] = {
            "call_id": "r",
            "path": "a",
            "t0": 0,
            "t1": 1,
            "parent_call_id": None,
            "output": "replacement",
        }
        _, _, key = PlatformTracingProvider._target(old)
        main, paused, go = threading.get_ident(), threading.Event(), threading.Event()
        outputs, refused = [], []
        real_send = PlatformTracingProvider.__dict__["_send"].__func__

        async def send(cls, *args, **kwargs):
            if threading.get_ident() != main and not go.is_set():
                paused.set()  # the flush has picked the run, with the old context owed
                go.wait(10)
            return await real_send(cls, *args, **kwargs)

        async def request(method, path, headers=None, content=None, **kwargs):
            if method == "PATCH":
                raw = gzip.decompress(content) if (headers or {}).get("Content-Encoding") == "gzip" else content
                sending = json.loads(raw)["trace"][0]["output"]
                if not refused or (sending == "replacement" and not fresh_lands):
                    refused.append(sending)
                    raise PlatformError("upstream", status_code=503)
                response = await server.request(method, path, headers=headers, content=content, **kwargs)
                outputs.append(whole_root(server, old.id)["output"])
                return response
            return await server.request(method, path, headers=headers, content=content, **kwargs)

        monkeypatch.setattr(PlatformTracingProvider, "_send", classmethod(send))
        flushed = []
        with patch("timbal.platform.utils._request", request):
            assert await asyncio.wait_for(old._save_trace(barrier=True), 5) is False  # the old run's end: owed
            other_loop = threading.Thread(target=lambda: flushed.append(asyncio.run(provider.flush(5))))
            other_loop.start()
            assert await asyncio.to_thread(paused.wait, 10)
            assert await asyncio.wait_for(fresh._save_trace(barrier=True), 5) is fresh_lands  # the id, used again
            go.set()
            await asyncio.wait_for(asyncio.to_thread(other_loop.join, 10), 15)
        assert "previous" not in outputs  # the old context was never stored over the fresh one
        if fresh_lands:
            assert outputs == ["replacement"] and flushed == [[]]
        else:
            assert outputs == [] and flushed == [[old.id]]  # the fresh run is still owed, and reported
            assert PlatformTracingProvider._unsaved[key].context is fresh

    @pytest.mark.parametrize("fresh_lands", [True, False], ids=["the fresh run landed", "the fresh run failed"])
    async def test_a_late_save_of_an_older_context_of_a_reused_run_id_is_dropped(self, monkeypatch, fresh_lands):
        """Work that outlived a finished run saves it again after its id was used by a fresh context (or its
        keep-alive, or a buffered upload, does): that save is not stored, and does not take the fresh run's place
        as what is owed."""
        monkeypatch.setattr(PlatformTracingProvider, "_retry_delays_s", (300.0,))
        server = FakePlatform(deltas=False)
        provider = self.redacting()
        old = self.context(provider, t1=1, output="previous")
        fresh = RunContext(id=old.id, tracing_provider=provider)
        fresh.platform_config = CONFIG
        fresh._trace["r"] = {"call_id": "r", "path": "a", "t0": 0, "t1": 1, "parent_call_id": None, "output": "new"}
        _, _, key = PlatformTracingProvider._target(old)
        with patch("timbal.platform.utils._request", server.request):
            await asyncio.wait_for(provider.put(old), 5)
            if fresh_lands:
                await asyncio.wait_for(provider.put(fresh), 5)
            else:
                server.fail_next = 1
                with pytest.raises(PlatformError):
                    await asyncio.wait_for(provider.put(fresh), 5)
            old._trace["r"].output = "previous, saved late"
            await asyncio.wait_for(provider.put(old), 5)
        if fresh_lands:
            assert whole_root(server, old.id)["output"] == "new"
            assert not PlatformTracingProvider._unsaved
        else:
            assert whole_root(server, old.id)["output"] == "previous"  # nothing newer landed
            assert PlatformTracingProvider._unsaved[key].context is fresh  # still owed: the fresh run

    async def test_a_buffered_run_id_used_again_uploads_the_fresh_run(self, monkeypatch):
        """Buffered uploads keep retrying a finished run whose end failed; its id is then used again by a fresh
        context. What is uploaded from then on is the fresh run, and a late save of the old one is not."""
        from timbal.state.tracing import uploads

        monkeypatch.setattr(PlatformTracingProvider, "_retry_delays_s", (300.0,))
        server, failed = FakePlatform(deltas=False), []
        down = True

        async def request(method, path, **kwargs):
            if method == "PATCH" and down:
                failed.append(None)
                raise PlatformError("upstream", status_code=503)
            return await server.request(method, path, **kwargs)

        async def until(predicate):
            while not predicate():
                await asyncio.sleep(0.01)

        provider = PlatformTracingProvider.configured(
            platform_config=CONFIG, durability="buffered", flush_interval_s=0.05
        )
        old = self.context(provider, output="old")
        fresh = RunContext(id=old.id, tracing_provider=provider)
        fresh.platform_config = CONFIG
        fresh._trace["r"] = {"call_id": "r", "path": "a", "t0": 0, "parent_call_id": None, "output": "fresh"}
        try:
            with patch("timbal.platform.utils._request", request):
                assert uploads.schedule(old)
                await asyncio.wait_for(until(lambda: failed), 5)  # its mid-run upload failed: retried
                old._trace["r"].t1 = 1
                with pytest.raises(PlatformError):
                    await asyncio.wait_for(uploads.write_through(old, provider.put), 5)  # its end, refused too
                down = False
                assert uploads.schedule(fresh)  # the id, used again
                await asyncio.wait_for(until(lambda: old.id in server.runs and not uploads.pending_runs()), 5)
                old._trace["r"].output = "old, saved late"
                assert uploads.schedule(old)
                assert await asyncio.wait_for(uploads.write_through(old, provider.put), 5) is False
        finally:
            self.reset_uploads()
        root = whole_root(server, old.id)
        assert root["output"] == "fresh" and root.get("t1") is None

    @staticmethod
    def reset_uploads():
        from timbal.state.tracing import uploads

        for run in list(uploads._RUNS.values()):
            if run.task is not None:
                run.task.cancel()
        uploads._reset_for_tests()

    async def test_a_fresh_run_taking_a_buffered_entry_over_still_falls_back_to_awaited_saves(self):
        """The old run left an upload pending; the fresh context that takes the entry over and falls behind too
        goes back to awaited saves, as any run does."""
        from timbal.state.tracing import uploads

        async def request(**_):
            raise PlatformError("upstream", status_code=503)  # the platform is down

        provider = PlatformTracingProvider.configured(
            platform_config=CONFIG, durability="buffered", flush_interval_s=0.05, max_lag_s=0
        )
        old = self.context(provider, t1=1, output="old")
        fresh = RunContext(id=old.id, tracing_provider=provider)
        fresh.platform_config = CONFIG
        fresh._trace["r"] = {"call_id": "r", "path": "a", "t0": 0, "parent_call_id": None, "output": "fresh"}
        try:
            with patch("timbal.platform.utils._request", request):
                assert uploads.schedule(old)
                assert not uploads.schedule(old)  # behind: the caller saves now
                assert uploads.schedule(fresh)  # the id, used again
                assert not uploads.schedule(fresh)  # behind as well
        finally:
            self.reset_uploads()

    async def test_an_old_runs_barrier_waiting_when_the_id_is_used_again_does_not_store_it(self):
        """Work that outlived a finished run waits to store it behind the run's end; meanwhile the id is used again.
        That save stores nothing, and does not count the fresh run's save as done: the fresh run is uploaded."""
        from timbal.state.tracing import uploads

        server, sent = FakePlatform(advertise=False), []
        on_wire, release = asyncio.Event(), asyncio.Event()

        async def request(method, path, headers=None, content=None, **kwargs):
            if method == "PATCH":
                sent.append(json.loads(content)["trace"][0]["output"])
                if len(sent) == 1:
                    on_wire.set()
                    await release.wait()
            return await server.request(method, path, headers=headers, content=content, **kwargs)

        async def until(predicate):
            while not predicate():
                await asyncio.sleep(0.01)

        provider = PlatformTracingProvider.configured(
            platform_config=CONFIG, durability="buffered", flush_interval_s=0.05
        )
        old = self.context(provider, t1=1, output="old")
        fresh = RunContext(id=old.id, tracing_provider=provider)
        fresh.platform_config = CONFIG
        fresh._trace["r"] = {"call_id": "r", "path": "a", "t0": 0, "parent_call_id": None, "output": "fresh"}
        try:
            with patch("timbal.platform.utils._request", request):
                end = asyncio.create_task(uploads.write_through(old, provider.put))
                await asyncio.wait_for(on_wire.wait(), 5)
                late = asyncio.create_task(uploads.write_through(old, provider.put))
                run = uploads._RUNS[uploads._key(old)]
                await asyncio.wait_for(until(lambda: run.writers == 2), 5)
                assert uploads.schedule(fresh)  # the id, used again
                release.set()
                assert await asyncio.wait_for(end, 5) is True
                assert await asyncio.wait_for(late, 5) is False
                await asyncio.wait_for(until(lambda: "fresh" in sent and not uploads.pending_runs()), 5)
        finally:
            self.reset_uploads()
        assert sent == ["old", "fresh"]
        assert whole_root(server, old.id)["output"] == "fresh"

    async def test_a_shutdown_flush_sends_an_end_that_failed_past_a_stalled_patch(self, monkeypatch):
        """The run's end goes past a stalled mid-run PATCH and fails; a shutdown flush then runs while the stalled
        one is still out. The flush sends the end, which lands."""
        monkeypatch.setattr(PlatformTracingProvider, "_stall_s", 0.0)
        monkeypatch.setattr(PlatformTracingProvider, "_retry_delays_s", (300.0,))
        server, ends, patches = FakePlatform(deltas=False), [], []
        first_sent, release = asyncio.Event(), asyncio.Event()
        provider = self.redacting()
        ctx = self.context(provider)
        record = self.recording(server, ctx.id, ends)

        async def request(method, path, headers=None, **kwargs):
            if method != "PATCH":
                return await server.request(method, path, headers=headers, **kwargs)
            patches.append(None)
            if len(patches) == 1:
                first_sent.set()
                await release.wait()  # the mid-run PATCH: stalled, then it lands
            elif len(patches) == 2:
                raise PlatformError("upstream", status_code=503)  # the end, past it
            return await record(method, path, headers=headers, **kwargs)

        async def flush_sent():
            while len(patches) < 3:
                await asyncio.sleep(0)

        with patch("timbal.platform.utils._request", request):
            mid_run = asyncio.create_task(ctx._save_trace(barrier=True))
            await asyncio.wait_for(first_sent.wait(), 5)
            ctx._trace["r"].t1 = 5
            assert await asyncio.wait_for(ctx._save_trace(barrier=True), 5) is False
            flushing = asyncio.create_task(provider.flush(5))
            await asyncio.wait_for(flush_sent(), 5)
            release.set()
            assert await asyncio.wait_for(flushing, 10) == []
            await asyncio.wait_for(mid_run, 5)
        assert self.never_back(ends)
        assert not PlatformTracingProvider._unsaved

    async def test_a_new_save_is_sent_even_when_every_other_owed_run_is_being_sent(self, monkeypatch):
        """``_unsaved`` is full of runs whose PATCHes are out; a new run's end is asked: it is sent and lands, and
        nothing is left owed once those land too."""
        monkeypatch.setattr(platform_module, "_UNSAVED_MAX", 3)
        server, held, release = FakePlatform(deltas=False), [], asyncio.Event()

        async def request(method, path, **kwargs):
            if method == "PATCH" and len(held) < 3:
                held.append(path)
                await release.wait()
            return await server.request(method, path, **kwargs)

        async def all_out():
            while len(held) < 3:
                await asyncio.sleep(0)

        provider = PlatformTracingProvider.configured(platform_config=CONFIG)
        out = [self.context(provider, t1=1, output=f"run {i}") for i in range(3)]
        newest = self.context(provider, t1=1, output="newest")
        with patch("timbal.platform.utils._request", request):
            sending = [asyncio.create_task(provider.put(ctx)) for ctx in out]
            await asyncio.wait_for(all_out(), 5)
            assert await asyncio.wait_for(newest._save_trace(barrier=True), 5) is True
            assert (whole_root(server, newest.id) or {}).get("output") == "newest"
            release.set()
            await asyncio.wait_for(asyncio.gather(*sending), 5)
            assert await PlatformTracingProvider.flush(timeout=1) == []
        assert all(whole_root(server, ctx.id)["output"] == f"run {i}" for i, ctx in enumerate(out))

    async def test_a_save_is_resent_to_the_destination_that_accepted_it(self):
        """A save fails at destination A; the provider's ``_run_path`` then routes the run to B (a registry that
        changed). The resend goes to A, and only A's answer settles it."""
        server, paths = FakePlatform(deltas=False), []
        server.fail_next = 1

        class Registry(PlatformTracingProvider):
            platform_config = CONFIG
            where = "a"

            @classmethod
            def _run_path(cls, platform_config, run_id):  # noqa: ARG003
                return RunPath(f"orgs/1/{cls.where}/runs/{run_id}", family=f"orgs/1/{cls.where}/runs")

        async def request(method, path, **kwargs):
            if method == "PATCH":
                paths.append(path)
            return await server.request(method, path, **kwargs)

        ctx = self.context(Registry, t1=1)
        with patch("timbal.platform.utils._request", request):
            with pytest.raises(PlatformError):
                await asyncio.wait_for(Registry.put(ctx), 5)
            Registry.where = "b"
            assert await PlatformTracingProvider.flush(timeout=1) == []
        assert paths == [f"orgs/1/a/runs/{ctx.id}"] * 2

    @pytest.mark.parametrize("old_saves", ["deltas", "whole only"])
    @pytest.mark.parametrize("old_context", ["kept", "collected"])
    async def test_a_run_id_taken_over_is_stored_whole_until_the_host_has_it(self, old_context, old_saves):
        """A finished run's id is used again by a fresh context, on a host that takes deltas: the fresh run's
        first PATCH replaces what the old run left instead of being merged into it, and once the fresh run ends
        the host holds it alone: its root and its child, none of the old run's spans. Also when the old context is
        gone, and when the old run never sent a delta (its only save went whole, before the host said it takes
        them)."""
        import gc

        server = FakePlatform()
        if old_saves == "deltas":
            PlatformTracingProvider._host_caps(CONFIG.host).delta = True
        provider = PlatformTracingProvider.configured(platform_config=CONFIG)

        def context(run_id, child, output):
            ctx = (
                RunContext(tracing_provider=provider)
                if run_id is None
                else RunContext(id=run_id, tracing_provider=provider)
            )
            ctx.platform_config = CONFIG
            ctx._trace["r"] = {"call_id": "r", "path": "a", "t0": 0, "parent_call_id": None, "output": output}
            ctx._trace[child] = {"call_id": child, "path": "a.t", "t0": 1, "t1": 2, "parent_call_id": "r"}
            return ctx

        old = context(None, "c1", "old")
        run_id = old.id
        with patch("timbal.platform.utils._request", server.request):
            if old_saves == "deltas":
                await asyncio.wait_for(provider.put(old), 5)
            old._trace["r"].t1 = 3
            await asyncio.wait_for(provider.put(old), 5)
            assert PlatformTracingProvider._host_caps(CONFIG.host).delta  # the fresh run's saves may go as deltas
            if old_context == "collected":
                del old
                gc.collect()
            fresh = context(run_id, "c2", "fresh")
            await asyncio.wait_for(provider.put(fresh), 5)
            fresh._trace["r"].t1 = 6
            await asyncio.wait_for(provider.put(fresh), 5)
            response = await server.request("GET", f"orgs/1/apps/2/runs/{run_id}")
        trace = response.json()["trace"]
        assert sorted(span["call_id"] for span in trace) == ["c2", "r"]
        assert next(span for span in trace if span["call_id"] == "r")["output"] == "fresh"

    async def test_a_save_not_sent_yet_is_never_pushed_out(self, monkeypatch):
        """A save is noted and not sent yet (another thread noted it and is about to send it) when ``_unsaved`` is
        full: the next save does not push it out, and it lands when it is sent."""
        monkeypatch.setattr(platform_module, "_UNSAVED_MAX", 1)
        server = FakePlatform(deltas=False)
        provider = PlatformTracingProvider.configured(platform_config=CONFIG)
        first, second = self.context(provider, t1=1, output="first"), self.context(provider, t1=1, output="second")
        target = PlatformTracingProvider._target(first)
        with patch("timbal.platform.utils._request", server.request):
            assert platform_module._owe(target, first, provider) is not None  # noted; its send has not begun
            await asyncio.wait_for(provider.put(second), 5)
            assert target[2] in PlatformTracingProvider._unsaved
            await asyncio.wait_for(provider._send(target[2]), 5)
            assert await PlatformTracingProvider.flush(timeout=1) == []
        assert whole_root(server, first.id)["output"] == "first"

    async def test_two_credentials_storing_one_run_are_one_destination(self):
        """Two providers store one run at the same place with different credentials: an older copy sent through
        one is not stored over a newer one sent through the other."""
        from timbal.state.tracing.providers import base

        other = PlatformConfig(
            host=CONFIG.host,
            auth=PlatformAuth(type=PlatformAuthType.BEARER, token="another_token"),
            subject=CONFIG.subject,
        )
        server = FakePlatform(deltas=False)
        policy = masking("secret-out", "[OUT]")
        one = PlatformTracingProvider.configured(platform_config=CONFIG, _trace_redactor=policy)
        two = PlatformTracingProvider.configured(trace_platform_config=other, _trace_redactor=policy)
        ctx = self.context(one)
        assert one._target(ctx)[2] == two._target(ctx)[2]  # one place, two tokens
        assert one._recent_key(*one._target(ctx)[:2]) != two._recent_key(*two._target(ctx)[:2])
        older = base._capture(ctx, policy)
        ctx._trace["r"].t1 = 5
        newer = base._capture(ctx, policy)
        with patch("timbal.platform.utils._request", server.request):
            await asyncio.wait_for(two._store(newer), 5)
            await asyncio.wait_for(one._store(older), 5)
            assert await PlatformTracingProvider.flush(timeout=1) == []
        assert whole_root(server, ctx.id)["t1"] == 5

    @pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
    async def test_a_forked_child_never_sends_the_parents_runs(self):
        """A save fails, so the run is owed; the process forks. The child has nothing to send (at its exit, or a
        flush): what it inherited is the run as it was at the fork, and the parent keeps sending it."""
        import warnings

        server = FakePlatform(deltas=False)
        server.fail_next = 1
        provider = PlatformTracingProvider.configured(platform_config=CONFIG)
        ctx = self.context(provider)
        with patch("timbal.platform.utils._request", server.request):
            assert await asyncio.wait_for(ctx._save_trace(barrier=True), 5) is False
            assert PlatformTracingProvider._unsaved
            read, write = os.pipe()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)  # forking a process that has threads
                pid = os.fork()
            if pid == 0:  # the child: report and leave, running nothing of the parent's
                report = "error"
                try:
                    owed = len(PlatformTracingProvider._unsaved) + len(uploads._RUNS)
                    platform_module._flush_at_exit()
                    report = f"{owed} {len(server.patches)}"
                finally:
                    os.write(write, report.encode())
                    os._exit(0)
            os.close(write)
            _, status = await asyncio.wait_for(asyncio.to_thread(os.waitpid, pid, 0), 10)
            report = os.read(read, 100).decode()
            os.close(read)
            assert report == "0 0"  # nothing owed in the child, nothing sent by it
            ctx._trace["r"].t1 = 5
            assert await asyncio.wait_for(ctx._save_trace(barrier=True), 5) is True
        assert whole_root(server, ctx.id)["t1"] == 5

    async def test_a_keep_alive_due_during_another_loops_flush_still_saves(self):
        """A run's keep-alive comes due while a flush runs on another event loop: it saves as usual, and its timer
        goes on."""
        server = FakePlatform(deltas=False)
        provider = PlatformTracingProvider.configured(platform_config=CONFIG, keepalive_s=3600)
        ctx = self.context(provider)
        _, _, key = PlatformTracingProvider._target(ctx)
        flushing = object()  # the other loop

        async def saved():
            while ctx.id not in server.runs:
                await asyncio.sleep(0)

        with patch("timbal.platform.utils._request", server.request):
            provider._arm_keepalive(key, ctx, asyncio.get_running_loop())
            PlatformTracingProvider._keepalives[key].cancel()  # as when its timer fires: no timer left
            PlatformTracingProvider._closing_loops.add(flushing)
            try:
                PlatformTracingProvider._keep_alive(key)
                await asyncio.wait_for(saved(), 5)
                assert PlatformTracingProvider._keepalives[key].handle is not None
            finally:
                PlatformTracingProvider._closing_loops.discard(flushing)
                entry = PlatformTracingProvider._keepalives.pop(key, None)
                if entry is not None:
                    entry.cancel()

    async def test_a_keep_alive_whose_save_is_dropped_keeps_its_timer(self):
        """A keep-alive holds a copy of the run older than what its destination is owed (another save of the run
        was asked since, one that did not set the timer again): the store drops that save, and the keep-alives go
        on."""
        from timbal.state.tracing.providers import base

        server = FakePlatform(deltas=False)
        policy = masking("secret-out", "[OUT]")
        provider = PlatformTracingProvider.configured(platform_config=CONFIG, keepalive_s=3600, _trace_redactor=policy)
        ctx = self.context(provider)
        target = PlatformTracingProvider._target(ctx)
        key = target[2]
        with patch("timbal.platform.utils._request", server.request):
            await asyncio.wait_for(provider._store(base._capture(ctx, policy)), 5)  # sets the timer, with this copy
            assert platform_module._owe(target, ctx, provider) is not None  # a newer save of the run is owed
            PlatformTracingProvider._keepalives[key].cancel()  # its timer fires
            before = asyncio.all_tasks()
            try:
                PlatformTracingProvider._keep_alive(key)
                started = asyncio.all_tasks() - before
                await asyncio.wait_for(asyncio.gather(*started, return_exceptions=True), 5)
                assert PlatformTracingProvider._keepalives[key].handle is not None
            finally:
                entry = PlatformTracingProvider._keepalives.pop(key, None)
                if entry is not None:
                    entry.cancel()

    async def test_a_save_whose_body_cannot_be_made_records_nothing(self):
        """A redactor raises while a PATCH body is being made: the save fails with that error, nothing waits for a
        PATCH that never went out (a read of the run would wait for it), and the request is not kept owed (it would
        fail the same way on every resend, and hold a place in ``_unsaved`` for good)."""
        server = FakePlatform(deltas=False)

        def broken(_value):
            raise RuntimeError("redactor bug")

        provider = PlatformTracingProvider.configured(platform_config=CONFIG, _trace_redactor=broken)
        ctx = self.context(provider)
        _, _, key = PlatformTracingProvider._target(ctx)
        with patch("timbal.platform.utils._request", server.request):
            with pytest.raises(RuntimeError, match="redactor bug"):
                await asyncio.wait_for(provider.put(ctx), 5)
        assert key not in PlatformTracingProvider._saving
        assert key not in PlatformTracingProvider._unsaved  # dropped: it would fail the same way on every resend
        assert not server.patches

    async def test_a_read_through_other_credentials_is_not_answered_from_an_older_copy(self):
        """Two providers store one run at one place with different credentials; the second stores a newer state.
        A read through the first is not answered with what this process kept of the first one's save."""
        other = PlatformConfig(
            host=CONFIG.host,
            auth=PlatformAuth(type=PlatformAuthType.BEARER, token="another_token"),
            subject=CONFIG.subject,
        )
        server = FakePlatform(deltas=False)
        one = PlatformTracingProvider.configured(platform_config=CONFIG)
        two = PlatformTracingProvider.configured(trace_platform_config=other)
        ctx = self.context(one, t1=1, output="v1")
        with patch("timbal.platform.utils._request", server.request):
            await asyncio.wait_for(one.put(ctx), 5)
            ctx._trace["r"].output = "v2"
            await asyncio.wait_for(two.put(ctx), 5)
            child = RunContext(parent_id=ctx.id, tracing_provider=one)
            child.platform_config = CONFIG
            trace = await asyncio.wait_for(one.get(child), 5)
        assert trace["r"].output == "v2"

    async def test_a_save_out_on_one_loop_is_never_pushed_out(self, monkeypatch):
        """A run's PATCH is out on this loop when a flush on another loop sends the run and fails, and its flight
        there ends: a new save then fills ``_unsaved``. The run is not pushed out: when this loop's PATCH is
        answered "too large", the run is sent again and lands."""
        import threading

        monkeypatch.setattr(platform_module, "_UNSAVED_MAX", 1)
        server = FakePlatform(deltas=False)
        main = threading.get_ident()
        first_sent, answer = asyncio.Event(), asyncio.Event()

        async def request(method, path, headers=None, **kwargs):
            if method == "PATCH" and threading.get_ident() != main:
                raise PlatformError("bad request", status_code=400)  # the flush's PATCH, on the other loop
            if method == "PATCH" and not first_sent.is_set():
                first_sent.set()
                await answer.wait()
                raise PlatformError("payload too large", status_code=413)
            return await server.request(method, path, headers=headers, **kwargs)

        provider = PlatformTracingProvider.configured(platform_config=CONFIG)
        first, second = self.context(provider, t1=1, output="first"), self.context(provider, t1=1, output="second")
        with patch("timbal.platform.utils._request", request):
            held = asyncio.create_task(provider.put(first))
            await asyncio.wait_for(first_sent.wait(), 5)
            flushed = await asyncio.wait_for(asyncio.to_thread(lambda: asyncio.run(provider.flush(5))), 15)
            assert flushed == [first.id]  # its PATCH there was refused
            await asyncio.wait_for(provider.put(second), 5)
            answer.set()
            await asyncio.wait_for(held, 5)
            assert await PlatformTracingProvider.flush(timeout=1) == []
        assert whole_root(server, first.id)["output"] == "first"
        assert whole_root(server, second.id)["output"] == "second"

    async def test_a_fresh_context_does_not_inherit_the_old_runs_size_refusal(self):
        """A finished run was refused as too large, so its saves went out cut down; its id is used again by a fresh
        context whose trace fits. The fresh run is stored as it is, not cut."""
        server = FakePlatform(deltas=False, limit=150_000)
        provider = PlatformTracingProvider.configured(platform_config=CONFIG)
        old = self.context(provider, t1=1, output="x" * 200_000)
        with patch("timbal.platform.utils._request", server.request):
            await asyncio.wait_for(provider.put(old), 5)
            assert "trace_truncated" in whole_root(server, old.id)["metadata"]
            fresh = RunContext(id=old.id, tracing_provider=provider)
            fresh.platform_config = CONFIG
            fresh._trace["r"] = {
                "call_id": "r",
                "path": "a",
                "t0": 0,
                "t1": 2,
                "parent_call_id": None,
                "output": "y" * 100_000,
            }
            await asyncio.wait_for(provider.put(fresh), 5)
        root = whole_root(server, old.id)
        assert root["output"] == "y" * 100_000 and "trace_truncated" not in (root.get("metadata") or {})

    async def test_acknowledgements_do_not_cross_redaction_policies(self):
        """Two providers with different policies save one run to one destination that takes deltas: a finished span
        the first one sent is sent again by the second, so the destination holds it as the second one's policy
        has it."""
        server = FakePlatform()
        PlatformTracingProvider._host_caps(CONFIG.host).delta = True
        plain = PlatformTracingProvider.configured(platform_config=CONFIG)
        masked = PlatformTracingProvider.configured(platform_config=CONFIG, _trace_redactor=masking("secret", "[S]"))
        ctx = RunContext(tracing_provider=plain)
        ctx.platform_config = CONFIG
        ctx._trace["r"] = {"call_id": "r", "path": "a", "t0": 0, "parent_call_id": None}
        ctx._trace["c"] = {
            "call_id": "c",
            "path": "a.t",
            "t0": 1,
            "t1": 2,
            "parent_call_id": "r",
            "output": "secret out",
        }
        with patch("timbal.platform.utils._request", server.request):
            await asyncio.wait_for(plain.put(ctx), 5)
            await asyncio.wait_for(masked.put(ctx), 5)
            response = await server.request("GET", f"orgs/1/apps/2/runs/{ctx.id}")
        child = next(span for span in response.json()["trace"] if span["call_id"] == "c")
        assert child["output"] == "[S] out"


class TestRedactionAcrossPaths:
    """Every PATCH applies the redaction policy of the provider that accepted the save, and those of the copies it
    was handed: whichever path sends it, and whatever was sent before it."""

    @staticmethod
    def context(provider=None, **root):
        ctx = RunContext(tracing_provider=provider)
        ctx.platform_config = CONFIG
        ctx._trace["r"] = {
            "call_id": "r",
            "path": "a",
            "t0": 0,
            "t1": 1,
            "parent_call_id": None,
            "input": {"prompt": "secret-in"},
            "output": "secret-out",
            **root,
        }
        return ctx

    async def test_a_store_of_its_own_gets_the_redacted_copy(self):
        """A subclass that stores runs its own way (it overrides ``_store``) persists what it is given."""
        stored = []

        class OwnStore(PlatformTracingProvider):
            platform_config = CONFIG
            _trace_redactor = staticmethod(masking("secret-out", "[OUT]"))

            @classmethod
            async def _store(cls, run_context):
                stored.append(run_context._trace.model_dump())

        await OwnStore.put(self.context())
        (trace,) = stored
        assert trace[0]["output"] == "[OUT]"

    async def test_a_store_of_its_own_that_scrubs_more_and_passes_it_on_keeps_both(self):
        """A subclass's ``_store`` scrubs the redacted copy it is given further, then hands it to the built-in store:
        what is stored has both, and the live run neither."""
        server = FakePlatform(deltas=False)

        class ScrubsMore(PlatformTracingProvider):
            platform_config = CONFIG
            _trace_redactor = staticmethod(masking("secret-out", "[OUT]"))

            @classmethod
            async def _store(cls, run_context):
                run_context._trace["r"].input = {"prompt": "[IN]"}  # its copy only
                await super()._store(run_context)

        ctx = self.context(ScrubsMore)
        with patch("timbal.platform.utils._request", server.request):
            await asyncio.wait_for(ScrubsMore.put(ctx), 5)
        root = whole_root(server, ctx.id)
        assert root["input"] == {"prompt": "[IN]"} and root["output"] == "[OUT]"
        assert ctx._trace["r"].input == {"prompt": "secret-in"} and ctx._trace["r"].output == "secret-out"

    async def test_a_copy_held_in_a_store_of_its_own_never_lands_over_a_newer_one(self):
        """A subclass's ``_store`` awaits before passing its redacted copy on; a mid-run save is held there while
        the run's end goes through and lands. The held copy, older, is not stored over it."""
        server = FakePlatform(deltas=False)
        held, release = asyncio.Event(), asyncio.Event()

        class Waits(PlatformTracingProvider):
            platform_config = CONFIG
            _trace_redactor = staticmethod(masking("secret-out", "[OUT]"))

            @classmethod
            async def _store(cls, run_context):
                if not held.is_set():
                    held.set()
                    await release.wait()
                await super()._store(run_context)

        ctx = self.context(Waits, t1=None)
        with patch("timbal.platform.utils._request", server.request):
            mid_run = asyncio.create_task(Waits.put(ctx))
            await asyncio.wait_for(held.wait(), 5)
            ctx._trace["r"].t1 = 5
            await asyncio.wait_for(Waits.put(ctx), 5)  # the end
            release.set()
            await asyncio.wait_for(mid_run, 5)
            assert await PlatformTracingProvider.flush(timeout=1) == []
        root = whole_root(server, ctx.id)
        assert root["t1"] == 5 and root["output"] == "[OUT]"

    async def test_a_copy_made_before_a_save_of_the_run_itself_went_out_is_not_stored_over_it(self):
        """One destination gets saves of the run itself (a built-in provider) and copies (a subclass whose own
        ``_store`` awaits before passing its copy on), with one policy. A copy made while a save of the run itself
        waited to go out is older than what that save sent: it is not stored after it."""
        server, applied = FakePlatform(deltas=False), []
        on_wire, release, copied, pass_on = asyncio.Event(), asyncio.Event(), asyncio.Event(), asyncio.Event()
        redactor = masking("secret-out", "[OUT]")
        live = PlatformTracingProvider.configured(platform_config=CONFIG, _trace_redactor=redactor)

        class OwnStore(PlatformTracingProvider):
            platform_config = CONFIG
            _trace_redactor = staticmethod(redactor)

            @classmethod
            async def _store(cls, run_context):
                copied.set()
                await pass_on.wait()
                await super()._store(run_context)

        async def request(method, path, **kwargs):
            if method == "PATCH" and not on_wire.is_set():
                on_wire.set()
                await release.wait()
            response = await server.request(method, path, **kwargs)
            if method == "PATCH":
                applied.append(whole_root(server, ctx.id)["output"])
            return response

        ctx = self.context(live, t1=None, output="v0")
        with patch("timbal.platform.utils._request", request):
            first = asyncio.create_task(live.put(ctx))
            await asyncio.wait_for(on_wire.wait(), 5)
            ctx._trace["r"].output = "v1"
            queued = asyncio.create_task(live.put(ctx))  # waits behind the PATCH on the wire
            await asyncio.wait_for(until_joined(PlatformTracingProvider._target(ctx)[2]), 5)
            copying = asyncio.create_task(OwnStore.put(ctx))  # a copy of v1, held in its _store
            await asyncio.wait_for(copied.wait(), 5)
            ctx._trace["r"].output = "v2"
            release.set()
            await asyncio.wait_for(asyncio.gather(first, queued), 5)  # the queued round sends the run as it is: v2
            pass_on.set()
            await asyncio.wait_for(copying, 5)
            assert await PlatformTracingProvider.flush(timeout=1) == []
        assert applied[-1] == "v2" and "v1" not in applied[applied.index("v2") :]

    async def test_a_field_an_exporter_took_out_of_its_copy_stays_out(self):
        """An exporter takes a field out of the redacted copy it is given and forwards it to a platform provider
        with a policy of its own: what is stored has the field out and the platform's policy applied."""
        server = FakePlatform(deltas=False)
        platform = PlatformTracingProvider.configured(
            platform_config=CONFIG, _trace_redactor=masking("secret-out", "[OUT]")
        )

        class Forward:
            async def export(self, run_context):
                run_context._trace["r"].input = None  # its copy only
                await platform.put(run_context)

        local = InMemoryTracingProvider.configured(_trace_redactor=masking("secret-in", "[IN]"), _exporters=[Forward()])
        ctx = self.context(local)
        with patch("timbal.platform.utils._request", server.request):
            await asyncio.wait_for(local.put(ctx), 5)
        root = whole_root(server, ctx.id)
        assert root.get("input") is None and root["output"] == "[OUT]"
        assert ctx._trace["r"].input == {"prompt": "secret-in"}

    async def test_a_copy_whose_capture_began_first_is_never_stored_over_a_later_one(self):
        """A copy's capture has read the running root when its redactor is held up; meanwhile the run ends and a
        copy of its end is captured and stored. The first copy, finished last, holds the older state: it is not
        stored over the end."""
        from timbal.state.tracing.providers import base

        server = FakePlatform(deltas=False)
        policy = masking("secret-out", "[OUT]")
        provider = PlatformTracingProvider.configured(platform_config=CONFIG, _trace_redactor=policy)
        ctx = self.context(provider, t1=None)
        end = []

        def slow(value):  # the first copy's redactor: the run ends while it works
            if not end:
                ctx._trace["r"].t1 = 5
                end.append(base._capture(ctx, policy))
            return policy(value)

        first = base._capture(ctx, slow)
        assert first._trace["r"].t1 is None and end[0]._trace["r"].t1 == 5
        with patch("timbal.platform.utils._request", server.request):
            await asyncio.wait_for(provider._store(end[0]), 5)
            await asyncio.wait_for(provider._store(first), 5)
            assert await PlatformTracingProvider.flush(timeout=1) == []
        assert whole_root(server, ctx.id)["t1"] == 5

    def test_captures_never_overlap_across_threads(self):
        """While a copy of a run is being captured on one thread, no other capture can begin: copies are numbered
        in the order their states were read."""
        import threading

        from timbal.state.tracing.providers import base

        ctx = self.context(None)
        reading, done = threading.Event(), threading.Event()

        def slow(value):
            reading.set()
            assert done.wait(10)
            return value

        capturing = threading.Thread(target=lambda: base._capture(ctx, slow))
        capturing.start()
        try:
            assert reading.wait(10)
            assert not base._CAPTURE_LOCK.acquire(blocking=False)
        finally:
            done.set()
            capturing.join(10)
        assert base._CAPTURE_LOCK.acquire(blocking=False)
        base._CAPTURE_LOCK.release()

    async def test_a_save_queued_behind_another_providers_is_sent_with_its_own_policy(self, monkeypatch):
        """A run id used again by a context saved with a redacting provider, queued behind a PATCH of the old one
        from a provider that does not redact: the queued round sends the fresh save with its own provider's
        policy."""
        monkeypatch.setattr(PlatformTracingProvider, "_stall_s", 3600.0)
        server = FakePlatform(deltas=False)
        plain = PlatformTracingProvider.configured(platform_config=CONFIG)
        masked = PlatformTracingProvider.configured(
            platform_config=CONFIG, _trace_redactor=masking("secret-out", "[OUT]")
        )
        old = self.context(plain)
        fresh = RunContext(id=old.id, tracing_provider=masked)
        fresh.platform_config = CONFIG
        fresh._trace["r"] = dict(old._trace["r"].model_dump(), output="secret-out again")
        on_wire, release = asyncio.Event(), asyncio.Event()

        async def request(method, path, **kwargs):
            if method == "PATCH" and not on_wire.is_set():
                on_wire.set()
                await release.wait()
            return await server.request(method, path, **kwargs)

        with patch("timbal.platform.utils._request", request):
            first = asyncio.create_task(old._save_trace(barrier=True))
            await asyncio.wait_for(on_wire.wait(), 5)
            replacing = asyncio.create_task(fresh._save_trace(barrier=True))
            await asyncio.wait_for(until_joined(PlatformTracingProvider._target(fresh)[2]), 5)
            release.set()
            assert await asyncio.wait_for(asyncio.gather(first, replacing), 5) == [True, True]
        assert whole_root(server, old.id)["output"] == "[OUT] again"

    async def test_a_shutdown_flush_sends_with_the_provider_that_accepted_the_save(self):
        """A configured provider called directly, for a run context that names no provider of its own: its save
        fails; the process-wide flush resends it with that provider's policy, not the base class's."""
        server = FakePlatform(deltas=False)
        server.fail_next = 1
        masked = PlatformTracingProvider.configured(
            platform_config=CONFIG, _trace_redactor=masking("secret-out", "[OUT]")
        )
        ctx = self.context(None)
        with patch("timbal.platform.utils._request", server.request):
            with pytest.raises(PlatformError):
                await masked.put(ctx)
            assert await PlatformTracingProvider.flush(timeout=1) == []
        assert whole_root(server, ctx.id)["output"] == "[OUT]"

    async def test_a_copy_forwarded_from_another_provider_gets_both_policies(self):
        """An exporter of a provider that masks the input forwards its redacted copy to a platform provider that
        masks the output: what the platform stores has both masked."""
        server = FakePlatform(deltas=False)
        platform = PlatformTracingProvider.configured(
            platform_config=CONFIG, _trace_redactor=masking("secret-out", "[OUT]")
        )

        class Forward:
            async def export(self, run_context):
                await platform.put(run_context)

        local = InMemoryTracingProvider.configured(_trace_redactor=masking("secret-in", "[IN]"), _exporters=[Forward()])
        ctx = self.context(local)
        with patch("timbal.platform.utils._request", server.request):
            await local.put(ctx)
        root = whole_root(server, ctx.id)
        assert root["input"] == {"prompt": "[IN]"} and root["output"] == "[OUT]"


class TestDestinations:
    async def test_each_destination_settles_on_its_own_acknowledgement(self):
        """One run saved to two destinations (two providers, two run paths). The first destination's answer comes
        after the second has stored the run: both are settled, and a flush has nothing left to send."""

        class OtherPlace(PlatformTracingProvider):
            platform_config = CONFIG

            @classmethod
            def _run_path(cls, platform_config, run_id):  # noqa: ARG003
                return RunPath(f"orgs/1/elsewhere/runs/{run_id}", family="orgs/1/elsewhere/runs")

        here, there = FakePlatform(deltas=False), FakePlatform(deltas=False)
        first_stored, answer_first, sent = asyncio.Event(), asyncio.Event(), []

        async def request(method, path, **kwargs):
            if method != "PATCH":
                return await (there if "elsewhere" in path else here).request(method, path, **kwargs)
            sent.append(path)
            if "elsewhere" in path:
                return await there.request(method, path, **kwargs)
            response = await here.request(method, path, **kwargs)
            if not first_stored.is_set():
                first_stored.set()
                await answer_first.wait()  # stored; its answer comes after the other destination's save
            return response

        default = PlatformTracingProvider.configured(platform_config=CONFIG)
        ctx = TestRedactionAcrossPaths.context(default)
        with patch("timbal.platform.utils._request", request):
            first = asyncio.create_task(default.put(ctx))
            await asyncio.wait_for(first_stored.wait(), 5)
            await asyncio.wait_for(OtherPlace.put(ctx), 5)
            answer_first.set()
            await asyncio.wait_for(first, 5)
            before = len(sent)
            assert await PlatformTracingProvider.flush(timeout=1) == []
            assert await PlatformTracingProvider.flush(timeout=1) == []
        assert len(sent) == before  # nothing owed at either destination
        assert ctx.id in here.runs and ctx.id in there.runs


class TestHousekeeping:
    def test_a_save_collected_while_the_snapshot_lock_is_held_does_not_wait_for_itself(self, monkeypatch):
        """A save left suspended when its event loop was closed is finished by the garbage collector, on whatever
        thread a collection happens to run: also on one that holds the lock the provider's bookkeeping uses."""
        import gc
        import sys
        import threading

        from timbal.state.tracing.providers import base

        lock_kind = type(base._CAPTURE_LOCK)
        monkeypatch.setattr(
            base, "_CAPTURE_LOCK", threading.RLock() if lock_kind is type(threading.RLock()) else threading.Lock()
        )
        monkeypatch.setattr(sys, "unraisablehook", lambda _info: None)  # "Task was destroyed but it is pending"
        never = []

        async def request(method, **_):
            assert method == "PATCH"
            if not never:
                never.append(None)
                await asyncio.get_running_loop().create_future()  # left on the wire: its loop closes
            return SimpleNamespace(raise_for_status=lambda: None, headers={}, json=lambda: {})

        provider = PlatformTracingProvider.configured(platform_config=CONFIG)
        left, other = (TestOrderAcrossPaths.context(provider) for _ in range(2))
        _, _, key = PlatformTracingProvider._target(left)
        allocations = []

        class Collecting(platform_module._Write):
            """A collection that lands on this allocation, made with the lock held."""

            __slots__ = ()

            def __init__(self, *args):
                if not allocations:
                    allocations.append(gc.collect())
                super().__init__(*args)

        gc.collect()
        gc.disable()
        try:
            with patch("timbal.platform.utils._request", request):

                def leave_a_save():
                    loop = asyncio.new_event_loop()
                    loop.create_task(provider.put(left))

                    async def out():
                        while not never:
                            await asyncio.sleep(0)

                    loop.run_until_complete(asyncio.wait_for(out(), 5))
                    loop.close()

                first = threading.Thread(target=leave_a_save)
                first.start()
                first.join(10)
                flushed = []  # the run sent again from another loop: its first loop's save is now only garbage
                flush = threading.Thread(target=lambda: flushed.append(asyncio.run(provider.flush(5))))
                flush.start()
                flush.join(10)
                assert flushed == [[]]
                monkeypatch.setattr(platform_module, "_Write", Collecting)
                saving = threading.Thread(target=lambda: asyncio.run(provider.put(other)), daemon=True)
                saving.start()
                saving.join(10)
        finally:
            gc.enable()
        assert not saving.is_alive()  # the save did not wait for itself
        assert allocations and allocations[0] > 0  # a collection ran inside the lock, and found the save left behind

    async def test_a_stalled_patch_keeps_no_finished_rounds(self, monkeypatch):
        """While one PATCH is out, the run's end goes past it again and again and fails each time: nothing of
        those failed rounds is kept (each failure holds the body it sent)."""
        monkeypatch.setattr(PlatformTracingProvider, "_stall_s", 0.0)
        first_sent, release, patches = asyncio.Event(), asyncio.Event(), []

        async def request(method, **_):
            assert method == "PATCH"
            patches.append(None)
            if len(patches) == 1:
                first_sent.set()
                await release.wait()
                return SimpleNamespace(raise_for_status=lambda: None, headers={}, json=lambda: {})
            raise PlatformError("forbidden", status_code=403)

        provider = PlatformTracingProvider.configured(platform_config=CONFIG)
        ctx = TestOrderAcrossPaths.context(provider)
        _, _, key = PlatformTracingProvider._target(ctx)
        with patch("timbal.platform.utils._request", request):
            first = asyncio.create_task(provider.put(ctx))
            await asyncio.wait_for(first_sent.wait(), 5)
            ctx._trace["r"].t1 = 5
            for _ in range(32):
                with pytest.raises(PlatformError):
                    await asyncio.wait_for(provider.put(ctx), 5)
            flight = PlatformTracingProvider._flights[key]
            assert len(patches) == 33
            assert len(flight.sent) <= 1  # the stalled round alone
            release.set()
            await asyncio.wait_for(first, 5)


class TestShutdownFlush:
    async def test_saves_owed_land_before_shutdown_ends(self, monkeypatch):
        monkeypatch.setattr(PlatformTracingProvider, "_stall_s", 60.0)
        ctx, provider, request, sent, first_sent, release = TestSingleFlight.stalled_run()
        with patch("timbal.platform.utils._request", request):
            caller = asyncio.create_task(provider._store(ctx))
            await first_sent.wait()
            caller.cancel()  # the turn went away; its save did not
            ctx._trace["r"].t1 = 1
            end = asyncio.create_task(provider._store(ctx))
            await asyncio.sleep(0)  # joined the next round
            end.cancel()
            from timbal.state.tracing.uploads import flush_all

            flushing = asyncio.create_task(flush_all(timeout=2))
            await asyncio.sleep(0.05)
            assert sent[-1]["trace"][0]["t1"] == 1  # the end did not wait out the 60 s stall
            release.set()
            await flushing
        assert not PlatformTracingProvider._unsaved

    async def test_saves_that_cannot_land_are_reported_within_the_bound(self):
        ctx, provider, request, sent, first_sent, release = TestSingleFlight.stalled_run()
        with patch("timbal.platform.utils._request", request), capture_logs() as logs:
            caller = asyncio.create_task(provider._store(ctx))
            await first_sent.wait()
            caller.cancel()
            left = await asyncio.wait_for(PlatformTracingProvider.flush(timeout=0.1), 1)
            release.set()
        assert left == [ctx.id]
        assert any(log["event"] == "platform_trace_flush_incomplete" and log["run_ids"] == [ctx.id] for log in logs)

    async def test_saves_that_died_with_an_earlier_loop_are_sent_again(self):
        server = FakePlatform(advertise=False)
        _, provider = tool_loop(1)
        ctx = RunContext(tracing_provider=provider)
        ctx.platform_config = CONFIG
        ctx._trace["r"] = {"call_id": "r", "path": "a", "t0": 0, "t1": 1, "parent_call_id": None}
        _, _, key = PlatformTracingProvider._target(ctx)
        platform_module._owe(
            PlatformTracingProvider._target(ctx), ctx, provider
        )  # asked for; its loop was torn down before it landed
        with patch("timbal.platform.utils._request", server.request):
            assert await PlatformTracingProvider.flush(timeout=1) == []
        assert server.runs[ctx.id]["trace"][0]["t1"] == 1

    async def test_a_flush_and_a_read_from_outside_the_run_use_its_own_credentials(self):
        # Requests take host and credentials from the current run: a shutdown flush (or a read) runs outside it.
        server = FakePlatform(advertise=False)
        seen = []

        async def request(method, path, **kwargs):
            current = get_run_context()
            seen.append((method, current.platform_config.host if current and current.platform_config else None))
            return await server.request(method, path, **kwargs)

        _, provider = tool_loop(1)
        ctx = RunContext(tracing_provider=provider)
        ctx.platform_config = CONFIG
        ctx._trace["r"] = {"call_id": "r", "path": "a", "t0": 0, "t1": 1, "parent_call_id": None}
        _, _, key = PlatformTracingProvider._target(ctx)
        platform_module._owe(PlatformTracingProvider._target(ctx), ctx, provider)
        set_run_context(None)
        with patch("timbal.platform.utils._request", request):
            assert await PlatformTracingProvider.flush(timeout=1) == []
            PlatformTracingProvider._recent.clear()
            reader = RunContext(parent_id=ctx.id, tracing_provider=provider)
            reader.platform_config = CONFIG
            assert await provider.get(reader) is not None
        assert seen == [("PATCH", CONFIG.host), ("GET", CONFIG.host)]
        assert get_run_context() is None  # restored after each request

    def test_a_script_whose_loop_ended_mid_save_still_lands_at_exit(self, tmp_path):
        import subprocess
        import sys

        log = tmp_path / "patches.jsonl"
        script = tmp_path / "script.py"
        script.write_text(
            f"""
import asyncio, json
from types import SimpleNamespace
from unittest.mock import patch
from timbal.state.config import PlatformAuth, PlatformAuthType, PlatformConfig, PlatformSubject
from timbal.state.context import RunContext
from timbal.state.tracing.providers.platform import PlatformTracingProvider

CONFIG = PlatformConfig(host="api.exit.test", auth=PlatformAuth(type=PlatformAuthType.BEARER, token="t"),
                        subject=PlatformSubject(org_id="1", app_id="2"))
calls = []

async def request(method, path, content=None, **_):
    calls.append(1)
    if len(calls) == 1:
        await asyncio.Event().wait()  # the first PATCH hangs; asyncio.run cancels it on the way out
    with open({str(log)!r}, "a") as f:
        f.write(content.decode() + "\\n")
    return SimpleNamespace(raise_for_status=lambda: None, headers={{}}, json=lambda: {{}})

patch("timbal.platform.utils._request", request).start()
provider = PlatformTracingProvider.configured(platform_config=CONFIG)

async def main():
    ctx = RunContext(tracing_provider=provider)
    ctx.platform_config = CONFIG
    ctx._trace["r"] = {{"call_id": "r", "path": "a", "t0": 0, "t1": 1, "parent_call_id": None}}
    asyncio.create_task(provider._store(ctx))
    await asyncio.sleep(0.05)

asyncio.run(main())
"""
        )
        subprocess.run([sys.executable, str(script)], check=True, timeout=30)
        lines = log.read_text().splitlines()
        assert len(lines) == 1 and json.loads(lines[0])["trace"][0]["t1"] == 1

    def test_patch_reads_are_bounded_unless_configured(self, monkeypatch):
        monkeypatch.delenv("TIMBAL_HTTP_READ_TIMEOUT", raising=False)
        assert PlatformTracingProvider._patch_timeout().read == PlatformTracingProvider._patch_read_timeout_s
        monkeypatch.setenv("TIMBAL_HTTP_READ_TIMEOUT", "none")
        assert PlatformTracingProvider._patch_timeout().read is None


class TestTransientFailures:
    @staticmethod
    def context(provider, *, ended=True):
        ctx = RunContext(tracing_provider=provider)
        ctx.platform_config = CONFIG
        root = {"call_id": "r", "path": "a", "t0": 0, "parent_call_id": None}
        ctx._trace["r"] = {**root, "t1": 1} if ended else root
        return ctx

    @staticmethod
    async def until(predicate, timeout=5.0):
        # A real time bound, and sleeps longer than a clock tick: on Windows before 3.13 asyncio's clock moves
        # in 15.6 ms steps, so sleep(0.01) can come back at once and a step count passes in no time at all.
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            if predicate():
                return
            await asyncio.sleep(0.02)

    async def test_an_ended_run_whose_save_failed_is_resent_in_the_background(self, monkeypatch):
        monkeypatch.setattr(PlatformTracingProvider, "_retry_delays_s", (0.01,))
        server = FakePlatform(advertise=False)
        server.fail_next = 3  # the save, then two background resends: a platform restart
        _, provider = tool_loop(1)
        ctx = self.context(provider)
        with patch("timbal.platform.utils._request", server.request):
            with pytest.raises(PlatformError):
                await provider._store(ctx)
            await self.until(lambda: not PlatformTracingProvider._retries)
        assert server.runs[ctx.id]["trace"][0]["t1"] == 1
        assert not PlatformTracingProvider._unsaved

    @pytest.mark.parametrize("by", ["background_resend", "shutdown_flush"])
    async def test_resends_keep_the_configured_provider(self, monkeypatch, by):
        """``deltas=False`` on a host that advertises deltas: a resend from a run's context must not fall back
        to the base class's negotiation."""
        monkeypatch.setattr(PlatformTracingProvider, "_retry_delays_s", (0.01 if by == "background_resend" else 60.0,))
        PlatformTracingProvider._host_caps(CONFIG.host).delta = True
        kinds = []

        async def request(*_args, headers=None, **_kwargs):
            kinds.append(headers["Content-Type"])
            if len(kinds) == 1:
                raise PlatformError("down", status_code=503)
            echo = {"X-Timbal-Snapshot": headers["X-Timbal-Snapshot"]}
            return SimpleNamespace(raise_for_status=lambda: None, headers=echo, json=lambda: {})

        provider = PlatformTracingProvider.configured(platform_config=CONFIG, deltas=False)
        ctx = self.context(provider)
        with patch("timbal.platform.utils._request", request):
            with pytest.raises(PlatformError):
                await provider._store(ctx)
            if by == "shutdown_flush":
                assert await PlatformTracingProvider.flush(timeout=1) == []
            await self.until(lambda: not PlatformTracingProvider._retries and len(kinds) > 1)
        assert kinds == ["application/json", "application/json"]

    async def test_a_buffered_run_whose_end_failed_is_resent_in_the_background(self, monkeypatch):
        """The end barrier holds the run in buffered uploads while it fails, then lets it go: the background
        resend must not take "uploads hold it" at that moment as "uploads will retry it"."""
        from timbal.state.tracing import uploads

        monkeypatch.setattr(PlatformTracingProvider, "_retry_delays_s", (0.01,))
        server = FakePlatform(advertise=False)
        server.fail_next = 1
        provider = PlatformTracingProvider.configured(
            platform_config=CONFIG, durability="buffered", flush_interval_s=0.05
        )
        ctx = self.context(provider)
        with patch("timbal.platform.utils._request", server.request):
            with pytest.raises(PlatformError):
                await uploads.write_through(ctx, provider.put)
            await self.until(lambda: ctx.id in server.runs)
        assert server.runs[ctx.id]["trace"][0]["t1"] == 1

    async def test_the_background_resends_give_up_after_the_window(self, monkeypatch):
        monkeypatch.setattr(PlatformTracingProvider, "_retry_delays_s", (0.01,))
        monkeypatch.setattr(PlatformTracingProvider, "_retry_window_s", 0.05)
        server = FakePlatform(advertise=False)
        server.fail_next = 10**6
        _, provider = tool_loop(1)
        ctx = self.context(provider)
        with patch("timbal.platform.utils._request", server.request), capture_logs() as logs:
            with pytest.raises(PlatformError):
                await provider._store(ctx)
            await self.until(lambda: not PlatformTracingProvider._retries)
        assert not PlatformTracingProvider._retries
        assert any(log["event"] == "platform_trace_retry_exhausted" for log in logs)
        assert [write.context for write in PlatformTracingProvider._unsaved.values()] == [ctx]  # still owed

    async def test_a_run_buffered_uploads_hold_all_window_is_not_reported_as_given_up(self, monkeypatch):
        from timbal.state.tracing import uploads

        monkeypatch.setattr(PlatformTracingProvider, "_retry_delays_s", (0.01,))
        monkeypatch.setattr(PlatformTracingProvider, "_retry_window_s", 0.05)
        monkeypatch.setattr(uploads, "tracked", lambda _context: True)
        server = FakePlatform(advertise=False)
        server.fail_next = 1
        _, provider = tool_loop(1)
        ctx = self.context(provider)
        with patch("timbal.platform.utils._request", server.request), capture_logs() as logs:
            with pytest.raises(PlatformError):
                await provider._store(ctx)
            await self.until(lambda: not PlatformTracingProvider._retries)
        events = [log["event"] for log in logs]
        assert "platform_trace_retry_handed_to_uploads" in events
        assert "platform_trace_retry_exhausted" not in events
        assert ctx.id not in server.runs  # every turn was left to the uploads

    async def test_a_rejected_or_mid_run_save_is_not_resent_in_the_background(self):
        async def forbidden(*_args, **_kwargs):
            raise PlatformError("forbidden", status_code=403)

        async def down(*_args, **_kwargs):
            raise PlatformError("down", status_code=503)

        _, provider = tool_loop(1)
        with patch("timbal.platform.utils._request", forbidden), pytest.raises(PlatformError):
            await provider._store(self.context(provider))
        with patch("timbal.platform.utils._request", down), pytest.raises(PlatformError):
            await provider._store(self.context(provider, ended=False))  # its next save carries it
        assert not PlatformTracingProvider._retries

    async def test_a_shutdown_flush_takes_over_the_background_resends(self, monkeypatch):
        monkeypatch.setattr(PlatformTracingProvider, "_retry_delays_s", (60.0,))
        server = FakePlatform(advertise=False)
        server.fail_next = 1
        _, provider = tool_loop(1)
        ctx = self.context(provider)
        with patch("timbal.platform.utils._request", server.request):
            with pytest.raises(PlatformError):
                await provider._store(ctx)
            assert PlatformTracingProvider._retries
            assert await PlatformTracingProvider.flush(timeout=1) == []  # does not wait out the 60 s
            await asyncio.sleep(0)
        assert server.runs[ctx.id]["trace"][0]["t1"] == 1
        assert not PlatformTracingProvider._retries

    async def test_a_delta_the_server_did_not_confirm_is_resent_whole(self):
        # Today's API reads a delta as a PATCH without `trace` and answers 204, echoing no snapshot.
        PlatformTracingProvider._host_caps(CONFIG.host).delta = True
        kinds = []

        async def todays_api(*_args, headers=None, **_kwargs):
            kinds.append(headers["Content-Type"])
            return SimpleNamespace(raise_for_status=lambda: None, headers={}, json=lambda: {})

        _, provider = tool_loop(1)
        ctx = self.context(provider)
        with patch("timbal.platform.utils._request", todays_api):
            await provider._store(ctx)
            await provider._store(ctx)
        assert kinds == [DELTA_MEDIA_TYPE, "application/json", "application/json"]
        assert PlatformTracingProvider._host_caps(CONFIG.host).delta_rejected

    async def test_no_span_takes_the_roots_seq(self):
        server = FakePlatform()
        PlatformTracingProvider._host_caps(CONFIG.host).delta = True
        _, provider = tool_loop(1)
        ctx = RunContext(tracing_provider=provider)
        ctx.platform_config = CONFIG
        ctx._trace["c"] = {"call_id": "c", "path": "a.x", "t0": 1, "t1": 2, "parent_call_id": "r"}  # before the root
        ctx._trace["r"] = {"call_id": "r", "path": "a", "t0": 0, "parent_call_id": None}
        with patch("timbal.platform.utils._request", server.request):
            await provider._store(ctx)
        assert [s["seq"] for s in server.patches[-1]["body"]["spans"]] == [1]

    async def test_large_bodies_compress_off_the_event_loop(self, monkeypatch):
        offloaded = []
        original = asyncio.to_thread

        async def spy(func, *args, **kwargs):
            offloaded.append(func)
            return await original(func, *args, **kwargs)

        monkeypatch.setattr(asyncio, "to_thread", spy)
        server = FakePlatform()
        PlatformTracingProvider._host_caps(CONFIG.host).gzip = True
        _, provider = tool_loop(1)
        ctx = self.context(provider)
        blob = "x" * (512 * 1024)
        ctx._trace["r"].metadata["blob"] = blob
        with patch("timbal.platform.utils._request", server.request):
            await provider._store(ctx)
        assert offloaded == [gzip.compress]
        assert server.patches[-1]["gzip"] and server.runs[ctx.id]["trace"][0]["metadata"]["blob"] == blob

    async def test_an_owed_save_pushed_out_of_the_bound_is_logged(self, monkeypatch):
        from timbal.state.tracing.providers import platform as platform_module

        monkeypatch.setattr(platform_module, "_UNSAVED_MAX", 1)

        async def down(*_args, **_kwargs):
            raise PlatformError("down", status_code=503)

        _, provider = tool_loop(1)
        first, second = self.context(provider, ended=False), self.context(provider, ended=False)
        with patch("timbal.platform.utils._request", down), capture_logs() as logs:
            for ctx in (first, second):
                with pytest.raises(PlatformError):
                    await provider._store(ctx)
        assert [log["run_id"] for log in logs if log["event"] == "platform_trace_unsaved_evicted"] == [first.id]

    def test_snapshots_are_unique_across_threads(self):
        from concurrent.futures import ThreadPoolExecutor

        from timbal.state.tracing.providers.platform import _next_snapshot

        with ThreadPoolExecutor(8) as pool:
            batches = list(pool.map(lambda _: [_next_snapshot() for _ in range(5000)], range(8)))
        flat = [snapshot for batch in batches for snapshot in batch]
        assert len(set(flat)) == len(flat)


def long_tool(provider, seconds, marks):
    """One tool call that takes ``seconds``; ``marks`` records how many PATCHes had landed as it started and ended."""

    async def wait(server):
        marks.append(len(server.patches))
        await asyncio.sleep(seconds)
        marks.append(len(server.patches))

    def handler(messages):
        if any(m.role == "tool" for m in messages):
            return "done"
        return Message(
            role="assistant", content=[ToolUseContent(id="c0", name="wait", input={})], stop_reason="tool_use"
        )

    def agent(server):
        async def wait_tool() -> str:
            await wait(server)
            return "waited"

        return Agent(
            name="agent",
            model=TestModel(handler=handler),
            tools=[Tool(name="wait", handler=wait_tool)],
            tracing_provider=provider,
        )

    return agent


class TestKeepAlive:
    @pytest.fixture(autouse=True)
    def no_timers(self):
        yield
        for entry in PlatformTracingProvider._keepalives.values():
            entry.cancel()
        PlatformTracingProvider._keepalives.clear()

    async def test_a_long_tool_keeps_the_run_saving_with_root_headers_only(self):
        server, marks = FakePlatform(), []
        provider = PlatformTracingProvider.configured(platform_config=CONFIG, keepalive_s=0.05)
        with patch("timbal.platform.utils._request", server.request):
            ctx, _ = await run(long_tool(provider, 0.4, marks)(server), provider)
            # The model step's own save may land as the tool starts; the rest are keep-alives.
            during = [p for p in server.patches[marks[0] : marks[1]] if p["body"].get("spans") == []]
            assert len(during) >= 4  # every ~50 ms of a 400 ms tool
            assert all(p["kind"] == "delta" and not p["body"]["ended"] for p in during)
            assert all(set(p["body"]["root"]) == {"call_id", "parent_call_id", "path", "t0"} for p in during)
            ended = len(server.patches)
            await asyncio.sleep(0.2)
            assert len(server.patches) == ended  # nothing after the end
            assert not any(entry.context is ctx for entry in PlatformTracingProvider._keepalives.values())
            same_trace(await read_back(provider, ctx.id), ctx, server)

    async def test_off_and_end_only_runs_do_not_keep_alive(self, monkeypatch):
        for provider in (
            PlatformTracingProvider.configured(platform_config=CONFIG, keepalive_s=0),
            PlatformTracingProvider.configured(platform_config=CONFIG, keepalive_s=0.05, durability="end"),
        ):
            server, marks = FakePlatform(), []
            with patch("timbal.platform.utils._request", server.request):
                await run(long_tool(provider, 0.3, marks)(server), provider)
            assert marks[1] == marks[0], provider.durability
        monkeypatch.setenv("TIMBAL_TRACE_KEEPALIVE_SECS", "0")
        server, marks = FakePlatform(), []
        provider = PlatformTracingProvider.configured(platform_config=CONFIG)
        with patch("timbal.platform.utils._request", server.request):
            await run(long_tool(provider, 0.3, marks)(server), provider)
        assert marks[1] == marks[0]

    async def test_a_run_that_stopped_saving_for_itself_stops_keeping_alive(self, monkeypatch):
        monkeypatch.setenv("TIMBAL_TRACE_KEEPALIVE_MAX_SECS", "0.12")
        server, marks = FakePlatform(), []
        provider = PlatformTracingProvider.configured(platform_config=CONFIG, keepalive_s=0.03)
        with patch("timbal.platform.utils._request", server.request), capture_logs() as logs:
            await run(long_tool(provider, 0.5, marks)(server), provider)
        during = marks[1] - marks[0]
        assert 2 <= during <= 5  # ~120 ms of keep-alives, then nothing until the tool ends
        assert [log for log in logs if log["event"] == "platform_trace_keepalive_stopped"]

    async def test_a_failed_keep_alive_is_logged_and_the_next_one_tries_again(self):
        server, marks = FakePlatform(), []
        provider = PlatformTracingProvider.configured(platform_config=CONFIG, keepalive_s=0.05)

        async def flaky(method, path, **kwargs):
            if method == "PATCH" and marks and len(marks) == 1 and not flaky.failed:
                flaky.failed = True
                raise PlatformError("upstream", status_code=503)
            return await server.request(method, path, **kwargs)

        flaky.failed = False
        with patch("timbal.platform.utils._request", flaky), capture_logs() as logs:
            await run(long_tool(provider, 0.35, marks)(server), provider)
        assert flaky.failed and marks[1] - marks[0] >= 2
        assert [log for log in logs if log["event"] == "platform_trace_keepalive_failed"]


class TestChatSessionOverDeltas:
    async def test_two_turns_chain_through_the_platform(self):
        from timbal.core.chat_session import ChatSession

        server = FakePlatform()
        PlatformTracingProvider._host_caps(CONFIG.host).delta = True
        agent, _ = tool_loop(2)
        with (
            patch("timbal.platform.utils._request", server.request),
            patch.object(PlatformTracingProvider, "platform_config", CONFIG),
        ):
            async with ChatSession(agent) as session:
                session.send("go")
                finished = []
                async for item in session.events():
                    if isinstance(item, dict) and item.get("kind") == "turn_finished":
                        finished.append(item)
                        if len(finished) == 1:
                            PlatformTracingProvider._recent.clear()  # the next turn reads the platform's copy
                            session.send("again")
                        else:
                            break
        assert [f["status"]["code"] for f in finished] == ["success", "success"]
        second = server.runs[finished[1]["run_id"]]["trace"]
        root = next(s for s in second if s["parent_call_id"] is None)
        users = [m for m in root["memory"] if m["role"] == "user"]
        assert len(users) == 2  # the first turn's prompt came back from the platform
        assert all(p["kind"] == "delta" for p in server.patches)
