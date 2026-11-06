"""PlatformTracingProvider span deltas and negotiated gzip, against a fake platform server.

The fake implements the server side of rfcs/platform-trace-storage.md (A + C): request
decompression, `Accept-Encoding` / `Accept-Patch` advertisement, span upserts ordered by
`snapshot`, assembly at `ended`, and GET of a never-ended run from its span rows.
"""

import asyncio
import gzip
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from timbal import Agent, Tool
from timbal.core.test_model import TestModel
from timbal.errors import PlatformError
from timbal.state import get_run_context, set_call_id, set_parent_call_id, set_run_context
from timbal.state.config import PlatformAuth, PlatformAuthType, PlatformConfig, PlatformSubject
from timbal.state.context import RunContext
from timbal.state.tracing.providers.platform import DELTA_MEDIA_TYPE, PlatformTracingProvider
from timbal.types.content import ToolUseContent
from timbal.types.message import Message

CONFIG = PlatformConfig(
    host="api.deltas.test",
    auth=PlatformAuth(type=PlatformAuthType.BEARER, token="t2_test"),
    subject=PlatformSubject(org_id="1", app_id="2"),
)


@pytest.fixture(autouse=True)
def clean():
    PlatformTracingProvider._caps.clear()
    PlatformTracingProvider._delta_runs.clear()
    PlatformTracingProvider._recent.clear()
    PlatformTracingProvider._flights.clear()
    set_run_context(None)
    set_call_id(None)
    set_parent_call_id(None)
    yield
    PlatformTracingProvider._caps.clear()
    PlatformTracingProvider._delta_runs.clear()
    PlatformTracingProvider._recent.clear()


class FakePlatform:
    def __init__(self, *, deltas=True, gzip_ok=True, advertise=True):
        self.deltas, self.gzip_ok, self.advertise = deltas, gzip_ok, advertise
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

    def _ok(self, payload=None):
        return SimpleNamespace(raise_for_status=lambda: None, headers=self._headers(), json=lambda: payload or {})

    async def request(self, method, path, headers=None, content=None, **_):
        run_id = path.rsplit("/", 1)[-1]
        if method == "GET":
            run = self.runs.get(run_id)
            if run is None:
                return self._ok({})
            if run.get("trace") is not None:
                return self._ok({"trace": run["trace"], "parent_id": run["parent_id"]})
            rows = [span for _, (_seq, _snap, span) in sorted(run["spans"].items(), key=lambda kv: kv[1][0])]
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
        body = json.loads(raw)
        kind = "delta" if headers.get("Content-Type") == DELTA_MEDIA_TYPE else "full"
        if kind == "delta" and not self.deltas:
            raise PlatformError("unsupported media type", status_code=415)
        self.patches.append(
            {"kind": kind, "bytes": len(content), "body": body, "gzip": headers.get("Content-Encoding") == "gzip"}
        )
        run = self.runs.setdefault(run_id, {"spans": {}, "trace": None, "root": None, "parent_id": None, "snapshot": 0})
        run["parent_id"] = body.get("parent_id")
        if kind == "full":  # authoritative: replaces the span rows
            run.update(trace=body["trace"], spans={}, snapshot=0)
            return self._ok()
        if body["snapshot"] <= run["snapshot"]:
            return self._ok()  # stale or retried delta: ack, no-op
        run["snapshot"] = body["snapshot"]
        for entry in body["spans"]:
            assert set(entry) == {"seq", "span"}, entry
            run["spans"][entry["span"]["call_id"]] = (entry["seq"], body["snapshot"], entry["span"])
        run["root"] = body["root"]
        if body["ended"]:
            rows = [span for _, (_seq, _snap, span) in sorted(run["spans"].items(), key=lambda kv: kv[1][0])]
            run["trace"] = [body["root"], *rows]
        else:
            run["trace"] = None  # live again (background work after the end) — assembled on read
        return self._ok()


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
        assert "memory" not in deltas[0]["root"]
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

    async def test_snapshots_keep_rising_in_a_new_process(self):
        from timbal.state.tracing.providers import platform as platform_module

        server = FakePlatform()
        PlatformTracingProvider._host_caps(CONFIG.host).delta = True
        agent, provider = tool_loop(2)
        with patch("timbal.platform.utils._request", server.request):
            ctx, _ = await run(agent, provider)
            first = max(p["body"]["snapshot"] for p in server.patches)
            # Another process continues the run (an approval resumed elsewhere): no counter carried over.
            platform_module._last_snapshot = 0
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
        from timbal.state.tracing.providers import platform as platform_module

        ahead = platform_module._last_snapshot + 10**15  # another process' clock runs ahead
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

    def test_patch_reads_are_bounded_unless_configured(self, monkeypatch):
        monkeypatch.delenv("TIMBAL_HTTP_READ_TIMEOUT", raising=False)
        assert PlatformTracingProvider._patch_timeout().read == PlatformTracingProvider._patch_read_timeout_s
        monkeypatch.setenv("TIMBAL_HTTP_READ_TIMEOUT", "none")
        assert PlatformTracingProvider._patch_timeout().read is None


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
