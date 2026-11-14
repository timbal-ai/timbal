"""Platform saves from chat sessions: buffered by default (sync once the host takes deltas), optional gzip bodies."""

import gzip
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from timbal import Agent, Tool
from timbal.core.chat_session import ChatSession
from timbal.core.test_model import TestModel
from timbal.state import get_run_context, set_call_id, set_parent_call_id, set_run_context
from timbal.state.config import PlatformAuth, PlatformAuthType, PlatformConfig, PlatformSubject
from timbal.state.context import RunContext
from timbal.state.tracing import uploads
from timbal.state.tracing.providers import InMemoryTracingProvider
from timbal.state.tracing.providers.platform import PlatformTracingProvider
from timbal.types.content import ToolUseContent
from timbal.types.message import Message

CONFIG = PlatformConfig(
    host="api.example.test",
    auth=PlatformAuth(type=PlatformAuthType.BEARER, token="t2_test"),
    subject=PlatformSubject(org_id="1", app_id="2"),
)


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.delenv(uploads.DURABILITY_ENV, raising=False)
    monkeypatch.delenv("TIMBAL_TRACE_GZIP", raising=False)
    set_run_context(None)
    set_call_id(None)
    set_parent_call_id(None)
    yield
    set_run_context(None)


class FakePlatform(PlatformTracingProvider):
    """A platform provider whose storage is a dict: no credentials, no network."""

    runs: dict = {}

    @classmethod
    async def get(cls, run_context):
        return cls.runs.get(run_context.parent_id)

    @classmethod
    async def _store(cls, run_context):
        from timbal.state.tracing.trace import Trace

        cls.runs[run_context.id] = Trace(run_context._trace.model_dump())


def probe_agent(provider):
    seen = []

    def probe() -> str:
        seen.append(uploads.durability(get_run_context()._tracing_provider))
        return "ok"

    def handler(messages):
        if messages[-1].role == "tool":
            return "done"
        return Message(
            role="assistant", content=[ToolUseContent(id="c1", name="probe", input={})], stop_reason="tool_use"
        )

    agent = Agent(
        name="agent",
        model=TestModel(handler=handler),
        tools=[Tool(name="probe", handler=probe)],
        tracing_provider=provider,
    )
    return agent, seen


async def one_turn(agent, **kwargs):
    async with ChatSession(agent, **kwargs) as session:
        session.send("go")
        async for item in session.events():
            if isinstance(item, dict) and item.get("kind") == "turn_finished":
                return item


class TestChatTurnDurability:
    async def test_platform_turns_are_buffered_by_default(self):
        agent, seen = probe_agent(FakePlatform)
        finished = await one_turn(agent)
        assert finished["status"]["code"] == "success"
        assert seen == ["buffered"]

    async def test_platform_turns_are_sync_once_the_host_takes_deltas(self):
        caps = PlatformTracingProvider._host_caps(CONFIG.host)
        caps.delta = True
        try:
            agent, seen = probe_agent(FakePlatform.configured(platform_config=CONFIG))
            await one_turn(agent)
            caps.delta_rejected = True
            await one_turn(agent)
        finally:
            PlatformTracingProvider._caps.pop(CONFIG.host, None)
        assert seen == ["sync", "buffered"]

    async def test_explicit_durability_wins(self):
        agent, seen = probe_agent(FakePlatform)
        await one_turn(agent, durability="sync")
        assert seen == ["sync"]

    async def test_a_provider_that_chose_keeps_its_mode(self):
        agent, seen = probe_agent(FakePlatform.configured(durability="end"))
        await one_turn(agent)
        assert seen == ["end"]

    async def test_the_env_setting_keeps_its_mode(self, monkeypatch):
        monkeypatch.setenv(uploads.DURABILITY_ENV, "sync")
        agent, seen = probe_agent(FakePlatform)
        await one_turn(agent)
        assert seen == ["sync"]

    async def test_other_providers_are_left_alone(self):
        agent, seen = probe_agent(InMemoryTracingProvider)
        await one_turn(agent)
        assert seen == ["sync"]

    async def test_history_still_chains_across_buffered_turns(self):
        agent, _ = probe_agent(FakePlatform)
        async with ChatSession(agent) as session:
            session.send("go")
            finished = []
            async for item in session.events():
                if isinstance(item, dict) and item.get("kind") == "turn_finished":
                    finished.append(item)
                    if len(finished) == 1:
                        session.send("again")
                    else:
                        break
        second = FakePlatform.runs[finished[1]["run_id"]]
        root = next(s for s in second.as_records() if s.parent_call_id is None)
        assert [m["role"] for m in root.memory][:4] == ["user", "assistant", "tool", "assistant"]

    def test_invalid_durability(self):
        agent, _ = probe_agent(FakePlatform)
        with pytest.raises(ValueError, match="durability"):
            ChatSession(agent, durability="later")


def _context():
    ctx = RunContext(tracing_provider=PlatformTracingProvider)
    ctx.platform_config = CONFIG
    ctx._trace["c1"] = {"call_id": "c1", "parent_call_id": None, "path": "agent", "t0": 0}
    return ctx


def _ok():
    res = MagicMock()
    res.raise_for_status = MagicMock()
    return res


class TestCompression:
    @pytest.mark.parametrize("setting", ["attr", "env"])
    async def test_gzip_body(self, setting, monkeypatch):
        provider = PlatformTracingProvider.configured(platform_config=CONFIG, compress=True)
        if setting == "env":
            provider = PlatformTracingProvider.configured(platform_config=CONFIG)
            monkeypatch.setenv("TIMBAL_TRACE_GZIP", "1")
        request = AsyncMock(return_value=_ok())
        with patch("timbal.platform.utils._request", request):
            await provider._store(_context())
        kwargs = request.call_args.kwargs
        assert kwargs["headers"]["Content-Encoding"] == "gzip"
        payload = json.loads(gzip.decompress(kwargs["content"]))
        assert payload["trace"][0]["call_id"] == "c1"

    async def test_off_by_default(self):
        provider = PlatformTracingProvider.configured(platform_config=CONFIG)
        request = AsyncMock(return_value=_ok())
        with patch("timbal.platform.utils._request", request):
            await provider._store(_context())
        kwargs = request.call_args.kwargs
        assert "Content-Encoding" not in kwargs["headers"]
        assert json.loads(kwargs["content"])["trace"][0]["call_id"] == "c1"

    async def test_the_cache_keeps_the_plain_body(self):
        provider = PlatformTracingProvider.configured(platform_config=CONFIG, compress=True)
        ctx = _context()
        request = AsyncMock(return_value=_ok())
        with patch("timbal.platform.utils._request", request):
            await provider._store(ctx)
            reader = RunContext(parent_id=ctx.id, tracing_provider=provider)
            reader.platform_config = CONFIG
            trace = await provider.get(reader)
        assert request.call_count == 1  # read back from memory
        assert trace["c1"].path == "agent"
