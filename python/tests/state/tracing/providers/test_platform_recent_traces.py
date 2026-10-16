"""PlatformTracingProvider reads back what this process just stored, without the network."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from timbal.state.config import PlatformAuth, PlatformAuthType, PlatformConfig, PlatformSubject
from timbal.state.context import RunContext
from timbal.state.tracing.providers.platform import PlatformTracingProvider

CONFIG = PlatformConfig(
    host="api.example.test",
    auth=PlatformAuth(type=PlatformAuthType.BEARER, token="t2_test"),
    subject=PlatformSubject(org_id="1", app_id="2"),
)
Provider = PlatformTracingProvider.configured(platform_config=CONFIG)

RECORD = {
    "call_id": "c1",
    "parent_call_id": None,
    "path": "voice_agent",
    "t0": 0,
    "memory": [{"role": "user", "content": [{"type": "text", "text": "hola"}]}],
}


def _ok(json_body=None):
    res = MagicMock()
    res.raise_for_status = MagicMock()
    res.json = MagicMock(return_value=json_body or {})
    return res


def _context(run_id: str | None = None, parent_id: str | None = None, config: PlatformConfig = CONFIG) -> RunContext:
    ctx = RunContext(tracing_provider=Provider)
    if run_id is not None:
        object.__setattr__(ctx, "id", run_id)
    object.__setattr__(ctx, "parent_id", parent_id)
    ctx.platform_config = config
    return ctx


@pytest.fixture(autouse=True)
def _forget():
    PlatformTracingProvider._recent.clear()
    yield
    PlatformTracingProvider._recent.clear()


async def _store(run_id: str, request: AsyncMock) -> None:
    stored = _context(run_id=run_id)
    stored._trace["c1"] = dict(RECORD)
    stored._trace["c1"]._memory_dump = RECORD["memory"]
    with patch("timbal.platform.utils._request", request):
        await Provider._store(stored)


@pytest.mark.asyncio
async def test_the_next_turn_reads_the_previous_one_without_a_request():
    request = AsyncMock(return_value=_ok())
    await _store("run-1", request)
    request.reset_mock()

    with patch("timbal.platform.utils._request", request):
        trace = await Provider.get(_context(parent_id="run-1"))

    request.assert_not_called()
    assert trace is not None
    assert trace["c1"].memory[0]["content"][0]["text"] == "hola"


@pytest.mark.asyncio
async def test_a_reader_cannot_change_what_the_next_reader_sees():
    await _store("run-1", AsyncMock(return_value=_ok()))

    first = await Provider.get(_context(parent_id="run-1"))
    first["c1"].memory.append({"role": "assistant", "content": []})
    second = await Provider.get(_context(parent_id="run-1"))

    assert len(second["c1"].memory) == 1


@pytest.mark.asyncio
async def test_a_failed_store_is_not_remembered():
    failing = _ok()
    failing.raise_for_status = MagicMock(side_effect=RuntimeError("500"))
    with pytest.raises(RuntimeError):
        await _store("run-1", AsyncMock(return_value=failing))

    assert not PlatformTracingProvider._recent


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "other",
    [
        CONFIG.model_copy(update={"auth": PlatformAuth(type=PlatformAuthType.BEARER, token="t2_someone_else")}),
        CONFIG.model_copy(update={"subject": PlatformSubject(org_id="9", app_id="2")}),
        CONFIG.model_copy(update={"subject": PlatformSubject(org_id="1", app_id="9")}),
    ],
    ids=["token", "org", "app"],
)
async def test_another_token_org_or_app_asks_the_platform(other):
    await _store("run-1", AsyncMock(return_value=_ok()))
    request = AsyncMock(return_value=_ok({"trace": [dict(RECORD)]}))

    with patch("timbal.platform.utils._request", request):
        await Provider.get(_context(parent_id="run-1", config=other))

    request.assert_awaited_once()


@pytest.mark.asyncio
async def test_an_old_or_unknown_run_comes_from_the_platform(monkeypatch):
    await _store("run-1", AsyncMock(return_value=_ok()))
    monkeypatch.setattr(PlatformTracingProvider, "_recent_ttl_s", -1.0)
    request = AsyncMock(return_value=_ok({"trace": [dict(RECORD)]}))

    with patch("timbal.platform.utils._request", request):
        old = await Provider.get(_context(parent_id="run-1"))
        unknown = await Provider.get(_context(parent_id="run-9"))

    assert request.await_count == 2
    assert old is not None and unknown is not None


def test_keeps_only_the_most_recent(monkeypatch):
    monkeypatch.setattr(PlatformTracingProvider, "_recent_max", 2)
    for run_id in ("a", "b", "c"):
        Provider._remember(run_id, [dict(RECORD)])

    assert list(PlatformTracingProvider._recent) == ["b", "c"]
