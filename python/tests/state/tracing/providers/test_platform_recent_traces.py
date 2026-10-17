"""PlatformTracingProvider reads back what this process just stored, without the network."""

from types import SimpleNamespace
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
def _forget(monkeypatch):
    monkeypatch.delenv("TIMBAL_TRACE_CACHE_MAX_ENTRIES", raising=False)
    monkeypatch.delenv("TIMBAL_TRACE_CACHE_TTL_SECS", raising=False)
    PlatformTracingProvider._recent.clear()
    yield
    PlatformTracingProvider._recent.clear()


@pytest.fixture
def clock(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(
        "timbal.state.tracing.providers.platform.time", SimpleNamespace(monotonic=lambda: now[0])
    )
    return now


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
async def test_an_old_or_unknown_run_comes_from_the_platform(clock):
    await _store("run-1", AsyncMock(return_value=_ok()))
    clock[0] += 300.0
    request = AsyncMock(return_value=_ok({"trace": [dict(RECORD)]}))

    with patch("timbal.platform.utils._request", request):
        old = await Provider.get(_context(parent_id="run-1"))
        unknown = await Provider.get(_context(parent_id="run-9"))

    assert request.await_count == 2
    assert old is not None and unknown is not None
    assert not PlatformTracingProvider._recent


def test_keeps_only_the_most_recent(monkeypatch):
    monkeypatch.setenv("TIMBAL_TRACE_CACHE_MAX_ENTRIES", "2")
    for run_id in ("a", "b", "c"):
        Provider._remember(run_id, [dict(RECORD)])

    assert list(PlatformTracingProvider._recent) == ["b", "c"]


@pytest.mark.parametrize("operation", ["hit", "miss", "write"])
def test_cache_activity_removes_all_expired_entries(monkeypatch, clock, operation):
    monkeypatch.setenv("TIMBAL_TRACE_CACHE_TTL_SECS", "6.5")
    Provider._remember("a", [dict(RECORD)])
    clock[0] += 4
    Provider._remember("b", [dict(RECORD)])
    clock[0] += 4
    Provider._remember("c", [dict(RECORD)])
    clock[0] = 110.5  # a and b expired; c is still fresh.

    if operation == "write":
        Provider._remember("d", [dict(RECORD)])
    else:
        result = Provider._recall("c" if operation == "hit" else "missing")
        assert (result is not None) == (operation == "hit")

    assert list(PlatformTracingProvider._recent) == (["c", "d"] if operation == "write" else ["c"])


def test_resaving_a_run_replaces_its_snapshot_and_refreshes_expiry(monkeypatch, clock):
    monkeypatch.setenv("TIMBAL_TRACE_CACHE_MAX_ENTRIES", "2")
    monkeypatch.setenv("TIMBAL_TRACE_CACHE_TTL_SECS", "10")
    Provider._remember("a", [dict(RECORD)])
    Provider._remember("b", [dict(RECORD)])
    clock[0] += 6
    Provider._remember("a", [{**RECORD, "metadata": {"updated": True}}])
    assert list(PlatformTracingProvider._recent) == ["b", "a"]

    clock[0] += 4
    assert Provider._recall("a")["c1"].metadata == {"updated": True}
    assert list(PlatformTracingProvider._recent) == ["a"]
    clock[0] += 6  # Reads did not extend the TTL.
    assert Provider._recall("a") is None
    assert not PlatformTracingProvider._recent


@pytest.mark.parametrize("setting", ["TIMBAL_TRACE_CACHE_MAX_ENTRIES", "TIMBAL_TRACE_CACHE_TTL_SECS"])
@pytest.mark.asyncio
async def test_zero_disables_cache_but_keeps_platform_reads_and_writes(monkeypatch, setting):
    await _store("run-1", AsyncMock(return_value=_ok()))
    monkeypatch.setenv(setting, "0")
    request = AsyncMock(return_value=_ok({"trace": [dict(RECORD)]}))
    await _store("run-2", request)
    with patch("timbal.platform.utils._request", request):
        assert await Provider.get(_context(parent_id="run-1")) is not None
    assert [call.kwargs["method"] for call in request.await_args_list] == ["PATCH", "GET"]
    assert not PlatformTracingProvider._recent


def test_lowering_capacity_applies_on_reads(monkeypatch):
    for key in ("a", "b", "c"):
        Provider._remember(key, [dict(RECORD)])
    monkeypatch.setenv("TIMBAL_TRACE_CACHE_MAX_ENTRIES", "1")
    assert Provider._recall("a") is None
    assert list(PlatformTracingProvider._recent) == ["c"]


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("TIMBAL_TRACE_CACHE_MAX_ENTRIES", "bad"),
        ("TIMBAL_TRACE_CACHE_MAX_ENTRIES", "-1"),
        ("TIMBAL_TRACE_CACHE_MAX_ENTRIES", "1.5"),
        ("TIMBAL_TRACE_CACHE_TTL_SECS", "bad"),
        ("TIMBAL_TRACE_CACHE_TTL_SECS", "-1"),
        ("TIMBAL_TRACE_CACHE_TTL_SECS", "nan"),
        ("TIMBAL_TRACE_CACHE_TTL_SECS", "inf"),
    ],
)
def test_invalid_settings_fall_back_to_bounded_defaults(monkeypatch, clock, setting, value):
    monkeypatch.setenv(setting, value)
    for i in range(17):
        Provider._remember(str(i), [dict(RECORD)])
    assert list(PlatformTracingProvider._recent) == [str(i) for i in range(1, 17)]
    clock[0] += 299
    assert Provider._recall("16") is not None
    clock[0] += 1
    assert Provider._recall("16") is None
    assert not PlatformTracingProvider._recent
