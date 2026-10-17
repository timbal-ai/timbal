"""PlatformTracingProvider reads back what this process just stored, without the network."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from structlog.testing import capture_logs
from timbal.state.config import PlatformAuth, PlatformAuthType, PlatformConfig, PlatformSubject
from timbal.state.context import RunContext
from timbal.state.tracing.providers.platform import PlatformTracingProvider, _parse_cache_limits
from timbal.state.tracing.trace import Trace

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


def _body(**extra) -> bytes:
    return json.dumps({"parent_id": None, "trace": [{**RECORD, **extra}]}).encode()


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
def _default_settings(monkeypatch):
    monkeypatch.delenv("TIMBAL_TRACE_CACHE_MAX_ENTRIES", raising=False)
    monkeypatch.delenv("TIMBAL_TRACE_CACHE_TTL_SECS", raising=False)
    _parse_cache_limits.cache_clear()


@pytest.fixture
def clock(monkeypatch):
    now = [100.0]
    monkeypatch.setattr("timbal.state.tracing.providers.platform.time", SimpleNamespace(monotonic=lambda: now[0]))
    return now


def _remember(key: str, body: bytes) -> None:
    Provider._remember(key, next(PlatformTracingProvider._snapshots), body)


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

    request = AsyncMock(return_value=_ok({"trace": [dict(RECORD)]}))
    with patch("timbal.platform.utils._request", request):
        await Provider.get(_context(parent_id="run-1"))

    request.assert_awaited_once()


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
        _remember(run_id, _body())

    assert list(PlatformTracingProvider._recent) == ["b", "c"]


@pytest.mark.parametrize("operation", ["hit", "miss", "write"])
def test_cache_activity_removes_all_expired_entries(monkeypatch, clock, operation):
    monkeypatch.setenv("TIMBAL_TRACE_CACHE_TTL_SECS", "6.5")
    _remember("a", _body())
    clock[0] += 4
    _remember("b", _body())
    clock[0] += 4
    _remember("c", _body())
    clock[0] = 110.5  # a and b expired; c is still fresh.

    if operation == "write":
        _remember("d", _body())
    else:
        result = Provider._recall("c" if operation == "hit" else "missing")
        assert (result is not None) == (operation == "hit")

    assert list(PlatformTracingProvider._recent) == (["c", "d"] if operation == "write" else ["c"])


def test_resaving_a_run_replaces_its_snapshot_and_refreshes_expiry(monkeypatch, clock):
    monkeypatch.setenv("TIMBAL_TRACE_CACHE_MAX_ENTRIES", "2")
    monkeypatch.setenv("TIMBAL_TRACE_CACHE_TTL_SECS", "10")
    _remember("a", _body())
    _remember("b", _body())
    clock[0] += 6
    _remember("a", _body(metadata={"updated": True}))
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
        _remember(key, _body())
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
    capacity = PlatformTracingProvider._recent_max
    for i in range(capacity + 1):
        _remember(str(i), _body())
    assert list(PlatformTracingProvider._recent) == [str(i) for i in range(1, capacity + 1)]
    clock[0] += 299
    assert Provider._recall(str(capacity)) is not None
    clock[0] += 1
    assert Provider._recall(str(capacity)) is None
    assert not PlatformTracingProvider._recent


def test_an_invalid_setting_warns_once(monkeypatch):
    monkeypatch.setenv("TIMBAL_TRACE_CACHE_TTL_SECS", "5m")
    with capture_logs() as logs:
        for i in range(5):
            _remember(str(i), _body())
            Provider._recall(str(i))

    assert [log["event"] for log in logs] == ["invalid_trace_cache_ttl"]


@pytest.mark.asyncio
async def test_a_failed_resave_stops_serving_the_older_snapshot():
    await _store("run-1", AsyncMock(return_value=_ok()))
    failing = _ok()
    failing.raise_for_status = MagicMock(side_effect=RuntimeError("timed out after the write landed"))
    with pytest.raises(RuntimeError):
        await _store("run-1", AsyncMock(return_value=failing))

    request = AsyncMock(return_value=_ok({"trace": [dict(RECORD)]}))
    with patch("timbal.platform.utils._request", request):
        await Provider.get(_context(parent_id="run-1"))

    request.assert_awaited_once()


@pytest.mark.asyncio
async def test_the_cache_holds_what_was_sent_not_what_the_run_appended_since():
    stored = _context(run_id="run-1")
    stored._trace["c1"] = dict(RECORD)
    stored._trace["c1"]._memory_dump = [dict(RECORD["memory"][0])]

    async def request(**_):
        stored._trace["c1"]._memory_dump.append({"role": "assistant", "content": []})
        return _ok()

    with patch("timbal.platform.utils._request", request):
        await Provider._store(stored)
        trace = await Provider.get(_context(parent_id="run-1"))

    assert len(trace["c1"].memory) == 1


def test_an_oversized_snapshot_is_not_kept_and_drops_the_older_one(monkeypatch):
    _remember("a", _body())
    _remember("b", _body())
    monkeypatch.setattr(PlatformTracingProvider, "_recent_max_bytes", len(_body()) * 2)

    _remember("a", _body(metadata={"padding": "x" * len(_body()) * 2}))

    assert Provider._recall("a") is None
    assert Provider._recall("b") is not None


def test_the_byte_budget_evicts_the_oldest(monkeypatch):
    monkeypatch.setattr(PlatformTracingProvider, "_recent_max_bytes", len(_body()) * 2)
    for key in ("a", "b", "c"):
        _remember(key, _body())

    assert list(PlatformTracingProvider._recent) == ["b", "c"]


async def _overlapping_saves(older_fails: bool, newer_fails: bool) -> tuple[list[str], Trace | None]:
    """Save run-1 twice; the older snapshot's PATCH finishes after the newer one's. Returns the request methods."""
    stored = _context(run_id="run-1")
    stored._trace["c1"] = dict(RECORD)
    older_sent, release_older = asyncio.Event(), asyncio.Event()
    methods = []

    async def request(**kwargs):
        methods.append(kwargs["method"])
        if kwargs["method"] == "GET":
            return _ok({"trace": [dict(RECORD)]})
        if len(methods) == 1:
            older_sent.set()
            await release_older.wait()
            failed = older_fails
        else:
            failed = newer_fails
        if failed:
            raise RuntimeError("timed out after the write landed")
        return _ok()

    with patch("timbal.platform.utils._request", request):
        older = asyncio.create_task(Provider._store(stored))
        await older_sent.wait()
        stored._trace["c1"].metadata["snapshot"] = "newer"
        try:
            await Provider._store(stored)
        except RuntimeError:
            pass
        release_older.set()
        await asyncio.gather(older, return_exceptions=True)
        trace = await Provider.get(_context(parent_id="run-1"))
    return methods, trace


@pytest.mark.asyncio
async def test_an_older_save_finishing_last_does_not_replace_a_newer_one():
    methods, trace = await _overlapping_saves(older_fails=False, newer_fails=False)

    assert methods == ["PATCH", "PATCH"]
    assert trace["c1"].metadata == {"snapshot": "newer"}


@pytest.mark.asyncio
async def test_an_older_save_failing_last_keeps_the_newer_one():
    methods, trace = await _overlapping_saves(older_fails=True, newer_fails=False)

    assert methods == ["PATCH", "PATCH"]
    assert trace["c1"].metadata == {"snapshot": "newer"}


@pytest.mark.asyncio
async def test_an_older_save_succeeding_after_a_newer_one_failed_is_not_served():
    methods, _ = await _overlapping_saves(older_fails=False, newer_fails=True)

    assert methods == ["PATCH", "PATCH", "GET"]


def test_interleaved_conversations_keep_their_previous_turn():
    conversations = 64
    for conv in range(conversations):
        _remember(f"{conv}-turn-1", _body())

    assert all(Provider._recall(f"{conv}-turn-1") is not None for conv in range(conversations))
