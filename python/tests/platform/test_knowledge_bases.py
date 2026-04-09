"""Tests for timbal.platform.knowledge_bases (SDK-aligned ``query``)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from timbal.platform.knowledge_bases import query
from timbal.state import set_run_context
from timbal.state.config import PlatformAuth, PlatformAuthType, PlatformConfig, PlatformSubject
from timbal.state.context import RunContext


@pytest.fixture(autouse=True)
def _ctx(monkeypatch):
    monkeypatch.delenv("TIMBAL_ORG_ID", raising=False)
    monkeypatch.delenv("TIMBAL_KB_ID", raising=False)
    monkeypatch.setattr("timbal.state.config_loader._cached_default_config", None)
    monkeypatch.setattr("timbal.state.config_loader._default_config_resolved", False)
    cfg = PlatformConfig(
        host="api.timbal.ai",
        auth=PlatformAuth(type=PlatformAuthType.BEARER, token="t"),
        subject=PlatformSubject(org_id="org_1", app_id="400", version_id="v1"),
    )
    set_run_context(RunContext(platform_config=cfg, tracing_provider=None))
    yield
    set_run_context(None)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_query_k2_path_and_body():
    mock_res = MagicMock()
    mock_res.json.return_value = {"rows": []}
    with patch("timbal.platform.knowledge_bases.query._request", AsyncMock(return_value=mock_res)) as m:
        out = await query("SELECT 1", ["a"], kb_id="kb9", explain=True)
    m.assert_awaited_once_with(
        "POST",
        "orgs/org_1/k2/kb9/query",
        json={"sql": "SELECT 1", "params": ["a"], "explain": True},
    )
    assert out == {"rows": []}


@pytest.mark.asyncio
async def test_query_default_params_empty_list():
    mock_res = MagicMock()
    mock_res.json.return_value = {}
    with patch("timbal.platform.knowledge_bases.query._request", AsyncMock(return_value=mock_res)) as m:
        await query("SELECT 1", kb_id="k")
    body = m.await_args.kwargs["json"]
    assert body["params"] == []


@pytest.mark.asyncio
async def test_query_explicit_org_id():
    mock_res = MagicMock()
    mock_res.json.return_value = {}
    with patch("timbal.platform.knowledge_bases.query._request", AsyncMock(return_value=mock_res)) as m:
        await query("SELECT 1", org_id="other_org", kb_id="k")
    m.assert_awaited_once_with(
        "POST",
        "orgs/other_org/k2/k/query",
        json={"sql": "SELECT 1", "params": []},
    )


@pytest.mark.asyncio
async def test_query_env_fallback(monkeypatch):
    set_run_context(RunContext(platform_config=None, tracing_provider=None))
    monkeypatch.setenv("TIMBAL_ORG_ID", "env_org")
    monkeypatch.setenv("TIMBAL_KB_ID", "env_kb")
    mock_res = MagicMock()
    mock_res.json.return_value = {"rows": [1]}
    with patch("timbal.platform.knowledge_bases.query._request", AsyncMock(return_value=mock_res)) as m:
        await query("SELECT 1")
    m.assert_awaited_once_with(
        "POST",
        "orgs/env_org/k2/env_kb/query",
        json={"sql": "SELECT 1", "params": []},
    )


@pytest.mark.asyncio
async def test_query_legacy_path_wraps_list():
    mock_res = MagicMock()
    mock_res.json.return_value = [{"id": 1}]
    with patch("timbal.platform.knowledge_bases.query._request", AsyncMock(return_value=mock_res)) as m:
        out = await query("SELECT 1", kb_id="kb1", legacy=True)
    m.assert_awaited_once_with(
        "POST",
        "orgs/org_1/kbs/kb1/query",
        json={"sql": "SELECT 1", "params": []},
    )
    assert out == {"rows": [{"id": 1}]}


@pytest.mark.asyncio
async def test_query_requires_kb_id():
    with pytest.raises(ValueError, match="kb_id"):
        await query("SELECT 1", org_id="x")


@pytest.mark.asyncio
async def test_query_requires_org_id(monkeypatch):
    set_run_context(RunContext(platform_config=None, tracing_provider=None))
    monkeypatch.delenv("TIMBAL_ORG_ID", raising=False)
    with pytest.raises(ValueError, match="org_id"):
        await query("SELECT 1", kb_id="k")
