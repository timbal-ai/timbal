"""Tests for Jira integration tools.

Unit tests run by default (mocked / no network).

Live Jira API tests are marked ``integration`` and skipped unless credentials are set.
Run them with::

    pytest -m integration python/tests/tools/test_jira.py

Site auth env vars::

    JIRA_BASE_URL=https://your-site.atlassian.net
    JIRA_API_TOKEN=...
    JIRA_EMAIL=you@example.com

OAuth env vars::

    JIRA_ACCESS_TOKEN=...   # or ATLASSIAN_ACCESS_TOKEN
    JIRA_CLOUD_ID=...       # optional; required if multiple sites
"""

from __future__ import annotations

import base64
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from timbal.codegen.tool_discovery import get_framework_tools
from timbal.platform.integrations import Integration
from timbal.tools.jira import (
    JiraGetMyself,
    JiraListFields,
    JiraListProjects,
    _jira_connection,
    _jira_request,
    _jira_use_site_auth,
    _resolve_cloud_id,
    _resolve_token,
)


def _jira_site_auth_configured() -> bool:
    base = (os.getenv("JIRA_BASE_URL") or "").strip()
    token = (os.getenv("JIRA_API_TOKEN") or "").strip()
    email = (os.getenv("JIRA_EMAIL") or os.getenv("JIRA_USER_EMAIL") or "").strip()
    return bool(base and token and email)


def _jira_oauth_configured() -> bool:
    return bool((os.getenv("JIRA_ACCESS_TOKEN") or os.getenv("ATLASSIAN_ACCESS_TOKEN") or "").strip())


def _skip_if_jira_live_env_not_set() -> None:
    if _jira_site_auth_configured() or _jira_oauth_configured():
        return
    pytest.skip(
        "Jira live integration: set site auth (JIRA_BASE_URL, JIRA_API_TOKEN, JIRA_EMAIL) "
        "or OAuth (JIRA_ACCESS_TOKEN or ATLASSIAN_ACCESS_TOKEN). "
        "Optional: JIRA_CLOUD_ID / ATLASSIAN_CLOUD_ID when using OAuth with multiple sites."
    )


class TestJiraPackageExports:
    def test_lazy_import_from_timbal_tools(self) -> None:
        from timbal.tools import JiraListProjects as exported

        assert exported is JiraListProjects

    def test_framework_discovery_registers_jira_provider(self) -> None:
        tools = get_framework_tools(no_cache=True)
        jira_tools = [ft for ft in tools.values() if ft.provider == "jira"]
        assert len(jira_tools) == 38
        assert any(ft.name == "jira_list_projects" for ft in jira_tools)


class TestJiraToolConfig:
    def test_get_config_includes_integration_and_token(self) -> None:
        tool = JiraListProjects(integration=Integration("jira", "org-int-1"))
        cfg = tool.get_config()
        assert "integration" in cfg
        assert "token" in cfg

    def test_tool_name_and_integration_annotation(self) -> None:
        tool = JiraListProjects()
        assert tool.name == "jira_list_projects"
        field = JiraListProjects.model_fields["integration"]
        assert field.annotation is not None


@pytest.mark.asyncio
async def test_resolve_token_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JIRA_ACCESS_TOKEN", "oauth-token")
    tool = JiraListProjects()
    assert await _resolve_token(tool) == "oauth-token"


@pytest.mark.asyncio
async def test_resolve_token_from_explicit_secret() -> None:
    tool = JiraListProjects(token=SecretStr("explicit-token"))
    assert await _resolve_token(tool) == "explicit-token"


@pytest.mark.asyncio
async def test_resolve_token_from_integration(monkeypatch: pytest.MonkeyPatch) -> None:
    tool = JiraListProjects(integration=Integration("jira", "int-1"))

    async def fake_resolve() -> dict[str, str]:
        return {"access_token": "platform-token"}

    monkeypatch.setattr(tool.integration, "resolve", fake_resolve)
    assert await _resolve_token(tool) == "platform-token"


@pytest.mark.asyncio
async def test_resolve_token_missing_raises() -> None:
    tool = JiraListProjects()
    with pytest.raises(ValueError, match="Jira credentials not found"):
        await _resolve_token(tool)


@pytest.mark.asyncio
async def test_jira_use_site_auth_when_env_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JIRA_BASE_URL", "https://example.atlassian.net")
    monkeypatch.setenv("JIRA_API_TOKEN", "api-token")
    monkeypatch.setenv("JIRA_EMAIL", "user@example.com")
    tool = JiraListProjects()
    assert await _jira_use_site_auth(tool) is True


@pytest.mark.asyncio
async def test_jira_use_site_auth_false_when_oauth_token_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JIRA_BASE_URL", "https://example.atlassian.net")
    monkeypatch.setenv("JIRA_API_TOKEN", "api-token")
    monkeypatch.setenv("JIRA_EMAIL", "user@example.com")
    tool = JiraListProjects(token=SecretStr("oauth-override"))
    assert await _jira_use_site_auth(tool) is False


@pytest.mark.asyncio
async def test_jira_connection_site_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JIRA_BASE_URL", "https://acme.atlassian.net")
    monkeypatch.setenv("JIRA_API_TOKEN", "secret")
    monkeypatch.setenv("JIRA_EMAIL", "dev@acme.com")
    tool = JiraListProjects()
    base, headers = await _jira_connection(tool, None, None)
    assert base == "https://acme.atlassian.net"
    assert headers["Accept"] == "application/json"
    assert headers["Authorization"].startswith("Basic ")
    decoded = base64.b64decode(headers["Authorization"].removeprefix("Basic ").encode()).decode()
    assert decoded == "dev@acme.com:secret"


@pytest.mark.asyncio
async def test_jira_connection_oauth_with_cloud_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JIRA_ACCESS_TOKEN", "oauth-token")
    monkeypatch.setenv("JIRA_CLOUD_ID", "cloud-abc")
    tool = JiraListProjects()
    base, headers = await _jira_connection(tool, None, None)
    assert base == "https://api.atlassian.com/ex/jira/cloud-abc"
    assert headers["Authorization"] == "Bearer oauth-token"


@pytest.mark.asyncio
async def test_resolve_cloud_id_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JIRA_ACCESS_TOKEN", "oauth-token")
    monkeypatch.setenv("JIRA_CLOUD_ID", "cloud-from-env")
    tool = JiraListProjects()
    assert await _resolve_cloud_id(tool, None, None) == "cloud-from-env"


@pytest.mark.asyncio
async def test_jira_request_site_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JIRA_BASE_URL", "https://site.atlassian.net")
    monkeypatch.setenv("JIRA_API_TOKEN", "tok")
    monkeypatch.setenv("JIRA_EMAIL", "a@b.com")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b'{"values":[]}'
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = {"values": []}
    mock_response.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.request = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    tool = JiraListProjects()
    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await _jira_request(tool, "GET", None, None, "/rest/api/3/project/search")

    assert result == {"values": []}
    call_kwargs = mock_client.request.call_args.kwargs
    assert call_kwargs["url"] == "https://site.atlassian.net/rest/api/3/project/search"
    assert call_kwargs["method"] == "GET"


@pytest.mark.asyncio
async def test_jira_list_projects_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JIRA_BASE_URL", "https://site.atlassian.net")
    monkeypatch.setenv("JIRA_API_TOKEN", "tok")
    monkeypatch.setenv("JIRA_EMAIL", "a@b.com")

    async def fake_request(
        _tool: Any,
        method: str,
        cloud_id: str | None,
        site_name: str | None,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        assert method == "GET"
        assert path == "/rest/api/3/project/search"
        assert params == {"startAt": 0, "maxResults": 50}
        return {"values": [{"key": "ENG"}]}

    monkeypatch.setattr("timbal.tools.jira._jira_request", fake_request)
    tool = JiraListProjects()
    out = await tool(start_at=0, max_results=50).collect()
    assert out.output == {"values": [{"key": "ENG"}]}


# ---------------------------------------------------------------------------
# Live Jira API (requires credentials + network; excluded from default pytest)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_jira_get_myself() -> None:
    """Call GET /rest/api/3/myself against a real Jira site."""
    _skip_if_jira_live_env_not_set()
    out = await JiraGetMyself().collect()
    assert out.status.code == "success", out.error
    assert isinstance(out.output, dict)
    assert out.output.get("accountId") or out.output.get("emailAddress")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_jira_list_projects() -> None:
    """Call project search against a real Jira site."""
    _skip_if_jira_live_env_not_set()
    out = await JiraListProjects(max_results=5).collect()
    assert out.status.code == "success", out.error
    assert isinstance(out.output, dict)
    assert "values" in out.output
    assert isinstance(out.output["values"], list)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_jira_list_fields() -> None:
    """Call GET /rest/api/3/field against a real Jira site."""
    _skip_if_jira_live_env_not_set()
    out = await JiraListFields().collect()
    assert out.status.code == "success", out.error
    assert isinstance(out.output, list)
    assert len(out.output) > 0
    assert "id" in out.output[0]
