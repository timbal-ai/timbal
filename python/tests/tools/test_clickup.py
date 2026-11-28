from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr
from timbal.platform.integrations import Integration
from timbal.tools.clickup import (
    ClickUpCreateTask,
    ClickUpDeleteTask,
    ClickUpGetTask,
    ClickUpListWorkspaces,
    _auth_headers,
    _resolve_token,
)


def test_auth_headers_passes_the_token_verbatim() -> None:
    assert _auth_headers("pk_123") == {
        "Authorization": "pk_123",
        "Content-Type": "application/json",
    }


class TestClickUpPackageExports:
    def test_lazy_import_from_timbal_tools(self) -> None:
        from timbal.tools import ClickUpListWorkspaces as exported

        assert exported is ClickUpListWorkspaces

    def test_framework_discovery_registers_clickup_provider(self) -> None:
        from timbal.codegen.tool_discovery import get_framework_tools

        tools = get_framework_tools(no_cache=True)
        clickup_tools = [ft for ft in tools.values() if ft.provider == "clickup"]
        assert len(clickup_tools) == 15
        assert any(ft.name == "clickup_create_task" for ft in clickup_tools)


class TestClickUpToolConfig:
    def test_get_config_includes_integration_and_token(self) -> None:
        tool = ClickUpGetTask(integration=Integration("clickup", "org-int-1"))
        cfg = tool.get_config()
        assert "integration" in cfg
        assert "token" in cfg

    def test_tool_name_and_integration_annotation(self) -> None:
        tool = ClickUpGetTask()
        assert tool.name == "clickup_get_task"
        field = ClickUpGetTask.model_fields["integration"]
        assert field.annotation is not None


@pytest.mark.asyncio
async def test_resolve_token_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLICKUP_API_TOKEN", "env-token")
    tool = SimpleNamespace(integration=None, token=None)
    assert await _resolve_token(tool) == "env-token"


@pytest.mark.asyncio
async def test_resolve_token_from_explicit_secret() -> None:
    tool = SimpleNamespace(integration=None, token=SecretStr("explicit-token"))
    assert await _resolve_token(tool) == "explicit-token"


@pytest.mark.asyncio
async def test_resolve_token_from_integration_token_key() -> None:
    integration = MagicMock(spec=Integration)
    integration.resolve = AsyncMock(return_value={"token": "platform-token"})
    tool = SimpleNamespace(integration=integration, token=None)
    assert await _resolve_token(tool) == "platform-token"


@pytest.mark.asyncio
async def test_resolve_token_from_integration_access_token_key() -> None:
    integration = MagicMock(spec=Integration)
    integration.resolve = AsyncMock(return_value={"access_token": "oauth-access"})
    tool = SimpleNamespace(integration=integration, token=None)
    assert await _resolve_token(tool) == "oauth-access"


@pytest.mark.asyncio
async def test_resolve_token_missing_raises() -> None:
    tool = SimpleNamespace(integration=None, token=None)
    with pytest.raises(ValueError, match="ClickUp token not found"):
        await _resolve_token(tool)


@pytest.mark.asyncio
async def test_list_workspaces_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLICKUP_API_TOKEN", "tok")

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"teams": [{"id": "1", "name": "Acme"}]}

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=mock_client):
        out = await ClickUpListWorkspaces()().collect()

    assert out.output == {"teams": [{"id": "1", "name": "Acme"}]}
    call_kwargs = mock_client.get.call_args.kwargs
    assert call_kwargs["headers"]["Authorization"] == "tok"


@pytest.mark.asyncio
async def test_create_task_handler_sends_only_provided_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLICKUP_API_TOKEN", "tok")

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"id": "task-1", "name": "Ship it"}

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=mock_client):
        out = await ClickUpCreateTask()(list_id="list-1", name="Ship it").collect()

    assert out.output == {"id": "task-1", "name": "Ship it"}
    call_kwargs = mock_client.post.call_args.kwargs
    assert call_kwargs["json"] == {"name": "Ship it"}
    assert str(mock_client.post.call_args.args[0]).endswith("/list/list-1/task")


@pytest.mark.asyncio
async def test_delete_task_handler_handles_empty_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLICKUP_API_TOKEN", "tok")

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.content = b""

    mock_client = MagicMock()
    mock_client.delete = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=mock_client):
        out = await ClickUpDeleteTask()(task_id="task-1").collect()

    assert out.output == {"success": True}
