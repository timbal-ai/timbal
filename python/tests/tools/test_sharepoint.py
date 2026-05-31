"""Unit tests for SharePoint integration tools (mocked / no network).

Live Microsoft Graph tests are marked ``integration`` and require platform OAuth or
``SHAREPOINT_ACCESS_TOKEN`` plus ``SHAREPOINT_SITE_ID`` in the environment.

Run live tests with::

    uv run pytest python/tests/tools/test_sharepoint.py -m integration -v
"""

from __future__ import annotations

import base64
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from timbal.codegen.tool_discovery import get_framework_tools
from timbal.platform.integrations import Integration
from timbal.tools.sharepoint import (
    SharePointCopyItem,
    SharePointCreateShareLink,
    SharePointDeletePermission,
    SharePointDownloadFile,
    SharePointGetFile,
    SharePointInvite,
    SharePointListPermissions,
    SharePointMoveItem,
    SharePointSearchSites,
    _resolve_token,
)

SHAREPOINT_TOOL_COUNT = 21


def _skip_if_sharepoint_live_env_not_set() -> None:
    if os.getenv("SHAREPOINT_ACCESS_TOKEN") and os.getenv("SHAREPOINT_SITE_ID"):
        return
    pytest.skip(
        "SharePoint live integration: set SHAREPOINT_ACCESS_TOKEN and SHAREPOINT_SITE_ID "
        "(Microsoft Graph OAuth bearer token and site ID)."
    )


def _tool_with_integration() -> SharePointSearchSites:
    return SharePointSearchSites(integration=Integration("sharepoint", "org-int-1"))


def _mock_httpx_client(*, get_responses: list[MagicMock] | None = None) -> MagicMock:
    """Build an AsyncClient mock; sequential .get() returns from get_responses."""
    get_responses = list(get_responses or [])
    get_index = {"i": 0}

    async def _get(*_args: Any, **_kwargs: Any) -> MagicMock:
        if get_index["i"] >= len(get_responses):
            raise AssertionError("Unexpected GET call")
        resp = get_responses[get_index["i"]]
        get_index["i"] += 1
        return resp

    mock_client = MagicMock()
    mock_client.get = AsyncMock(side_effect=_get)
    mock_client.post = AsyncMock()
    mock_client.patch = AsyncMock()
    mock_client.put = AsyncMock()
    mock_client.delete = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    return mock_client


def _json_response(data: dict[str, Any], *, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = data
    resp.raise_for_status = MagicMock()
    return resp


def _bytes_response(
    content: bytes,
    *,
    content_type: str = "application/octet-stream",
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.content = content
    resp.headers = {"content-type": content_type}
    resp.raise_for_status = MagicMock()
    return resp


class TestSharePointPackageExports:
    def test_lazy_import_from_timbal_tools(self) -> None:
        from timbal.tools import SharePointSearchSites as exported

        assert exported is SharePointSearchSites

    def test_framework_discovery_registers_sharepoint_provider(self) -> None:
        tools = get_framework_tools(no_cache=True)
        sp_tools = [ft for ft in tools.values() if ft.provider == "sharepoint"]
        assert len(sp_tools) == SHAREPOINT_TOOL_COUNT
        names = {ft.name for ft in sp_tools}
        assert "sharepoint_search_sites" in names
        assert "sharepoint_move_item" in names
        assert "sharepoint_delete_permission" in names


class TestSharePointToolConfig:
    def test_get_config_requires_integration(self) -> None:
        tool = _tool_with_integration()
        cfg = tool.get_config()
        assert "integration" in cfg
        assert cfg["integration"]["value"] == "org-int-1"

    def test_tool_name(self) -> None:
        assert SharePointSearchSites().name == "sharepoint_search_sites"


@pytest.mark.asyncio
async def test_resolve_token_from_integration() -> None:
    tool = _tool_with_integration()

    async def fake_resolve() -> dict[str, str]:
        return {"token": "graph-token"}

    tool.integration.resolve = fake_resolve  # type: ignore[method-assign]
    assert await _resolve_token(tool) == "graph-token"


@pytest.mark.asyncio
async def test_resolve_token_missing_raises() -> None:
    tool = SharePointSearchSites()
    with pytest.raises(ValueError, match="SharePoint integration not configured"):
        await _resolve_token(tool)


@pytest.mark.asyncio
async def test_search_sites_handler() -> None:
    tool = _tool_with_integration()

    async def fake_resolve() -> dict[str, str]:
        return {"token": "tok"}

    tool.integration.resolve = fake_resolve  # type: ignore[method-assign]

    mock_response = _json_response({"value": [{"id": "site-1", "displayName": "Marketing"}]})
    mock_client = _mock_httpx_client(get_responses=[mock_response])

    with patch("httpx.AsyncClient", return_value=mock_client):
        out = await tool(query="marketing").collect()

    assert out.status.code == "success", out.error
    assert out.output == {"value": [{"id": "site-1", "displayName": "Marketing"}]}
    call_kwargs = mock_client.get.call_args.kwargs
    assert call_kwargs["params"] == {"search": "marketing"}
    assert call_kwargs["headers"]["Authorization"] == "Bearer tok"


@pytest.mark.asyncio
async def test_get_file_returns_base64_for_binary() -> None:
    tool = SharePointGetFile(integration=Integration("sharepoint", "int-1"))
    pdf_bytes = b"%PDF-1.4 fake"

    async def fake_resolve() -> dict[str, str]:
        return {"token": "tok"}

    tool.integration.resolve = fake_resolve  # type: ignore[method-assign]

    meta_resp = _json_response({"id": "file-1", "name": "doc.pdf"})
    content_resp = _bytes_response(pdf_bytes, content_type="application/pdf")
    mock_client = _mock_httpx_client(get_responses=[meta_resp, content_resp])

    with patch("httpx.AsyncClient", return_value=mock_client):
        out = await tool(site_id="site-1", file_id="file-1").collect()

    assert out.status.code == "success", out.error
    assert out.output["content_type"] == "application/pdf"
    assert base64.b64decode(out.output["content_base64"]) == pdf_bytes
    assert "content" not in out.output


@pytest.mark.asyncio
async def test_get_file_includes_text_for_plain_text() -> None:
    tool = SharePointGetFile(integration=Integration("sharepoint", "int-1"))
    text_bytes = b"hello sharepoint"

    async def fake_resolve() -> dict[str, str]:
        return {"token": "tok"}

    tool.integration.resolve = fake_resolve  # type: ignore[method-assign]

    meta_resp = _json_response({"id": "file-2", "name": "notes.txt"})
    content_resp = _bytes_response(text_bytes, content_type="text/plain")
    mock_client = _mock_httpx_client(get_responses=[meta_resp, content_resp])

    with patch("httpx.AsyncClient", return_value=mock_client):
        out = await tool(site_id="site-1", file_id="file-2").collect()

    assert out.output["content"] == "hello sharepoint"
    assert out.output["content_base64"] == base64.b64encode(text_bytes).decode("ascii")


@pytest.mark.asyncio
async def test_download_file_binary() -> None:
    tool = SharePointDownloadFile(integration=Integration("sharepoint", "int-1"))
    raw = b"\x00\x01\x02"

    async def fake_resolve() -> dict[str, str]:
        return {"token": "tok"}

    tool.integration.resolve = fake_resolve  # type: ignore[method-assign]

    mock_client = _mock_httpx_client(get_responses=[_bytes_response(raw)])
    with patch("httpx.AsyncClient", return_value=mock_client):
        out = await tool(site_id="site-1", file_id="file-3").collect()

    assert out.output["size"] == 3
    assert base64.b64decode(out.output["content_base64"]) == raw


@pytest.mark.asyncio
async def test_move_item_handler() -> None:
    tool = SharePointMoveItem(integration=Integration("sharepoint", "int-1"))

    async def fake_resolve() -> dict[str, str]:
        return {"token": "tok"}

    tool.integration.resolve = fake_resolve  # type: ignore[method-assign]

    mock_response = _json_response({"id": "item-1", "name": "moved.docx"})
    mock_client = _mock_httpx_client()
    mock_client.patch = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_client):
        out = await tool(
            site_id="site-1",
            item_id="item-1",
            new_parent_folder_id="folder-2",
            new_name="moved.docx",
        ).collect()

    assert out.output["name"] == "moved.docx"
    patch_kwargs = mock_client.patch.call_args.kwargs
    assert patch_kwargs["json"] == {
        "parentReference": {"id": "folder-2"},
        "name": "moved.docx",
    }


@pytest.mark.asyncio
async def test_copy_item_returns_monitor_url() -> None:
    tool = SharePointCopyItem(integration=Integration("sharepoint", "int-1"))

    async def fake_resolve() -> dict[str, str]:
        return {"token": "tok"}

    tool.integration.resolve = fake_resolve  # type: ignore[method-assign]

    mock_response = MagicMock()
    mock_response.status_code = 202
    mock_response.headers = {"Location": "https://graph.microsoft.com/v1.0/monitor/abc"}
    mock_response.raise_for_status = MagicMock()

    mock_client = _mock_httpx_client()
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_client):
        out = await tool(
            site_id="site-1",
            item_id="item-1",
            target_parent_folder_id="folder-dest",
        ).collect()

    assert out.output["monitor_url"] == "https://graph.microsoft.com/v1.0/monitor/abc"


@pytest.mark.asyncio
async def test_list_permissions_handler() -> None:
    tool = SharePointListPermissions(integration=Integration("sharepoint", "int-1"))

    async def fake_resolve() -> dict[str, str]:
        return {"token": "tok"}

    tool.integration.resolve = fake_resolve  # type: ignore[method-assign]

    mock_response = _json_response({"value": [{"id": "perm-1", "roles": ["read"]}]})
    mock_client = _mock_httpx_client(get_responses=[mock_response])

    with patch("httpx.AsyncClient", return_value=mock_client):
        out = await tool(site_id="site-1", item_id="item-1").collect()

    assert len(out.output["value"]) == 1


@pytest.mark.asyncio
async def test_invite_handler() -> None:
    tool = SharePointInvite(integration=Integration("sharepoint", "int-1"))

    async def fake_resolve() -> dict[str, str]:
        return {"token": "tok"}

    tool.integration.resolve = fake_resolve  # type: ignore[method-assign]

    mock_response = _json_response({"value": [{"email": "user@example.com"}]})
    mock_client = _mock_httpx_client()
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_client):
        out = await tool(
            site_id="site-1",
            item_id="item-1",
            emails=["user@example.com"],
            roles=["read"],
        ).collect()

    post_json = mock_client.post.call_args.kwargs["json"]
    assert post_json["recipients"] == [{"email": "user@example.com"}]
    assert post_json["roles"] == ["read"]


@pytest.mark.asyncio
async def test_create_share_link_handler() -> None:
    tool = SharePointCreateShareLink(integration=Integration("sharepoint", "int-1"))

    async def fake_resolve() -> dict[str, str]:
        return {"token": "tok"}

    tool.integration.resolve = fake_resolve  # type: ignore[method-assign]

    mock_response = _json_response({"link": {"webUrl": "https://contoso.sharepoint.com/:u:/s/link"}})
    mock_client = _mock_httpx_client()
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_client):
        out = await tool(
            site_id="site-1",
            item_id="item-1",
            link_type="view",
            scope="organization",
        ).collect()

    assert "webUrl" in out.output["link"]
    assert mock_client.post.call_args.kwargs["json"] == {"type": "view", "scope": "organization"}


@pytest.mark.asyncio
async def test_delete_permission_handler() -> None:
    tool = SharePointDeletePermission(integration=Integration("sharepoint", "int-1"))

    async def fake_resolve() -> dict[str, str]:
        return {"token": "tok"}

    tool.integration.resolve = fake_resolve  # type: ignore[method-assign]

    mock_response = MagicMock()
    mock_response.status_code = 204
    mock_response.raise_for_status = MagicMock()
    mock_client = _mock_httpx_client()
    mock_client.delete = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_client):
        out = await tool(site_id="site-1", item_id="item-1", permission_id="perm-9").collect()

    assert out.output == {"status": "deleted", "permission_id": "perm-9"}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_sharepoint_list_files() -> None:
    """Smoke test against real Microsoft Graph (optional)."""
    _skip_if_sharepoint_live_env_not_set()
    from timbal.tools import SharePointListFiles

    site_id = os.environ["SHAREPOINT_SITE_ID"]
    token = os.environ["SHAREPOINT_ACCESS_TOKEN"]
    tool = SharePointListFiles(integration=Integration("sharepoint", "live"))

    async def fake_resolve() -> dict[str, str]:
        return {"token": token}

    tool.integration.resolve = fake_resolve  # type: ignore[method-assign]
    out = await tool(site_id=site_id).collect()
    assert out.status.code == "success", out.error
    assert "value" in out.output
