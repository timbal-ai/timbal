"""Tests for Google Docs tools.

Unit tests use mocked httpx. Live integration tests require platform config
(``TIMBAL_API_KEY``, ``TIMBAL_ORG_ID``) and a connected Google Docs integration.

Run integration tests explicitly::

    GOOGLE_DOCS_INTEGRATION_ID=173 uv run python -m pytest python/tests/tools/test_google_docs.py -m integration -v
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr
from timbal.core.tool import Tool
from timbal.platform.integrations import Integration
from timbal.state import RunContext, set_run_context
from timbal.state.config_loader import resolve_platform_config
from timbal.tools.google_docs import (
    GoogleDocsAppendImage,
    GoogleDocsAppendText,
    GoogleDocsCreate,
    GoogleDocsCreateFromTemplate,
    GoogleDocsFindDocument,
    GoogleDocsGetDocument,
    GoogleDocsGetTabContent,
    GoogleDocsInsertPageBreak,
    GoogleDocsInsertTable,
    GoogleDocsInsertText,
    GoogleDocsReplaceImage,
    GoogleDocsReplaceText,
    _document_end_index,
    _escape_drive_query,
)

PUBLIC_IMAGE = "https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png"


def _skip_if_google_docs_integration_not_configured(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setattr("timbal.state.config_loader._cached_default_config", None)
    monkeypatch.setattr("timbal.state.config_loader._default_config_resolved", False)
    org = os.environ.get("TIMBAL_ORG_ID")
    token = os.environ.get("TIMBAL_API_KEY") or os.environ.get("TIMBAL_API_TOKEN")
    integration_id = os.environ.get("GOOGLE_DOCS_INTEGRATION_ID", "173")
    if not (org and token):
        pytest.skip(
            "Google Docs integration: set TIMBAL_ORG_ID and TIMBAL_API_KEY (or TIMBAL_API_TOKEN)."
        )
    cfg = resolve_platform_config()
    if cfg is None or not getattr(cfg, "host", None):
        pytest.skip("Google Docs integration: platform host not resolved.")
    set_run_context(RunContext(platform_config=cfg, tracing_provider=None))
    return integration_id


def _integration_tool(tool_cls: type[Tool], integration_id: str) -> Tool:
    return tool_cls(integration=Integration("google_docs", integration_id))


async def _invoke(tool: Tool, **kwargs):
    result = await tool(**kwargs).collect()
    if result.error:
        message = result.error.get("message", result.error) if isinstance(result.error, dict) else result.error
        raise AssertionError(f"{tool.name} failed: {message}")
    return result.output


def _mock_httpx_context(mock_client: MagicMock) -> MagicMock:
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_client)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def test_escape_drive_query():
    assert _escape_drive_query("Tom & Jerry") == "Tom & Jerry"
    assert _escape_drive_query("O'Brien") == "O\\'Brien"


def test_document_end_index():
    doc = {"body": {"content": [{"endIndex": 1}, {"startIndex": 1, "endIndex": 42}]}}
    assert _document_end_index(doc) == 41


@pytest.mark.asyncio
async def test_find_document_uses_fulltext_query():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "files": [{"id": "doc1", "name": "Proposal", "webViewLink": "https://docs.google.com/document/d/doc1"}]
    }

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = GoogleDocsFindDocument(token=SecretStr("token"))
        out = await tool.handler(search_name="Proposal", folder_id=None, max_results=5)

    assert out["total"] == 1
    assert out["documents"][0]["documentId"] == "doc1"
    params = mock_client.get.await_args.kwargs["params"]
    assert "fullText contains 'Proposal'" in params["q"]
    assert "application/vnd.google-apps.document" in params["q"]


@pytest.mark.asyncio
async def test_create_document_uses_drive_and_batch_update():
    create_response = MagicMock()
    create_response.raise_for_status = MagicMock()
    create_response.json.return_value = {
        "id": "new-doc",
        "name": "My Doc",
        "webViewLink": "https://docs.google.com/document/d/new-doc",
    }

    batch_response = MagicMock()
    batch_response.raise_for_status = MagicMock()
    batch_response.json.return_value = {"documentId": "new-doc"}

    mock_client = MagicMock()
    mock_client.post = AsyncMock(side_effect=[create_response, batch_response])

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = GoogleDocsCreate(token=SecretStr("token"))
        out = await tool.handler(title="My Doc", content="Hello", folder_id="folder123")

    assert out["documentId"] == "new-doc"
    assert mock_client.post.await_count == 2
    drive_call = mock_client.post.await_args_list[0]
    assert drive_call.args[0].endswith("/drive/v3/files")
    assert drive_call.kwargs["json"]["parents"] == ["folder123"]
    batch_call = mock_client.post.await_args_list[1]
    assert batch_call.args[0].endswith("/documents/new-doc:batchUpdate")
    assert batch_call.kwargs["json"]["requests"][0]["insertText"]["text"] == "Hello"


@pytest.mark.asyncio
async def test_replace_text_posts_batch_update():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"replies": [{"replaceAllText": {"occurrencesChanged": 3}}]}

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = GoogleDocsReplaceText(token=SecretStr("token"))
        out = await tool.handler(
            document_id="doc123",
            find_text="{{name}}",
            replace_text="Alice",
            match_case=True,
        )

    assert out["occurrencesChanged"] == 3
    body = mock_client.post.await_args.kwargs["json"]
    assert body["requests"][0]["replaceAllText"]["containsText"]["text"] == "{{name}}"


@pytest.mark.asyncio
async def test_get_document_passes_include_tabs_content():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"documentId": "doc123", "title": "Test"}

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = GoogleDocsGetDocument(token=SecretStr("token"))
        out = await tool.handler(document_id="doc123", include_tabs_content=True)

    assert out["documentId"] == "doc123"
    params = mock_client.get.await_args.kwargs["params"]
    assert params["includeTabsContent"] == "true"


@pytest.mark.asyncio
async def test_append_text_inserts_at_end_index():
    get_response = MagicMock()
    get_response.raise_for_status = MagicMock()
    get_response.json.return_value = {"body": {"content": [{"endIndex": 1}, {"startIndex": 1, "endIndex": 10}]}}

    batch_response = MagicMock()
    batch_response.raise_for_status = MagicMock()
    batch_response.json.return_value = {"documentId": "doc123"}

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=get_response)
    mock_client.post = AsyncMock(return_value=batch_response)

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = GoogleDocsAppendText(token=SecretStr("token"))
        out = await tool.handler(document_id="doc123", text=" appended", tab_id=None)

    assert out["appendedLength"] == len(" appended")
    batch_body = mock_client.post.await_args.kwargs["json"]
    assert batch_body["requests"][0]["insertText"]["location"]["index"] == 9


@pytest.mark.integration
@pytest.mark.asyncio
async def test_google_docs_live_all_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end smoke test for all 12 Google Docs tools via platform OAuth."""
    integration_id = _skip_if_google_docs_integration_not_configured(monkeypatch)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    title = f"Timbal Docs Integration {stamp}"

    created = await _invoke(
        _integration_tool(GoogleDocsCreate, integration_id),
        title=title,
        content="Initial content.\n",
    )
    doc_id = created["documentId"]

    found = await _invoke(
        _integration_tool(GoogleDocsFindDocument, integration_id),
        search_name="Timbal",
        max_results=5,
    )
    assert any(d["documentId"] == doc_id for d in found["documents"])

    doc = await _invoke(
        _integration_tool(GoogleDocsGetDocument, integration_id),
        document_id=doc_id,
        include_tabs_content=True,
    )
    assert doc.get("title") == title

    await _invoke(
        _integration_tool(GoogleDocsInsertText, integration_id),
        document_id=doc_id,
        text="Inserted line.\n",
        index=1,
    )
    await _invoke(
        _integration_tool(GoogleDocsAppendText, integration_id),
        document_id=doc_id,
        text="Appended line.\n",
    )
    replaced = await _invoke(
        _integration_tool(GoogleDocsReplaceText, integration_id),
        document_id=doc_id,
        find_text="Initial",
        replace_text="Updated",
        match_case=True,
    )
    assert replaced["occurrencesChanged"] >= 1

    await _invoke(
        _integration_tool(GoogleDocsInsertTable, integration_id),
        document_id=doc_id,
        rows=2,
        columns=2,
    )
    await _invoke(_integration_tool(GoogleDocsInsertPageBreak, integration_id), document_id=doc_id)

    appended_image = await _invoke(
        _integration_tool(GoogleDocsAppendImage, integration_id),
        document_id=doc_id,
        image_uri=PUBLIC_IMAGE,
        width_pt=120,
        height_pt=40,
    )
    image_object_id = appended_image["objectId"]
    assert image_object_id

    await _invoke(
        _integration_tool(GoogleDocsReplaceImage, integration_id),
        document_id=doc_id,
        image_object_id=image_object_id,
        image_uri=PUBLIC_IMAGE,
    )

    template = await _invoke(
        _integration_tool(GoogleDocsCreate, integration_id),
        title=f"Timbal Template {stamp}",
        content="Hello {{name}}",
    )
    from_template = await _invoke(
        _integration_tool(GoogleDocsCreateFromTemplate, integration_id),
        template_id=template["documentId"],
        title=f"Timbal From Template {stamp}",
        replacements={"{{name}}": "Bruno"},
    )
    assert from_template["documentId"]

    doc_with_tabs = await _invoke(
        _integration_tool(GoogleDocsGetDocument, integration_id),
        document_id=doc_id,
        include_tabs_content=True,
    )
    tabs = doc_with_tabs.get("tabs") or []
    if tabs:
        tab_id = tabs[0]["tabProperties"]["tabId"]
        tab_content = await _invoke(
            _integration_tool(GoogleDocsGetTabContent, integration_id),
            document_id=doc_id,
            tab_id=tab_id,
        )
        assert tab_content["tabId"] == tab_id
