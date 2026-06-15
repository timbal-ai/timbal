"""Unit tests for Google Docs tools (httpx mocked)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr
from timbal.tools.google_docs import (
    GoogleDocsAppendText,
    GoogleDocsCreate,
    GoogleDocsFindDocument,
    GoogleDocsGetDocument,
    GoogleDocsReplaceText,
    _document_end_index,
    _escape_drive_query,
)


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
