from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from timbal.tools import CalaQuery, CalaSearch


def _mock_httpx_context(mock_client):
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_client)
    ctx.__aexit__ = AsyncMock(return_value=None)
    return ctx


@pytest.mark.asyncio
async def test_cala_search_posts_full_search_request():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"content": "answer", "entities": []}

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = CalaSearch(api_key=SecretStr("cala-key"))
        out = await tool.handler(
            input="What are the biggest AI startups in Europe?",
            explainability=False,
            return_entities=False,
        )

    assert out == {"content": "answer", "entities": []}
    url = mock_client.post.await_args[0][0]
    assert url == "https://api.cala.ai/v1/knowledge/search"
    assert mock_client.post.await_args.kwargs["headers"]["x-api-key"] == "cala-key"
    assert mock_client.post.await_args.kwargs["json"] == {
        "input": "What are the biggest AI startups in Europe?",
        "explainability": False,
        "return_entities": False,
    }


@pytest.mark.asyncio
async def test_cala_query_posts_structured_query_request():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"results": [], "entities": []}

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = CalaQuery(api_key=SecretStr("cala-key"), base_url="https://custom.api/v1")
        out = await tool.handler(
            input="startups.location=Spain.funding>10M",
            return_entities=True,
        )

    assert out == {"results": [], "entities": []}
    url = mock_client.post.await_args[0][0]
    assert url == "https://custom.api/v1/knowledge/query"
    assert mock_client.post.await_args.kwargs["json"] == {
        "input": "startups.location=Spain.funding>10M",
        "return_entities": True,
    }
