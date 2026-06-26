"""Tests for Gemini image tools.

Unit tests mock httpx. Live integration tests require GEMINI_API_KEY.

Run integration tests::

    uv run python -m pytest python/tests/tools/test_gemini_images.py -m integration -v
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr
from timbal.tools.gemini_images import (
    GeminiImagesAnalyzeImage,
    GeminiImagesEditImage,
    GeminiImagesGenerateImage,
    GeminiImagesGenerateImageWithReference,
    GeminiImagesImageToBase64,
    _parse_gemini_response,
)

_TINY_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


def _mock_httpx_context(mock_client: MagicMock) -> MagicMock:
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_client)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def _image_response(b64: str = "aW1hZ2U=", mime: str = "image/png", text: str | None = None) -> dict[str, Any]:
    parts: list[dict[str, Any]] = []
    if text:
        parts.append({"text": text})
    parts.append({"inlineData": {"data": b64, "mimeType": mime}})
    return {"candidates": [{"content": {"parts": parts}}]}


def test_parse_gemini_response_extracts_image_and_text():
    parsed = _parse_gemini_response(_image_response(text="done"))
    assert parsed["text"] == "done"
    assert parsed["image_base64"] == "aW1hZ2U="
    assert parsed["mime_type"] == "image/png"


@pytest.mark.asyncio
async def test_generate_image_uses_goog_api_key_header():
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = _image_response()

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=response)

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = GeminiImagesGenerateImage(api_key=SecretStr("test-gemini-key"))
        out = await tool(prompt="A red circle").collect()

    assert out.status.code == "success"
    assert out.output["image_base64"] == "aW1hZ2U="
    headers = mock_client.post.call_args.kwargs["headers"]
    assert headers["x-goog-api-key"] == "test-gemini-key"
    assert "Authorization" not in headers


@pytest.mark.asyncio
async def test_analyze_image_returns_text():
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": "Mostly red."}]}}],
    }

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=response)

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = GeminiImagesAnalyzeImage(api_key=SecretStr("test-gemini-key"))
        out = await tool(
            prompt="What color?",
            image_base64=_TINY_PNG_B64,
            image_mime_type="image/png",
        ).collect()

    assert out.status.code == "success"
    assert out.output == {"text": "Mostly red."}


@pytest.mark.asyncio
async def test_edit_image_posts_inline_data():
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = _image_response(text="edited")

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=response)

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = GeminiImagesEditImage(api_key=SecretStr("test-gemini-key"))
        out = await tool(
            prompt="Add a border",
            image_base64=_TINY_PNG_B64,
            image_mime_type="image/png",
        ).collect()

    assert out.status.code == "success"
    payload = mock_client.post.call_args.kwargs["json"]
    parts = payload["contents"][0]["parts"]
    assert parts[0]["text"] == "Add a border"
    assert parts[1]["inline_data"]["data"] == _TINY_PNG_B64


@pytest.mark.asyncio
async def test_generate_with_reference_includes_all_images():
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = _image_response()

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=response)

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = GeminiImagesGenerateImageWithReference(api_key=SecretStr("test-gemini-key"))
        out = await tool(
            prompt="Make it blue",
            reference_images_base64=[_TINY_PNG_B64, _TINY_PNG_B64],
            reference_mime_types=["image/png", "image/jpeg"],
        ).collect()

    assert out.status.code == "success"
    parts = mock_client.post.call_args.kwargs["json"]["contents"][0]["parts"]
    assert len(parts) == 3
    assert parts[1]["inline_data"]["mime_type"] == "image/png"
    assert parts[2]["inline_data"]["mime_type"] == "image/jpeg"


@pytest.mark.asyncio
async def test_image_url_to_base64_fetches_url():
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.content = b"\x89PNG\r\n"
    response.headers = {"content-type": "image/png; charset=utf-8"}

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=response)

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = GeminiImagesImageToBase64()
        out = await tool(url="https://example.com/logo.png").collect()

    assert out.status.code == "success"
    assert out.output["mime_type"] == "image/png"
    assert out.output["image_base64"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_gemini_generate_image_live():
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("Set GEMINI_API_KEY for Gemini image integration test")
    tool = GeminiImagesGenerateImage()
    prompt = "A minimalist red circle on white background, flat vector style."
    out = None
    for _ in range(2):
        out = await tool(prompt=prompt).collect()
        if out.status.code == "success" and out.output.get("image_base64"):
            break
    assert out is not None
    assert out.status.code == "success", out.error
    assert out.output.get("image_base64"), f"expected image output, got: {out.output}"
    assert out.output.get("mime_type")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_gemini_analyze_image_live():
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("Set GEMINI_API_KEY for Gemini image integration test")
    out = await GeminiImagesAnalyzeImage()(
        prompt="Describe this image in one short sentence.",
        image_base64=_TINY_PNG_B64,
        image_mime_type="image/png",
    ).collect()
    assert out.status.code == "success", out.error
    assert out.output["text"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_gemini_edit_image_live():
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("Set GEMINI_API_KEY for Gemini image integration test")
    gen = await GeminiImagesGenerateImage()(
        prompt="A red circle on white background, minimalist.",
    ).collect()
    assert gen.status.code == "success", gen.error
    image_b64 = gen.output["image_base64"]

    out = await GeminiImagesEditImage()(
        prompt="Add a thin black border.",
        image_base64=image_b64,
        image_mime_type=gen.output.get("mime_type") or "image/jpeg",
    ).collect()
    assert out.status.code == "success", out.error
    assert out.output["image_base64"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_gemini_generate_with_reference_live():
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("Set GEMINI_API_KEY for Gemini image integration test")
    gen = await GeminiImagesGenerateImage()(
        prompt="A red circle on white background, minimalist.",
    ).collect()
    assert gen.status.code == "success", gen.error

    out = await GeminiImagesGenerateImageWithReference()(
        prompt="Create a similar icon but make the circle blue.",
        reference_images_base64=[gen.output["image_base64"]],
        reference_mime_types=[gen.output.get("mime_type") or "image/jpeg"],
    ).collect()
    assert out.status.code == "success", out.error
    assert out.output["image_base64"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_gemini_image_url_to_base64_live():
    out = await GeminiImagesImageToBase64()(
        url="https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_92x30dp.png",
    ).collect()
    assert out.status.code == "success", out.error
    assert out.output["image_base64"]
    assert out.output["mime_type"].startswith("image/")
