"""Tests for Higgsfield AI platform tools.

Unit tests mock httpx. Integration tests require ``HF_KEY`` or ``HF_API_KEY`` + ``HF_API_SECRET``.

Run integration tests explicitly::

    uv run python -m pytest python/tests/tools/test_higgsfield.py -m integration -v
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr
from timbal.core.tool import Tool
from timbal.tools.higgsfield import (
    DEFAULT_TEXT_TO_IMAGE_MODEL,
    HiggsfieldCancelRequest,
    HiggsfieldCheckStatus,
    HiggsfieldGetResult,
    HiggsfieldSubmit,
    HiggsfieldTextToImage,
    HiggsfieldUploadFile,
    _resolve_higgsfield_credential_key,
)

_TEST_KEY = "test-key-id:test-secret"


def _skip_if_higgsfield_not_configured() -> None:
    if os.getenv("HF_KEY") or (os.getenv("HF_API_KEY") and os.getenv("HF_API_SECRET")):
        return
    pytest.skip("Higgsfield integration: set HF_KEY or HF_API_KEY + HF_API_SECRET")


async def _invoke(tool: Tool, **kwargs: Any):
    result = await tool(**kwargs).collect()
    if result.error:
        message = result.error.get("message", result.error) if isinstance(result.error, dict) else result.error
        raise AssertionError(f"{tool.name} failed: {message}")
    return result


def _mock_async_client(*, get_side_effect=None, post_side_effect=None, put_side_effect=None) -> tuple[MagicMock, MagicMock]:
    client = MagicMock()

    if get_side_effect is not None:
        client.get = AsyncMock(side_effect=get_side_effect)
    else:
        client.get = AsyncMock()

    if post_side_effect is not None:
        client.post = AsyncMock(side_effect=post_side_effect)
    else:
        client.post = AsyncMock()

    if put_side_effect is not None:
        client.put = AsyncMock(side_effect=put_side_effect)
    else:
        client.put = AsyncMock()

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=client)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm, client


@pytest.mark.asyncio
async def test_resolve_credentials_from_hf_key_field():
    tool = HiggsfieldSubmit(hf_key=SecretStr(_TEST_KEY))
    assert await _resolve_higgsfield_credential_key(tool) == _TEST_KEY


@pytest.mark.asyncio
async def test_resolve_credentials_from_api_key_and_secret_fields():
    tool = HiggsfieldSubmit(api_key=SecretStr("kid"), api_secret=SecretStr("sec"))
    assert await _resolve_higgsfield_credential_key(tool) == "kid:sec"


@pytest.mark.asyncio
async def test_submit_posts_model_and_arguments():
    submit_response = MagicMock()
    submit_response.raise_for_status = MagicMock()
    submit_response.json.return_value = {
        "request_id": "req-123",
        "status_url": "https://platform.higgsfield.ai/requests/req-123/status",
        "cancel_url": "https://platform.higgsfield.ai/requests/req-123/cancel",
    }

    cm, client = _mock_async_client(post_side_effect=[submit_response])

    with patch("httpx.AsyncClient", return_value=cm):
        tool = HiggsfieldSubmit(hf_key=SecretStr(_TEST_KEY))
        out = await _invoke(
            tool,
            model="bytedance/seedream/v4/text-to-image",
            arguments={"prompt": "sunset"},
        )

    assert out.output["request_id"] == "req-123"
    assert out.usage.get("higgsfield_submit:requests") == 1
    call_args = client.post.call_args
    assert call_args.args[0] == "bytedance/seedream/v4/text-to-image"
    assert call_args.kwargs["json"] == {"prompt": "sunset"}
    headers = call_args.kwargs["headers"]
    assert headers["Authorization"] == f"Key {_TEST_KEY}"


@pytest.mark.asyncio
async def test_check_status_returns_normalized_payload():
    status_response = MagicMock()
    status_response.raise_for_status = MagicMock()
    status_response.json.return_value = {"status": "in_progress"}

    cm, _client = _mock_async_client(get_side_effect=[status_response])

    with patch("httpx.AsyncClient", return_value=cm):
        tool = HiggsfieldCheckStatus(hf_key=SecretStr(_TEST_KEY))
        out = await _invoke(tool, request_id="req-abc")

    assert out.output["request_id"] == "req-abc"
    assert out.output["status"] == "in_progress"


@pytest.mark.asyncio
async def test_get_result_polls_until_completed():
    running = MagicMock()
    running.raise_for_status = MagicMock()
    running.json.return_value = {"status": "in_progress"}

    completed = MagicMock()
    completed.raise_for_status = MagicMock()
    completed.json.return_value = {
        "status": "completed",
        "images": [{"url": "https://cdn.example/img.png"}],
    }

    cm, client = _mock_async_client(get_side_effect=[running, completed])

    with patch("httpx.AsyncClient", return_value=cm), patch("asyncio.sleep", new=AsyncMock()):
        tool = HiggsfieldGetResult(hf_key=SecretStr(_TEST_KEY))
        out = await _invoke(tool, request_id="req-poll", poll_interval=0.01, timeout=5.0)

    assert out.output["status"] == "completed"
    assert out.output["images"][0]["url"] == "https://cdn.example/img.png"
    assert client.get.await_count == 2


@pytest.mark.asyncio
async def test_text_to_image_subscribe_waits_by_default():
    submit_response = MagicMock()
    submit_response.raise_for_status = MagicMock()
    submit_response.json.return_value = {"request_id": "req-img"}

    completed = MagicMock()
    completed.raise_for_status = MagicMock()
    completed.json.return_value = {"status": "completed", "images": [{"url": "https://cdn.example/out.png"}]}

    cm, client = _mock_async_client(post_side_effect=[submit_response], get_side_effect=[completed])

    with patch("httpx.AsyncClient", return_value=cm):
        tool = HiggsfieldTextToImage(hf_key=SecretStr(_TEST_KEY))
        out = await _invoke(tool, prompt="A red balloon")

    assert out.output["status"] == "completed"
    submit_call = client.post.call_args
    assert submit_call.args[0] == DEFAULT_TEXT_TO_IMAGE_MODEL
    assert submit_call.kwargs["json"]["prompt"] == "A red balloon"
    assert out.usage.get("higgsfield_text_to_image:requests") == 1


@pytest.mark.asyncio
async def test_text_to_image_wait_false_returns_submit_payload():
    submit_response = MagicMock()
    submit_response.raise_for_status = MagicMock()
    submit_response.json.return_value = {"request_id": "req-async", "status_url": "https://x/status"}

    cm, client = _mock_async_client(post_side_effect=[submit_response])

    with patch("httpx.AsyncClient", return_value=cm):
        tool = HiggsfieldTextToImage(hf_key=SecretStr(_TEST_KEY))
        out = await _invoke(tool, prompt="wave", wait=False)

    assert out.output["submitted"] is True
    assert out.output["request_id"] == "req-async"
    client.get.assert_not_called()


@pytest.mark.asyncio
async def test_upload_file_returns_public_url(tmp_path):
    image = tmp_path / "sample.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    meta_response = MagicMock()
    meta_response.raise_for_status = MagicMock()
    meta_response.json.return_value = {
        "public_url": "https://cdn.example/public.png",
        "upload_url": "https://upload.example/put",
    }
    put_response = MagicMock()
    put_response.raise_for_status = MagicMock()

    cm, client = _mock_async_client(post_side_effect=[meta_response], put_side_effect=[put_response])

    with patch("httpx.AsyncClient", return_value=cm):
        tool = HiggsfieldUploadFile(hf_key=SecretStr(_TEST_KEY))
        out = await _invoke(tool, file_path=str(image))

    assert out.output["url"] == "https://cdn.example/public.png"
    assert out.output["content_type"] == "image/png"
    client.put.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancel_request_posts_cancel_endpoint():
    cancel_response = MagicMock()
    cancel_response.raise_for_status = MagicMock()

    cm, client = _mock_async_client(post_side_effect=[cancel_response])

    with patch("httpx.AsyncClient", return_value=cm):
        tool = HiggsfieldCancelRequest(hf_key=SecretStr(_TEST_KEY))
        out = await _invoke(tool, request_id="req-cancel")

    assert out.output["cancelled"] is True
    assert client.post.await_args.args[0] == "/requests/req-cancel/cancel"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_text_to_image_live():
    _skip_if_higgsfield_not_configured()
    tool = HiggsfieldTextToImage()
    out = await _invoke(
        tool,
        prompt="A single red apple on white background, product photo",
        aspect_ratio="1:1",
        wait=True,
        timeout=600.0,
    )
    assert out.status.code == "success"
    assert str(out.output.get("status", "")).lower() in {"completed", "failed", "nsfw", "canceled", "cancelled"}
