"""Tests for Krea.ai tools.

Unit tests use mocked httpx. Live integration tests require ``KREA_API_KEY``.

Run integration tests explicitly::

    uv run python -m pytest python/tests/tools/test_krea.py -m integration -v
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr
from timbal.tools.krea import (
    KreaGenerateImage,
    KreaGenerateVideo,
    KreaGetJob,
    KreaListJobs,
    _build_generate_path,
    _extract_job_urls,
)


def _mock_httpx_context(mock_client: MagicMock) -> MagicMock:
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_client)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def _completed_job(job_id: str = "job-abc", url: str = "https://cdn.example.com/out.mp4") -> dict[str, Any]:
    return {
        "job_id": job_id,
        "status": "completed",
        "created_at": "2026-01-01T00:00:00Z",
        "completed_at": "2026-01-01T00:01:00Z",
        "result": {"urls": [url]},
    }


def test_build_generate_path_video():
    assert _build_generate_path("video", "google/veo-3.1-fast") == "/generate/video/google/veo-3.1-fast"
    assert _build_generate_path("video", "video/kling/kling-2.5") == "/generate/video/kling/kling-2.5"


def test_build_generate_path_image():
    assert _build_generate_path("image", "bfl/flux-1-dev") == "/generate/image/bfl/flux-1-dev"


def test_extract_job_urls():
    assert _extract_job_urls({"result": {"urls": ["https://a", "https://b"]}}) == ["https://a", "https://b"]
    assert _extract_job_urls({"result": None}) == []


@pytest.mark.asyncio
async def test_krea_list_jobs():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"items": [], "next_cursor": None}

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = KreaListJobs(api_key=SecretStr("krea-test"))
        out = await tool.handler(limit=10, cursor=None, types=None, status=None)

    assert out == {"items": [], "next_cursor": None}
    url = mock_client.get.await_args[0][0]
    assert url == "https://api.krea.ai/jobs"
    headers = mock_client.get.await_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer krea-test"


@pytest.mark.asyncio
async def test_krea_get_job():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = _completed_job()

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = KreaGetJob(api_key=SecretStr("krea-test"))
        out = await tool.handler(job_id="job-abc")

    assert out["status"] == "completed"
    assert mock_client.get.await_args[0][0] == "https://api.krea.ai/jobs/job-abc"


@pytest.mark.asyncio
async def test_krea_generate_video_submit_and_poll():
    submit_response = MagicMock()
    submit_response.raise_for_status = MagicMock()
    submit_response.json.return_value = {
        "job_id": "job-video-1",
        "status": "queued",
        "created_at": "2026-01-01T00:00:00Z",
        "completed_at": None,
        "result": None,
    }

    poll_pending = MagicMock()
    poll_pending.raise_for_status = MagicMock()
    poll_pending.json.return_value = {"job_id": "job-video-1", "status": "processing", "created_at": "2026-01-01T00:00:00Z"}

    poll_done = MagicMock()
    poll_done.raise_for_status = MagicMock()
    poll_done.json.return_value = _completed_job("job-video-1", "https://cdn.example.com/video.mp4")

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=submit_response)
    mock_client.get = AsyncMock(side_effect=[poll_pending, poll_done])

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = KreaGenerateVideo(api_key=SecretStr("krea-test"))
        out = await tool.handler(
            prompt="A boat on water",
            model="google/veo-3.1-fast",
            aspect_ratio="16:9",
            duration=4,
            resolution="720p",
            generate_audio=False,
            start_image=None,
            end_image=None,
            reference_images=None,
            provider_params=None,
            webhook_url=None,
            poll_interval=0,
            timeout=30,
        )

    assert out["success"] is True
    assert out["video_url"] == "https://cdn.example.com/video.mp4"
    assert out["job_id"] == "job-video-1"

    post_url = mock_client.post.await_args[0][0]
    assert post_url == "https://api.krea.ai/generate/video/google/veo-3.1-fast"
    body = mock_client.post.await_args.kwargs["json"]
    assert body["prompt"] == "A boat on water"
    assert body["duration"] == 4
    assert body["aspect_ratio"] == "16:9"


@pytest.mark.asyncio
async def test_krea_generate_video_empty_prompt():
    tool = KreaGenerateVideo(api_key=SecretStr("krea-test"))
    with pytest.raises(ValueError, match="prompt is required"):
        await tool.handler(
            prompt="   ",
            model="google/veo-3.1-fast",
            aspect_ratio="16:9",
            duration=4,
            resolution="720p",
            generate_audio=False,
            start_image=None,
            end_image=None,
            reference_images=None,
            provider_params=None,
            webhook_url=None,
            poll_interval=0,
            timeout=30,
        )


@pytest.mark.asyncio
async def test_krea_generate_image_submit_and_poll():
    submit_response = MagicMock()
    submit_response.raise_for_status = MagicMock()
    submit_response.json.return_value = {
        "job_id": "job-img-1",
        "status": "queued",
        "created_at": "2026-01-01T00:00:00Z",
        "completed_at": None,
        "result": None,
    }

    poll_done = MagicMock()
    poll_done.raise_for_status = MagicMock()
    poll_done.json.return_value = _completed_job("job-img-1", "https://cdn.example.com/image.png")

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=submit_response)
    mock_client.get = AsyncMock(return_value=poll_done)

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = KreaGenerateImage(api_key=SecretStr("krea-test"))
        out = await tool.handler(
            prompt="A red balloon",
            model="bfl/flux-1-dev",
            width=1024,
            height=1024,
            seed=42,
            image_url=None,
            provider_params={"steps": 20},
            webhook_url=None,
            poll_interval=0,
            timeout=30,
        )

    assert out["success"] is True
    assert out["image_url"] == "https://cdn.example.com/image.png"
    post_url = mock_client.post.await_args[0][0]
    assert post_url == "https://api.krea.ai/generate/image/bfl/flux-1-dev"
    body = mock_client.post.await_args.kwargs["json"]
    assert body["steps"] == 20
    assert body["seed"] == 42


@pytest.mark.integration
@pytest.mark.asyncio
async def test_krea_list_jobs_live():
    if not os.getenv("KREA_API_KEY"):
        pytest.skip("Set KREA_API_KEY for Krea integration test")
    tool = KreaListJobs()
    out = await tool.handler(limit=5, cursor=None, types=None, status=None)
    assert "items" in out


@pytest.mark.integration
@pytest.mark.asyncio
async def test_krea_generate_video_live_or_balance_error():
    if not os.getenv("KREA_API_KEY"):
        pytest.skip("Set KREA_API_KEY for Krea integration test")
    tool = KreaGenerateVideo()
    try:
        out = await tool.handler(
            prompt="A paper boat on calm water, macro cinematic.",
            model="google/veo-3.1-fast",
            aspect_ratio="16:9",
            duration=4,
            resolution="720p",
            generate_audio=False,
            start_image=None,
            end_image=None,
            reference_images=None,
            provider_params=None,
            webhook_url=None,
            poll_interval=5,
            timeout=600,
        )
    except (RuntimeError, TimeoutError) as exc:
        # API errors (e.g. insufficient balance) now propagate instead of being swallowed.
        assert str(exc)
    else:
        assert out["success"] is True
        assert out["video_url"]
        assert out["job_id"]
