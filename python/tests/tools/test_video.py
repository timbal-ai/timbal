"""Tests for model-agnostic video generation tool.

Unit tests use mocked httpx. Live integration tests require provider API keys.

Run integration tests explicitly::

    uv run pytest python/tests/tools/test_video.py -m integration -v
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr
from timbal.tools.video import (
    _VIDEO_MODELS,
    VideoGenerate,
    VideoListModels,
    _build_bytedance_payload,
    _build_fal_payload,
    _build_google_payload,
    _resolve_model_config,
    _validate_fal_constraints,
)


def _mock_httpx_context(mock_client: MagicMock) -> MagicMock:
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_client)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def _google_done_response(video_uri: str = "https://example.com/veo.mp4") -> dict[str, Any]:
    return {
        "done": True,
        "response": {
            "generateVideoResponse": {
                "generatedSamples": [{"video": {"uri": video_uri}}],
            }
        },
    }


def _bytedance_succeeded_response(
    video_url: str = "https://example.com/seedance.mp4",
    *,
    total_tokens: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"status": "succeeded", "content": {"video_url": video_url}}
    if total_tokens is not None:
        payload["usage"] = {"total_tokens": total_tokens}
    return payload


async def _collect_video(tool: VideoGenerate, **kwargs: Any):
    result = await tool(**kwargs).collect()
    return result


def _fal_completed_status() -> dict[str, Any]:
    return {"status": "COMPLETED"}


def _fal_result_response(video_url: str = "https://example.com/fal.mp4") -> dict[str, Any]:
    return {"video": {"url": video_url}}


@pytest.mark.asyncio
async def test_google_video_generate_submit_and_poll():
    submit_response = MagicMock()
    submit_response.raise_for_status = MagicMock()
    submit_response.json.return_value = {"name": "models/veo-3.1-generate-preview/operations/op1"}

    poll_pending = MagicMock()
    poll_pending.raise_for_status = MagicMock()
    poll_pending.json.return_value = {"done": False}

    poll_done = MagicMock()
    poll_done.raise_for_status = MagicMock()
    poll_done.json.return_value = _google_done_response()

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=submit_response)
    mock_client.get = AsyncMock(side_effect=[poll_pending, poll_done])

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = VideoGenerate(google_api_key=SecretStr("gemini-test"))
        out = await tool.handler(
            prompt="A drone over the ocean at sunset",
            model="google/veo-3.1",
            aspect_ratio="16:9",
            resolution="720p",
            duration=8,
            negative_prompt=None,
            image_url=None,
            generate_audio=True,
            seed=42,
            provider_params=None,
            poll_interval=0,
            timeout=30,
        )

    assert out["success"] is True
    assert out["provider"] == "google"
    assert out["model"] == "google/veo-3.1"
    assert out["video_url"] == "https://example.com/veo.mp4"

    mock_client.post.assert_awaited_once()
    post_url = mock_client.post.await_args[0][0]
    assert post_url.endswith("veo-3.1-generate-preview:predictLongRunning")
    headers = mock_client.post.await_args.kwargs["headers"]
    assert headers["x-goog-api-key"] == "gemini-test"
    body = mock_client.post.await_args.kwargs["json"]
    assert body["instances"][0]["prompt"] == "A drone over the ocean at sunset"
    assert body["parameters"]["aspectRatio"] == "16:9"
    assert body["parameters"]["resolution"] == "720p"
    assert body["parameters"]["durationSeconds"] == 8
    assert body["parameters"]["seed"] == 42


@pytest.mark.asyncio
async def test_bytedance_video_generate_submit_and_poll():
    submit_response = MagicMock()
    submit_response.raise_for_status = MagicMock()
    submit_response.json.return_value = {"id": "task-abc"}

    poll_pending = MagicMock()
    poll_pending.raise_for_status = MagicMock()
    poll_pending.json.return_value = {"status": "running"}

    poll_done = MagicMock()
    poll_done.raise_for_status = MagicMock()
    poll_done.json.return_value = _bytedance_succeeded_response()

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=submit_response)
    mock_client.get = AsyncMock(side_effect=[poll_pending, poll_done])

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = VideoGenerate(bytedance_api_key=SecretStr("ark-test"))
        out = await tool.handler(
            prompt="Product demo in a studio",
            model="bytedance/seedance-2.0",
            aspect_ratio="16:9",
            resolution="720p",
            duration=5,
            negative_prompt=None,
            image_url=None,
            generate_audio=False,
            seed=None,
            provider_params=None,
            poll_interval=0,
            timeout=30,
        )

    assert out["success"] is True
    assert out["provider"] == "bytedance"
    assert out["video_url"] == "https://example.com/seedance.mp4"

    body = mock_client.post.await_args.kwargs["json"]
    assert body["model"] == "dreamina-seedance-2-0-260128"
    assert body["ratio"] == "16:9"
    assert body["resolution"] == "720p"
    assert body["duration"] == 5
    assert body["generate_audio"] is False
    assert body["content"][0]["text"] == "Product demo in a studio"


@pytest.mark.asyncio
async def test_fal_fallback_video_generate():
    submit_response = MagicMock()
    submit_response.raise_for_status = MagicMock()
    submit_response.json.return_value = {
        "request_id": "req-1",
        "status_url": "https://queue.fal.run/fal-ai/kling/requests/req-1/status",
        "response_url": "https://queue.fal.run/fal-ai/kling/requests/req-1",
    }

    status_pending = MagicMock()
    status_pending.raise_for_status = MagicMock()
    status_pending.json.return_value = {"status": "IN_PROGRESS"}

    status_done = MagicMock()
    status_done.raise_for_status = MagicMock()
    status_done.json.return_value = _fal_completed_status()

    result_response = MagicMock()
    result_response.raise_for_status = MagicMock()
    result_response.json.return_value = _fal_result_response()

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=submit_response)
    mock_client.get = AsyncMock(side_effect=[status_pending, status_done, result_response])

    fal_model = "fal-ai/kling-video/v2/master/text-to-video"
    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = VideoGenerate(fal_api_key=SecretStr("fal-test"))
        out = await tool.handler(
            prompt="A cat walking in the rain",
            model=fal_model,
            aspect_ratio="9:16",
            resolution="720p",
            duration=5,
            negative_prompt="blurry",
            image_url=None,
            generate_audio=True,
            seed=7,
            provider_params=None,
            poll_interval=0,
            timeout=30,
        )

    assert out["success"] is True
    assert out["provider"] == "fal"
    assert out["model"] == fal_model
    assert out["video_url"] == "https://example.com/fal.mp4"

    post_url = mock_client.post.await_args[0][0]
    assert "fal-ai/kling-video/v2/master/text-to-video" in post_url
    body = mock_client.post.await_args.kwargs["json"]
    assert body["prompt"] == "A cat walking in the rain"
    assert body["aspect_ratio"] == "9:16"
    assert body["duration"] == "5"
    assert body["negative_prompt"] == "blurry"
    assert body["seed"] == 7


def test_build_google_payload_provider_params_override():
    config = _resolve_model_config("google/veo-3.1")
    payload = _build_google_payload(
        config,
        prompt="test",
        aspect_ratio="16:9",
        resolution="1080p",
        duration=6,
        negative_prompt="noise",
        image_url=None,
        seed=1,
        provider_params={"duration": 4},
    )
    assert payload["parameters"]["durationSeconds"] == 4
    assert payload["parameters"]["resolution"] == "1080p"
    assert payload["parameters"]["negativePrompt"] == "noise"


def test_build_bytedance_payload_with_image_url():
    config = _resolve_model_config("bytedance/seedance-2.0")
    payload = _build_bytedance_payload(
        config,
        prompt="animate this",
        aspect_ratio="9:16",
        resolution="720p",
        duration=5,
        generate_audio=True,
        image_url="https://example.com/frame.png",
        seed=None,
        provider_params=None,
    )
    assert payload["content"][1]["image_url"]["url"] == "https://example.com/frame.png"
    assert payload["ratio"] == "9:16"


def test_build_fal_payload_whitelist_strips_unknown():
    config = _resolve_model_config("fal-ai/unknown-model")
    payload = _build_fal_payload(
        config,
        prompt="hello",
        aspect_ratio="16:9",
        resolution="720p",
        duration=5,
        negative_prompt=None,
        image_url=None,
        generate_audio=True,
        seed=None,
        provider_params={"unknown_key": "drop-me", "aspect_ratio": "1:1"},
    )
    assert "unknown_key" not in payload
    assert payload["aspect_ratio"] == "1:1"


def test_build_fal_veo_payload_duration_enum():
    config = _resolve_model_config("fal-ai/veo3.1/fast")
    payload = _build_fal_payload(
        config,
        prompt="boat",
        aspect_ratio="16:9",
        resolution="720p",
        duration=4,
        negative_prompt=None,
        image_url=None,
        generate_audio=False,
        seed=None,
        provider_params=None,
    )
    assert payload["duration"] == "4s"
    assert payload["resolution"] == "720p"


def test_validate_fal_veo_rejects_invalid_duration():
    config = _resolve_model_config("fal-ai/veo3.1/fast")
    with pytest.raises(ValueError, match="only supports duration"):
        _validate_fal_constraints(config, aspect_ratio="16:9", resolution="720p", duration=5)


def test_validate_fal_veo_rejects_invalid_aspect_ratio():
    config = _resolve_model_config("fal-ai/veo3.1/fast")
    with pytest.raises(ValueError, match="only supports aspect ratios"):
        _validate_fal_constraints(config, aspect_ratio="1:1", resolution="720p", duration=4)


@pytest.mark.asyncio
async def test_fal_invalid_duration_fails_before_http():
    tool = VideoGenerate(fal_api_key=SecretStr("fal-test"))
    out = await tool.handler(
        prompt="test",
        model="fal-ai/veo3.1/fast",
        aspect_ratio="16:9",
        resolution="720p",
        duration=5,
        negative_prompt=None,
        image_url=None,
        generate_audio=False,
        seed=None,
        provider_params=None,
        poll_interval=0,
        timeout=30,
    )
    assert out["success"] is False
    assert out["error_type"] == "ValueError"
    assert "duration" in out["error"].lower()


@pytest.mark.asyncio
async def test_fal_minimax_usage_no_video_seconds():
    submit_response = MagicMock()
    submit_response.raise_for_status = MagicMock()
    submit_response.json.return_value = {
        "request_id": "req-mm",
        "status_url": "https://queue.fal.run/fal-ai/minimax/requests/req-mm/status",
        "response_url": "https://queue.fal.run/fal-ai/minimax/requests/req-mm",
    }

    status_done = MagicMock()
    status_done.raise_for_status = MagicMock()
    status_done.json.return_value = _fal_completed_status()

    result_response = MagicMock()
    result_response.raise_for_status = MagicMock()
    result_response.json.return_value = _fal_result_response()

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=submit_response)
    mock_client.get = AsyncMock(side_effect=[status_done, result_response])

    fal_model = "fal-ai/minimax/video-01-live/text-to-video"
    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = VideoGenerate(fal_api_key=SecretStr("fal-test"))
        output = await _collect_video(
            tool,
            prompt="minimal prompt only",
            model=fal_model,
            aspect_ratio="16:9",
            resolution="720p",
            duration=5,
            negative_prompt=None,
            image_url=None,
            generate_audio=False,
            seed=None,
            provider_params=None,
            poll_interval=0,
            timeout=30,
        )

    assert output.output["success"] is True
    assert f"{fal_model}:generations" in output.usage
    assert not any(k.endswith(":video_seconds_720p") for k in output.usage)
    post_body = mock_client.post.await_args.kwargs["json"]
    assert post_body == {"prompt": "minimal prompt only"}


@pytest.mark.asyncio
async def test_video_generate_returns_error_on_provider_failure():
    submit_response = MagicMock()
    submit_response.raise_for_status = MagicMock()
    submit_response.json.return_value = {"id": "task-fail"}

    poll_failed = MagicMock()
    poll_failed.raise_for_status = MagicMock()
    poll_failed.json.return_value = {"status": "failed", "error": "quota exceeded"}

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=submit_response)
    mock_client.get = AsyncMock(return_value=poll_failed)

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = VideoGenerate(bytedance_api_key=SecretStr("ark-test"))
        out = await tool.handler(
            prompt="fail case",
            model="bytedance/seedance-2.0-fast",
            aspect_ratio="16:9",
            resolution="720p",
            duration=5,
            negative_prompt=None,
            image_url=None,
            generate_audio=True,
            seed=None,
            provider_params=None,
            poll_interval=0,
            timeout=10,
        )

    assert out["success"] is False
    assert out["video_url"] is None
    assert out["error_type"] == "RuntimeError"
    assert "failed" in out["error"].lower()


@pytest.mark.asyncio
async def test_video_generate_empty_prompt():
    tool = VideoGenerate(google_api_key=SecretStr("gemini-test"))
    out = await tool.handler(
        prompt="   ",
        model="google/veo-3.1",
        aspect_ratio="16:9",
        resolution="720p",
        duration=8,
        negative_prompt=None,
        image_url=None,
        generate_audio=True,
        seed=None,
        provider_params=None,
        poll_interval=0,
        timeout=30,
    )
    assert out["success"] is False
    assert "prompt" in out["error"].lower()


@pytest.mark.asyncio
async def test_google_video_usage_via_collect():
    submit_response = MagicMock()
    submit_response.raise_for_status = MagicMock()
    submit_response.json.return_value = {"name": "models/veo-3.1-fast-generate-preview/operations/op1"}

    poll_done = MagicMock()
    poll_done.raise_for_status = MagicMock()
    poll_done.json.return_value = _google_done_response()

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=submit_response)
    mock_client.get = AsyncMock(return_value=poll_done)

    model = "google/veo-3.1-fast"
    duration = 4
    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = VideoGenerate(google_api_key=SecretStr("gemini-test"))
        output = await _collect_video(
            tool,
            prompt="Usage test prompt",
            model=model,
            aspect_ratio="16:9",
            resolution="720p",
            duration=duration,
            negative_prompt=None,
            image_url=None,
            generate_audio=False,
            seed=None,
            provider_params=None,
            poll_interval=0,
            timeout=30,
        )

    assert output.output["success"] is True
    assert output.usage.get(f"{model}:video_seconds_720p") == duration
    assert output.usage.get(f"{model}:generations") == 1
    assert output.usage.get(f"{model}:requests") == 1
    assert "video_generate:requests" not in output.usage


@pytest.mark.asyncio
async def test_bytedance_video_usage_via_collect_with_tokens():
    submit_response = MagicMock()
    submit_response.raise_for_status = MagicMock()
    submit_response.json.return_value = {"id": "task-tokens"}

    poll_done = MagicMock()
    poll_done.raise_for_status = MagicMock()
    poll_done.json.return_value = _bytedance_succeeded_response(total_tokens=412_880)

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=submit_response)
    mock_client.get = AsyncMock(return_value=poll_done)

    model = "bytedance/seedance-2.0"
    duration = 5
    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = VideoGenerate(bytedance_api_key=SecretStr("ark-test"))
        output = await _collect_video(
            tool,
            prompt="Token usage test",
            model=model,
            aspect_ratio="16:9",
            resolution="720p",
            duration=duration,
            negative_prompt=None,
            image_url=None,
            generate_audio=False,
            seed=None,
            provider_params=None,
            poll_interval=0,
            timeout=30,
        )

    assert output.output["success"] is True
    assert output.usage.get(f"{model}:video_seconds_720p") == duration
    assert output.usage.get(f"{model}:generations") == 1
    assert output.usage.get(f"{model}:requests") == 1
    assert output.usage.get(f"{model}:tokens") == 412_880


@pytest.mark.asyncio
async def test_fal_video_usage_via_collect():
    submit_response = MagicMock()
    submit_response.raise_for_status = MagicMock()
    submit_response.json.return_value = {
        "request_id": "req-usage",
        "status_url": "https://queue.fal.run/fal-ai/kling/requests/req-usage/status",
        "response_url": "https://queue.fal.run/fal-ai/kling/requests/req-usage",
    }

    status_done = MagicMock()
    status_done.raise_for_status = MagicMock()
    status_done.json.return_value = _fal_completed_status()

    result_response = MagicMock()
    result_response.raise_for_status = MagicMock()
    result_response.json.return_value = _fal_result_response()

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=submit_response)
    mock_client.get = AsyncMock(side_effect=[status_done, result_response])

    fal_model = "fal-ai/kling-video/v2/master/text-to-video"
    duration = 5
    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = VideoGenerate(fal_api_key=SecretStr("fal-test"))
        output = await _collect_video(
            tool,
            prompt="fal usage test",
            model=fal_model,
            aspect_ratio="16:9",
            resolution="720p",
            duration=duration,
            negative_prompt=None,
            image_url=None,
            generate_audio=False,
            seed=None,
            provider_params=None,
            poll_interval=0,
            timeout=30,
        )

    assert output.output["success"] is True
    assert output.usage.get(f"{fal_model}:video_seconds_720p") == duration
    assert output.usage.get(f"{fal_model}:generations") == 1
    assert output.usage.get(f"{fal_model}:requests") == 1
    assert "video_generate:requests" not in output.usage


@pytest.mark.asyncio
async def test_failure_records_no_video_usage():
    submit_response = MagicMock()
    submit_response.raise_for_status = MagicMock()
    submit_response.json.return_value = {"id": "task-fail-usage"}

    poll_failed = MagicMock()
    poll_failed.raise_for_status = MagicMock()
    poll_failed.json.return_value = {"status": "failed", "error": "quota exceeded"}

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=submit_response)
    mock_client.get = AsyncMock(return_value=poll_failed)

    model = "bytedance/seedance-2.0-fast"
    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(mock_client)):
        tool = VideoGenerate(bytedance_api_key=SecretStr("ark-test"))
        output = await _collect_video(
            tool,
            prompt="fail usage test",
            model=model,
            aspect_ratio="16:9",
            resolution="720p",
            duration=5,
            negative_prompt=None,
            image_url=None,
            generate_audio=True,
            seed=None,
            provider_params=None,
            poll_interval=0,
            timeout=10,
        )

    assert output.output["success"] is False
    assert not any(k.startswith(f"{model}:") for k in output.usage)
    assert "video_generate:requests" not in output.usage


@pytest.mark.integration
@pytest.mark.asyncio
async def test_google_veo_live():
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("Set GEMINI_API_KEY for Google Veo integration test")
    tool = VideoGenerate()
    out = await tool.handler(
        prompt="A red balloon floating over a calm lake at dawn, cinematic.",
        model="google/veo-3.1-fast",
        aspect_ratio="16:9",
        resolution="720p",
        duration=4,
        negative_prompt=None,
        image_url=None,
        generate_audio=False,
        seed=None,
        provider_params=None,
        poll_interval=10,
        timeout=600,
    )
    assert out["success"] is True, out.get("error")
    assert out["video_url"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_bytedance_seedance_live():
    if not (os.getenv("BYTEPLUS_API_KEY") or os.getenv("ARK_API_KEY")):
        pytest.skip("Set BYTEPLUS_API_KEY or ARK_API_KEY for Seedance integration test")
    tool = VideoGenerate()
    out = await tool.handler(
        prompt="Slow camera move over a minimalist product on white background.",
        model="bytedance/seedance-2.0-fast",
        aspect_ratio="16:9",
        resolution="720p",
        duration=5,
        negative_prompt=None,
        image_url=None,
        generate_audio=False,
        seed=None,
        provider_params=None,
        poll_interval=5,
        timeout=600,
    )
    assert out["success"] is True, out.get("error")
    assert out["video_url"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_fal_fallback_live():
    if not os.getenv("FAL_KEY"):
        pytest.skip("Set FAL_KEY for fal integration test")
    tool = VideoGenerate()
    out = await tool.handler(
        prompt="A paper boat drifting on a puddle, macro shot.",
        model="fal-ai/minimax/video-01",
        aspect_ratio="16:9",
        resolution="720p",
        duration=5,
        negative_prompt=None,
        image_url=None,
        generate_audio=False,
        seed=None,
        provider_params=None,
        poll_interval=5,
        timeout=600,
    )
    assert out["success"] is True, out.get("error")
    assert out["video_url"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_fal_veo31_fast_live():
    if not os.getenv("FAL_KEY"):
        pytest.skip("Set FAL_KEY for fal integration test")
    tool = VideoGenerate()
    out = await tool.handler(
        prompt="A paper boat drifting on calm water, macro cinematic.",
        model="fal-ai/veo3.1/fast",
        aspect_ratio="16:9",
        resolution="720p",
        duration=4,
        negative_prompt=None,
        image_url=None,
        generate_audio=False,
        seed=None,
        provider_params=None,
        poll_interval=5,
        timeout=600,
    )
    assert out["success"] is True, out.get("error")
    assert out["video_url"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_fal_invalid_params_fail_fast():
    if not os.getenv("FAL_KEY"):
        pytest.skip("Set FAL_KEY for fal integration test")
    tool = VideoGenerate()
    out = await tool.handler(
        prompt="test",
        model="fal-ai/veo3.1/fast",
        aspect_ratio="1:1",
        resolution="720p",
        duration=4,
        negative_prompt=None,
        image_url=None,
        generate_audio=False,
        seed=None,
        provider_params=None,
        poll_interval=5,
        timeout=60,
    )
    assert out["success"] is False
    assert out["error_type"] == "ValueError"
    assert "aspect ratio" in out["error"].lower()


@pytest.mark.asyncio
async def test_video_list_models_returns_catalog():
    tool = VideoListModels()
    out = await tool.handler(include_billing=True)
    assert out["default_model"] == "google/veo-3.1"
    assert len(out["direct_models"]) == 4
    assert out["fal_fallback"]["enabled"] is True
    veo = next(m for m in out["direct_models"] if m["id"] == "google/veo-3.1")
    units = {c["unit"] for c in veo["billing"]["costs"]}
    assert "video_seconds_720p" in units
    assert set(_VIDEO_MODELS) == {m["id"] for m in out["direct_models"]}


def test_list_video_cost_rows_matches_usage_keys():
    from timbal.core.video_models import list_video_cost_rows

    rows = list_video_cost_rows()
    assert len(rows) == 18
    names = {row["name"] for row in rows}
    assert names == {
        "google/veo-3.1",
        "google/veo-3.1-fast",
        "bytedance/seedance-2.0",
        "bytedance/seedance-2.0-fast",
    }
    assert all(row["currency"] == "USD" for row in rows)
