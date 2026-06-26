"""
Krea.ai REST API tools.

Async job flow: POST /generate/{media}/{provider}/{model} → job_id → poll GET /jobs/{id}.
Docs: https://docs.krea.ai/
"""

from __future__ import annotations

import asyncio
import time
from typing import Annotated, Any, Literal

import structlog
from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration
from ._creds import resolve_api_key

logger = structlog.get_logger("timbal.tools.krea")

_KREA_BASE = "https://api.krea.ai"
_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
_FAILED_STATUSES = frozenset({"failed", "cancelled"})

MediaKind = Literal["video", "image"]


def _krea_headers(api_key: str, *, webhook_url: str | None = None) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if webhook_url:
        headers["X-Webhook-URL"] = webhook_url
    return headers


def _format_krea_error(response: Any) -> str:
    try:
        body = response.json()
    except Exception:
        text = getattr(response, "text", "") or ""
        return text[:500] or f"HTTP {getattr(response, 'status_code', '?')}"

    if isinstance(body, dict):
        for key in ("message", "error", "detail"):
            value = body.get(key)
            if isinstance(value, str) and value:
                return value
    return str(body)[:500]


async def _raise_for_krea_response(response: Any) -> None:
    import httpx

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = _format_krea_error(response)
        raise RuntimeError(f"Krea API error ({response.status_code}): {detail}") from exc


def _build_generate_path(media: MediaKind, model: str) -> str:
    normalized = (model or "").strip().strip("/")
    if not normalized:
        raise ValueError("model is required (e.g. google/veo-3.1-fast or bfl/flux-1-dev)")
    prefix = f"generate/{media}/"
    if normalized.startswith(prefix):
        return f"/{normalized}"
    if normalized.startswith(f"{media}/"):
        return f"/generate/{normalized}"
    return f"/generate/{media}/{normalized}"


def _extract_job_urls(job: dict[str, Any]) -> list[str]:
    result = job.get("result")
    if not isinstance(result, dict):
        return []
    urls = result.get("urls")
    if isinstance(urls, list):
        return [str(u) for u in urls if isinstance(u, str) and u]
    return []


async def _poll_krea_job(
    client: Any,
    *,
    api_key: str,
    job_id: str,
    poll_interval: float,
    timeout: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last: dict[str, Any] = {}
    headers = _krea_headers(api_key)

    while time.monotonic() < deadline:
        response = await client.get(f"{_KREA_BASE}/jobs/{job_id}", headers=headers)
        await _raise_for_krea_response(response)
        last = response.json()
        status = (last.get("status") or "").lower()
        if status in _TERMINAL_STATUSES:
            if status in _FAILED_STATUSES:
                err = last.get("error") or {}
                message = err.get("message") if isinstance(err, dict) else str(err)
                raise RuntimeError(f"Krea job {job_id} {status}: {message or last}")
            return last
        if poll_interval > 0:
            await asyncio.sleep(poll_interval)

    raise TimeoutError(f"Krea job {job_id} timed out after {timeout}s (last status={last.get('status')})")


class KreaListJobs(Tool):
    name: str = "krea_list_jobs"
    description: str | None = (
        "List Krea generation jobs for the authenticated API key with optional pagination and filters."
    )
    integration: Annotated[str, Integration("krea")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _krea_list_jobs(
            limit: int = Field(100, description="Number of jobs to return (1-1000)."),
            cursor: str | None = Field(
                None,
                description="ISO 8601 timestamp cursor (jobs created before this time).",
            ),
            types: str | None = Field(
                None,
                description='Comma-separated job types filter (e.g. "flux,k1,externalImage").',
            ),
            status: str | None = Field(None, description="Filter by job status."),
        ) -> dict[str, Any]:
            api_key = await resolve_api_key(tool=self, provider_name="Krea", env_var="KREA_API_KEY")
            import httpx

            params: dict[str, Any] = {"limit": max(1, min(limit, 1000))}
            if cursor:
                params["cursor"] = cursor
            if types:
                params["types"] = types
            if status:
                params["status"] = status

            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, read=None)) as client:
                response = await client.get(
                    f"{_KREA_BASE}/jobs",
                    headers=_krea_headers(api_key),
                    params=params,
                )
                await _raise_for_krea_response(response)
                return response.json()

        super().__init__(handler=_krea_list_jobs, **kwargs)


class KreaGetJob(Tool):
    name: str = "krea_get_job"
    description: str | None = (
        "Get the current state of a Krea job by id. When status is completed, result.urls contains output URLs."
    )
    integration: Annotated[str, Integration("krea")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _krea_get_job(
            job_id: str = Field(..., description="Job UUID returned by a Krea generate call."),
        ) -> dict[str, Any]:
            api_key = await resolve_api_key(tool=self, provider_name="Krea", env_var="KREA_API_KEY")
            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, read=None)) as client:
                response = await client.get(
                    f"{_KREA_BASE}/jobs/{job_id}",
                    headers=_krea_headers(api_key),
                )
                await _raise_for_krea_response(response)
                return response.json()

        super().__init__(handler=_krea_get_job, **kwargs)


class KreaCancelJob(Tool):
    name: str = "krea_cancel_job"
    description: str | None = "Delete/cancel a Krea job by id."
    integration: Annotated[str, Integration("krea")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _krea_cancel_job(
            job_id: str = Field(..., description="Job UUID to delete/cancel."),
        ) -> dict[str, Any]:
            api_key = await resolve_api_key(tool=self, provider_name="Krea", env_var="KREA_API_KEY")
            import httpx

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=None)) as client:
                response = await client.delete(
                    f"{_KREA_BASE}/jobs/{job_id}",
                    headers=_krea_headers(api_key),
                )
                await _raise_for_krea_response(response)
                if response.content:
                    body = response.json()
                    if isinstance(body, dict):
                        return body
                return {"deleted": True, "job_id": job_id}

        super().__init__(handler=_krea_cancel_job, **kwargs)


class KreaGenerateVideo(Tool):
    name: str = "krea_generate_video"
    description: str | None = (
        "Generate a video via Krea.ai. Submits to POST /generate/video/{provider}/{model}, "
        "polls /jobs/{id} until complete, and returns output URLs. "
        "Examples: google/veo-3.1-fast, kling/kling-2.5, alibaba/wan-2.5, bytedance/seedance-2-fast."
    )
    integration: Annotated[str, Integration("krea")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _krea_generate_video(
            prompt: str = Field(..., description="Text description of the video to generate."),
            model: str = Field(
                "google/veo-3.1-fast",
                description="Krea video model path: provider/model (e.g. google/veo-3.1-fast).",
            ),
            aspect_ratio: str = Field("16:9", description="Aspect ratio when supported by the model."),
            duration: int | None = Field(None, description="Duration in seconds when supported by the model."),
            resolution: str | None = Field(None, description="Output resolution when supported (e.g. 720p, 1080p)."),
            generate_audio: bool | None = Field(None, description="Whether to generate audio when supported."),
            start_image: str | None = Field(
                None,
                description="Optional first-frame image URL, base64 data URI, or Krea asset URL.",
            ),
            end_image: str | None = Field(None, description="Optional last-frame image URL when supported."),
            reference_images: list[str] | None = Field(
                None,
                description="Optional reference image URLs when supported by the model.",
            ),
            provider_params: dict[str, Any] | None = Field(
                None,
                description="Additional model-native fields merged into the request body.",
            ),
            webhook_url: str | None = Field(
                None,
                description="Optional X-Webhook-URL header; Krea POSTs completion payload here.",
            ),
            poll_interval: float = Field(5.0, description="Seconds between job status polls."),
            timeout: float = Field(600.0, description="Max seconds to wait for job completion."),
        ) -> dict[str, Any]:
            start = time.monotonic()
            resolved_model = model.strip()

            try:
                if not prompt or not prompt.strip():
                    raise ValueError("prompt is required and must be non-empty")

                payload: dict[str, Any] = {"prompt": prompt.strip()}
                if aspect_ratio:
                    payload["aspect_ratio"] = aspect_ratio
                if duration is not None:
                    payload["duration"] = duration
                if resolution is not None:
                    payload["resolution"] = resolution
                if generate_audio is not None:
                    payload["generate_audio"] = generate_audio
                if start_image:
                    payload["start_image"] = start_image
                if end_image:
                    payload["end_image"] = end_image
                if reference_images:
                    payload["reference_images"] = reference_images
                if provider_params:
                    for key, value in provider_params.items():
                        if value is not None:
                            payload[key] = value

                path = _build_generate_path("video", resolved_model)
                api_key = await resolve_api_key(tool=self, provider_name="Krea", env_var="KREA_API_KEY")
                import httpx

                async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, read=None)) as client:
                    submit = await client.post(
                        f"{_KREA_BASE}{path}",
                        headers=_krea_headers(api_key, webhook_url=webhook_url),
                        json=payload,
                    )
                    await _raise_for_krea_response(submit)
                    submit_data = submit.json()

                    job_id = submit_data.get("job_id")
                    if not isinstance(job_id, str) or not job_id:
                        raise ValueError(f"Krea video API returned no job_id: {submit_data}")

                    job = await _poll_krea_job(
                        client,
                        api_key=api_key,
                        job_id=job_id,
                        poll_interval=poll_interval,
                        timeout=timeout,
                    )

                urls = _extract_job_urls(job)
                if not urls:
                    raise ValueError(f"No output URLs in completed Krea job: {job}")

                elapsed = time.monotonic() - start
                logger.info(
                    "Krea video generated",
                    model=resolved_model,
                    job_id=job_id,
                    elapsed_s=round(elapsed, 2),
                )
                return {
                    "success": True,
                    "model": resolved_model,
                    "job_id": job_id,
                    "video_url": urls[0],
                    "urls": urls,
                    "job": job,
                    "elapsed_seconds": round(elapsed, 2),
                }

            except Exception as exc:
                elapsed = time.monotonic() - start
                logger.error("Krea video generation failed", model=resolved_model, error=str(exc))
                return {
                    "success": False,
                    "model": resolved_model,
                    "job_id": None,
                    "video_url": None,
                    "urls": [],
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "elapsed_seconds": round(elapsed, 2),
                }

        super().__init__(handler=_krea_generate_video, **kwargs)


class KreaGenerateImage(Tool):
    name: str = "krea_generate_image"
    description: str | None = (
        "Generate an image via Krea.ai. Submits to POST /generate/image/{provider}/{model}, "
        "polls /jobs/{id} until complete, and returns output URLs. "
        "Examples: bfl/flux-1-dev, krea/krea-2/medium, google/imagen-4."
    )
    integration: Annotated[str, Integration("krea")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _krea_generate_image(
            prompt: str = Field(..., description="Text description of the image to generate."),
            model: str = Field(
                "bfl/flux-1-dev",
                description="Krea image model path: provider/model (e.g. bfl/flux-1-dev).",
            ),
            width: int | None = Field(None, description="Output width in pixels when supported."),
            height: int | None = Field(None, description="Output height in pixels when supported."),
            seed: int | None = Field(None, description="Optional random seed."),
            image_url: str | None = Field(
                None,
                description="Optional init/reference image URL for img2img when supported.",
            ),
            provider_params: dict[str, Any] | None = Field(
                None,
                description="Additional model-native fields merged into the request body.",
            ),
            webhook_url: str | None = Field(None, description="Optional X-Webhook-URL header."),
            poll_interval: float = Field(3.0, description="Seconds between job status polls."),
            timeout: float = Field(300.0, description="Max seconds to wait for job completion."),
        ) -> dict[str, Any]:
            start = time.monotonic()
            resolved_model = model.strip()

            try:
                if not prompt or not prompt.strip():
                    raise ValueError("prompt is required and must be non-empty")

                payload: dict[str, Any] = {"prompt": prompt.strip()}
                if width is not None:
                    payload["width"] = width
                if height is not None:
                    payload["height"] = height
                if seed is not None:
                    payload["seed"] = seed
                if image_url:
                    payload["image_url"] = image_url
                if provider_params:
                    for key, value in provider_params.items():
                        if value is not None:
                            payload[key] = value

                path = _build_generate_path("image", resolved_model)
                api_key = await resolve_api_key(tool=self, provider_name="Krea", env_var="KREA_API_KEY")
                import httpx

                async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, read=None)) as client:
                    submit = await client.post(
                        f"{_KREA_BASE}{path}",
                        headers=_krea_headers(api_key, webhook_url=webhook_url),
                        json=payload,
                    )
                    await _raise_for_krea_response(submit)
                    submit_data = submit.json()

                    job_id = submit_data.get("job_id")
                    if not isinstance(job_id, str) or not job_id:
                        raise ValueError(f"Krea image API returned no job_id: {submit_data}")

                    job = await _poll_krea_job(
                        client,
                        api_key=api_key,
                        job_id=job_id,
                        poll_interval=poll_interval,
                        timeout=timeout,
                    )

                urls = _extract_job_urls(job)
                if not urls:
                    raise ValueError(f"No output URLs in completed Krea job: {job}")

                elapsed = time.monotonic() - start
                logger.info(
                    "Krea image generated",
                    model=resolved_model,
                    job_id=job_id,
                    elapsed_s=round(elapsed, 2),
                )
                return {
                    "success": True,
                    "model": resolved_model,
                    "job_id": job_id,
                    "image_url": urls[0],
                    "urls": urls,
                    "job": job,
                    "elapsed_seconds": round(elapsed, 2),
                }

            except Exception as exc:
                elapsed = time.monotonic() - start
                logger.error("Krea image generation failed", model=resolved_model, error=str(exc))
                return {
                    "success": False,
                    "model": resolved_model,
                    "job_id": None,
                    "image_url": None,
                    "urls": [],
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "elapsed_seconds": round(elapsed, 2),
                }

        super().__init__(handler=_krea_generate_image, **kwargs)
