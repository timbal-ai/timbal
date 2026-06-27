"""Higgsfield AI platform tools (https://platform.higgsfield.ai).

Uses the public REST API with ``Authorization: Key {api_key}`` where ``api_key`` is
``key_id:secret`` (set ``HF_KEY``, pass ``api_key`` on the tool, or use a platform integration).
Application strings are model routes accepted by ``platform.higgsfield.ai`` —
hierarchical paths (``bytedance/seedream/v4/text-to-image``) or ``/v1/…``
endpoints from the official JS SDK. Override ``model`` when your plan exposes
different routes in the Cloud dashboard.

Default routes are aligned with official Higgsfield documentation:
- Python SDK (higgsfield-client README): Seedream text-to-image
- JS SDK (higgsfield-js): FLUX Kontext, Speak, custom-references
- CLI catalog (higgsfield-ai/cli MODELS.md): outpaint, background remover
- ComfyUI-Higgsfield-Direct (higgsfield-client routes): Soul, Reve, Seedream edit, Seedance/Kling I2V
"""

from __future__ import annotations

import asyncio
import base64
import time
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlencode

import httpx
from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..platform.integrations import Integration
from ._creds import resolve_api_key

_HF_BASE = "https://platform.higgsfield.ai"
_HF_TERMINAL_STATUSES = frozenset({"completed", "failed", "nsfw", "canceled", "cancelled"})

# Default application routes — override per call via the ``model`` parameter.
# higgsfield-client README
DEFAULT_TEXT_TO_IMAGE_MODEL = "bytedance/seedream/v4/text-to-image"
# ComfyUI-Higgsfield-Direct (higgsfield-client)
DEFAULT_IMAGE_TO_IMAGE_MODEL = "bytedance/seedream/v4/edit"
# higgsfield-ai/cli MODELS.md job_set_type
DEFAULT_REMOVE_BACKGROUND_MODEL = "image_background_remover"
# higgsfield-ai/cli MODELS.md job_set_type
DEFAULT_EXPAND_IMAGE_MODEL = "outpaint"
# Seedance v1 Pro (documented I2V sibling; MCP unified verified family)
DEFAULT_TEXT_TO_VIDEO_MODEL = "bytedance/seedance/v1/pro/text-to-video"
# ComfyUI-Higgsfield-Direct
DEFAULT_IMAGE_TO_VIDEO_MODEL = "bytedance/seedance/v1/pro/image-to-video"
# higgsfield-js Speak v2 endpoint
DEFAULT_LIPSYNC_MODEL = "/v1/speak/higgsfield"
# higgsfield-js createSoulId endpoint
DEFAULT_SOUL_TRAIN_MODEL = "/v1/custom-references"
# platform.higgsfield.ai Soul character generation (custom_reference_id consumer)
DEFAULT_SOUL_GENERATE_MODEL = "higgsfield-ai/soul/character"

_HF_CREDENTIAL_KEYS = ("api_key", "hf_key", "credential_key")


def _auth_headers(credential_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Key {credential_key}",
        "Content-Type": "application/json",
        "User-Agent": "timbal-higgsfield/1.0",
    }


def _clean_arguments(**kwargs: Any) -> dict[str, Any]:
    return {key: value for key, value in kwargs.items() if value is not None}


def _application_url(application: str, webhook_url: str | None = None) -> str:
    if not webhook_url:
        return application
    return f"{application}?{urlencode({'hf_webhook': webhook_url})}"


async def _credential_key(tool: Any) -> str:
    return await resolve_api_key(
        tool=tool,
        provider_name="Higgsfield",
        env_var="HF_KEY",
        integration_keys=_HF_CREDENTIAL_KEYS,
    )


class _HiggsfieldTool(Tool):
    integration: Annotated[str, Integration("higgsfield")] | None = None
    api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config({"integration": self.integration, "api_key": self.api_key}),
        }


async def _hf_submit(
    *,
    credential_key: str,
    application: str,
    arguments: dict[str, Any],
    webhook_url: str | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    async with httpx.AsyncClient(base_url=_HF_BASE, timeout=httpx.Timeout(timeout, connect=10.0)) as client:
        response = await client.post(
            _application_url(application, webhook_url),
            headers=_auth_headers(credential_key),
            json=arguments,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Unexpected Higgsfield submit response.")
        return payload


async def _hf_status(
    *,
    credential_key: str,
    request_id: str,
    timeout: float = 60.0,
) -> dict[str, Any]:
    async with httpx.AsyncClient(base_url=_HF_BASE, timeout=httpx.Timeout(timeout, connect=10.0)) as client:
        response = await client.get(
            f"/requests/{request_id}/status",
            headers=_auth_headers(credential_key),
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Unexpected Higgsfield status response.")
        return payload


async def _hf_cancel(*, credential_key: str, request_id: str) -> None:
    async with httpx.AsyncClient(base_url=_HF_BASE, timeout=httpx.Timeout(60.0, connect=10.0)) as client:
        response = await client.post(
            f"/requests/{request_id}/cancel",
            headers=_auth_headers(credential_key),
        )
        response.raise_for_status()


async def _hf_poll_result(
    *,
    credential_key: str,
    request_id: str,
    poll_interval: float = 2.0,
    timeout: float = 600.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status_payload = await _hf_status(credential_key=credential_key, request_id=request_id)
        status = str(status_payload.get("status", "")).lower()
        if status in _HF_TERMINAL_STATUSES:
            return status_payload
        if poll_interval > 0:
            await asyncio.sleep(poll_interval)
    raise TimeoutError(f"Higgsfield request {request_id} did not complete within {timeout}s.")


async def _hf_subscribe(
    *,
    credential_key: str,
    application: str,
    arguments: dict[str, Any],
    webhook_url: str | None = None,
    poll_interval: float = 2.0,
    timeout: float = 600.0,
) -> dict[str, Any]:
    submit_payload = await _hf_submit(
        credential_key=credential_key,
        application=application,
        arguments=arguments,
        webhook_url=webhook_url,
        timeout=timeout,
    )
    request_id = submit_payload.get("request_id")
    if not isinstance(request_id, str):
        raise ValueError("Higgsfield submit response missing request_id.")
    result = await _hf_poll_result(
        credential_key=credential_key,
        request_id=request_id,
        poll_interval=poll_interval,
        timeout=timeout,
    )
    result.setdefault("request_id", request_id)
    return result


async def _hf_upload_bytes(
    *,
    credential_key: str,
    data: bytes,
    content_type: str,
) -> str:
    async with httpx.AsyncClient(base_url=_HF_BASE, timeout=httpx.Timeout(120.0, connect=10.0)) as client:
        upload_meta = await client.post(
            "/files/generate-upload-url",
            headers=_auth_headers(credential_key),
            json={"content_type": content_type},
        )
        upload_meta.raise_for_status()
        meta = upload_meta.json()
        public_url = meta.get("public_url")
        upload_url = meta.get("upload_url")
        if not isinstance(public_url, str) or not isinstance(upload_url, str):
            raise ValueError("Unexpected Higgsfield upload URL response.")

        put_response = await client.put(
            upload_url,
            content=data,
            headers={"Content-Type": content_type},
        )
        put_response.raise_for_status()
        return public_url


class HiggsfieldSubmit(_HiggsfieldTool):
    name: str = "higgsfield_submit"
    description: str | None = (
        "Submit an async Higgsfield generation job. Returns request_id and status URLs. "
        "Use higgsfield_check_status / higgsfield_get_result to poll, or prefer a "
        "high-level higgsfield_* generation tool with wait=true."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _submit(
            model: str = Field(..., description="Higgsfield application route, e.g. bytedance/seedream/v4/text-to-image."),
            arguments: dict[str, Any] = Field(..., description="Model-specific JSON arguments."),
            webhook_url: str | None = Field(None, description="Optional webhook URL notified on completion."),
        ) -> dict[str, Any]:
            credential_key = await _credential_key(self)
            return await _hf_submit(
                credential_key=credential_key,
                application=model,
                arguments=arguments,
                webhook_url=webhook_url,
            )

        super().__init__(handler=_submit, **kwargs)


class HiggsfieldCheckStatus(_HiggsfieldTool):
    name: str = "higgsfield_check_status"
    description: str | None = (
        "Check Higgsfield request status. Returns queued | in_progress | completed | failed | nsfw | canceled."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _check_status(
            request_id: str = Field(..., description="Request ID from higgsfield_submit or a generation tool."),
        ) -> dict[str, Any]:
            credential_key = await _credential_key(self)
            payload = await _hf_status(credential_key=credential_key, request_id=request_id)
            return {
                "request_id": request_id,
                "status": payload.get("status"),
                "raw": payload,
            }

        super().__init__(handler=_check_status, **kwargs)


class HiggsfieldGetResult(_HiggsfieldTool):
    name: str = "higgsfield_get_result"
    description: str | None = (
        "Poll a Higgsfield request until terminal and return the final payload (images, videos, etc.)."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _get_result(
            request_id: str = Field(..., description="Request ID to wait on."),
            poll_interval: float = Field(2.0, description="Seconds between status polls."),
            timeout: float = Field(600.0, description="Max seconds to wait before raising TimeoutError."),
        ) -> dict[str, Any]:
            credential_key = await _credential_key(self)
            result = await _hf_poll_result(
                credential_key=credential_key,
                request_id=request_id,
                poll_interval=poll_interval,
                timeout=timeout,
            )
            result.setdefault("request_id", request_id)
            return result

        super().__init__(handler=_get_result, **kwargs)


class HiggsfieldCancelRequest(_HiggsfieldTool):
    name: str = "higgsfield_cancel_request"
    description: str | None = "Cancel a queued Higgsfield request (cannot cancel in-progress jobs)."

    def __init__(self, **kwargs: Any) -> None:
        async def _cancel(
            request_id: str = Field(..., description="Request ID to cancel."),
        ) -> dict[str, Any]:
            credential_key = await _credential_key(self)
            await _hf_cancel(credential_key=credential_key, request_id=request_id)
            return {"request_id": request_id, "cancelled": True}

        super().__init__(handler=_cancel, **kwargs)


class HiggsfieldUploadFile(_HiggsfieldTool):
    name: str = "higgsfield_upload_file"
    description: str | None = (
        "Upload an image or video to Higgsfield storage. Returns a public URL for use in generation tools."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _upload(
            file_path: str | None = Field(None, description="Local path to upload."),
            file_base64: str | None = Field(None, description="Base64-encoded file bytes (alternative to file_path)."),
            content_type: str | None = Field(
                None,
                description="MIME type (e.g. image/jpeg). Guessed from file_path when omitted.",
            ),
        ) -> dict[str, Any]:
            if not file_path and not file_base64:
                raise ValueError("Provide file_path or file_base64.")

            credential_key = await _credential_key(self)

            if file_path:
                path = Path(file_path)
                data = path.read_bytes()
                mime = content_type or _guess_mime_type(path.suffix)
            else:
                data = base64.b64decode(file_base64 or "")
                mime = content_type or "application/octet-stream"

            url = await _hf_upload_bytes(credential_key=credential_key, data=data, content_type=mime)
            return {"url": url, "content_type": mime}

        super().__init__(handler=_upload, **kwargs)


def _guess_mime_type(suffix: str) -> str:
    mapping = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
    }
    return mapping.get(suffix.lower(), "application/octet-stream")


async def _run_generation(
    tool: Any,
    *,
    model: str,
    arguments: dict[str, Any],
    wait: bool,
    webhook_url: str | None,
    poll_interval: float,
    timeout: float,
) -> dict[str, Any]:
    credential_key = await _credential_key(tool)
    if wait:
        return await _hf_subscribe(
            credential_key=credential_key,
            application=model,
            arguments=arguments,
            webhook_url=webhook_url,
            poll_interval=poll_interval,
            timeout=timeout,
        )
    submit_payload = await _hf_submit(
        credential_key=credential_key,
        application=model,
        arguments=arguments,
        webhook_url=webhook_url,
    )
    return {"submitted": True, **submit_payload}


class HiggsfieldTextToImage(_HiggsfieldTool):
    name: str = "higgsfield_text_to_image"
    description: str | None = (
        "Generate an image from text via Higgsfield (default Seedream 4). "
        "Models: Seedream, Nano Banana, GPT Image 2, Flux 2, Soul 2.0, etc."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _text_to_image(
            prompt: str = Field(..., description="Text prompt."),
            model: str = Field(DEFAULT_TEXT_TO_IMAGE_MODEL, description="Higgsfield application route."),
            resolution: str | None = Field(None, description='e.g. "2K", "1080p".'),
            aspect_ratio: str | None = Field(None, description='e.g. "16:9", "1:1".'),
            camera_fixed: bool | None = Field(None, description="Fix camera for video-style generations."),
            seed: int | None = Field(None, description="Optional seed."),
            wait: bool = Field(True, description="If true, poll until complete; if false, return request_id only."),
            webhook_url: str | None = Field(None, description="Optional completion webhook."),
            poll_interval: float = Field(2.0, description="Poll interval when wait=true."),
            timeout: float = Field(600.0, description="Max wait seconds when wait=true."),
        ) -> dict[str, Any]:
            arguments = _clean_arguments(
                prompt=prompt,
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                camera_fixed=camera_fixed,
                seed=seed,
            )
            return await _run_generation(
                self,
                model=model,
                arguments=arguments,
                wait=wait,
                webhook_url=webhook_url,
                poll_interval=poll_interval,
                timeout=timeout,
            )

        super().__init__(handler=_text_to_image, **kwargs)


class HiggsfieldImageToImage(_HiggsfieldTool):
    name: str = "higgsfield_image_to_image"
    description: str | None = "Edit or transform an image with a prompt (default Seedream 4 edit)."

    def __init__(self, **kwargs: Any) -> None:
        async def _image_to_image(
            image_url: str = Field(..., description="Source image URL (upload via higgsfield_upload_file first)."),
            prompt: str = Field(..., description="Edit instruction."),
            model: str = Field(DEFAULT_IMAGE_TO_IMAGE_MODEL, description="Higgsfield application route."),
            strength: float | None = Field(None, description="Edit strength when supported by the model."),
            seed: int | None = Field(None, description="Optional seed."),
            wait: bool = Field(True, description="Poll until complete when true."),
            webhook_url: str | None = None,
            poll_interval: float = 2.0,
            timeout: float = 600.0,
        ) -> dict[str, Any]:
            arguments = _clean_arguments(
                image_url=image_url,
                prompt=prompt,
                strength=strength,
                seed=seed,
            )
            return await _run_generation(
                self,
                model=model,
                arguments=arguments,
                wait=wait,
                webhook_url=webhook_url,
                poll_interval=poll_interval,
                timeout=timeout,
            )

        super().__init__(handler=_image_to_image, **kwargs)


class HiggsfieldRemoveBackground(_HiggsfieldTool):
    name: str = "higgsfield_remove_background"
    description: str | None = "Remove the background from an image."

    def __init__(self, **kwargs: Any) -> None:
        async def _remove_background(
            image_url: str = Field(..., description="Image URL."),
            model: str = Field(DEFAULT_REMOVE_BACKGROUND_MODEL, description="Higgsfield application route."),
            wait: bool = True,
            webhook_url: str | None = None,
            poll_interval: float = 2.0,
            timeout: float = 600.0,
        ) -> dict[str, Any]:
            arguments = _clean_arguments(image_url=image_url)
            return await _run_generation(
                self,
                model=model,
                arguments=arguments,
                wait=wait,
                webhook_url=webhook_url,
                poll_interval=poll_interval,
                timeout=timeout,
            )

        super().__init__(handler=_remove_background, **kwargs)


class HiggsfieldExpandImage(_HiggsfieldTool):
    name: str = "higgsfield_expand_image"
    description: str | None = "Outpaint / expand image borders."

    def __init__(self, **kwargs: Any) -> None:
        async def _expand_image(
            image_url: str = Field(..., description="Image URL."),
            direction: str | None = Field(None, description='Expansion direction, e.g. "left", "right", "all".'),
            aspect_ratio: str | None = Field(None, description='Target aspect ratio, e.g. "16:9".'),
            model: str = Field(DEFAULT_EXPAND_IMAGE_MODEL, description="Higgsfield application route."),
            wait: bool = True,
            webhook_url: str | None = None,
            poll_interval: float = 2.0,
            timeout: float = 600.0,
        ) -> dict[str, Any]:
            arguments = _clean_arguments(image_url=image_url, direction=direction, aspect_ratio=aspect_ratio)
            return await _run_generation(
                self,
                model=model,
                arguments=arguments,
                wait=wait,
                webhook_url=webhook_url,
                poll_interval=poll_interval,
                timeout=timeout,
            )

        super().__init__(handler=_expand_image, **kwargs)


class HiggsfieldTextToVideo(_HiggsfieldTool):
    name: str = "higgsfield_text_to_video"
    description: str | None = (
        "Generate video from text. Default Seedance; also Kling 3.0, Veo 3.1, Wan 2.6, Cinema Studio, etc."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _text_to_video(
            prompt: str = Field(..., description="Video prompt."),
            model: str = Field(DEFAULT_TEXT_TO_VIDEO_MODEL, description="Higgsfield application route."),
            duration: int | None = Field(None, description="Duration in seconds when supported."),
            resolution: str | None = Field(None, description='e.g. "720p", "1080p".'),
            aspect_ratio: str | None = Field(None, description='e.g. "16:9".'),
            seed: int | None = Field(None, description="Optional seed."),
            wait: bool = True,
            webhook_url: str | None = None,
            poll_interval: float = 2.0,
            timeout: float = 900.0,
        ) -> dict[str, Any]:
            arguments = _clean_arguments(
                prompt=prompt,
                duration=duration,
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                seed=seed,
            )
            return await _run_generation(
                self,
                model=model,
                arguments=arguments,
                wait=wait,
                webhook_url=webhook_url,
                poll_interval=poll_interval,
                timeout=timeout,
            )

        super().__init__(handler=_text_to_video, **kwargs)


class HiggsfieldImageToVideo(_HiggsfieldTool):
    name: str = "higgsfield_image_to_video"
    description: str | None = "Animate a still image into video (default Seedance image-to-video)."

    def __init__(self, **kwargs: Any) -> None:
        async def _image_to_video(
            image_url: str = Field(..., description="Start frame image URL."),
            prompt: str | None = Field(None, description="Optional motion prompt."),
            model: str = Field(DEFAULT_IMAGE_TO_VIDEO_MODEL, description="Higgsfield application route."),
            motion_id: str | None = Field(None, description="Preset motion identifier when supported."),
            motion_strength: float | None = Field(None, description="Motion strength 0-1 when supported."),
            duration: int | None = Field(None, description="Duration in seconds."),
            seed: int | None = Field(None, description="Optional seed."),
            wait: bool = True,
            webhook_url: str | None = None,
            poll_interval: float = 2.0,
            timeout: float = 900.0,
        ) -> dict[str, Any]:
            arguments = _clean_arguments(
                image_url=image_url,
                prompt=prompt,
                motion_id=motion_id,
                motion_strength=motion_strength,
                duration=duration,
                seed=seed,
            )
            return await _run_generation(
                self,
                model=model,
                arguments=arguments,
                wait=wait,
                webhook_url=webhook_url,
                poll_interval=poll_interval,
                timeout=timeout,
            )

        super().__init__(handler=_image_to_video, **kwargs)


class HiggsfieldLipsync(_HiggsfieldTool):
    name: str = "higgsfield_lipsync"
    description: str | None = "Talking avatar / lipsync — sync face to audio or TTS text."

    def __init__(self, **kwargs: Any) -> None:
        async def _lipsync(
            image_url: str = Field(..., description="Face/portrait image URL."),
            audio_url: str | None = Field(None, description="Audio URL for lipsync."),
            text: str | None = Field(None, description="Text for TTS when audio_url is omitted."),
            voice: str | None = Field(None, description="Voice id/name when using text TTS."),
            model: str = Field(DEFAULT_LIPSYNC_MODEL, description="Higgsfield application route."),
            wait: bool = True,
            webhook_url: str | None = None,
            poll_interval: float = 2.0,
            timeout: float = 900.0,
        ) -> dict[str, Any]:
            if not audio_url and not text:
                raise ValueError("Provide audio_url or text for lipsync.")
            arguments = _clean_arguments(image_url=image_url, audio_url=audio_url, text=text, voice=voice)
            return await _run_generation(
                self,
                model=model,
                arguments=arguments,
                wait=wait,
                webhook_url=webhook_url,
                poll_interval=poll_interval,
                timeout=timeout,
            )

        super().__init__(handler=_lipsync, **kwargs)


class HiggsfieldSoulTrain(_HiggsfieldTool):
    name: str = "higgsfield_soul_train"
    description: str | None = "Train a Soul character (consistent identity) from reference photos."

    def __init__(self, **kwargs: Any) -> None:
        async def _soul_train(
            name: str = Field(..., description="Character name."),
            image_urls: list[str] = Field(..., description="Reference photo URLs (5-20 recommended)."),
            style: str | None = Field(None, description="Optional style preset."),
            model: str = Field(DEFAULT_SOUL_TRAIN_MODEL, description="Higgsfield Soul training route."),
            wait: bool = True,
            webhook_url: str | None = None,
            poll_interval: float = 5.0,
            timeout: float = 3600.0,
        ) -> dict[str, Any]:
            arguments = _clean_arguments(name=name, image_urls=image_urls, style=style)
            return await _run_generation(
                self,
                model=model,
                arguments=arguments,
                wait=wait,
                webhook_url=webhook_url,
                poll_interval=poll_interval,
                timeout=timeout,
            )

        super().__init__(handler=_soul_train, **kwargs)


class HiggsfieldSoulGenerate(_HiggsfieldTool):
    name: str = "higgsfield_soul_generate"
    description: str | None = "Generate image or video using a trained Soul character."

    def __init__(self, **kwargs: Any) -> None:
        async def _soul_generate(
            soul_id: str = Field(..., description="reference_id from higgsfield_soul_train (sent as custom_reference_id)."),
            prompt: str = Field(..., description="Generation prompt."),
            reference_strength: float = Field(
                1.0, description="Custom reference strength (0-1). Defaults to 1.", ge=0.0, le=1.0
            ),
            model: str = Field(DEFAULT_SOUL_GENERATE_MODEL, description="Higgsfield Soul generation route."),
            aspect_ratio: str | None = Field(None, description='One of "9:16", "16:9", "4:3", "3:4", "1:1", "2:3", "3:2".'),
            resolution: str | None = Field(None, description='"720p" or "1080p".'),
            batch_size: int | None = Field(None, description="1 or 4."),
            seed: int | None = Field(None, description="Optional seed (1-1000000)."),
            enhance_prompt: bool | None = Field(None, description="Let Higgsfield expand the prompt."),
            wait: bool = True,
            webhook_url: str | None = None,
            poll_interval: float = 2.0,
            timeout: float = 900.0,
        ) -> dict[str, Any]:
            arguments = _clean_arguments(
                prompt=prompt,
                custom_reference_id=soul_id,
                custom_reference_strength=reference_strength,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                batch_size=batch_size,
                seed=seed,
                enhance_prompt=enhance_prompt,
            )
            return await _run_generation(
                self,
                model=model,
                arguments=arguments,
                wait=wait,
                webhook_url=webhook_url,
                poll_interval=poll_interval,
                timeout=timeout,
            )

        super().__init__(handler=_soul_generate, **kwargs)
