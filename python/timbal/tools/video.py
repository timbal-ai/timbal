"""
Model-agnostic video generation tool.

Routes unified inputs to Google Veo (direct), ByteDance Seedance (BytePlus direct),
or fal.ai (fallback for any other model string). Each provider has its own payload
shape and polling contract; complexity lives here so callers only pass prompt + model.
"""

from __future__ import annotations

import asyncio
import base64
import os
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal
from urllib.parse import quote, urlparse

import structlog
from pydantic import Field, SecretStr

from ..core.tool import Tool
from ..core.video_models import get_default_video_model, list_video_models
from ..platform.integrations import Integration
from ..state import get_run_context

logger = structlog.get_logger("timbal.tools.video")

_GOOGLE_BASE = "https://generativelanguage.googleapis.com/v1beta"
_BYTEDANCE_BASE = "https://ark.ap-southeast.bytepluses.com/api/v3/contents/generations/tasks"
_FAL_QUEUE_BASE = "https://queue.fal.run"

DEFAULT_MODEL = get_default_video_model()
VALID_ASPECT_RATIOS = ("16:9", "9:16", "1:1")
VALID_RESOLUTIONS = ("720p", "1080p", "480p")

ProviderName = Literal["google", "bytedance", "fal"]


DurationFormat = Literal["int", "enum_s", "int_str"]


@dataclass(frozen=True, slots=True)
class _VideoModelConfig:
    """Static configuration for a single video model."""

    provider: ProviderName
    model_id: str
    supports: frozenset[str]
    defaults: dict[str, Any] = field(default_factory=dict)
    supports_audio: bool = False
    supports_image_to_video: bool = True
    fal_image_key: str = "image_url"
    """Key used for first-frame image on fal endpoints (image_url vs image_urls)."""
    duration_format: DurationFormat = "int"
    """How to encode ``duration`` in fal payloads (int, ``4s`` enum, or ``\"5\"`` string)."""
    allowed_durations: frozenset[int] | None = None
    allowed_aspect_ratios: frozenset[str] | None = None
    allowed_resolutions: frozenset[str] | None = None


def _optional_int_frozenset(values: list[Any] | None) -> frozenset[int] | None:
    if not values:
        return None
    return frozenset(int(v) for v in values)


def _optional_str_frozenset(values: list[Any] | None) -> frozenset[str] | None:
    if not values:
        return None
    return frozenset(str(v) for v in values)


def _yaml_entry_to_config(entry: dict[str, Any]) -> _VideoModelConfig:
    provider = entry.get("provider")
    if provider not in ("google", "bytedance", "fal"):
        raise ValueError(f"Invalid provider in video_models.yaml for {entry.get('id')}: {provider!r}")

    supports = entry.get("supports") or []
    allowed_durations = entry.get("allowed_durations")
    return _VideoModelConfig(
        provider=provider,
        model_id=str(entry.get("model_id") or entry["id"]),
        supports=frozenset(str(s) for s in supports),
        defaults=dict(entry.get("defaults") or {}),
        supports_audio=bool(entry.get("supports_audio", False)),
        supports_image_to_video=bool(entry.get("supports_image_to_video", True)),
        fal_image_key=str(entry.get("fal_image_key") or "image_url"),
        duration_format=entry.get("duration_format") or "int",
        allowed_durations=_optional_int_frozenset(allowed_durations),
        allowed_aspect_ratios=_optional_str_frozenset(entry.get("allowed_aspect_ratios")),
        allowed_resolutions=_optional_str_frozenset(entry.get("allowed_resolutions")),
    )


def _load_direct_video_models() -> dict[str, _VideoModelConfig]:
    catalog: dict[str, _VideoModelConfig] = {}
    for entry in list_video_models(include_billing=False, include_runtime=True):
        model_id = entry.get("id")
        if not isinstance(model_id, str):
            continue
        catalog[model_id] = _yaml_entry_to_config(entry)
    return catalog


_VIDEO_MODELS: dict[str, _VideoModelConfig] = _load_direct_video_models()

_FAL_VEO_SUPPORTS = frozenset(
    {
        "prompt",
        "aspect_ratio",
        "duration",
        "resolution",
        "negative_prompt",
        "generate_audio",
        "seed",
        "auto_fix",
        "safety_tolerance",
    }
)
_FAL_KLING_SUPPORTS = frozenset(
    {
        "prompt",
        "aspect_ratio",
        "duration",
        "negative_prompt",
        "cfg_scale",
        "seed",
    }
)
_FAL_MINIMAX_SUPPORTS = frozenset({"prompt"})
_FAL_LUMA_SUPPORTS = frozenset({"prompt", "aspect_ratio", "loop"})
_FAL_WAN_SUPPORTS = frozenset({"prompt", "aspect_ratio", "resolution", "num_frames", "frames_per_second", "seed"})

_FAL_VIDEO_CATALOG: dict[str, _VideoModelConfig] = {
    "fal-ai/veo3.1": _VideoModelConfig(
        provider="fal",
        model_id="fal-ai/veo3.1",
        supports=_FAL_VEO_SUPPORTS,
        defaults={"resolution": "720p", "generate_audio": True},
        supports_audio=True,
        duration_format="enum_s",
        allowed_durations=frozenset({4, 6, 8}),
        allowed_aspect_ratios=frozenset({"16:9", "9:16"}),
        allowed_resolutions=frozenset({"720p", "1080p", "4k"}),
    ),
    "fal-ai/veo3.1/fast": _VideoModelConfig(
        provider="fal",
        model_id="fal-ai/veo3.1/fast",
        supports=_FAL_VEO_SUPPORTS,
        defaults={"resolution": "720p", "generate_audio": True},
        supports_audio=True,
        duration_format="enum_s",
        allowed_durations=frozenset({4, 6, 8}),
        allowed_aspect_ratios=frozenset({"16:9", "9:16"}),
        allowed_resolutions=frozenset({"720p", "1080p", "4k"}),
    ),
    "fal-ai/veo3": _VideoModelConfig(
        provider="fal",
        model_id="fal-ai/veo3",
        supports=_FAL_VEO_SUPPORTS,
        defaults={"resolution": "720p", "generate_audio": True},
        supports_audio=True,
        duration_format="enum_s",
        allowed_durations=frozenset({4, 6, 8}),
        allowed_aspect_ratios=frozenset({"16:9", "9:16"}),
        allowed_resolutions=frozenset({"720p", "1080p", "4k"}),
    ),
    "fal-ai/veo3/fast": _VideoModelConfig(
        provider="fal",
        model_id="fal-ai/veo3/fast",
        supports=_FAL_VEO_SUPPORTS,
        defaults={"resolution": "720p", "generate_audio": True},
        supports_audio=True,
        duration_format="enum_s",
        allowed_durations=frozenset({4, 6, 8}),
        allowed_aspect_ratios=frozenset({"16:9", "9:16"}),
        allowed_resolutions=frozenset({"720p", "1080p", "4k"}),
    ),
    "fal-ai/kling-video/v2/master/text-to-video": _VideoModelConfig(
        provider="fal",
        model_id="fal-ai/kling-video/v2/master/text-to-video",
        supports=_FAL_KLING_SUPPORTS,
        defaults={"duration": "5", "aspect_ratio": "16:9"},
        duration_format="int_str",
        allowed_durations=frozenset({5, 10}),
        allowed_aspect_ratios=frozenset({"16:9", "9:16", "1:1"}),
    ),
    "fal-ai/minimax/video-01-live/text-to-video": _VideoModelConfig(
        provider="fal",
        model_id="fal-ai/minimax/video-01-live/text-to-video",
        supports=_FAL_MINIMAX_SUPPORTS,
        supports_image_to_video=False,
    ),
    "fal-ai/minimax/video-01": _VideoModelConfig(
        provider="fal",
        model_id="fal-ai/minimax/video-01",
        supports=_FAL_MINIMAX_SUPPORTS,
        supports_image_to_video=False,
    ),
    "fal-ai/luma-dream-machine": _VideoModelConfig(
        provider="fal",
        model_id="fal-ai/luma-dream-machine",
        supports=_FAL_LUMA_SUPPORTS,
        allowed_aspect_ratios=frozenset({"16:9", "9:16", "4:3", "3:4", "21:9", "9:21"}),
    ),
    "fal-ai/wan/v2.1/1.3b/text-to-video": _VideoModelConfig(
        provider="fal",
        model_id="fal-ai/wan/v2.1/1.3b/text-to-video",
        supports=_FAL_WAN_SUPPORTS,
        allowed_aspect_ratios=frozenset({"16:9", "9:16", "1:1"}),
        allowed_resolutions=frozenset({"480p", "580p", "720p"}),
    ),
}

_FAL_CATALOG_PREFIXES: tuple[tuple[str, _VideoModelConfig], ...] = (
    ("fal-ai/kling-video/", _FAL_VIDEO_CATALOG["fal-ai/kling-video/v2/master/text-to-video"]),
    ("fal-ai/veo3.1", _FAL_VIDEO_CATALOG["fal-ai/veo3.1"]),
    ("fal-ai/veo3", _FAL_VIDEO_CATALOG["fal-ai/veo3"]),
    ("fal-ai/minimax/video-01", _FAL_VIDEO_CATALOG["fal-ai/minimax/video-01"]),
    ("fal-ai/wan/", _FAL_VIDEO_CATALOG["fal-ai/wan/v2.1/1.3b/text-to-video"]),
)

_FAL_FALLBACK_SUPPORTS = frozenset(
    {
        "prompt",
        "aspect_ratio",
        "duration",
        "resolution",
        "image_url",
        "image_urls",
        "seed",
        "negative_prompt",
        "generate_audio",
    }
)


def _lookup_fal_catalog(model_id: str) -> _VideoModelConfig | None:
    if model_id in _FAL_VIDEO_CATALOG:
        return _FAL_VIDEO_CATALOG[model_id]
    for prefix, template in _FAL_CATALOG_PREFIXES:
        if model_id.startswith(prefix):
            return _VideoModelConfig(
                provider=template.provider,
                model_id=model_id,
                supports=template.supports,
                defaults=dict(template.defaults),
                supports_audio=template.supports_audio,
                supports_image_to_video=template.supports_image_to_video,
                fal_image_key=template.fal_image_key,
                duration_format=template.duration_format,
                allowed_durations=template.allowed_durations,
                allowed_aspect_ratios=template.allowed_aspect_ratios,
                allowed_resolutions=template.allowed_resolutions,
            )
    return None


def _infer_fal_heuristics(model_id: str) -> _VideoModelConfig:
    """Best-effort config for unknown fal endpoints based on path segments."""
    lower = model_id.lower()
    if "veo3" in lower or "/veo/" in lower:
        template = _FAL_VIDEO_CATALOG["fal-ai/veo3.1"]
        return _VideoModelConfig(
            provider="fal",
            model_id=model_id,
            supports=template.supports,
            defaults=dict(template.defaults),
            supports_audio=template.supports_audio,
            duration_format=template.duration_format,
            allowed_durations=template.allowed_durations,
            allowed_aspect_ratios=template.allowed_aspect_ratios,
            allowed_resolutions=template.allowed_resolutions,
        )
    if "kling" in lower:
        template = _FAL_VIDEO_CATALOG["fal-ai/kling-video/v2/master/text-to-video"]
        return _VideoModelConfig(
            provider="fal",
            model_id=model_id,
            supports=template.supports,
            defaults=dict(template.defaults),
            duration_format=template.duration_format,
            allowed_durations=template.allowed_durations,
            allowed_aspect_ratios=template.allowed_aspect_ratios,
        )
    return _VideoModelConfig(
        provider="fal",
        model_id=model_id,
        supports=_FAL_FALLBACK_SUPPORTS,
        defaults={},
        supports_audio=True,
        fal_image_key="image_url",
    )


def _resolve_model_config(model: str) -> _VideoModelConfig:
    """Return catalog config or synthesize a fal fallback config for unknown model strings."""
    normalized = (model or DEFAULT_MODEL).strip()
    if normalized in _VIDEO_MODELS:
        return _VIDEO_MODELS[normalized]
    catalog = _lookup_fal_catalog(normalized)
    if catalog is not None:
        return catalog
    return _infer_fal_heuristics(normalized)


def _format_fal_duration(config: _VideoModelConfig, duration: int) -> str | int:
    if config.duration_format == "enum_s":
        return f"{duration}s"
    if config.duration_format == "int_str":
        return str(duration)
    return duration


def _validate_fal_constraints(
    config: _VideoModelConfig,
    *,
    aspect_ratio: str,
    resolution: str,
    duration: int,
) -> int:
    """Validate unified inputs against fal catalog constraints; return billed duration seconds."""
    billed = duration
    if config.allowed_durations is not None and duration not in config.allowed_durations:
        allowed = ", ".join(str(d) for d in sorted(config.allowed_durations))
        raise ValueError(
            f"Model '{config.model_id}' only supports duration values (seconds): {allowed}. Got {duration}."
        )
    if config.allowed_aspect_ratios is not None and aspect_ratio not in config.allowed_aspect_ratios:
        allowed = ", ".join(sorted(config.allowed_aspect_ratios))
        raise ValueError(f"Model '{config.model_id}' only supports aspect ratios: {allowed}. Got '{aspect_ratio}'.")
    if config.allowed_resolutions is not None and resolution not in config.allowed_resolutions:
        allowed = ", ".join(sorted(config.allowed_resolutions))
        raise ValueError(f"Model '{config.model_id}' only supports resolutions: {allowed}. Got '{resolution}'.")
    return billed


def _format_fal_http_error(response: Any) -> str:
    """Extract fal ``detail`` from an httpx response for clearer tool errors."""
    try:
        body = response.json()
    except Exception:
        return str(response.text)[:500] or f"HTTP {response.status_code}"

    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, str):
            return detail
        if isinstance(detail, list):
            parts: list[str] = []
            for item in detail:
                if isinstance(item, dict):
                    loc = ".".join(str(x) for x in item.get("loc", ()))
                    msg = item.get("msg", "")
                    parts.append(f"{loc}: {msg}" if loc else str(msg))
                else:
                    parts.append(str(item))
            if parts:
                return "; ".join(parts)
    return str(body)[:500]


async def _raise_for_fal_response(response: Any) -> None:
    import httpx

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = _format_fal_http_error(response)
        raise RuntimeError(f"fal API error ({response.status_code}): {detail}") from exc


async def _fetch_fal_result(
    client: Any,
    *,
    api_key: str,
    submit_model_id: str,
    queue_model_path: str,
    request_id: str,
    response_url: str | None,
) -> dict[str, Any]:
    """Fetch fal queue result, trying canonical and submit paths when fal rewrites routes."""
    import httpx

    headers = {"Authorization": f"Key {api_key}"}
    candidates: list[str] = []
    if response_url:
        candidates.append(response_url)
    for model_path in (queue_model_path, submit_model_id):
        encoded = _encode_fal_model_path(model_path)
        built = f"{_FAL_QUEUE_BASE}/{encoded}/requests/{request_id}"
        if built not in candidates:
            candidates.append(built)

    last_error: str | None = None
    for url in candidates:
        result_response = await client.get(
            url,
            headers=headers,
            timeout=httpx.Timeout(120.0, read=None),
        )
        if result_response.status_code == 404:
            last_error = _format_fal_http_error(result_response)
            continue
        await _raise_for_fal_response(result_response)
        return result_response.json()

    raise RuntimeError(
        f"fal result fetch failed for '{submit_model_id}' (request_id={request_id}). "
        f"Tried {len(candidates)} URL(s). Last error: {last_error}"
    )


def _encode_fal_model_path(model_id: str) -> str:
    return "/".join(quote(part, safe="") for part in model_id.split("/"))


def _fal_queue_model_path_from_submit(data: dict[str, Any]) -> str | None:
    url = data.get("status_url") or data.get("response_url")
    if not isinstance(url, str) or "/requests/" not in url:
        return None
    try:
        path = urlparse(url).path.strip("/").split("/")
        i = path.index("requests")
        if i > 0:
            return "/".join(path[:i])
    except ValueError:
        return None
    return None


async def _resolve_google_key(tool: Any) -> str:
    if getattr(tool, "google_integration", None) is not None and isinstance(tool.google_integration, Integration):
        credentials = await tool.google_integration.resolve()
        key = credentials.get("api_key")
        if key:
            return str(key)
    if tool.google_api_key is not None:
        return tool.google_api_key.get_secret_value()
    env_key = os.getenv("GEMINI_API_KEY")
    if env_key:
        return env_key
    raise ValueError(
        "Google API key not found. Set GEMINI_API_KEY, pass google_api_key, or configure a gemini integration."
    )


async def _resolve_bytedance_key(tool: Any) -> str:
    if getattr(tool, "bytedance_integration", None) is not None and isinstance(tool.bytedance_integration, Integration):
        credentials = await tool.bytedance_integration.resolve()
        key = credentials.get("api_key")
        if key:
            return str(key)
    if tool.bytedance_api_key is not None:
        return tool.bytedance_api_key.get_secret_value()
    env_key = os.getenv("BYTEPLUS_API_KEY") or os.getenv("ARK_API_KEY")
    if env_key:
        return env_key
    raise ValueError(
        "BytePlus API key not found. Set BYTEPLUS_API_KEY or ARK_API_KEY, "
        "pass bytedance_api_key, or configure an integration."
    )


async def _resolve_fal_key(tool: Any) -> str:
    if getattr(tool, "fal_integration", None) is not None and isinstance(tool.fal_integration, Integration):
        credentials = await tool.fal_integration.resolve()
        key = credentials.get("api_key")
        if key:
            return str(key)
    if tool.fal_api_key is not None:
        return tool.fal_api_key.get_secret_value()
    env_key = os.getenv("FAL_KEY")
    if env_key:
        return env_key
    raise ValueError("fal API key not found. Set FAL_KEY, pass fal_api_key, or configure a fal integration.")


async def _fetch_image_as_base64(image_url: str) -> str:
    import httpx

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        response = await client.get(image_url)
        response.raise_for_status()
        return base64.b64encode(response.content).decode("utf-8")


def _build_google_payload(
    config: _VideoModelConfig,
    *,
    prompt: str,
    aspect_ratio: str,
    resolution: str,
    duration: int,
    negative_prompt: str | None,
    image_url: str | None,
    seed: int | None,
    provider_params: dict[str, Any] | None,
) -> dict[str, Any]:
    instance: dict[str, Any] = {"prompt": prompt.strip()}
    parameters: dict[str, Any] = dict(config.defaults)
    parameters["aspectRatio"] = aspect_ratio if aspect_ratio in VALID_ASPECT_RATIOS else "16:9"
    parameters["resolution"] = resolution if resolution in VALID_RESOLUTIONS else "720p"
    parameters["durationSeconds"] = duration
    if negative_prompt:
        parameters["negativePrompt"] = negative_prompt
    if seed is not None:
        parameters["seed"] = seed
    if provider_params:
        for key, value in provider_params.items():
            if value is not None:
                if key in ("aspect_ratio",):
                    parameters["aspectRatio"] = value
                elif key in ("duration",):
                    parameters["durationSeconds"] = value
                else:
                    parameters[key] = value

    payload: dict[str, Any] = {"instances": [instance], "parameters": parameters}
    if image_url:
        payload["_image_url"] = image_url  # resolved async before submit
    supports = config.supports
    filtered_params = {k: v for k, v in parameters.items() if k in supports}
    filtered_instance = {k: v for k, v in instance.items() if k in supports or k == "prompt"}
    return {"instances": [filtered_instance], "parameters": filtered_params, "_image_url": image_url}


def _build_bytedance_payload(
    config: _VideoModelConfig,
    *,
    prompt: str,
    aspect_ratio: str,
    resolution: str,
    duration: int,
    generate_audio: bool,
    image_url: str | None,
    seed: int | None,
    provider_params: dict[str, Any] | None,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt.strip()}]
    if image_url:
        content.append({"type": "image_url", "image_url": {"url": image_url}})

    payload: dict[str, Any] = {
        "model": config.model_id,
        "content": content,
        **dict(config.defaults),
        "ratio": aspect_ratio if aspect_ratio in VALID_ASPECT_RATIOS else "16:9",
        "resolution": resolution if resolution in VALID_RESOLUTIONS else "720p",
        "duration": duration,
        "generate_audio": generate_audio,
    }
    if seed is not None:
        payload["seed"] = seed
    if provider_params:
        for key, value in provider_params.items():
            if value is not None:
                payload[key] = value

    supports = config.supports
    return {k: v for k, v in payload.items() if k in supports or k in ("model", "content")}


def _build_fal_payload(
    config: _VideoModelConfig,
    *,
    prompt: str,
    aspect_ratio: str,
    resolution: str,
    duration: int,
    negative_prompt: str | None,
    image_url: str | None,
    generate_audio: bool,
    seed: int | None,
    provider_params: dict[str, Any] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = dict(config.defaults)
    payload["prompt"] = prompt.strip()
    supports = config.supports

    if "aspect_ratio" in supports and aspect_ratio in VALID_ASPECT_RATIOS:
        payload["aspect_ratio"] = aspect_ratio
    if "resolution" in supports and resolution in VALID_RESOLUTIONS:
        payload["resolution"] = resolution
    if "duration" in supports:
        payload["duration"] = _format_fal_duration(config, duration)
    if negative_prompt and "negative_prompt" in supports:
        payload["negative_prompt"] = negative_prompt
    if image_url and config.supports_image_to_video:
        payload[config.fal_image_key] = image_url
    if generate_audio and config.supports_audio and "generate_audio" in supports:
        payload["generate_audio"] = generate_audio
    if seed is not None and "seed" in supports:
        payload["seed"] = seed
    if provider_params:
        for key, value in provider_params.items():
            if value is not None:
                payload[key] = value

    return {k: v for k, v in payload.items() if k in supports or k == "prompt"}


async def _poll_until_done(
    poll_fn: Any,
    *,
    is_done: Any,
    interval: float,
    timeout: float,
) -> Any:
    """Poll ``poll_fn`` until ``is_done`` returns True or timeout elapses."""
    deadline = time.monotonic() + timeout
    last: Any = None
    while time.monotonic() < deadline:
        last = await poll_fn()
        if is_done(last):
            return last
        if interval > 0:
            await asyncio.sleep(interval)
    raise TimeoutError(f"Video generation timed out after {timeout}s")


def _google_generated_samples(data: dict[str, Any]) -> list[dict[str, Any]]:
    response = data.get("response") or {}
    gen = response.get("generateVideoResponse") or {}
    samples = gen.get("generatedSamples") or []
    return [s for s in samples if isinstance(s, dict)]


def _extract_google_video_url(data: dict[str, Any]) -> str | None:
    samples = _google_generated_samples(data)
    if samples:
        video = samples[0].get("video") or {}
        uri = video.get("uri")
        if isinstance(uri, str) and uri:
            return uri
    return None


def _extract_bytedance_tokens(data: dict[str, Any]) -> int | None:
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return None
    total = usage.get("total_tokens")
    if total is None:
        return None
    try:
        return int(total)
    except (TypeError, ValueError):
        return None


def _count_fal_generations(data: dict[str, Any]) -> int:
    videos = data.get("videos")
    if isinstance(videos, list) and videos:
        return len(videos)
    if _extract_fal_video_url(data):
        return 1
    return 0


def _record_video_usage(
    model: str,
    resolution: str,
    duration: int | None,
    generations: int | None,
    tokens: int | None,
) -> None:
    """Record per-model usage quantities for backend Costs(name, unit) lookup."""
    ctx = get_run_context()
    if ctx is None:
        return
    gens = max(int(generations or 1), 1)
    if duration is not None:
        secs = max(int(duration), 0) * gens
        if secs > 0:
            ctx.update_usage(f"{model}:video_seconds_{resolution}", secs)
    ctx.update_usage(f"{model}:generations", gens)
    ctx.update_usage(f"{model}:requests", 1)
    if tokens:
        ctx.update_usage(f"{model}:tokens", int(tokens))


def _extract_bytedance_video_url(data: dict[str, Any]) -> str | None:
    content = data.get("content") or {}
    url = content.get("video_url")
    if isinstance(url, str) and url:
        return url
    return None


def _extract_fal_video_url(data: dict[str, Any]) -> str | None:
    video = data.get("video")
    if isinstance(video, dict):
        url = video.get("url")
        if isinstance(url, str) and url:
            return url
    videos = data.get("videos")
    if isinstance(videos, list) and videos:
        first = videos[0]
        if isinstance(first, dict):
            url = first.get("url")
            if isinstance(url, str) and url:
                return url
        elif isinstance(first, str):
            return first
    for key in ("output", "result"):
        nested = data.get(key)
        if isinstance(nested, dict):
            found = _extract_fal_video_url(nested)
            if found:
                return found
    return None


async def _run_google(
    tool: Any,
    config: _VideoModelConfig,
    payload: dict[str, Any],
    *,
    poll_interval: float,
    timeout: float,
) -> dict[str, Any]:
    import httpx

    api_key = await _resolve_google_key(tool)
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}

    submit_body = {"instances": payload["instances"], "parameters": payload["parameters"]}
    image_url = payload.get("_image_url")
    if image_url and submit_body["instances"]:
        b64 = await _fetch_image_as_base64(image_url)
        submit_body["instances"][0]["image"] = {"bytesBase64Encoded": b64}

    url = f"{_GOOGLE_BASE}/models/{config.model_id}:predictLongRunning"
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, read=None)) as client:
        response = await client.post(url, headers=headers, json=submit_body)
        response.raise_for_status()
        submit_data = response.json()

        operation_name = submit_data.get("name")
        if not isinstance(operation_name, str) or not operation_name:
            raise ValueError(f"Google video API returned no operation name: {submit_data}")

        poll_url = f"{_GOOGLE_BASE}/{operation_name.lstrip('/')}"

        async def _poll() -> dict[str, Any]:
            r = await client.get(poll_url, headers={"x-goog-api-key": api_key})
            r.raise_for_status()
            return r.json()

        def _is_done(data: dict[str, Any]) -> bool:
            if data.get("done"):
                if data.get("error"):
                    raise RuntimeError(f"Google video generation failed: {data['error']}")
                return True
            return False

        result = await _poll_until_done(_poll, is_done=_is_done, interval=poll_interval, timeout=timeout)
        samples = _google_generated_samples(result)
        generations = len(samples) if samples else 1
        video_url = _extract_google_video_url(result)
        if not video_url:
            raise ValueError(f"No video URL in Google response: {result}")
        return {
            "video_url": video_url,
            "raw": result,
            "generations": generations,
            "tokens": None,
        }


async def _run_bytedance(
    tool: Any,
    _config: _VideoModelConfig,
    payload: dict[str, Any],
    *,
    poll_interval: float,
    timeout: float,
) -> dict[str, Any]:
    import httpx

    api_key = await _resolve_bytedance_key(tool)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, read=None)) as client:
        response = await client.post(_BYTEDANCE_BASE, headers=headers, json=payload)
        response.raise_for_status()
        submit_data = response.json()

        task_id = submit_data.get("id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"BytePlus video API returned no task id: {submit_data}")

        poll_url = f"{_BYTEDANCE_BASE}/{task_id}"

        async def _poll() -> dict[str, Any]:
            r = await client.get(poll_url, headers={"Authorization": f"Bearer {api_key}"})
            r.raise_for_status()
            return r.json()

        def _is_done(data: dict[str, Any]) -> bool:
            status = (data.get("status") or "").lower()
            if status in ("failed", "error", "cancelled"):
                raise RuntimeError(f"BytePlus video generation failed: {data}")
            return status == "succeeded"

        result = await _poll_until_done(_poll, is_done=_is_done, interval=poll_interval, timeout=timeout)
        video_url = _extract_bytedance_video_url(result)
        if not video_url:
            raise ValueError(f"No video URL in BytePlus response: {result}")
        return {
            "video_url": video_url,
            "raw": result,
            "generations": 1,
            "tokens": _extract_bytedance_tokens(result),
        }


async def _run_fal(
    tool: Any,
    config: _VideoModelConfig,
    payload: dict[str, Any],
    *,
    poll_interval: float,
    timeout: float,
) -> dict[str, Any]:
    import httpx

    api_key = await _resolve_fal_key(tool)
    headers = {"Authorization": f"Key {api_key}", "Content-Type": "application/json"}
    path = _encode_fal_model_path(config.model_id)
    submit_url = f"{_FAL_QUEUE_BASE}/{path}"

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, read=None)) as client:
        response = await client.post(submit_url, headers=headers, json=payload)
        await _raise_for_fal_response(response)
        submit_data = response.json()

        request_id = submit_data.get("request_id")
        status_url = submit_data.get("status_url")
        response_url = submit_data.get("response_url")
        queue_model_path = _fal_queue_model_path_from_submit(submit_data) or config.model_id

        if not request_id or not status_url:
            raise ValueError(f"fal queue submit returned incomplete response: {submit_data}")

        async def _poll() -> dict[str, Any]:
            r = await client.get(
                status_url,
                headers={"Authorization": f"Key {api_key}"},
            )
            await _raise_for_fal_response(r)
            return r.json()

        def _is_done(data: dict[str, Any]) -> bool:
            status = (data.get("status") or "").upper()
            if status in ("FAILED", "ERROR", "CANCELLED"):
                raise RuntimeError(f"fal video generation failed: {data}")
            return status == "COMPLETED"

        await _poll_until_done(_poll, is_done=_is_done, interval=poll_interval, timeout=timeout)

        result = await _fetch_fal_result(
            client,
            api_key=api_key,
            submit_model_id=config.model_id,
            queue_model_path=queue_model_path,
            request_id=request_id,
            response_url=response_url if isinstance(response_url, str) else None,
        )

        video_url = _extract_fal_video_url(result)
        if not video_url:
            raise ValueError(f"No video URL in fal response: {result}")
        return {
            "video_url": video_url,
            "raw": result,
            "generations": _count_fal_generations(result),
            "tokens": None,
        }


class VideoGenerate(Tool):
    name: str = "video_generate"
    record_default_request_usage: bool = False
    description: str | None = (
        "Generate a video from a text prompt using a model-agnostic interface. "
        "Supported direct models: google/veo-3.1, google/veo-3.1-fast, "
        "bytedance/seedance-2.0, bytedance/seedance-2.0-fast. "
        "Any other model string is routed to fal.ai (e.g. fal-ai/veo3.1, fal-ai/kling-video/v2/master/text-to-video). "
        "Submits the job, polls until complete, and returns video_url. "
        "Optional image_url enables image-to-video when the model supports it."
    )
    google_integration: Annotated[str, Integration("gemini")] | None = None
    bytedance_integration: Annotated[str, Integration("byteplus")] | None = None
    fal_integration: Annotated[str, Integration("fal")] | None = None
    google_api_key: SecretStr | None = None
    bytedance_api_key: SecretStr | None = None
    fal_api_key: SecretStr | None = None

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            **self._annotate_config(
                {
                    "google_integration": self.google_integration,
                    "bytedance_integration": self.bytedance_integration,
                    "fal_integration": self.fal_integration,
                    "google_api_key": self.google_api_key,
                    "bytedance_api_key": self.bytedance_api_key,
                    "fal_api_key": self.fal_api_key,
                }
            ),
        }

    def __init__(self, **kwargs: Any) -> None:
        async def _video_generate(
            prompt: str = Field(..., description="Text description of the video to generate."),
            model: str = Field(
                DEFAULT_MODEL,
                description=(
                    "Unified model id. Direct: google/veo-3.1, google/veo-3.1-fast, "
                    "bytedance/seedance-2.0, bytedance/seedance-2.0-fast. "
                    "Any other value is treated as a fal.ai endpoint id."
                ),
            ),
            aspect_ratio: str = Field(
                "16:9",
                description="Aspect ratio: 16:9, 9:16, or 1:1.",
            ),
            resolution: str = Field("720p", description="Output resolution: 720p, 1080p, or 480p."),
            duration: int = Field(8, description="Target duration in seconds (model limits apply)."),
            negative_prompt: str | None = Field(None, description="Optional negative prompt (Google/fal)."),
            image_url: str | None = Field(
                None,
                description="Optional first-frame image URL for image-to-video when supported.",
            ),
            generate_audio: bool = Field(True, description="Whether to generate synchronized audio when supported."),
            seed: int | None = Field(None, description="Optional random seed for reproducibility."),
            provider_params: dict[str, Any] | None = Field(
                None,
                description="Provider-native parameter overrides merged before whitelist filtering.",
            ),
            poll_interval: float = Field(5.0, description="Seconds between status polls."),
            timeout: float = Field(600.0, description="Max seconds to wait for generation."),
        ) -> dict[str, Any]:
            start = time.monotonic()
            resolved_model = (model or DEFAULT_MODEL).strip()

            try:
                if not prompt or not prompt.strip():
                    raise ValueError("prompt is required and must be non-empty")

                config = _resolve_model_config(resolved_model)

                if image_url and not config.supports_image_to_video:
                    raise ValueError(f"Model '{resolved_model}' does not support image-to-video")

                aspect = aspect_ratio if aspect_ratio in VALID_ASPECT_RATIOS else "16:9"
                res = resolution if resolution in VALID_RESOLUTIONS else "720p"
                billing_duration: int | None = duration

                if config.provider == "google":
                    payload = _build_google_payload(
                        config,
                        prompt=prompt,
                        aspect_ratio=aspect,
                        resolution=res,
                        duration=duration,
                        negative_prompt=negative_prompt,
                        image_url=image_url,
                        seed=seed,
                        provider_params=provider_params,
                    )
                    result = await _run_google(self, config, payload, poll_interval=poll_interval, timeout=timeout)
                elif config.provider == "bytedance":
                    payload = _build_bytedance_payload(
                        config,
                        prompt=prompt,
                        aspect_ratio=aspect,
                        resolution=res,
                        duration=duration,
                        generate_audio=generate_audio,
                        image_url=image_url,
                        seed=seed,
                        provider_params=provider_params,
                    )
                    result = await _run_bytedance(self, config, payload, poll_interval=poll_interval, timeout=timeout)
                else:
                    _validate_fal_constraints(
                        config,
                        aspect_ratio=aspect,
                        resolution=res,
                        duration=duration,
                    )
                    billing_duration = duration if "duration" in config.supports else None
                    payload = _build_fal_payload(
                        config,
                        prompt=prompt,
                        aspect_ratio=aspect,
                        resolution=res,
                        duration=duration,
                        negative_prompt=negative_prompt,
                        image_url=image_url,
                        generate_audio=generate_audio,
                        seed=seed,
                        provider_params=provider_params,
                    )
                    result = await _run_fal(self, config, payload, poll_interval=poll_interval, timeout=timeout)

                _record_video_usage(
                    resolved_model,
                    res,
                    billing_duration,
                    result.get("generations"),
                    result.get("tokens"),
                )

                elapsed = time.monotonic() - start
                logger.info(
                    "Video generated",
                    model=resolved_model,
                    provider=config.provider,
                    elapsed_s=round(elapsed, 2),
                )
                return {
                    "success": True,
                    "provider": config.provider,
                    "model": resolved_model,
                    "video_url": result["video_url"],
                    "raw": result.get("raw"),
                    "elapsed_seconds": round(elapsed, 2),
                }

            except Exception as exc:
                elapsed = time.monotonic() - start
                logger.error("Video generation failed", model=resolved_model, error=str(exc))
                return {
                    "success": False,
                    "provider": _resolve_model_config(resolved_model).provider,
                    "model": resolved_model,
                    "video_url": None,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "elapsed_seconds": round(elapsed, 2),
                }

        super().__init__(handler=_video_generate, **kwargs)


class VideoListModels(Tool):
    name: str = "video_list_models"
    description: str | None = (
        "List direct video generation models from the Timbal catalog "
        "(google/veo-3.1, bytedance/seedance-2.0, etc.) with billing units. "
        "Any other model string passed to video_generate is routed to fal.ai dynamically."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _video_list_models(
            include_billing: bool = Field(
                True,
                description="Include billing.costs rows aligned with Costs(name, unit).",
            ),
        ) -> dict[str, Any]:
            models = list_video_models(include_billing=include_billing)
            return {
                "default_model": get_default_video_model(),
                "direct_models": models,
                "fal_fallback": {
                    "enabled": True,
                    "note": (
                        "Pass any fal.ai endpoint id as model to video_generate "
                        "(e.g. fal-ai/kling-video/v2/master/text-to-video). "
                        "Add Costs rows per fal endpoint separately."
                    ),
                },
            }

        super().__init__(handler=_video_list_models, **kwargs)
