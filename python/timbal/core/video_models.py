"""Video model metadata and type definitions.

The ``VideoModel`` Literal is auto-generated from ``video_models.yaml`` by
``scripts/generate_video_models.py``. Runtime helpers load the same YAML for
catalog discovery and Costs(name, unit) alignment.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml

_VIDEO_MODELS_YAML = Path(__file__).parent.parent / "video_models.yaml"


@lru_cache(maxsize=1)
def _load_video_models_yaml() -> dict[str, Any]:
    if not _VIDEO_MODELS_YAML.exists():
        return {"default": "google/veo-3.1", "models": []}
    with _VIDEO_MODELS_YAML.open() as f:
        data = yaml.safe_load(f) or {}
    if "models" not in data:
        data["models"] = []
    return data


@lru_cache(maxsize=1)
def _models_by_id() -> dict[str, dict[str, Any]]:
    data = _load_video_models_yaml()
    return {m["id"]: m for m in data.get("models", []) if m.get("id")}


def get_default_video_model() -> str:
    """Return the default unified video model id."""
    data = _load_video_models_yaml()
    default = data.get("default")
    if isinstance(default, str) and default.strip():
        return default.strip()
    models = data.get("models") or []
    if models and isinstance(models[0], dict):
        first_id = models[0].get("id")
        if isinstance(first_id, str):
            return first_id
    return "google/veo-3.1"


def get_video_model(model_id: str) -> dict[str, Any] | None:
    """Return YAML metadata for a direct video model, or None if unknown."""
    return _models_by_id().get(model_id)


def list_video_models(*, include_billing: bool = True, include_runtime: bool = False) -> list[dict[str, Any]]:
    """Return catalog entries for all direct video models from video_models.yaml."""
    entries: list[dict[str, Any]] = []
    for model in _load_video_models_yaml().get("models", []):
        if not isinstance(model, dict) or not model.get("id"):
            continue
        item: dict[str, Any] = {
            "id": model["id"],
            "provider": model.get("provider"),
            "model_id": model.get("model_id"),
            "display_name": model.get("display_name"),
            "description": model.get("description"),
            "supports_audio": model.get("supports_audio", False),
            "supports_image_to_video": model.get("supports_image_to_video", True),
            "allowed_aspect_ratios": model.get("allowed_aspect_ratios"),
            "allowed_resolutions": model.get("allowed_resolutions"),
            "defaults": model.get("defaults") or {},
        }
        if include_runtime:
            item["supports"] = model.get("supports") or []
            item["duration_format"] = model.get("duration_format")
            item["fal_image_key"] = model.get("fal_image_key")
            item["allowed_durations"] = model.get("allowed_durations")
        if include_billing:
            billing = model.get("billing") or {}
            item["billing"] = {
                "primary_unit": billing.get("primary_unit"),
                "costs": billing.get("costs") or [],
            }
        entries.append(item)
    return entries


def list_video_cost_rows() -> list[dict[str, Any]]:
    """Flatten billing.costs from video_models.yaml into Costs(name, unit) rows."""
    rows: list[dict[str, Any]] = []
    for model in list_video_models(include_billing=True):
        model_id = model["id"]
        billing = model.get("billing") or {}
        for cost in billing.get("costs") or []:
            if not isinstance(cost, dict):
                continue
            unit = cost.get("unit")
            if not unit:
                continue
            rows.append(
                {
                    "name": model_id,
                    "unit": unit,
                    "rate": cost.get("rate"),
                    "description": cost.get("description"),
                    "currency": "USD",
                }
            )
    return rows


# ---------------------------------------------------------------------------
# VideoModel type — direct models only (fal endpoints are dynamic)
VideoModel = Literal[
    "google/veo-3.1",
    "google/veo-3.1-fast",
    "bytedance/seedance-2.0",
    "bytedance/seedance-2.0-fast",
]
