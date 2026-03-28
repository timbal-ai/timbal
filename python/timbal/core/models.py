"""Utility for loading model metadata from models.yaml."""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


@lru_cache(maxsize=1)
def _load_models() -> dict[str, dict[str, Any]]:
    """Load models.yaml and return a dict keyed by model id."""
    models_path = Path(__file__).parent.parent / "models.yaml"
    if not models_path.exists():
        return {}
    with open(models_path) as f:
        data = yaml.safe_load(f)
    return {m["id"]: m for m in data.get("models", [])}


def get_context_window(model_id: str) -> int | None:
    """Get the context window size (in tokens) for a model.

    Args:
        model_id: Model identifier (e.g., 'openai/gpt-4.1-nano').

    Returns:
        Context window in tokens, or None if unknown.
    """
    models = _load_models()
    model = models.get(model_id)
    if model is None:
        return None
    return model.get("context_window")
