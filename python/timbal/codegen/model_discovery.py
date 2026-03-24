from pathlib import Path

import yaml

_MODELS_YAML = Path(__file__).parent.parent / "models.yaml"

_PROVIDER_LOGOS = {
    "anthropic": "https://content.timbal.ai/assets/anthropic_favicon.svg",
    "openai": "https://content.timbal.ai/assets/openai_favicon.svg",
    "google": "https://content.timbal.ai/assets/google_favicon.svg",
    "togetherai": "https://content.timbal.ai/assets/togetherai_favicon.svg",
    "xai": "https://content.timbal.ai/assets/xai_favicon.svg",
    "groq": "https://content.timbal.ai/assets/groq_favicon.svg",
    "fireworks": "https://content.timbal.ai/assets/fireworks_favicon.svg",
    "xiaomi": "https://content.timbal.ai/assets/xiaomi_favicon.svg",
    "byteplus": "https://content.timbal.ai/assets/byteplus_favicon.svg",
}


def get_models() -> list[dict]:
    """Return all supported LLM models from models.yaml.

    Each entry contains: id, provider, display_name, description, input_price,
    output_price, context_window, capabilities.
    """
    with _MODELS_YAML.open() as f:
        data = yaml.safe_load(f)
    return data.get("models", [])


def get_provider_summaries() -> list[dict]:
    """Return provider summaries with model counts, sorted by count descending."""
    groups: dict[str, dict] = {}
    for model in get_models():
        provider = model["provider"]
        if provider not in groups:
            groups[provider] = {
                "name": provider,
                "logo": _PROVIDER_LOGOS.get(provider),
                "model_count": 0,
            }
        groups[provider]["model_count"] += 1
    return sorted(groups.values(), key=lambda g: g["model_count"], reverse=True)
