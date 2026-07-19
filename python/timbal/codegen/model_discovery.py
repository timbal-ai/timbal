from pathlib import Path

import yaml

_MODELS_YAML = Path(__file__).parent.parent / "models.yaml"

_PROVIDER_LOGOS = {
    "anthropic": "https://timbalusercontent.com/assets/anthropic_favicon.svg",
    "openai": "https://timbalusercontent.com/assets/openai_favicon.svg",
    "google": "https://timbalusercontent.com/assets/google_favicon.svg",
    "togetherai": "https://timbalusercontent.com/assets/togetherai_favicon.svg",
    "xai": "https://timbalusercontent.com/assets/xai_favicon.svg",
    "groq": "https://timbalusercontent.com/assets/groq_favicon.svg",
    "fireworks": "https://timbalusercontent.com/assets/fireworks_favicon.svg",
    "xiaomi": "https://timbalusercontent.com/assets/xiaomi_favicon.svg",
    "byteplus": "https://timbalusercontent.com/assets/byteplus_favicon.svg",
    "cerebras": "https://timbalusercontent.com/assets/cerebras_favicon.svg",
    "sambanova": "https://timbalusercontent.com/assets/sambanova_favicon.svg",
    "moonshot": "https://timbalusercontent.com/assets/moonshot_favicon.svg",
}


def get_models() -> list[dict]:
    """Return all supported LLM models from models.yaml.

    Each entry contains: id, provider, display_name, description, input_price,
    output_price, context_window, capabilities, and optional availability fields
    (requires_activation, dedicated_only, notes).
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
