"""Model metadata and type definitions.

This module is the single source of truth for supported LLM models.
The ``Model`` Literal type is auto-generated from ``models.yaml`` by
``scripts/generate_models.py``.  ``get_context_window`` provides a
runtime lookup into the same YAML for context-window sizes.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml

# Suffixes collectors append to token usage units for non-standard pricing.
LONG_CONTEXT_USAGE_SUFFIX = "_long_context"
FAST_USAGE_SUFFIX = "_fast"
FLEX_USAGE_SUFFIX = "_flex"
ULTRAFAST_USAGE_SUFFIX = "_ultrafast"
_PRICING_USAGE_SUFFIXES = (ULTRAFAST_USAGE_SUFFIX, FAST_USAGE_SUFFIX, FLEX_USAGE_SUFFIX, LONG_CONTEXT_USAGE_SUFFIX)


def base_usage_metric(metric: str) -> str:
    """Strip all pricing-tier suffixes from a usage metric name.

    ``output_text_tokens_long_context_fast`` -> ``output_text_tokens``. Use this wherever
    usage is aggregated for display or assertions rather than billing.
    """
    changed = True
    while changed:
        changed = False
        for suffix in _PRICING_USAGE_SUFFIXES:
            if metric.endswith(suffix):
                metric = metric[: -len(suffix)]
                changed = True
                break
    return metric


@lru_cache(maxsize=1)
def _load_models() -> dict[str, dict[str, Any]]:
    """Load models.yaml and return a dict keyed by model id."""
    models_path = Path(__file__).parent.parent / "models.yaml"
    if not models_path.exists():
        return {}
    with open(models_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {m["id"]: m for m in data.get("models", [])}


def get_context_window(model_id: str) -> int | None:
    """Get the context window size (in tokens) for a model.

    Args:
        model_id: Model identifier (e.g., 'openai/gpt-5.4-nano').

    Returns:
        Context window in tokens, or None if unknown.
    """
    models = _load_models()
    model = models.get(model_id)
    if model is None:
        return None
    return model.get("context_window")


def get_long_context_threshold(model_id: str) -> int | None:
    """Get the input-token threshold above which a model bills the full request at long-context rates.

    Providers such as OpenAI (>272K on 1.05M-context models) and xAI (>=200K) reprice
    the *entire* request — not just the overflow — at a model-specific boundary.

    Args:
        model_id: Model identifier (e.g., 'openai/gpt-6-astra').

    Returns:
        Threshold in input tokens, or None if the model has no long-context tier (or is unknown).
    """
    models = _load_models()
    model = models.get(model_id)
    if model is None:
        return None
    long_context = model.get("long_context")
    if not isinstance(long_context, dict):
        return None
    threshold = long_context.get("threshold")
    return int(threshold) if threshold is not None else None


def uses_long_context_pricing(model_id: str, input_tokens: int) -> bool:
    """Whether this prompt falls in the model's long-context pricing tier."""
    model = _load_models().get(model_id)
    if model is None:
        return False
    long_context = model.get("long_context")
    if not isinstance(long_context, dict):
        return False
    threshold = long_context.get("threshold")
    if threshold is None:
        return False
    if long_context.get("inclusive", False):
        return input_tokens >= int(threshold)
    return input_tokens > int(threshold)


def service_tier_usage_suffix(model_id: str, service_tier: str | None) -> str:
    """Return the priced usage suffix for the actual provider service tier."""
    if not service_tier:
        return ""
    tier = "fast" if service_tier == "priority" else service_tier
    model = _load_models().get(model_id)
    tiers = model.get("service_tiers") if model is not None else None
    if not isinstance(tiers, dict) or tier not in tiers:
        return ""
    if tier == "fast":
        return FAST_USAGE_SUFFIX
    if tier == "flex":
        return FLEX_USAGE_SUFFIX
    if tier == "ultrafast":
        return ULTRAFAST_USAGE_SUFFIX
    return ""


def has_cache_write_pricing(model_id: str) -> bool:
    """Whether the catalog prices prompt-cache writes separately for a model.

    Collectors only split ``cache_write_tokens`` into their own usage unit when this is
    True; otherwise those tokens stay in ``input_text_tokens`` (billed at the input rate)
    rather than landing in a unit no cost table can price.
    """
    models = _load_models()
    model = models.get(model_id)
    if model is None:
        return False
    return model.get("cache_write_price") is not None


# ---------------------------------------------------------------------------
# Model type with provider prefixes
Model = Literal[
    "anthropic/claude-fable-5-1",
    "anthropic/claude-fable-5",
    "anthropic/claude-opus-5-5",
    "anthropic/claude-opus-5",
    "anthropic/claude-opus-4-8",
    "anthropic/claude-sonnet-5-5",
    "anthropic/claude-sonnet-5",
    "anthropic/claude-opus-4-7",
    "anthropic/claude-opus-4-6",
    "anthropic/claude-opus-4-5",
    "anthropic/claude-sonnet-4-6",
    "anthropic/claude-sonnet-4-5",
    "anthropic/claude-haiku-5-5",
    "anthropic/claude-haiku-4-5",
    "openai/gpt-6-astra",
    "openai/gpt-6.1-sol",
    "openai/gpt-6-sol",
    "openai/gpt-6-luna",
    "openai/gpt-5.5",
    "openai/gpt-5.5-pro",
    "openai/gpt-5.6-sol",
    "openai/gpt-5.6-terra",
    "openai/gpt-5.6-luna",
    "openai/gpt-5.4",
    "openai/gpt-5.4-pro",
    "openai/gpt-5.4-mini",
    "openai/gpt-5.4-nano",
    "openai/gpt-5.2",
    "openai/gpt-5.2-pro",
    "openai/gpt-5.1",
    "openai/gpt-5",
    "openai/gpt-5-mini",
    "openai/gpt-5-nano",
    "openai/gpt-4.1",
    "openai/gpt-4.1-mini",
    "openai/gpt-4.1-nano",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "openai/o4-mini",
    "openai/o3",
    "openai/o3-mini",
    "openai/o3-pro",
    "openai/o1",
    "openai/gpt-5.5-2026-04-23",
    "togetherai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "togetherai/Qwen/Qwen3.5-397B-A17B",
    "togetherai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
    "togetherai/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
    "togetherai/Qwen/Qwen3-Coder-Next-FP8",
    "togetherai/Qwen/Qwen3-Next-80B-A3B-Instruct",
    "togetherai/Qwen/Qwen2.5-7B-Instruct-Turbo",
    "togetherai/deepseek-ai/DeepSeek-V3.1",
    "togetherai/deepseek-ai/DeepSeek-V4-Pro",
    "togetherai/moonshotai/Kimi-K2.6",
    "togetherai/moonshotai/Kimi-K2.7-Code",
    "togetherai/MiniMaxAI/MiniMax-M2.7",
    "togetherai/MiniMaxAI/MiniMax-M3",
    "togetherai/zai-org/GLM-5.1",
    "togetherai/zai-org/GLM-5.2",
    "togetherai/openai/gpt-oss-120b",
    "togetherai/openai/gpt-oss-20b",
    "togetherai/moonshotai/Kimi-K3",
    "togetherai/deepseek-ai/DeepSeek-V4-Pro-0813",
    "togetherai/deepseek-ai/DeepSeek-V4-Flash-0731",
    "togetherai/deepseek-ai/DeepSeek-V4.1-Flash",
    "togetherai/zai-org/GLM-5.3",
    "togetherai/zai-org/GLM-5.3-Flash",
    "togetherai/thinkingmachines/Inkling",
    "togetherai/meta-models/Muse-Glimmer-30B",
    "togetherai/Qwen/Qwen3.5-9B",
    "togetherai/Qwen/Qwen3.8-2.4T-A95B",
    "google/gemini-3.8-flash",
    "google/gemini-3.7-flash",
    "google/gemini-3.6-flash",
    "google/gemini-3.5-flash",
    "google/gemini-3.5-flash-lite",
    "google/gemini-3.1-pro-preview",
    "google/gemini-3.1-pro-preview-customtools",
    "google/gemini-3.1-flash-lite",
    "google/gemini-3-flash-preview",
    "google/gemini-2.5-pro",
    "google/gemini-2.5-pro-preview-tts",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-flash-lite",
    "google/gemini-2.5-flash-image",
    "google/gemini-2.5-flash-preview-tts",
    "xai/grok-build-0.1",
    "xai/grok-4.7",
    "xai/grok-4.6",
    "xai/grok-4.5",
    "xai/grok-4.3",
    "groq/qwen/qwen3.8-27b",
    "groq/openai/gpt-oss-120b",
    "groq/openai/gpt-oss-20b",
    "fireworks/accounts/fireworks/models/deepseek-v4-flash-0731",
    "fireworks/accounts/fireworks/models/qwen3p8-max",
    "fireworks/accounts/fireworks/models/kimi-k2p6",
    "fireworks/accounts/fireworks/models/kimi-k2p7-code",
    "fireworks/accounts/fireworks/models/minimax-m3",
    "fireworks/accounts/fireworks/models/gpt-oss-120b",
    "fireworks/accounts/fireworks/models/glm-5p2",
    "fireworks/accounts/fireworks/models/deepseek-v4p1-flash",
    "fireworks/accounts/fireworks/models/kimi-k3",
    "fireworks/accounts/fireworks/models/glm-5p3",
    "fireworks/accounts/fireworks/models/glm-5p3-flash",
    "fireworks/accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b",
    "fireworks/accounts/fireworks/models/nemotron-3-ultra-nvfp4",
    "fireworks/accounts/fireworks/models/ember-1",
    "xiaomi/mimo-v2.5",
    "xiaomi/mimo-v2.5-pro",
    "byteplus/dola-seed-2-1-turbo-260628",
    "byteplus/seed-2-0-lite-260428",
    "byteplus/seed-2-0-mini-260428",
    "byteplus/deepseek-v4-pro-ga-260813",
    "byteplus/deepseek-v4-flash-ga-260731",
    "byteplus/glm-5-2-260617",
    "byteplus/glm-5-3-flash-260828",
    "byteplus/deepseek-v4-1-flash-260910",
    "byteplus/seed-2-0-lite-260228",
    "byteplus/seed-2-0-mini-260215",
    "byteplus/seed-1-8-251228",
    "byteplus/seed-1-6-250915",
    "byteplus/seed-2-0-pro-260328",
    "byteplus/deepseek-v4-pro-260425",
    "byteplus/deepseek-v4-flash-260425",
    "byteplus/deepseek-v3-2-251201",
    "byteplus/gpt-oss-120b-250805",
    "byteplus/glm-4-7-251222",
    "byteplus/seed-2-0-code-preview-260328",
    "cerebras/gpt-oss-120b",
    "cerebras/qwen-3.8-27b",
    "sambanova/DeepSeek-V3.1",
    "sambanova/DeepSeek-V3.2",
    "sambanova/Meta-Llama-3.3-70B-Instruct",
    "sambanova/gpt-oss-120b",
    "sambanova/MiniMax-M2.7",
    "sambanova/gemma-4-31B-it",
    "sambanova/MiniMax-M3",
    "moonshot/kimi-k3",
    "moonshot/kimi-k2.7-code",
    "moonshot/kimi-k2.7-code-highspeed",
    "moonshot/kimi-k2.6",
]
