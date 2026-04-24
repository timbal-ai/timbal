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


# ---------------------------------------------------------------------------
# Model type with provider prefixes
Model = Literal[
    "anthropic/claude-opus-4-7",
    "anthropic/claude-opus-4-6",
    "anthropic/claude-opus-4-5",
    "anthropic/claude-opus-4-1",
    "anthropic/claude-opus-4-0",
    "anthropic/claude-sonnet-4-6",
    "anthropic/claude-sonnet-4-5",
    "anthropic/claude-sonnet-4-0",
    "anthropic/claude-haiku-4-5",
    "anthropic/claude-3-7-sonnet-latest",
    "anthropic/claude-3-5-haiku-latest",
    "anthropic/claude-3-opus-latest",
    "anthropic/claude-3-haiku-20240307",
    "openai/gpt-5.5",
    "openai/gpt-5.4",
    "openai/gpt-5.4-mini",
    "openai/gpt-5.4-nano",
    "openai/gpt-5.3-chat-latest",
    "openai/gpt-5.2",
    "openai/gpt-5.2-pro",
    "openai/gpt-5.1",
    "openai/gpt-5.1-codex",
    "openai/gpt-5",
    "openai/gpt-5-mini",
    "openai/gpt-5-nano",
    "openai/gpt-4.1",
    "openai/gpt-4.1-mini",
    "openai/gpt-4.1-nano",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "openai/o4-mini",
    "openai/o4-mini-deep-research",
    "openai/o3",
    "openai/o3-mini",
    "openai/o3-pro",
    "openai/o3-deep-research",
    "openai/o1",
    "openai/o1-mini",
    "togetherai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "togetherai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "togetherai/meta-llama/Llama-3.2-3B-Instruct-Turbo",
    "togetherai/Qwen/Qwen3.5-397B-A17B",
    "togetherai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
    "togetherai/Qwen/Qwen3-235B-A22B-Thinking-2507",
    "togetherai/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
    "togetherai/Qwen/Qwen3-Coder-Next-FP8",
    "togetherai/Qwen/Qwen3-Next-80B-A3B-Instruct",
    "togetherai/Qwen/Qwen2.5-7B-Instruct-Turbo",
    "togetherai/deepseek-ai/DeepSeek-V3.1",
    "togetherai/deepseek-ai/DeepSeek-R1",
    "togetherai/moonshotai/Kimi-K2.5",
    "togetherai/moonshotai/Kimi-K2-Instruct-0905",
    "togetherai/moonshotai/Kimi-K2-Thinking",
    "togetherai/MiniMaxAI/MiniMax-M2.5",
    "togetherai/zai-org/GLM-5",
    "togetherai/zai-org/GLM-4.7",
    "togetherai/openai/gpt-oss-120b",
    "togetherai/google/gemma-3n-E4B-it",
    "togetherai/google/gemma-3-27b-it",
    "togetherai/deepcogito/cogito-v2-1-671b",
    "togetherai/mistralai/Mistral-Small-24B-Instruct-2501",
    "google/gemini-3.1-pro-preview",
    "google/gemini-3.1-flash-lite-preview",
    "google/gemini-3-flash-preview",
    "google/gemini-2.5-pro",
    "google/gemini-2.5-pro-preview-tts",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-flash-lite",
    "google/gemini-2.5-flash-native-audio-preview-12-2025",
    "google/gemini-2.5-flash-image",
    "google/gemini-2.5-flash-preview-tts",
    "xai/grok-4",
    "xai/grok-4-fast-reasoning",
    "xai/grok-4-fast-non-reasoning",
    "xai/grok-4-1-fast-reasoning",
    "xai/grok-4-1-fast-non-reasoning",
    "groq/meta-llama/llama-4-scout-17b-16e-instruct",
    "groq/llama-3.3-70b-versatile",
    "groq/llama-3.1-8b-instant",
    "groq/qwen/qwen3-32b",
    "groq/openai/gpt-oss-120b",
    "groq/openai/gpt-oss-20b",
    "groq/moonshotai/kimi-k2-instruct-0905",
    "fireworks/accounts/fireworks/models/llama4-maverick-instruct-basic",
    "fireworks/accounts/fireworks/models/llama4-scout-instruct-basic",
    "fireworks/accounts/fireworks/models/llama-v3p3-70b-instruct",
    "fireworks/accounts/fireworks/models/llama-v3p1-405b-instruct",
    "fireworks/accounts/fireworks/models/llama-v3p1-70b-instruct",
    "fireworks/accounts/fireworks/models/llama-v3p1-8b-instruct",
    "fireworks/accounts/fireworks/models/qwen3-coder-480b-a35b-instruct",
    "fireworks/accounts/fireworks/models/qwen3-235b-a22b",
    "fireworks/accounts/fireworks/models/qwen3-32b",
    "fireworks/accounts/fireworks/models/qwen3-8b",
    "fireworks/accounts/fireworks/models/qwen2p5-72b-instruct",
    "fireworks/accounts/fireworks/models/deepseek-v3p2",
    "fireworks/accounts/fireworks/models/deepseek-v3p1",
    "fireworks/accounts/fireworks/models/deepseek-r1",
    "fireworks/accounts/fireworks/models/deepseek-r1-0528",
    "fireworks/accounts/fireworks/models/deepseek-r1-distill-llama-70b",
    "fireworks/accounts/fireworks/models/kimi-k2p5",
    "fireworks/accounts/fireworks/models/kimi-k2-instruct-0905",
    "fireworks/accounts/fireworks/models/kimi-k2-thinking",
    "fireworks/accounts/fireworks/models/minimax-m2p5",
    "fireworks/accounts/fireworks/models/gpt-oss-120b",
    "fireworks/accounts/fireworks/models/gpt-oss-20b",
    "fireworks/accounts/fireworks/models/mistral-large-3-fp8",
    "fireworks/accounts/fireworks/models/mistral-small-24b-instruct-2501",
    "fireworks/accounts/fireworks/models/gemma-3-27b-it",
    "fireworks/accounts/fireworks/models/glm-5",
    "fireworks/accounts/fireworks/models/glm-4p5",
    "fireworks/accounts/fireworks/models/qwq-32b",
    "xiaomi/mimo-v2-pro",
    "xiaomi/mimo-v2-omni",
    "xiaomi/mimo-v2-flash",
    "byteplus/seed-2-0-pro-260215",
    "byteplus/seed-2-0-lite-260228",
    "byteplus/seed-2-0-mini-260215",
    "byteplus/seed-1-8-251228",
    "byteplus/seed-1-6-250915",
    "cerebras/llama3.1-8b",
    "cerebras/gpt-oss-120b",
    "cerebras/qwen-3-235b-a22b-instruct-2507",
    "cerebras/zai-glm-4.7",
    "sambanova/DeepSeek-R1-0528",
    "sambanova/DeepSeek-V3-0324",
    "sambanova/DeepSeek-V3.1",
    "sambanova/DeepSeek-V3.1-cb",
    "sambanova/DeepSeek-V3.1-Terminus",
    "sambanova/DeepSeek-V3.2",
    "sambanova/Meta-Llama-3.3-70B-Instruct",
    "sambanova/Llama-3.3-Swallow-70B-Instruct-v0.4",
    "sambanova/Llama-4-Maverick-17B-128E-Instruct",
    "sambanova/Meta-Llama-3.1-8B-Instruct",
    "sambanova/Qwen3-235B-A22B-Instruct-2507",
    "sambanova/Qwen3-32B",
    "sambanova/MiniMax-M2.5",
    "sambanova/gpt-oss-120b",
]
