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
        model_id: Model identifier (e.g., 'openai/gpt-5.4-nano').

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
    "anthropic/claude-fable-5",
    "anthropic/claude-opus-4-8",
    "anthropic/claude-sonnet-5",
    "anthropic/claude-opus-4-7",
    "anthropic/claude-opus-4-6",
    "anthropic/claude-opus-4-5",
    "anthropic/claude-opus-4-1",
    "anthropic/claude-sonnet-4-6",
    "anthropic/claude-sonnet-4-5",
    "anthropic/claude-haiku-4-5",
    "openai/gpt-5.5",
    "openai/gpt-5.5-pro",
    "openai/gpt-5.6-sol",
    "openai/gpt-5.6-terra",
    "openai/gpt-5.6-luna",
    "openai/gpt-5.4",
    "openai/gpt-5.4-pro",
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
    "togetherai/zai-org/GLM-5",
    "togetherai/zai-org/GLM-5.1",
    "togetherai/zai-org/GLM-5.2",
    "togetherai/zai-org/GLM-4.7",
    "togetherai/openai/gpt-oss-120b",
    "togetherai/openai/gpt-oss-20b",
    "togetherai/google/gemma-3n-E4B-it",
    "togetherai/google/gemma-3-27b-it",
    "togetherai/deepcogito/cogito-v2-1-671b",
    "togetherai/mistralai/Mistral-Small-24B-Instruct-2501",
    "google/gemini-3.6-flash",
    "google/gemini-3.5-flash",
    "google/gemini-3.5-flash-lite",
    "google/gemini-3.1-pro-preview",
    "google/gemini-3.1-flash-lite",
    "google/gemini-3-flash-preview",
    "google/gemini-2.5-pro",
    "google/gemini-2.5-pro-preview-tts",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-flash-lite",
    "google/gemini-2.5-flash-image",
    "google/gemini-2.5-flash-preview-tts",
    "xai/grok-4.5",
    "xai/grok-4.3",
    "groq/llama-3.3-70b-versatile",
    "groq/llama-3.1-8b-instant",
    "groq/qwen/qwen3.6-27b",
    "groq/openai/gpt-oss-120b",
    "groq/openai/gpt-oss-20b",
    "fireworks/accounts/fireworks/models/deepseek-v4-pro",
    "fireworks/accounts/fireworks/models/deepseek-v4-flash",
    "fireworks/accounts/fireworks/models/qwen3p6-plus",
    "fireworks/accounts/fireworks/models/qwen3p7-plus",
    "fireworks/accounts/fireworks/models/kimi-k2p6",
    "fireworks/accounts/fireworks/models/kimi-k2p7-code",
    "fireworks/accounts/fireworks/models/kimi-k2p5",
    "fireworks/accounts/fireworks/models/minimax-m2p5",
    "fireworks/accounts/fireworks/models/minimax-m2p7",
    "fireworks/accounts/fireworks/models/minimax-m3",
    "fireworks/accounts/fireworks/models/gpt-oss-120b",
    "fireworks/accounts/fireworks/models/gpt-oss-20b",
    "fireworks/accounts/fireworks/models/glm-5p1",
    "fireworks/accounts/fireworks/models/glm-5p2",
    "xiaomi/mimo-v2.5",
    "xiaomi/mimo-v2.5-pro",
    "byteplus/seed-2-0-lite-260228",
    "byteplus/seed-2-0-mini-260215",
    "byteplus/seed-1-8-251228",
    "byteplus/seed-1-6-250915",
    "byteplus/seed-2-0-pro-260328",
    "byteplus/kimi-k2-250905",
    "byteplus/kimi-k2-thinking-251104",
    "byteplus/deepseek-v4-pro-260425",
    "byteplus/deepseek-v4-flash-260425",
    "byteplus/deepseek-v3-2-251201",
    "byteplus/deepseek-r1-250528",
    "byteplus/gpt-oss-120b-250805",
    "byteplus/glm-4-7-251222",
    "byteplus/seed-2-0-code-preview-260328",
    "cerebras/gpt-oss-120b",
    "cerebras/zai-glm-4.7",
    "sambanova/DeepSeek-V3.1",
    "sambanova/DeepSeek-V3.2",
    "sambanova/Meta-Llama-3.3-70B-Instruct",
    "sambanova/Llama-4-Maverick-17B-128E-Instruct",
    "sambanova/gpt-oss-120b",
    "sambanova/MiniMax-M2.7",
    "sambanova/gemma-3-12b-it",
    "sambanova/gemma-4-31B-it",
    "moonshot/kimi-k3",
    "moonshot/kimi-k2.7-code",
    "moonshot/kimi-k2.7-code-highspeed",
    "moonshot/kimi-k2.6",
    "moonshot/kimi-k2.5",
]
