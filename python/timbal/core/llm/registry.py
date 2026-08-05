"""Provider registry — static per-provider configuration.

Adding an OpenAI-compatible provider is one ``_PROVIDERS`` entry here;
adding a new API shape is a new module (see package docstring).
"""

import os
from dataclasses import dataclass
from typing import Literal

import structlog

logger = structlog.get_logger("timbal.core.llm")

TIMBAL_OPENAI_API = os.getenv("TIMBAL_OPENAI_API", "responses")
if TIMBAL_OPENAI_API != "responses":
    logger.warning(
        "Using legacy Chat Completions API. OpenAI is transitioning to the new Responses API, "
        "which should be preferred for all new development. Set TIMBAL_OPENAI_API=responses to switch."
    )


@dataclass(frozen=True, slots=True)
class _ProviderConfig:
    """Static configuration for a single LLM provider."""

    env_key: str
    """Environment variable name for the API key (e.g. ``OPENAI_API_KEY``)."""

    default_base_url: str | None = None
    """Default API base URL.  ``None`` uses the SDK default."""

    proxy_name: str = "openai-completions"
    """Platform proxy path segment (e.g. ``openai-responses``, ``anthropic``)."""

    proxy_suffix: str = "/v1"
    """Appended to the proxy URL.  Anthropic uses ``""``."""

    client_type: Literal["openai", "anthropic"] = "openai"
    """Which SDK client to create."""

    flatten_text_content: bool = False
    """Flatten text-only content arrays to plain strings for providers with incomplete Chat Completions support."""

    supports_stream_options: bool = True
    """Whether the provider supports ``stream_options`` in Chat Completions."""

    supports_platform_proxy: bool = True
    """If False, never fall back to the Timbal platform proxy — require the provider API key."""

    supports_chat_reasoning_content: bool = False
    """If True, serialize ThinkingContent as top-level ``reasoning_content`` (Moonshot/Fireworks-style).

    Otherwise thinking is omitted from outbound chat-completions messages (Vercel/LiteLLM default).
    """


_PROVIDERS: dict[str, _ProviderConfig] = {
    "openai": _ProviderConfig(
        env_key="OPENAI_API_KEY",
        proxy_name="openai-responses" if TIMBAL_OPENAI_API == "responses" else "openai-completions",
    ),
    "anthropic": _ProviderConfig(
        env_key="ANTHROPIC_API_KEY",
        proxy_name="anthropic",
        proxy_suffix="",
        client_type="anthropic",
    ),
    "google": _ProviderConfig(
        env_key="GEMINI_API_KEY",
        default_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    ),
    "togetherai": _ProviderConfig(
        env_key="TOGETHER_API_KEY",
        default_base_url="https://api.together.xyz/v1/",
        supports_chat_reasoning_content=True,
    ),
    "xai": _ProviderConfig(
        env_key="XAI_API_KEY",
        default_base_url="https://api.x.ai/v1",
        proxy_name="openai-responses",
    ),
    "groq": _ProviderConfig(
        env_key="GROQ_API_KEY",
        default_base_url="https://api.groq.com/openai/v1",
    ),
    "fireworks": _ProviderConfig(
        env_key="FIREWORKS_API_KEY",
        default_base_url="https://api.fireworks.ai/inference/v1",
        supports_chat_reasoning_content=True,
    ),
    "byteplus": _ProviderConfig(
        env_key="BYTEPLUS_API_KEY",
        default_base_url="https://ark.ap-southeast.bytepluses.com/api/v3",
        supports_chat_reasoning_content=True,
    ),
    "xiaomi": _ProviderConfig(
        env_key="XIAOMI_API_KEY",
        default_base_url="https://api.xiaomimimo.com/v1",
        flatten_text_content=True,
        supports_stream_options=False,
    ),
    "cerebras": _ProviderConfig(
        env_key="CEREBRAS_API_KEY",
        default_base_url="https://api.cerebras.ai/v1",
    ),
    "moonshot": _ProviderConfig(
        env_key="MOONSHOT_API_KEY",
        default_base_url="https://api.moonshot.ai/v1",
        supports_platform_proxy=False,
        supports_chat_reasoning_content=True,
    ),
    "sambanova": _ProviderConfig(
        env_key="SAMBANOVA_API_KEY",
        default_base_url="https://api.sambanova.ai/v1",
        flatten_text_content=True,
    ),
}
