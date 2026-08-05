"""
INTERNAL USE ONLY

This package is intended for internal use and will be subject to frequent
changes as LLM providers constantly update their APIs and add new features.
The external APIs (Runnables, Agents, Workflows) will remain stable, but this
package will evolve to match provider changes.

Do not rely on this package's interface in external code.

Layout:

- ``router`` — model-string dispatch (``_llm_router``)
- ``registry`` — static per-provider configuration (``_PROVIDERS``)
- ``clients`` — SDK client cache/resolution, platform proxy fallback, warmup
- ``retry`` — retry loop for transient provider failures

One module per provider API shape (not per provider; Chat Completions alone
serves a dozen providers):

- ``messages`` — Anthropic Messages API
- ``responses`` — OpenAI Responses API (openai, xai)
- ``chat_completions`` — OpenAI Chat Completions and compatible providers

Each API module builds the request kwargs once per request and returns a
``(create_stream, context_label)`` pair consumed by the router's single retry
loop, so the split adds zero per-chunk overhead.

Provider SDKs (openai, anthropic) are imported lazily — at client resolution
and error classification, never at package import. They account for ~460ms
(a third) of ``from timbal import Agent`` otherwise.
"""

from .clients import _CLIENT_CACHE, _get_client, _resolve_client, warmup_llm_connection
from .registry import _PROVIDERS, _ProviderConfig
from .retry import _retry_on_error
from .router import _llm_router

__all__ = [
    "_CLIENT_CACHE",
    "_PROVIDERS",
    "_ProviderConfig",
    "_get_client",
    "_llm_router",
    "_resolve_client",
    "_retry_on_error",
    "warmup_llm_connection",
]
