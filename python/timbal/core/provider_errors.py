"""Lazy access to provider-SDK exception classes.

Importing ``openai`` + ``anthropic`` costs ~460 ms — a third of the entire
``from timbal import Agent`` import — and these classes are only needed to
*classify* errors at LLM-call time, by which point the active provider's SDK
is imported anyway. Built once on first use and cached.
"""

_classes: dict[str, tuple[type, ...]] | None = None


def provider_error_classes() -> dict[str, tuple[type, ...]]:
    """Exception-class tuples for retry/fallback classification.

    Keys: ``rate_limit``, ``timeout``, ``connection``, ``status``. Each value
    holds the corresponding (OpenAI, Anthropic) SDK exception classes.
    """
    global _classes
    if _classes is None:
        from anthropic import APIConnectionError as AnthropicAPIConnectionError
        from anthropic import APIStatusError as AnthropicAPIStatusError
        from anthropic import APITimeoutError as AnthropicAPITimeoutError
        from anthropic import RateLimitError as AnthropicRateLimitError
        from openai import APIConnectionError as OpenAIAPIConnectionError
        from openai import APIStatusError as OpenAIAPIStatusError
        from openai import APITimeoutError as OpenAIAPITimeoutError
        from openai import RateLimitError as OpenAIRateLimitError

        _classes = {
            "rate_limit": (OpenAIRateLimitError, AnthropicRateLimitError),
            "timeout": (OpenAIAPITimeoutError, AnthropicAPITimeoutError),
            "connection": (OpenAIAPIConnectionError, AnthropicAPIConnectionError),
            "status": (OpenAIAPIStatusError, AnthropicAPIStatusError),
        }
    return _classes
