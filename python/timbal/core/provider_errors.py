"""Lazy access to provider-SDK exception classes.

Importing ``openai`` + ``anthropic`` costs ~460 ms — a third of the entire
``from timbal import Agent`` import — and these classes are only needed to
*classify* errors at LLM-call time.

Classification never imports an SDK: an exception can only be an instance of
a class from an SDK that is already in ``sys.modules`` (it raised it), so the
tuples are built from whichever provider SDKs the process has actually used.
An Anthropic-only process therefore never loads ``openai`` at all, and vice
versa. Rebuilt per call (a few attribute lookups, error paths only) so a
fallback chain that touches a second provider later is classified correctly.
"""

import sys


def provider_error_classes() -> dict[str, tuple[type, ...]]:
    """Exception-class tuples for retry/fallback classification.

    Keys: ``rate_limit``, ``timeout``, ``connection``, ``status``. Each value
    contains the classes from the provider SDKs currently imported — possibly
    empty tuples (``isinstance(e, ())`` is False), e.g. for TestModel-only
    processes where no provider SDK ever loads.
    """
    rate_limit: list[type] = []
    timeout: list[type] = []
    connection: list[type] = []
    status: list[type] = []

    openai = sys.modules.get("openai")
    if openai is not None:
        rate_limit.append(openai.RateLimitError)
        timeout.append(openai.APITimeoutError)
        connection.append(openai.APIConnectionError)
        status.append(openai.APIStatusError)

    anthropic = sys.modules.get("anthropic")
    if anthropic is not None:
        rate_limit.append(anthropic.RateLimitError)
        timeout.append(anthropic.APITimeoutError)
        connection.append(anthropic.APIConnectionError)
        status.append(anthropic.APIStatusError)

    return {
        "rate_limit": tuple(rate_limit),
        "timeout": tuple(timeout),
        "connection": tuple(connection),
        "status": tuple(status),
    }
