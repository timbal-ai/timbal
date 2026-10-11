"""``timbal/auto``: let the platform pick the model per turn.

Nothing to configure: ``Agent(model="timbal/auto")`` behaves like any other
model string. The platform looks at the conversation and at which files are
attached to the latest user message. Documents are sent upstream as extracted
text, so the SDK names the attachments in ``metadata.timbal_attachments`` on
the chat-completions request; the platform removes ``timbal_*`` keys before
the request reaches a provider and forwards any other ``metadata`` untouched.
"""

from __future__ import annotations

from ...state import get_billing_id, get_call_id, get_run_context, set_billing_id
from ...types.content import FileContent
from ...types.message import Message

AUTO_PROVIDER = "timbal"
"""Provider prefix whose requests carry routing metadata (``timbal/auto``)."""

_AUTO_BILLING_PREFIX = f"{AUTO_PROVIDER}/"

# Bare model id → provider, for the model the response reports.
_SERVED_PREFIXES: tuple[tuple[str, str], ...] = (
    ("claude", "anthropic"),
    ("gpt", "openai"),
    ("chatgpt", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
    ("gemini", "google"),
    ("grok", "xai"),
    ("kimi", "moonshot"),
)


def served_model_id(api_model: str) -> str:
    """``provider/model`` for the bare model id a routed response reports."""
    if "/" in api_model:
        return api_model
    lowered = api_model.lower()
    for prefix, provider in _SERVED_PREFIXES:
        if lowered.startswith(prefix):
            return f"{provider}/{api_model}"
    return f"{AUTO_PROVIDER}/{api_model}"


def record_served_model(api_model: str | None) -> str | None:
    """Relabel the active call from ``timbal/auto`` to the model that actually served it.

    The SDK sets the billing id to the *declared* model before the call; for
    ``timbal/auto`` that is meaningless for usage and traces. The first response
    chunk carries the routed model, so collectors call this on every chunk and
    it acts once: usage keys and span metadata move to ``provider/model``.
    Returns the new billing id, or ``None`` when nothing changed.
    """
    if not api_model or api_model == "auto":
        return None
    billing_id = get_billing_id()
    if not billing_id or not billing_id.startswith(_AUTO_BILLING_PREFIX):
        return None
    served = served_model_id(api_model)
    set_billing_id(served)
    provider, _, name = served.partition("/")
    run_context = get_run_context()
    if run_context is not None:
        span = run_context._trace.get(get_call_id())
        if span is not None:
            span.metadata.update(model_provider=provider, model_name=name)
    return served


def _attachment_names(messages: list[Message]) -> list[str]:
    last_user = next((m for m in reversed(messages) if m.role == "user"), None)
    if last_user is None:
        return []
    names: list[str] = []
    for content in last_user.content:
        if not isinstance(content, FileContent):
            continue
        name = content.name
        if not name:
            ext = object.__getattribute__(content.file, "__source_extension__") or ""
            name = f"file{ext}"
        names.append(name)
    return names


def auto_metadata(messages: list[Message]) -> dict[str, str]:
    """``metadata`` entries for a ``timbal/auto`` request: the attachments on the latest user turn.

    Keys are prefixed ``timbal_`` so the platform can remove exactly these.
    """
    names = _attachment_names(messages)
    if not names:
        return {}
    return {"timbal_attachments": ",".join(dict.fromkeys(n.strip() for n in names if n.strip()))}
