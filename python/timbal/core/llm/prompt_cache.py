"""Prompt-cache request shaping for OpenAI, Groq, and Anthropic.

Verified against the provider docs (4 Nov 2026):

* OpenAI prompt caching — prefixes of 1,024 tokens and up are cached
  automatically. ``prompt_cache_key`` routes requests that share a prefix on
  models before GPT-5.6 and only separates cache accounting from GPT-5.6 on.
  GPT-5.6 and later do not reuse a shorter stable prefix unless the request
  marks it with ``prompt_cache_breakpoint``; the marker lives on a Responses
  ``input_text`` block. Reads come back as
  ``usage.input_tokens_details.cached_tokens`` (Responses) or
  ``usage.prompt_tokens_details.cached_tokens`` (Chat Completions).
  https://developers.openai.com/api/docs/guides/prompt-caching

* Anthropic prompt caching — top-level ``cache_control`` is automatic caching
  (the breakpoint follows the conversation). An explicit breakpoint on the
  last stable block is what a varying suffix (caller, clock) needs; automatic
  caching alone writes the breakpoint on the changing block and the lookback
  never hits. Up to 4 breakpoints. Minimum prefix 512 tokens on Haiku 5.5 and
  4,096 on Haiku 4.5. Reads are ``cache_read_input_tokens``.
  https://platform.claude.com/docs/en/build-with-claude/prompt-caching

* AWS Bedrock — InvokeModel / Messages uses the same ``cache_control``
  breakpoints. Converse uses a ``cachePoint`` block after the cached text.
  Top-level ``cache_control`` (automatic caching) is accepted for Haiku 4.5
  and Haiku 5.5; the legacy Bedrock integration (Opus 4.6 and earlier) returns
  400 for that field, so the block breakpoint is the one Bedrock always
  honors. Same 4-breakpoint cap, 5-minute default or 1-hour TTL, and the same
  per-model minimums.
  https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html

* Groq — exact-prefix caching is automatic, only on the GPT-OSS models, and
  unused entries expire after 2 hours. There is no session key. Hits are
  ``usage.prompt_tokens_details.cached_tokens``.
  https://console.groq.com/docs/prompt-caching

LiveKit Agents sends ``prompt_cache_key`` on the OpenAI request and
``prompt_cache_breakpoint`` markers once the model is GPT-5.6 or later
(Apache-2.0, ``livekit-plugins-openai``). Same fields here; the request
builder is Timbal's.

The platform proxy forwards the session key. The framework sends it as
``prompt_cache_key`` on OpenAI request bodies and as
``x-timbal-prompt-cache-key`` on every provider call that has a session.
"""

from __future__ import annotations

import contextvars
import hashlib
import re
from typing import Any

from pydantic import BaseModel, ConfigDict

# OpenAI's published minimum cacheable prefix. Shorter prefixes are not cached
# and are not an error. Before GPT-5.6 the exact minimum also depends on
# request settings (tools, images, schemas).
OPENAI_MIN_CACHE_TOKENS = 1024

# Anthropic / Bedrock minimums for the voice models. A shorter prefix is
# processed without caching and without an error.
ANTHROPIC_MIN_CACHE_TOKENS: dict[str, int] = {
    "claude-haiku-5-5": 512,
    "claude-haiku-4-5": 4096,
}

# Groq caches an exact prefix on these model ids only (the part after ``groq/``).
GROQ_PROMPT_CACHE_MODELS = frozenset(
    {
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-safeguard-20b",
    }
)

PROMPT_CACHE_KEY_HEADER = "x-timbal-prompt-cache-key"

# GPT-5.6 and later: explicit breakpoints.
# gpt-5.6, gpt-5.6-luna, gpt-5.10, gpt-6, gpt-6-luna, gpt-6.1-sol.
_EXPLICIT_BREAKPOINT_MODEL_RE = re.compile(
    r"^gpt-(?:5\.(?:[6-9]|[1-9]\d+)|[6-9]\d*(?:\.\d+)?)(?:-|$)",
    re.IGNORECASE,
)

_prompt_cache_key: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "timbal_prompt_cache_key",
    default=None,
)

_MISSING = object()


class SystemPrompt(BaseModel):
    """Static instructions first, per-turn text last.

    Providers cache an exact prefix. A clock, caller id, or customer record
    inside ``static`` changes the prefix every turn, so nothing hits. Those
    values belong in ``dynamic``, which is sent after the cached prefix.
    """

    model_config = ConfigDict(extra="forbid")

    static: str = ""
    dynamic: str = ""

    def as_text(self) -> str:
        """One string, static bytes first. Empty parts are omitted."""
        parts = [part for part in (self.static, self.dynamic) if part]
        return "\n\n".join(parts)


def voice_prompt(static: str, dynamic: str = "") -> SystemPrompt:
    """Build a cache-friendly system prompt.

    ``static`` is the spoken behavior and anything else that stays byte-for-byte
    the same across turns. ``dynamic`` is this turn's caller, time, or customer
    data. On a voice turn Timbal appends the call's own caller, time, and
    customer fields after ``dynamic``.
    """
    return SystemPrompt(static=static, dynamic=dynamic)


def bind_prompt_cache_key(key: str | None) -> contextvars.Token[str | None]:
    """Pin the OpenAI session cache key for the current context."""
    return _prompt_cache_key.set(key)


def reset_prompt_cache_key(token: contextvars.Token[str | None]) -> None:
    _prompt_cache_key.reset(token)


def current_prompt_cache_key() -> str | None:
    return _prompt_cache_key.get()


def normalize_cache_key(key: Any) -> str | None:
    """A single-line key. Longer than 256 characters is hashed so it stays stable."""
    if not isinstance(key, str):
        return None
    key = key.strip()
    if not key or "\n" in key or "\r" in key:
        return None
    if len(key) > 256:
        return hashlib.sha256(key.encode()).hexdigest()
    return key


def session_prompt_cache_key(provider_params: dict[str, Any] | None = None) -> str | None:
    """Explicit ``prompt_cache_key`` wins; otherwise the bound session key."""
    explicit = (provider_params or {}).get("prompt_cache_key", _MISSING)
    raw = current_prompt_cache_key() if explicit is _MISSING else explicit
    return normalize_cache_key(raw)


def supports_explicit_prompt_cache_breakpoint(model_name: str) -> bool:
    """GPT-5.6 and later mark cache boundaries; earlier models cache a prefix."""
    return bool(_EXPLICIT_BREAKPOINT_MODEL_RE.match((model_name or "").strip()))


def groq_prompt_cache_model(model_name: str) -> bool:
    """True when Groq caches this model (GPT-OSS exact-prefix caching)."""
    return (model_name or "").strip() in GROQ_PROMPT_CACHE_MODELS


def normalize_system_prompt(system_prompt: str | SystemPrompt | None) -> SystemPrompt | None:
    if system_prompt is None:
        return None
    if isinstance(system_prompt, SystemPrompt):
        if not system_prompt.static and not system_prompt.dynamic:
            return None
        return system_prompt
    if isinstance(system_prompt, str):
        return SystemPrompt(static=system_prompt) if system_prompt else None
    return None


def system_prompt_text(system_prompt: str | SystemPrompt | None) -> str | None:
    """Flatten a prompt for a one-shot caller that wants a single string."""
    if isinstance(system_prompt, str):
        return system_prompt or None
    prompt = normalize_system_prompt(system_prompt)
    if prompt is None:
        return None
    return prompt.as_text() or None


def apply_openai_prompt_cache_key(
    provider: str,
    kwargs: dict[str, Any],
    provider_params: dict[str, Any],
) -> None:
    """Put ``prompt_cache_key`` on an OpenAI body. Drop it for every other provider.

    Groq rejects nothing it does not document only when the field is absent;
    its cache is automatic and has no session key. The key is popped here so a
    later ``dict.update`` cannot put it back on a Groq request.
    """
    explicit = provider_params.pop("prompt_cache_key", _MISSING)
    if provider != "openai":
        return
    raw = current_prompt_cache_key() if explicit is _MISSING else explicit
    key = normalize_cache_key(raw)
    if key:
        kwargs["prompt_cache_key"] = key


def anthropic_system_param(
    system_prompt: str | SystemPrompt | None,
    cache_control: dict[str, Any] | None,
) -> str | list[dict[str, Any]] | None:
    """Messages / InvokeModel ``system`` value.

    A plain string when the whole prompt is stable (top-level ``cache_control``
    covers it). When a dynamic suffix is present and caching is on, the static
    block carries ``cache_control`` and the suffix does not — the shape Bedrock
    InvokeModel caches, including on the legacy integration that rejects a
    top-level ``cache_control``. ``cache_control`` None means the caller opted
    out: one string, no marker.
    """
    prompt = normalize_system_prompt(system_prompt)
    if prompt is None:
        return None
    if cache_control is None or not prompt.dynamic or not prompt.static:
        text = prompt.as_text()
        return text or None
    return [
        {"type": "text", "text": prompt.static, "cache_control": dict(cache_control)},
        {"type": "text", "text": prompt.dynamic},
    ]


def bedrock_converse_system(
    system_prompt: str | SystemPrompt | None,
    *,
    ttl: str | None = None,
) -> list[dict[str, Any]]:
    """Converse ``system`` blocks: ``cachePoint`` after the static text.

    Timbal speaks the Messages API (InvokeModel ``cache_control``). This is the
    same split in Converse's ``cachePoint`` shape, for a proxy that translates
    the request onto Bedrock Converse. ``ttl`` is ``"5m"`` or ``"1h"``; omit
    it for the 5-minute default. A model that does not support the field
    rejects it, so the default request leaves it off.
    """
    prompt = normalize_system_prompt(system_prompt)
    if prompt is None:
        return []
    point: dict[str, Any] = {"type": "default"}
    if ttl:
        point["ttl"] = ttl
    blocks: list[dict[str, Any]] = []
    if prompt.static:
        blocks.append({"text": prompt.static})
        if prompt.dynamic:
            blocks.append({"cachePoint": point})
    if prompt.dynamic:
        blocks.append({"text": prompt.dynamic})
    elif prompt.static:
        # Stable prompt, no suffix: the checkpoint still marks the prefix.
        blocks.append({"cachePoint": point})
    return blocks


def openai_responses_instructions(
    system_prompt: str | SystemPrompt | None,
    model_name: str,
) -> tuple[str | None, list[dict[str, Any]]]:
    """``(instructions, leading input items)`` for the Responses API.

    GPT-5.6 and later get an explicit breakpoint after the static text and the
    dynamic suffix as the next developer item, matching OpenAI's guide. Earlier
    models keep one ``instructions`` string with the static bytes first; a
    matching prefix of that string is what automatic caching reuses.
    """
    prompt = normalize_system_prompt(system_prompt)
    if prompt is None:
        return None, []
    if prompt.static and prompt.dynamic and supports_explicit_prompt_cache_breakpoint(model_name):
        static_item = {
            "role": "developer",
            "content": [
                {
                    "type": "input_text",
                    "text": prompt.static,
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                }
            ],
        }
        dynamic_item = {"role": "developer", "content": prompt.dynamic}
        return None, [static_item, dynamic_item]
    text = prompt.as_text()
    return (text or None), []


def openai_chat_system_message(system_prompt: str | SystemPrompt | None) -> dict[str, Any] | None:
    """One Chat Completions system message, static text first.

    Groq and OpenAI-before-5.6 cache an exact prefix of this string. The
    Responses breakpoint field is not sent here: Chat Completions does not
    document it, and Groq would see an unknown content field.
    """
    prompt = normalize_system_prompt(system_prompt)
    if prompt is None:
        return None
    text = prompt.as_text()
    if not text:
        return None
    return {"role": "system", "content": text}
