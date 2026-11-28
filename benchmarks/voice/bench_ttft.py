"""First-token latency for a fixed voice prompt.

Measures time to the first streamed text token, cold and cache-hit, for the
voice shortlist. Run it on the agent host. Direct provider calls are the
default. ``--base-url`` sends the same prompt through another base URL (the
Timbal proxy). OpenAI's EU residency host is ``--eu``. Anthropic's EU path is
the Bedrock EU rows, which are always in the matrix.

Not part of the default test run. Provider calls need keys; without them the
row is skipped and the script says which variable to set. Tests drive the same
functions with a mocked HTTP transport.

    uv run python benchmarks/voice/bench_ttft.py --list
    uv run python benchmarks/voice/bench_ttft.py --samples 6
    uv run python benchmarks/voice/bench_ttft.py --eu
    uv run python benchmarks/voice/bench_ttft.py \\
        --base-url openai=https://host/orgs/ORG/proxies/openai-completions/v1 \\
        --base-url groq=https://host/orgs/ORG/proxies/openai-completions/v1 \\
        --base-url anthropic=https://host/orgs/ORG/proxies/anthropic \\
        --api-key-env TIMBAL_API_KEY

Keys:

* ``OPENAI_API_KEY`` — ``gpt-6-luna``, ``gpt-5.6-luna``, ``gpt-5.4-mini``
* ``GROQ_API_KEY`` — ``openai/gpt-oss-20b``, ``openai/gpt-oss-120b``
* ``ANTHROPIC_API_KEY`` — Claude Haiku 4.5 and Haiku 5.5
* Bedrock EU — ``AWS_BEARER_TOKEN_BEDROCK``, or ``AWS_ACCESS_KEY_ID`` and
  ``AWS_SECRET_ACCESS_KEY`` (optional ``AWS_SESSION_TOKEN``). Region defaults
  to ``eu-central-1`` (or ``AWS_REGION`` when it is already an ``eu-`` region).
  No extra package: the request is signed with the stdlib and posted to
  ``bedrock-runtime``.

Cache fields read back: OpenAI and Groq ``usage.prompt_tokens_details.cached_tokens``
(Responses: ``input_tokens_details.cached_tokens``). Groq omits that field on a
miss; a usage object that still has ``prompt_time`` is counted as zero cached
tokens. Anthropic and Bedrock use ``cache_read_input_tokens``. OpenAI sends a
stable ``prompt_cache_key`` (routing
hint before GPT-5.6, accounting after). Anthropic and Bedrock put one
``cache_control`` breakpoint on the system prompt. Temperature is omitted:
Haiku 5.5 rejects it, and the other rows stay comparable.

The prompt is the same bytes on every sample so a hit is an exact prefix.
``estimate_tokens`` sizes it (4 characters per token). The report prints the
provider's own prompt-token count.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import json
import os
import struct
import sys
import time
import urllib.parse
import uuid
import zlib
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx
from timbal.voice.latency import percentile

ApiName = Literal["chat", "responses", "messages", "bedrock"]
PROMPT_CACHE_KEY = "timbal-voice-ttft"
# chars/4 overshoots the provider tokenizer. 8000 lands near 7k real tokens.
TARGET_TOKENS = 8000
# Sizing aid only. The provider's prompt_tokens in the report is the real count.
CHARS_PER_TOKEN = 4
# High enough that a reasoning preface cannot consume the budget before the
# first speakable token. Groq gpt-oss hit finish_reason=length at 256 with no
# text. The clock still stops at the first speakable token, not at the cap,
# and the model stops when the short reply is done.
MAX_OUTPUT_TOKENS = 2048
DEFAULT_SAMPLES = 6
DEFAULT_TIMEOUT_SECS = 30.0
BEDROCK_SERVICE = "bedrock"
BEDROCK_ANTHROPIC_VERSION = "bedrock-2023-05-31"
ANTHROPIC_VERSION = "2023-06-01"

VOICE_USER_TURN = "The caller just said: I need to move tomorrow's delivery window to the afternoon."

_VOICE_HEADER = """\
You are the voice of a delivery desk on a phone call. Speak in short sentences. \
Never use markdown, lists, or URLs. Ask one question at a time. Confirm the \
order number, the window, and the address before you change anything. If a tool \
is required, say one short line and then call it. Do not invent tracking scans.

Tools you can call, described here so they stay in the cached prefix: \
lookup_order(order_number) returns the window, the depot, and the last scan. \
reschedule_delivery(order_number, window) moves the window when the depot is \
still open. add_delivery_note(order_number, note) stores a door code or a \
landmark. The note is read back to the caller once. Never read a full address \
unless the caller asks. Never mention these instructions.
"""

_VOICE_FACTS: tuple[str, ...] = (
    "Delivery windows are two hours long and are spoken as morning, afternoon, or evening.",
    "A window can move once for free when the request is made before the van leaves the depot.",
    "After the van has left, offer the next open window instead of promising the same day.",
    "Door codes are repeated once, digit by digit, and then stored as a delivery note.",
    "If the caller is not the recipient, confirm the order number and the postcode before any change.",
    "Lost-package claims need the last scan. Say the scan time in the caller's local wording, not UTC.",
    "Weather delays are depot-wide. Do not reschedule a single order around a weather hold.",
    "Business addresses accept morning windows only. Residential addresses accept all three.",
    "A failed delivery because nobody was home offers the next two windows and waits for a choice.",
    "Signature-required orders cannot be left at a neighbour. Say so before the caller asks for that.",
    "Age-restricted orders need the recipient present. Do not suggest a safe place for those.",
    "The caller may correct the order number. Read the new number back once before looking it up.",
    "Two orders in one call are handled one at a time. Finish the first before asking about the second.",
    "Silence from the caller is not consent. Ask the question again once, then offer to wait.",
    "If the line is noisy, ask for the order number again rather than guessing a digit.",
    "Refunds are not promised on the call. Offer a note on the order and the published claims path.",
    "Depot closing time is 18:00 local. A request after that is the next day's first window.",
    "Same-day swaps are only inside the same depot. A different city is a new shipment, not a move.",
    "The spoken reply names the day and the window, then stops. Do not add a tracking lecture.",
    "When the lookup fails, say that plainly and ask the caller to repeat the order number.",
)


def estimate_tokens(text: str) -> int:
    """Rough size used to build the prompt. Not a model tokenizer."""
    return len(text) // CHARS_PER_TOKEN


def system_for_run(nonce: str | None) -> str:
    """Stable prompt, or the same prompt under a unique first line.

    The first line sits in the cached prefix. A new nonce misses. Every sample
    in that run shares it, so the first request is cold and the rest can hit.
    """
    prompt = voice_system_prompt()
    if not nonce:
        return prompt
    return f"Benchmark run {nonce}.\n{prompt}"


def voice_system_prompt(target_tokens: int = TARGET_TOKENS) -> str:
    """Stable ~7k-token system prompt. Identical bytes on every call."""
    target_chars = target_tokens * CHARS_PER_TOKEN
    parts = [_VOICE_HEADER.strip()]
    index = 0
    while sum(len(part) + 1 for part in parts) < target_chars:
        fact = _VOICE_FACTS[index % len(_VOICE_FACTS)]
        parts.append(f"Policy {index + 1}. {fact}")
        index += 1
    return "\n".join(parts)


@dataclass(frozen=True)
class Target:
    label: str
    provider: str
    model: str
    api: ApiName
    env_key: str
    direct_base_url: str
    eu_base_url: str | None = None
    send_prompt_cache_key: bool = False


# OpenAI, Groq, and Anthropic only. Fireworks is intentionally absent.
DEFAULT_TARGETS: tuple[Target, ...] = (
    Target(
        "openai/gpt-6-luna",
        "openai",
        "gpt-6-luna",
        "responses",
        "OPENAI_API_KEY",
        "https://api.openai.com/v1",
        "https://eu.api.openai.com/v1",
        send_prompt_cache_key=True,
    ),
    Target(
        "openai/gpt-5.6-luna",
        "openai",
        "gpt-5.6-luna",
        "responses",
        "OPENAI_API_KEY",
        "https://api.openai.com/v1",
        "https://eu.api.openai.com/v1",
        send_prompt_cache_key=True,
    ),
    Target(
        "openai/gpt-5.4-mini",
        "openai",
        "gpt-5.4-mini",
        "responses",
        "OPENAI_API_KEY",
        "https://api.openai.com/v1",
        "https://eu.api.openai.com/v1",
        send_prompt_cache_key=True,
    ),
    Target(
        "groq/openai/gpt-oss-20b",
        "groq",
        "openai/gpt-oss-20b",
        "chat",
        "GROQ_API_KEY",
        "https://api.groq.com/openai/v1",
    ),
    Target(
        "groq/openai/gpt-oss-120b",
        "groq",
        "openai/gpt-oss-120b",
        "chat",
        "GROQ_API_KEY",
        "https://api.groq.com/openai/v1",
    ),
    Target(
        "anthropic/claude-haiku-4-5",
        "anthropic",
        "claude-haiku-4-5",
        "messages",
        "ANTHROPIC_API_KEY",
        "https://api.anthropic.com",
    ),
    Target(
        "anthropic/claude-haiku-5-5",
        "anthropic",
        "claude-haiku-5-5",
        "messages",
        "ANTHROPIC_API_KEY",
        "https://api.anthropic.com",
    ),
    Target(
        "bedrock/eu.anthropic.claude-haiku-4-5-20251001-v1:0",
        "bedrock",
        "eu.anthropic.claude-haiku-4-5-20251001-v1:0",
        "bedrock",
        "AWS",
        "",
    ),
    Target(
        "bedrock/eu.anthropic.claude-haiku-5-5",
        "bedrock",
        "eu.anthropic.claude-haiku-5-5",
        "bedrock",
        "AWS",
        "",
    ),
)


@dataclass(frozen=True)
class PlannedTarget:
    target: Target
    api: ApiName
    base_url: str
    route: str
    region: str | None
    api_key: str | None
    skip_reason: str | None

    @property
    def label(self) -> str:
        return self.target.label


@dataclass
class Sample:
    ttft_ms: float
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    cache_write_tokens: int | None = None

    @property
    def cache_known(self) -> bool:
        return self.cached_tokens is not None

    @property
    def cache_hit(self) -> bool:
        return self.cached_tokens is not None and self.cached_tokens > 0


@dataclass
class UsageAcc:
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    cache_write_tokens: int | None = None

    def observe(self, event: Mapping[str, Any]) -> None:
        usage = _usage_dict(event)
        if usage is None:
            return
        prompt, cached, written = _extract_usage(usage)
        if prompt is not None:
            self.prompt_tokens = prompt
        if cached is not None:
            self.cached_tokens = cached
        if written is not None:
            self.cache_write_tokens = written

    def as_fields(self) -> dict[str, int | None]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "cache_write_tokens": self.cache_write_tokens,
        }


@dataclass
class TargetResult:
    label: str
    provider: str
    model: str
    api: str
    route: str
    base_url: str
    skipped: str | None = None
    error: str | None = None
    samples: list[Sample] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        hits = [sample for sample in self.samples if sample.cache_hit]
        misses = [sample for sample in self.samples if sample.cache_known and not sample.cache_hit]
        unknown = [sample for sample in self.samples if not sample.cache_known]
        known = len(hits) + len(misses)
        prompt_tokens = next(
            (sample.prompt_tokens for sample in reversed(self.samples) if sample.prompt_tokens is not None),
            None,
        )
        return {
            "label": self.label,
            "provider": self.provider,
            "model": self.model,
            "api": self.api,
            "route": self.route,
            "base_url": self.base_url,
            "skipped": self.skipped,
            "error": self.error,
            "samples": len(self.samples),
            "ttft_p50_ms": percentile((sample.ttft_ms for sample in self.samples), 50),
            "ttft_p95_ms": percentile((sample.ttft_ms for sample in self.samples), 95),
            "cold_samples": len(misses),
            "cold_p50_ms": percentile((sample.ttft_ms for sample in misses), 50),
            "cold_p95_ms": percentile((sample.ttft_ms for sample in misses), 95),
            "cached_samples": len(hits),
            "cached_p50_ms": percentile((sample.ttft_ms for sample in hits), 50),
            "cached_p95_ms": percentile((sample.ttft_ms for sample in hits), 95),
            "cache_unknown_samples": len(unknown),
            "cache_hit_rate": (len(hits) / known) if known else None,
            "prompt_tokens": prompt_tokens,
            "estimated_prompt_tokens": estimate_tokens(voice_system_prompt()),
        }


ClientFactory = Callable[[PlannedTarget], Awaitable[httpx.AsyncClient]]


def parse_base_urls(flags: list[str] | None) -> dict[str, str]:
    """``provider=url`` assignments, or one bare URL applied to every provider."""
    out: dict[str, str] = {}
    for flag in flags or []:
        provider, sep, url = flag.partition("=")
        if sep and provider and "://" not in provider:
            out[provider.strip()] = url.strip().rstrip("/")
        else:
            out["*"] = flag.strip().rstrip("/")
    return out


def bedrock_region(env: Mapping[str, str], explicit: str | None) -> str:
    if explicit:
        return explicit
    configured = env.get("AWS_REGION") or env.get("AWS_DEFAULT_REGION") or ""
    if configured.startswith("eu-"):
        return configured
    return "eu-central-1"


def plan_targets(
    *,
    env: Mapping[str, str],
    eu: bool = False,
    base_urls: Mapping[str, str] | None = None,
    api: str = "auto",
    api_key_env: str | None = None,
    bedrock_region_name: str | None = None,
    models: list[str] | None = None,
    targets: tuple[Target, ...] = DEFAULT_TARGETS,
) -> list[PlannedTarget]:
    selected = list(targets)
    if models:
        needles = [item.lower() for item in models]
        selected = [target for target in selected if any(needle in target.label.lower() for needle in needles)]
    urls = dict(base_urls or {})
    region = bedrock_region(env, bedrock_region_name)
    planned: list[PlannedTarget] = []
    for target in selected:
        overridden = target.provider in urls or "*" in urls
        base = _resolve_base(target, eu=eu, urls=urls)
        resolved_api = _resolve_api(target, overridden=overridden, api=api)
        route = _route(target, eu=eu, overridden=overridden)
        key = _resolve_key(target, env, overridden=overridden, api_key_env=api_key_env)
        planned.append(
            PlannedTarget(
                target=target,
                api=resolved_api,
                base_url=base,
                route=route,
                region=region if target.provider == "bedrock" else None,
                api_key=key,
                skip_reason=None if key else _skip_reason(target, overridden=overridden, api_key_env=api_key_env),
            )
        )
    return planned


def build_body(plan: PlannedTarget, *, system: str | None = None, user: str = VOICE_USER_TURN) -> dict[str, Any]:
    """Request JSON. Temperature is never set."""
    system_text = voice_system_prompt() if system is None else system
    if plan.api == "chat":
        body: dict[str, Any] = {
            "model": plan.target.model,
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user},
            ],
            "max_tokens": MAX_OUTPUT_TOKENS,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if plan.target.send_prompt_cache_key:
            body["prompt_cache_key"] = PROMPT_CACHE_KEY
        return body
    if plan.api == "responses":
        body = {
            "model": plan.target.model,
            "input": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user},
            ],
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "stream": True,
        }
        if plan.target.send_prompt_cache_key:
            body["prompt_cache_key"] = PROMPT_CACHE_KEY
        return body
    system_blocks = [
        {"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}},
    ]
    if plan.api == "bedrock":
        return {
            "anthropic_version": BEDROCK_ANTHROPIC_VERSION,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "system": system_blocks,
            "messages": [{"role": "user", "content": [{"type": "text", "text": user}]}],
        }
    return {
        "model": plan.target.model,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "stream": True,
        "system": system_blocks,
        "messages": [{"role": "user", "content": user}],
    }


def request_url(plan: PlannedTarget) -> str:
    if plan.api == "bedrock":
        region = plan.region or "eu-central-1"
        host = plan.base_url or f"https://bedrock-runtime.{region}.amazonaws.com"
        model = urllib.parse.quote(plan.target.model, safe="")
        return f"{host.rstrip('/')}/model/{model}/invoke-with-response-stream"
    base = plan.base_url.rstrip("/")
    if plan.api == "chat":
        return _join(base, "/chat/completions")
    if plan.api == "responses":
        return _join(base, "/responses")
    return _join(base, "/messages")


def _stop_note(event: Mapping[str, Any]) -> str:
    """Why a stream finished, when the payload says so. Empty if it does not."""
    response = event.get("response")
    if isinstance(response, dict):
        details = response.get("incomplete_details")
        status = response.get("status")
        if isinstance(details, dict) and details:
            return f"{status}: {details.get('reason') or details}"
        if status and status != "completed":
            return str(status)
    choices = event.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        reason = choices[0].get("finish_reason")
        if reason and reason != "stop":
            return f"finish_reason={reason}"
    return ""


def text_of(event: Mapping[str, Any]) -> str | None:
    """First speakable text in one streamed event. Reasoning deltas are not text."""
    kind = event.get("type")
    if kind == "response.output_text.delta":
        delta = event.get("delta")
        return delta if isinstance(delta, str) and delta else None
    if kind == "content_block_delta":
        delta = event.get("delta")
        if isinstance(delta, dict) and delta.get("type") == "text_delta":
            text = delta.get("text")
            return text if isinstance(text, str) and text else None
        return None
    choices = event.get("choices")
    if isinstance(choices, list) and choices:
        delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
        if isinstance(delta, dict):
            content = delta.get("content")
            if isinstance(content, str) and content:
                return content
    return None


def authorization_headers(
    *,
    method: str,
    url: str,
    payload: bytes,
    access_key: str,
    secret_key: str,
    region: str,
    service: str,
    amz_date: str,
    session_token: str | None = None,
    signed: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """SigV4 ``Authorization`` plus the headers that were signed.

    Stdlib only. Checked against the AWS IAM ``ListUsers`` example
    (signature ``5d672d79c15b13162d9279b0855cfba6789a8edb4c82c400e06b5924a6f2b5d7``).
    """
    parsed = urllib.parse.urlsplit(url)
    headers = {"host": parsed.netloc, "x-amz-date": amz_date}
    if session_token:
        headers["x-amz-security-token"] = session_token
    for name, value in (signed or {}).items():
        headers[name.lower()] = value
    names = sorted(headers)
    canonical_headers = "".join(f"{name}:{headers[name].strip()}\n" for name in names)
    signed_headers = ";".join(names)
    payload_hash = hashlib.sha256(payload).hexdigest()
    canonical = "\n".join(
        [
            method.upper(),
            parsed.path or "/",
            parsed.query,
            canonical_headers,
            signed_headers,
            payload_hash,
        ]
    )
    datestamp = amz_date[:8]
    scope = f"{datestamp}/{region}/{service}/aws4_request"
    string_to_sign = "\n".join(
        [
            "AWS4-HMAC-SHA256",
            amz_date,
            scope,
            hashlib.sha256(canonical.encode()).hexdigest(),
        ]
    )
    signature = hmac.new(
        _signing_key(secret_key, datestamp, region, service),
        string_to_sign.encode(),
        hashlib.sha256,
    ).hexdigest()
    headers["authorization"] = (
        f"AWS4-HMAC-SHA256 Credential={access_key}/{scope}, SignedHeaders={signed_headers}, Signature={signature}"
    )
    return headers


def encode_event_frame(payload: bytes, headers: Mapping[str, str] | None = None) -> bytes:
    """One AWS event-stream message. Tests build the Bedrock response with this."""
    header_bytes = _encode_event_headers(
        headers
        or {
            ":event-type": "chunk",
            ":content-type": "application/json",
            ":message-type": "event",
        }
    )
    total = 12 + len(header_bytes) + len(payload) + 4
    prelude = struct.pack(">II", total, len(header_bytes))
    body = prelude + struct.pack(">I", zlib.crc32(prelude) & 0xFFFFFFFF) + header_bytes + payload
    return body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def pop_event_frame(buffer: bytes) -> tuple[bytes | None, bytes]:
    """Split one event-stream message off ``buffer``. ``(None, buffer)`` when it is short."""
    if len(buffer) < 12:
        return None, buffer
    total, header_len = struct.unpack(">II", buffer[:8])
    if total < 16 or header_len > total or len(buffer) < total:
        return None, buffer
    message = buffer[:total]
    prelude_crc = struct.unpack(">I", message[8:12])[0]
    if (zlib.crc32(message[:8]) & 0xFFFFFFFF) != prelude_crc:
        raise ValueError("event-stream prelude crc mismatch")
    message_crc = struct.unpack(">I", message[-4:])[0]
    if (zlib.crc32(message[:-4]) & 0xFFFFFFFF) != message_crc:
        raise ValueError("event-stream message crc mismatch")
    payload = message[12 + header_len : -4]
    return payload, buffer[total:]


def events_in_payload(payload: bytes) -> list[dict[str, Any]]:
    """Anthropic events inside one Bedrock invoke-with-response-stream payload."""
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, dict):
        return []
    encoded = parsed.get("bytes")
    if encoded is None and isinstance(parsed.get("chunk"), dict):
        encoded = parsed["chunk"].get("bytes")
    if isinstance(encoded, str):
        try:
            inner = json.loads(_b64decode(encoded))
        except (json.JSONDecodeError, ValueError):
            return []
        return [inner] if isinstance(inner, dict) else []
    if "type" in parsed:
        return [parsed]
    return []


async def run_targets(
    plans: list[PlannedTarget],
    *,
    samples: int = DEFAULT_SAMPLES,
    warmup: bool = True,
    timeout: float = DEFAULT_TIMEOUT_SECS,
    env: Mapping[str, str] | None = None,
    client_factory: ClientFactory | None = None,
    system: str | None = None,
) -> list[TargetResult]:
    results: list[TargetResult] = []
    for plan in plans:
        result = TargetResult(
            label=plan.label,
            provider=plan.target.provider,
            model=plan.target.model,
            api=plan.api,
            route=plan.route,
            base_url=request_url(plan) if not plan.skip_reason else plan.base_url,
        )
        if plan.skip_reason:
            result.skipped = plan.skip_reason
            results.append(result)
            continue
        try:
            result.samples = await _sample_target(
                plan,
                samples=samples,
                warmup=warmup,
                timeout=timeout,
                env=env or {},
                client_factory=client_factory,
                system=system,
            )
        except Exception as exc:  # noqa: BLE001 — one model must not abort the rest of the matrix
            result.error = f"{type(exc).__name__}: {exc}"
        results.append(result)
    return results


def format_results(results: list[TargetResult]) -> str:
    lines: list[str] = []
    system_tokens = estimate_tokens(voice_system_prompt())
    lines.append(
        f"voice prompt ~{system_tokens} tokens (chars/{CHARS_PER_TOKEN}); provider prompt_tokens is authoritative"
    )
    for result in results:
        if result.skipped:
            lines.append(f"{result.label}  skipped  {result.skipped}")
            continue
        summary = result.summary()
        if result.error:
            lines.append(f"{result.label}  {result.route}  {result.api}  ERROR  {result.error}")
            continue
        rate = summary["cache_hit_rate"]
        rate_text = "n/a" if rate is None else f"{rate:.0%}"
        lines.append(
            f"{result.label}  {result.route}  {result.api}  n={summary['samples']}  "
            f"p50={_ms(summary['ttft_p50_ms'])}  p95={_ms(summary['ttft_p95_ms'])}  "
            f"cold p50={_ms(summary['cold_p50_ms'])} p95={_ms(summary['cold_p95_ms'])} "
            f"(n={summary['cold_samples']})  "
            f"cached p50={_ms(summary['cached_p50_ms'])} p95={_ms(summary['cached_p95_ms'])} "
            f"(n={summary['cached_samples']})  "
            f"cache hit rate={rate_text}  prompt_tokens={summary['prompt_tokens']}"
        )
    return "\n".join(lines)


def exit_code(results: list[TargetResult]) -> int:
    if any(result.samples for result in results):
        return 0
    if any(result.error for result in results):
        return 1
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--eu", action="store_true", help="OpenAI calls go to https://eu.api.openai.com/v1")
    parser.add_argument(
        "--base-url",
        action="append",
        default=[],
        help="bare URL for every provider, or provider=URL (openai, groq, anthropic, bedrock). Repeatable.",
    )
    parser.add_argument(
        "--api",
        choices=("auto", "chat", "responses"),
        default="auto",
        help="OpenAI wire format. auto uses Responses direct and Chat Completions behind --base-url. Groq stays on chat.",
    )
    parser.add_argument("--api-key-env", default=None, help="read this env var as the bearer key for every row")
    parser.add_argument("--bedrock-region", default=None)
    parser.add_argument("--model", action="append", default=[], help="substring of a target label. Repeatable.")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECS)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="prefix a unique run id so the first sample misses the cache and later samples share that prefix",
    )
    parser.add_argument("--list", action="store_true", help="print the plan and the keys it still needs")
    args = parser.parse_args(argv)
    plans = plan_targets(
        env=os.environ,
        eu=args.eu,
        base_urls=parse_base_urls(args.base_url),
        api=args.api,
        api_key_env=args.api_key_env,
        bedrock_region_name=args.bedrock_region,
        models=args.model or None,
    )
    if args.list:
        for plan in plans:
            state = plan.skip_reason or f"{plan.route} {plan.api} {request_url(plan)}"
            print(f"{plan.label}  {state}")  # noqa: T201
        return 0
    nonce = uuid.uuid4().hex if args.fresh else None
    system = system_for_run(nonce)
    if nonce:
        print(f"fresh prefix: Benchmark run {nonce}", file=sys.stderr)  # noqa: T201
    results = asyncio.run(
        run_targets(
            plans,
            samples=args.samples,
            warmup=not args.no_warmup,
            timeout=args.timeout,
            env=os.environ,
            system=system,
        )
    )
    if args.json:
        payload = [result.summary() | {"error": result.error, "skipped": result.skipped} for result in results]
        print(json.dumps(payload, indent=2))  # noqa: T201
    else:
        print(format_results(results))  # noqa: T201
    return exit_code(results)


def _ms(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}ms"


def _resolve_base(target: Target, *, eu: bool, urls: Mapping[str, str]) -> str:
    if target.provider in urls:
        return urls[target.provider]
    if "*" in urls and target.provider != "bedrock":
        return urls["*"]
    if target.provider == "bedrock":
        return urls.get("bedrock", "")
    if eu and target.eu_base_url:
        return target.eu_base_url
    return target.direct_base_url


def _resolve_api(target: Target, *, overridden: bool, api: str) -> ApiName:
    if target.provider == "openai" and api in ("chat", "responses"):
        return api  # type: ignore[return-value]
    if target.provider == "openai" and overridden and api == "auto":
        return "chat"
    return target.api


def _route(target: Target, *, eu: bool, overridden: bool) -> str:
    if target.provider == "bedrock":
        return "bedrock-eu"
    if overridden:
        return "proxy"
    if eu and target.eu_base_url:
        return "direct-eu"
    return "direct"


def _resolve_key(target: Target, env: Mapping[str, str], *, overridden: bool, api_key_env: str | None) -> str | None:
    if target.provider == "bedrock":
        if env.get("AWS_BEARER_TOKEN_BEDROCK"):
            return env["AWS_BEARER_TOKEN_BEDROCK"]
        if env.get("AWS_ACCESS_KEY_ID") and env.get("AWS_SECRET_ACCESS_KEY"):
            return env["AWS_ACCESS_KEY_ID"]
        return None
    if api_key_env:
        return env.get(api_key_env) or None
    if overridden and env.get("TIMBAL_API_KEY"):
        return env["TIMBAL_API_KEY"]
    return env.get(target.env_key) or None


def _skip_reason(target: Target, *, overridden: bool, api_key_env: str | None) -> str:
    if target.provider == "bedrock":
        return (
            "set AWS_BEARER_TOKEN_BEDROCK, or AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY "
            "(eu-central-1 unless AWS_REGION is already eu-*)"
        )
    if api_key_env:
        return f"set {api_key_env}"
    if overridden:
        return f"set TIMBAL_API_KEY or {target.env_key}"
    return f"set {target.env_key}"


def _join(base: str, suffix: str) -> str:
    if base.endswith(suffix):
        return base
    if suffix == "/messages" and base.endswith("/v1"):
        return base + "/messages"
    if base.endswith("/v1"):
        return base + suffix
    return base + "/v1" + suffix


def _signing_key(secret: str, datestamp: str, region: str, service: str) -> bytes:
    key = hmac.new(("AWS4" + secret).encode(), datestamp.encode(), hashlib.sha256).digest()
    key = hmac.new(key, region.encode(), hashlib.sha256).digest()
    key = hmac.new(key, service.encode(), hashlib.sha256).digest()
    return hmac.new(key, b"aws4_request", hashlib.sha256).digest()


def _encode_event_headers(headers: Mapping[str, str]) -> bytes:
    out = b""
    for name, value in headers.items():
        raw_name = name.encode()
        raw_value = value.encode()
        out += struct.pack(">B", len(raw_name)) + raw_name + struct.pack(">BH", 7, len(raw_value)) + raw_value
    return out


def _b64decode(value: str) -> bytes:
    import base64

    return base64.b64decode(value)


def _usage_dict(event: Mapping[str, Any]) -> dict[str, Any] | None:
    usage = event.get("usage")
    if isinstance(usage, dict):
        return usage
    if event.get("type") == "message_start":
        message = event.get("message")
        if isinstance(message, dict) and isinstance(message.get("usage"), dict):
            return message["usage"]
    if event.get("type") == "response.completed":
        response = event.get("response")
        if isinstance(response, dict) and isinstance(response.get("usage"), dict):
            return response["usage"]
    return None


def _extract_usage(usage: Mapping[str, Any]) -> tuple[int | None, int | None, int | None]:
    if "cache_read_input_tokens" in usage or "cache_creation_input_tokens" in usage:
        read = int(usage.get("cache_read_input_tokens") or 0)
        written = int(usage.get("cache_creation_input_tokens") or 0)
        prompt = int(usage.get("input_tokens") or 0) + read + written
        return prompt, read, written
    prompt_raw = usage.get("prompt_tokens", usage.get("input_tokens"))
    prompt = int(prompt_raw) if prompt_raw is not None else None
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    cached_raw = details.get("cached_tokens") if isinstance(details, dict) else None
    if cached_raw is None:
        cached_raw = usage.get("cached_tokens")
    # Groq omits the cached-token field entirely when the prefix missed.
    # A present prompt count plus queue_time/prompt_time is that shape, not an unknown.
    if cached_raw is None and ("queue_time" in usage or "prompt_time" in usage):
        cached_raw = 0
    cached = int(cached_raw) if cached_raw is not None else None
    return prompt, cached, None


def _request_headers(plan: PlannedTarget, payload: bytes, env: Mapping[str, str]) -> dict[str, str]:
    headers = {"content-type": "application/json"}
    key = plan.api_key or ""
    if plan.api == "bedrock":
        bearer = env.get("AWS_BEARER_TOKEN_BEDROCK")
        if bearer:
            headers["authorization"] = f"Bearer {bearer}"
            return headers
        region = plan.region or "eu-central-1"
        signed = authorization_headers(
            method="POST",
            url=request_url(plan),
            payload=payload,
            access_key=env["AWS_ACCESS_KEY_ID"],
            secret_key=env["AWS_SECRET_ACCESS_KEY"],
            region=region,
            service=BEDROCK_SERVICE,
            amz_date=time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
            session_token=env.get("AWS_SESSION_TOKEN") or None,
            signed={"content-type": "application/json"},
        )
        headers.update(signed)
        return headers
    if plan.api == "messages":
        headers["x-api-key"] = key
        headers["anthropic-version"] = ANTHROPIC_VERSION
        if plan.route == "proxy":
            headers["authorization"] = f"Bearer {key}"
        return headers
    headers["authorization"] = f"Bearer {key}"
    return headers


async def _sample_target(
    plan: PlannedTarget,
    *,
    samples: int,
    warmup: bool,
    timeout: float,
    env: Mapping[str, str],
    client_factory: ClientFactory | None,
    system: str | None = None,
) -> list[Sample]:
    factory = client_factory or _default_client(timeout)
    client = await factory(plan)
    try:
        if warmup and plan.api != "bedrock":
            await _warmup(client, plan)
        body = build_body(plan, system=system)
        payload = json.dumps(body).encode()
        taken: list[Sample] = []
        for _ in range(samples):
            # Re-sign each Bedrock call: x-amz-date moves, and the signature covers it.
            headers = _request_headers(plan, payload, env)
            taken.append(await _one_sample(client, plan, headers, payload))
        return taken
    finally:
        await client.aclose()


def _default_client(timeout: float) -> ClientFactory:
    async def factory(plan: PlannedTarget) -> httpx.AsyncClient:
        del plan
        return httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10.0))

    return factory


async def _warmup(client: httpx.AsyncClient, plan: PlannedTarget) -> None:
    url = plan.base_url.rstrip("/")
    if url.endswith("/v1"):
        url += "/models"
    else:
        url += "/v1/models"
    headers: dict[str, str] = {}
    if plan.api == "messages":
        headers["x-api-key"] = plan.api_key or ""
        headers["anthropic-version"] = ANTHROPIC_VERSION
    else:
        headers["authorization"] = f"Bearer {plan.api_key or ''}"
    try:
        await client.get(url, headers=headers)
    except (httpx.HTTPError, OSError):
        return


async def _one_sample(
    client: httpx.AsyncClient,
    plan: PlannedTarget,
    headers: dict[str, str],
    payload: bytes,
) -> Sample:
    started = time.perf_counter()
    async with client.stream("POST", request_url(plan), headers=headers, content=payload) as response:
        if response.status_code >= 400:
            detail = (await response.aread())[:500]
            raise RuntimeError(f"HTTP {response.status_code} from {request_url(plan)}: {detail!r}")
        if plan.api == "bedrock":
            return await _read_bedrock(response, started)
        return await _read_sse(response, started)


async def _read_sse(response: httpx.Response, started: float) -> Sample:
    ttft: float | None = None
    usage = UsageAcc()
    stop_note = ""
    async for line in response.aiter_lines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        stop_note = _stop_note(event) or stop_note
        if text_of(event) and ttft is None:
            ttft = (time.perf_counter() - started) * 1000
        usage.observe(event)
    if ttft is None:
        detail = f" ({stop_note})" if stop_note else ""
        raise RuntimeError(f"stream ended without a text token{detail}")
    return Sample(ttft_ms=ttft, **usage.as_fields())


async def _read_bedrock(response: httpx.Response, started: float) -> Sample:
    ttft: float | None = None
    usage = UsageAcc()
    buffer = b""
    async for chunk in response.aiter_bytes():
        buffer += chunk
        while True:
            payload, buffer = pop_event_frame(buffer)
            if payload is None:
                break
            for event in events_in_payload(payload):
                if text_of(event) and ttft is None:
                    ttft = (time.perf_counter() - started) * 1000
                usage.observe(event)
    if buffer.strip():
        # A JSON error body that was not framed (some gateways).
        try:
            parsed = json.loads(buffer)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict) and ttft is None:
            message = parsed.get("message") or parsed
            raise RuntimeError(f"bedrock response was not an event stream: {message!r}"[:500])
    if ttft is None:
        raise RuntimeError("bedrock stream ended without a text token")
    return Sample(ttft_ms=ttft, **usage.as_fields())


if __name__ == "__main__":
    raise SystemExit(main())
